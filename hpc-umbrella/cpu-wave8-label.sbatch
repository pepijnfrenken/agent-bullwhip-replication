#!/bin/bash
# =============================================================================
# cpu-wave8-label.sbatch — WAVE8 round-1 labels for the R2 arm (soft-target SFT warm
# start), w8_g26b candidates, CPU partition. Clone of cpu-wave7-label.sbatch.
#
# 700 episodes x 136 states x 26 candidates x M=16 futures. Wave 7 measured 186.7 s for
# the 21-candidate version on 96 cores (labels.manifest.json wall_sec); scaling by 26/21
# gives ~4 min [INFERENCE: not measured for w8_g26b until this job runs].
#
#   Submit:  cd /scratch-shared/$USER/bullwhip/kit && sbatch cpu-wave8-label.sbatch
#   Results: results/wave8_labels/{labels.jsonl,labels.npz,labels.manifest.json}
#   R1 does NOT need this job (it labels its own on-policy states each round).
# =============================================================================
#SBATCH --job-name=bw8-label
#SBATCH --partition=tue.cpu3.q
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=16G
#SBATCH --output=/scratch-shared/%u/bullwhip/logs/wave8-label_%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=umbrella-cluster@pepijnfrenken.nl

set -euo pipefail

KIT_DIR="${SLURM_SUBMIT_DIR:-}"
if [ -z "${KIT_DIR}" ] || [ ! -r "${KIT_DIR}/config.sh" ]; then
  KIT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
if [ ! -r "${KIT_DIR}/config.sh" ]; then
  echo "FATAL: config.sh not found in KIT_DIR=${KIT_DIR} — submit from the kit dir (cd <kit> && sbatch <file>)" >&2
  exit 1
fi
# shellcheck source=config.sh
. "${KIT_DIR}/config.sh"

bw_load_env
export HF_HUB_OFFLINE=1
mkdir -p "${LOG_DIR}"
cd "${REPO_DIR}"

REPO_HEAD="$(git -C "${REPO_DIR}" rev-parse HEAD 2>/dev/null || echo unknown)"
REPO_DIRTY="$(git -C "${REPO_DIR}" status --porcelain 2>/dev/null | wc -l)"
echo "repo ${REPO_DIR} HEAD=${REPO_HEAD} dirty_files=${REPO_DIRTY}"
if [ -n "${EXPECTED_COMMIT:-}" ] && [ "${REPO_HEAD}" != "${EXPECTED_COMMIT}" ]; then
  echo "FATAL: EXPECTED_COMMIT=${EXPECTED_COMMIT} but repo HEAD=${REPO_HEAD} — refusing to run stale code" >&2
  exit 1
fi

W8_LABEL_SEED="${W8_LABEL_SEED:-6000}"          # MUST stay out of eval band 1000..1029
W8_LABEL_EPISODES="${W8_LABEL_EPISODES:-700}"
W8_LABEL_M="${W8_LABEL_M:-16}"
LABEL_OUT="${RESULTS_DIR}/wave8_labels/labels"

echo "=== wave8-label | seed=${W8_LABEL_SEED} episodes=${W8_LABEL_EPISODES} M=${W8_LABEL_M} procs=${SLURM_CPUS_PER_TASK} at $(date -Is) ==="
RC=0
python3 -u -m agent_bullwhip.decision_labeler \
  --grid w8_g26b \
  --seed "${W8_LABEL_SEED}" \
  --episodes "${W8_LABEL_EPISODES}" \
  --m-futures "${W8_LABEL_M}" \
  --processes "${SLURM_CPUS_PER_TASK}" \
  --hash-rows 2000 \
  --out "${LABEL_OUT}" || RC=$?

echo "=== labeler exit=${RC} at $(date -Is) ==="
if [ "${RC}" -ne 0 ]; then
  exit "${RC}"
fi
python3 - "${LABEL_OUT}.manifest.json" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
for k in ("grid_id", "n_states", "flat_rate", "top_edge_rate", "zero_rate",
          "target_mass_top_letter", "edge_argmin_rate", "mean_target_entropy_nats",
          "hash", "wall_sec"):
    print(f"{k}: {m.get(k)}")
PY
