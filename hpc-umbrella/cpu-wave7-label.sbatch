#!/bin/bash
# =============================================================================
# cpu-wave7-label.sbatch — decision_labeler on the CPU partition (no GPU needed).
#
# Produces the wave-7 training labels (spec D3/D7): N = episodes x 136 states
# (36 weeks - 2 late-week exclusions, x4 roles) of CRN E[J] over M futures.
# Throughput measured on the hub: ~0.5 s/state/core at M=16 -> 700 episodes
# (~95k states) is ~14 core-hours -> one 96-thread node finishes in minutes.
#
# Determinism: the labeler derives every rng from stable SHA strings, so the
# output must be byte-identical to a hub run of the same seed/episodes/M. The
# job prints the manifest hash; compare it with the hub's:
#     python -m agent_bullwhip.decision_labeler --seed 6000 --episodes 700 \
#         --out /tmp/hub_labels --hash-rows 2000
#     python3 -c "import json;print(json.load(open('/tmp/hub_labels.manifest.json'))['hash'])"
#
#   Submit:  cd /scratch-shared/$USER/bullwhip/kit && sbatch cpu-wave7-label.sbatch
#   Results: results/wave7_labels/{labels.jsonl,labels.npz,labels.manifest.json}
# =============================================================================
#SBATCH --job-name=bw7-label
#SBATCH --partition=tue.cpu3.q
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=16G
#SBATCH --output=/scratch-shared/%u/bullwhip/logs/wave7-label_%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=umbrella-cluster@pepijnfrenken.nl

set -euo pipefail

# Slurm runs batch scripts from its own spool dir (/var/spool/slurmd/jobNNN/slurm_script),
# so $BASH_SOURCE does NOT point at this file. The kit dir is where `sbatch` was invoked
# from (runbook: cd <kit> && sbatch <file>); BASH_SOURCE is only the manual-run fallback.
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
export HF_HUB_OFFLINE=1            # compute nodes have no egress (AUDIT-1 6 P3)
mkdir -p "${LOG_DIR}"
cd "${REPO_DIR}"

REPO_HEAD="$(git -C "${REPO_DIR}" rev-parse HEAD 2>/dev/null || echo unknown)"
REPO_DIRTY="$(git -C "${REPO_DIR}" status --porcelain 2>/dev/null | wc -l)"
echo "repo ${REPO_DIR} HEAD=${REPO_HEAD} dirty_files=${REPO_DIRTY}"
if [ -n "${EXPECTED_COMMIT:-}" ] && [ "${REPO_HEAD}" != "${EXPECTED_COMMIT}" ]; then
  echo "FATAL: EXPECTED_COMMIT=${EXPECTED_COMMIT} but repo HEAD=${REPO_HEAD} — refusing to run stale code" >&2
  exit 1
fi

# Knobs (override on the sbatch command line: W7_LABEL_EPISODES=100 sbatch ...)
W7_LABEL_SEED="${W7_LABEL_SEED:-6000}"        # MUST stay out of eval band 1000..1029
W7_LABEL_EPISODES="${W7_LABEL_EPISODES:-700}" # ~95k states
W7_LABEL_M="${W7_LABEL_M:-16}"                # CRN futures per state (>=16, AUDIT-1 P1)
LABEL_OUT="${RESULTS_DIR}/wave7_labels/labels"

echo "=== wave7-label | seed=${W7_LABEL_SEED} episodes=${W7_LABEL_EPISODES} M=${W7_LABEL_M} procs=${SLURM_CPUS_PER_TASK} at $(date -Is) ==="
echo "python: $(command -v python3) $(python3 -V 2>&1)"

python3 -u -m agent_bullwhip.decision_labeler \
  --seed "${W7_LABEL_SEED}" \
  --episodes "${W7_LABEL_EPISODES}" \
  --m-futures "${W7_LABEL_M}" \
  --processes "${SLURM_CPUS_PER_TASK}" \
  --hash-rows 2000 \
  --out "${LABEL_OUT}"

echo "=== labeler exit=$? at $(date -Is) ==="
echo "manifest:"
python3 - <<'PY'
import json
m = json.load(open("${LABEL_OUT}.manifest.json"))
for k in ("n_states", "flat_rate", "edge_argmin_rate",
          "mean_target_entropy_nats", "hash", "continuation_theta",
          "continuation_lam", "wall_sec"):
    print(f"{k}: {m.get(k)}")
if m.get("edge_argmin_rate", 1.0) > 0.9:
    print("WARN: edge-argmin rate very high — most targets sit at the grid "
          "boundary; consider lowering --p-random (extreme behavior states)")
PY