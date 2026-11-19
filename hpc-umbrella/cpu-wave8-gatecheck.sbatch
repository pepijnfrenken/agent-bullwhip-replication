#!/bin/bash
# =============================================================================
# cpu-wave8-gatecheck.sbatch — refuse to train on a drifted gate.
#
# Runs scripts/verify_rung_gates.py against the re-frozen 1.7B and 4B gate files
# and exits non-zero on any deviation from the values pre-registered in
# docs/waves/W8-RERUN-LOG.md. Submit it with a Slurm dependency so training can
# never start on a mismatched gate:
#
#   sbatch --dependency=afterok:<preflight_1.7b>:<preflight_4b> cpu-wave8-gatecheck.sbatch
#   GATE_JOB=<that job id>
#   W8_RUNG=1.7b W8_LRS=1e-5 W8_SEEDS=3 sbatch --dependency=afterok:${GATE_JOB} --array=0-2 gpu-wave8-train.sbatch
#
# Why a separate job: the dependency chain then runs unattended, so a VPN
# session expiring overnight cannot leave the pipeline half-submitted.
# =============================================================================
#SBATCH --job-name=bw8-gatecheck
#SBATCH --partition=tue.cpu3.q
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=/scratch-shared/%u/bullwhip/logs/wave8-gatecheck_%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=umbrella-cluster@pepijnfrenken.nl

set -uo pipefail

KIT_DIR="${SLURM_SUBMIT_DIR:-}"
if [ -z "${KIT_DIR}" ] || [ ! -r "${KIT_DIR}/config.sh" ]; then
  KIT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
if [ ! -r "${KIT_DIR}/config.sh" ]; then
  echo "FATAL: config.sh not found in KIT_DIR=${KIT_DIR} — submit from the kit dir" >&2
  exit 1
fi
# shellcheck source=config.sh
. "${KIT_DIR}/config.sh"

bw_load_env
mkdir -p "${LOG_DIR}"
cd "${REPO_DIR}"

echo "repo ${REPO_DIR} HEAD=$(git rev-parse HEAD 2>/dev/null || echo unknown) at $(date -Is)"
echo "=== gate files under test ==="
ls -l results/wave8/gates_1.7b.json results/wave8/gates_4b.json 2>&1

python3 scripts/verify_rung_gates.py \
  --gates results/wave8/gates_1.7b.json --rung 1.7b \
  --gates results/wave8/gates_4b.json  --rung 4b \
  --json results/wave8/gate_verification.json
rc=$?
echo "gatecheck exit=${rc}"
if [ "${rc}" -ne 0 ]; then
  echo "REFUSING to let dependent training jobs start (Slurm afterok will cancel them)." >&2
fi
exit "${rc}"
