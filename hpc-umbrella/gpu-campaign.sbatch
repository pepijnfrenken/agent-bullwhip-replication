#!/bin/bash
# =============================================================================
# gpu-campaign.sbatch — the 13-config ablation as a Slurm job array (one task = one config).
#
# Shape SCC/Umbrella likes: independent single-GPU runs, not one distributed job.
# Sizing: ~5 GPU-h per 300-step 0.6B run => ~65 GPU-h for the sweep (fairshare).
#
#   Submit:  cd /scratch-shared/$USER/bullwhip/kit
#            sbatch --array=0-12 gpu-campaign.sbatch     # rows 2..14 of campaign.tsv
#   Watch:   squeue --me
#   Results: results/grpo_<name>/  +  logs/campaign_<ArrayJob>_<Task>.log
#
# Walltime: 18h is requested, but the L4 partitions may cap lower (config.sh --check
# tests this). If the cap is under the needed time, lower STEPS_CAMPAIGN — every run
# checkpoints (--save-every) and resumes (--resume), so a split run is not lost work.
# Slurm sends SIGTERM TERM_GRACE_SECONDS before the kill; the trap below forwards it
# to python, which checkpoints and exits. Resubmit the same array task to continue.
# =============================================================================
#SBATCH --job-name=bw-campaign
#SBATCH --partition=tue.gpu3.q
#SBATCH --gres=gpu:1
#SBATCH --constraint=l4
#SBATCH --time=18:00:00
#SBATCH --signal=B:TERM@600
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/scratch-shared/%u/bullwhip/logs/campaign_%A_%a.log

set -euo pipefail

KIT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=config.sh
. "${KIT_DIR}/config.sh"

bw_load_env
mkdir -p "${LOG_DIR}"
cd "${REPO_DIR}"

if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
  echo "this script is an array job: sbatch --array=0-12 $0" >&2
  exit 2
fi
ROWS=$((SLURM_ARRAY_TASK_ID + 2))                    # +2 = skip the header row, 1-index
LINE=$(sed -n "${ROWS}p" "${CAMPAIGN_TSV}")
if [ -z "${LINE}" ]; then
  echo "no config row ${ROWS} for task ${SLURM_ARRAY_TASK_ID} in ${CAMPAIGN_TSV}" >&2
  exit 1
fi
IFS=$'\t' read -r NAME GROUP_ HORIZON_ CURRICULA_ LR_ BETA_KL_ STEPS_ <<< "${LINE}"
echo "=== task ${SLURM_ARRAY_TASK_ID} -> ${NAME} | group=${GROUP_} horizon=${HORIZON_} curricula=${CURRICULA_} lr=${LR_} beta_kl=${BETA_KL_} steps=${STEPS_} rollout_batch=${ROLLOUT_BATCH} save_every=${SAVE_EVERY} (${CAMPAIGN_TSV}) ==="

nvidia-smi --query-gpu=name,memory.total --format=csv || true

# SIGTERM (Slurm walltime) -> python -> checkpoint -> clean exit; then resubmit.
python3 -u -m agent_bullwhip.grpo_train \
  --model "${MODEL_ID}" \
  --steps "${STEPS_}" --group "${GROUP_}" --horizon "${HORIZON_}" \
  --curricula "${CURRICULA_}" --lr "${LR_}" --beta-kl "${BETA_KL_}" \
  --rollout-batch "${ROLLOUT_BATCH}" --save-every "${SAVE_EVERY}" \
  --out "results/grpo_${NAME}" --resume &
PY_PID=$!
trap 'echo "[$(date -Is)] SIGTERM from Slurm -> forwarding to python ${PY_PID}"; kill -TERM "${PY_PID}" 2>/dev/null || true' TERM
RC=0
wait "${PY_PID}" || RC=$?
trap - TERM

echo "=== ${NAME} exit=${RC} at $(date -Is) ==="
ls -la "results/grpo_${NAME}" 2>/dev/null || echo "NO OUTPUT DIR - check the log above"
exit "${RC}"
