#!/bin/bash
# =============================================================================
# gpu-wave8-fsdp-probe.sbatch — FSDP acceptance probe for 8B LoRA (2 steps, 2x L4).
#
# Tests docs/waves/W8B-FSDP-SCOPE.md before any leg is attempted. Two ranks,
# micro-batch 1, no grad checkpointing. It is not enough for the process to
# exit: scripts/probe_fsdp_lora.py asserts that LoRA gradients are non-zero and
# finite, because a port that runs and trains nothing is the failure this
# project already has on record.
#
#   Submit:  cd /scratch-shared/$USER/bullwhip/kit
#            sbatch --export=ALL,EXPECTED_COMMIT=$(git -C ../agent-bullwhip-replication rev-parse HEAD) \
#                   gpu-wave8-fsdp-probe.sbatch
#
# Free lane, ~15 min including queue. Exit 0 = PROBE PASS, 1 = PROBE FAIL,
# 2 = refused (bad environment).
# =============================================================================
#SBATCH --job-name=bw8-fsdp-probe
#SBATCH --partition=tue.gpu.q
#SBATCH --gres=gpu:2
#SBATCH --constraint=l4
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/scratch-shared/%u/bullwhip/logs/wave8-fsdp-probe_%j.log
#SBATCH --mail-type=NONE
#SBATCH --mail-user=umbrella-cluster@pepijnfrenken.nl

set -euo pipefail

KIT_DIR="${W8_KIT_DIR:-}"
if [ -z "${KIT_DIR}" ] || [ ! -r "${KIT_DIR}/config.sh" ]; then
  # slurmd runs a spool copy, so BASH_SOURCE points at /var/spool/slurmd/job<id>
  # and not at where it was submitted from: SLURM_SUBMIT_DIR is the only reliable
  # anchor here (job 796150 died in 2 s learning this).
  KIT_DIR="${SLURM_SUBMIT_DIR:-}"
fi
if [ -z "${KIT_DIR}" ] || [ ! -r "${KIT_DIR}/config.sh" ]; then
  KIT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
fi
if [ ! -r "${KIT_DIR}/config.sh" ]; then
  echo "FATAL: config.sh not found in KIT_DIR=${KIT_DIR}" >&2
  exit 2
fi
# shellcheck source=config.sh
. "${KIT_DIR}/config.sh"

bw_load_env
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "${LOG_DIR}"
cd "${REPO_DIR}"

REPO_HEAD="$(git -C "${REPO_DIR}" rev-parse HEAD 2>/dev/null || echo unknown)"
echo "repo ${REPO_DIR} HEAD=${REPO_HEAD} ranks=${SLURM_NTASKS:-1}"
: "${EXPECTED_COMMIT:?FATAL: EXPECTED_COMMIT not exported — set it with --export=ALL}"
PINNED="$(git -C "${REPO_DIR}" rev-parse "${EXPECTED_COMMIT}" 2>/dev/null || echo "${EXPECTED_COMMIT}")"
if [ "${REPO_HEAD}" != "${PINNED}" ]; then
  if git -C "${REPO_DIR}" merge-base --is-ancestor "${PINNED}" "${REPO_HEAD}" 2>/dev/null; then
    moved="$(git -C "${REPO_DIR}" diff --name-only "${PINNED}" "${REPO_HEAD}" -- . ':(exclude)results' ':(exclude)docs' | head -5)"
    if [ -n "${moved}" ]; then
      echo "FATAL: code differs from pinned ${PINNED}: ${moved}" >&2
      exit 2
    fi
    echo "NOTE: HEAD is ahead of pin in results/docs only — running."
  else
    echo "FATAL: pinned ${PINNED} is not an ancestor of HEAD ${REPO_HEAD}" >&2
    exit 2
  fi
fi

export W8B_FSDP=1
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$(( 20000 + SLURM_JOB_ID % 20000 ))
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

echo "=== fsdp probe at $(date -Is) ranks=${SLURM_NTASKS} master=${MASTER_ADDR}:${MASTER_PORT} ==="
# Job 796505 died in FSDP's parameter-handle init and the exception never reached
# the log because the peer rank sat in a collective for 30 minutes. Three variants
# run in one slot so a queue wait buys a diagnosis, not a guess:
#   A auto-wrap, tied     (what failed)
#   B no auto-wrap, tied  (sidesteps the recursive wrap)
#   C auto-wrap, untied   (Qwen3 ties lm_head to embed_tokens)
export W8B_NCCL_TIMEOUT=120
PASSED=""
for V in "auto 0 A" "none 0 B" "auto 1 C"; do
  set -- $V
  export W8B_FSDP_WRAP="$1"
  export W8B_UNTIE="$2"
  TAG="$3"
  echo "--- variant ${TAG}: wrap=$1 untie=$2 ---"
  set +e
  srun --ntasks="${SLURM_NTASKS:-2}" --ntasks-per-node="${SLURM_NTASKS:-2}" \
    python3 scripts/probe_fsdp_lora.py 2>&1 | grep -vE "Loading checkpoint|it/s\]" | tail -30
  RCV=$?
  set -e
  echo "--- variant ${TAG}: rc=${RCV} ---"
  if [ "${RCV}" -eq 0 ]; then PASSED="${PASSED} ${TAG}"; fi
done
if [ -n "${PASSED}" ]; then
  echo "=== fsdp probe done: PASSING VARIANTS:${PASSED} ==="
  exit 0
fi
echo "=== fsdp probe done: ALL VARIANTS FAILED ==="
exit 1
