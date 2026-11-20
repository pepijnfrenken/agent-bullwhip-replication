#!/usr/bin/env bash
# =============================================================================
# w8-submit.sh — submit wave-8 jobs so the knobs actually reach the job.
#
# WHY: on this cluster `sbatch` does not carry the submitting shell's environment
# into the job. The documented form `W8_RUNG=1.7b sbatch gpu-wave8-train.sbatch`
# therefore runs as the 0.6b default, silently. Verified 2026-11-19 with a
# two-way probe: `sbatch --export=ALL,W8_RUNG=probeA x.sbatch` arrives,
# `W8_RUNG=probeB sbatch x.sbatch` arrives empty. This wrapper always uses
# --export=ALL and refuses to submit without an explicit rung.
#
# Usage (from the kit dir, e.g. /scratch-shared/$USER/bullwhip/kit):
#   bash w8-submit.sh preflight 1.7b
#   bash w8-submit.sh train     1.7b --lrs 1e-5 --seeds 3 --array 0-2
#   bash w8-submit.sh train     4b   --lrs 5e-5 --seeds 2 --array 0-1 --micro-batch 8
#   bash w8-submit.sh eval      4b   --seeds 0:1 --lr 5e-5 --dependency afterok:12345
#   bash w8-submit.sh check               # gate check over both rungs
#
# Every submit also exports EXPECTED_COMMIT (default: the repo HEAD as this script
# sees it), so the kit's own stale-checkout gate stays armed.
# =============================================================================
set -uo pipefail

KIT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=config.sh
. "${KIT_DIR}/config.sh"

STAGE="${1:-}"; shift 1 2>/dev/null || true
# `check` runs over both rungs and takes no rung argument; every other stage does.
if [ "${STAGE}" = "check" ]; then
  RUNG="both"
else
  RUNG="${1:-}"; shift 1 2>/dev/null || true
fi

LRS=""; SEEDS=""; ARRAY=""; MICRO=""; EVAL_LR=""; DEP=""; ARMS=""; EVAL_RUNS=""
while [ $# -gt 0 ]; do
  case "$1" in
    --lrs) LRS="$2"; shift 2 ;;
    --seeds) SEEDS="$2"; shift 2 ;;
    --array) ARRAY="$2"; shift 2 ;;
    --micro-batch) MICRO="$2"; shift 2 ;;
    --lr) EVAL_LR="$2"; shift 2 ;;
    --runs) EVAL_RUNS="$2"; shift 2 ;;
    --arms) ARMS="$2"; shift 2 ;;
    --dependency) DEP="$2"; shift 2 ;;
    *) echo "[w8-submit] unknown option: $1" >&2; exit 2 ;;
  esac
done

EXPORT="ALL,EXPECTED_COMMIT=$(git -C "${REPO_DIR}" rev-parse HEAD 2>/dev/null || echo unknown)"
add() { EXPORT="${EXPORT},$1"; }

case "${STAGE}" in
  check)
    # the gatecheck reads both rung files; the rung variable is irrelevant here.
    # The option loop already consumed --dependency into DEP, so rebuild the args.
    CHK_ARGS=(--export="${EXPORT}")
    [ -n "${DEP}" ] && CHK_ARGS+=(--dependency="${DEP}")
    echo "[w8-submit] check -> sbatch ${CHK_ARGS[*]} cpu-wave8-gatecheck.sbatch"
    exec sbatch "${CHK_ARGS[@]}" cpu-wave8-gatecheck.sbatch ;;
  preflight|train|eval) ;;
  ""|-h|--help)
    sed -n '2,25p' "${BASH_SOURCE[0]}"; exit 0 ;;
  *) echo "[w8-submit] stage must be preflight|train|eval|check (got '${STAGE}')" >&2; exit 2 ;;
esac

case "${RUNG}" in
  0.6b|1.7b|4b|8b|both) ;;
  *) echo "[w8-submit] rung must be 0.6b|1.7b|4b|8b (got '${RUNG}')" >&2; exit 2 ;;
esac
add "W8_RUNG=${RUNG}"

case "${STAGE}" in
  preflight)
    FILE="gpu-wave8-preflight.sbatch" ;;
  train)
    FILE="gpu-wave8-train.sbatch"
    [ -n "${LRS}" ] && add "W8_LRS=${LRS}"
    [ -n "${SEEDS}" ] && add "W8_SEEDS=${SEEDS}"
    [ -n "${MICRO}" ] && add "W8_MICRO_BATCH=${MICRO}" ;;
  eval)
    FILE="gpu-wave8-eval.sbatch"
    [ -n "${SEEDS}" ] && add "W8_EVAL_SEEDS=${SEEDS}"   # colon-separated: 0:1
    [ -n "${EVAL_LR}" ] && add "W8_LR=${EVAL_LR}"
    [ -n "${ARMS}" ] && add "W8_EVAL_ARMS=${ARMS}"
    [ -n "${EVAL_RUNS}" ] && add "W8_EVAL_RUNS=${EVAL_RUNS}" ;;
esac

ARGS=(--export="${EXPORT}")
[ -n "${DEP}" ] && ARGS+=(--dependency="${DEP}")
[ -n "${ARRAY}" ] && ARGS+=(--array="${ARRAY}")

echo "[w8-submit] ${STAGE} rung=${RUNG} export=${EXPORT}"
echo "[w8-submit] -> sbatch ${ARGS[*]} ${FILE}"
sbatch "${ARGS[@]}" "${FILE}"
