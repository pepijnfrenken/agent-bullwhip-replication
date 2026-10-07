#!/bin/bash
# =============================================================================
# gpu-wave8-eval.sbatch — WAVE8 protocol evals + gate reads (docs/waves/WAVE8_SPEC.md §6-§8).
#
# Per rung (W8_RUNG=0.6b|1.7b|4b): the untrained control (C0) and every trained adapter of the
# selected arms, each as step-argmax (n=1), step-sampled (n=30), noisy-argmax (n=30) with the
# per-decision traces filled (--traces), plus the permutation probe and the collapse-final read
# (S_nom, argmax histogram, calibration). Adapters are the FINAL checkpoint of round 3 of the
# winning lr (selected on the curriculum-validation band 9500..9529 — never on eval paths).
#
#   W8_EVAL_ARMS="r1"            default; also: r2sft, r2rl, r3a, r3b, r3c (space separated)
#   W8_LR=1e-5 | 5e-5            override the validation-selected lr of r1 (family of the rung)
#   W8_R2_LR=1e-4                lr of the r2 arms (required if r2sft/r2rl are evaluated)
#   W8_EVAL_SEEDS="0 1 2"        seeds present (4B: "0 1")
#
#   Submit:  cd /scratch-shared/$USER/bullwhip/kit && sbatch gpu-wave8-eval.sbatch
#   Outputs: results/eval/w8-<arm>[-<rung>]-seed<S>-<mode>-<pattern>.jsonl (+ .summary.json),
#            results/eval/w8-<arm>-seed<S>-perm.json, results/wave8/<arm>-seed<S>-S_nom.json,
#            results/wave8/lr_selection*.json
# =============================================================================
#SBATCH --job-name=bw8-eval
#SBATCH --partition=tue.gpu.q
#SBATCH --gres=gpu:1
#SBATCH --constraint=l4
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=/scratch-shared/%u/bullwhip/logs/wave8-eval_%j.log
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
bw_w8_rung

REPO_HEAD="$(git -C "${REPO_DIR}" rev-parse HEAD 2>/dev/null || echo unknown)"
echo "repo ${REPO_DIR} HEAD=${REPO_HEAD} rung=${W8_RUNG:-0.6b}"
if [ -n "${EXPECTED_COMMIT:-}" ] && [ "${REPO_HEAD}" != "${EXPECTED_COMMIT}" ]; then
  echo "FATAL: EXPECTED_COMMIT=${EXPECTED_COMMIT} but repo HEAD=${REPO_HEAD} — refusing to run stale code" >&2
  exit 1
fi
[ -s "${RUNG_GATES}" ] || { echo "FATAL: ${RUNG_GATES} missing (F3 not frozen for this rung)" >&2; exit 1; }

OUT="results/eval"
W8DIR="results/wave8"
mkdir -p "${OUT}" "${W8DIR}"
FAILED=0
EVAL_ARMS="${W8_EVAL_ARMS:-r1}"
SEEDS="${W8_EVAL_SEEDS:-0 1 2}"
SUF="${RUNG_SUFFIX}"
COMMON=(--model "${RUNG_MODEL}" --revision "${RUNG_REV}" --grid w8_g26b)
PERM_CASES="${W8_PERM_CASES:-results/wave8/S_nom.jsonl}"
LABELS_FOR_CAL=""
[ -s results/wave8_labels/labels.jsonl ] && LABELS_FOR_CAL="results/wave8_labels/labels.jsonl"

final_ckpt() {  # final_ckpt <train-out-dir> -> path of the last adapter dir (state.json)
  python3 -c "import json,sys;print(sys.argv[1]+'/'+json.load(open(sys.argv[1]+'/state.json'))['checkpoint_dir'])" "$1"
}

ev() {  # ev <tag> <model-name> <pattern> <mode> [adapter]
  local tag="$1" mname="$2" pattern="$3" mode="$4" adapter="${5:-}"
  local aargs=()
  [ -n "${adapter}" ] && aargs=(--adapter "${adapter}")
  echo "--- eval ${tag} (${pattern}/${mode}) at $(date -Is)"
  if python3 -u -m agent_bullwhip.decision_eval "${COMMON[@]}" --outdir "${OUT}" --tag "${tag}" \
      --model-name "${mname}" --pattern "${pattern}" --mode "${mode}" --traces 1 \
      ${aargs[@]+"${aargs[@]}"}; then
    echo "--- ok ${tag}"
  else
    echo "--- STEP FAILED: ${tag} rc=$?" >&2
    FAILED=1
  fi
}

reads() {  # reads <arm-tag> <adapter|->   (permutation probe + collapse-final; failures recorded, not fatal)
  local arm="$1" adapter="$2"
  local aargs=()
  [ "${adapter}" != "-" ] && aargs=(--adapter "${adapter}")
  echo "--- perm probe ${arm} at $(date -Is)"
  python3 -u scripts/wave8_gates.py perm-check "${COMMON[@]}" --states "${PERM_CASES}" --n 512 \
    --gate 0.90 ${aargs[@]+"${aargs[@]}"} --out "${OUT}/w8-${arm}-perm.json" \
    || echo "--- PERM ${arm}: gate failed or errored (recorded in the JSON; continuing)" >&2
  echo "--- collapse-final ${arm} at $(date -Is)"
  python3 -u scripts/wave8_gates.py collapse-final "${COMMON[@]}" --arm "${arm}" --frozen "${RUNG_GATES}" \
    ${aargs[@]+"${aargs[@]}"} ${LABELS_FOR_CAL:+--labels "${LABELS_FOR_CAL}"} --out "${W8DIR}/${arm}-S_nom.json" \
    || { echo "--- STEP FAILED: collapse-final ${arm}" >&2; FAILED=1; }
}

run_arm() {  # run_arm <tag-prefix> <model-name> <adapter|->
  local prefix="$1" mname="$2" adapter="$3"
  local aarg=""
  [ "${adapter}" != "-" ] && aarg="${adapter}"
  for PM in "argmax step" "sampled step" "argmax noisy"; do
    set -- ${PM}
    ev "w8-${prefix}-$1-$2" "${mname}" "$2" "$1" "${aarg}"
  done
  reads "${prefix}" "${adapter}"
}

# --- untrained control for this rung (C0; label mass >= 0.99 stays fatal here) -------------
run_arm "untrained${SUF}" "qwen3-untrained${SUF}" "-"

# --- r1 lr: validation-selected on the curriculum band 9500..9529 (never eval paths) ---------
LR_TAG=""
if [[ " ${EVAL_ARMS} " == *" r1 "* ]]; then
  if [ -n "${W8_LR:-}" ]; then
    LR_TAG="lr$(python3 -c "print(('%g' % float('${W8_LR}')).replace('e-0','e-').replace('e+0','e+'))")"
    echo "r1 lr override: ${LR_TAG}"
  else
    python3 scripts/wave8_analysis.py select-lr --val-dir results/wave8_val --family "r1${SUF}" \
      --out "${W8DIR}/lr_selection${SUF}.json"
    LR_TAG="$(python3 -c "import json;print(json.load(open('${W8DIR}/lr_selection${SUF}.json'))['winner'])")"
  fi
  echo "r1${SUF} winner lr tag: ${LR_TAG}"
fi
R2_TAG=""
if [[ " ${EVAL_ARMS} " == *" r2rl "* || " ${EVAL_ARMS} " == *" r2sft "* ]]; then
  R2_TAG="lr$(python3 -c "print(('%g' % float('${W8_R2_LR:?FATAL: r2 arms need W8_R2_LR}')).replace('e-0','e-').replace('e+0','e+'))")"
fi

for ARM in ${EVAL_ARMS}; do
  for SEED in ${SEEDS}; do
    case "${ARM}" in
      r1)    D="results/wave8_r1${SUF}_${LR_TAG}_seed${SEED}_round3" ;;
      r2rl)  D="results/wave8_r2rl${SUF}_${R2_TAG}_seed${SEED}_round3" ;;
      r2sft) D="results/wave8_r2sft_seed${SEED}" ;;
      r3a|r3b|r3c) D="results/wave8_${ARM}${SUF}_${LR_TAG:-${R2_TAG}}_seed${SEED}_round3" ;;
      *) echo "FATAL: unknown arm ${ARM}" >&2; exit 2 ;;
    esac
    if [ ! -s "${D}/state.json" ]; then
      echo "FATAL: missing train output ${D}/state.json — run gpu-wave8-train.sbatch first" >&2
      exit 1
    fi
    ADP="$(final_ckpt "${D}")"
    run_arm "${ARM}${SUF}-seed${SEED}" "w8-${ARM}${SUF}-seed${SEED}" "${ADP}"
  done
done

echo "--- artifacts ---"
ls -la "${OUT}" | tail -30
echo "=== wave8-eval done at $(date -Is) failed=${FAILED} ==="
exit "${FAILED}"
