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

KIT_DIR="${W8_KIT_DIR:-}"
if [ -z "${KIT_DIR}" ] || [ ! -r "${KIT_DIR}/config.sh" ]; then
  KIT_DIR="${SLURM_SUBMIT_DIR:-}"
fi
if [ -z "${KIT_DIR}" ] || [ ! -r "${KIT_DIR}/config.sh" ]; then
  KIT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
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
# 2026-11-19: this cluster does NOT carry the submitting shell environment into the job.
# `W8_RUNG=1.7b sbatch gpu-wave8-preflight.sbatch` therefore ran as the 0.6b default and
# silently re-froze the 0.6B gate (verified with a two-way probe: --export=ALL works, the
# prefix form does not). Every knob must travel with --export=ALL,VAR=value, and an unset
# rung now says so out loud instead of looking like a choice.
if [ -z "${W8_RUNG:-}" ]; then
  echo "WARN: W8_RUNG not in the job environment -> defaulting to 0.6b." >&2
  echo "WARN: submit with sbatch --export=ALL,W8_RUNG=<0.6b|1.7b|4b|8b> ..." >&2
fi
bw_w8_rung

REPO_HEAD="$(git -C "${REPO_DIR}" rev-parse HEAD 2>/dev/null || echo unknown)"
echo "repo ${REPO_DIR} HEAD=${REPO_HEAD} rung=${W8_RUNG:-0.6b}"
: "${EXPECTED_COMMIT:?FATAL: EXPECTED_COMMIT not exported — set it from the hub (git rev-parse HEAD) before submitting}"
EXPECTED_COMMIT="$(git -C "${REPO_DIR}" rev-parse "${EXPECTED_COMMIT}" 2>/dev/null || echo "${EXPECTED_COMMIT}")"
# 2026-11-19: the retention guard commits results to this same branch, so a queued job
# legitimately sees HEAD ahead of the pinned commit. Refuse only when CODE differs: the
# pin must be an ancestor and every path changed since it must sit under results/ or docs/.
if [ "${REPO_HEAD}" != "${EXPECTED_COMMIT}" ]; then
  if git -C "${REPO_DIR}" merge-base --is-ancestor "${EXPECTED_COMMIT}" "${REPO_HEAD}" 2>/dev/null; then
    moved="$(git -C "${REPO_DIR}" diff --name-only "${EXPECTED_COMMIT}" "${REPO_HEAD}" -- . ':(exclude)results' ':(exclude)docs' | head -5)"
    if [ -z "${moved}" ]; then
      echo "NOTE: HEAD ${REPO_HEAD} is ahead of pin ${EXPECTED_COMMIT} in results/docs only — running."
    else
      echo "FATAL: code differs from pinned ${EXPECTED_COMMIT}: ${moved}" >&2
      exit 1
    fi
  else
    echo "FATAL: pinned ${EXPECTED_COMMIT} is not an ancestor of HEAD ${REPO_HEAD}" >&2
    exit 1
  fi
fi
[ -s "${RUNG_GATES}" ] || { echo "FATAL: ${RUNG_GATES} missing (F3 not frozen for this rung)" >&2; exit 1; }

OUT="results/eval"
W8DIR="results/wave8"
mkdir -p "${OUT}" "${W8DIR}"
FAILED=0
EVAL_ARMS="${W8_EVAL_ARMS:-r1}"
SEEDS="${W8_EVAL_SEEDS:-0 1 2}"
# knobs travel via --export=ALL, where a comma is the separator, so a seed list
# with spaces cannot be passed. Accept W8_EVAL_SEEDS=0:1 as well.
SEEDS="${SEEDS//:/ }"
SUF="${RUNG_SUFFIX}"
COMMON=(--model "${RUNG_MODEL}" --revision "${RUNG_REV}" --grid w8_g26b)
PERM_CASES="${W8_PERM_CASES:-results/wave8/S_nom.jsonl}"
LABELS_FOR_CAL=""
[ -s results/wave8_labels/labels.jsonl ] && LABELS_FOR_CAL="results/wave8_labels/labels.jsonl"

# --- precondition: an lr can only be selected from a validation band that exists -------------
# Each R1 arm writes its band entry (w8-val-r1<SUF>-<lrtag>-seed<S>.jsonl) in its FINAL `validate`
# stage, i.e. only after that arm's round 3 finishes. With no band, `wave8_analysis.py select-lr`
# raises SystemExit — which used to cost 6 minutes of control evals before the job died with a
# one-line reason. Fail in two seconds instead, and say why.
if [ -z "${W8_LR:-}" ] && [[ " ${EVAL_ARMS} " == *" r1 "* ]]; then
  BAND=(results/wave8_val/w8-val-r1${SUF}-*.jsonl)
  if [ ! -e "${BAND[0]}" ]; then
    echo "FATAL: no validation band — results/wave8_val/w8-val-r1${SUF}-*.jsonl does not exist." >&2
    echo "       An R1 arm only writes it in its final 'validate' stage, after round 3 completes." >&2
    echo "       If the train array is still running, wait for it. A collapse stop-loss ends an arm" >&2
    echo "       BEFORE its validate stage, so a stopped arm never populates the band." >&2
    if command -v squeue >/dev/null 2>&1; then
      echo "       bw8-train tasks still in the queue:" >&2
      squeue -u "${USER}" -h -o "         %i %T %M %j" 2>/dev/null | grep -i bw8 \
        || echo "         (none — so the band is missing for a different reason)"
    fi
    echo "       Workaround, only if a single lr has round-3 adapters:" >&2
    echo "         W8_LR=<lr> W8_EVAL_SEEDS=\"<seeds with adapters>\" sbatch gpu-wave8-eval.sbatch" >&2
    exit 1
  fi
  echo "validation band: $(ls -1 results/wave8_val/w8-val-r1${SUF}-*.jsonl | wc -l) file(s)"
  ls -1 results/wave8_val/w8-val-r1${SUF}-*.jsonl | sed 's/^/  /'
  NLRS="$(basename -a results/wave8_val/w8-val-r1${SUF}-*.jsonl \
          | sed -E 's/^.*-(lr[0-9.]+e[-+]?[0-9]+)-seed[0-9]+\.jsonl$/\1/' | sort -u | wc -l)"
  if [ "${NLRS}" -lt 2 ]; then
    echo "NOTE: the band covers ${NLRS} lr — select-lr then has nothing to compare (a single" >&2
    echo "      candidate is not a selection; say so in the write-up if that is what you run)." >&2
  fi
fi

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

# --- untrained curriculum control for the validation band (argmax, runs 30, seeds 9500..9529;
#     without it the band numbers have no untrained counterpart — W8-R1-DRAINED.md §8) ---------
echo "--- untrained curriculum band control (argmax, runs 30, seeds 9500..9529) at $(date -Is)"
if python3 -u -m agent_bullwhip.decision_eval "${COMMON[@]}" --outdir "results/wave8_val" \
    --tag "untrained-curriculum" --model-name "qwen3-untrained${SUF}" \
    --pattern curriculum --mode argmax --runs 30 --seed-base 9500 --traces 1; then
  echo "--- ok untrained curriculum control"
else
  echo "--- STEP FAILED: untrained curriculum control rc=$?" >&2
  FAILED=1
fi

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
  R2_LR="${W8_R2_LR:?FATAL: r2 arms need W8_R2_LR}"
  R2_LR="${R2_LR#lr}"   # select-sft-lr returns the tag 'lr2e-4'; float() wants the plain form
  R2_TAG="lr$(python3 -c "print(('%g' % float('${R2_LR}')).replace('e-0','e-').replace('e+0','e+'))")"
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
