#!/bin/bash
# =============================================================================
# gpu-wave8-train.sbatch — WAVE8 training (docs/waves/WAVE8_SPEC.md §4, §5), one L4 per array task.
#
#   W8_ARM=r1     (default) outcome-scored RL from the untrained readout, 3 rounds. Array tasks
#                 enumerate (lr, seed): idx -> LR=W8_LRS[idx / W8_SEEDS], SEED=idx % W8_SEEDS.
#                 Default `sbatch --array=0-5` = lrs {1e-5, 5e-5} x seeds {0,1,2}.
#   W8_ARM=r2     soft-target SFT warm start (one full epoch) + post-hoc gate reads; the R2+RL
#                 rounds anchored to the frozen SFT adapter run ONLY with W8_R2_RL=1 (audit
#                 2026-10-08: SFT-first, RL gated on the SFT reads — the reference recipe for this
#                 model class is SFT-only). W8_R2_LR required; array 0-2 = seeds.
#   W8_ARM=r2sweep  the 500-step SFT lr sweep (seed 0; lrs 1e-5, 1e-4, 2e-4; array 0-2).
#   W8_RUNG=0.6b|1.7b|4b  model rung (config.sh pins); 4B: set W8_MICRO_BATCH (smoke first!).
#
#   Submit:  cd /scratch-shared/$USER/bullwhip/kit && sbatch --array=0-5 gpu-wave8-train.sbatch
#            W8_ARM=r2 W8_R2_LR=1e-4 sbatch --array=0-2 gpu-wave8-train.sbatch
#            W8_RUNG=1.7b W8_LRS=5e-5 sbatch --array=0-2 gpu-wave8-train.sbatch
#            W8_RUNG=4b W8_LRS=5e-5 W8_SEEDS=2 W8_MICRO_BATCH=8 sbatch --array=0-1 gpu-wave8-train.sbatch
#   Needs:   results/wave8/gates.json with F0+F1 passing and the rung's F3 frozen
#            (gpu-wave8-preflight.sbatch); R2 also needs results/wave8_labels/labels.jsonl.
#   Resume:  resubmit the same task; round_state.json + --resume continue at the interrupted stage.
#            Exit 75 = SIGTERM-partial (resubmit). A STOPPED_collapse.json in the arm dir = the
#            two-consecutive-collapse stop-loss fired (exit 0, by design).
# =============================================================================
#SBATCH --job-name=bw8-train
#SBATCH --partition=tue.gpu.q
#SBATCH --gres=gpu:1
#SBATCH --constraint=l4
#SBATCH --time=08:00:00
#SBATCH --signal=B:TERM@600
#SBATCH --array=0-5
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=/scratch-shared/%u/bullwhip/logs/wave8-train_%A_%a.log
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
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "${LOG_DIR}"
cd "${REPO_DIR}"
bw_w8_rung

REPO_HEAD="$(git -C "${REPO_DIR}" rev-parse HEAD 2>/dev/null || echo unknown)"
REPO_DIRTY="$(git -C "${REPO_DIR}" status --porcelain 2>/dev/null | wc -l)"
echo "repo ${REPO_DIR} HEAD=${REPO_HEAD} dirty_files=${REPO_DIRTY}"
: "${EXPECTED_COMMIT:?FATAL: EXPECTED_COMMIT not exported — set it from the hub (git rev-parse HEAD) before submitting}"
EXPECTED_COMMIT="$(git -C "${REPO_DIR}" rev-parse "${EXPECTED_COMMIT}" 2>/dev/null || echo "${EXPECTED_COMMIT}")"
if [ "${REPO_HEAD}" != "${EXPECTED_COMMIT}" ]; then
  echo "FATAL: EXPECTED_COMMIT=${EXPECTED_COMMIT} but repo HEAD=${REPO_HEAD} — refusing to run stale code" >&2
  exit 1
fi
if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
  echo "this script is an array job: sbatch --array=0-5 $0" >&2
  exit 2
fi

# knobs
W8_ARM="${W8_ARM:-r1}"
W8_LRS="${W8_LRS:-1e-5 5e-5}"
W8_SEEDS="${W8_SEEDS:-3}"
W8_ROUNDS="${W8_ROUNDS:-3}"
W8_STEPS_PER_ROUND="${W8_STEPS_PER_ROUND:-170}"
W8_BATCH="${W8_BATCH:-32}"
W8_MAX_LEN="${W8_MAX_LEN:-1024}"
W8_GRAD_CKPT="${W8_GRAD_CKPT:-1}"
W8_SAVE_EVERY="${W8_SAVE_EVERY:-10}"
W8_MICRO_BATCH="${W8_MICRO_BATCH:-0}"
W8_SFT_STEPS="${W8_SFT_STEPS:-2827}"
W8_R2_RL="${W8_R2_RL:-0}"          # 1 = also run the R2+RL rounds in the same job (gate them on the SFT reads first)

# pre-flight gates: F0 + F1 (hub) and this rung's frozen F3 must be on disk and passing
python3 - "${RUNG_GATES}" <<'PY' || exit 1
import json, sys
base = json.load(open("results/wave8/gates.json")) if __import__("os").path.exists("results/wave8/gates.json") else {}
rung = json.load(open(sys.argv[1])) if __import__("os").path.exists(sys.argv[1]) else {}
bad = [k for k in ("F0", "F1") if not ((base.get(k) or {}).get("verdict") or {}).get("pass")]
if bad:
    sys.exit(f"FATAL: pre-flight {bad} not passing in results/wave8/gates.json — nothing is spent before F0-F3 pass")
if not rung.get("F3") or not (rung.get("F2") or {}).get("prompt_sha_equal"):
    sys.exit(f"FATAL: {sys.argv[1]} lacks F2/F3 — run gpu-wave8-preflight.sbatch first (W8_RUNG matches?)")
print("pre-flight ok:", {k: base[k]["verdict"]["pass"] for k in ("F0", "F1")}, "F3 frozen in", sys.argv[1])
PY

IDX="${SLURM_ARRAY_TASK_ID}"
LRS=(${W8_LRS})

# Slurm SIGTERM (TERM@600) -> python child -> checkpoint -> exit 75 / 0; loop until it is really gone
run_py() {
  python3 -u "$@" &
  local pid=$! rc=0
  trap 'echo "[$(date -Is)] SIGTERM from Slurm -> forwarding to python ${pid}"; kill -TERM "${pid}" 2>/dev/null || true' TERM
  while kill -0 "${pid}" 2>/dev/null; do
    if wait "${pid}"; then rc=0; else rc=$?; fi
  done
  trap - TERM
  return "${rc}"
}

COMMON=(--model "${RUNG_MODEL}" --revision "${RUNG_REV}")
RC=0
case "${W8_ARM}" in
  r1)
    LR_IDX=$((IDX / W8_SEEDS)); SEED=$((IDX % W8_SEEDS)); LR="${LRS[${LR_IDX}]:-}"
    [ -n "${LR}" ] || { echo "FATAL: task ${IDX} -> lr index ${LR_IDX} outside W8_LRS='${W8_LRS}'" >&2; exit 1; }
    echo "=== wave8 r1 task ${IDX}: lr=${LR} seed=${SEED} rung=${W8_RUNG:-0.6b} rounds=${W8_ROUNDS}x${W8_STEPS_PER_ROUND} at $(date -Is) ==="
    run_py -m agent_bullwhip.decision_rounds run --arm r1 --seed "${SEED}" --lr "${LR}" "${COMMON[@]}" \
      --rung-suffix="${RUNG_SUFFIX}" --rounds "${W8_ROUNDS}" --steps-per-round "${W8_STEPS_PER_ROUND}" \
      --batch "${W8_BATCH}" --micro-batch "${W8_MICRO_BATCH}" --max-length "${W8_MAX_LEN}" \
      --grad-ckpt "${W8_GRAD_CKPT}" --save-every "${W8_SAVE_EVERY}" \
      --processes "${SLURM_CPUS_PER_TASK}" --gates "${RUNG_GATES}" || RC=$?
    echo "=== r1 seed=${SEED} lr=${LR} exit=${RC} at $(date -Is) ==="
    ;;
  r2)
    SEED="${IDX}"; LR="${W8_R2_LR:?FATAL: W8_ARM=r2 needs W8_R2_LR (from select-sft-lr)}"
    LR="${LR#lr}"   # select-sft-lr returns the tag 'lr2e-4'; the runners want the plain float '2e-4'
    python3 -c "float('${LR}')" 2>/dev/null \
      || { echo "FATAL: W8_R2_LR='${W8_R2_LR}' -> lr '${LR}' is not a float" >&2; exit 1; }
    LABELS="${RESULTS_DIR}/wave8_labels/labels.jsonl"
    [ -s "${LABELS}" ] || { echo "FATAL: ${LABELS} missing — run cpu-wave8-label.sbatch first" >&2; exit 1; }
    SFT_DIR="results/wave8_r2sft_seed${SEED}"
    echo "=== wave8 r2 seed=${SEED} lr=${LR}: SFT ${W8_SFT_STEPS} steps at $(date -Is) ==="
    run_py -m agent_bullwhip.decision_rounds sft --labels "${LABELS}" --seed "${SEED}" --lr "${LR}" \
      --steps "${W8_SFT_STEPS}" --save-every 250 --out "${SFT_DIR}" "${COMMON[@]}" \
      --max-length "${W8_MAX_LEN}" --micro-batch "${W8_MICRO_BATCH}" || RC=$?
    echo "=== r2 sft exit=${RC} at $(date -Is) ==="
    [ "${RC}" -eq 0 ] || exit "${RC}"
    run_py -m agent_bullwhip.decision_rounds sft-gates --sft-dir "${SFT_DIR}" --gates "${RUNG_GATES}" \
      "${COMMON[@]}" || RC=$?
    [ "${RC}" -eq 0 ] || { echo "sft-gates failed rc=${RC}" >&2; exit "${RC}"; }
    if [ "${W8_R2_RL}" = "1" ]; then
      SFT_FINAL="${SFT_DIR}/$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['checkpoint_dir'])" "${SFT_DIR}/state.json")"
      echo "=== wave8 r2rl seed=${SEED}: RL rounds from ${SFT_FINAL} at $(date -Is) ==="
      run_py -m agent_bullwhip.decision_rounds run --arm r2rl --seed "${SEED}" --lr "${LR}" "${COMMON[@]}" \
        --base-adapter "${SFT_FINAL}" --rung-suffix="${RUNG_SUFFIX}" --rounds "${W8_ROUNDS}" \
        --steps-per-round "${W8_STEPS_PER_ROUND}" --batch "${W8_BATCH}" --micro-batch "${W8_MICRO_BATCH}" \
        --max-length "${W8_MAX_LEN}" --grad-ckpt "${W8_GRAD_CKPT}" --save-every "${W8_SAVE_EVERY}" \
        --processes "${SLURM_CPUS_PER_TASK}" --gates "${RUNG_GATES}" || RC=$?
      echo "=== r2rl seed=${SEED} exit=${RC} at $(date -Is) ==="
    else
      echo "=== r2 SFT-only (W8_R2_RL=0): read sft-gates + eval on the hub before scheduling the RL rounds ==="
    fi
    ;;
  r2sweep)
    SWEEP=(1e-5 1e-4 2e-4); LR="${SWEEP[${IDX}]:-}"
    [ -n "${LR}" ] || { echo "FATAL: r2sweep tasks are 0-2" >&2; exit 1; }
    LABELS="${RESULTS_DIR}/wave8_labels/labels.jsonl"
    [ -s "${LABELS}" ] || { echo "FATAL: ${LABELS} missing — run cpu-wave8-label.sbatch first" >&2; exit 1; }
    TAG="lr${LR}"
    echo "=== wave8 r2sweep lr=${LR}: 500 SFT steps, val every 100 at $(date -Is) ==="
    run_py -m agent_bullwhip.decision_rounds sft --labels "${LABELS}" --seed 0 --lr "${LR}" --steps 500 \
      --val-every 100 --save-every 250 --out "results/wave8_r2sft_lrsweep_${TAG}" "${COMMON[@]}" \
      --max-length "${W8_MAX_LEN}" || RC=$?
    echo "=== r2sweep exit=${RC} at $(date -Is) ==="
    ;;
  *) echo "FATAL: W8_ARM must be r1|r2|r2sweep (got '${W8_ARM}')" >&2; exit 2 ;;
esac

if [ "${RC}" -ne 0 ]; then
  [ "${RC}" -eq 75 ] && echo "EXIT 75: SIGTERM-partial — resubmit this array task to continue"
  exit "${RC}"
fi
ls -d results/wave8_*"${SEED:-}"* 2>/dev/null | head -20 || true
for st in results/wave8_*/STOPPED_collapse.json; do
  [ -e "${st}" ] && echo "NOTE: stop-loss marker ${st}"
done
exit 0
