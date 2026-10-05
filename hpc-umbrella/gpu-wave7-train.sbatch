#!/bin/bash
# =============================================================================
# gpu-wave7-train.sbatch — wave-7 SFT + RLCD training as a 3-seed array on L4s.
#
# One array task per training seed {0,1,2} (spec D4). Each task:
#   1. SFT (soft-target CE, LoRA r=8/alpha=16 attn-only, permuted options) on
#      the labels from cpu-wave7-label.sbatch -> results/wave7_sft_seed<N>/
#   2. on-policy relabel round (DAgger-style, AUDIT-1 fix 8): roll the SFT
#      policy out on curriculum episodes (decision_eval --collect-states),
#      relabel the visited states (decision_labeler --states-jsonl)
#   3. RLCD from the SFT checkpoint on those states
#      -> results/wave7_rlcd_seed<N>/   (the SFT adapter is NOT mutated)
#
# Walltime 4h is generous (measured SFT throughput: ~1-2 s/batch on an L4 for
# batch 32). Slurm sends SIGTERM TERM_GRACE_SECONDS before the kill; the trap
# forwards it to python, which checkpoints and exits (--save-every / --resume).
# Resubmit the same array task to continue.
#
#   Submit:  cd /scratch-shared/$USER/bullwhip/kit && sbatch gpu-wave7-train.sbatch
#   Results: results/wave7_sft_seed{0,1,2}/ + results/wave7_rlcd_seed{0,1,2}/
#   Labels:  results/wave7_labels/ (run cpu-wave7-label.sbatch FIRST)
# =============================================================================
#SBATCH --job-name=bw7-train
#SBATCH --partition=tue.gpu.q
#SBATCH --gres=gpu:1
#SBATCH --constraint=l4
#SBATCH --time=04:00:00
#SBATCH --signal=B:TERM@600
#SBATCH --array=0-2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=/scratch-shared/%u/bullwhip/logs/wave7-train_%A_%a.log
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

SEED="${SLURM_ARRAY_TASK_ID}"                    # training seeds {0,1,2}
W7_LR="${W7_LR:-1e-5}"
W7_SFT_STEPS="${W7_SFT_STEPS:-120}"              # optimizer batches (batch 32)
W7_RLCD_STEPS="${W7_RLCD_STEPS:-40}"
W7_SAVE_EVERY="${W7_SAVE_EVERY:-10}"
W7_N_COLLECT="${W7_N_COLLECT:-40}"               # curriculum episodes for round-2
LABELS_JSONL="${RESULTS_DIR}/wave7_labels/labels.jsonl"
SFT_OUT="results/wave7_sft_seed${SEED}"
RLCD_OUT="results/wave7_rlcd_seed${SEED}"

if [ ! -f "${LABELS_JSONL}" ]; then
  echo "FATAL: labels not found at ${LABELS_JSONL} — run cpu-wave7-label.sbatch first" >&2
  exit 1
fi

# Record which label set is being trained on (canonical hub hash:
# 45d9f40d6ffc4ce93b0caaf7e9233c1e22e6b726efc980d92ffd8dc13e41f51b).
# Warn-only by default; set W7_EXPECT_LABEL_HASH=<hash> to make a mismatch fatal.
LABEL_MANIFEST="${RESULTS_DIR}/wave7_labels/labels.manifest.json"
if [ -f "${LABEL_MANIFEST}" ]; then
  LABEL_HASH="$(python3 -c "import json,sys;print(json.load(open(sys.argv[1])).get('hash','? '))" "${LABEL_MANIFEST}")"
  echo "labels: ${LABELS_JSONL} hash=${LABEL_HASH}"
  if [ -n "${W7_EXPECT_LABEL_HASH:-}" ] && [ "${LABEL_HASH}" != "${W7_EXPECT_LABEL_HASH}" ]; then
    echo "FATAL: label hash ${LABEL_HASH} != expected ${W7_EXPECT_LABEL_HASH}" >&2
    exit 1
  fi
else
  echo "WARN: no label manifest at ${LABEL_MANIFEST} — cannot report the label hash" >&2
fi

echo "=== wave7-train seed=${SEED} | sft_steps=${W7_SFT_STEPS} rlcd_steps=${W7_RLCD_STEPS} lr=${W7_LR} at $(date -Is) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv || true

# --- stage 1: SFT -----------------------------------------------------------
python3 -u -m agent_bullwhip.decision_train \
  --model "${MODEL_ID}" \
  --labels "${LABELS_JSONL}" \
  --phase sft \
  --steps "${W7_SFT_STEPS}" --batch 32 --lr "${W7_LR}" \
  --save-every "${W7_SAVE_EVERY}" \
  --seed "${SEED}" \
  --out "${SFT_OUT}" --resume &
PY_PID=$!
trap 'echo "[$(date -Is)] SIGTERM from Slurm -> forwarding to python ${PY_PID}"; kill -TERM "${PY_PID}" 2>/dev/null || true' TERM
RC=0
while kill -0 "${PY_PID}" 2>/dev/null; do
  if wait "${PY_PID}"; then RC=0; else RC=$?; fi
done
trap - TERM
echo "=== sft seed=${SEED} exit=${RC} at $(date -Is) ==="
if [ "${RC}" -ne 0 ]; then
  exit "${RC}"
fi

# final SFT adapter = highest checkpoint dir (resume-safe lookup)
# NB: keep the `|| true` — under `set -euo pipefail` a failing `ls` (no matches,
# exit 2) would otherwise kill the script silently, before the FATAL below.
SFT_FINAL="$(ls -d "${SFT_OUT}"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1 || true)"
if [ -z "${SFT_FINAL}" ]; then
  echo "FATAL: no sft checkpoint dirs under ${SFT_OUT}" >&2
  exit 1
fi
echo "sft final adapter: ${SFT_FINAL}"

# --- round 2: on-policy state collection + relabel (AUDIT-1 fix 8) -----------
POLICY_STATES="${RESULTS_DIR}/wave7_labels/policy_states_seed${SEED}.jsonl"
POLICY_LABELS="${RESULTS_DIR}/wave7_labels/policy_labels_seed${SEED}"
python3 -u -m agent_bullwhip.decision_eval \
  --model "${MODEL_ID}" \
  --adapter "${SFT_FINAL}" \
  --collect-states "${POLICY_STATES}" \
  --n-collect "${W7_N_COLLECT}" \
  --collect-seed "$((7000 + SEED))" || exit $?

python3 -u -m agent_bullwhip.decision_labeler \
  --states-jsonl "${POLICY_STATES}" \
  --seed "$((6000 + SEED))" \
  --processes "${SLURM_CPUS_PER_TASK}" \
  --hash-rows 2000 \
  --out "${POLICY_LABELS}" || exit $?

# --- stage 2: RLCD from the SFT checkpoint on the on-policy states ------------
python3 -u -m agent_bullwhip.decision_train \
  --model "${MODEL_ID}" \
  --labels "${POLICY_LABELS}.jsonl" \
  --phase rlcd \
  --base-adapter "${SFT_FINAL}" \
  --steps "${W7_RLCD_STEPS}" --batch 32 --lr "${W7_LR}" \
  --beta-kl 0.02 \
  --save-every "${W7_SAVE_EVERY}" \
  --seed "${SEED}" \
  --out "${RLCD_OUT}" --resume &
PY_PID=$!
trap 'echo "[$(date -Is)] SIGTERM from Slurm -> forwarding to python ${PY_PID}"; kill -TERM "${PY_PID}" 2>/dev/null || true' TERM
RC=0
while kill -0 "${PY_PID}" 2>/dev/null; do
  if wait "${PY_PID}"; then RC=0; else RC=$?; fi
done
trap - TERM
echo "=== rlcd seed=${SEED} exit=${RC} at $(date -Is) ==="

echo "=== wave7-train seed=${SEED} done at $(date -Is) ==="