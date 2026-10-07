#!/bin/bash
# =============================================================================
# gpu-w6fix.sbatch — WAVE6-FIX contract micro-probe (docs/waves/WAVE6-FIX_SPEC.md):
# six independent array tasks, one per row of campaign-w6fix.tsv (A0, A0b, A1 x2 seeds,
# A2 x2 seeds). Clone of gpu-campaign.sbatch (same KIT_DIR block, EXPECTED_COMMIT gate,
# TERM forwarding, `_seed<N>` suffix -> --seed N).
#
#   Submit:  cd /scratch-shared/$USER/bullwhip/kit
#            sbatch --array=0-5 gpu-w6fix.sbatch        # rows 2..7 of campaign-w6fix.tsv
#   Pilot:   W6FIX_STEPS=4 sbatch --array=2 gpu-w6fix.sbatch   # first 4 steps of the real
#            A1 run (same --out; the full submit later resumes from step 4)
#   Watch:   squeue --me
#   Results: results/grpo_<name>/  +  logs/w6fix_<ArrayJob>_<Task>.log
#
# Every run checkpoints (--save-every) and resumes (--resume); Slurm sends SIGTERM
# TERM_GRACE_SECONDS before the kill and the trap below forwards it to python.
# =============================================================================
#SBATCH --job-name=bw-w6fix
#SBATCH --partition=tue.gpu.q
#SBATCH --gres=gpu:1
#SBATCH --constraint=l4
#SBATCH --time=14:00:00
#SBATCH --signal=B:TERM@600
#SBATCH --array=0-5
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=/scratch-shared/%u/bullwhip/logs/w6fix_%A_%a.log
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
mkdir -p "${LOG_DIR}"
cd "${REPO_DIR}"

# F6: stamp the exact code this task ran (the sweep once ran pre-fix code and
# nothing in the log said so). The stage.sh EXPECTED_COMMIT gate is the hard
# version of this; the echo is the soft record in every task log.
REPO_HEAD="$(git -C "${REPO_DIR}" rev-parse HEAD 2>/dev/null || echo unknown)"
REPO_DIRTY="$(git -C "${REPO_DIR}" status --porcelain 2>/dev/null | wc -l)"
echo "repo ${REPO_DIR} HEAD=${REPO_HEAD} dirty_files=${REPO_DIRTY}"
# D3 (audit §4 item 9 — the sbatch-side hard gate): stage.sh's EXPECTED_COMMIT
# is the login-side gate; this is the job-side one. Empty (config.sh default)
# keeps the stamp log-only; a full SHA makes a stale checkout FAIL the task.
if [ -n "${EXPECTED_COMMIT:-}" ] && [ "${REPO_HEAD}" != "${EXPECTED_COMMIT}" ]; then
  echo "FATAL: EXPECTED_COMMIT=${EXPECTED_COMMIT} but repo HEAD=${REPO_HEAD} (${REPO_DIR}) — refusing to run stale code" >&2
  exit 1
fi

if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
  echo "this script is an array job: sbatch --array=0-5 $0" >&2
  exit 2
fi
ROWS=$((SLURM_ARRAY_TASK_ID + 2))                    # +2 = skip the header row, 1-index
LINE=$(sed -n "${ROWS}p" "${CAMPAIGN_W6FIX_TSV}")
if [ -z "${LINE}" ]; then
  echo "no config row ${ROWS} for task ${SLURM_ARRAY_TASK_ID} in ${CAMPAIGN_W6FIX_TSV}" >&2
  exit 1
fi
IFS=$'\t' read -r NAME GROUP_ HORIZON_ CURRICULA_ LR_ BETA_KL_ STEPS_ PFMT_ MAXNEW_ PARSEAWARE_ FMTW_ <<< "${LINE}"
# pilot knob: W6FIX_STEPS=4 runs only the first 4 steps of the selected row (resumable)
STEPS_="${W6FIX_STEPS:-${STEPS_}}"
FMT_ARGS=()
[ "${PARSEAWARE_}" = "1" ] && FMT_ARGS=(--fmt-weight "${FMTW_}")
# F12: a NAME suffix `_seed<N>` becomes --seed <N> (campaign.tsv is data and is
# not edited here); a11_seed2 must no longer silently clone the seed-12345 run.
SEED_=""
case "${NAME}" in
  *_seed[0-9]*)
    SEED_="${NAME##*_seed}"
    SEED_="${SEED_%%[^0-9]*}"
    ;;
esac
SEED_ARGS=()
[ -n "${SEED_}" ] && SEED_ARGS=(--seed "${SEED_}")
echo "=== task ${SLURM_ARRAY_TASK_ID} -> ${NAME} | group=${GROUP_} horizon=${HORIZON_} curricula=${CURRICULA_} lr=${LR_} beta_kl=${BETA_KL_} steps=${STEPS_} rollout_batch=${ROLLOUT_BATCH} save_every=${SAVE_EVERY} seed=${SEED_:-<default 12345>} prompt_format=${PFMT_} max_new=${MAXNEW_} parse_aware=${PARSEAWARE_} fmt_weight=${FMTW_} (${CAMPAIGN_W6FIX_TSV}) ==="

nvidia-smi --query-gpu=name,memory.total --format=csv || true

# SIGTERM (Slurm walltime) -> python -> checkpoint -> clean exit; then resubmit.
python3 -u -m agent_bullwhip.grpo_train \
  --model "${MODEL_ID}" \
  --steps "${STEPS_}" --group "${GROUP_}" --horizon "${HORIZON_}" \
  --curricula "${CURRICULA_}" --lr "${LR_}" --beta-kl "${BETA_KL_}" \
  --rollout-batch "${ROLLOUT_BATCH}" --save-every "${SAVE_EVERY}" \
  --prompt-format "${PFMT_}" --max-new "${MAXNEW_}" \
  --parse-aware-reward "${PARSEAWARE_}" "${FMT_ARGS[@]}" \
  --dump-completions-every 10 \
  "${SEED_ARGS[@]}" \
  --out "results/grpo_${NAME}" --resume &
PY_PID=$!
trap 'echo "[$(date -Is)] SIGTERM from Slurm -> forwarding to python ${PY_PID}"; kill -TERM "${PY_PID}" 2>/dev/null || true' TERM
# F10: a trapped TERM makes a plain `wait` return at once and the script would
# exit while python is still finishing its step -> Slurm reaps the child and the
# walltime checkpoint is lost. Loop until the child is REALLY gone.
# D2: that first `wait` reports 143 (the batch shell was signaled, not the
# child) and must not stick — the LAST wait's status is the child's REAL exit
# status, so a clean checkpointed exit-0 yields RC=0 and the job completes;
# RC=143 survives only if python really died on TERM (the correct FAILED case).
RC=0
while kill -0 "${PY_PID}" 2>/dev/null; do
  if wait "${PY_PID}"; then
    RC=0
  else
    RC=$?
  fi
done
trap - TERM

echo "=== ${NAME} exit=${RC} at $(date -Is) ==="
ls -la "results/grpo_${NAME}" 2>/dev/null || echo "NO OUTPUT DIR - check the log above"
exit "${RC}"
