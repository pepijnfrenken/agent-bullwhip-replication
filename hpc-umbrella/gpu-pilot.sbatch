#!/bin/bash
# =============================================================================
# gpu-pilot.sbatch — the unblock test + the Snellius-decision numbers.
#
# Why an L4 partition: gpuA (tue.gpu1.q) is 8x A30 MIG 1g.6gb (~6 GB VRAM,
# ~230 GB/s) — worse than Pino's desktop 3060 Ti. gpuB/gpuC carry full L4s
# (24 GB, ~300 GB/s); 24 GB is what makes batched rollouts comfortable.
#
# What it proves, in order (each step is independently useful in the log):
#   A  a real optimizer step completes where the desktop stalled (D2 hypothesis:
#      TORCHDYNAMO_DISABLE=1 + no compile; the module torch 2.1.2 is the second
#      test of the transformers-5.x-on-torch-2.5.1 version-pairing theory)
#   B  bench_rollout.py: batched vs sequential µs per generate(), tokens/s, peak
#      VRAM — the machine-readable numbers that decide the Snellius escalation
#   C  a short GRPO pilot with the D1 batching + D3 checkpointing on, walltime-safe
#
# Submit:  cd /scratch-shared/$USER/bullwhip/kit && sbatch gpu-pilot.sbatch
# Watch:   squeue --me ; tail -f /scratch-shared/$USER/bullwhip/logs/gpu-pilot_*.log
# =============================================================================
#SBATCH --job-name=bw-pilot
#SBATCH --partition=tue.gpu3.q
#SBATCH --gres=gpu:1
#SBATCH --constraint=l4
#SBATCH --time=02:00:00
#SBATCH --signal=B:TERM@600
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/scratch-shared/%u/bullwhip/logs/gpu-pilot_%j.log

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
JOB="${SLURM_JOB_ID:-manual}"
BENCH_OUT="results/grpo_bench/bench_${JOB}.json"

echo "=== bw-pilot ${JOB} on $(hostname) | $(date -Is) ==="
echo "partition=${SLURM_JOB_PARTITION:-?} gres=${SLURM_JOB_GRES:-?} env_mode=${ENV_MODE}"
echo "TORCHDYNAMO_DISABLE=${TORCHDYNAMO_DISABLE} HF_HOME=${HF_HOME}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true
python3 -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"
python3 -c "import transformers, peft; print('transformers', transformers.__version__, '| peft', peft.__version__)" 2>&1 || echo "WARN: transformers/peft import failed (see stage.sh ENV_MODE notes)"

# Forward Slurm's pre-walltime SIGTERM to the python child so its handler can
# checkpoint and exit cleanly (grpo_train installs the handler; see --save-every).
run_py() {
  python3 -u "$@" &
  local pid=$! rc=0
  trap 'echo "[$(date -Is)] SIGTERM from Slurm -> forwarding to python ${pid}"; kill -TERM "${pid}" 2>/dev/null || true' TERM
  wait "${pid}" || rc=$?
  trap - TERM
  return "${rc}"
}

echo "=== A: minimal optimizer step (the thing the desktop never completed) ==="
python3 -u - <<'PY'
import time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", dtype=torch.bfloat16).to("cuda")
t = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
if t.pad_token is None:
    t.pad_token = t.eos_token
opt = torch.optim.AdamW(m.parameters(), lr=1e-5)
b = {k: v.to("cuda") for k, v in t(["Order 12 units this week."] * 4, return_tensors="pt", padding=True).items()}
t0 = time.time()
for i in range(2):
    out = m(**b, labels=b["input_ids"]); out.loss.backward(); opt.step(); opt.zero_grad()
    print("step", i, "loss", round(float(out.loss), 4))
print("A_seconds", round(time.time() - t0, 2))
print("A_PEAK_VRAM_GB", round(torch.cuda.max_memory_allocated() / 1e9, 2))
PY

echo "=== B: batched vs sequential rollout benchmark (D1 numbers) ==="
run_py bench_rollout.py --model "${MODEL_ID}" --group "${GROUP}" --horizon "${HORIZON}" \
  --batches "${BENCH_BATCHES}" --repeats 2 --out "${BENCH_OUT}"
echo "--- ${BENCH_OUT} ---"
cat "${BENCH_OUT}"

echo "=== C: GRPO pilot (${STEPS_PILOT} steps, batched rollouts, resumable) ==="
run_py -m agent_bullwhip.grpo_train --model "${MODEL_ID}" \
  --steps "${STEPS_PILOT}" --group "${GROUP}" --horizon "${HORIZON}" \
  --curricula "${CURRICULA}" --lr "${LR}" --beta-kl "${BETA_KL}" \
  --rollout-batch "${ROLLOUT_BATCH}" --save-every 1 \
  --out "results/grpo_a0_pilot_${JOB}" --resume

echo "=== pilot artifacts ==="
ls -la "results/grpo_a0_pilot_${JOB}" 2>/dev/null || echo "NO OUTPUT DIR - check the log above"
ls -la "results/grpo_bench" 2>/dev/null || true
echo "=== done: $(date -Is) ==="
