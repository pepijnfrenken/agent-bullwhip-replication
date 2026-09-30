#!/bin/bash
# =============================================================================
# cpu-harness.sbatch — CPU-side validation on Umbrella (no GPU, no queue wait).
#
# Purpose: prove the cluster's python/env + the repo's simulation and analysis
# code work BEFORE spending GPU time, and prove the new Wave-6 batching and
# checkpoint/resume paths run on the cluster interpreter (stub policy, no torch).
#
# Pins live in config.sh; the #SBATCH lines below repeat them literally because
# Slurm cannot interpolate shell variables. `bash config.sh --check` fails if the
# two ever disagree.
#
# Submit:  cd /scratch-shared/$USER/bullwhip/kit && sbatch cpu-harness.sbatch
# Watch:   squeue --me ; tail -f /scratch-shared/$USER/bullwhip/logs/cpu-harness_*.log
# =============================================================================
#SBATCH --job-name=bw-cpu
#SBATCH --partition=tue.cpu3.q
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --output=/scratch-shared/%u/bullwhip/logs/cpu-harness_%j.log

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

if [ -n "${SLURM_JOB_PARTITION:-}" ] && [ "${SLURM_JOB_PARTITION}" != "${CPU_PARTITION}" ]; then
  echo "WARN: running on ${SLURM_JOB_PARTITION} but CPU_PARTITION=${CPU_PARTITION} (header/pin drift)"
fi

bw_load_env
mkdir -p "${LOG_DIR}"
echo "=== bw-cpu ${SLURM_JOB_ID:-manual} on $(hostname) | $(date -Is) ==="
echo "python: $(command -v python3) $(python3 -V 2>&1)"
cd "${REPO_DIR}"

# 1. environment sanity (informative: the CPU node may not see the module torch)
python3 - <<'PY' || true
import importlib
for mod in ("numpy", "pandas", "torch", "transformers", "peft"):
    try:
        m = importlib.import_module(mod)
        print(f"{mod:12s} {getattr(m, '__version__', '?')}")
    except Exception as exc:                      # noqa: BLE001 - informational
        print(f"{mod:12s} MISSING ({type(exc).__name__})")
PY

# 2. the suite. One test asserts the *hub* has no torch; on a node that does have
#    it that assertion is simply wrong for this host, so it is deselected here.
echo "=== pytest (expect all green) ==="
python3 -m pytest tests/ -q \
  --deselect tests/test_grpo_train.py::test_torch_available_reports_false_on_hub

# 3. deterministic floors (the CPU part of the campaign; hub anchor: 3681 on step demand)
if [ -f walkforward_paths.py ]; then
  echo "=== walk-forward floors ==="
  time python3 walkforward_paths.py
else
  echo "walkforward_paths.py not present in this checkout"
fi

# 4. the Wave-6 training loop, torch-free: batched rollouts + checkpoint/resume
#    on the cluster interpreter (stub policy). Proves the D1/D3 code paths run here.
echo "=== torch-free loop dry-run (stub policy) ==="
python3 - <<'PY'
import json, threading
from pathlib import Path
from agent_bullwhip.grpo_train import load_resume_state, parse_args, run_loop


class Stub:
    """Stands in for the real policy: counts generate() calls, no torch."""

    def __init__(self, stop=None):
        self.model = None
        self.tok = None
        self.stop = stop
        self.calls = 0

    def rollout(self, prompts, temperature=1.0, max_new=8):
        self.calls += 1
        return None, ["8"] * len(prompts)

    def grpo_step(self, prompts, completions, advantages, **kw):
        if self.stop is not None:
            self.stop.set()          # pretend Slurm's SIGTERM lands mid-step
        return 0.25


def args(steps, save_every="0"):
    return parse_args(["--steps", str(steps), "--group", "2", "--horizon", "6",
                       "--save-every", save_every, "--out", "/tmp/ignored"])


root = Path("results/grpo_dryrun_cpu")
for chunk in (1, 16, 0):
    out = root / f"chunk{chunk}"
    pol = Stub()
    run_loop(pol, args(3), out / "train_log.jsonl", out / "checkpoint",
             has_torch=lambda: False)
    print(f"rollout_batch={chunk:>2}  generate_calls={pol.calls:>3} for 3 steps "
          f"(144 decisions/step at G=2,T=6)")

# SIGTERM -> checkpoint -> resume, on the cluster interpreter
stop = threading.Event()
out = root / "sigterm"
run_loop(Stub(stop), args(6, save_every="0"), out / "train_log.jsonl",
         out / "checkpoint", has_torch=lambda: False, stop=stop)
start, state = load_resume_state(out)
print("sigterm run: state.json =",
      json.dumps({k: v for k, v in state.items() if k != "rng_state"}, sort_keys=True))
print("resume would continue at step", start)
PY

echo "=== done: $(date -Is) ==="
