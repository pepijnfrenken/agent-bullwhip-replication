"""Contract test for hpc-umbrella/gpu-wave7-train.sbatch (WAVE7-AUDIT-2 probe D).

Runs the REAL sbatch file under bash with a stub config.sh and a stub
`python3` on PATH. decision_train.main()/run_loop/save_checkpoint are the
real code (only _Policy is replaced by a torch-free stub); collect and
relabel are recorded fakes. Asserts the exit-status contract Slurm sees:

  (a) an RLCD failure makes the job exit non-zero (P1-2)
  (b) collect + relabel are skipped once the round-2 manifest exists (P1-5)
  (c) a SIGTERM'd SFT exits non-zero and never reaches round 2 (P1-1)
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SBATCH = ROOT / "hpc-umbrella" / "gpu-wave7-train.sbatch"

pytestmark = pytest.mark.skipif(shutil.which("bash") is None,
                                reason="bash not available")

DRIVER = r'''
"""decision_train.main() with a torch-free _Policy (probe D train_driver)."""
import os, sys, time
from pathlib import Path
sys.path.insert(0, os.environ["W7T_ROOT"])
import agent_bullwhip.decision_train as dt

class _M:
    def save_pretrained(self, p):
        Path(p, "adapter_model.safetensors").write_text("x")

class Stub:
    def __init__(self, *a, phase="sft", **k):
        if phase == "rlcd" and os.environ.get("FAIL_RLCD") == "1":
            raise RuntimeError("simulated RLCD failure (CUDA OOM / peft load)")
        self.model, self.tok = _M(), None
        with open(os.environ["W7T_CALLS"], "a") as f:
            f.write(f"train-{phase} adapter={k.get('adapter_dir')} "
                    f"base={k.get('base_adapter')}\n")
    def _step(self):
        time.sleep(float(os.environ.get("STUB_STEP_SLEEP", "0")))
    def sft_step(self, *a, **k):
        self._step(); return dt.StepResult(loss=1.0)
    def rlcd_step(self, *a, **k):
        self._step(); return dt.StepResult(loss=1.0, loss_kl=0.0)
    def compute_pi_sft(self, rows, batch):
        return {dt.pi_sft_key(r): [1 / 21] * 21 for r in rows}
    def set_pi_sft(self, table): pass
    def evaluate(self, rows): return {}
    def check_finite(self): return True
    def save_optimizer(self, p): return False
    def load_optimizer(self, p, lr): return True

dt._Policy = Stub
dt._torch_available = lambda: True
dt.run_loop.__kwdefaults__["has_torch"] = lambda: True
raise SystemExit(dt.main(sys.argv[1:]))
'''

PY_STUB = r'''#!/bin/bash
# fake python3: real decision_train with a stub policy; recorded fakes for
# collect/relabel; anything else -> the real interpreter.
case " $* " in
  *" agent_bullwhip.decision_train "*) shift 3; exec "$W7T_PY" "$W7T_DRIVER" "$@";;
  *" agent_bullwhip.decision_eval "*)
     out=""; ad=""
     while [ $# -gt 0 ]; do
       [ "$1" = --collect-states ] && out="$2"; [ "$1" = --adapter ] && ad="$2"; shift
     done
     echo "collect adapter=$ad" >> "$W7T_CALLS"; echo '{}' > "$out"; exit 0;;
  *" agent_bullwhip.decision_labeler "*)
     out=""
     while [ $# -gt 0 ]; do [ "$1" = --out ] && out="$2"; shift; done
     echo "relabel" >> "$W7T_CALLS"
     cp "$W7T_POLICY_TEMPLATE" "$out.jsonl"
     echo '{"hash": "fake"}' > "$out.manifest.json"; exit 0;;
  *"torch.cuda.is_available"*) echo "cuda-check" >> "$W7T_CALLS"; exit 0;;
  *) exec "$W7T_PY" "$@";;
esac
'''


def _rows(n: int, episode0: int = 0) -> list[dict]:
    roles = ["retailer", "wholesaler", "distributor", "factory"]
    out = []
    for i in range(n):
        j = [float(abs(k - i % 21)) for k in range(21)]
        out.append({"episode": episode0 + i // 24, "week": i % 24,
                    "role": roles[i % 4], "ctx": {}, "J": j,
                    "p": [1 / 21] * 21, "argmin_idx": i % 21})
    return out


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


@pytest.fixture
def harness(tmp_path):
    kit, repo, bin_ = tmp_path / "kit", tmp_path / "repo", tmp_path / "bin"
    for d in (kit, repo, bin_):
        d.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    shutil.copy(SBATCH, kit / SBATCH.name)
    (kit / "config.sh").write_text(
        f'REPO_DIR="{repo}"\nRESULTS_DIR="${{REPO_DIR}}/results"\n'
        f'LOG_DIR="{tmp_path / "logs"}"\nMODEL_ID=Qwen/Qwen3-0.6B\n'
        'bw_load_env() { :; }\n')
    (tmp_path / "driver.py").write_text(DRIVER)
    py = bin_ / "python3"
    py.write_text(PY_STUB)
    py.chmod(0o755)
    _write_jsonl(repo / "results" / "wave7_labels" / "labels.jsonl", _rows(96))
    _write_jsonl(tmp_path / "policy_template.jsonl", _rows(72, episode0=100))
    calls = tmp_path / "calls.log"
    calls.touch()
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("W7_", "SLURM_", "EXPECTED_COMMIT"))}
    env.update({
        "PATH": f"{bin_}{os.pathsep}{env.get('PATH', '')}",
        "SLURM_SUBMIT_DIR": str(kit), "SLURM_ARRAY_TASK_ID": "0",
        "SLURM_CPUS_PER_TASK": "2", "W7T_ROOT": str(ROOT),
        "W7T_PY": sys.executable, "W7T_DRIVER": str(tmp_path / "driver.py"),
        "W7T_CALLS": str(calls),
        "W7T_POLICY_TEMPLATE": str(tmp_path / "policy_template.jsonl"),
        "W7_SFT_STEPS": "6", "W7_RLCD_STEPS": "4", "W7_SAVE_EVERY": "2",
        "W7_BATCH": "8",
    })

    class H:
        def __init__(self):
            self.repo, self.calls_path, self.env = repo, calls, env
            self.cmd = ["bash", str(kit / SBATCH.name)]

        def calls(self) -> list[str]:
            return [x for x in calls.read_text().splitlines()
                    if x != "cuda-check"]

        def run(self, **extra) -> subprocess.CompletedProcess:
            return subprocess.run(self.cmd, env={**env, **extra}, cwd=repo,
                                  capture_output=True, text=True, timeout=300)
    return H()


def _info(p: subprocess.CompletedProcess) -> str:
    return f"rc={p.returncode}\n--stdout--\n{p.stdout}\n--stderr--\n{p.stderr}"


def test_clean_run_exits_zero_and_runs_every_stage(harness):
    p = harness.run()
    assert p.returncode == 0, _info(p)
    c = harness.calls()
    assert [x.split()[0] for x in c] == ["train-sft", "collect", "relabel",
                                         "train-rlcd"], c
    # the round-2 collect and RLCD start from the FINAL SFT checkpoint
    assert "checkpoint-5" in c[1] and c[3].endswith("wave7_sft_seed0/checkpoint-5")


def test_rlcd_failure_exits_nonzero(harness):
    """(a) AUDIT-2 P1-2 / probe D2."""
    p = harness.run(FAIL_RLCD="1")
    assert p.returncode != 0, _info(p)
    assert "rlcd seed=0 exit=1" in p.stdout


def test_round2_skipped_when_manifest_exists(harness):
    """(b) AUDIT-2 P1-5: a resubmit after a completed relabel must not
    re-collect/relabel (non-idempotent bytes -> refusal or adapter wipe)."""
    lab = harness.repo / "results" / "wave7_labels"
    shutil.copy(harness.env["W7T_POLICY_TEMPLATE"],
                lab / "policy_labels_seed0.jsonl")
    (lab / "policy_labels_seed0.manifest.json").write_text('{"hash": "x"}')
    p = harness.run()
    assert p.returncode == 0, _info(p)
    kinds = [x.split()[0] for x in harness.calls()]
    assert "collect" not in kinds and "relabel" not in kinds, kinds
    assert kinds[-1] == "train-rlcd"
    assert "skipping collect + relabel" in p.stdout


def test_sigtermed_sft_exits_nonzero_before_round2(harness):
    """(c) AUDIT-2 P1-1 / probe D1: Slurm's B:TERM reaches bash, the trap
    forwards it to python, python checkpoints — and the job must NOT go on
    to round 2 + RLCD on the partial adapter."""
    env = {**harness.env, "W7_SFT_STEPS": "200", "STUB_STEP_SLEEP": "0.05"}
    proc = subprocess.Popen(harness.cmd, env=env, cwd=harness.repo,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True)
    log = harness.repo / "results" / "wave7_sft_seed0" / "train_log.jsonl"
    deadline = time.time() + 60
    while time.time() < deadline:
        if log.exists() and '"step": 3' in log.read_text():
            break
        time.sleep(0.05)
    else:
        proc.kill()
        pytest.fail("SFT stub never reached step 3")
    proc.send_signal(signal.SIGTERM)
    out, err = proc.communicate(timeout=60)
    assert proc.returncode != 0, f"rc={proc.returncode}\n{out}\n{err}"
    assert "SIGTERM from Slurm" in out
    kinds = [x.split()[0] for x in harness.calls()]
    assert kinds == ["train-sft"], kinds
    st = json.loads((log.parent / "state.json").read_text())
    assert 3 <= st["step"] < 199
