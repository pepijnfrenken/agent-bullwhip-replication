"""verify_pull.py on a synthetic pull of a fully successful default run
(WAVE7-AUDIT-2 P2-1 / probe G: the SFT dirs end at W7_SFT_STEPS=120 because
RLCD writes its own wave7_rlcd_seed*/ — a 159 expectation called every
successful pull INCOMPLETE)."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "hpc-umbrella" / "verify_pull.py"

# what the default gpu-wave7-train.sbatch / campaign actually produce
DEFAULT_STEPS = {"wave7_sft": 120, "wave7_rlcd": 40}


def _expect() -> dict:
    spec = importlib.util.spec_from_file_location("verify_pull", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.EXPECT


def _fake_run(d: Path, steps: int, save_every: int = 10) -> None:
    d.mkdir(parents=True)
    (d / "train_log.jsonl").write_text(
        "".join(json.dumps({"step": s, "loss": 1.0}) + "\n" for s in range(steps)))
    (d / "state.json").write_text(json.dumps({"step": steps - 1}))
    for s in range(steps):
        if (s + 1) % save_every == 0 or s == steps - 1:
            ck = d / f"checkpoint-{s}"
            ck.mkdir()
            (ck / "adapter_model.safetensors").write_text("x")


def _verify(root: Path) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(SCRIPT), str(root)],
                          capture_output=True, text=True, timeout=60)


def test_default_successful_wave7_pull_is_complete(tmp_path):
    for name, target in _expect().items():
        steps = next((v for k, v in DEFAULT_STEPS.items()
                      if name.startswith(k)), target)
        _fake_run(tmp_path / "results" / name, steps)
    p = _verify(tmp_path)
    assert p.returncode == 0, p.stdout + p.stderr
    assert "VERDICT: COMPLETE" in p.stdout


def test_partial_sft_is_incomplete(tmp_path):
    for name, target in _expect().items():
        steps = next((v for k, v in DEFAULT_STEPS.items()
                      if name.startswith(k)), target)
        if name == "wave7_sft_seed1":
            steps = 60                       # SIGTERM'd half-way
        _fake_run(tmp_path / "results" / name, steps)
    p = _verify(tmp_path)
    assert p.returncode == 1
    assert "wave7_sft_seed1: train_log last step 59 < 119" in p.stdout
