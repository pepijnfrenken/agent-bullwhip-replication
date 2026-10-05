"""CLI entry-point smoke tests: every `-m` module must reach argparse.

Regression guard: `decision_train.py` once lacked its `if __name__ == "__main__"`
guard, so `python3 -m agent_bullwhip.decision_train` imported the module, defined
its functions and exited 0 having done nothing. Tests that import and call
`main()` directly all passed; the failure only showed up in the sbatch that
invokes the trainer via `-m` (a 1-second "sft exit=0" with no checkpoints).

Asserting on the `--help` banner catches both a missing guard and a broken
argparse before a cluster job is ever submitted.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

MODULES = [
    "agent_bullwhip.decision_train",
    "agent_bullwhip.decision_eval",
    "agent_bullwhip.decision_labeler",
    "agent_bullwhip.decision_readout",
]


@pytest.mark.parametrize("module", MODULES)
def test_module_dash_m_reaches_argparse(module: str) -> None:
    proc = subprocess.run(
        [sys.executable, "-m", module, "--help"],
        capture_output=True, text=True, timeout=180, cwd=str(REPO_ROOT),
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 0, (
        f"{module} --help rc={proc.returncode}:\n{output[-500:]}")
    assert "usage" in output.lower(), (
        f"{module} --help printed no usage text — missing "
        f"`if __name__ == \"__main__\"` guard or broken argparse?\n{output[-500:]}")
