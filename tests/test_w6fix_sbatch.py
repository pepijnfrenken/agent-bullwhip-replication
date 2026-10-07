"""hpc-umbrella/gpu-w6fix.sbatch + campaign-w6fix.tsv: the REAL sbatch under bash with a
stub config.sh and a recording `python3` shim (same harness idea as test_sbatch_contract)."""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
KITSRC = ROOT / "hpc-umbrella"
pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")

PY_STUB = '''#!/bin/bash
case " $* " in
  *" agent_bullwhip.grpo_train "*) echo "$*" >> "$W6_CALLS"; exit 0;;
  *) exec "$W6_PY" "$@";;
esac
'''


def _tsv_rows():
    lines = (KITSRC / "campaign-w6fix.tsv").read_text().splitlines()
    hdr = lines[0].split("\t")
    return hdr, [dict(zip(hdr, ln.split("\t"))) for ln in lines[1:]]


def _run(tmp_path, task, extra_env=None):
    kit, repo, bin_ = tmp_path / "kit", tmp_path / "repo", tmp_path / "bin"
    for d in (kit, repo, bin_):
        d.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    shutil.copy(KITSRC / "gpu-w6fix.sbatch", kit)
    shutil.copy(KITSRC / "campaign-w6fix.tsv", kit)
    (kit / "config.sh").write_text(
        f'REPO_DIR="{repo}"\nLOG_DIR="{tmp_path / "logs"}"\nMODEL_ID=Qwen/Qwen3-0.6B\n'
        f'CAMPAIGN_W6FIX_TSV="{kit / "campaign-w6fix.tsv"}"\nROLLOUT_BATCH=16\nSAVE_EVERY=10\n'
        'bw_load_env() { :; }\n')
    py = bin_ / "python3"
    py.write_text(PY_STUB)
    py.chmod(0o755)
    calls = tmp_path / f"calls{task}.log"
    calls.touch()
    env = {k: v for k, v in os.environ.items() if not k.startswith(("SLURM_", "EXPECTED_COMMIT", "W6FIX"))}
    import sys
    env.update({"PATH": f"{bin_}{os.pathsep}{env['PATH']}", "SLURM_SUBMIT_DIR": str(kit),
                "SLURM_ARRAY_TASK_ID": str(task), "W6_CALLS": str(calls), "W6_PY": sys.executable})
    env.update(extra_env or {})
    p = subprocess.run(["bash", str(kit / "gpu-w6fix.sbatch")], env=env, capture_output=True,
                       text=True, timeout=60)
    return p, calls.read_text().splitlines()


def test_campaign_tsv_shape():
    hdr, rows = _tsv_rows()
    assert hdr == ["name", "group", "horizon", "curricula", "lr", "beta_kl", "steps",
                   "prompt_format", "max_new", "parse_aware", "fmt_weight"]
    assert [r["name"] for r in rows] == ["w6fix_a0", "w6fix_a0b", "w6fix_a1", "w6fix_a1_seed2",
                                         "w6fix_a2", "w6fix_a2_seed2"]
    assert all(len(r) == 11 for r in rows)


@pytest.mark.parametrize("task", range(6))
def test_each_array_task_passes_its_rows_flags(tmp_path, task):
    _, rows = _tsv_rows()
    row = rows[task]
    p, calls = _run(tmp_path, task)
    assert p.returncode == 0, p.stdout + p.stderr
    assert len(calls) == 1
    argv = calls[0]
    assert f"--steps {row['steps']} " in argv
    assert f"--prompt-format {row['prompt_format']} " in argv
    assert f"--max-new {row['max_new']} " in argv
    assert f"--parse-aware-reward {row['parse_aware']} " in argv
    assert "--dump-completions-every 10" in argv and argv.rstrip().endswith("--resume")
    assert f"--out results/grpo_{row['name']}" in argv
    assert ("--fmt-weight 1.0" in argv) == (row["parse_aware"] == "1")
    assert ("--seed 2" in argv) == row["name"].endswith("_seed2")
    assert ("--seed" in argv) == row["name"].endswith("_seed2")


def test_pilot_steps_override_and_commit_gate(tmp_path):
    p, calls = _run(tmp_path / "a", 2, {"W6FIX_STEPS": "4"})
    assert p.returncode == 0 and "--steps 4 " in calls[0] and "--prompt-format chat_contract" in calls[0]
    p, calls = _run(tmp_path / "b", 2, {"EXPECTED_COMMIT": "deadbeef"})
    assert p.returncode == 1 and calls == [] and "EXPECTED_COMMIT" in p.stderr


def test_headers_match_kit_pins():
    txt = (KITSRC / "gpu-w6fix.sbatch").read_text()
    for needle in ("--partition=tue.gpu.q", "--constraint=l4", "--gres=gpu:1", "--time=14:00:00",
                   "--signal=B:TERM@600", "--array=0-5", "--cpus-per-task=8", "--mem=16G"):
        assert f"#SBATCH {needle}" in txt
