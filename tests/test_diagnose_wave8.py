"""Contract test for hpc-umbrella/diagnose-wave8.sh.

The script exists because the laptop cannot read the cluster (VPN-only), so its only job is to
turn a failed bw8-* job into a paste-back bundle. What must hold: it finds the newest log, surfaces
the error-shaped lines, and prints an inventory that lets a reader name the cause — in particular
the lr-tag mismatch that produces 'missing train output .../state.json'.

Runs entirely against a synthetic tree (BULLWHIP_ROOT override), so it needs no cluster.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

KIT = Path(__file__).resolve().parents[1] / "hpc-umbrella"
SCRIPT = KIT / "diagnose-wave8.sh"


def _mk_tree(tmp_path: Path, *, log_text: str, gates: bool = True, val_dir: str | None = "r1_lr5e-5_seed0_round0"):
    root = tmp_path / "bullwhip"
    repo = root / "agent-bullwhip-replication"
    (root / "logs").mkdir(parents=True)
    (repo / "results" / "wave8").mkdir(parents=True)
    (repo / "results" / "eval").mkdir(parents=True)
    (root / "logs" / "wave8-eval_401234.log").write_text(log_text)
    if gates:
        (repo / "results" / "wave8" / "gates.json").write_text(
            json.dumps({k: {"verdict": {"pass": True}} for k in ("F0", "F1", "F2", "F3")})
        )
    (repo / "results" / "wave8" / "S_nom.jsonl").write_text("{}\n")
    if val_dir:
        (repo / "results" / "wave8_val" / val_dir).mkdir(parents=True)
    return root


def _trained(repo_root: Path, name: str, checkpoint_dir: str | None = "checkpoint-169"):
    d = repo_root / "agent-bullwhip-replication" / "results" / name
    payload: dict[str, object] = {"step": 169}
    if checkpoint_dir:
        payload["checkpoint_dir"] = checkpoint_dir
        (d / checkpoint_dir).mkdir(parents=True)
    else:
        d.mkdir(parents=True)
    (d / "state.json").write_text(json.dumps(payload))
    return d


def _run(root: Path, selector: str = "eval"):
    env = dict(os.environ, BULLWHIP_ROOT=str(root))
    p = subprocess.run(["bash", str(SCRIPT), selector], env=env, capture_output=True, text=True, timeout=120)
    return p


def test_reports_lr_tag_mismatch_with_the_dirs_that_actually_exist(tmp_path):
    log = (
        "repo /scratch-shared/20234428/bullwhip/agent-bullwhip-replication HEAD=deadbeef rung=0.6b\n"
        "--- eval w8-untrained-argmax-step (argmax/step) at 2026-10-08T09:12:03+00:00\n"
        "--- ok w8-untrained-argmax-step\n"
        "r1 winner lr tag: lr1e-5\n"
        "FATAL: missing train output results/wave8_r1_lr1e-5_seed0_round3/state.json "
        "— run gpu-wave8-train.sbatch first\n"
    )
    root = _mk_tree(tmp_path, log_text=log)
    _trained(root, "wave8_r1_lr5e-5_seed0_round3")          # what was really trained
    p = _run(root)
    assert p.returncode == 0, p.stdout + p.stderr
    out = p.stdout
    assert "FATAL: missing train output" in out                      # error surfaced
    assert "winner lr tag: lr1e-5" in out                            # the guess that broke it
    assert "wave8_r1_lr5e-5_seed0_round3" in out and "adapter OK" in out   # what actually exists
    assert "W8_LR=<the lr you actually trained>" in out              # the fix is spelled out


def test_flags_state_json_without_checkpoint_dir(tmp_path):
    log = "repo /x HEAD=abc rung=0.6b\n--- ok w8-untrained-argmax-step\n"
    root = _mk_tree(tmp_path, log_text=log)
    _trained(root, "wave8_r1_lr5e-5_seed1_round3", checkpoint_dir=None)
    out = _run(root).stdout
    assert "no checkpoint_dir key" in out
    assert "ADAPTER DIR MISSING" in out


def test_missing_gates_is_called_out_as_the_f3_refusal(tmp_path):
    log = "repo /x HEAD=abc rung=0.6b\nFATAL: results/wave8/gates.json missing (F3 not frozen for this rung)\n"
    root = _mk_tree(tmp_path, log_text=log, gates=False)
    out = _run(root).stdout
    assert "MISSING/empty" in out and "F3 not frozen" in out


def test_missing_val_dir_is_called_out_before_select_lr(tmp_path):
    log = "repo /x HEAD=abc rung=0.6b\n--- ok w8-untrained-argmax-step\n"
    root = _mk_tree(tmp_path, log_text=log, val_dir=None)
    out = _run(root).stdout
    assert "results/wave8_val/: MISSING" in out
    assert "select-lr will fail" in out


def test_runs_without_a_kit_config_and_without_sacct(tmp_path):
    """Piped over ssh as `bash -s` there is no config.sh next to the script, and no sacct."""
    log = "repo /x HEAD=abc rung=0.6b\n--- ok w8-untrained-argmax-step\n"
    root = _mk_tree(tmp_path, log_text=log)
    _trained(root, "wave8_r1_lr5e-5_seed0_round3")
    bare = tmp_path / "nokit"
    bare.mkdir()
    copy = bare / "diagnose-wave8.sh"
    copy.write_text(SCRIPT.read_text())
    env = dict(os.environ, BULLWHIP_ROOT=str(root))
    p = subprocess.run(["bash", str(copy), "eval"], env=env, capture_output=True, text=True, timeout=120)
    assert p.returncode == 0, p.stdout + p.stderr
    assert "wave8-eval_401234.log" in p.stdout          # found the log without config.sh
    assert "rehearsal" not in p.stdout
    assert "wave8_r1_lr5e-5_seed0_round3" in p.stdout   # inventory still resolved the repo
