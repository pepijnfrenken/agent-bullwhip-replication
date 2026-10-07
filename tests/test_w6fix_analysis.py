"""scripts/w6fix_analysis.py: real historical logs + synthetic arm logs per rule."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("w6fix_analysis", ROOT / "scripts" / "w6fix_analysis.py")
wa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wa)
HIST = ROOT / "hpc-umbrella" / "umbrella-results" / "results"


@pytest.mark.skipif(not (HIST / "grpo_a0_base" / "train_log.jsonl").exists(),
                    reason="historical wave-6 log not in checkout")
def test_historical_a0_base_plateau_numbers():
    recs = wa.load_log(HIST / "grpo_a0_base" / "train_log.jsonl")
    assert len(recs) == 300
    assert round(wa.by_step(recs, 0, 50, "parse_failure_rate"), 4) == 0.8401   # W6F §1
    assert round(wa.by_step(recs, 250, 300, "parse_failure_rate"), 4) == 0.8586
    assert round(wa.by_step(recs, 45, 50, "parse_failure_rate"), 4) == 0.8491
    assert round(wa.by_step(recs, 95, 100, "parse_failure_rate"), 4) == 0.8377


def _write(dirp: Path, name: str, n: int, rate_fn, *, sha=None, pf=0.85, clip=0.003):
    d = dirp / name
    d.mkdir(parents=True)
    with (d / "train_log.jsonl").open("w") as f:
        for i in range(n):
            f.write(json.dumps({
                "step": i, "contract_parse_rate": rate_fn(i), "parse_failure_rate": pf,
                "parse_lenient_rate": 1 - pf, "clip_frac": clip, "mean_episode_cost": 1000.0 + i,
                "first_prompt_sha256": sha, "order_entropy_bits": 1.5}) + "\n")
        f.write(json.dumps({"event": "sigterm", "step": n}) + "\n")


def _mk(tmp_path, a1, a2, a1s2=None, a2s2=None):
    r = tmp_path / "results"
    _write(r, "grpo_w6fix_a0", 50, lambda i: 0.0, sha=wa.A0_FIRST_PROMPT_SHA)
    _write(r, "grpo_w6fix_a1", 100, a1, sha=wa.A1_FIRST_PROMPT_SHA)
    _write(r, "grpo_w6fix_a2", 100, a2, sha=wa.A1_FIRST_PROMPT_SHA)
    _write(r, "grpo_w6fix_a1_seed2", 100, a1s2 or a1)
    _write(r, "grpo_w6fix_a2_seed2", 100, a2s2 or a2)
    return wa.analyse(r, tmp_path / "nohist")


def test_rule_a_contract_was_the_blocker(tmp_path):
    res = _mk(tmp_path, lambda i: 0.95 if i >= 10 else 0.5, lambda i: 0.95)
    assert res["rules"]["final"] == "R-A" and res["validity_ok"] is True
    assert res["H"]["H1"] == "supports" and res["H"]["H2"] == "supports"
    assert res["P"]["a1"]["P50"] == pytest.approx(0.95)


def test_rule_b_reward_also_needed(tmp_path):
    res = _mk(tmp_path, lambda i: 0.3, lambda i: 0.3 if i < 30 else 0.95)
    assert res["rules"]["final"] == "R-B" and res["H"]["H3"] == "supports"


def test_rule_c_capacity(tmp_path):
    res = _mk(tmp_path, lambda i: 0.05, lambda i: 0.1)
    assert res["rules"]["final"] == "R-C" and res["H"]["H4"] == "supports"


def test_rule_d_middle_and_mixed_seeds(tmp_path):
    mid = _mk(tmp_path / "m", lambda i: 0.5, lambda i: 0.5)
    assert mid["rules"]["final"] == "R-D"
    mixed = _mk(tmp_path / "x", lambda i: 0.95, lambda i: 0.95, a1s2=lambda i: 0.5, a2s2=lambda i: 0.5)
    assert mixed["rules"]["per_seed"] == {1: "R-A", 2: "R-D"}
    assert mixed["rules"]["final"].startswith("R-D (mixed")


def test_v0_void_and_early_stop(tmp_path):
    r = tmp_path / "results"
    _write(r, "grpo_w6fix_a0", 50, lambda i: 0.0, sha=wa.A0_FIRST_PROMPT_SHA, pf=0.5)
    _write(r, "grpo_w6fix_a1", 60, lambda i: 0.01, sha="bad")
    res = wa.analyse(r, tmp_path / "nohist")
    assert res["validity"]["V0"]["pass"] is False and res["validity_ok"] is False
    assert res["validity"]["V1_a1"]["pass"] is False
    assert res["P"]["a1"]["P100"] == pytest.approx(0.01)
    assert res["P"]["a1"]["P100_flag"].startswith("stopped early")
    assert res["rules"]["final"].startswith("not run")


def test_missing_everything_prints_not_run(tmp_path):
    res = wa.analyse(tmp_path / "r", tmp_path / "h")
    assert res["P"]["a1"] == "not run" and "not run" in wa.render(res)
