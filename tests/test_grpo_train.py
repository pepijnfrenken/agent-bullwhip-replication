"""Wave 6: grpo_train.py tests — loop mechanics with a stub policy (no torch).

The hub has no torch/tranformers/peft; `grpo_train` must import and its loop
must run against any `PolicyLike` stub. `_Policy` itself (torch-backed) is not
constructed anywhere here.
"""
from __future__ import annotations

import json
import random

import pytest

from agent_bullwhip.grpo_train import (
    _torch_available,
    collect_step,
    extract_int,
    main,
    parse_args,
    run_loop,
)


class _FakeModel:
    def __init__(self):
        self.save_calls: list = []

    def save_pretrained(self, path):
        self.save_calls.append(str(path))


class _FakeTok:
    def __init__(self):
        self.save_calls: list = []

    def save_pretrained(self, path):
        self.save_calls.append(str(path))


class _StubPolicy:
    """PolicyLike stub: deterministic completion text, recorded calls."""

    def __init__(self, answer: str = "8", loss: float = 0.123):
        self.answer = answer
        self.loss = loss
        self.n_rollouts = 0
        self.model = _FakeModel()
        self.tok = None

    def rollout(self, prompts, temperature=1.0, max_new=8):
        self.n_rollouts += 1
        return None, [self.answer] * len(prompts)

    def grpo_step(self, batch_prompts, batch_completions, advantages, **kwargs):
        return self.loss


# --------------------------------------------------------------------------- #
# collect_step: shapes, standardized advantages, seed reproducibility
# --------------------------------------------------------------------------- #
def test_collect_step_shapes_and_standardized_advantages():
    G, T = 2, 4
    prompts, completions, advs = collect_step(
        _StubPolicy(), G=G, curriculum="poisson", horizon=T, seed=7)
    n = G * 4 * T
    assert len(prompts) == len(completions) == len(advs) == n
    # completions are what the stub emitted, one per decision
    assert all(c == "8" for c in completions)
    # records are ordered episode-major, then (t, agent_id): episode g occupies
    # [g*4*T : (g+1)*4*T], and within one episode position t*4 + agent_id
    for g in range(G):
        assert prompts[g * 4 * T] != ""  # a real prompt was built for every slot
    # advantages standardized per (agent, week) across the G episodes
    for k in range(4):
        for t in range(T):
            vals = [advs[g * 4 * T + t * 4 + k] for g in range(G)]
            mean = sum(vals) / G
            std = (sum((v - mean) ** 2 for v in vals) / G) ** 0.5
            assert abs(mean) < 1e-6
            assert abs(std - 1.0) < 1e-3


def test_collect_step_seed_reproducibility():
    pol = _StubPolicy()
    p1, c1, a1 = collect_step(pol, G=2, curriculum="poisson", horizon=6, seed=99)
    n1 = pol.n_rollouts
    p2, c2, a2 = collect_step(pol, G=2, curriculum="poisson", horizon=6, seed=99)
    assert p1 == p2 and c1 == c2 and a1 == a2
    assert pol.n_rollouts == 2 * n1  # same demand -> same number of decisions


def test_curricula_produce_different_demand_driven_patterns():
    # Same stub, same seed: poisson vs truncnorm draw different demand -> costs
    # differ -> the group-normalized advantages differ (demand-driven signal).
    _, c_poisson, a_poisson = collect_step(
        _StubPolicy(), G=2, curriculum="poisson", horizon=6, seed=7)
    _, c_trunc, a_trunc = collect_step(
        _StubPolicy(), G=2, curriculum="truncnorm", horizon=6, seed=7)
    assert a_poisson != a_trunc
    assert c_poisson == c_trunc  # same policy emits the same completions either way


# --------------------------------------------------------------------------- #
# extract_int
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("text,expected", [
    ("42", 42),
    ("Order: 17 units", 17),
    ("-3", -3),          # valid signed integer (env clamps negatives to 0)
    ("-", 0),            # negative-looking junk, no digit
    ("abc", 0),          # junk, no digit
    ("", 0),             # empty output
])
def test_extract_int(text, expected):
    assert extract_int(text) == expected


# --------------------------------------------------------------------------- #
# arg parsing / CLI short-circuit
# --------------------------------------------------------------------------- #
def test_steps_zero_short_circuits(tmp_path):
    out = tmp_path / "never"
    rc = main(["--steps", "0", "--out", str(out)])
    assert rc == 0
    assert not out.exists()          # no log write, no directory


def test_parse_args_defaults_and_override():
    args = parse_args(["--steps", "5", "--group", "2", "--horizon", "6"])
    assert args.steps == 5 and args.group == 2 and args.horizon == 6
    assert args.curricula == "poisson,truncnorm" and args.beta_kl == 0.02


# --------------------------------------------------------------------------- #
# run_loop: full step loop with a stub policy (no torch at any point)
# --------------------------------------------------------------------------- #
def _loop_args(**over):
    base = dict(steps=2, group=2, horizon=6, curricula="poisson,truncnorm",
                lr=1e-5, beta_kl=0.02)
    base.update(over)
    return parse_args([
        "--steps", str(base["steps"]), "--group", str(base["group"]),
        "--horizon", str(base["horizon"]), "--curricula", base["curricula"],
        "--lr", str(base["lr"]), "--beta-kl", str(base["beta_kl"]),
        "--out", "/tmp/ignored",
    ])


def test_run_loop_logs_one_line_per_step_no_ckpt_without_torch(tmp_path):
    out = tmp_path / "out"
    log_path = out / "train_log.jsonl"
    ckpt_path = out / "checkpoint"
    pol = _StubPolicy()
    run_loop(pol, _loop_args(steps=3, group=2), log_path, ckpt_path,
             has_torch=lambda: False)
    lines = log_path.read_text().splitlines()
    assert len(lines) == 3
    for i, line in enumerate(lines):
        rec = json.loads(line)
        assert rec["step"] == i
        assert rec["curriculum"] in ("poisson", "truncnorm")  # alternates
        for key in ("loss", "mean_adv", "min_adv", "max_adv", "elapsed_s"):
            assert key in rec
        assert isinstance(rec["loss"], float)
    assert pol.model.save_calls == []   # has_torch False -> checkpoint skipped
    assert not ckpt_path.exists()


def test_run_loop_saves_checkpoint_on_last_step_when_torch(tmp_path):
    out = tmp_path / "out"
    log_path = out / "train_log.jsonl"
    ckpt_path = out / "checkpoint"
    pol = _StubPolicy()
    pol.tok = _FakeTok()
    run_loop(pol, _loop_args(steps=1, group=1), log_path, ckpt_path,
             has_torch=lambda: True, rng=random.Random(1))
    assert pol.model.save_calls == [str(ckpt_path)]     # step 0 is also step%25==0
    assert pol.tok.save_calls == [str(ckpt_path)]


def test_run_loop_curriculum_alternates_across_steps(tmp_path):
    log_path = tmp_path / "train_log.jsonl"
    run_loop(_StubPolicy(), _loop_args(steps=4, group=1), log_path,
             tmp_path / "checkpoint", has_torch=lambda: False)
    recs = [json.loads(l) for l in log_path.read_text().splitlines()]
    assert [r["curriculum"] for r in recs] == ["poisson", "truncnorm", "poisson", "truncnorm"]


def test_torch_available_reports_false_on_hub():
    assert _torch_available() is False  # this box has no torch; the loop still ran