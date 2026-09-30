"""Wave 6: grpo_train.py tests — loop mechanics with a stub policy (no torch).

The hub has no torch/tranformers/peft; `grpo_train` must import and its loop
must run against any `PolicyLike` stub. `_Policy` itself (torch-backed) is not
constructed anywhere here.
"""
from __future__ import annotations

import json
import os
import random
import re
import signal
import threading
from pathlib import Path

import pytest

from agent_bullwhip import grpo_train
from agent_bullwhip.grpo_train import (
    _torch_available,
    collect_step,
    harden_compile_path,
    load_resume_state,
    main,
    parse_args,
    rng_from_state,
    run_loop,
    save_checkpoint,
    sigterm_handler,
)


class _FakeModel:
    """Writes a real file so checkpoint tests can assert on-disk state."""

    def __init__(self):
        self.save_calls: list = []

    def save_pretrained(self, path):
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        (p / "adapter_model.bin").write_text("fake-lora-weights")
        self.save_calls.append(str(path))


class _FakeTok:
    def __init__(self):
        self.save_calls: list = []

    def save_pretrained(self, path):
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        (p / "tok.bin").write_text("fake-tokenizer")
        self.save_calls.append(str(path))


class _StubPolicy:
    """PolicyLike stub: deterministic completion text, recorded calls.

    `calls` records the prompt list of every rollout() call, which is how the
    batching tests measure dispatch (one call per chunk, not per decision).
    """

    def __init__(self, answer: str = "8", loss: float = 0.123):
        self.answer = answer
        self.loss = loss
        self.n_rollouts = 0
        self.calls: list[list[str]] = []
        self.model = _FakeModel()
        self.tok = None

    def rollout(self, prompts, temperature=1.0, max_new=8):
        self.n_rollouts += 1
        self.calls.append(list(prompts))
        return None, [self.answer] * len(prompts)

    def grpo_step(self, batch_prompts, batch_completions, advantages, **kwargs):
        return self.loss


class _WeekPolicy:
    """Answers with the week number parsed from the prompt it was handed.

    The flat batch is episode-major then (t, agent_id), so a misaligned wrap of
    batched outputs would place the wrong week's answer in each slot; this stub
    makes alignment checkable from the returned completions alone.
    """

    def __init__(self):
        self.n_rollouts = 0
        self.model = _FakeModel()
        self.tok = None

    def rollout(self, prompts, temperature=1.0, max_new=8):
        self.n_rollouts += 1
        return None, [re.search(r"Current week: (\d+)", p).group(1) for p in prompts]

    def grpo_step(self, batch_prompts, batch_completions, advantages, **kwargs):
        return 0.0


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
# rollout batching (D1): same prompts, same alignment, fewer generate() calls
# --------------------------------------------------------------------------- #
def test_collect_step_batching_preserves_prompts_completions_advantages():
    # Same seed/curriculum: chunk=1 (per-decision dispatch) and chunk=0
    # (everything pending in one call) must produce identical flat batches.
    p_seq, c_seq, a_seq = collect_step(_StubPolicy(), G=2, curriculum="poisson",
                                       horizon=6, seed=7, rollout_batch=1)
    p_bat, c_bat, a_bat = collect_step(_StubPolicy(), G=2, curriculum="poisson",
                                       horizon=6, seed=7, rollout_batch=0)
    assert p_seq == p_bat
    assert c_seq == c_bat
    assert a_seq == a_bat


@pytest.mark.parametrize("chunk,expected_calls", [
    (1, 48),    # 48 decisions, one prompt per call (tiny-VRAM path)
    (4, 12),    # one same-week role group per call
    (8, 6),     # one week (2 episodes x 4 roles) per call
    (16, 6),    # default: still one week per call at G=2
    (0, 6),     # unbounded: all pending decisions of a week in one call
])
def test_collect_step_call_counts_scale_with_rollout_batch(chunk, expected_calls):
    G, T = 2, 6   # 48 decisions -> 48 * T prompts
    pol = _StubPolicy()
    collect_step(pol, G=G, curriculum="poisson", horizon=T, seed=7, rollout_batch=chunk)
    assert pol.n_rollouts == expected_calls
    assert sum(len(c) for c in pol.calls) == G * 4 * T   # every decision submitted once
    assert max(len(c) for c in pol.calls) <= (chunk if chunk > 0 else G * 4 * T)


@pytest.mark.parametrize("chunk", [1, 8, 0])
def test_collect_step_unwraps_batched_outputs_to_per_decision_alignment(chunk):
    # The stub answers with the week number parsed from its own prompt. The flat
    # batch is episode-major then (t, agent_id) -> slot (g, t, k) must hold str(t).
    G, T = 3, 5
    _, completions, advs = collect_step(_WeekPolicy(), G=G, curriculum="poisson",
                                        horizon=T, seed=3, rollout_batch=chunk)
    assert len(completions) == len(advs) == G * 4 * T
    for g in range(G):
        for t in range(T):
            for k in range(4):
                assert completions[g * 4 * T + t * 4 + k] == str(t)


def test_run_loop_dispatches_batched_rollout_calls(tmp_path):
    # Integration: the loop must hand the batch size through to collect_step.
    # 2 steps x 2 episodes x 4 roles x 6 weeks = 96 decisions; with the default
    # chunk=16 and 8 prompts/week that is 6 calls per step, not 48.
    pol = _StubPolicy()
    run_loop(pol, _loop_args(steps=2, group=2, horizon=6),
             tmp_path / "train_log.jsonl", tmp_path / "checkpoint",
             has_torch=lambda: False)
    assert pol.n_rollouts == 12
    assert all(len(c) == 8 for c in pol.calls)


# --------------------------------------------------------------------------- #
# compile-path hardening (D2)
# --------------------------------------------------------------------------- #
def test_harden_compile_path_sets_env_and_generation_knob(monkeypatch):
    monkeypatch.delenv("TORCHDYNAMO_DISABLE", raising=False)
    monkeypatch.delenv("TOKENIZERS_PARALLELISM", raising=False)

    class _GenCfg:
        compile = True

    class _Model:
        generation_config = _GenCfg()

    applied = harden_compile_path(_Model())
    assert os.environ["TORCHDYNAMO_DISABLE"] == "1"
    assert os.environ["TOKENIZERS_PARALLELISM"] == "false"
    assert _Model.generation_config.compile is False
    assert applied == {"TORCHDYNAMO_DISABLE": "1", "TOKENIZERS_PARALLELISM": "false",
                       "generation_config.compile": False}


def test_harden_compile_path_respects_explicit_operator_override(monkeypatch):
    # Documented escape hatch: re-enabling the compile path to test the parked
    # hypothesis must not be silently overridden.
    monkeypatch.setenv("TORCHDYNAMO_DISABLE", "0")
    assert harden_compile_path()["TORCHDYNAMO_DISABLE"] == "0"


def test_harden_compile_path_reports_absent_generation_knob(monkeypatch):
    # transformers 5.x may rename the decode-compile flag; the returned value is
    # what the pilot prints, so "absent" must be visible, not silent.
    monkeypatch.delenv("TORCHDYNAMO_DISABLE", raising=False)

    class _Model:
        pass

    assert harden_compile_path(_Model())["generation_config.compile"] == "absent"


# --------------------------------------------------------------------------- #
# checkpoint / resume (D3)
# --------------------------------------------------------------------------- #
def test_run_loop_checkpoint_cadence_and_metadata(tmp_path):
    out = tmp_path / "out"
    pol = _StubPolicy()
    pol.tok = _FakeTok()
    run_loop(pol, _loop_args(steps=4, group=1, save_every=2), out / "train_log.jsonl",
             out / "checkpoint", has_torch=lambda: True, rng=random.Random(5))
    state = json.loads((out / "state.json").read_text())
    assert state["step"] == 3                     # last completed step
    assert len(pol.model.save_calls) == 1         # saved once (last step is also save_every)
    assert not (out / "checkpoint.tmp").exists()  # atomic write left no debris
    for key in ("rng_state", "model", "group", "horizon", "curricula", "lr",
                "beta_kl", "rollout_batch", "saved_at"):
        assert key in state


def test_checkpoint_writes_metadata_without_torch(tmp_path):
    # The hub has no torch: step counter + RNG must still be checkpointed so the
    # dry-run/resume mechanics are testable (and a CPU harness can resume).
    out = tmp_path / "out"
    ckpt = out / "checkpoint"
    payload = save_checkpoint(_StubPolicy(), ckpt, out / "state.json", step=7,
                              rng=random.Random(3), args=_loop_args(steps=10),
                              has_torch=lambda: False)
    assert payload["adapter_saved"] is False and payload["optimizer_saved"] is False
    assert not ckpt.exists()
    assert json.loads((out / "state.json").read_text())["step"] == 7


def test_save_checkpoint_calls_optimizer_hook_when_present(tmp_path):
    out = tmp_path / "out"
    pol = _StubPolicy()
    pol.tok = _FakeTok()
    pol.save_optimizer = lambda path: (out / "opt_calls").write_text(str(path)) or True
    payload = save_checkpoint(pol, out / "checkpoint", out / "state.json", step=1,
                              rng=random.Random(3), args=_loop_args(steps=4),
                              has_torch=lambda: True)
    assert payload["optimizer_saved"] is True
    assert (out / "opt_calls").read_text() == str(out / "checkpoint")


def test_load_resume_state_absent_returns_start(tmp_path):
    assert load_resume_state(tmp_path / "nothing") == (0, None)


def test_resume_continues_step_counter_and_rng(tmp_path):
    out = tmp_path / "out"
    log_path, ckpt_path = out / "train_log.jsonl", out / "checkpoint"
    pol = _StubPolicy()
    pol.tok = _FakeTok()
    args = _loop_args(steps=2, group=1, save_every=2)
    rng = random.Random(99)
    run_loop(pol, args, log_path, ckpt_path, has_torch=lambda: True, rng=rng)

    start_step, state = load_resume_state(out)
    assert start_step == 2
    # the RNG continues exactly where the checkpoint left it
    assert rng_from_state(state).getstate() == rng.getstate()

    # a wider --steps rerun resumes and only appends the new steps
    run_loop(_StubPolicy(), _loop_args(steps=4, group=1, save_every=0), log_path,
             ckpt_path, start_step=start_step, rng=rng_from_state(state),
             has_torch=lambda: False)
    recs = [json.loads(l) for l in log_path.read_text().splitlines()]
    assert [r["step"] for r in recs] == [0, 1, 2, 3]
    assert json.loads((out / "state.json").read_text())["step"] == 3


def test_sigterm_handler_sets_stop_event():
    stop = threading.Event()
    sigterm_handler(stop)(signal.SIGTERM, None)
    assert stop.is_set()


class _SigtermPolicy(_StubPolicy):
    """Simulates SIGTERM landing during an optimizer step."""

    def __init__(self, stop: threading.Event, at_step: int = 1):
        super().__init__()
        self.stop, self.at_step, self.seen = stop, at_step, 0

    def grpo_step(self, batch_prompts, batch_completions, advantages, **kwargs):
        self.seen += 1
        if self.seen == self.at_step + 1:   # steps are 0-indexed in the log
            self.stop.set()
        return self.loss


def test_run_loop_checkpoints_and_exits_cleanly_on_sigterm(tmp_path):
    out = tmp_path / "out"
    log_path, ckpt_path = out / "train_log.jsonl", out / "checkpoint"
    stop = threading.Event()
    pol = _SigtermPolicy(stop, at_step=1)
    pol.tok = _FakeTok()
    run_loop(pol, _loop_args(steps=5, group=1, save_every=0), log_path, ckpt_path,
             has_torch=lambda: True, rng=random.Random(2), stop=stop)

    recs = [json.loads(l) for l in log_path.read_text().splitlines()]
    steps_run = [r["step"] for r in recs if r.get("event") != "sigterm"]
    assert steps_run == [0, 1]                             # step 2 never ran
    assert recs[-1] == {"step": 1, "event": "sigterm", "adapter_saved": True}
    # the job exits with a usable checkpoint even though save_every was 0
    assert (ckpt_path / "adapter_model.bin").exists()
    assert json.loads((out / "state.json").read_text())["step"] == 1
    assert pol.seen == 2                                  # no further optimizer steps


def test_main_installs_sigterm_handler_wired_to_the_loop(tmp_path, monkeypatch):
    registered: dict = {}
    monkeypatch.setattr(signal, "signal",
                        lambda sig, h: registered.__setitem__(sig, h) or None)
    monkeypatch.setattr(grpo_train, "_Policy", lambda model: _StubPolicy())
    seen: dict = {}

    def fake_run_loop(policy, args, log_path, ckpt_path, **kw):
        seen.update(kw)

    monkeypatch.setattr(grpo_train, "run_loop", fake_run_loop)
    assert main(["--steps", "3", "--out", str(tmp_path / "out")]) == 0

    handler = registered[signal.SIGTERM]
    stop = seen["stop"]
    assert isinstance(stop, threading.Event) and not stop.is_set()
    handler(signal.SIGTERM, None)                         # simulate Slurm's warning
    assert stop.is_set()


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
    assert args.rollout_batch == 16   # batched by default; 1 = per-decision


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
    # adapter + tokenizer land in the checkpoint dir, written atomically
    assert (ckpt_path / "adapter_model.bin").read_text() == "fake-lora-weights"
    assert (ckpt_path / "tok.bin").read_text() == "fake-tokenizer"
    assert not (out / "checkpoint.tmp").exists()
    state = json.loads((out / "state.json").read_text())
    assert state["step"] == 0 and state["adapter_saved"] is True


def test_run_loop_curriculum_alternates_across_steps(tmp_path):
    log_path = tmp_path / "train_log.jsonl"
    run_loop(_StubPolicy(), _loop_args(steps=4, group=1), log_path,
             tmp_path / "checkpoint", has_torch=lambda: False)
    recs = [json.loads(l) for l in log_path.read_text().splitlines()]
    assert [r["curriculum"] for r in recs] == ["poisson", "truncnorm", "poisson", "truncnorm"]


def test_torch_available_reports_false_on_hub():
    assert _torch_available() is False  # this box has no torch; the loop still ran