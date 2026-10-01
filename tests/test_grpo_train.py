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
    StepResult,
    _resolve_seed,
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
    warn_resume_lr_mismatch,
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
    rollout returns the F2 triple (texts, ids=None, old_logprobs=None); ids/logps
    are the real path's sampled identities and are None for stub-driven runs.
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
        return [self.answer] * len(prompts), None, None

    def grpo_step(self, batch_prompts, batch_completions, advantages, **kwargs):
        return StepResult(loss=self.loss)


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
        return ([re.search(r"Current week: (\d+)", p).group(1) for p in prompts],
                None, None)

    def grpo_step(self, batch_prompts, batch_completions, advantages, **kwargs):
        return StepResult(loss=0.0)


# --------------------------------------------------------------------------- #
# collect_step: shapes, standardized advantages, seed reproducibility
# --------------------------------------------------------------------------- #
class _AlternatingPolicy:
    """Answers "8" in even-numbered batch positions, "9" in odd ones. With the
    shared-demand path and a whole week in one call, positions 0..3 belong to
    episode 0 and 4..7 to episode 1 -> the two episodes emit different actions,
    so the group-normalized advantage has real (policy-sample) variance. With
    chunk=1 every call is a single prompt at position 0 -> identical actions
    (episodes differ only through demand, which F17 removes)."""

    def __init__(self):
        self.i = 0

    def rollout(self, prompts, temperature=1.0, max_new=8):
        out = []
        for _ in prompts:
            out.append("8" if (self.i // 4) % 2 == 0 else "9")
            self.i += 1
        return out, None, None

    def grpo_step(self, *a, **kw):
        return StepResult(loss=0.0)


def test_collect_step_shapes_and_standardized_advantages():
    # FLAGGED (was: per-episode demands). F17 makes all G episodes of a step
    # share ONE demand path (common random numbers), so with a deterministic
    # stub there is zero group variance and the advantages are exactly 0 — that
    # IS the fix: the z-score must measure policy samples, not demand luck.
    G, T = 2, 4
    roll = collect_step(_StubPolicy(), G=G, curriculum="poisson", horizon=T, seed=7)
    n = G * 4 * T
    assert len(roll.prompts) == len(roll.completions) == len(roll.advantages) == n
    assert all(c == "8" for c in roll.completions)
    for g in range(G):
        assert roll.prompts[g * 4 * T] != ""
    assert all(abs(a) < 1e-3 for a in roll.advantages)   # zero-variance group
    # with policy-sample variance the standardization property still holds
    roll_v = collect_step(_AlternatingPolicy(), G=G, curriculum="poisson",
                          horizon=T, seed=7, rollout_batch=0)
    for k in range(4):
        for t in range(T):
            vals = [roll_v.advantages[g * 4 * T + t * 4 + k] for g in range(G)]
            mean = sum(vals) / G
            std = (sum((v - mean) ** 2 for v in vals) / G) ** 0.5
            assert abs(mean) < 1e-6
            assert abs(std - 1.0) < 1e-3


def test_collect_step_shared_demand_common_path():
    # F17: the G episodes of one step observe the SAME demand realization — the
    # group differs only in policy samples. With a deterministic stub the prompt
    # at slot (t, agent) is bit-identical across episodes.
    G, T = 2, 6
    roll = collect_step(_StubPolicy(), G=G, curriculum="poisson", horizon=T, seed=99)
    for t in range(T):
        for k in range(4):
            assert roll.prompts[t * 4 + k] == roll.prompts[4 * T + t * 4 + k]
    # legacy per-episode draws still available for comparison
    roll_legacy = collect_step(_StubPolicy(), G=G, curriculum="poisson",
                               horizon=T, seed=99, shared_demand=False)
    assert roll_legacy.prompts[:4 * T] == roll.prompts[:4 * T]       # same first draw
    assert roll_legacy.prompts[4 * T:] != roll.prompts[4 * T:]       # g=1 differs


def test_collect_step_seed_reproducibility():
    pol = _StubPolicy()
    r1 = collect_step(pol, G=2, curriculum="poisson", horizon=6, seed=99)
    n1 = pol.n_rollouts
    r2 = collect_step(pol, G=2, curriculum="poisson", horizon=6, seed=99)
    assert r1.prompts == r2.prompts and r1.completions == r2.completions
    assert r1.advantages == r2.advantages
    assert pol.n_rollouts == 2 * n1  # same demand -> same number of decisions


def test_curricula_produce_different_demand_driven_patterns():
    # FLAGGED (was: a_poisson != a_trunc with per-episode demands). F17 shares
    # one demand per step, so with a deterministic stub BOTH curricula give an
    # all-zero advantage (no signal — the honest state of a constant policy);
    # the curricula still differ in the demand paths the agents see.
    r_p = collect_step(_StubPolicy(), G=2, curriculum="poisson", horizon=6, seed=7)
    r_t = collect_step(_StubPolicy(), G=2, curriculum="truncnorm", horizon=6, seed=7)
    assert r_p.prompts != r_t.prompts          # different demand -> different prompts
    assert r_p.completions == r_t.completions  # same policy emits the same text
    assert all(abs(a) < 1e-3 for a in r_p.advantages)
    assert r_p.advantages == r_t.advantages    # constant policy: zero signal either way
    # with policy-sample variance both curricula give REAL signal — and thanks
    # to F17's common random numbers the demand effect is factored OUT: the
    # same action sequence produced the same per-(agent, week) differences even
    # under a completely different demand path (group normalization is
    # shift-invariant, and the shared demand shifts every episode alike).
    v_p = collect_step(_AlternatingPolicy(), G=2, curriculum="poisson",
                       horizon=6, seed=7, rollout_batch=0)
    v_t = collect_step(_AlternatingPolicy(), G=2, curriculum="truncnorm",
                       horizon=6, seed=7, rollout_batch=0)
    assert v_p.advantages == v_t.advantages          # CRN: demand-independent signal
    assert v_p.costs != v_t.costs                    # the demands themselves differ
    assert any(abs(a) > 0.5 for a in v_p.advantages) # and the signal is real


# --------------------------------------------------------------------------- #
# rollout batching (D1): same prompts, same alignment, fewer generate() calls
# --------------------------------------------------------------------------- #
def test_collect_step_batching_preserves_prompts_completions_advantages():
    # Same seed/curriculum: chunk=1 (per-decision dispatch) and chunk=0
    # (everything pending in one call) must produce identical flat batches.
    r_seq = collect_step(_StubPolicy(), G=2, curriculum="poisson",
                         horizon=6, seed=7, rollout_batch=1)
    r_bat = collect_step(_StubPolicy(), G=2, curriculum="poisson",
                         horizon=6, seed=7, rollout_batch=0)
    assert r_seq.prompts == r_bat.prompts
    assert r_seq.completions == r_bat.completions
    assert r_seq.advantages == r_bat.advantages


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
    roll = collect_step(_WeekPolicy(), G=G, curriculum="poisson",
                        horizon=T, seed=3, rollout_batch=chunk)
    assert len(roll.completions) == len(roll.advantages) == G * 4 * T
    for g in range(G):
        for t in range(T):
            for k in range(4):
                assert roll.completions[g * 4 * T + t * 4 + k] == str(t)


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
    # F11: the optimizer hook runs against the TMP dir BEFORE the rename, so a
    # walltime kill cannot leave optimizer.pt missing next to a finished adapter
    assert (out / "opt_calls").read_text() == str(out / "checkpoint-1.tmp")
    assert (out / "checkpoint-1" / "adapter_model.bin").exists()


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
        return StepResult(loss=self.loss)


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
    assert (out / "checkpoint-1" / "adapter_model.bin").exists()   # F11: versioned
    assert not (out / "checkpoint-1.tmp").exists()
    assert json.loads((out / "state.json").read_text())["step"] == 1
    assert pol.seen == 2                                  # no further optimizer steps


def test_main_installs_sigterm_handler_wired_to_the_loop(tmp_path, monkeypatch):
    registered: dict = {}
    monkeypatch.setattr(signal, "signal",
                        lambda sig, h: registered.__setitem__(sig, h) or None)
    monkeypatch.setattr(grpo_train, "_Policy",
                        lambda model, seed=None: _StubPolicy())
    seen: dict = {}

    def fake_run_loop(policy, args, log_path, ckpt_path, **kw):
        seen.update(kw)

    monkeypatch.setattr(grpo_train, "run_loop", fake_run_loop)
    assert main(["--steps", "3", "--seed", "7", "--out", str(tmp_path / "out")]) == 0

    handler = registered[signal.SIGTERM]
    stop = seen["stop"]
    assert isinstance(stop, threading.Event) and not stop.is_set()
    handler(signal.SIGTERM, None)                         # simulate Slurm's warning
    assert stop.is_set()
    # F12/F6: run_info.json records the seed and the git stamp
    info = json.loads((tmp_path / "out" / "run_info.json").read_text())
    assert info["seed"] == 7
    assert info["git_head"]  # repo HEAD logged


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
    assert args.seed is None and _resolve_seed(args) == 12345   # F12 default
    assert _resolve_seed(parse_args(["--seed", "2"])) == 2


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
    # adapter + tokenizer land in the checkpoint dir, written atomically (F11:
    # versioned dirs; a SIGKILL mid-write can never half-remove a shared dir)
    assert (out / "checkpoint-0" / "adapter_model.bin").read_text() == "fake-lora-weights"
    assert (out / "checkpoint-0" / "tok.bin").read_text() == "fake-tokenizer"
    assert not (out / "checkpoint-0.tmp").exists()
    state = json.loads((out / "state.json").read_text())
    assert state["step"] == 0 and state["adapter_saved"] is True
    assert state["checkpoint_dir"] == "checkpoint-0"   # F11: resume finds it


def test_run_loop_curriculum_alternates_across_steps(tmp_path):
    log_path = tmp_path / "train_log.jsonl"
    run_loop(_StubPolicy(), _loop_args(steps=4, group=1), log_path,
             tmp_path / "checkpoint", has_torch=lambda: False)
    recs = [json.loads(l) for l in log_path.read_text().splitlines()]
    assert [r["curriculum"] for r in recs] == ["poisson", "truncnorm", "poisson", "truncnorm"]


def test_torch_available_reports_false_on_hub():
    assert _torch_available() is False  # this box has no torch; the loop still ran


# --------------------------------------------------------------------------- #
# F1: non-finite tripwire — abort (exit 3), pre-NaN checkpoint, no silent skip
# --------------------------------------------------------------------------- #
class _NonfinitePolicy(_StubPolicy):
    """Reports a non-finite loss with diagnostics from the first bad micro."""

    def grpo_step(self, batch_prompts, batch_completions, advantages, **kwargs):
        return StepResult(
            loss=float("nan"), n_nonfinite=1,
            first_nonfinite={"micro_idx": 0, "loss_pg": -2.0, "loss_kl": 0.1,
                             "s_min": -30.0, "max_abs_logit": 90.0})


def test_run_loop_aborts_nonfinite_step_saves_prenan_exits_3(tmp_path):
    out = tmp_path / "out"
    log_path, ckpt_path = out / "train_log.jsonl", out / "checkpoint"
    pol = _NonfinitePolicy()
    pol.tok = _FakeTok()
    with pytest.raises(SystemExit) as ei:
        run_loop(pol, _loop_args(steps=5, group=1, save_every=25), log_path,
                 ckpt_path, has_torch=lambda: True, rng=random.Random(1))
    assert ei.value.code == 3
    # the step record carries the metric + the event line says nonfinite
    recs = [json.loads(l) for l in log_path.read_text().splitlines()]
    assert recs[0]["step"] == 0 and recs[0]["n_nonfinite"] == 1
    assert recs[1] == {"event": "nonfinite", "step": 0, "micro_idx": 0,
                       "loss_pg": -2.0, "loss_kl": 0.1, "s_min": -30.0,
                       "max_abs_logit": 90.0}
    # still-finite weights saved to checkpoint-prenan/, state marked aborted
    assert (out / "checkpoint-prenan" / "adapter_model.bin").exists()
    state = json.loads((out / "state.json").read_text())
    assert state["aborted"] == "nonfinite"
    assert state["checkpoint_dir"] == "checkpoint-prenan"
    # no further optimizer steps ran (the step was skipped, not silently kept)
    assert len(recs) == 2


# --------------------------------------------------------------------------- #
# F13: save_checkpoint refuses non-finite params (abort, no NaN checkpoint)
# --------------------------------------------------------------------------- #
def test_save_checkpoint_refuses_nonfinite_params(tmp_path):
    out = tmp_path / "out"
    pol = _StubPolicy()
    pol.tok = _FakeTok()
    pol.check_finite = lambda: False
    with pytest.raises(RuntimeError, match="non-finite"):
        save_checkpoint(pol, out / "checkpoint", out / "state.json", step=3,
                        rng=random.Random(3), args=_loop_args(steps=4),
                        has_torch=lambda: True)
    assert not list(out.rglob("adapter_model.bin"))   # nothing persisted
    assert not (out / "state.json").exists()
    # finite params save normally
    pol.check_finite = lambda: True
    save_checkpoint(pol, out / "checkpoint", out / "state.json", step=3,
                    rng=random.Random(3), args=_loop_args(steps=4),
                    has_torch=lambda: True)
    assert (out / "checkpoint-3" / "adapter_model.bin").exists()


# --------------------------------------------------------------------------- #
# F12: seed recorded in state.json + drives the loop RNG
# --------------------------------------------------------------------------- #
def test_save_checkpoint_records_effective_seed(tmp_path):
    out = tmp_path / "out"
    payload = save_checkpoint(_StubPolicy(), out / "checkpoint", out / "state.json",
                              step=1, rng=random.Random(3),
                              args=parse_args(["--seed", "42"]),
                              has_torch=lambda: False)
    assert payload["seed"] == 42
    payload2 = save_checkpoint(_StubPolicy(), out / "checkpoint",
                               out / "state2.json", step=1, rng=random.Random(3),
                               args=parse_args(["--steps", "2"]),
                               has_torch=lambda: False)
    assert payload2["seed"] == 12345       # default is explicit in the payload


def test_loop_rng_respects_seed_arg(tmp_path):
    # same --seed -> identical rollouts; different --seed -> different demand draw
    a1 = parse_args(["--steps", "2", "--group", "2", "--horizon", "6", "--seed", "5"])
    a2 = parse_args(["--steps", "2", "--group", "2", "--horizon", "6", "--seed", "5"])
    a3 = parse_args(["--steps", "2", "--group", "2", "--horizon", "6", "--seed", "6"])
    log1, log2, log3 = (tmp_path / f"t{i}.jsonl" for i in (1, 2, 3))
    run_loop(_StubPolicy(), a1, log1, tmp_path / "c1", has_torch=lambda: False)
    run_loop(_StubPolicy(), a2, log2, tmp_path / "c2", has_torch=lambda: False)
    run_loop(_StubPolicy(), a3, log3, tmp_path / "c3", has_torch=lambda: False)
    r1 = [json.loads(l) for l in log1.read_text().splitlines()]
    r2 = [json.loads(l) for l in log2.read_text().splitlines()]
    r3 = [json.loads(l) for l in log3.read_text().splitlines()]
    assert r1 == r2                       # same seed -> identical steps
    assert [x["order_max"] for x in r1] != [x["order_max"] for x in r3] or \
        r1[0]["samples"][0]["prompt_hash"] != r3[0]["samples"][0]["prompt_hash"]


# --------------------------------------------------------------------------- #
# F2: sampled ids + parsed orders + old logprobs ride the StepRollout
# --------------------------------------------------------------------------- #
class _IdsPolicy(_StubPolicy):
    """Emits one-token completions whose ids/logps differ per decision."""

    def __init__(self):
        super().__init__(answer="1")
        self._i = 0

    def rollout(self, prompts, temperature=1.0, max_new=8):
        self.n_rollouts += 1
        self.calls.append(list(prompts))
        texts, ids, logps = [], [], []
        for _ in prompts:
            texts.append(str(self._i))
            ids.append([1000 + self._i])
            logps.append([-0.5 - self._i])
            self._i += 1
        return texts, ids, logps


def test_collect_step_carries_ids_orders_old_logprobs():
    G, T = 1, 3
    roll = collect_step(_IdsPolicy(), G=G, curriculum="poisson", horizon=T,
                        seed=5, rollout_batch=0)
    n = G * 4 * T
    assert len(roll.completion_ids) == len(roll.old_logprobs) == n
    for i in range(n):
        assert roll.completion_ids[i] == [1000 + i]
        assert roll.old_logprobs[i] == [-0.5 - i]
        assert roll.orders[i] == i
    # ids must NEVER be the re-tokenized parsed order (F2: score what was sampled)
    assert all(i == [1000 + k] for k, i in enumerate(roll.completion_ids))


# --------------------------------------------------------------------------- #
# F1/F11 resume guards in main()
# --------------------------------------------------------------------------- #
def _main_with_stub_policy(monkeypatch):
    monkeypatch.setattr(grpo_train, "_Policy",
                        lambda model, seed=None: _StubPolicy())


def test_resume_refuses_missing_adapter_dir(tmp_path, monkeypatch):
    _main_with_stub_policy(monkeypatch)
    out = tmp_path / "out"
    out.mkdir()
    (out / "state.json").write_text(json.dumps(
        {"step": 4, "rng_state": list(random.Random(1).getstate()),
         "adapter_saved": True, "checkpoint_dir": "checkpoint-4"}))
    rc = main(["--resume", "--steps", "6", "--out", str(out)])
    assert rc == 1     # F11: abort rather than silently continue from a fresh LoRA


def test_resume_refuses_nonfinite_aborted_state(tmp_path, monkeypatch):
    _main_with_stub_policy(monkeypatch)
    out = tmp_path / "out"
    out.mkdir()
    (out / "state.json").write_text(json.dumps(
        {"step": 2, "aborted": "nonfinite", "adapter_saved": False}))
    rc = main(["--resume", "--steps", "6", "--out", str(out)])
    assert rc == 3


# --------------------------------------------------------------------------- #
# F7: the per-step log record carries the collapse/diversity metrics
# --------------------------------------------------------------------------- #
def test_run_loop_emits_metric_fields(tmp_path):
    log_path = tmp_path / "train_log.jsonl"
    run_loop(_StubPolicy(), _loop_args(steps=2, group=2), log_path,
             tmp_path / "checkpoint", has_torch=lambda: False)
    recs = [json.loads(l) for l in log_path.read_text().splitlines()]
    for rec in recs:
        for key in ("loss_pg", "loss_kl", "policy_push", "s_min", "entropy_first",
                    "grad_norm", "lora_norm", "max_abs_logratio", "clip_frac",
                    "n_nonfinite", "n_unique_orders",
                    "n_unique_completions", "parse_failure_rate", "order_mean",
                    "order_std", "order_max", "mean_episode_cost", "samples"):
            assert key in rec
        assert rec["n_unique_orders"] == 1          # constant stub: collapsed
        assert rec["parse_failure_rate"] == 0.0
        assert rec["order_max"] == 8
        assert len(rec["samples"]) == 3             # 3 sampled rows per step
        assert {"prompt_hash", "text", "order", "ids"} == set(rec["samples"][0])
        assert rec["mean_episode_cost"] is not None


# --------------------------------------------------------------------------- #
# B4/D6: the run_loop log record carries the detached policy-push diagnostic
# and the ratio statistics a stub policy returns (StepResult passthrough).
# --------------------------------------------------------------------------- #
class _MetricsPolicy(_StubPolicy):
    """Stub that reports non-None B4/D6 diagnostics (as _Policy does on GPU)."""

    def grpo_step(self, batch_prompts, batch_completions, advantages, **kwargs):
        return StepResult(
            loss=self.loss, loss_pg=0.01, loss_kl=0.02,
            policy_push=-0.31,                       # detached -(a * seq_logp).mean()
            max_abs_logratio=0.117,                  # D6: max |cur_logp - old_logp|
            clip_frac=0.0,                           # ratio==1 in-step -> 0 clipped
            grad_norm=0.42, lora_norm=1.33,
        )


def test_run_loop_logs_policy_push_and_ratio_stats(tmp_path):
    log_path = tmp_path / "train_log.jsonl"
    run_loop(_MetricsPolicy(), _loop_args(steps=2, group=2), log_path,
             tmp_path / "checkpoint", has_torch=lambda: False)
    recs = [json.loads(l) for l in log_path.read_text().splitlines()]
    for rec in recs:
        assert rec["policy_push"] == -0.31
        assert rec["max_abs_logratio"] == 0.117
        assert rec["clip_frac"] == 0.0
        assert rec["grad_norm"] == 0.42 and rec["lora_norm"] == 1.33


# --------------------------------------------------------------------------- #
# D8: --lr on --resume differs from the SAVED optimizer lr -> clear warning,
# saved value wins. The helper is CPU-testable; main() just prints it.
# --------------------------------------------------------------------------- #
def test_warn_resume_lr_mismatch_warns_on_differing_lr():
    w = warn_resume_lr_mismatch(cli_lr=2e-5, saved_lr=1e-5)
    assert w is not None
    assert "2e-05" in w and "1e-05" in w
    assert "SAVED value wins" in w


def test_warn_resume_lr_mismatch_silent_on_matching_lr():
    assert warn_resume_lr_mismatch(cli_lr=1e-5, saved_lr=1e-5) is None
    # tiny float noise around the same value: no warning
    assert warn_resume_lr_mismatch(cli_lr=1e-5, saved_lr=1e-5 + 1e-13) is None


def test_optimizer_lr_reads_saved_param_group_without_torch():
    """optimizer_lr() must return the SAVED param-group lr (D8's data source)
    without importing torch — build the object via __new__ and stub _opt."""
    pol = grpo_train._Policy.__new__(grpo_train._Policy)
    assert pol.optimizer_lr() is None          # no optimizer yet

    class _FakeOpt:
        param_groups = [{"lr": 2.5e-5}]

    pol._opt = _FakeOpt()
    assert pol.optimizer_lr() == 2.5e-5

def test_policy_class_exposes_grpo_step():
    """Regression for pilot 360562 step B: the D8 splice orphaned `_Policy.grpo_step`
    into a nested dead def under `warn_resume_lr_mismatch`. No hub test instantiates
    the real `_Policy` (no torch on this host), so nothing noticed — the GPU pilot did.
    This pins the class surface torch-free."""
    import inspect

    from agent_bullwhip.grpo_train import _Policy

    assert callable(getattr(_Policy, "rollout", None)), "_Policy.rollout missing"
    assert callable(getattr(_Policy, "grpo_step", None)), "_Policy.grpo_step missing"
    params = set(inspect.signature(_Policy.grpo_step).parameters)
    assert {"batch_prompts", "batch_completions", "advantages"} <= params
    assert callable(getattr(_Policy, "save_optimizer", None))
    assert callable(getattr(_Policy, "load_optimizer", None))
