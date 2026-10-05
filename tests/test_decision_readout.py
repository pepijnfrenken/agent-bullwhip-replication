"""Wave 7 P1: decision_readout — render invariants + batched letter-logit readout.

Hub-safe split (mirrors the grpo test pattern): everything that needs only the
engine (renderers, grid/label invariants) runs on the torch-less hub; the torch
tests importorskip and live in one class with a shared class-scoped model so the
memory-tight host never holds more than one fp32 0.6B at a time.
"""
from __future__ import annotations

import string

import pytest

from agent_bullwhip.engine import GameStepper, ROLES, SimConfig, make_demand
from agent_bullwhip.agents import OrderUpToAgent
from agent_bullwhip.decision_readout import (
    LABELS, ORDER_GRID, QUESTION, ReadoutModel, prompt_hash, render_body,
    render_options, render_prompt, render_state, resolve_label_ids,
)

# Canonical rendered-prompt hash fixture (AUDIT-1 3c P3: template-drift pin).
# Computed 2026-10-05 on hub transformers 4.57.6 / Qwen3-0.6B tokenizer_config;
# the cluster sbatch asserts the same bytes so 4.55.4 drift is caught before any
# eval forward. If the template/messages legitimately change, re-compute via
# `python -m agent_bullwhip.decision_readout --prompt-hash` and update here.
PROMPT_HASH_FIXTURE = "b96fb847317aed3ce670a015a0e4bc269b5573299ee19f1f577499803b99ebb7"

CANONICAL_CTX = {
    "role": "retailer", "t": 0, "on_hand": 0, "backlog": 0, "outstanding": 0,
    "incoming_last": 4, "incoming_now": 4, "last_orders": [],
}


def engine_ctxs(n_weeks: int = 4, seed: int = 1001) -> list[dict]:
    """Real engine observations under a floor policy (never hand-typed)."""
    step = GameStepper(make_demand(n_weeks, "noisy", seed=seed), SimConfig(horizon=n_weeks))
    agents = {r: OrderUpToAgent(3.0, 0.5) for r in ROLES}
    ctxs = []
    while not step.done():
        obs = step.observations()
        ctxs.extend(obs.values())
        step.advance({r: agents[r](obs[r]) for r in ROLES})
    return ctxs


# --------------------------------------------------------------------------- #
# Grid / label invariants + renderers (no torch)
# --------------------------------------------------------------------------- #
class TestGridInvariants:
    def test_grid_ascending_0_to_40_step_2(self):
        assert ORDER_GRID == list(range(0, 41, 2))
        assert len(ORDER_GRID) == 21

    def test_labels_aligned_a_through_u(self):
        assert LABELS == list(string.ascii_uppercase[:21])
        assert len(LABELS) == len(ORDER_GRID)

    def test_render_options_ascending_order(self):
        opts = render_options()
        lines = opts.splitlines()
        assert len(lines) == 21
        assert lines[0] == "A. order 0 units"
        assert lines[20] == "U. order 40 units"
        # fixed ascending: A<->0, B<->2, ... U<->40
        for i, line in enumerate(lines):
            assert line.startswith(f"{LABELS[i]}. order {ORDER_GRID[i]} units")

    def test_render_options_rejects_none_ascending(self):
        with pytest.raises(ValueError):
            render_options([0, 6, 4])

    def test_render_options_rejects_wrong_count(self):
        with pytest.raises(ValueError):
            render_options(list(range(0, 44, 2)))  # 22 options but only 21 labels


class TestRenderState:
    def test_engine_golden_all_roles_all_weeks(self):
        """Renderer must accept every ctx GameStepper.observations() produces."""
        ctxs = engine_ctxs(n_weeks=4)
        assert len(ctxs) == 4 * 4
        for ctx in ctxs:
            text = render_state(ctx)
            assert f"Current week: {ctx['t']}" in text
            assert f"- On-hand inventory: {ctx['on_hand']}" in text
            assert f"- Backlog (unfilled orders owed to your customer): {ctx['backlog']}" in text
            assert f"- Outstanding orders (ordered but not yet received): {ctx['outstanding']}" in text
            assert f"- Incoming order from your customer last week: {ctx['incoming_last']}" in text
            assert f"- Your recent orders: {list(ctx['last_orders'])}" in text

    def test_retailer_sees_incoming_now_upstream_does_not(self):
        for ctx in engine_ctxs(n_weeks=3):
            text = render_state(ctx)
            if ctx["role"] == "retailer":
                assert f"this week: {ctx['incoming_now']}" in text
                assert "not yet known" not in text
            else:
                assert "this week: not yet known (you decide on last week's information)" in text
                assert ctx["incoming_now"] is None  # the engine's leak-free contract

    def test_rejects_hand_typed_leaky_upstream_ctx(self):
        """A non-retailer ctx carrying incoming_now is an engine leak — must raise
        (AUDIT-1 3c P1: hand-typed states are how the probe went engine-unreachable)."""
        with pytest.raises(ValueError):
            render_state({"role": "factory", "t": 3, "on_hand": 0, "backlog": 0,
                          "outstanding": 0, "incoming_last": 8, "incoming_now": 8,
                          "last_orders": [8, 8, 8]})

    def test_rejects_missing_observation_field(self):
        ctx = dict(CANONICAL_CTX)
        del ctx["backlog"]
        with pytest.raises(KeyError):
            render_state(ctx)

    def test_body_contains_material_question_options(self):
        body = render_body(CANONICAL_CTX)
        assert "Material:" in body
        assert f"Question:\n{QUESTION}" in body
        assert "Options:\nA. order 0 units" in body


# --------------------------------------------------------------------------- #
# Torch-backed: label ids, batched readout, determinism, hash fixture
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def rm_and_prompts():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    rm = ReadoutModel("Qwen/Qwen3-0.6B", device="cpu")
    ctxs = engine_ctxs(n_weeks=2)
    prompts = [render_prompt(c, rm.tokenizer) for c in ctxs[:2]]
    return rm, prompts


@pytest.mark.usefixtures("rm_and_prompts")
class TestReadoutTorch:

    def test_label_ids_rendered_continuation(self, rm_and_prompts):
        """AUDIT-1 3c P2 fix: ids resolve on the rendered continuation, and each
        letter is single-token (A..U -> 32..52 in Qwen3)."""
        rm, prompts = rm_and_prompts
        ids = resolve_label_ids(rm.tokenizer, prompts[0])
        assert len(ids) == 21
        for label in LABELS:
            bare = rm.tokenizer.encode(label, add_special_tokens=False)
            assert len(bare) == 1, f"{label} is not a single token"
            assert ids[label] == bare[0], "continuation id != bare letter id"
            # the read position genuinely sees the letter: prompt+letter ends on it
            assert rm.tokenizer(prompts[0] + label, add_special_tokens=False).input_ids[-1] == ids[label]
        assert [ids[l] for l in LABELS] == list(range(32, 53))

    def test_prompt_hash_fixture(self, rm_and_prompts):
        """Template drift on 4.55.4 would change these bytes — the cluster sbatch
        asserts the same value before any eval forward (spec §5 fallback, made
        mandatory by AUDIT-1 3c P3)."""
        rm, prompts = rm_and_prompts
        assert rm.prompt_hash == PROMPT_HASH_FIXTURE
        assert prompt_hash(render_prompt(CANONICAL_CTX, rm.tokenizer)) == PROMPT_HASH_FIXTURE

    def test_readout_determinism_fixed_prompt(self, rm_and_prompts):
        rm, prompts = rm_and_prompts
        a = rm.readout_batch(prompts[:2])
        b = rm.readout_batch(prompts[:2])
        assert a == b      # exact equality on CPU (same weights/code path)

    def test_batch_mass_and_composition(self, rm_and_prompts):
        rm, prompts = rm_and_prompts
        recs = rm.readout_batch(prompts)
        for rec, prompt in zip(recs, prompts):
            assert rec["mass"] >= 0.99
            assert rec["argmax_index"] in range(21)
            assert rec["argmax_order"] == ORDER_GRID[rec["argmax_index"]]
            assert rec["argmax_label"] == LABELS[rec["argmax_index"]]
            assert rec["tokens"] == len(rm.tokenizer(prompt, add_special_tokens=False).input_ids)
            # secondary composition = expectation over the grid
            want = sum(p * o for p, o in zip(rec["probs"], ORDER_GRID))
            assert rec["expect_order"] == pytest.approx(want, abs=1e-3)

    def test_batch_equals_single_within_tolerance(self, rm_and_prompts):
        """Batching (left padding) must not change any decision: argmax is
        identical and probs/mass agree within fp32 GEMM noise (~1e-5)."""
        rm, prompts = rm_and_prompts
        single = [rm.readout(p) for p in prompts]
        batch = rm.readout_batch(prompts)
        for s, b in zip(single, batch):
            assert s["argmax_index"] == b["argmax_index"]
            assert abs(s["mass"] - b["mass"]) < 1e-3
            assert max(abs(a - c) for a, c in zip(s["probs"], b["probs"])) < 1e-3
            assert b["tokens"] == s["tokens"]

    def test_max_length_guard_raises_instead_of_truncating(self, rm_and_prompts):
        rm, prompts = rm_and_prompts
        rm.max_length = 16
        try:
            with pytest.raises(ValueError):
                rm.readout_batch(prompts)
        finally:
            rm.max_length = 1024