"""Tests for the AUDIT2 fix: same-week incoming_now leak removed for upstream tiers."""
from agent_bullwhip.engine import ROLES, SimConfig, make_demand, run_game
from agent_bullwhip.agents import MirrorAgent, build_prompt


def test_engine_upstream_incoming_now_is_none():
    """Upstream tiers must NOT see the tier-below's same-week order (AUDIT2 §3)."""
    demand = make_demand(10, "step")
    agents = {r: MirrorAgent() for r in ROLES}
    # MirrorAgent ignores ctx, but we need to capture ctx; wrap it
    captured = {}

    def make_capture(role):
        base = MirrorAgent()
        def call(ctx):
            captured[role] = dict(ctx)
            return base(ctx)
        return call

    agents = {r: make_capture(r) for r in ROLES}
    run_game(agents, demand, SimConfig(horizon=10))

    # retailer sees current demand; upstream sees None
    assert captured["retailer"]["incoming_now"] is not None
    for r in ["wholesaler", "distributor", "factory"]:
        assert captured[r]["incoming_now"] is None, f"{r} leaked incoming_now={captured[r]['incoming_now']}"


def test_prompt_hides_incoming_now_upstream():
    """The rendered prompt for upstream tiers must not contain a same-week number."""
    ctx_upstream = {"role": "wholesaler", "t": 3, "on_hand": 0, "backlog": 5,
                    "outstanding": 10, "incoming_last": 8, "incoming_now": None,
                    "last_orders": [8, 8, 8]}
    p = build_prompt("wholesaler", ctx_upstream)
    assert "not yet known" in p, "upstream prompt should say not yet known"
    assert "Incoming order from your customer this week: 8" not in p

    ctx_retailer = dict(ctx_upstream, role="retailer", incoming_now=8)
    p2 = build_prompt("retailer", ctx_retailer)
    assert "Incoming order from your customer this week: 8" in p2


# --------------------------------------------------------------------------- #
# Wave 7: the decision-native readout renderer must hold the same no-leak
# contract (it renders from engine observations, bypassing build_prompt).
# --------------------------------------------------------------------------- #
def test_readout_renderer_upstream_hides_incoming_now():
    """render_state/render_prompt take GameStepper.observations() ctxs; an
    upstream ctx (incoming_now=None) must render 'not yet known' and never a
    same-week number, while the retailer renders its incoming_now."""
    from agent_bullwhip.decision_readout import render_state

    ctx_upstream = {"role": "wholesaler", "t": 3, "on_hand": 0, "backlog": 5,
                    "outstanding": 10, "incoming_last": 8, "incoming_now": None,
                    "last_orders": [8, 8, 8]}
    text = render_state(ctx_upstream)
    assert "not yet known (you decide on last week's information)" in text
    assert "Incoming order from your customer this week: " not in text or \
        "not yet known" in text

    ctx_retailer = dict(ctx_upstream, role="retailer", incoming_now=8)
    text_r = render_state(ctx_retailer)
    assert "this week: 8" in text_r
    assert "not yet known" not in text_r


def test_readout_renders_live_engine_observations_no_leak():
    """Full pipeline: capture GameStepper.observations() during a run and render
    every week; no upstream row may contain a same-week incoming number."""
    from agent_bullwhip.decision_readout import render_state
    from agent_bullwhip.engine import GameStepper, SimConfig, make_demand
    from agent_bullwhip.agents import MirrorAgent

    agents = {r: MirrorAgent() for r in ROLES}
    step = GameStepper(make_demand(6, "noisy", seed=1003), SimConfig(horizon=6))
    while not step.done():
        obs = step.observations()
        for r in ROLES:
            text = render_state(obs[r])
            if r == "retailer":
                assert f"this week: {obs[r]['incoming_now']}" in text
            else:
                assert "not yet known" in text
                assert f"this week: {obs[r]['incoming_now']}" not in text
        step.advance({r: agents[r](obs[r]) for r in ROLES})
