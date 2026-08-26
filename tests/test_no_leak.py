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
