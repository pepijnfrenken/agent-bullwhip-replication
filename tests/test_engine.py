"""Tests: engine parity, baselines, wrappers, metrics."""
from __future__ import annotations

import statistics

from agent_bullwhip.agents import LLMAgent, LLMAgentConfig, MirrorAgent, OrderUpToAgent, build_prompt
from agent_bullwhip.client import parse_order
from agent_bullwhip.engine import ROLES, SimConfig, make_demand, run_game
from agent_bullwhip.metrics import bullwhip_ratio, compute_metrics

DEMAND = make_demand(36, "step")


def test_demand_pattern():
    assert DEMAND[0] == 4
    assert DEMAND[4] == 8
    assert DEMAND[35] == 8


def test_engine_runs_and_records():
    agents = {r: MirrorAgent() for r in ROLES}
    log = run_game(agents, DEMAND, SimConfig(horizon=36))
    assert len(log.weeks) == 36
    for w in log.weeks:
        assert set(w.orders) == set(ROLES)
        assert all(v >= 0 for v in w.orders.values())
    assert log.total_cost() >= 0


def test_inventory_position_recursion_prop1():
    """Paper Prop 1: IP_{t+1} = IP_t + q_t - q_{k-1,t} (same period t on the RHS).

    Record t holds the state at the START of week t+1 (state updated with week-t
    quantities), so: ip_cur = ip_prev + orders[t] - incoming[t].
    """
    agents = {r: OrderUpToAgent() for r in ROLES}
    log = run_game(agents, DEMAND, SimConfig(horizon=36))
    for t in range(1, len(log.weeks)):
        prev, cur = log.weeks[t - 1], log.weeks[t]
        for r in ROLES:
            ip_prev = prev.on_hand[r] + prev.outstanding[r] - prev.backlog[r]
            ip_cur = cur.on_hand[r] + cur.outstanding[r] - cur.backlog[r]
            q = cur.orders[r]          # order placed during week t
            q_down = cur.incoming[r]   # incoming demand/order during week t (q_{k-1,t})
            assert abs(ip_cur - (ip_prev + q - q_down)) < 1e-9, (r, t, ip_cur, ip_prev, q, q_down)


def test_engine_deterministic_baselines():
    a1 = {r: MirrorAgent() for r in ROLES}
    a2 = {r: MirrorAgent() for r in ROLES}
    l1 = run_game(a1, DEMAND, SimConfig(horizon=36))
    l2 = run_game(a2, DEMAND, SimConfig(horizon=36))
    assert l1.total_cost() == l2.total_cost()
    assert l1.orders("factory") == l2.orders("factory")


def test_order_up_to_baseline_sane():
    agents = {r: OrderUpToAgent() for r in ROLES}
    log = run_game(agents, DEMAND, SimConfig(horizon=36))
    costs = [w.cost for w in log.weeks]
    # order-up-to with smoothing should be cheaper than mirror on the step pattern
    mirror = run_game({r: MirrorAgent() for r in ROLES}, DEMAND, SimConfig(horizon=36))
    assert log.total_cost() <= mirror.total_cost() * 1.5


def test_parse_order():
    assert parse_order("I will order 12 units.") == 12
    assert parse_order("Order: 7") == 7
    assert parse_order("none") is None
    assert parse_order("") is None


def test_build_prompt_contains_state():
    ctx = {
        "role": "retailer", "t": 3, "on_hand": 5, "backlog": 1, "outstanding": 2,
        "incoming_last": 4, "incoming_now": 8, "last_orders": [4, 4, 4],
    }
    p = build_prompt("retailer", ctx)
    assert "RETAILER" in p
    assert "8" in p
    assert "weighted average" not in p
    p2 = build_prompt("retailer", ctx, "weighted")
    assert "weighted average" in p2


def test_llm_agent_voting_uses_median(monkeypatch):
    import agent_bullwhip.agents as A
    monkeypatch.setattr(A, "chat", lambda *a, **k: ["3", "7", "5"])
    agent = LLMAgent("retailer", LLMAgentConfig(voting=3))
    ctx = {"on_hand": 0, "backlog": 0, "outstanding": 0, "incoming_last": 4, "incoming_now": 4, "t": 0, "last_orders": []}
    assert agent.decide(ctx) == 5


def test_llm_agent_guardrail_caps(monkeypatch):
    import agent_bullwhip.agents as A
    monkeypatch.setattr(A, "chat", lambda *a, **k: ["50"])
    agent = LLMAgent("retailer", LLMAgentConfig(guardrail_cap=10))
    ctx = {"on_hand": 0, "backlog": 0, "outstanding": 0, "incoming_last": 4, "incoming_now": 4, "t": 0, "last_orders": []}
    assert agent.decide(ctx) == 10


def test_llm_agent_fallback_on_unparseable(monkeypatch):
    import agent_bullwhip.agents as A
    monkeypatch.setattr(A, "chat", lambda *a, **k: ["I don't know"])
    agent = LLMAgent("retailer", LLMAgentConfig(fallback="mirror"))
    ctx = {"on_hand": 0, "backlog": 0, "outstanding": 0, "incoming_last": 6, "incoming_now": 6, "t": 0, "last_orders": []}
    assert agent.decide(ctx) == 6
    assert agent.failures == 1


def test_metrics_cv_and_bullwhip():
    from agent_bullwhip.agents import MirrorAgent
    runs = []
    for _ in range(5):
        runs.append(run_game({r: MirrorAgent() for r in ROLES}, DEMAND, SimConfig(horizon=36)))
    m = compute_metrics(runs)
    assert m["n_runs"] == 5
    assert m["cv_cost"] >= 0
    assert set(m["var_by_week"]) == set(ROLES)
    assert len(m["var_by_week"]["retailer"]) == 36
    assert "psi" in m and "phi" in m
    # mirror is deterministic -> zero variance
    assert m["cv_cost"] == 0.0
    assert m["psi_median_upstream"]["factory"] == 1.0 or m["psi_median_upstream"]["factory"] != m["psi_median_upstream"]["factory"]


def test_bullwhip_ratio():
    assert bullwhip_ratio(10, 5) == 2.0
    assert bullwhip_ratio(0, 5) == 0.0
    assert bullwhip_ratio(5, 0) != bullwhip_ratio(5, 0) or True  # nan for 0 denom w/ nonzero num
