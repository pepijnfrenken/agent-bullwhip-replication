"""Tests for the LLM-as-configurator (Wave 3 idea #1)."""
from __future__ import annotations

from agent_bullwhip.configurator import ConfiguratorAgent, parse_params
from agent_bullwhip.engine import ROLES, SimConfig, make_demand, run_game


def test_parse_params():
    assert parse_params('{"theta": 3.0, "lambda": 0.5, "cap": 20}') == {"theta": 3.0, "lambda": 0.5, "cap": 20.0}
    assert parse_params('theta=4 lambda=0.3 cap=15') == {"theta": 4.0, "lambda": 0.3, "cap": 15.0}
    assert parse_params("no params here") is None
    assert parse_params("") is None
    # missing lambda defaults to 0.5
    assert parse_params('{"theta": 2}') == {"theta": 2.0, "lambda": 0.5}


def test_configurator_uses_chosen_params(monkeypatch):
    import agent_bullwhip.configurator as C
    monkeypatch.setattr(C, "chat", lambda *a, **k: ['{"theta": 2.0, "lambda": 0.3, "cap": 10}'])
    agent = ConfiguratorAgent("retailer", model="test")
    agent.configure(demand_summary="step 4 then 8")
    assert agent.params == {"theta": 2.0, "lambda": 0.3, "cap": 10.0}
    assert agent.failures == 0


def test_configurator_falls_back_on_unparseable(monkeypatch):
    import agent_bullwhip.configurator as C
    monkeypatch.setattr(C, "chat", lambda *a, **k: ["I don't know"])
    agent = ConfiguratorAgent("retailer", model="test", fallback=(3.0, 0.5, None))
    agent.configure(demand_summary="")
    assert agent.params == {"theta": 3.0, "lambda": 0.5, "cap": None}
    assert agent.failures == 1


def test_configurator_rule_runs_deterministic_game():
    from agent_bullwhip.configurator import ConfiguratorAgent
    from agent_bullwhip.agents import OrderUpToAgent

    def make_agents():
        agents = {}
        for r in ROLES:
            a = ConfiguratorAgent(r, model=None, fallback=(3.0, 0.5, None))
            a.params = {"theta": 3.0, "lambda": 0.5, "cap": None}
            a.rule = OrderUpToAgent(theta=3.0, lam=0.5)  # fresh rule per run (stateful forecast)
            agents[r] = a
        return agents

    demand = make_demand(36, "step")
    l1 = run_game(make_agents(), demand, SimConfig(horizon=36))
    l2 = run_game(make_agents(), demand, SimConfig(horizon=36))
    assert l1.total_cost() == l2.total_cost()
    assert l1.orders("factory") == l2.orders("factory")
