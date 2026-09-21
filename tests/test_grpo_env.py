"""Wave 6: domain-neutral GRPO env tests (no model calls — deterministic)."""
from __future__ import annotations

import random

import pytest

from agent_bullwhip.engine import SimConfig, make_demand, ROLES
from agent_bullwhip.grpo_env import (
    BeerGameEpisode,
    group_normalize,
    return_to_go_rewards,
    sample_curriculum_demand,
)


class _ConstantPolicy:
    """Policy stub: always orders the same integer."""

    def __init__(self, q: int = 8):
        self.q = q
        self.n_calls = 0

    def __call__(self, agent_id: int, t: int, prompt: str) -> str:
        self.n_calls += 1
        assert "Reply with ONLY an integer" in prompt  # same family as waves 1-5
        assert "not yet known" in prompt if agent_id > 0 else "Incoming order from your customer this week: 8" in prompt or "Incoming order from your customer this week: 4" in prompt
        return str(self.q)


def test_episode_shape_and_leak_guard():
    pol = _ConstantPolicy(8)
    ep = BeerGameEpisode(SimConfig(horizon=12))
    res = ep.run(lambda a, t, p: pol(a, t, p), demand=make_demand(12, "step"))
    # 4 agents x 12 weeks = 48 decisions, one prompt per decision
    assert len(res.records) == 4 * 12
    assert pol.n_calls == 48
    assert set(res.orders) == {0, 1, 2, 3}
    assert all(len(v) == 12 for v in res.orders.values())
    # every agent gets every (agent, week) reward position
    rew = return_to_go_rewards(res.sys_cost_by_week)
    assert set(rew) == {0, 1, 2, 3}
    assert all(len(v) == 12 for v in rew.values())
    # return-to-go: rewards are negative, magnitude grows toward t=0 (suffix of
    # positive costs) -> series is monotonically DECREASING in t
    for k, series in rew.items():
        for t in range(len(series) - 1):
            assert series[t] <= series[t + 1] + 1e-9
    # total cost equals the engine's own accounting
    assert abs(sum(res.sys_cost_by_week) - res.total_sys_cost) < 1e-6


def test_costs_differ_by_policy_constant_zero_blowup():
    # At the real 36-week horizon: naive constant policies (order 0 or order 8
    # forever) cost ~2.6x the untuned order-up-to floor on the step demand. That
    # gap is the entire optimization target GRPO must close — assert it here so a
    # broken env (wrong cost wiring) can't train silently.
    ep = BeerGameEpisode(SimConfig(horizon=36))
    demand = make_demand(36, "step")
    res0 = ep.run(lambda a, t, p: "0", demand=demand)
    res8 = ep.run(lambda a, t, p: "8", demand=demand)
    assert res0.total_sys_cost > 0 and res8.total_sys_cost > 0
    from agent_bullwhip.agents import OrderUpToAgent
    from agent_bullwhip.engine import run_game
    agents: dict[str, object] = {r: OrderUpToAgent() for r in ROLES}
    logf = run_game(agents, demand, SimConfig(horizon=36))
    assert logf.total_cost() == 3681.0  # the README's known anchor_only score
    assert logf.total_cost() < 0.5 * res8.total_sys_cost
    assert logf.total_cost() < 0.5 * res0.total_sys_cost


def test_group_normalize_matches_paper_properties():
    rng = random.Random(0)
    G, K, T = 8, 4, 12
    rewards = [
        {k: [-abs(rng.gauss(10, 5)) for _ in range(T)] for k in range(K)}
        for _ in range(G)
    ]
    adv = group_normalize(rewards)
    assert len(adv) == G
    for k in range(K):
        for t in range(T):
            vals = [adv[i][k][t] for i in range(G)]
            mean = sum(vals) / G
            std = (sum((v - mean) ** 2 for v in vals) / G) ** 0.5
            assert abs(mean) < 1e-6          # zero mean across the group
            assert abs(std - 1.0) < 1e-3     # unit variance (eps=1e-4)
    # identical rewards across the group -> advantages ~ 0 (no signal, no update)
    same = [{k: [-100.0] * T for k in range(K)} for _ in range(G)]
    adv_same = group_normalize(same)
    for i in range(G):
        for k in range(K):
            for v in adv_same[i][k]:
                assert abs(v) < 1e-6


@pytest.mark.parametrize("curriculum", ["poisson", "truncnorm"])
def test_curricula_resample_and_bounds(curriculum):
    rng = random.Random(42)
    d1 = sample_curriculum_demand(rng, curriculum, horizon=36)
    d2 = sample_curriculum_demand(rng, curriculum, horizon=36)
    assert len(d1) == len(d2) == 36
    assert d1 != d2                       # resampled per episode
    assert all(0 <= x <= 50 for x in d1)  # paper bounds (Poisson tail < 50 in practice)


def test_policy_failure_yields_zero_not_crash():
    ep = BeerGameEpisode(SimConfig(horizon=6))
    res = ep.run(lambda a, t, p: "not-a-number", demand=make_demand(6, "step"))
    assert all(q == 0 for qs in res.orders.values() for q in qs)


def test_shared_policy_sees_all_four_roles():
    seen = set()
    def pol(agent_id, t, prompt):
        seen.add(agent_id)
        return "8"
    ep = BeerGameEpisode(SimConfig(horizon=4))
    ep.run(pol, demand=make_demand(4, "step"))
    assert seen == {0, 1, 2, 3}
