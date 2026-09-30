"""Wave 6: domain-neutral GRPO env tests (no model calls — deterministic)."""
from __future__ import annotations

import random
import re

import pytest

from agent_bullwhip.engine import SimConfig, make_demand, ROLES
from agent_bullwhip.grpo_env import (
    BeerGameEpisode,
    extract_int,
    group_normalize,
    return_to_go_rewards,
    run_episodes_batched,
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


# --------------------------------------------------------------------------- #
# extract_int: env-side parse (policy text -> order quantity)
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
# Batched rollout driver: it must be the engine's week loop, only dispatched
# differently. The action rule is derived FROM THE PROMPT so both drivers see
# exactly the same information.
# --------------------------------------------------------------------------- #
def _action_from_prompt(prompt: str) -> str:
    t = int(re.search(r"Current week: (\d+)", prompt).group(1))
    role = re.search(r"You are the ([A-Z]+) in a four-stage", prompt).group(1).lower()
    return str((ROLES.index(role) * 3 + t * 5) % 11)


@pytest.mark.parametrize("chunk", [1, 4, 0])           # 1 = per-decision, 0 = unbounded
@pytest.mark.parametrize("delay,init_oh,init_b,init_o", [
    (2, 0, 0, 0),      # paper defaults
    (1, 5, 2, 3),      # non-zero initial state, 1-week lag
    (3, 0, 4, 1),      # 3-week lag + initial backlog
])
def test_batched_driver_matches_engine(chunk, delay, init_oh, init_b, init_o):
    cfg = SimConfig(horizon=12, shipment_delay=delay, init_on_hand=init_oh,
                    init_backlog=init_b, init_outstanding=init_o)
    demands = [make_demand(12, "step"),
               make_demand(12, "noisy", seed=11),
               make_demand(12, "wild", seed=5)]
    ep = BeerGameEpisode(cfg)
    sequential = [ep.run(lambda a, t, p: _action_from_prompt(p), demand=d) for d in demands]
    batched = run_episodes_batched(
        lambda prompts: [_action_from_prompt(p) for p in prompts], demands, cfg, chunk=chunk)

    assert len(batched) == len(sequential) == 3
    for seq, bat in zip(sequential, batched):
        # records carry (t, agent_id, prompt, completion) -> covers prompt text,
        # prompt order, action alignment and the clamped-order bookkeeping
        assert bat.records == seq.records
        assert bat.orders == seq.orders
        assert bat.sys_cost_by_week == seq.sys_cost_by_week
        assert bat.total_sys_cost == seq.total_sys_cost
    # prompt family unchanged (frozen waves 1-5 family)
    assert all("Reply with ONLY an integer" in r.prompt
               for res in batched for r in res.records)


def test_batched_driver_rejects_wrong_completion_count():
    # A desynced batch must fail loudly rather than silently misalign decisions
    # (the AUDIT2 silent-fallback lesson).
    with pytest.raises(ValueError, match="completions for"):
        run_episodes_batched(lambda prompts: prompts[:1],
                             [make_demand(4, "step")], SimConfig(horizon=4))


def test_same_week_decisions_are_independent_of_each_other():
    """The property batching rests on: a week-t decision cannot see a same-week
    order from another tier (engine AUDIT2 fix: no one-week lookahead). Changing
    the retailer's order at week 1 must leave the other three tiers' week-1
    prompts bit-identical; if a same-week lookahead ever creeps into the engine,
    batching these four decisions together would be wrong and this fails."""
    cfg = SimConfig(horizon=6)
    demand = make_demand(6, "step")

    def run_with_retailer_order(order_at_week1: int) -> dict[str, str]:
        def policy(prompt: str) -> str:
            t = int(re.search(r"Current week: (\d+)", prompt).group(1))
            role = re.search(r"You are the ([A-Z]+) in a four-stage", prompt).group(1).lower()
            if role == "retailer" and t == 1:
                return str(order_at_week1)
            return _action_from_prompt(prompt)
        res = run_episodes_batched(lambda prompts: [policy(p) for p in prompts],
                                   [demand], cfg)[0]
        return {(r.agent_id, r.t): r.prompt for r in res.records}

    base = run_with_retailer_order(0)
    other = run_with_retailer_order(100)
    for aid in (1, 2, 3):
        assert base[(aid, 1)] == other[(aid, 1)]

