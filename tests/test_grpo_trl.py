"""Wave 6 tests: rollout aggregation + TRL reward glue (no torch needed)."""
from __future__ import annotations

import random

from agent_bullwhip.engine import SimConfig, make_demand
from agent_bullwhip.grpo_env import BeerGameEpisode, group_normalize
from agent_bullwhip.grpo_trl import collect_rollouts, ep_result_to_rewards, make_reward_fn


def _episode(seed: int, q: int = 8, horizon: int = 8):
    ep = BeerGameEpisode(SimConfig(horizon=horizon))
    return ep.run(lambda a, t, p: str(q), demand=make_demand(horizon, "step"))


def test_collect_rollouts_shapes_and_standardization():
    G, T = 6, 8
    eps = [_episode(s, horizon=T) for s in range(G)]
    batch = collect_rollouts(eps)
    assert len(batch.prompts) == len(batch.completions) == len(batch.advantages) == G * 4 * T
    # advantages standardized per (agent, week): overall mean ~ 0
    m = sum(batch.advantages) / len(batch.advantages)
    assert abs(m) < 1e-6
    # every decision's completion recorded as emitted
    assert all(c.isdigit() for c in batch.completions)


def test_advantage_sign_better_episode_gets_positive():
    # same prompts; one episode costs less than the group -> its decisions get
    # positive advantage, the worst episode's get negative.
    G, T = 4, 8
    eps = []
    for i, q in enumerate([2, 4, 8, 30]):   # 30-ordering blows up backlog
        ep = BeerGameEpisode(SimConfig(horizon=T))
        eps.append(ep.run(lambda a, t, p, q=q: str(q), demand=make_demand(T, "step")))
    batch = collect_rollouts(eps)
    # index the advantages by episode: first 4*T rows = episode 0 (q=2), last = q=30
    n = 4 * T
    first_ep_adv = batch.advantages[:n]
    last_ep_adv = batch.advantages[-n:]
    assert sum(first_ep_adv) > 0     # cheapest policy reinforced
    assert sum(last_ep_adv) < 0      # blowup policy discouraged


def test_reward_fn_cursor_and_desync_guard():
    eps = [_episode(s, horizon=6) for s in range(4)]
    batch = collect_rollouts(eps)
    reward_fn, set_batch = make_reward_fn()
    set_batch(batch)
    out1 = reward_fn([], batch.completions[:10])
    assert len(out1) == 10 and all(isinstance(x, float) for x in out1)
    out2 = reward_fn([], batch.completions[10:])
    assert len(out2) == len(batch.completions) - 10
    # cursor exhausted -> hard error (row-order desync must never train silently)
    try:
        reward_fn([], ["8"])
        raised = False
    except RuntimeError:
        raised = True
    assert raised


def test_group_normalize_uses_return_to_go():
    # Return-to-go geometry, pinned precisely:
    # - a cost appearing ONLY at the final week inflates the suffix from every
    #   earlier t -> every position's advantage differs across the two episodes
    # - a cost appearing ONLY at week 0 sits outside every later suffix -> only
    #   position 0 differs; later positions stay at zero advantage
    T = 6
    from agent_bullwhip.grpo_env import return_to_go_rewards
    # late blowup: suffix from t=0 includes it
    r_late1 = return_to_go_rewards([1.0] * T)
    r_late2 = return_to_go_rewards([1.0] * (T - 1) + [50.0])
    adv = group_normalize([r_late1, r_late2])
    for t in range(T):
        d = abs(adv[0][0][t] - adv[1][0][t])
        assert d > 0.5, f"late cost must propagate to week {t} (got {d})"
    # early-only cost: contained to its own position
    r_early1 = return_to_go_rewards([1.0] * T)
    r_early2 = return_to_go_rewards([50.0] + [1.0] * (T - 1))
    adv2 = group_normalize([r_early1, r_early2])
    assert abs(adv2[0][0][0] - adv2[1][0][0]) > 0.5     # position 0 sees it
    for t in range(1, T):
        assert abs(adv2[0][0][t] - adv2[1][0][t]) < 1e-9, (
            f"early cost must NOT leak into week {t}")
