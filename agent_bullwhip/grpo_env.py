"""Domain-neutral delayed-inventory decision environment for GRPO training.

Wave 6 (WAVE6_SPEC.md). The paper's GRPO recipe (arXiv:2605.17036 §6–7) is defined over
a four-echelon beer-game. We implement the same mechanism over a DOMAIN-NEUTRAL interface
so the identical training loop generalizes to financial-services problems with the same
geometry (Pino's D5 constraint):

    beer game            <->  financial services analogue
    -------------------------+-------------------------------------------
    on-hand inventory        |  cash / liquidity buffer (or token float)
    backlog (unfilled)       |  unfilled client orders / shortfall
    outstanding (pipeline)   |  in-flight settlements (T+2 style lag)
    shipment delay (2 wks)   |  settlement lag
    holding cost $1          |  carry cost of buffer (funding spread)
    backlog cost $2          |  shortfall / buy-in penalty
    customer demand process  |  order-flow process (curriculum = regimes)

The beer game (engine.run_game) is instance #1. A finance env only has to yield the same
per-decision records: (prompt_completions, costs per agent-week, order vector). The GRPO
loop consumes records; it never imports the beer game.

Paper recipe implemented here (§6):
- one shared policy across tiers/agents (centralized training, decentralized execution)
- G episodes per training step under the current policy
- reward scope: system-level; attribution: return-to-go per agent-week
  r_{k,t} = -sum_{tau>=t} c_tau^{sys}; advantage = group-normalized per (agent, week)
- curricula: Poisson(lambda ~ U[5,20]) and TruncNormal(mu ~ U[8,20], sigma ~ U[2,6])
  resampled per episode; eval stays on the repo's protocol paths (never trained on).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable

from .engine import ROLES, SimConfig, make_demand, run_game
from .agents import build_prompt

# --------------------------------------------------------------------------- #
# Domain-neutral record type. `agent_id` indexes the decision-maker (tier, desk,
# account); the training loop is agnostic to what the agent *is*.
# --------------------------------------------------------------------------- #
@dataclass
class DecisionRecord:
    agent_id: int          # 0..K-1 (shared policy indexes all agents)
    t: int                 # decision time within the episode
    prompt: str            # observation serialized for the policy
    completion: str        # the action as emitted (integer string)
    logprob_ready: bool = True

@dataclass
class EpisodeResult:
    records: list[DecisionRecord]
    orders: dict[int, list[int]]           # agent_id -> per-t orders
    sys_cost_by_week: list[float]
    total_sys_cost: float

# --------------------------------------------------------------------------- #
# Reward: system-level, return-to-go per agent-week (paper Eq. under §6
# "Cost structure and reward signals"). r_{k,t} = -sum_{tau>=t} c_tau.
# --------------------------------------------------------------------------- #
def return_to_go_rewards(sys_cost_by_week: list[float]) -> dict[int, list[float]]:
    T = len(sys_cost_by_week)
    suffix = [0.0] * (T + 1)
    for t in range(T - 1, -1, -1):
        suffix[t] = suffix[t + 1] + sys_cost_by_week[t]
    # r_{k,t} = -sum_{tau>=t} c_tau  (costs are positive; rewards are negative and
    # grow MORE NEGATIVE as t decreases — the paper's return-to-go signal).
    K = 4
    return {k: [-suffix[t] for t in range(T)] for k in range(K)}

# --------------------------------------------------------------------------- #
# Demand curricula (paper §6): Poisson and truncated-normal, resampled per episode.
# --------------------------------------------------------------------------- #
def sample_curriculum_demand(rng: random.Random, curriculum: str, horizon: int = 36) -> list[int]:
    import math

    if curriculum == "poisson":
        # stdlib Random has no .poisson — Knuth's multiplicative method (fine for
        # lambda <= ~30, which the U[5,20] curriculum guarantees)
        lam = rng.uniform(5, 20)
        out = []
        for _ in range(horizon):
            L = math.exp(-lam)
            k, p = 0, 1.0
            while True:
                p *= rng.random()
                if p <= L:
                    break
                k += 1
            out.append(min(k, 50))
        return out
    if curriculum == "truncnorm":
        mu = rng.uniform(8, 20)
        sigma = rng.uniform(2, 6)
        out = []
        while len(out) < horizon:
            v = rng.gauss(mu, sigma)
            if 0 <= v <= 50:
                out.append(int(round(v)))
        return out
    raise ValueError(f"unknown curriculum: {curriculum}")

# --------------------------------------------------------------------------- #
# Beer-game adapter: builds prompts from the SAME builder as waves 1-5
# (leak-free engine; upstream incoming_now=None), collects orders + costs.
# --------------------------------------------------------------------------- #
class BeerGameEpisode:
    """Runs one episode with a `policy` callable: (agent_id, t, prompt) -> action_str.

    The policy here is intentionally dumb-from-the-loop's perspective: during
    training it is 'generate + record logprob' (wired to vLLM/TRL inside the
    trainer), during eval it is 'serve via llama.cpp/OpenAI endpoint'. The env
    never knows which.
    """

    def __init__(self, sim_cfg: SimConfig | None = None,
                 prompt_variant: str = "default"):
        self.sim_cfg = sim_cfg or SimConfig(horizon=36)
        self.prompt_variant = prompt_variant

    @staticmethod
    def _ctx_to_agent_id(role: str) -> int:
        return ROLES.index(role)

    def run(self, policy: Callable[[int, int, str], str],
            demand: list[int] | None = None,
            rng: random.Random | None = None,
            curriculum: str | None = None) -> EpisodeResult:
        if demand is None:
            demand = sample_curriculum_demand(rng or random.Random(), curriculum or "poisson",
                                              self.sim_cfg.horizon)
        prompts: dict[str, list[str]] = {r: [] for r in ROLES}

        class _Agent:
            # thin adapter so engine.run_game drives the policy; records prompts
            # (engine handles the leak-free ctx fields).
            def __init__(self, outer, role):
                self.outer, self.role = outer, role

            def __call__(self, ctx: dict) -> int:
                p = build_prompt(self.role, ctx, self.outer.prompt_variant)
                prompts[self.role].append(p)
                action = policy(self._aid(), ctx["t"], p)
                try:
                    q = int(action)
                except (TypeError, ValueError):
                    q = 0
                return max(0, q)

            def _aid(self):
                return self._aid_static(self.role)
            @staticmethod
            def _aid_static(role):
                return ROLES.index(role)

        agents: dict[str, object] = {r: _Agent(self, r) for r in ROLES}
        log = run_game(agents, demand, self.sim_cfg)
        costs = [w.cost for w in log.weeks]
        records: list[DecisionRecord] = []
        orders: dict[int, list[int]] = {}
        for aid, role in enumerate(ROLES):
            qs = log.orders(role)
            orders[aid] = qs
            for t, p in enumerate(prompts[role]):
                records.append(DecisionRecord(agent_id=aid, t=t, prompt=p,
                                              completion=str(qs[t])))
        records.sort(key=lambda d: (d.t, d.agent_id))
        return EpisodeResult(records=records, orders=orders,
                             sys_cost_by_week=costs,
                             total_sys_cost=log.total_cost())

# --------------------------------------------------------------------------- #
# GRPO advantage (paper Eq. 25): normalize per (agent, week) across the G
# episodes of one training step. Episode-level/system-level signals are special
# cases; we implement return-to-go per agent-week as specified.
# --------------------------------------------------------------------------- #
def group_normalize(rewards: list[dict[int, list[float]]],
                    eps: float = 1e-4) -> list[dict[int, list[float]]]:
    """rewards: G entries of {agent_id: [r_t]}. Returns advantages, same shape.

    Two explicit passes (mean, then std) — auditable over clever (repo rule).
    """
    G = len(rewards)
    K = len(rewards[0])
    T = len(rewards[0][0])
    out: list[dict[int, list[float]]] = [dict() for _ in range(G)]
    for k in range(K):
        for t in range(T):
            vals = [r[k][t] for r in rewards]
            mean = sum(vals) / G
            var = sum((v - mean) ** 2 for v in vals) / G
            std = (var ** 0.5) + eps
            for i in range(G):
                out[i].setdefault(k, [0.0] * T)[t] = (rewards[i][k][t] - mean) / std
    return out
