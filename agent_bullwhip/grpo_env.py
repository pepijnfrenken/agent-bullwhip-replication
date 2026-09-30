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
import re
from dataclasses import dataclass, field
from typing import Callable

from .engine import ROLES, GameStepper, RunLog, SimConfig, make_demand, run_game
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
        return episode_result_from_log(log, prompts)


def extract_int(text: str) -> int:
    """First signed integer in `text`; 0 if none (env clamps negatives to 0).

    Lives here rather than in grpo_train because it is the env-side parse:
    policy text -> order quantity. Same rule as the waves 1-5 `parse_order`
    (first integer anywhere), so the trained policy is served by the same
    contract the earlier waves used.
    """
    m = re.search(r"-?\d+", text or "")
    return int(m.group()) if m else 0


def episode_result_from_log(log: RunLog, prompts: dict[str, list[str]]) -> EpisodeResult:
    """Assemble an EpisodeResult from a RunLog + the prompts observed per role.

    Shared by the synchronous (BeerGameEpisode.run) and batched drivers, so both
    produce byte-identical record layout/order: episode-major, then (t, agent_id),
    with `completion` = the order the engine actually placed (clamped to >= 0).
    """
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
                         sys_cost_by_week=[w.cost for w in log.weeks],
                         total_sys_cost=log.total_cost())


def run_episodes_batched(batch_policy: Callable[[list[str]], list[str]],
                         demands: list[list[int]],
                         sim_cfg: SimConfig | None = None, *,
                         prompt_variant: str = "default", chunk: int = 16,
                         parse: Callable[[str], int] = extract_int) -> list[EpisodeResult]:
    """Run `len(demands)` episodes in lockstep, batching generate() across
    decisions that are causally independent.

    Why batching is semantically safe here:
      * across TIME: a week-t decision reads state through week t-1 only, so the
        lockstep week loop visits decisions in exactly `engine.run_game`'s order.
      * across ROLES within one week: `engine.GameStepper.observations` builds every
        role's ctx from the state at the START of week t and exposes no same-week
        order to anybody (`incoming_now` is the retailer's exogenous demand, not a
        downstream order — the AUDIT2 leak fix). The 4 role decisions of a week are
        therefore mutually independent and may share one generate() call.
      * across EPISODES: separate demand realizations, separate state, nothing flows
        between them, so all G episodes' week-t decisions may share a call.

    Only the *dispatch* is batched: every decision keeps its own prompt and its own
    sampled completion. `chunk=1` disables the batching (one prompt per call).

    Week semantics come from `engine.GameStepper` — the SAME stepper `run_game`
    drives — so there is exactly one implementation (STEER-01) and no mirror to
    drift. `tests/test_grpo_env.py::test_batched_driver_matches_engine` guards the
    driver logic (prompt order, record assembly, chunking) against the synchronous
    driver, and `test_same_week_decisions_are_independent_of_each_other` guards the
    independence this batching depends on.

    `chunk` = max prompts per batch_policy call (<= 0 = every pending decision of
    the week in one call). Returns one EpisodeResult per demand, in order.
    """
    cfg = sim_cfg or SimConfig(horizon=36)
    steppers = [GameStepper(demand, cfg) for demand in demands]
    if not steppers:
        return []
    prompts: list[dict[str, list[str]]] = [{r: [] for r in ROLES} for _ in steppers]
    max_chunk = chunk if chunk > 0 else 1 << 30

    while True:
        pending: list[tuple[int, str, str]] = []
        for g, stepper in enumerate(steppers):
            if stepper.done():
                continue
            for role, ctx in stepper.observations().items():
                p = build_prompt(role, ctx, prompt_variant)
                prompts[g][role].append(p)
                pending.append((g, role, p))
        if not pending:
            break
        texts: list[str] = []
        for i in range(0, len(pending), max_chunk):
            texts.extend(batch_policy([p for _, _, p in pending[i:i + max_chunk]]))
        if len(texts) != len(pending):
            raise ValueError(
                f"batch_policy returned {len(texts)} completions for {len(pending)} prompts")
        orders: list[dict[str, int]] = [{} for _ in steppers]
        for (g, role, _), text in zip(pending, texts):
            orders[g][role] = int(parse(text))
        for g, stepper in enumerate(steppers):
            if not stepper.done():
                stepper.advance(orders[g])

    return [episode_result_from_log(stepper.log(), prompts[g])
            for g, stepper in enumerate(steppers)]


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
