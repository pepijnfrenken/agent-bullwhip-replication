"""Wave 6: TRL GRPOTrainer wiring for the delayed-inventory env.

D1 (decided): TRL first, hand-rolled fallback only if this stalls. This module is
the TRL integration: it feeds `BeerGameEpisode` rollouts to `trl.GRPOTrainer`
through a custom reward function, using the paper's reward design:

- system-level cost scope, return-to-go attribution per agent-week
- group normalization per (agent, week) across the G rollouts (Eq. 25)

TRL computes its own group-relative advantages over per-completion rewards. To
preserve the paper's per-(agent, week) return-to-go credit assignment we do the
aggregation here: each generated completion (one weekly decision) gets its own
reward = the group-normalized return-to-go advantage of that (agent, week)
position, computed across this training step's G episodes. TRL then normalizes
again per prompt group; because ours are already standardized, that second
normalization is a no-op rescale (documented in WAVE6_SPEC.md §3.2).

Training runs on the desktop (D3: Modal OUT). Everything here is CPU/GPU-agnostic;
the pilot config lives in grpo_train.py.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from .grpo_env import BeerGameEpisode, EpisodeResult, group_normalize


@dataclass
class RolloutBatch:
    """One training step's worth of rollouts under the current policy."""

    prompts: list[str]          # flat list, len = G * 4 * T (one per decision)
    completions: list[str]      # the emitted integer strings (recorded, not re-generated)
    advantages: list[float]     # per-decision advantage (paper Eq. 25, return-to-go)


def collect_rollouts(episodes: list[EpisodeResult]) -> RolloutBatch:
    """Aggregate G episode results into a flat per-decision batch with paper
    advantages. Episodes must come from the SAME policy snapshot (one step)."""
    rewards = [ep_result_to_rewards(ep) for ep in episodes]
    adv = group_normalize(rewards)  # list of {agent_id: [per-week advantages]}
    prompts: list[str] = []
    completions: list[str] = []
    advantages: list[float] = []
    for ep, adv_ep in zip(episodes, adv):
        for rec in ep.records:
            prompts.append(rec.prompt)
            completions.append(rec.completion)
            advantages.append(adv_ep[rec.agent_id][rec.t])
    return RolloutBatch(prompts=prompts, completions=completions, advantages=advantages)


def ep_result_to_rewards(ep: EpisodeResult) -> dict[int, list[float]]:
    from .grpo_env import return_to_go_rewards
    return return_to_go_rewards(ep.sys_cost_by_week)


# --------------------------------------------------------------------------- #
# vLLM/TRL glue — imported lazily so the hub (no torch) can still run tests.
# --------------------------------------------------------------------------- #
def make_reward_fn():
    """TRL reward_fn(prompts, completions, **kwargs) -> list[float].

    The trainer passes per-row extras via `trainer.state`-style plumbing; we use
    the closure over the current step's advantage map instead: TRL calls the
    reward function with the SAME row order as the generated batch, so we map
    (prompt, completion) -> advantage through the batch collected this step.
    """

    state: dict = {"batch": None, "cursor": 0}

    def set_batch(batch: RolloutBatch):
        state["batch"] = batch
        state["cursor"] = 0

    def reward_fn(prompts, completions, **kwargs):
        batch = state["batch"]
        if batch is None:
            raise RuntimeError("reward_fn called before set_batch()")
        n = len(completions)
        out = batch.advantages[state["cursor"]: state["cursor"] + n]
        state["cursor"] += n
        if len(out) != n:
            raise RuntimeError(
                f"advantage cursor exhausted: wanted {n}, had {len(out)}. "
                "Row order desync between generation and reward call.")
        return list(out)

    return reward_fn, set_batch
