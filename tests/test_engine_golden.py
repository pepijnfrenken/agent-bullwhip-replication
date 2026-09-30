"""Waves 1-5 comparability guard (STEER-01 step 1).

`fixtures/engine_golden.json` holds per-week orders + costs from `engine.run_game`
captured on 2026-09-30, *before* the STEER-01 refactor moved the week steps into
`engine.GameStepper` (refactor taken so the synchronous and batched drivers share
one implementation instead of two copies). Every number must come back identical —
exact equality, no tolerance — because waves 1-5 results were computed on these
exact trajectories.

Do NOT regenerate the fixture to make this test pass: that re-pins behaviour and
destroys the guard. An intentional change to engine semantics needs its own
decision record plus a re-run of the affected waves.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from agent_bullwhip.engine import ROLES, SimConfig, make_demand, run_game
from agent_bullwhip.grpo_env import sample_curriculum_demand

GOLDEN = json.loads((Path(__file__).parent / "fixtures" / "engine_golden.json").read_text())


def _rule(name: str):
    if name == "A":
        return lambda aid, t: (aid * 3 + t * 5) % 11
    if name == "B":                      # emits 0s and negatives (clamp exercised)
        return lambda aid, t: ((aid * 7 - t * 4) % 13) - 2
    raise AssertionError(f"unknown action rule: {name}")


def _case_id(case: dict) -> str:
    return (f"{case['pattern']}-s{case['seed']}-h{case['horizon']}"
            f"-{case['config_name']}-r{case['rule']}")


def test_golden_fixture_covers_the_reported_matrix():
    assert GOLDEN["n_cases"] == len(GOLDEN["cases"]) == 108
    assert {c["pattern"] for c in GOLDEN["cases"]} == {"step", "poisson", "truncnorm"}
    assert {c["horizon"] for c in GOLDEN["cases"]} == {12, 36}
    assert {c["seed"] for c in GOLDEN["cases"]} == {1, 2, 3}
    assert len({c["config_name"] for c in GOLDEN["cases"]}) == 3
    assert len({c["rule"] for c in GOLDEN["cases"]}) == 2


@pytest.mark.parametrize("case", GOLDEN["cases"], ids=_case_id)
def test_engine_week_trajectories_match_waves_1_5_golden(case):
    horizon, seed = case["horizon"], case["seed"]
    if case["pattern"] == "step":
        demand = make_demand(horizon, "step")
    else:
        demand = sample_curriculum_demand(random.Random(seed), case["pattern"], horizon)
    assert demand == case["demand"]          # the demand generators are pinned too

    rule = _rule(case["rule"])
    agents = {r: (lambda idx, rd: (lambda ctx: rd(idx, ctx["t"])))(idx, rule)
              for idx, r in enumerate(ROLES)}
    log = run_game(agents, demand, SimConfig(**case["config"]))

    assert {r: log.orders(r) for r in ROLES} == case["orders"]
    assert [w.cost for w in log.weeks] == case["costs"]
    assert log.total_cost() == case["total_cost"]
