#!/usr/bin/env python3
"""wave8_teacher_check.py — WAVE8 pre-flight F1 (docs/waves/WAVE8_SPEC.md §6.1).

Closed-loop PI1 (every role, every week, plays argmin_a E[J(s,a)] over the w8_g26b candidates,
labeler M=16 CRN futures) vs the continuation floor (theta 3.5, lam 0.2) on the SAME 30
curriculum paths, seeds 9000..9029 (never eval paths). If PI1's mean cost exceeds the floor's,
the teacher is defective (§2.1 defect 2): no R-arm is staged.

  python3 scripts/wave8_teacher_check.py --processes 6 --out results/wave8/f1.json --gates results/wave8/gates.json
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_bullwhip import decision_labeler as L                              # noqa: E402
from agent_bullwhip.agents import OrderUpToAgent                              # noqa: E402
from agent_bullwhip.decision_gates import f1_gate, update_json_key            # noqa: E402
from agent_bullwhip.decision_readout import get_grid                          # noqa: E402
from agent_bullwhip.engine import ROLES, GameStepper, SimConfig               # noqa: E402

HORIZON = 36
CURR = ("poisson", "truncnorm")


def play(seed: int, i: int, grid: list[int], m: int, mode: str) -> float:
    """One closed-loop game. mode 'pi1' | 'floor'."""
    rng = L._seeded_rng(f"{seed}:ep:0")
    params, demand = L.sample_curriculum_demand_ex(rng, CURR[i % 2], HORIZON)
    step = GameStepper(demand, SimConfig(horizon=HORIZON))
    floor = {r: OrderUpToAgent(L.CONTINUATION_THETA, L.CONTINUATION_LAM) for r in ROLES}
    il: dict[str, list[int]] = {r: [] for r in ROLES}
    while not step.done():
        week = step.t
        obs = step.observations()
        for r in ROLES:
            il[r].append(obs[r]["incoming_last"])
        if mode == "floor" or week >= HORIZON - 2:
            if mode == "floor":
                orders = {r: floor[r](obs[r]) for r in ROLES}
            else:     # orders at t >= H-2 land after the horizon: play the continuation floor
                cont = L.make_continuation(L.CONTINUATION_THETA, L.CONTINUATION_LAM, il, week)
                orders = {r: cont[r](obs[r]) for r in ROLES}
        else:
            cont = L.make_continuation(L.CONTINUATION_THETA, L.CONTINUATION_LAM, il, week)
            orders = {}
            for r in ROLES:
                rng_f = L._seeded_rng(f"{seed}:f1:{week}:{r}")
                fut = [L.draw_future_demand(rng_f, CURR[i % 2], params, HORIZON - week)
                       for _ in range(m)]
                J = L.score_candidates(step, week, r, grid, cont, fut, L.CONTINUATION_THETA,
                                       L.CONTINUATION_LAM, il)
                orders[r] = grid[min(range(len(J)), key=J.__getitem__)]
        step.advance(orders)
    return float(step.log().total_cost())


def _game(t):
    seed, i, grid, m = t
    return {"seed": seed, "pi1": play(seed, i, grid, m, "pi1"), "floor": play(seed, i, grid, m, "floor")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=30)
    ap.add_argument("--seed-base", type=int, default=9000)
    ap.add_argument("--m-futures", type=int, default=16)
    ap.add_argument("--grid", default="w8_g26b")
    ap.add_argument("--processes", type=int, default=1)
    ap.add_argument("--out", default="results/wave8/f1.json")
    ap.add_argument("--gates", default=None)
    a = ap.parse_args(argv)
    seeds = range(a.seed_base, a.seed_base + a.games)
    if any(L.EVAL_SEED_MIN <= x < L.EVAL_SEED_MAX for x in seeds) or a.seed_base + a.games > 9100:
        raise SystemExit("F1 seeds must stay out of the eval band 1000..1029 and below S_nom 9100")
    grid = list(get_grid(a.grid).values)
    tasks = [(a.seed_base + i, i, grid, a.m_futures) for i in range(a.games)]
    t0 = time.time()
    if a.processes > 1:
        with mp.get_context("fork").Pool(a.processes) as pool:
            res = list(pool.imap(_game, tasks))
    else:
        res = [_game(t) for t in tasks]
    v = f1_gate([r["pi1"] for r in res], [r["floor"] for r in res])
    out = {"grid": a.grid, "m_futures": a.m_futures, "seeds": [r["seed"] for r in res],
           "pi1": [r["pi1"] for r in res], "floor": [r["floor"] for r in res], "verdict": v,
           "wall_sec": round(time.time() - t0, 1), "processes": a.processes}
    p = Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    if a.gates:
        update_json_key(a.gates, "F1", {"verdict": v, "seeds": [a.seed_base, a.seed_base + a.games - 1],
                                        "m_futures": a.m_futures, "source": str(p)})
    print(f"F1: PI1 mean {v['pi1_mean']:,.0f} vs floor mean {v['floor_mean']:,.0f} "
          f"(PI1 cheaper in {v['wins']}/{v['n']}) -> {'PASS' if v['pass'] else 'FAIL (teacher defective)'} "
          f"wall={out['wall_sec']}s -> {p}")
    return 0 if v["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
