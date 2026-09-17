#!/usr/bin/env python3
"""walkforward_paths.py — per-path walk-forward floor numbers (for paired comparisons).

Same demand paths as the interleaved protocol runs (seeds 1000+i, horizon 36).
Dumps one record per (pattern, pass): the walk-forward tuned cost + the (theta,lam)
it picked from the first 18 weeks, the untuned a-priori cost, and the full-path
oracle cost. Written to results/walkforward_paths.json so protocol arms can be
compared path-by-path (paired deltas) instead of by means across different paths.
"""
from __future__ import annotations

import json

from agent_bullwhip.agents import OrderUpToAgent
from agent_bullwhip.engine import ROLES, SimConfig, make_demand, run_game

HORIZON = 36
TUNE_UNTIL = 18
SEEDS = [1000 + i for i in range(10)]
PATTERNS = ["step", "noisy", "chaotic", "wild"]
THETAS = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0]
LAMS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def cost(demand, theta, lam):
    agents = {r: OrderUpToAgent(theta=theta, lam=lam) for r in ROLES}
    return run_game(agents, demand, SimConfig(horizon=len(demand))).total_cost()


def best_on_prefix(demand, until):
    best = None
    for th in THETAS:
        for la in LAMS:
            c = cost(demand[:until], th, la)
            if best is None or c < best[0]:
                best = (c, th, la)
    return best[1], best[2]


def main() -> None:
    out = []
    for p in PATTERNS:
        for i, s in enumerate(SEEDS):
            d = make_demand(HORIZON, p, seed=s)
            th, la = best_on_prefix(d, TUNE_UNTIL)
            rec = {
                "pattern": p, "pass": i, "seed": s,
                "wf_theta": th, "wf_lam": la,
                "wf_cost": cost(d, th, la),
                "untuned_cost": cost(d, 3.0, 0.5),
                "oracle_cost": min(cost(d, x, y) for x in THETAS for y in LAMS),
            }
            out.append(rec)
            print(f"{p:>8} pass {i}: wf=({th},{la}) {rec['wf_cost']:>8,.0f}  "
                  f"untuned {rec['untuned_cost']:>8,.0f}  oracle {rec['oracle_cost']:>8,.0f}")
    with open("results/walkforward_paths.json", "w") as f:
        json.dump(out, f, indent=1)
    print("\nwrote results/walkforward_paths.json")


if __name__ == "__main__":
    main()
