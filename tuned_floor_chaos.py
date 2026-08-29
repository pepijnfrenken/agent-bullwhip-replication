#!/usr/bin/env python3
"""Tuned floor per chaos environment — the fair formula baseline for the crossover run."""
from agent_bullwhip.engine import make_demand, ROLES, SimConfig, run_game
from agent_bullwhip.agents import OrderUpToAgent

HORIZON = 36
SEEDS = [1000 + i for i in range(10)]
PATTERNS = ["noisy", "chaotic", "wild"]

def run_floor(pattern, theta, lam, seeds):
    costs = []
    for s in seeds:
        demand = make_demand(HORIZON, pattern, seed=s)
        agents = {r: OrderUpToAgent(theta=theta, lam=lam) for r in ROLES}
        log = run_game(agents, demand, SimConfig(horizon=HORIZON))
        costs.append(log.total_cost())
    return costs

print(f"{'pattern':<10}{'untuned(3,0.5)':>16}{'best tuned':>22}{'(theta,lam)':>16}{'tuned CV':>10}")
for p in PATTERNS:
    untuned = sum(run_floor(p, 3.0, 0.5, SEEDS)) / 10
    best = None
    for theta in [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0]:
        for lam in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            m = sum(run_floor(p, theta, lam, SEEDS)) / 10
            if best is None or m < best[0]:
                best = (m, theta, lam)
    bm, bt, bl = best
    costs = run_floor(p, bt, bl, SEEDS)
    cv = (sum((x - bm)**2 for x in costs)/len(costs))**0.5 / bm
    print(f"{p:<10}{untuned:>16,.0f}{bm:>22,.0f}{f'({bt},{bl})':>16}{cv:>10.3f}")
