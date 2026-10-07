#!/usr/bin/env python3
"""Walk-forward tuning: the formula picks (theta,lam) from the FIRST HALF of the game
only, then deploys frozen on the second half. The fair 'you don't know the future' test.
Compares against the LLM agents (which need no tuning) on the same seeded paths."""
from agent_bullwhip.engine import make_demand, ROLES, SimConfig, run_game
from agent_bullwhip.agents import OrderUpToAgent

HORIZON = 36
SEEDS = [1000 + i for i in range(10)]
PATTERNS = ["noisy", "chaotic", "wild"]
TUNE_UNTIL = 18  # see first half only

def run_floor(demand, theta, lam):
    agents = {r: OrderUpToAgent(theta=theta, lam=lam) for r in ROLES}
    log = run_game(agents, demand, SimConfig(horizon=len(demand)))
    return log.total_cost()

def best_on_prefix(demand, until):
    """Grid-search (theta,lam) on the first `until` weeks of demand (simulate the game)."""
    prefix = demand[:until]
    best = None
    for theta in [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0]:
        for lam in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            agents = {r: OrderUpToAgent(theta=theta, lam=lam) for r in ROLES}
            log = run_game(agents, prefix, SimConfig(horizon=until))
            c = log.total_cost()
            if best is None or c < best[0]:
                best = (c, theta, lam)
    return best[1], best[2]

print(f"{'pattern':<10}{'wf-tuned (theta,lam)':>24}{'wf floor mean':>16}{'untuned mean':>14}{'oracle mean':>14}")
for p in PATTERNS:
    wf_costs, unt_costs, oracle_costs = [], [], []
    for s in SEEDS:
        d = make_demand(HORIZON, p, seed=s)
        th, la = best_on_prefix(d, TUNE_UNTIL)
        wf_costs.append(run_floor(d, th, la))
        unt_costs.append(run_floor(d, 3.0, 0.5))
        # oracle = tuned on the full path (unfair baseline, for reference)
        best = None
        for theta in [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0]:
            for lam in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                c = run_floor(d, theta, lam)
                if best is None or c < best[0]:
                    best = (c, theta, lam)
        oracle_costs.append(best[0])
    wf_mean = sum(wf_costs)/len(wf_costs)
    print(f"{p:<10}{'see first 18wk':>24}{wf_mean:>16,.0f}{sum(unt_costs)/10:>14,.0f}{sum(oracle_costs)/10:>14,.0f}")

print("\n=== LLM reference (seeded runs, same paths) ===")
print("chaotic: kb_pointer_verbal 9,156 | kb_system_gated 11,525")
print("wild:    kb_pointer_verbal ~14,196 (partial) | kb_system_gated ~13,755 (partial)")
print("noisy:   kb_pointer_verbal 5,424 | kb_system_gated 5,973")
