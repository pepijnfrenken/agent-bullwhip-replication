#!/usr/bin/env python3
"""Out-of-sample tuned-floor check (AUDIT2 §1b): tune (theta,lam) on 5 seeds, score on the other 5."""
from agent_bullwhip.engine import make_demand, ROLES, SimConfig, run_game
from agent_bullwhip.agents import OrderUpToAgent

HORIZON = 36
SEEDS = [1000 + i for i in range(10)]

def run_floor(theta, lam, seeds):
    costs = []
    for s in seeds:
        demand = make_demand(HORIZON, "noisy", seed=s)
        agents = {r: OrderUpToAgent(theta=theta, lam=lam) for r in ROLES}
        log = run_game(agents, demand, SimConfig(horizon=HORIZON))
        costs.append(log.total_cost())
    return costs

def best_on(seeds):
    best = None
    for theta in [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0]:
        for lam in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            m = sum(run_floor(theta, lam, seeds)) / len(seeds)
            if best is None or m < best[0]:
                best = (m, theta, lam)
    return best

print("=== Out-of-sample tuned floor (split-half, leak-free engine) ===")
for name, tr, te in [
    ("train{0-4}/test{5-9}", SEEDS[:5], SEEDS[5:]),
    ("train{5-9}/test{0-4}", SEEDS[5:], SEEDS[:5]),
    ("train evens/test odds", SEEDS[::2], SEEDS[1::2]),
]:
    bm, bt, bl = best_on(tr)
    oos = sum(run_floor(bt, bl, te)) / len(te)
    print(f"  {name}: tuned=({bt},{bl}) train_mean={bm:,.0f} -> OOS mean={oos:,.0f}")

# reference: full-sample tuned + untuned + leak-free LLM means
full = best_on(SEEDS)
print(f"\n  full-sample tuned: ({full[1]},{full[2]}) mean={full[0]:,.0f}")
print(f"  untuned (3.0, 0.5): {sum(run_floor(3.0, 0.5, SEEDS))/10:,.0f}")
print(f"  leak-free kb_system_gated: 5,973 | kb_pointer_verbal: 5,424 (from results/noisy_leakfree)")
