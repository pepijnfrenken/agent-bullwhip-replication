#!/usr/bin/env python3
"""Tuned-floor analysis (AUDIT2 fix): sweep order-up-to (theta, lambda) on the noisy
demand paths and report the honest baseline. Also computes the leaked-floor variant
to quantify the information asymmetry the leak-fix removes.

Run: .venv/bin/python tuned_floor.py
"""
from __future__ import annotations

from agent_bullwhip.engine import make_demand, ROLES, SimConfig, run_game
from agent_bullwhip.agents import OrderUpToAgent

HORIZON = 36
SEEDS = [1000 + i for i in range(10)]


def run_floor(theta, lam, leak=False, seeds=SEEDS):
    """Run the order-up-to policy on the 10 noisy paths. leak=True = every tier sees
    incoming_now (the same-week lookahead the LLM used to get) — quantifies the leak.
    With leak, the forecast advances on the CURRENT week's incoming for all tiers."""
    costs = []
    for s in seeds:
        demand = make_demand(HORIZON, "noisy", seed=s)
        agents = {}
        for r in ROLES:
            a = OrderUpToAgent(theta=theta, lam=lam)
            if leak:
                orig = a.__call__
                def call(ctx, _o=orig, _a=a):
                    # full leak: use incoming_now for EVERY tier (incl. retailer)
                    obs = ctx.get("incoming_now")
                    if obs is None:
                        obs = ctx["incoming_last"]
                    _a._forecast = _a.lam * obs + (1 - _a.lam) * _a._forecast
                    ip = ctx["on_hand"] + ctx["outstanding"] - ctx["backlog"]
                    return int(max(0, round(_a.theta * _a._forecast - ip)))
                a.__call__ = call
            agents[r] = a
        log = run_game(agents, demand, SimConfig(horizon=HORIZON))
        costs.append(log.total_cost())
    return costs


def main():
    print("=== Tuned floor sweep (no leak) — the honest baseline ===")
    best = None
    for theta in [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0]:
        for lam in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            costs = run_floor(theta, lam)
            m = sum(costs) / len(costs)
            if best is None or m < best[0]:
                best = (m, theta, lam, costs)
    print(f"study floor (3.0, 0.5): mean={sum(run_floor(3.0, 0.5))/10:,.0f}")
    print(f"BEST tuned floor: mean={best[0]:,.0f} at (theta={best[1]}, lam={best[2]})")

    print("\n=== Leaked floor (same-week lookahead) — DEPRECATED ===")
    print("The engine no longer leaks incoming_now upstream (AUDIT2 fix), so the leaked")
    print("variant can't be reproduced without re-adding the leak. Audit2 measured it at")
    print("2,920-3,042 (37-40% below the LLM) under the OLD engine — that asymmetry is now removed.")
    best_leak = None
    for theta in [2.0, 2.5, 3.0, 3.5, 4.0]:
        for lam in [0.1, 0.2, 0.3, 0.4, 0.5]:
            costs = run_floor(theta, lam, leak=True)
            m = sum(costs) / len(costs)
            if best_leak is None or m < best_leak[0]:
                best_leak = (m, theta, lam)
    print(f"best leaked floor: mean={best_leak[0]:,.0f} at (theta={best_leak[1]}, lam={best_leak[2]})")

    print("\n=== LLM noisy means (from results/noisy_interleaved) for comparison ===")
    print("kb_system_gated: 4,849 | kb_pointer_verbal: 5,421 (leak-tainted, see AUDIT2)")


if __name__ == "__main__":
    main()
