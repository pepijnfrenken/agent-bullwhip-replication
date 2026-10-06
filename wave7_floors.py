#!/usr/bin/env python3
"""wave7_floors.py — 30-pass protocol floors for the wave-7 eval (AUDIT-1 fix 5).

C1/C3/C4 live on the noisy n=30 arm; the paired analysis needs the walk-forward
referenced floor on the SAME demand paths (seeds 1000+i, horizon 36), plus a
record for the single deterministic step game (the n=1 step-argmax arm). The
floor logic is imported from the wave-5/6 machinery verbatim
(walkforward_paths.cost / best_on_prefix) — nothing is re-derived here.

Output: results/wave7_floors.json — one record per (pattern, pass) for
pattern in {step, noisy}, pass 0..29 (seed = 1000+pass), plus one
{"pattern": "step", "pass": -1, "seed": null} record for the single game.

Run (CPU, any host, ~1-2 min):  python3 wave7_floors.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from agent_bullwhip.engine import make_demand
from walkforward_paths import HORIZON, TUNE_UNTIL, THETAS, LAMS, cost, best_on_prefix

PASSES = 30
UNTUNED = (3.0, 0.5)          # the a-priori (theta, lam) reference


def rec(pattern: str, pass_i: int, seed: int | None, demand: list[int]) -> dict:
    th, la = best_on_prefix(demand, TUNE_UNTIL)
    return {
        "pattern": pattern, "pass": pass_i, "seed": seed,
        "wf_theta": th, "wf_lam": la,
        "wf_cost": cost(demand, th, la),
        "untuned_cost": cost(demand, *UNTUNED),
        "oracle_cost": min(cost(demand, x, y) for x in THETAS for y in LAMS),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="wave-7 30-pass protocol floors")
    ap.add_argument("--passes", type=int, default=PASSES)
    ap.add_argument("--out", default="results/wave7_floors.json")
    args = ap.parse_args()

    out: list[dict] = []
    for pattern in ("step", "noisy"):
        for i in range(args.passes):
            d = make_demand(HORIZON, pattern, seed=1000 + i)
            r = rec(pattern, i, 1000 + i, d)
            out.append(r)
            print(f"{pattern:>6} pass {i:>2}: wf=({r['wf_theta']},{r['wf_lam']}) "
                  f"{r['wf_cost']:>9,.0f}  untuned {r['untuned_cost']:>9,.0f}  "
                  f"oracle {r['oracle_cost']:>9,.0f}")

    d1 = make_demand(HORIZON, "step")            # default seed: the n=1 arm
    r = rec("step", -1, None, d1)
    r["note"] = "single deterministic step game (n=1 step-argmax arm)"
    out.append(r)
    print(f" step single game (n=1): wf={r['wf_cost']:>9,.0f}  "
          f"untuned {r['untuned_cost']:>9,.0f}  oracle {r['oracle_cost']:>9,.0f}")

    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        json.dump(out, f, indent=1)
    wf = sorted(r["wf_cost"] for r in out[:-1])
    print(f"\nwrote {p}  ({len(out)} records)")
    print(f"wf_cost over {len(wf)} seeded passes: median {wf[len(wf)//2]:,.0f} "
          f"min {wf[0]:,.0f} max {wf[-1]:,.0f}")


if __name__ == "__main__":
    main()
