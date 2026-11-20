#!/usr/bin/env python3
"""wave8_g1_read.py — the frozen G1 read for one arm against one matched control.

G1 (docs/waves/WAVE8_SPEC.md:298): paired noisy-argmax generation, n=30 paths,
pass requires a >=15% reduction by BOTH the median ratio and the median of the
per-path proportional reductions, at least 20/30 paired wins, one-sided sign-test
p < 0.02. At n=30 that p needs 22 wins, which is why the paper says 22/30 while
the spec says 20/30; this script prints both so the discrepancy is visible.

Usage:
  python3 scripts/wave8_g1_read.py CONTROL.jsonl ARM.jsonl [--label NAME]

Pairs by the row's "run" field when present, else by row order. Ties are counted
and reported, never silently dropped.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys


def load(path: str) -> dict[int, float]:
    out: dict[int, float] = {}
    with open(path) as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            key = r.get("run", i)
            if isinstance(key, dict):
                key = i
            out[int(key)] = float(r["total_cost"])
    return out


def sign_test(wins: int, losses: int) -> float:
    """One-sided exact binomial p for wins out of wins+losses at p=0.5."""
    n = wins + losses
    if n == 0:
        return 1.0
    return sum(math.comb(n, k) for k in range(wins, n + 1)) / 2 ** n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("control")
    ap.add_argument("arm")
    ap.add_argument("--label", default="")
    a = ap.parse_args()

    ctl = load(a.control)
    arm = load(a.arm)
    keys = sorted(set(ctl) & set(arm))
    if not keys:
        sys.exit("no shared run keys — refusing to compare unpaired rows")

    pairs = [(ctl[k], arm[k]) for k in keys]
    ties = sum(1 for c, t in pairs if c == t)
    wins = sum(1 for c, t in pairs if t < c)
    losses = sum(1 for c, t in pairs if t > c)

    med_c = statistics.median(c for c, _ in pairs)
    med_t = statistics.median(t for _, t in pairs)
    ratio = med_t / med_c if med_c else float("nan")
    per_path = [(t / c if c else float("nan")) for c, t in pairs]
    med_prop = statistics.median(per_path)

    p = sign_test(wins, losses)
    label = a.label or a.arm.split("/")[-1]
    print(f"arm            {label}")
    print(f"pairs          {len(pairs)}  (ties {ties})")
    print(f"control median {med_c:,.2f}")
    print(f"arm median     {med_t:,.2f}")
    print(f"median ratio   {ratio:.4f}   reduction {(1-ratio)*100:6.2f}%")
    print(f"median of per-path reductions {med_prop:.4f}   ({(1-med_prop)*100:6.2f}%)")
    print(f"paired wins    {wins}/{len(pairs)}  losses {losses}  one-sided p {p:.3g}")

    both15 = ratio <= 0.85 and med_prop <= 0.85
    print(f"gate  >=15% both         {'PASS' if both15 else 'FAIL'}")
    print(f"gate  >=20/30 wins       {'PASS' if wins >= 20 else 'FAIL'}")
    print(f"gate  p < 0.02           {'PASS' if p < 0.02 else 'FAIL'}")
    print(f"gate  22/30 (paper text) {'PASS' if wins >= 22 else 'FAIL'}")
    print(f"VERDICT {'PASS' if both15 and wins >= 20 and p < 0.02 else 'FAIL'}")


if __name__ == "__main__":
    main()
