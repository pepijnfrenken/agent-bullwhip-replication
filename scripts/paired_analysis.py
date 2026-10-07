#!/usr/bin/env python3
"""paired_analysis.py — per-path (paired) comparisons of the Jev protocol arms.

The interleaved runner gives every config the SAME demand path per pass, so the
right statistic is the per-pass delta, not the difference of means across
different paths. This script pairs:

  results/jev_protocol/*.jsonl       (config costs per pass, 'run' = pass index)
  results/walkforward_paths.json     (per-path walk-forward / untuned / oracle floor)

and prints mean paired delta, SE, t and win counts for every config vs every
reference. References: anchor_tuned/anchor_only (from the same run set) and the
floor numbers computed on the identical paths.

Usage: python3 paired_analysis.py [--dir results/jev_protocol]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics

PATTERNS = ["step", "noisy", "chaotic", "wild"]


def load_runs(d: str) -> dict[str, dict[str, dict[int, float]]]:
    """{pattern: {config: {pass: cost}}} — pattern inferred from file order."""
    out: dict[str, dict[str, dict[int, float]]] = {}
    files = sorted(glob.glob(f"{d}/*.jsonl"), key=os.path.getmtime)
    for i, f in enumerate(files):
        pat = PATTERNS[i] if i < len(PATTERNS) else f"file{i}"
        per: dict[str, dict[int, float]] = {}
        for line in open(f):
            if not line.strip():
                continue
            r = json.loads(line)
            if not r.get("orders"):
                continue
            per.setdefault(r["config"], {})[r["run"]] = r["total_cost"]
        out[pat] = per
    return out


def paired(deltas: list[float]) -> str:
    m = statistics.mean(deltas)
    if len(deltas) < 2:
        return f"{m:>+9,.0f} (n=1)"
    se = statistics.stdev(deltas) / len(deltas) ** 0.5
    t = m / se if se else float("inf")
    wins = sum(1 for x in deltas if x < 0)
    return f"{m:>+9,.0f}  SE {se:>7,.0f}  t={t:>6.2f}  better {wins}/{len(deltas)}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/jev_protocol")
    args = ap.parse_args()
    runs = load_runs(args.dir)
    wf = {}
    try:
        for r in json.load(open("results/walkforward_paths.json")):
            wf.setdefault(r["pattern"], {})[r["pass"]] = r
    except FileNotFoundError:
        wf = {}

    for pat, per in runs.items():
        if not per:
            continue
        print(f"===== {pat} =====")
        refs: dict[str, dict[int, float]] = {}
        for name in ("anchor_tuned", "jev_anchor_only"):
            if name in per:
                refs[name] = per[name]
        if pat in wf and wf[pat]:
            refs["wf_floor"] = {i: r["wf_cost"] for i, r in wf[pat].items()}
            refs["untuned_cold"] = {i: r["untuned_cost"] for i, r in wf[pat].items()}
            refs["oracle"] = {i: r["oracle_cost"] for i, r in wf[pat].items()}
        for cfg, costs in sorted(per.items()):
            if cfg in refs:
                continue
            line = f"  {cfg:>16}  n={len(costs)}"
            for rname, rcosts in refs.items():
                common = sorted(set(costs) & set(rcosts))
                if len(common) < 2:
                    continue
                d = [costs[i] - rcosts[i] for i in common]
                line += f"\n      vs {rname:<15}: {paired(d)}"
            print(line)
        print()


if __name__ == "__main__":
    main()
