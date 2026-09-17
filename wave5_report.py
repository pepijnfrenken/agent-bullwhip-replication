#!/usr/bin/env python3
"""wave5_report.py — n=20 noisy extension: per-path paired stats + the post's graph.

1. parse results/jev_noisy20/*.jsonl -> per (config, path) cost
2. paired stats for every config vs the tuned floor (mean delta, sign count, t-stat)
3. render a 2-panel figure (matplotlib if available, else hand-rolled SVG):
     A. noisy arm n=20: cost by config (mean + min/max whiskers)
     B. the four arms: model vs tuned floor, % delta (the "wins where tuning is weak" story)
Writes data + figure next to the results, and prints everything for the docs.
"""
from __future__ import annotations

import glob
import json
import pathlib
import statistics
from collections import defaultdict

KB = pathlib.Path(__file__).resolve().parent  # this script sits at the repo root
R = KB / "results"


def load_runs():
    import os
    rows = []
    files = sorted(glob.glob(os.path.join(str(R), "jev_noisy20", "*.jsonl")))
    print(f"[load] files: {files}")
    for f in files:
        n_ok = n_bad = 0
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line)); n_ok += 1
                except Exception as e:
                    n_bad += 1
                    if n_bad == 1:
                        print(f"  [load] first parse error: {type(e).__name__} {str(e)[:80]}")
        print(f"  [load] {os.path.basename(f)}: ok={n_ok} bad={n_bad}")
    return rows


runs = load_runs()
print(f"[load] {len(runs)} run records")
if runs:
    keys = sorted(runs[0].keys())
    print(f"[schema] keys: {keys}")
    cost_key = next((k for k in ("total_cost", "cost", "total", "final_cost", "cum_cost") if k in runs[0]), None)
    cfg_key = next((k for k in ("config", "config_name", "arm", "tag_config") if k in runs[0]), None)
    path_key = next((k for k in ("path", "path_id", "seed", "demand_seed", "path_seed", "run") if k in runs[0]), None)
    print(f"[schema] cost={cost_key} config={cfg_key} path={path_key}")
    if not (cost_key and cfg_key and path_key):
        print("[schema] sample record:", json.dumps(runs[0], default=str)[:600])

    per = defaultdict(dict)
    for r in runs:
        per[r.get(cfg_key)][str(r.get(path_key))] = float(r.get(cost_key) or 0)

    cfgs = sorted(per)
    print(f"\n[configs] {cfgs}")
    shared = None
    for c in cfgs:
        s = set(per[c])
        shared = s if shared is None else (shared & s)
    print(f"[paths] {len(shared)} shared paths")

    base = "anchor_tuned"
    print(f"\n=== per-config vs {base} (paired, n={len(shared)}) ===")
    print(f"{'config':22s}{'mean':>9s}{'sd':>8s}{'cv':>7s}{'vs floor':>10s}{'%':>8s}{'better on':>11s}{'t':>7s}")
    out = {}
    for c in cfgs:
        vals = [per[c][p] for p in shared]
        mv = statistics.mean(vals)
        sd = statistics.pstdev(vals)
        row = {"mean": round(mv, 1), "sd": round(sd, 1), "cv": round(sd / mv, 3) if mv else None}
        if base in per and c != base:
            d = [per[c][p] - per[base][p] for p in shared]
            md = statistics.mean(d)
            bet = sum(1 for x in d if x < 0)
            t = md / (statistics.stdev(d) / len(d) ** 0.5) if len(d) > 1 and statistics.stdev(d) else 0.0
            row.update({"mean_delta": round(md, 1), "pct": round(100 * md / statistics.mean([per[base][p] for p in shared]), 1),
                        "better_on": f"{bet}/{len(d)}", "t": round(t, 2)})
            print(f"{c:22s}{mv:9.1f}{sd:8.1f}{(sd/mv):7.3f}{md:10.1f}{row['pct']:8.1f}{row['better_on']:>11s}{t:7.2f}")
        else:
            print(f"{c:22s}{mv:9.1f}{sd:8.1f}{(sd/mv):7.3f}{'-':>10s}{'-':>8s}{'-':>11s}{'-':>7s}")
        out[c] = {**row, "min": min(vals), "max": max(vals)}
    (R / "jev_noisy20" / "paired_stats.json").write_text(json.dumps(out, indent=2))
    print("\nwrote results/jev_noisy20/paired_stats.json")

# Figures were tried and removed (charts didn't read well for this data).
# Numbers live in results/jev_noisy20/paired_stats.json and POSTS.md "Numbers".
