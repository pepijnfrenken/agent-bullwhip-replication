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

# ---------------- graph ----------------
AXES = {
    "noisy n=10": (5214, 4263), "noisy n=20": (4773, 4437),
    "step": (None, None), "chaotic": (7677, 9587), "wild": (12111, 13710),
}
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    have_mpl = True
except Exception:
    have_mpl = False
print(f"\n[graph] matplotlib available: {have_mpl}")

if have_mpl and runs:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))
    cs = [c for c in cfgs]
    means = [out[c]["mean"] for c in cs]
    lo = [out[c]["mean"] - out[c]["min"] for c in cs]
    hi = [out[c]["max"] - out[c]["mean"] for c in cs]
    colors = ["#1f6feb" if c == "jev_gate_never" else "#d29922" if c == base else "#8b949e" for c in cs]
    ax1.bar(range(len(cs)), means, yerr=[lo, hi], color=colors, capsize=4, alpha=0.9, edgecolor="white")
    ax1.set_xticks(range(len(cs)))
    ax1.set_xticklabels([c.replace("jev_", "") for c in cs], rotation=20, ha="right", fontsize=9)
    ax1.set_ylabel("mean total cost (lower = better)")
    ax1.set_title("Noisy demand, n=20 paired paths\nblue = model · amber = tuned floor · grey = controls", fontsize=10)
    ax1.grid(axis="y", alpha=0.25)
    for i, c in enumerate(cs):
        if c != base:
            ax1.annotate(f"{out[c]['pct']:+.1f}%", (i, means[i]), ha="center", va="bottom", fontsize=8)
    arms = ["noisy n=10", "noisy n=20", "chaotic", "wild"]
    pcts = [round(100 * (AXES[a][1] - AXES[a][0]) / AXES[a][0], 1) for a in arms]
    ax2.bar(range(len(arms)), pcts, color=["#1f6feb", "#1f6feb", "#d29922", "#d29922"], alpha=0.9, edgecolor="white")
    ax2.axhline(0, color="#30363d", lw=1)
    ax2.set_xticks(range(len(arms)))
    ax2.set_xticklabels(arms, rotation=15, ha="right", fontsize=9)
    ax2.set_ylabel("% vs the tuned floor (+ = model worse)")
    ax2.set_title("Model vs the tuned floor, by arm\n(it wins where tuning is weak)", fontsize=10)
    ax2.grid(axis="y", alpha=0.25)
    for i, p in enumerate(pcts):
        ax2.annotate(f"{p:+.1f}%", (i, p), ha="center", va="bottom" if p >= 0 else "top", fontsize=8)
    fig.suptitle("Jev (decision-native model) vs the tuned order-up-to formula", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out_png = KB / "docs" / "wave5_jev_vs_tuned.png"
    out_png.parent.mkdir(exist_ok=True)
    fig.savefig(out_png, dpi=160)
    print(f"[graph] wrote {out_png}")
else:
    print("[graph] matplotlib missing -- run: .venv/bin/pip install matplotlib, or use the numbers above")
