#!/usr/bin/env python3
"""make_charts.py — three clean chart directions for the Wave-5 post. Pick one.

A. paired dumbbell  : n=20 noisy paths, model vs tuned floor, one line per path (shows 15/20)
B. editorial bars   : noisy n=20 cost by config, horizontal, value labels, one accent colour
C. wins-where-weak  : two series per arm — % vs untuned policy and % vs tuned floor
"""
from __future__ import annotations

import glob
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

KB = pathlib.Path(__file__).resolve().parent
R = KB / "results"
OUT = KB / "docs"
OUT.mkdir(exist_ok=True)

MODEL, FLOOR, GREY = "#1f6feb", "#d29922", "#8b949e"
plt.rcParams.update({"font.size": 11, "axes.edgecolor": "#d0d7de", "axes.linewidth": 0.8,
                     "axes.labelcolor": "#24292f", "text.color": "#24292f",
                     "xtick.color": "#57606a", "ytick.color": "#57606a", "figure.facecolor": "white"})

# ---- data ----
per = {}
for f in glob.glob(str(R / "jev_noisy20" / "*.jsonl")):
    for line in open(f):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        per.setdefault(r["config"], {})[str(r["run"])] = float(r["total_cost"])
paths = sorted(set.intersection(*[set(v) for v in per.values()]), key=int)
model, floor = per["jev_gate_never"], per["anchor_tuned"]

# ================= A. paired dumbbell =================
fig, ax = plt.subplots(figsize=(8.4, 6.2))
for i, p in enumerate(paths):
    m, fl = model[p], floor[p]
    better = m < fl
    ax.plot([fl, m], [i, i], color=MODEL if better else GREY, lw=2.0, alpha=0.85, solid_capstyle="round", zorder=2)
    ax.scatter([fl], [i], s=34, color=FLOOR, zorder=3)
    ax.scatter([m], [i], s=34, color=MODEL if better else GREY, zorder=3)
ax.set_yticks(range(len(paths)))
ax.set_yticklabels([f"path {p}" for p in paths], fontsize=8.5)
ax.set_xlabel("total cost  (lower is better)")
n_better = sum(1 for p in paths if model[p] < floor[p])
ax.set_title(f"Every path, paired — noisy demand (n={len(paths)})\n"
             f"model (blue) beats the tuned floor (amber) on {n_better}/{len(paths)} paths",
             fontsize=12, loc="left")
ax.grid(axis="x", color="#eaeef2", lw=0.8)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.invert_yaxis()
fig.tight_layout()
fig.savefig(OUT / "chart_A_paired_dumbbell.png", dpi=170)
print("wrote chart_A_paired_dumbbell.png")

# ================= B. editorial bars =================
stats = json.loads((R / "jev_noisy20" / "paired_stats.json").read_text())
labels = {"jev_gate_never": "Jev (model)", "anchor_tuned": "tuned formula",
          "jev_gate_never_m3": "model, ±3 band", "jev_anchor_only": "anchor only (no model)",
          "jev_gate_never_m12": "model, ±12 band", "jev_jitter6_a": "random ±6 jitter"}
order = sorted(stats, key=lambda c: stats[c]["mean"])
vals = [stats[c]["mean"] for c in order]
cols = [MODEL if c == "jev_gate_never" else FLOOR if c == "anchor_tuned" else GREY for c in order]
fig, ax = plt.subplots(figsize=(8.4, 4.4))
bars = ax.barh([labels.get(c, c) for c in order], vals, color=cols, height=0.62)
for b, c in zip(bars, order):
    v = stats[c]["mean"]
    tag = "" if c == "anchor_tuned" else f"   {stats[c]['pct']:+.1f}%"
    ax.text(v + 60, b.get_y() + b.get_height() / 2, f"{v:,.0f}{tag}", va="center", fontsize=10)
ax.set_xlabel("mean total cost — noisy demand, n=20 paths (lower is better)")
ax.set_title("The model is the cheapest config — and ±3 or ±12 makes it worse than the formula",
             fontsize=11.5, loc="left")
ax.set_xlim(0, max(vals) * 1.16)
ax.grid(axis="x", color="#eaeef2", lw=0.8)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "chart_B_editorial_bars.png", dpi=170)
print("wrote chart_B_editorial_bars.png")

# ================= C. wins where tuning is weak =================
arms = ["noisy\n(n=20)", "noisy\n(n=10)", "chaotic", "wild"]
vs_untuned = [-22.8, -22.8, -24.3, -15.2]      # model vs the untuned policy (from the protocol arms)
vs_tuned = [stats["jev_gate_never"]["pct"], -18.2, 24.9, 13.2]  # vs the walk-forward floor
import numpy as np
x = np.arange(len(arms))
w = 0.36
fig, ax = plt.subplots(figsize=(8.4, 4.6))
b1 = ax.bar(x - w / 2, vs_untuned, w, color=MODEL, label="vs untuned policy")
b2 = ax.bar(x + w / 2, vs_tuned, w, color=FLOOR, label="vs tuned formula (walk-forward floor)")
ax.axhline(0, color="#24292f", lw=1)
for bs, data in ((b1, vs_untuned), (b2, vs_tuned)):
    for b, v in zip(bs, data):
        ax.text(b.get_x() + b.get_width() / 2, v + (1.4 if v >= 0 else -3.2), f"{v:+.1f}%",
                ha="center", fontsize=9.5)
ax.set_xticks(x)
ax.set_xticklabels(arms)
ax.set_ylabel("change in cost vs comparator  (− = better)")
ax.set_title("Wins where tuning is weak: it always beats the untuned policy,\nbut only beats the tuned formula when tuning has little to buy",
             fontsize=11.5, loc="left")
ax.legend(frameon=False, fontsize=10, loc="upper left")
ax.grid(axis="y", color="#eaeef2", lw=0.8)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "chart_C_wins_where_tuning_is_weak.png", dpi=170)
print("wrote chart_C_wins_where_tuning_is_weak.png")
