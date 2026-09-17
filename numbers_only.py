#!/usr/bin/env python3
"""numbers_only.py — drop the figures, make the post number-driven.

1. delete the chart PNGs + the chart script (Pino: charts don't read well)
2. strip the plotting block from wave5_report.py (stats stay reproducible)
3. add a "Numbers" section to POSTS.md: a plain-text block (thread-friendly) + a markdown table
4. point WAVE5.md at the numbers instead of the figure
"""
from __future__ import annotations

import pathlib
import subprocess

KB = pathlib.Path(__file__).resolve().parent

# ---- 1. remove figures + chart script ----
for rel in ("docs/wave5_jev_vs_tuned.png", "docs/chart_A_paired_dumbbell.png",
            "docs/chart_B_editorial_bars.png", "docs/chart_C_wins_where_tuning_is_weak.png",
            "make_charts.py"):
    p = KB / rel
    if p.exists():
        subprocess.run(["git", "rm", "-q", "--cached", rel], cwd=KB, check=False)
        p.unlink()
        print(f"removed {rel}")

# ---- 2. strip plotting from wave5_report.py ----
p = KB / "wave5_report.py"
t = p.read_text()
if "# ---------------- graph ----------------" in t:
    t = t.split("# ---------------- graph ----------------")[0] + \
        "# Figures were tried and removed (charts didn't read well for this data).\n" \
        "# Numbers live in results/jev_noisy20/paired_stats.json and POSTS.md \"Numbers\".\n"
    p.write_text(t)
    print("stripped plotting from wave5_report.py")

# ---- 3. POSTS.md numbers section ----
NUMBERS = """### Numbers (the whole thing, no charts)

```
NOISY DEMAND — 20 paired paths (lower cost = better)
  untuned policy (anchor only)   5,488      +15.0% vs tuned
  Jev (model)                    4,437       -7.1% vs tuned   <- wins on 15/20 paths, t=-2.15
  tuned formula (walk-fwd floor) 4,773         -
  model with +-3 band            4,820       +1.0% vs tuned   <- over-tightening kills it
  model with +-12 band           5,664      +18.7% vs tuned
  random +-6 jitter (control)    6,751      +41.4% vs tuned   <- wins on 1/20

  first read at 10 paths: 5,214 -> 4,263 = -18.2%, 10/10, t=-3.89
  the n=20 re-run shrank it to -7.1% (15/20). Direction holds, magnitude doesn't.

ALL FOUR ARMS — model vs the tuned formula (10 paths each, 240 runs, 0 failures)
  noisy (n=10)   -18.2%     (n=20: -7.1%)
  step          +6.2%      worse on 10/10
  chaotic      +24.9%      loses to the formula, beats the untuned policy by 24.3%
  wild         +13.2%      loses to the formula, beats the untuned policy by 15.2%

  vs the untuned policy the model is better on every stochastic arm: -22.8 / -24.3 / -15.2%.
  What varies is how much tuning buys (5.6% noisy vs 39.3% chaotic) -> it wins where tuning is weak.
```

| noisy n=20 (paired) | untuned | **Jev** | tuned floor | model vs floor | better on |
|---|---|---|---|---|---|
| cost (lower better) | 5,488 | **4,437** | 4,773 | **−7.1%** | **15/20** (t=−2.15) |
| model ±3 band | | 4,820 | 4,773 | +1.0% | 8/20 (ns) |
| model ±12 band | | 5,664 | 4,773 | +18.7% | 3/20 |
| random ±6 jitter | | 6,751 | 4,773 | +41.4% | 1/20 |

| arm (10 paths) | Jev vs tuned floor | Jev vs untuned policy |
|---|---|---|
| noisy | −18.2% (10/10) | −22.8% |
| step | +6.2% (worse 10/10) | +6.2% (worse) |
| chaotic | +24.9% (3/10, ns) | −24.3% |
| wild | +13.2% (3/10, ns) | −15.2% |

"""

p = KB / "POSTS.md"
t = p.read_text()
anchor = "### Single-post version"
assert anchor in t
t = t.replace(anchor, NUMBERS + anchor, 1)
p.write_text(t)
print("added Numbers section to POSTS.md")

# ---- 4. WAVE5.md pointer ----
p = KB / "WAVE5.md"
t = p.read_text()
old = "Paired per-path stats: `results/jev_noisy20/paired_stats.json`; figure: `docs/wave5_jev_vs_tuned.png`."
new = "Paired per-path stats: `results/jev_noisy20/paired_stats.json`; numbers (no charts): `POSTS.md` § Numbers."
assert old in t
p.write_text(t.replace(old, new, 1))
print("WAVE5.md pointer updated")
