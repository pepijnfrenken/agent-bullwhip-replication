#!/usr/bin/env python3
"""finalize_wave5.py — fold the n=20 + band-sweep results into the docs, write AUDIT5 (post audit).

Edits are exact-anchor replacements; the script fails loudly if an anchor is missing
(no silent no-ops). Prints a summary of what changed.
"""
from __future__ import annotations

import pathlib

KB = pathlib.Path(__file__).resolve().parent
changed = []


def edit(rel: str, old: str, new: str, must: bool = True):
    p = KB / rel
    t = p.read_text()
    if old not in t:
        if must:
            raise SystemExit(f"ANCHOR MISSING in {rel}:\n{old[:120]}")
        return
    p.write_text(t.replace(old, new, 1))
    changed.append(rel)


# ---------------- POSTS.md ----------------
edit("POSTS.md",
     "On noisy demand it beat the tuned 1970s formula by **18.2%** (paired, 10/10 paths).",
     "On noisy demand it beat the tuned 1970s formula by **7.1%** (paired, 20 paths, 15/20, t=−2.15).\n"
     "The first read was **18.2%** (10/10 paths) — the n=20 re-run shrank it, and it still holds. That's the honest number.")

edit("POSTS.md",
     "10 paired demand paths per arm. The walk-forward-tuned floor was recomputed on the *identical* demand paths",
     "20 paired demand paths on the noisy arm (10 on the others). The walk-forward-tuned floor was recomputed on the *identical* demand paths")

edit("POSTS.md",
     "noisy demand: formula **5,214** → Jev **4,263** = **−18.2%**, better on 10/10 paths",
     "noisy demand (n=20): formula **4,773** → Jev **4,437** = **−7.1%**, better on 15/20 paths (t=−2.15)\n"
     "  (n=10 first read: 5,214 → 4,263 = −18.2%, 10/10, t=−3.89 — same direction, smaller magnitude, wider sample)\n"
     "  band sweep: ±3 → **+1.0%** (8/20, ns: tightening the clamp kills the win) · ±12 → **+18.7%** — so the default ±6 is the working point")

edit("POSTS.md",
     "Open: the chaotic/wild arms are still running, plus an n=20 extension, a ±3/±6/±12 band sweep, and the mechanism question — *why* is it right under noise and wrong under flat demand?",
     "Closed since the first draft: the chaotic and wild arms are complete (all four arms, 240 runs, 0 failures) and the n=20 noisy extension + ±3/±12 band sweep finished 2026-09-17 (`results/jev_noisy20/`).\n"
     "Still open: the mechanism — *why* is it right under noise and wrong under flat demand? — and a second decision-native model for generality.")

edit("POSTS.md",
     "Under noisy demand it beat the walk-forward-tuned 1970s formula by **18.2%** (paired, 10/10 paths). Under flat demand it lost by **22%**.",
     "Under noisy demand it beat the walk-forward-tuned 1970s formula by **7.1%** (paired, n=20; 15/20 paths). Under flat demand it lost by **22%**.")

# ---------------- WAVE5.md ----------------
edit("WAVE5.md",
     "**Status:** all four arms complete — 10 passes each, 240 runs, 0 failures (interleaved, seeded, paired). An n=20 noisy extension + ±3/±12 band sweep is running.",
     "**Status:** all four arms complete — 240 runs, 0 failures (interleaved, seeded, paired). "
     "The n=20 noisy extension + ±3/±12 band sweep completed 2026-09-17: the model beats the tuned floor by "
     "**−7.1% (15/20 paths, t=−2.15)** vs −18.2% (10/10, t=−3.89) at n=10 — direction holds, magnitude shrank; "
     "±3 loses to the floor (+1.0%, 8/20) and ±12 is much worse (+18.7%). Data: `results/jev_noisy20/`.")

edit("WAVE5.md",
     "1. Noisy extension to n=20 paths + band sweep ±3/±12 (running).",
     "1. ~~Noisy extension to n=20 + band sweep~~ — **done 2026-09-17**: model 4,436.6 (CV 0.144) vs tuned floor 4,773.4 (CV 0.094) = **−7.1%, 15/20, t=−2.15**; ±3 4,819.9 (+1.0%, 8/20, ns); ±12 5,663.9 (+18.7%); anchor-only 5,487.9 (+15.0%); jitter control 6,750.9 (+41.4%, 1/20). Paired per-path stats: `results/jev_noisy20/paired_stats.json`; figure: `docs/wave5_jev_vs_tuned.png`.")

# ---------------- CHANGES.md ----------------
edit("CHANGES.md",
     "- **Running:** n=20 noisy extension + ±3/±12 band sweep (`results/jev_noisy20/`).",
     "- **Complete (2026-09-17): n=20 noisy extension + ±3/±12 band sweep** (`results/jev_noisy20/`, 120 calls, 0 failures). "
     "Paired, per-path: model **4,436.6** vs tuned floor **4,773.4** = **−7.1%, 15/20 paths, t=−2.15** (n=10 read: −18.2%, 10/10, t=−3.89 — magnitude shrank with the bigger sample, direction holds). "
     "Band sweep: ±3 **+1.0%** (8/20, ns), ±12 **+18.7%** → default ±6 is the working point. Controls at n=20: anchor-only +15.0%, jitter +41.4% (1/20). Model CV 0.144 vs floor 0.094.")

# ---------------- AUDIT5.md ----------------
(KB / "AUDIT5.md").write_text("""# AUDIT5.md — audit of the Wave 5 post (claims vs data)

Scope: every numeric claim in `POSTS.md` (Wave 5 draft), checked against the run data and the
analysis scripts. Compiled 2026-09-17, after the n=20 noisy extension + band sweep completed.

| # | claim in the post | source | status |
|---|---|---|---|
| 1 | noisy: model beats the tuned floor by **18.2%**, 10/10 paths | `results/jev_protocol/` (n=10), `walkforward_paths.json`, `paired_analysis.py` (t=−3.89) | **SUPERSEDED** → n=20 gives **−7.1%, 15/20, t=−2.15**. Post now leads with the n=20 number and shows the n=10 read for honesty. |
| 2 | fixed demand: model loses by **22%** | protocol arm (n=10) | verified (unchanged) |
| 3 | chaotic: **+25%** vs tuned floor, **−24%** vs untuned policy | protocol arm (n=10) | verified (unchanged) |
| 4 | wild: **+13%** vs tuned floor, −15% vs untuned, 10/10 | protocol arm (n=10), commit `78e2968` (t=−4.29 vs untuned) | verified (unchanged) |
| 5 | "correction is consistent on all three stochastic arms (15–24% vs untuned)" | protocol arms | verified (unchanged) |
| 6 | tuning buys 5.6% (noisy) vs 39.3% (chaotic) | `walkforward_paths.json` | verified (unchanged) |
| 7 | controls: jitter +1,659, offset +912, misaligned replay +2,230 (n=10) | `results/jev_controls_noisy/` | verified for n=10; **extension**: n=20 re-run puts jitter at +41.4% (1/20) and anchor-only +15.0% (1/20) — same direction, larger margin |
| 8 | gate defers 85% (noisy) / 43% (fixed), median confidences 0.21 / 0.56 | `results/jev_protocol/` summaries | verified (unchanged) |
| 9 | "two of my own claims died" (one-decision-deep fixed win; reads pick) | `AUDIT4.md` | verified (retractions stand) |
| 10 | all four arms complete, 240 runs, 0 failures | run summaries | verified — post draft updated (previously said "still running") |
| 11 | *new*: ±3 band loses to the floor (**+1.0%, 8/20, ns**), ±12 much worse (**+18.7%**) | `results/jev_noisy20/paired_stats.json` | added to the post (item 3/) as the "it isn't just boldness" evidence |
| 12 | *new*: model variance is ~1.5× the floor's (CV 0.144 vs 0.094) | same | added as a caveat line |

## Framing checks

- **Headline**: "same model, opposite verdicts" + "wins where tuning is weak" — both survive the n=20
  re-run (the win is smaller, not gone). The post now leads with n=20 and *shows* the n=10 read;
  hiding the shrink would contradict the project's own audit style.
- **No overclaiming of significance**: n=20 gives t=−2.15 (≈p<0.05, two-sided) on ONE arm. The post
  does not claim generality; the pending list keeps "second decision-native model" open.
- **Retraction integrity**: the two retracted claims are still labelled retracted in the post body
  (item 6/) and in `AUDIT4.md` — no resurrection via the n=20 numbers.

## Reproduce

```
.venv/bin/python wave5_report.py        # parses results/jev_noisy20/*.jsonl -> paired stats + figure
cat results/jev_noisy20/paired_stats.json
```
""")
changed.append("AUDIT5.md")

print("edited:", ", ".join(dict.fromkeys(changed)))
