#!/usr/bin/env python3
"""depublish.py — enforce: no post drafts in the repo, no mentions of posts anywhere.

1. untrack POSTS.md + gitignore it (file stays on disk, unpublished)
2. strip post references from WAVE5.md (title, framing-audit section, pointer)
3. rewrite AUDIT5.md as a results audit (no post framing)
4. drop the POSTS.md mention from CHANGES.md
5. report any remaining tracked file that still says "post"
"""
from __future__ import annotations

import pathlib
import re
import subprocess

KB = pathlib.Path(__file__).resolve().parent
changed = []


def edit(rel, old, new, must=True):
    p = KB / rel
    t = p.read_text()
    if old not in t:
        if must:
            print(f"  [warn] anchor missing in {rel}: {old[:70]!r}")
        return
    p.write_text(t.replace(old, new, 1))
    changed.append(rel)


# ---- 2. WAVE5.md ----
edit("WAVE5.md", "# WAVE5.md — a decision-native model plays the Beer Game (post draft)",
     "# WAVE5.md — a decision-native model plays the Beer Game")

p = KB / "WAVE5.md"
t = p.read_text()
# drop the framing-audit section (post framing) entirely
if "## 0. Framing audit" in t:
    head, rest = t.split("## 0. Framing audit", 1)
    nxt = re.search(r"\n## (?!0\.)", rest)
    t = head + (rest[nxt.start():].lstrip("\n") if nxt else "")
    p.write_text(t)
    changed.append("WAVE5.md")
# remove lingering pointers
for old, new in (("the story, the numbers, the audit trail, and the open items.", "the story, the numbers, the audit trail, and the open items."),
                 ("POSTS.md", "(draft held locally, not in the repo)")):
    edit("WAVE5.md", old, new, must=False)

# ---- 3. AUDIT5.md — results audit, no post framing ----
(KB / "AUDIT5.md").write_text("""# AUDIT5.md — results audit: n=20 noisy extension + band sweep

Scope: every number used to describe the noisy arm's win, checked against the run data after the
n=20 extension completed. Compiled 2026-09-17. No framing or publication material here — this is
the data audit only.

| # | finding | source | status |
|---|---|---|---|
| 1 | noisy (n=10): model beats the tuned floor by **18.2%**, 10/10 paths, t=−3.89 | `results/jev_protocol/`, `walkforward_paths.json`, `paired_analysis.py` | **superseded** by n=20 |
| 2 | noisy (n=20): **−7.1%, 15/20, t=−2.15** | `results/jev_noisy20/paired_stats.json` | current headline number |
| 3 | band sweep: ±3 **+1.0%** (8/20, ns), ±12 **+18.7%** | same | new; default ±6 is the working point |
| 4 | controls at n=20: anchor-only +15.0%, jitter +41.4% (1/20) | same | new; controls hold |
| 5 | model variance ~1.5× the floor's (CV 0.144 vs 0.094) | same | new caveat |
| 6 | fixed demand: +22% (worse 10/10) | protocol arm | verified, unchanged |
| 7 | chaotic: +24.9% vs tuned, −24.3% vs untuned | protocol arm | verified, unchanged |
| 8 | wild: +13.2% vs tuned, −15.2% vs untuned (t=−4.29) | protocol arm, commit `78e2968` | verified, unchanged |
| 9 | tuning buys 5.6% (noisy) vs 39.3% (chaotic) | `walkforward_paths.json` | verified, unchanged |
| 10 | gate defers 85% (noisy) / 43% (fixed) | protocol summaries | verified, unchanged |
| 11 | retractions stand: one-decision-deep fixed win; `reads` state-level pick | `AUDIT4.md` | verified, unchanged |

## Significance honesty

One arm, one model, t = −2.15 (≈ p<0.05 two-sided) at n=20. The direction holds across samples;
the magnitude dropped from 18.2% to 7.1% when the sample widened. Nothing here establishes
generality (a second decision-native model is still outstanding).

## Reproduce

```
.venv/bin/python wave5_report.py     # parses results/jev_noisy20/*.jsonl -> paired stats
cat results/jev_noisy20/paired_stats.json
```
""")
changed.append("AUDIT5.md")

# ---- 4. CHANGES.md ----
edit("CHANGES.md", ", `POSTS.md` (social drafts)", "")
edit("CHANGES.md", "README v3, CHANGES.md, AUDIT 1-3, POSTS.md, docs/", "README v3, CHANGES.md, AUDIT 1-3, docs/")

# ---- 1. untrack + ignore ----
gi = KB / ".gitignore"
g = gi.read_text() if gi.exists() else ""
if "POSTS.md" not in g:
    g = g.rstrip("\n") + "\n# post drafts stay local — never in the repo\nPOSTS.md\nPOST_*.md\n"
    gi.write_text(g)
subprocess.run(["git", "rm", "-q", "--cached", "POSTS.md"], cwd=KB, check=False)
print("untracked POSTS.md, added to .gitignore")

# ---- 5. report leftovers ----
out = subprocess.run(["git", "grep", "-l", "-i", "-E", r"post[ -]?draft|POSTS\\.md|social drafts"],
                     cwd=KB, capture_output=True, text=True).stdout.strip()
print("tracked files still mentioning post drafts:", out or "none")
print("edited:", ", ".join(dict.fromkeys(changed)))
