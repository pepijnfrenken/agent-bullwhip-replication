# AUDIT5.md — audit of the Wave 5 post (claims vs data)

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
