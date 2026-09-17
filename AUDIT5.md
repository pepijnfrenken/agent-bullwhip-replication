# AUDIT5.md — results audit: n=20 noisy extension + band sweep

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
