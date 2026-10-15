# W8 R2+RL VERDICT (job 419129) — 2026-10-15

## What ran
- R2+RL (W8_ARM=r2, W8_R2_LR=lr2e-4, --array=0-2): 3 seeds x 3 RL rounds, all 3 seeds
  completed all 3 rounds (round_state.json shows the full r1..r3 collect/label/merge/anchor/train/gate
  chain). No collapse stop-markers were created: `collapsed=false` on all 9 round-gates.
  This run is therefore a pure success from a stability angle: the RLCD round loop ran clean.
- The val band was pushed as 30 rows per seed (~90 total), readout composition = argmax on
  a 26-point order grid, zero prompt-hash dups.

## R2+RL band (30 val runs per seed; lower cost = better)
| seed | median | mean | min | max | % under untrained (14194) |
|-----:|-------:|-----:|----:|----:|--------------------------:|
| 0    | 11,234 | 11,412 | 9,015 | 14,686 | 29/30 |
| 1    | 27,638 | 31,952 | 15,232 | 57,747 |  0/30 |
| 2    | 12,696 | 12,769 | 7,143 | 26,259 | 19/30 |
| pooled | 13,528 | — | 7,143 | 57,747 | 48/90 |

Untrained noisy-argmax median = 14,194 (W8-R2-SFT-VERDICT §3).

## G1 pass/fail read (bar: >=15% median reduction vs own untrained readout)
- **Seed0: PASS** (+20.9% improvement, 29/30 val runs under untrained median).
- **Seed2: MARGINAL FAIL** (+10.6% improvement, 19/30 under; not the 15% bar, and
  the high tail is fat: p90 17,854 > untrained median).
- **Seed1: FAILED HARD, in the WRONG DIRECTION** (median 27,638 = 94.7% WORSE than
  untrained; its best val run 15,232 is still worse than untrained 14,194).
  Pooled n=90 median is +4.7% vs untrained — i.e. at the fleet level RL added ~5% value,
  completely cancelled by seed1's blow-up on a per-seed G1 test. Pooled sign test
  48/90 under untrained median -> p ~0.60, no detectable improvement.

## Why seed1 blew up (diagnosis from logs)
- Round-end KL from anchor: seed1 r1 1.77 -> r2 1.64 -> r3 1.91 — it drifted ~2-3x
  farther from the frozen anchor than seed0 (0.62-0.86) or seed2 (0.93-1.54).
- train_log r3 grad_norm spiking to 9.0 (vs seed0/seed2 ~0.8): RLCD's policy-gradient
  step sizes went unstable late in training on seed1 only.
- Collapse reads were always `collapsed=false` at gate time: min_mass stayed
  0.76-0.89 and modal_argmax_share 0.21-0.34, so the collapse monitor did NOT flag it.
  This arm's failure mode is **cost blow-up without distribution collapse** — the
  policy sharpened locally at catastrophic moments instead of converging tightly and
  consistently.
- Seed1 p10-p90 band is 15,600-54,579 on a scale where untrained is 14,194: the RL
  basically UNSHOT the policy in a narrow tail of states.

## Verdict
- **G1: FAIL on the pre-registered bar.** 2/3 seeds below the bar; one seed actively
  catastrophic; pooled sign test shows no improvement over untrained noisy-argmax.
  This is the SAME verdict shape as the SFT-only arm (r2 SFT was harmful in
  W8-R2-SFT-VERDICT), with an important difference: the RL step did not rescue the
  SFT degradation and on one seed made it dramatically worse.
- **G2/G4 (collapse gates): no failure fired** — the guard was silent, but the guard
  is measuring the wrong thing for this arm: min_mass/modality did not capture the
  cost blow-up.

## What this changes for next steps
1. The r2 (2-transshipment) arm at lr 2e-4 is a **dead end** under this pipeline.
   SFT-only harmful, SFT+RL not better, single-seed catastrophic. Do not pursue
   r2 with SFT/RLCD further without a different seed pool / arm budget.
2. Next GPU-h should go to the 1.7B ladder (stage C), which needs the pre-flight
   B1 weight snapshot fixed (job 432656 crashed in the HF download path) before
   any submission.
3. Checkpoint retention: skip-gate() left old model artifacts on scratch; the
   push-side exclusion keeps ONLY metric metadata. If a later re-labelling or
   re-eval needs the adapters, they're at
   `results/wave8_r2rl_lr0.0002_seed{N}_round{K}/checkpoint-*/` on scratch, NOT
   in the repo.
