# Canary verdict — array 361358 (tasks 1 & 11)

**Date:** 2026-10-02 · **Code:** HEAD `17eb83b0f4cbf35d86c45eee453d89f53366c727`, both runs `git_dirty=true` (1 file) at launch
**Hardware:** 1x NVIDIA L4 per task · **Seed:** 12345 (both)

## Runs

| task | config | lr | steps | exit | wall (CEST) | checkpoints |
|---|---|---|---|---|---|---|
| 361358_1 | a0_base | 1e-5 | 300/300 | 0 | 02:23 -> 10:48 (8h25m) | 9 ... 299 (every 10) |
| 361358_11 | a10_lr_high | 2e-5 | 300/300 | 0 | 02:27 -> 10:44 (8h17m) | 9 ... 299 (every 10) |

## Verdict: PASS — both rows healthy through the full 300 steps. Release the remaining rows (0, 2-10, 12) when ready.

## Evidence vs the two legacy failure regimes

| signal | legacy freeze (351673 round-2) | legacy collapse (361032_10) | canary _1 (1e-5) | canary _11 (2e-5) |
|---|---|---|---|---|
| loss | frozen 0.031-0.034 plateau | frozen ~0.0356 | oscillates +/-1e-3, no plateau | oscillates +/-1e-3, no plateau |
| loss_kl | not logged | 1.78 | 0 -> 0.0186 (linear drift) | 0 -> 0.0379 (scales with 2x lr) |
| entropy_first | not logged | 3.7e-06 -> 4.3e-05 (collapsed) | 4.19-4.24 stable | 4.14-4.22 (dip to 3.95 ~s185, recovers) |
| grad_norm | not logged | ~1e-5 | 0.28-0.37 | 0.27-0.38 |
| lora_norm | not logged | 17.62 static | 17.2858 -> 17.2959 | 17.2858 -> 17.3022 (2x drift) |
| n_unique_orders | not logged | 1-4 | 16-35 | 10-32 |
| order_max | not logged | 4.44e+07 constant (attractor) | <=5000 sporadic spikes | <=2e4 early spikes -> <=100 late |
| n_nonfinite | - | 0 | 0 (all steps) | 0 (all steps) |

The 44M-order attractor, entropy collapse (~1e-6), gradient death (~1e-5) and the frozen-loss
plateau are all absent. Policy drift scales coherently with the swept knob (2x lr -> 2x KL drift,
~1.6x lora_norm drift) — the dynamics respond to lr the way they should.

## Watch items (carry into the full sweep)

1. **a10 (2e-5) order diversity declines late**: n_unique_orders 32 -> 10-12 across the run;
   order_mean 0.18-0.29 by the end. Mild concentration, not collapse — expected direction at 2x lr,
   but first thing to check on longer runs.
2. **order_max tail spikes**: isolated samples up to 2e4 (run _11 s40) / 5000 (run _1). Means stay
   <=19 — one-off rogue samples, not a regime.
3. **parse_failure_rate ~0.83-0.90 in both canaries** — also present in gpu-pilot_360642 (2026-10-01),
   so it is baseline-wide on this codebase, NOT a canary regression. (The legacy collapsed run logged
   0.0 only because it emitted a constant parseable integer.) Worth one dedicated look at what the
   reward does with unparsed samples; not a canary blocker.
4. **git_dirty=true (1 file) at launch** for both runs — fine for the canary; record/commit the dirty
   file before the full sweep for clean provenance.

## Capacity note for the full sweep

~101 s/step at G=8, T=36, rollout_batch=16 -> ~8.5h per 300-step row on one L4.
Remaining: tasks 0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12 (11 rows). With 12x L4 concurrently it is a
single overnight wave; queueing may split it into two.

## Reproduce this read

```
cd hpc-umbrella/umbrella-results/logs
python3 ~/.hermes/scripts/bw-log-trajectory.py campaign_361358_1.log campaign_361358_11.log
```
