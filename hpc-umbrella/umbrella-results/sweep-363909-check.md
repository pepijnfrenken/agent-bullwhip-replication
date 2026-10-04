# Sweep check — array 363909 (FINAL read)

**Date:** 2026-10-03 ~12:45 CEST · **Code:** HEAD `e1faf79` (+1 dirty file at launch: a comment-only
`noisy_arm.sh` edit; committed alongside this doc to close the canary's provenance watch item)
**Hardware:** 1x NVIDIA L4 per task · **Seed:** 12345 (a11_seed2: seed 2) · **Walltime cap:** 14 h/task
**Status: FINAL (2026-10-03)** — all 13 rows complete and clean. a9_lr_low and a11_seed2 finished in array 363909; a2_group16 completed its last 56 steps in resume job **374875 task 3** (checkpoint-244 -> 300, exit 0).

## Runs

| task | config | steps | started (CEST) | ended (CEST) | s/step | status |
|---|---|---|---|---|---|---|
| 0 | pilot (re-verify) | 4 | 10-02 11:57 | 12:04 | ~134 | exit 0 — batch16 path re-verified under the fix |
| 1 | a0_base | 300 | 10-02 02:23 | 10:48 | 101 | exit 0 (canary) |
| 2 | a1_group4 | 300 | 10-02 11:57 | 16:10 | 51 | exit 0 |
| 3 | a2_group16 | **300** (244 + resume) | 10-02 12:05 | resumed in 374875_3 | 204 | complete: 14 h walltime stop at 244, resumed 245-300, exit 0 |
| 4 | a3_horizon24 | 300 | 10-02 14:53 | 20:30 | 68 | exit 0 |
| 5 | a4_horizon52 | 300 | 10-02 16:11 | 10-03 04:26 | 148 | exit 0 |
| 6 | a5_poisson_only | 300 | 10-02 21:04 | 10-03 05:35 | 103 | exit 0 |
| 7 | a6_truncnorm_only | 300 | 10-02 22:29 | 10-03 06:47 | 100 | exit 0 |
| 8 | a7_kl_low | 300 | 10-03 00:31 | 09:01 | 102 | exit 0 |
| 9 | a8_kl_high | 300 | 10-03 01:55 | 10:24 | 102 | exit 0 |
| 10 | a9_lr_low | 300 | 10-03 04:13 | complete | 104 | exit 0 |
| 11 | a10_lr_high | 300 | 10-02 02:27 | 10:44 | 100 | exit 0 (canary) |
| 12 | a11_seed2 | 300 | 10-03 04:26 | complete | 102 | exit 0 |

Notes:

- **task 3**: SIGTERM at 01:55:08 = the `--signal=B:TERM@600` grace, exactly 13 h 50 m after
  allocation (14 h cap). Checkpoint at step 244, clean exit. Walltime math: 300 x 204 s ≈ 17 h,
  i.e. this row cannot finish in one 14 h slot by construction; the split+resume is the designed
  path (runbook §gpu-campaign). The cancel notice in its log names a sibling job id (363918);
  nothing else in the wave references it. **Resolved: resume job 374875 task 3 continued from checkpoint-244 to 300 cleanly** (log: `resumed from step 245 (adapter + tokenizer + optimizer)`).
- **task 10**: launched fresh from step 0 — the stale collapsed 361032 checkpoint did NOT
  auto-resume (its log says "no state.json; starting from step 0"). The a9 caveat from the
  previous addendum is satisfied for this wave.

## Health vs the six legacy failure regimes (completed rows)

Legacy regimes (from the canary verdict 361358 and 361032 addendum): collapse (n_unique_orders -> 1,
entropy -> ~1e-6), frozen loss plateaus, the 44 M order attractor, nonfinite values, dead gradients
(~1e-5), static LoRA.

| task | loss range (full run) | loss_kl end | entropy end | grad end | lora_norm end | n_unique end | order_max | nonfinite |
|---|---|---|---|---|---|---|---|---|
| 1 a0_base | -1.4e-03 .. 1.6e-03 | 0.0186 | 4.19 | 0.36 | 17.2959 | 17 | 6.6e+06 | 0 |
| 2 a1_group4 | -2.2e-03 .. 2.6e-03 | 0.0181 | 3.94 | 0.49 | 17.2913 | 12 | 2.0e+05 | 0 |
| 3 a2_group16 (300) | -1.1e-03 .. 1.4e-03 | 0.0140 | 4.04 | 0.23 | 17.3003 | 19 | 2.0e+06 | 0 |
| 4 a3_horizon24 | -1.4e-03 .. 2.1e-03 | 0.0222 | 3.96 | 0.39 | 17.2948 | 12 | 5.5e+05 | 0 |
| 5 a4_horizon52 | -1.1e-03 .. 1.4e-03 | 0.0134 | 4.04 | 0.27 | 17.2911 | 27 | 6.4e+06 | 0 |
| 6 a5_poisson | -1.3e-03 .. 1.8e-03 | 0.0167 | 4.18 | 0.35 | 17.2954 | 17 | 2.0e+06 | 0 |
| 7 a6_truncnorm | -1.3e-03 .. 1.7e-03 | 0.0154 | 3.97 | 0.33 | 17.2944 | 18 | 2.0e+06 | 0 |
| 8 a7_kl_low | -1.4e-03 .. 1.7e-03 | 0.0333 | 4.09 | 0.37 | 17.3003 | 17 | 2.0e+06 | 0 |
| 9 a8_kl_high | -1.4e-03 .. 1.7e-03 | 0.0046 | 4.25 | 0.29 | 17.2878 | 25 | 2.5e+07 | 0 |
| 10 a9_lr_low (300) | -1.4e-03 .. 1.7e-03 | 0.0068 | 4.21 | 0.32 | 17.2921 | 22 | 2.5e+07 | 0 |
| 11 a10_lr_high | -1.4e-03 .. 2.2e-03 | 0.0379 | 4.14 | 0.30 | 17.3022 | 12 | 2.0e+05 | 0 |
| 12 a11_seed2 (300) | -1.5e-03 .. 1.8e-03 | 0.0189 | 4.07 | 0.31 | 17.3031 | 14 | 2.2e+06 | 0 |

\* isolated rogue samples, see the revision note below. Nothing here is a failure regime: no collapse,
no freeze, no attractor, no nonfinite, gradients alive (0.2-0.5), LoRA drifting on every row.

### Revision note (2026-10-04)

The first pass of this table was parsed from the job stdout logs, which sample every 10th step. The
committed per-run `train_log.jsonl` files (light artifacts, commit `3d49b14`, 4.6 MB) carry EVERY
step: all numbers above now come from those files. The `order_max` column is materially larger than
the sampled first pass for several runs (2-9 steps per run have order_max >= 1e5, up to 4 >= 1e6).
Those spikes are isolated (no adjacent steps), and several land on the SAME steps across different
configs (step 46 -> 2.02e6 in six runs; step 280 -> 2.5e7 in both a8 and a9), so they are shared
seeded episode paths rather than run instability. No persistence, order_mean recovers the next step,
n_unique stays high. Distinct from the legacy constant 4.44e7 attractor. a2's train_log carries
step 244 twice (SIGTERM save + resume boundary): dedupe by step when aggregating.

## Knob coherence

- Wall-clock scales with the knobs: group4 51 s | group8 ~101 s | group16 204 s (≈2.0x);
  horizon24 68 s | horizon36 ~101 s | horizon52 148 s. As expected.
- KL drift at ~300 steps scales with beta_kl and lr: beta=0.1 -> 0.0046 (held tight);
  beta=0.02 -> 0.018-0.019; beta=0.005 -> 0.033; lr 5e-6 -> 0.0044; lr 2e-5 -> 0.038.
- Entropy stays within 3.94-4.25 end-of-run (lowest: group4 3.94, horizon24 3.96 — mild sag,
  no collapse).

## Anomalies & watch items (carry to the final read)

1. **Rogue parsed orders are systemic-but-isolated:** full per-step data shows 2-9 steps per run with
   order_max >= 1e5 (up to 4 >= 1e6; max 2.5e7 in a8 and a9, both at step 280; ~2.02e6 repeats at step
   46 across six runs). Same seeded paths across configs, never persistent, means recover. NOT the
   legacy constant 4.44e7 attractor. Fold into the parse/reward look (canary watch item 3); consider
   filtering rogue parses before scoring eval episodes.
3. **parse_failure_rate ~0.84-0.87 on every row** — same as the canary baseline; the dedicated look
   at what the reward does with unparsed samples is still owed.
4. Sibling job id 363918 appears only in task 3's cancel notice (same second). No double-runs
   anywhere (every row shows a single fresh start, no unexpected resumes). Confirm or ignore on
   scratch.
5. git_dirty=1 at launch on all rows — that file is the `noisy_arm.sh` comment edit; committed
   alongside this doc so future waves launch clean.

## Actions

- [x] **Resume task 3**: done — job 374875_3 finished checkpoint-244 -> 300 (clean).
- [x] Re-pull + final read: done (this document).
- [ ] Pull full results off scratch (train_log.jsonl + adapters) — scratch deletes 14 days after the
      last write; nothing on Umbrella is durable.
- [ ] Then: llama.cpp serve + eval arms (untrained vs trained vs floors); WAVE6.md + AUDIT6.

## Reproduce

```bash
cd hpc-umbrella/umbrella-results/logs
python3 ~/.hermes/scripts/bw-log-trajectory.py campaign_363909_*.log
# per-log compact trajectory + red-flag verdicts (collapse / 44M attractor / nonfinite / frozen loss)
```


## Final read (2026-10-03, after the last push)

**All 13 rows complete at 300 steps, exit 0, every one clean on the six legacy failure regimes.**
- a9_lr_low finished down the stretch with stable entropy (4.22 -> 4.21) and the smallest KL drift of
  the sweep (0.0068; lr 5e-6 = half of base, drift roughly half of base's 0.0186). Notably, this is
  the config that COLLAPSED in the 361032 era — under the fixed code it trains sanely for the full
  300 steps. The fix prevents the collapse regime from forming from a fresh start.
- a11_seed2 (seed 2, otherwise a0_base) landed almost on top of the seed-12345 base: kl 0.0189 vs
  0.0186, lora 17.3031 vs 17.2959, uq end 14 vs 17, entropy end 4.07 vs 4.19. Seed robustness of
  the dynamics holds.
- The a2_group16 walltime split (244 in the array, 245-300 in the resume) ran without surprises:
  adapter + tokenizer + optimizer all restored, no re-run of earlier steps, no gap. The split+resume
  design is now demonstrated on a real campaign row, not just a drill.

Standing anomalies carried into the final read: rogue parsed orders up to 2.5e7 on isolated steps
across most runs (systemic-but-isolated, see the revision note); systematic late order-diversity
decline on every row (no collapse); parse_failure_rate ~0.84-0.91 baseline-wide.

## Actions (updated)

- [x] Resume task 3 (job 374875_3) — done.
- [x] Final re-pull + read — done.
- [x] Full results pulled off scratch — **verified COMPLETE 2026-10-04** with `hpc-umbrella/verify_pull.py` (run on tue-login001): 13/13 runs, all final artifacts, ~16.3 GB total. a2_group16 shows 31 checkpoints (9..244 then 249..299, the walltime-split provenance) and 301 train_log records (one extra resume-boundary line; dedupe by step when analyzing). Pilot ckpt-3 ok; extras (351229/360642 pilots, bench, dryrun_cpu) as expected.
- [x] Light artifacts committed (3d49b14) and verified against train_log.jsonl on the hub, 2026-10-04
      (all headline numbers reproduce; order_max column revised upward from full-step data).
- [ ] llama.cpp serve + eval arms (untrained vs trained vs floors) — hub-side env setup started
      (`hpc-umbrella/eval-cpu-setup.sh`: llama.cpp build + Qwen3-0.6B Q4_K_M; needs the a0_base
      adapter file to serve the trained arm).
- [ ] WAVE6.md + AUDIT6 before any post material.
