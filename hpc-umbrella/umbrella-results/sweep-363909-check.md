# Sweep check — array 363909 (preliminary read)

**Date:** 2026-10-03 ~12:45 CEST · **Code:** HEAD `e1faf79` (+1 dirty file at launch: a comment-only
`noisy_arm.sh` edit; committed alongside this doc to close the canary's provenance watch item)
**Hardware:** 1x NVIDIA L4 per task · **Seed:** 12345 (a11_seed2: seed 2) · **Walltime cap:** 14 h/task
**Status: PRELIMINARY** — a9_lr_low and a11_seed2 were still running at pull time (step 250/300);
a2_group16 is incomplete (see below). Final verdict once those three land and are re-pulled.

## Runs

| task | config | steps | started (CEST) | ended (CEST) | s/step | status |
|---|---|---|---|---|---|---|
| 0 | pilot (re-verify) | 4 | 10-02 11:57 | 12:04 | ~134 | exit 0 — batch16 path re-verified under the fix |
| 1 | a0_base | 300 | 10-02 02:23 | 10:48 | 101 | exit 0 (canary) |
| 2 | a1_group4 | 300 | 10-02 11:57 | 16:10 | 51 | exit 0 |
| 3 | a2_group16 | **244/300** | 10-02 12:05 | 10-03 01:55 | 204 | **14 h walltime stop; checkpoint-244 saved, exit 0 → RESUME REQUIRED** |
| 4 | a3_horizon24 | 300 | 10-02 14:53 | 20:30 | 68 | exit 0 |
| 5 | a4_horizon52 | 300 | 10-02 16:11 | 10-03 04:26 | 148 | exit 0 |
| 6 | a5_poisson_only | 300 | 10-02 21:04 | 10-03 05:35 | 103 | exit 0 |
| 7 | a6_truncnorm_only | 300 | 10-02 22:29 | 10-03 06:47 | 100 | exit 0 |
| 8 | a7_kl_low | 300 | 10-03 00:31 | 09:01 | 102 | exit 0 |
| 9 | a8_kl_high | 300 | 10-03 01:55 | 10:24 | 102 | exit 0 |
| 10 | a9_lr_low | 250→ | 10-03 04:13 | ~13:00 (eta) | 104 | running, on pace |
| 11 | a10_lr_high | 300 | 10-02 02:27 | 10:44 | 100 | exit 0 (canary) |
| 12 | a11_seed2 | 250→ | 10-03 04:26 | ~12:55 (eta) | 102 | running, on pace |

Notes:

- **task 3**: SIGTERM at 01:55:08 = the `--signal=B:TERM@600` grace, exactly 13 h 50 m after
  allocation (14 h cap). Checkpoint at step 244, clean exit. Walltime math: 300 x 204 s ≈ 17 h,
  i.e. this row cannot finish in one 14 h slot by construction; the split+resume is the designed
  path (runbook §gpu-campaign). The cancel notice in its log names a sibling job id (363918);
  nothing else in the wave references it.
- **task 10**: launched fresh from step 0 — the stale collapsed 361032 checkpoint did NOT
  auto-resume (its log says "no state.json; starting from step 0"). The a9 caveat from the
  previous addendum is satisfied for this wave.

## Health vs the six legacy failure regimes (completed rows)

Legacy regimes (from the canary verdict 361358 and 361032 addendum): collapse (n_unique_orders -> 1,
entropy -> ~1e-6), frozen loss plateaus, the 44 M order attractor, nonfinite values, dead gradients
(~1e-5), static LoRA.

| task | loss range (full run) | loss_kl end | entropy end | grad end | lora_norm end | n_unique end | order_max | nonfinite |
|---|---|---|---|---|---|---|---|---|
| 1 a0_base | -7.7e-4 .. 1.2e-3 | 0.0186 | 4.19 | 0.36 | 17.2959 | 17 | 5e3 | 0 |
| 2 a1_group4 | -2.2e-3 .. 1.6e-3 | 0.0181 | 3.94 | 0.49 | 17.2913 | 12 | 2.8e4 | 0 |
| 3 a2_group16 (244) | -1.0e-3 .. 8.8e-4 | 0.0115 | 4.13 | 0.22 | 17.2947 | 26 | 1.7e4 | 0 |
| 4 a3_horizon24 | -1.2e-3 .. 1.4e-3 | 0.0222 | 3.96 | 0.39 | 17.2948 | 12 | 2.0e4 | 0 |
| 5 a4_horizon52 | -8.9e-4 .. 1.2e-3 | 0.0134 | 4.04 | 0.27 | 17.2911 | 27 | 6.6e4 | 0 |
| 6 a5_poisson | -5.7e-4 .. 1.0e-3 | 0.0167 | 4.18 | 0.35 | 17.2954 | 17 | 5.6e3 | 0 |
| 7 a6_truncnorm | -1.3e-3 .. 9.7e-4 | 0.0154 | 3.97 | 0.33 | 17.2944 | 18 | 2.0e4 | 0 |
| 8 a7_kl_low | -1.4e-3 .. 1.2e-3 | 0.0333 | 4.09 | 0.37 | 17.3003 | 17 | 5e3 | 0 |
| 9 a8_kl_high | -1.1e-3 .. 1.6e-3 | 0.0046 | 4.25 | 0.29 | 17.2878 | 25 | 2.5e7 * | 0 |
| 10 a9_lr_low (250) | -1.0e-3 .. 1.1e-3 | 0.0044 | 4.22 | 0.35 | 17.2902 | 24 | 1.0e4 | 0 |
| 11 a10_lr_high | -7.3e-4 .. 1.2e-3 | 0.0379 | 4.14 | 0.30 | 17.3022 | 12 | 2.0e4 | 0 |
| 12 a11_seed2 (250) | -8.4e-4 .. 1.4e-3 | 0.0166 | 4.10 | 0.33 | 17.3008 | 17 | 1.0e5 | 0 |

\* single step (s280), one rogue sample — see below. Nothing here is a failure regime: no collapse,
no freeze, no attractor, no nonfinite, gradients alive (0.2-0.5), LoRA drifting on every row.

## Knob coherence

- Wall-clock scales with the knobs: group4 51 s | group8 ~101 s | group16 204 s (≈2.0x);
  horizon24 68 s | horizon36 ~101 s | horizon52 148 s. As expected.
- KL drift at ~300 steps scales with beta_kl and lr: beta=0.1 -> 0.0046 (held tight);
  beta=0.02 -> 0.018-0.019; beta=0.005 -> 0.033; lr 5e-6 -> 0.0044; lr 2e-5 -> 0.038.
- Entropy stays within 3.94-4.25 end-of-run (lowest: group4 3.94, horizon24 3.96 — mild sag,
  no collapse).

## Anomalies & watch items (carry to the final read)

1. **a8_kl_high s280: single-step order_max 2.50e7** (order_mean 21667, std 735k; n_unique 25, KL and
   entropy unaffected). One rogue sample with a giant parsed integer — NOT the legacy constant
   4.44e7 attractor. Fold into the parse/reward look (canary watch item 3).
2. **Order-diversity decline is systematic**: every row loses diversity late (n_unique ~32 -> 12-27,
   order_mean drifting low). Not collapse (legacy floor was 1-4). Eval-side question: does it cost
   performance vs the floors?
3. **parse_failure_rate ~0.84-0.87 on every row** — same as the canary baseline; the dedicated look
   at what the reward does with unparsed samples is still owed.
4. Sibling job id 363918 appears only in task 3's cancel notice (same second). No double-runs
   anywhere (every row shows a single fresh start, no unexpected resumes). Confirm or ignore on
   scratch.
5. git_dirty=1 at launch on all rows — that file is the `noisy_arm.sh` comment edit; committed
   alongside this doc so future waves launch clean.

## Actions

- [ ] **Resume task 3**: `cd /scratch-shared/20234428/bullwhip/kit && sbatch --array=3 gpu-campaign.sbatch`
      (auto-resumes from step 244; ~56 steps x 204 s ≈ 3.2 h; fits the 14 h slot; a free L4 appears
      when a9/a11 finish).
- [ ] Re-pull logs after a9/a11 finish (~11:00 UTC) and re-run this check -> final sweep verdict.
- [ ] Pull full results off scratch (train_log.jsonl + adapters) — scratch deletes 14 days after the
      last write; nothing on Umbrella is durable.
- [ ] Then: llama.cpp serve + eval arms (untrained vs trained vs floors); WAVE6.md + AUDIT6.

## Reproduce

```bash
cd hpc-umbrella/umbrella-results/logs
python3 ~/.hermes/scripts/bw-log-trajectory.py campaign_363909_*.log
# per-log compact trajectory + red-flag verdicts (collapse / 44M attractor / nonfinite / frozen loss)
```
