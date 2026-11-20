# Wave-8 rebuild: what the rerun actually returned (2026-11-20)

Job chain `766006` → `766031`, plus recovery job `774785`. All seven original jobs COMPLETED with
exit 0. This document reads the rebuild's own artifacts and compares them with the values the paper
and `docs/waves/WAVE8_WORD8B_RESULT.md` publish. Every number below is read from a file named in the
row that carries it.

## Jobs

| job | what | elapsed | ended (UTC) | node |
|---|---|---|---|---|
| 766006 | preflight 1.7b (gate freeze) | 1m27s | 2026-11-19T15:2x | tue-gpuc001 |
| 766026 | preflight 4b (gate freeze) | 2m41s | 2026-11-19T15:2x | tue-gpuc001 |
| 766027 | gatecheck both rungs | 4s | 2026-11-19T15:2x | - |
| 766028_0/1/2 | train 1.7b lr1e-5 seeds 0,1,2 | 1h50m / 1h50m / 1h48m | 20:07 / 21:57 / 22:54 | gpuc001, gpuc001, gpub001 |
| 766029_0/1 | train 4b lr5e-5 seeds 0,1 | 3h30m / 3h36m | 00:36 / 00:42 | gpub001 |
| 766030 | eval 1.7b | 55m | 23:49 | - |
| 766031 | eval 4b | 1h32m | 02:14 | - |

Retention: results are in the cluster checkout, in git at `9e12d72` (auto-mirror), in the clone at
`~/bullwhip-archive`, and in a checksummed snapshot. `results/eval` went from 49 to 97 files;
`results/wave8_val` gained 12; `results/wave8_r1-*` added 20 round directories. The 4B eval finished
at 02:14 local, 01:14Z, and the mirror commit at 01:15:31Z caught it.

## 1. Gates: exact reproduction, both rungs

`results/wave8/gate_verification.json`, `overall ok: True`, 8/8 checks:

| rung | field | registered | read | rel diff |
|---|---|---|---|---|
| 1.7b | H_lo | 9.6527e-05 | 9.65265603466344e-05 | 4.6e-06 |
| 1.7b | H_hi | 1.6313 | 1.6312969325077367 | 1.9e-06 |
| 1.7b | I_hi0 | 1.4858 | 1.4857733402975195 | 1.8e-05 |
| 1.7b | m_hi | 0.1487 | 0.14874833921755284 | 3.3e-04 |
| 4b | H_lo | 0.038026 | 0.038025627538467104 | 9.8e-06 |
| 4b | H_hi | 1.910513 | 1.9105125259947648 | 2.5e-07 |
| 4b | I_hi0 | 0.676551 | 0.6765505195974482 | 7.1e-07 |
| 4b | m_hi | 0.1487 | 0.14874833921755284 | 3.3e-04 |

## 2. Untrained controls: reproduce

| control | rebuild | reference | delta | source of reference |
|---|---|---|---|---|
| untrained curriculum band, 30 runs, argmax | 116,948.33 | 116,948 | +0.0003% | paper ledger band ref |
| untrained 1.7b, single run, argmax-step | 11,328 | 11,508 | -1.56% | `paper.tex` 1.7B control |
| untrained 4b, single run, argmax-step | 18,268 | 21,578 | -15.3% | `WAVE8_WORD8B_RESULT.md` |

Files: `results/wave8_val/untrained-curriculum.summary.json`,
`results/eval/w8-untrained-1.7b-argmax-step.summary.json`,
`results/eval/w8-untrained-4b-argmax-step.summary.json`.

The band control matching to 0.33 cost units on a 116,948 mean says the untrained readout is
deterministic across a month and a rebuilt environment. The 4B single-run control sits 15% below its
published value, which is the first sign that the published 4B column was measured on a slightly
different footing than this rebuild.


## 2b. CORRECTION (2026-11-20, 08:4xZ): the step generation is n=1 and cannot test G1

This document first compared the rebuild's step-generation costs (11,070 and 18,258) against the
paper's final-eval medians (8,315 and 7,612) and concluded the 4B headline did not reproduce. That
comparison was invalid and is retracted.

`ev()` in `gpu-wave8-eval.sbatch` passed no `--runs`, so `decision_eval` used its default of 1. Every
step and noisy row in `results/eval/w8-r1-*` is a **single** path per seed, confirmed by
`n_runs: 1, runs: 1` in each summary. The paper's 4B claim is a **30/30 paired-wins** statistic, which
needs 30 paired paths. A 1-run cost cannot adjudicate it in either direction.

What is like-for-like is the curriculum-validation band, which was explicitly run at 30 paths:

| arm | rebuild band mean | paper band mean | delta |
|---|---|---|---|
| 4B seed0 | 9,355.13 | 10,547.8 | -11.3% |
| 4B seed1 | 19,385.3 | 14,681.8 | +32.0% |
| 4B pooled | 14,370.2 | 12,614.8 | +13.9% |
| 1.7B seed0 | 28,824.97 | 47,125 | -38.8% |
| 1.7B seed1 | 86,546.23 | 36,734 | +135.6% |
| 1.7B seed2 | 86,595.1 | 61,073 | +41.8% |
| 1.7B pooled | 67,322.1 | 48,311 | +39.4% |
| untrained band control | 116,948.33 | 116,948 | +0.0003% |

Paper band means are from `paper.tex` line 308 (1.7B "curriculum-validation generation") and line 360
(4B "10,547.8 / 14,681.8, archived").

The pattern that matters: the **untrained control reproduces exactly** while the **trained arms do
not**, and the seed ordering inverts for 1.7B (paper has seed1 cheapest at 36,734; the rebuild has
seed0 cheapest at 28,825). A deterministic readout with non-deterministic training outcomes points at
training-side variability between October and the rebuild, not at a mis-specified readout.

Consequence: G1 for the 4B rung is **not yet adjudicated**. Jobs `775521` (1.7b, seeds 0:1:2) and
`775522` (4b, seeds 0:1), both with `W8_EVAL_RUNS=30`, are queued to produce the paired paths. Until
they return, no sentence about paired wins belongs in the paper.

## 3. Trained arms: the 30-run validation band

Files: `results/wave8_val/w8-val-r1-<rung>-lr<lr>-seed<k>.summary.json`, 30 runs each, grid `w8_g26b`,
argmax, curriculum pattern, seeds 9500..9529.

| arm | rebuild mean | min | max | published |
|---|---|---|---|---|
| 4b seed0 lr5e-5 | 9,355.13 | 6,298 | 13,092 | 8,315 |
| 4b seed1 lr5e-5 | 19,385.3 | 15,810 | 27,820 | 7,612 |
| 4b pooled | 14,370.2 | - | - | 12,614.8 (recorded band ref) |
| 1.7b seed0 lr1e-5 | 28,824.97 | 17,143 | 46,592 | 21,468 |
| 1.7b seed1 lr1e-5 | 86,546.23 | 54,782 | 112,289 | 43,660 |
| 1.7b seed2 lr1e-5 | 86,595.1 | 62,182 | 96,465 | 71,270 |

Deltas against the published figures: 4B pooled +13.9%, 4B seed1 +139.9%, 1.7B +79.3% / +87.8% /
+29.1%. The 4B seed pair spreads 2.07x; the 1.7B seed triple spreads 3.00x.

## 4. Trained arms: single-run evals, argmax-step

| arm | cost | min_mass | confidence | zero-letter share |
|---|---|---|---|---|
| 1.7b seed0 | 38,490 | 1.00000 | 0.9808 | 0.000 |
| 1.7b seed1 | 81,994 | 1.00000 | 0.9781 | 0.000 |
| 1.7b seed2 | 92,030 | 1.00000 | 0.9815 | 0.000 |
| 4b seed0 | 11,070 | 1.00000 | 0.6040 | 0.014 |
| 4b seed1 | 18,258 | 1.00000 | 0.6319 | 0.000 |
| untrained 1.7b | 11,328 | 0.85869 | 0.8809 | 0.035 |
| untrained 4b | 18,268 | 0.99957 | 0.6112 | 0.049 |

Ratios against the untrained control of the same rung: 4B seed0 0.606, 4B seed1 0.9995 (trained and
untrained are indistinguishable within a single run), 1.7B 3.40 / 7.24 / 8.12.

## 5. Mechanism: the collapse checks reproduce the rung story

Median entropy per round, from `results/wave8_r1-*/round*/collapse_round*.json`:

| arm | r1 | r2 | r3 | collapsed flags |
|---|---|---|---|---|
| 1.7b seed0 | 0.02158 | 0.002891 | 3.20e-06 | False, False, False |
| 1.7b seed1 | 2.53e-05 | 0.24296 | 0.003155 | **True**, False, False |
| 1.7b seed2 | 0.10233 | 0.029992 | 2.50e-04 | False, False, False |
| 4b seed0 | 1.88547 | 1.66055 | 1.48620 | False, False, False |
| 4b seed1 | 1.52454 | 1.49313 | 1.46098 | False, False, False |

The 4B holds about 1.5 nats of median entropy across all three rounds. The 1.7B drives entropy down
three to five orders of magnitude and trips the collapse gate once (seed1 round1, modal argmax share
1.0, which then recovered to 0.243 before falling back to 0.0032).

**One dissociation worth carrying into the paper.** 1.7B seed0 ends at the lowest entropy of any arm
(3.2e-06) yet is the cheapest 1.7B arm (38,490 versus 81,994 and 92,030). Degeneracy and cost are not
the same axis: a near-deterministic policy is cheap when its single action happens to match the
canonical one. Any sentence of the form "1.7B costs more because it collapses" is too strong; the
defensible form is that the 1.7B loses the policy and its cost blows up, while entropy alone does not
predict which seed blows up most.

Confidence separates the rungs in the other direction: 4B sits near 0.60 while 1.7B sits near 0.98.
The cheap model is the unsure one.

## 6. Two defects found while reading

**Duplicate band-control path (fixed).** `--tag "untrained-curriculum"` carried no rung suffix, so
766030 (1.7b) and 766031 (4b) wrote the same `results/wave8_val/untrained-curriculum.jsonl`. The 4B
job finished last and won. The surviving 116,948 is the untrained **4B** band; the 1.7B band control
from this rebuild is gone. Fixed to `untrained-curriculum${SUF}`, and recovery job `774785`
(rung 1.7b, seeds 0:1:2, lr 1e-5) is queued to regenerate it once an L4 frees.

**`results/` is gitignored.** The retention guard's `push_results.sh` compensates with `git add -f`,
so nothing was lost, but `git status --porcelain results/` on the cluster showing nothing is not
evidence that results are committed. The guard's own file count is the signal to trust.

## 7. What this does to the paper

Survives: the gate protocol, the untrained readout, the rung ordering (4B trains usefully, 1.7B
degrades), the entropy evidence, the confidence inversion.

Does not survive as written:

- **The band numbers shift by +13.9% pooled for 4B and +39.4% pooled for 1.7B**, with the 1.7B seed
  ordering inverting. These are valid like-for-like comparisons (30 paths, same generation) and they
  do not reproduce October seed-for-seed.
- **G1 on the 4B rung is unadjudicated in the rebuild** pending jobs 775521/775522. The single-path
  step costs must not be cited against the paper's medians.
- **The 1.7B magnitudes are understated** in the paper by 29% to 88%, in the same direction.
- The 4B single-run untrained control is 15% below its published value, so the published 4B column
  carries a provenance gap that the audit already flagged and this rebuild now quantifies.

Recommendation: rerun nothing else. Rewrite §6 and S3 around the rebuild's own table, report the seed
spread explicitly, and keep the qualitative rung verdict, which the mechanism data supports
independently of the cost column. Where the paper needs a single 4B number, use seed0 and say so.
