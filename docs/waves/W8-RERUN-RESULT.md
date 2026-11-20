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

## 8. G1 on the generation the criterion actually names (2026-11-20, 10:5xZ)

`WAVE8_SPEC.md:298` fixes G1 as: paired **noisy-argmax n=30**, median cost reduction >= 15% in *both*
variants (ratio of medians and median of per-path reductions) **and** >= 20/30 paired wins **and**
one-sided sign-test p < 0.02; a seed passes iff all three hold, and the arm needs >= 2/3 seeds. The
paper states the binding win count as >= 22/30, since p < 0.02 at n=30 needs 22.

`decision_eval.py` also carries a binding P0 fix from 2026-10-05 (`ee63271`): step+argmax is
deterministic, so that arm is emitted as a **single** run by design and the sign tests live on noisy
n=30, where the 30 demand paths genuinely differ. Section 2b of this document treated the step arm's
n=1 as a rebuild defect. It is not a defect, it is the spec, and jobs 775521/775522 were therefore
partly redundant. The noisy arm has been n=30 since the first rebuild eval.

Demand paths depend only on `seed_base + i` (`make_demands`, model-independent), so control and
trained rows pair by run index. Control medians reproduce the paper's own figures exactly:
4B noisy median **21,578** (paper: 21,578) and 1.7B noisy median **11,508** with mean 11,405 (paper:
11,508 / 11,405).

### 4B, noisy n=30, matched untrained control

| seed | wins | med ctl | med trained | ratio of medians | median per-path ratio | mean trained | G1 |
|---|---|---|---|---|---|---|---|
| 0 | 30/30 | 21,578 | 9,346 | 0.43 (57% reduction) | 0.40 (60%) | 9,279 | **PASS** |
| 1 | 25/30 | 21,578 | 17,958 | 0.83 (17% reduction) | 0.75 (25%) | 17,288 | **PASS** |

p(wins) one-sided: 30/30 = 9.3e-10, 25/30 = 1.6e-04. Both seeds clear 20/30 and p < 0.02. The arm is
2/2 on seeds, so **G1 passes for 4B** on the rebuild's own artifacts.

### 1.7B, noisy n=30, matched untrained control

| seed | wins | med ctl | med trained | ratio of medians | median per-path ratio | mean trained | G1 |
|---|---|---|---|---|---|---|---|
| 0 | 7/30 | 11,508 | 27,706 | 2.41x | 2.28x | 28,739 | FAIL |
| 1 | 0/30 | 11,508 | 79,094 | 6.87x | 7.88x | 92,064 | FAIL |
| 2 | 0/30 | 11,508 | 94,887 | 8.25x | 8.23x | 93,854 | FAIL |

**G1 fails on all three seeds, so the 1.7B rung is adverse**, as published. Zero ties occurred in any
pair, matching the paper's tie check.

### Same arms on the curriculum band n=30 (secondary)

| arm | wins | med ctl | med trained | ratio |
|---|---|---|---|---|
| 4B seed0 | 30/30 | 133,898 | 9,854 | 0.07x |
| 4B seed1 | 25/30 | 133,898 | 18,334 | 0.14x |
| 1.7B seed0 | 6/30 | 20,563 | 28,069 | 1.37x |
| 1.7B seed1 | 0/30 | 20,563 | 80,033 | 3.89x |
| 1.7B seed2 | 0/30 | 20,563 | 87,512 | 4.26x |

Band control means: untrained 4B 116,948.33, untrained 1.7B 20,154.80 (`runs=30`,
`results/wave8_val/untrained-curriculum-4b.summary.json` and `-1.7b.summary.json`). The 4B band
control came back at 116,948.33 twice, 21 hours apart, bit-identical.

## 9. What survives and what does not, on the rebuild's evidence

Survives, from artifacts read today:

- **4B passes G1 on both seeds.** Seed0 comfortably, seed1 narrowly: 17% reduction against a 15%
  threshold, 25/30 wins rather than 30/30.
- **1.7B is adverse on all three seeds**, at larger magnitudes than published (2.41x / 6.87x / 8.25x
  versus 1.87x / 3.79x / 6.19x), with seed0 short of the published 0/30 (it gets 7/30).
- The rung ordering, the mechanism (4B entropy stable near 1.5 nats, 1.7B driven to 1e-06..1e-04 with
  one collapse trip), and the exact determinism of the untrained readout.

Does not survive as written:

- **The per-seed medians.** 4B seed1 is 17,958 in the rebuild against 7,612 published; the published
  pair (8,315 and 7,612) is tighter than any two seeds this protocol produces. 4B seed0 at 9,346
  against 8,315 is a 12% gap, which is defensible as reproduction; seed1 is not.
- **"30/30 per seed" for 4B.** Seed1 gives 25/30. The verdict does not depend on this, because the
  frozen criterion needs 20/30, but the abstract's phrasing does.
- **"both effect sizes in full"** for 4B seed1, which clears 15% by two points.

Two caveats that must travel with the 1.7B row:

- The untrained 1.7B band read has `min_mass` 0.60945, and its single-run reads are 0.85869
  (argmax-step), 0.543 (sampled), 0.823 (argmax-noisy). The control's readout is partly outside the
  26-letter vocabulary, so the 1.7B comparison rests on a control that fails the C0 mass threshold.
  It does not change the direction, but it belongs in the limitations paragraph.
- The 1.7B band numbers in the paper (47,125 / 36,734 / 61,073) do not reproduce: the rebuild gives
  28,825 / 86,546 / 86,595, and the seed ordering inverts (seed0 cheapest here, seed1 cheapest there).
