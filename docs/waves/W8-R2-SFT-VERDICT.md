# WAVE8 R2-SFT - read of the three SFT-only seeds (2026-10-12)

Source: cluster read of `results/wave8_r2sft_seed{0,1,2}/` (`train_log.jsonl`,
`sft_gate_reads.jsonl`), the F3 block of `results/wave8/gates.json`, job 407398 accounting,
job 405888 (untrained control), the labels file and manifest, and the r2 lr-sweep dirs.
All numbers are copied from those artifacts. Derived quantities are marked `[DERIVED]`.

## 1. Arm status

| seed | job | state | elapsed | final val regret | nll | top1 |
|---|---|---|---|---|---|---|
| 0 | 407398_0 | COMPLETED 0:0 | 04:31:24 | 167.995168 | 3.152042 | 0.137185 |
| 1 | 407398_1 | COMPLETED 0:0 | 04:32:45 | 169.637907 | 3.151062 | 0.152311 |
| 2 | 407398_2 | COMPLETED 0:0 | 04:32:02 | 121.384454 | 3.152204 | 0.103151 |

Final val row is `step 2826`, `n_val 4760`. Start record (seed 0): `grid_id w8_g26b`,
`n_batches 2827`, `n_train 90440`, `git_head 4f84981528928e9e89312a9b556ccd65bb53882b`,
`labels_sha 3aeaa7e8c0cf5c494417b70a778ebe310e83ad3f5801c3ae7dd24ab3aff53a0a`.
SubmitLine: `EXPECTED_COMMIT=4f849815... W8_ARM=r2 W8_R2_LR=lr2e-4 --array=0-2`.

**The §1 confound is closed.** 2,827 steps over 90,440 train rows at batch 32 is 2826.25, so every
seed got exactly one full epoch (§4.2), and all three exited 0:0. Wave 7 trained 4.2 % of an epoch;
this run cannot be explained by under-training. The `event: resume, start_step 0` record is expected:
`sft_argv` always passes `--resume`.

## 2. Collapse reads: 36/36 tripped

`wave8_gates.py sft-gates` writes `sft_gate_reads.jsonl` into the arm dir and prints nothing, so the
`grep "sft-gates"` over the slurm log is empty by construction. The reads themselves:

| quantity | observed (36 reads = 12 checkpoints x 3 seeds) |
|---|---|
| `collapsed` | **true in 36/36**, from checkpoint-249 through checkpoint-2826 |
| `I` (state-dependence, nats) | 0.001294 to 0.035171 |
| `median_entropy` | 3.064915 to 3.248811 |
| `modal_argmax_share` | 0.140625 to 0.917969 |
| `perm_rate` | 0.0 in 28/36 reads, max 0.011719 (gate 0.90) |

Frozen F3 control (`results/wave8/gates.json`, n_states 1024):

| `H_lo` | `H_hi` | `H_median_untrained` | `m_star` | `m_hi` | `I0` | `I_hi0` | untrained modal share |
|---|---|---|---|---|---|---|---|
| 1.393329 | 2.021895 | 1.798150 | 0.126953 | 0.148748 | 0.090728 | 0.096992 | 0.876953 |

**Branch = high, "diffuse and state-blind"** (§6.2 rule: `median H > H_hi AND I <= I_hi0`). Every
read has `median_entropy` 3.065 to 3.249 against `H_hi` = 2.022, and `I` at most 0.0352 against
`I_hi0` = 0.096992. The modal-share limb is not what trips: reads with share 0.141 and share 0.918 are
both flagged. So no checkpoint, at any point in a full epoch, carries state information above the
untrained readout's bootstrap bound.

## 3. Controls

Untrained readout on the same S_nom (job 405888, 2026-10-08T09:35):

| quantity | value |
|---|---|
| step-argmax (n=1) | 13,048 |
| step-sampled n=30 | mean 26,112, min 10,632, max 52,808 |
| noisy-argmax n=30 | mean 14,194, min 12,722, max 16,128 |
| permutation invariance (512 states, 4 grids) | 0 invariant, rate 0.000, FAIL |
| collapse-final | collapsed=False, argmax histogram top3 [(4, 898), (7, 113), (3, 8)] |

So on S_nom the untrained model is narrow (median entropy 1.798) and letter-locked (index 4 in
898/1024 states) yet **passes** the collapse gate. The gate is not trivially unsatisfiable, and the
trained readout moved from narrow-and-slightly-state-responsive (`I0` 0.090728) to flat-and-blind.

## 4. Mechanism: the readout learned the letter marginal, not the conditional

Three independent quantities agree.

1. **The label target is not state-blind.** Strided sample of `labels.jsonl` (10,578 rows, every 9th;
   the file is 95,200 rows): `poisson` n=5289 `I_target`=0.4801 `H_target`=2.7692 `H_mean_p`=3.2493;
   `truncnorm` n=5289 `I_target`=0.4689 `H_target`=2.7825 `H_mean_p`=3.2514. Whole-sample
   `I_target`=0.4733 against a gate bar of 0.096992 and a 26-letter ceiling of 3.258. Both curricula
   carry comparable state information, so this is not a labeling artifact.
2. **The trained entropy lands on the marginal, not on the target.** `median_entropy` 3.065 to 3.249
   versus `H(mean_a p)` = 3.2503 (the fixed letter marginal) and the target's per-state entropy 2.777.
   The readout reproduces the marginal's entropy and none of the conditional structure.
3. **The val NLL says the same.** A state-independent predictor that outputs the letter marginal
   scores exactly `H(mean p)` = 3.2503. [DERIVED] The trained readout scores 3.1520 to 3.1522, i.e.
   better than the best possible state-blind predictor by 0.098 nats out of the 0.4733 nats available:
   **about 21 % of the label target's state information recovered, 79 % discarded.** Caveat: the
   marginal baseline is from the strided whole-file sample, the NLL from the 4760-row val split
   (strided 1-in-20, so representative).

Conclusion: the SFT converged to the state-independent letter marginal plus roughly 0.1 nat of state
signal. "The SFT learned nothing" is not supported; "the SFT learned a fifth of what the labels
contain" is.

### 4.1 The gate reads a cross-regime set, and the verdict still stands

The training labels are behaviour-mix states (manifest `behavior`: `p_floor` 0.75, `p_random` 0.25,
per WAVE8 §96) with demand curricula `poisson` / `truncnorm` only. S_nom is the nominal-floor regime
(§6.2/§276: every 2nd state of 16 nominal-floor theta=3.5, lambda=0.2 episodes, seeds 9100 to 9115,
weeks < 34, all 4 roles). So the gate is a read on a regime the labels do not contain.

That does not rescue the arm, because `I` measures output variation **within** the read set, and the
read set varies week, role, episode and demand rate (`params.lam` 13.126 in the S_nom first row). A
learned function of the state features would show `I` > 0 on those states. It shows 0.0013 to 0.0352.
The high-entropy reading also rules out the benign explanation: saturation on out-of-range inputs
would produce a narrow lock, not a 3.1-nat flat distribution. Two separate claims follow, and the
writeup must keep them apart:

* **Supported:** after one full epoch the readout is a state-independent letter prior.
* **Not yet supported:** that this is failure on the training distribution. The in-distribution
  number is the 21 % CE gap in §4, and the protocol eval (G1/G4) has not run for this arm.

## 5. Metric-validity findings

**F-C. The pre-registered lr rule selects on the metric that this failure minimises.** §4.2 fixes the
r2 lr as `{1e-5, 1e-4}` chosen on label-side validation NLL, seed 0, first 500 steps. The 500-step
sweep gives NLL 3.237886 (1e-5), 3.197980 (1e-4), 3.181659 (2e-4) - so NLL selects **2e-4**, which is
`W8_R2_LR=lr2e-4`, the value the three seeds ran. The selection is explained and reproducible. But
NLL against a near-uniform letter marginal (3.2503 versus the 3.2581 uniform ceiling) is lowered most
easily by fitting that marginal, which is exactly what the collapse gate then flags. The §4.2
selection rule and the §6.2 gate are in tension by construction: the NLL-selected lr is the one that
found the marginal fastest. Deviation to record: 2e-4 is not in the §4.2 candidate set; the sbatch
`SWEEP` array is 1e-5 / 1e-4 / 2e-4.

**F-D. At 500 steps the three read metrics disagree violently.**

| lr | val regret @499 | nll @499 | top1 @499 |
|---|---|---|---|
| 1e-5 | 3039.173805 | 3.237886 | 0.126681 |
| 1e-4 | 176.173936 | 3.197980 | **0.013445** |
| 2e-4 | 297.037736 | 3.181659 | 0.064496 |

The best-regret arm (1e-4, 176.17) has top1 1.3 %, a readout answering the same letter everywhere. The
best-top1 arm (1e-5, 0.1267, at the modal share) has a regret of 3,039. Selecting on regret would have
picked a constant policy. And the ordering does not survive training: at step 2826 the 2e-4 seeds land
at 121.4 to 169.6, better than the 500-step 1e-4 arm's 176.17. **A 500-step sweep cannot select an lr
for a 2,827-step run**, and no single scalar can: the only discriminator observed here is the collapse
read itself.

**F-E. The permutation augmentation bought nothing.** `p_perm` = 0.5 for the whole run; `perm_rate` is
0.0 at 28 of 36 checkpoint reads with a maximum of 0.011719 against a 0.90 gate, and the untrained
control also reads 0.000. This is not new information about the grid, but it closes the §4.2 addition
as ineffective at every checkpoint of a full epoch.

## 6. Provenance and deviations

* `git_dirty: true`, 3 files, at both runtime records. They are: `results/wave8/gates.json` (the F3
  block added by the pre-flight, +59 lines, expected), `results/wave8/grid_w8_g26b.json`, and the
  untracked `results-wave1-351673/`. **Commit the grid**: it must not stay uncommitted under a run.
* The grid diff is metadata only: `git_head` 9050447 to b61cea6 and `transformers` "5.14.1" to
  "4.55.4". The `labels` and `values` arrays are untouched, so there is no letter-to-order remap and no
  label/prompt semantics mismatch. Timestamps agree: grid 2026-10-08 00:06:53, `labels.jsonl`
  12:15:36 (labels generated against this grid), `S_nom.jsonl` 2026-10-07 21:42:49.
* The r2 lr-selection record is `results/wave8/sft_lr_selection.json` (the `--out` default of
  `select-sft-lr`). An earlier read of this session reported it missing: that was a glob error,
  `lr_selection*.json` does not match `sft_lr_selection.json`.
* `results/wave8_val/` holds 4 jsonl files (`lr1e-5` seeds 0, 1, 2 and `lr5e-5` seed 0). The eval
  sbatch's band requirement is satisfied, but the r1 band still cannot compare lrs (n=3 against n=1),
  as W8-R1-DRAINED.md §3 records.

## 7. What this licenses, and what the spec says next

R2-SFT collapsed with the mid-letter signature (§6.6 branch: pile-up at a middle letter means
letter-lock, not an edge defect, so R3 does not apply). R1's six arms collapsed from the opposite
side (W8-R1-DRAINED.md: final reads median entropy 0.879 to 1.760, modal share 0.246 to 0.742), so
this wave now has both failure directions on record: R1 locks narrow, R2-SFT goes flat.

Per §6.6 the R2+RL rounds are the next rung, and per §6.5 that rung will stop itself: the in-training
collapse read runs on 256 S_nom states per round and the SFT init already trips the identical read at
every checkpoint, so two consecutive round reads trip and the run halts. [DERIVED] Running those
rounds is still worth its ~1.4 GPU-h, because it turns the §6.6 row into a result instead of an
inference. The ladder (§4.4) is also licensed now: pre-flight passed and the 0.6B run completed with
no infrastructure fault.

Not yet run for this arm: the protocol eval, so G1, G4, G6 (histogram + calibration), G7 (`min_mass`)
and the per-letter argmax histogram are all absent. `wave8_gates.py collapse-final` is what writes the
histogram the §6.6 edge check needs, and it has not been run for these adapters.

## 8. Recommended next submits, in order

1. **Add the collapse read to the lr-sweep arms and select on collapse-clean AND NLL.** F-C and F-D
   are the reason: on this grid NLL alone rewards the marginal, and regret alone rewards a constant.
   The sweeps already save checkpoints every 250 steps, and S_nom plus F3 are frozen, so this is a
   read per checkpoint, not new machinery. Proposal only, the spec change is the owner's call.
2. **Protocol eval for `r2sft` with `W8_R2_LR=lr2e-4`** (the band is present). About 40 minutes: it
   produces G1, G4, G6, G7 and the S_nom histogram, i.e. the paired record against the untrained
   numbers in §3 and the edge check §6.6 needs.
3. **The two R2+RL rounds** (~1.4 GPU-h) to complete the §6.6 row as a measured stop-and-report.
4. **The ladder: R1 recipe at 1.7B, then 4B** (§4.4). This is the submit that converts a 0.6B null
   into a scaling result, and it is the one that decides whether the wave's story is "capacity" or
   "recipe".

## 9. Recompute commands

Independent recomputation of the three central claims, from raw artifacts only:

```bash
R=/scratch-shared/$USER/bullwhip/agent-bullwhip-replication; cd $R
# collapse reads: every row must show collapsed true
cat results/wave8_r2sft_seed*/sft_gate_reads.jsonl
# F3 thresholds and the untrained reference
python3 -c "import json;print(json.load(open('results/wave8/gates.json'))['F3'])"
# I_target and the marginal baseline (the §4 numbers)
python3 - <<'PY'
import json, math
P=[]
for i,l in enumerate(open('results/wave8_labels/labels.jsonl')):
    if i%9==0: P.append(json.loads(l)['p'])
K=len(P[0]); H=lambda p: -sum(x*math.log(x+1e-12) for x in p)
mean=[sum(p[a] for p in P)/len(P) for a in range(K)]; hb=sum(H(p) for p in P)/len(P)
print("H_target_mean %.4f H_mean_p %.4f I_target %.4f logK %.4f"%(hb,H(mean),H(mean)-hb,math.log(K)))
PY
```

## 10. Protocol eval, job 418889 (2026-10-12): the arm is worse than the base model

Nine modes across three seeds, `completed` 30/30 each, `failed=0`, `parse_fail=0`. So the parse half
of G7 passes; everything else fails, and the primary metric fails in the wrong direction.

| mode | untrained (405888) | seed0 | seed1 | seed2 |
| --- | --- | --- | --- | --- |
| step/argmax (n=1) | 13,048 | 21,358 | 29,950 | 45,514 |
| noisy/argmax (n=30, binding for G1/G2) | 14,194 | 21,806 | 33,041 | 40,923 |
| step/sampled (n=30, G4) | 26,112 | 47,982 | 80,712 | 59,576 |

* **G1 FAIL on 3/3 seeds** (the arm needs 2/3). The gate asks for a median cost reduction of at least
  15 % against the arm's own untrained readout; every seed sits 54 % to 188 % *above* it. The paired
  medians, wins and sign-test still need the per-path files (§11), but no pairing can rescue a 1.5x
  to 2.9x worse mean.
* **G2 FAIL at every tier on 3/3.** Against the `wf_tuned` floor 5,346: seed0 4.08x (the *strong* tier
  is <= 4x, so it misses even the weakest tier by 2 %), seed1 6.18x, seed2 7.65x. The untrained
  reference is 2.16x.
* **G4 FAIL on 3/3.** The sampled mean is 2.2x to 3.6x the untrained sampled median (22,207) where the
  gate wants a >= 20 % reduction, and the spread is far wider than the untrained CV 0.48 (seed1: min
  17,346, max 345,882, mean 80,712).
* **G6 recorded:** collapse tripped on 3/3 (I 0.0810 / 0.0905 / 0.0923, bar I_hi0 0.096992, branch
  high, reason "diffuse and state-blind"), permutation gate FAIL on 3/3 (invariance 0.00195 / 0.00000
  / 0.00586 against gate 0.90), with the permuted grids disagreeing with the canonical choice in
  460 to 491 of 512 states. The base model fails the same probe at 0.000, so the failure is inherited
  rather than SFT-induced: **the choice tracks the letter's position in the list, not the option's
  value.**
* **G7:** parse half passes, `min_mass` 0.466 to 0.629 across all nine runs against a bar of 0.9
  (wave-7 observed 0.924 to 0.982). By the spec's own phrasing the readout has left the label set.

Do not compare the §3 S_nom yardsticks (206 / 239 / 1503) to these `mean_cost` numbers: the yardsticks
are teacher cost J per state over the 1,024 nominal states, these are realised multi-week episode
costs over the validation band. Different currency, not comparable.

### 10.1 Correction to §2 and §4: the checkpoint reads are a 256-state subsample

`sft-gates` invokes the collapse read with `--n 256` over `results/wave8/S_nom.jsonl`
(`decision_rounds.py:445`), while the eval's `collapse-final` reads all 1,024 states. S_nom is every
second state of 16 nominal episodes, so a 256-state draw is a narrower slice of state space and
understates I: the *same* final checkpoint reads I 0.0352 in the 256-state read and I 0.0905 in the
1,024-state read. Both sit below the 0.096992 bar, so every branch is unchanged, but the magnitude to
quote is the full read, **I 0.0810 to 0.0923, which is the untrained model's own level (I0 0.0907)**.

That refines §4. The SFT did not destroy state information; it retained about as little as the base
model had and rewrote the marginal. The untrained model puts 0.877 of its argmax on one letter; the
trained models spread it to a modal share of 0.126 to 0.159, with a heavy load on the *minimum* order
(letter A) in greedy decoding on the mixed band (`zero_letter_share` 0.451 to 0.644, versus 0.031 to
0.038 in sampled mode). That is where the episode cost comes from: a diffuse policy that defaults to
ordering nothing accumulates backlog, while a concentrated one at a middling order does not.

`calibration_table`'s `accuracy` means "the chosen candidate's J is within 1 % of that state's
minimum J": 47.5 to 47.9 % per seed, with auroc 0.56 to 0.69 and ece 0.42 against a mean confidence of
0.052. One-step near-optimality in roughly half the rows coexists with a 1.6x to 3.5x worse episode
cost, so the damage is concentrated in the weeks where the choice is not near-optimal.

### 10.2 The spec's own branch

WAVE8 §6.6 maps "R1 fails G1 and collapse tripped **mid-letter**" to "letter-lock = objective or lr,
not grid -> run R2 (warm start); if R2+RL also locks -> INCONCLUSIVE-negative at 0.6B, go to the
ladder". The S_nom argmax histogram peaks for seed2 are indices 13, 1 and 19 (letter A is 0, Z is 25),
so the mid-letter branch applies: **R2+RL is the licensed next run, and R3 (grid) is not licensed.**
The ladder stays licensed as the §6.6 exit.

## 11. Paired G1 statistics (needs the per-path files)

```bash
python3 - <<'PY'
import json, glob, statistics as st, math
def costs(p):
    out = []
    for line in open(p):
        r = json.loads(line)
        out.append(r.get("cost", r.get("mean_cost")))
    return out
base = costs("results/eval/w8-untrained-argmax-noisy.jsonl")
print(f"untrained n={len(base)} median={st.median(base):.1f}")
for p in sorted(glob.glob("results/eval/w8-r2sft-seed*-argmax-noisy.jsonl")):
    t = costs(p); k = min(len(base), len(t))
    d = [t[i] - base[i] for i in range(k)]
    wins = sum(1 for x in d if x < 0)
    pv = sum(math.comb(k, i) for i in range(wins, k + 1)) / 2 ** k
    red = (st.median(base[:k]) - st.median(t[:k])) / st.median(base[:k])
    print(f"{p.split('/')[-1]:44s} median={st.median(t):.1f} ratio_of_medians_red={red:+.1%} "
          f"median_of_per_path_red={st.median([(base[i]-t[i])/base[i] for i in range(k)]):+.1%} wins={wins}/{k} sign_p={pv:.4f}")
PY
```

G1 wants `ratio_of_medians_red` and `median_of_per_path_red` both >= +15 %, `wins >= 20/30` and
`sign_p < 0.02`.
