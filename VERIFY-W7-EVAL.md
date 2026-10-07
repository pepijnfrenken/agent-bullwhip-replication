# VERIFY-W7-EVAL.md — independent recomputation of the wave-7 eval readout

**Scope.** Independent verification of the wave-7 gate readout (Slurm job 398422, eval artifacts at
commit `020f43d`) and adversarial test of its mechanism claim. Recomputed from the raw artifacts
with separate code (no import of `wave7_analysis.py`); scripts run ad hoc, not committed.

**Artifacts read** (all under `results/`): `eval/w7-*.jsonl` (21 protocol arms), `eval/w7-*-perm.json`
(7 probes), `eval/*.summary.json`, `wave7_floors.json`, `wave7_report.md`, `wave7_logs/wave7-eval_398422.log`,
`local-eval/trained-{step,noisy}/qwen3-0.6b-trained-baseline.jsonl` (C3 refs). Spec: `WAVE7_SPEC.md`
(§4 + AUDIT-1 amendment 5 + A1). Code read: `wave7_analysis.py`, `agent_bullwhip/decision_readout.py`,
`agent_bullwhip/decision_eval.py`, `probe_permutation_invariance.py`, `wave7_floors.py`.

---

## Item 1 — per-arm table from raw jsonl — **CONFIRMED (0/21 mismatches)**

Independent recompute over `results/eval/w7-{arm}-{mode}-{pattern}.jsonl` (21 arms). `runs` =
records with a non-null `total_cost` and non-empty `orders`; `fail` = complement; `CV` = stdev/mean;
`uniq` = distinct order values across all roles/weeks; `edge %` = share of orders in {0,40};
`rogue` = runs with any order ≥ 1e5.

| arm | mode | pattern | runs | fail | median | mean | CV | uniq orders | edge % | rogue |
|---|---|---|---|---|---|---|---|---|---|---|
| untrained | argmax | step | 1 | 0 | 10,464 | 10,464 | 0.00 | 5 | 0.0 | 0 |
| untrained | sampled | step | 30 | 0 | 22,207 | 23,493 | 0.48 | 14 | 6.1 | 0 |
| untrained | argmax | noisy | 30 | 0 | 11,569 | 11,442 | 0.08 | 7 | 0.0 | 0 |
| sft-seed0 | argmax | step | 1 | 0 | 19,788 | 19,788 | 0.00 | 11 | 20.1 | 0 |
| sft-seed0 | sampled | step | 30 | 0 | 45,872 | 48,503 | 0.45 | 21 | 10.6 | 0 |
| sft-seed0 | argmax | noisy | 30 | 0 | 20,594 | 21,604 | 0.16 | 14 | 20.5 | 0 |
| sft-seed1 | argmax | step | 1 | 0 | 21,120 | 21,120 | 0.00 | 10 | 11.1 | 0 |
| sft-seed1 | sampled | step | 30 | 0 | 46,954 | 49,318 | 0.42 | 21 | 10.0 | 0 |
| sft-seed1 | argmax | noisy | 30 | 0 | 25,721 | 25,900 | 0.12 | 17 | 8.7 | 0 |
| sft-seed2 | argmax | step | 1 | 0 | 26,038 | 26,038 | 0.00 | 6 | 0.0 | 0 |
| sft-seed2 | sampled | step | 30 | 0 | 46,752 | 47,566 | 0.47 | 21 | 8.4 | 0 |
| sft-seed2 | argmax | noisy | 30 | 0 | 25,508 | 25,970 | 0.06 | 17 | 0.3 | 0 |
| rlcd-seed0 | argmax | step | 1 | 0 | 37,526 | 37,526 | 0.00 | 1 | 0.0 | 0 |
| rlcd-seed0 | sampled | step | 30 | 0 | 35,982 | 37,440 | 0.19 | 8 | 3.5 | 0 |
| rlcd-seed0 | argmax | noisy | 30 | 0 | 37,384 | 37,468 | 0.01 | 1 | 0.0 | 0 |
| rlcd-seed1 | argmax | step | 1 | 0 | 32,646 | 32,646 | 0.00 | 1 | 0.0 | 0 |
| rlcd-seed1 | sampled | step | 30 | 0 | 33,200 | 34,301 | 0.18 | 7 | 0.0 | 0 |
| rlcd-seed1 | argmax | noisy | 30 | 0 | 32,516 | 32,601 | 0.01 | 2 | 0.0 | 0 |
| rlcd-seed2 | argmax | step | 1 | 0 | 27,778 | 27,778 | 0.00 | 1 | 0.0 | 0 |
| rlcd-seed2 | sampled | step | 30 | 0 | 33,104 | 31,043 | 0.37 | 15 | 0.8 | 0 |
| rlcd-seed2 | argmax | noisy | 30 | 0 | 27,652 | 27,727 | 0.01 | 1 | 0.0 | 0 |

Machine diff against the `## Per-arm stats` table in `results/wave7_report.md`: **21/21 rows match**
(runs, fail, median, mean, CV to 2 dp, uniq orders, edge %, rogue). No mismatch to report. All 21
designed arms present; no rogue orders anywhere (as expected for a letter-readout and stated in the
report).

---

## Item 2 — C1 recomputation and gate audit — **CONFIRMED (numbers); gate applied consistently**

Paired per-run (`run` index) vs the *same-mode-same-pattern* `untrained` arm. `wins` = runs with
trained cost < untrained cost; `sign p` = one-sided exact binomial P(X≥wins); `red1` = 1 − median
ratio; `red2` = median of per-path relative reductions.

| trained arm | mode | pattern | n | wins | med trained | med untrained | red1 | red2 | sign p | C1 |
|---|---|---|---|---|---|---|---|---|---|---|
| sft-seed0 | argmax | step | 1 | 0/1 | 19,788 | 10,464 | −89.1% | −89.1% | 1.00e+00 | n/a (n=1) |
| sft-seed0 | sampled | step | 30 | 2/30 | 45,872 | 22,207 | −106.6% | −106.4% | 1.00e+00* | fail |
| sft-seed0 | argmax | noisy | 30 | 0/30 | 20,594 | 11,569 | −78.0% | −84.9% | 1.00e+00 | fail |
| sft-seed1 | argmax | step | 1 | 0/1 | 21,120 | 10,464 | −101.8% | −101.8% | 1.00e+00 | n/a (n=1) |
| sft-seed1 | sampled | step | 30 | 1/30 | 46,954 | 22,207 | −111.4% | −109.4% | 1.00e+00* | fail |
| sft-seed1 | argmax | noisy | 30 | 0/30 | 25,721 | 11,569 | −122.3% | −126.0% | 1.00e+00 | fail |
| sft-seed2 | argmax | step | 1 | 0/1 | 26,038 | 10,464 | −148.8% | −148.8% | 1.00e+00 | n/a (n=1) |
| sft-seed2 | sampled | step | 30 | 2/30 | 46,752 | 22,207 | −110.5% | −109.2% | 1.00e+00* | fail |
| sft-seed2 | argmax | noisy | 30 | 0/30 | 25,508 | 11,569 | −120.5% | −129.4% | 1.00e+00 | fail |
| rlcd-seed0 | argmax | step | 1 | 0/1 | 37,526 | 10,464 | −258.6% | −258.6% | 1.00e+00 | n/a (n=1) |
| rlcd-seed0 | sampled | step | 30 | 4/30 | 35,982 | 22,207 | −62.0% | −70.7% | 1.00e+00* | fail |
| rlcd-seed0 | argmax | noisy | 30 | 0/30 | 37,384 | 11,569 | −223.1% | −223.3% | 1.00e+00 | fail |
| rlcd-seed1 | argmax | step | 1 | 0/1 | 32,646 | 10,464 | −212.0% | −212.0% | 1.00e+00 | n/a (n=1) |
| rlcd-seed1 | sampled | step | 30 | 5/30 | 33,200 | 22,207 | −49.5% | −70.7% | 1.00e+00* | fail |
| rlcd-seed1 | argmax | noisy | 30 | 0/30 | 32,516 | 11,569 | −181.1% | −181.0% | 1.00e+00 | fail |
| rlcd-seed2 | argmax | step | 1 | 0/1 | 27,778 | 10,464 | −165.5% | −165.5% | 1.00e+00 | n/a (n=1) |
| rlcd-seed2 | sampled | step | 30 | 12/30 | 33,104 | 22,207 | −49.1% | −25.8% | 9.00e−01* | fail |
| rlcd-seed2 | argmax | noisy | 30 | 0/30 | 27,652 | 11,569 | −139.0% | −138.7% | 1.00e+00 | fail |

Every wins / median / red1 / red2 value matches the report's C1 table. The only differences are
display rounding of `sign p`: the report prints `1.00e+00` where the exact one-sided binomial p is
`0.9999958` (4 wins), `0.9999703` (5), `0.99999997` (2), etc. No PASS anywhere; all 30-pair rows fail
hard (negative reductions, ≤12/30 wins); the three n=1 rows are report-only.

**Gate application vs pre-registered wording (WAVE7_SPEC §4 + AUDIT-1 fix 5).**

- *Win threshold "≥20/30".* `wave7_analysis.gate_c1` uses `need = 20` for n ≥ 30 and
  `ceil(2n/3)` for n < 30, plus `red1 ≥ 15%` AND `red2 ≥ 15%` AND `sign_p < 0.02`. On the binding
  arm (n = 30) `need = 20`, exactly the pre-registered number. The `ceil(2n/3)` scaling and the
  "both reduction variants" requirement are analyzer syntheses not literally in §4, but both are
  **disclosed in the report text**, and both are *stricter* readings — they cannot turn a fail into a
  pass. No verdict changes.
- *Which variant binds.* §4 (draft) said "the paired step arm"; AUDIT-1 amendment 5 moves C1/C3/C4
  to the **noisy n=30** arm and demotes step-argmax to a single labeled cost. The analyzer's header
  and the report both state this. The analyzer additionally gates the sampled-step arm (the wave-6
  T=0.7 counterpart, not a pre-registered binding arm). **Verdict is invariant under either reading**:
  every trained arm loses on *both* step-sampled and noisy.
- *n=1 step-argmax treatment.* Report-only; `gate_c1` marks n < 5 as `n/a (n=1)`. Matches AUDIT-1
  §5 P0. Correct.

**No deviation here changes the verdict.**

---

## Item 3 — permutation artifacts — **CONFIRMED (invariance); letter/value mode diagnosed**

Recomputed from `results/eval/w7-*-perm.json`. `values[k][s]` = argmax order under grid `k`;
`grids[0]` = canonical ascending; implied letter index `L = grid[k].index(value)`. `letter_const` =
share of states whose letter index is identical across all 4 grids; `letter_const`/`mean|dV|` etc.
are over 512 states × 4 grids.

| arm | n | invariance | disagree vs canonical (grids 1..3) | modal L (share) | letter_const | mean \|ΔV\| | mean \|ΔL\| | mode |
|---|---|---|---|---|---|---|---|---|
| untrained | 512 | 0.000 (0/512) | [490, 512, 512] | 4 (0.486) | 0.0% | 10.57 | 1.13 | neither |
| rlcd-seed0 | 512 | 0.000 (0/512) | [512, 512, 512] | 17 (0.626) | 0.0% | 8.22 | 0.37 | neither (*near-letter*) |
| rlcd-seed1 | 512 | 0.000 (0/512) | [511, 512, 512] | 15 (0.962) | 85.5% | 16.64 | 0.06 | **letter-dominated** |
| rlcd-seed2 | 512 | 0.000 (0/512) | [491, 512, 512] | 13 (0.758) | 7.4% | 14.06 | 0.48 | neither (*near-letter*) |
| sft-seed0 | 512 | 0.000 (0/512) | [458, 446, 453] | 5 (0.289) | 0.0% | 13.66 | 5.87 | neither |
| sft-seed1 | 512 | 0.000 (0/512) | [431, 510, 482] | 5 (0.279) | 0.0% | 13.97 | 5.67 | neither |
| sft-seed2 | 512 | 0.000 (0/512) | [490, 498, 457] | 5 (0.445) | 0.0% | 7.16 | 7.04 | neither |

- Invariance rates and per-grid disagreement match the report's table and the job log
  (`results/wave7_logs/wave7-eval_398422.log`) exactly. **No trained arm is value-preserving; the
  gate fails for all six (0.000 < 0.90).** Untrained is report-only and also 0.000. **CONFIRMED.**
- **Value-preserving / letter-preserving / neither** (as asked): **none value-preserving.** rlcd-seed1
  is *letter-dominated* (96.2% of picks at one letter position, 85.5% fully letter-constant across
  grids; mean |ΔL| = 0.06). rlcd-seed0 and rlcd-seed2 are *near-letter* (one modal letter index at
  63%/76%, mean |ΔL| ≤ 0.48) but not literally constant. All three SFT arms are **neither**: both the
  value and the letter index swing (mean |ΔL| 5.7–7.0). The report only prints invariance + gate; the
  letter-index breakdown above is new here and shows the failure *mode* differs by arm — RLCD answers
  by *option position*, SFT by neither value nor position.

Consistency cross-check: in argmax mode the RLCD arms are constant-order policies (Item 4), so a
*value-driven* policy would score 100% invariance — the measured 0% is direct proof the RLCD readout
answers by letter position, not by order value.

---

## Item 4 — confound test: is "trained worse" a dominant-order artifact? — **CONFIRMED: yes**

**Constant-order cost curve** (all four roles order `k` every week, engine `SimConfig` defaults,
horizon 36; `make_demand(36,"step")` = the n=1 game path; noisy = median over seeds 1000..1029):

```
k     :  0     2     4     6     8    10    12    14    16    18    20    22    24    26    28    30    32    34    36    38    40
step  : 9552  9568  9584  9600  9616  9692 11280 13444 15752 18120 20512 22928 25344 27778 30212 32646 35080 37526 39972 42418 44864
noisy : 9817  9833  9849  9865  9881  9950 11362 13445 15716 18053 20426 22817 25233 27652 30081 32515 34949 37383 39826 42272 44718
```

Monotone increasing; **best constant = 0** on both. The curve is steep: const-26/30/34 ≈ 27.8k / 32.6k
/ 37.5k, vs const-8 = 9.6k and the wf_tuned floor = 3,206.

**Per-arm dominant order, modal share, within-game distinct orders** (argmax mode; the gate-binding
mode):

| arm | dominant order | modal share (argmax/step, noisy) | distinct orders/game | cost of that constant order |
|---|---|---|---|---|
| untrained | 8 | 0.74 / 0.78 | 5 (step), 4–6 (noisy) | 9,616 |
| sft-seed0 | 34 | 0.56 / 0.56 | 11 (step), 8–11 (noisy) | 37,526 |
| sft-seed1 | 34 | 0.59 / 0.58 | 10 (step), 7–12 (noisy) | 37,526 |
| sft-seed2 | 34 | 0.76 / 0.83 | 6 (step), 5–11 (noisy) | 37,526 |
| rlcd-seed0 | 34 | **1.00 / 1.00** | **1 (step), 1 (noisy)** | 37,526 |
| rlcd-seed1 | 30 | **1.00 / 1.00** | **1 (step), 1–2 (noisy)** | 32,646 |
| rlcd-seed2 | 26 | **1.00 / 1.00** | **1 (step), 1 (noisy)** | 27,778 |

**Per-run identity to a constant-order policy** (arm cost vs `const_cost(demand, dominant_k)` on the
*same* demand path):

- rlcd-seed0 argmax-noisy: **30/30 runs exact** (mean |Δ| = 0.0); argmax-step 1/1 exact.
- rlcd-seed2 argmax-noisy: **30/30 exact**; argmax-step 1/1 exact.
- rlcd-seed1 argmax-noisy: **28/30 exact** (mean |Δ| = 9.3); argmax-step 1/1 exact.
- untrained / sft arms: 0/30 exact (they are not constant).

**Answer, plainly.**

1. **The RLCD arms are effectively constant-order policies** in the gate-binding argmax mode: each
   plays a single order (34 / 30 / 26) on every role/week, exactly reproducing the constant-k cost.
   The SFT arms are *near*-constant: dominant order 34 at 56–83% share with 6–12 distinct orders.
2. **The untrained arm is not constant** (dominant 8, 5 distinct orders on the step game — a low-order
   regime near the cheap end of the curve).
3. **The C1 failure follows directly from dominant-order drift, not from an unusually good control.**
   Training moved the argmax to large orders (26–34) that sit high on a monotone cost curve; the
   untrained control happened to sit at order 8. Constant-34 alone costs 37,526 — ~3.9× the untrained
   cost and ~11.7× the floor. The untrained control is *itself* not optimal (best constant order 0 =
   9,552 < 10,464, and the tuned floor is 3,206), so "trained worse than untrained" is a
   dominant-order artifact plus a weak control, not evidence that training damaged a strong policy.
   Mechanism claim **CONFIRMED**.

---

## Item 5 — harness sanity — **CONFIRMED, with two minor labeling gaps**

- **Run counts 1/30.** All 21 arms present; `*-argmax-step.jsonl` = 1 record each; `*-sampled-step`
  and `*-argmax-noisy` = 30 each. Matches `decision_eval` (`n_runs = 1 if pattern=="step" and
  mode=="argmax" else 30`). **CONFIRMED.**
- **Canonical ascending grid for eval renders.** `decision_eval` calls
  `render_prompt(ctx, tokenizer)` with no grid → `render_options(ORDER_GRID)` default
  `require_ascending=True`; `ORDER_GRID = [0,2,…,40]`. Passing a reversed grid raises
  `ValueError: option grid must be ascending`. Only the perm probe renders shuffled grids (its own
  probe artifact, not the eval protocol). **CONFIRMED.**
- **`prompt_hash_dups`.** 0 in all 427 protocol records across the 21 arms (7 arms × 61). **CONFIRMED.**
- **Parse failures.** `failures` dicts all zero for every run (`assert` in `run_batched_eval`);
  0 records rejected by the loader; every `*.summary.json` reports `parse_fail: 0`, `failed: 0`.
  **CONFIRMED.**
- **n=1 arm marked report-only.** Correct in C1 (`n/a (n=1)` on all six step-argmax rows), C2
  (column header `step-argmax (n=1)`), C4 (`n/a (n=1)`). **Gaps:** the R1 table prints the three n=1
  step-argmax rows (`0/1`, `p=1.00e+00`) with no report-only/`n/a` caveat, and the C3 *step* table
  shows the argmax rows at `n shared = 1` / `clean n = 0` without a report-only note. The `n` column
  makes n=1 visible in both, and neither table feeds a gate, so this is cosmetic. **PARTIAL — minor.**
- Light checks (not required, cheap): C3 reference files exist and their rogue-run lists recompute
  to exactly the report's `[0,3,4,12,13,14,19,20,27,29]` (step) and `[8,9]` (noisy); R1 sampled-step
  wins recompute to 21/22/23 with median Δ −10,565 / −11,520 / −13,760, matching the report. Floors
  recompute to step wf/untuned/oracle = 3,206 / 3,681 / 3,206 and noisy = 5,346 / 5,423 / 4,594,
  matching the report's floors table. **CONFIRMED.**
- Trivial nit: report says "strong tier ≤ 12,800 (4×) / headline ≤ 6,400 (2×)"; 4×3,206 = 12,824 and
  2×3,206 = 6,412. The code constants are 12,800 / 6,400, so the report quotes the code, not the
  literal product. Immaterial (no arm is near either boundary).

---

## Item 6 — `WAVE7.md` cross-check — **PRESENT NOW; recomputed (10 minor claims fail, none changes the verdict)**

`WAVE7.md` appeared at the repo root partway through this verification (another agent). It asserts
every number was recomputed; I re-checked its numbers against mine.

**Core result claims — all CONFIRMED against my recomputation.** §3 C1 table (all six noisy rows
+ the step-argmax/step-sampled summaries), C2 table, C3 step/noisy medians (wave-6 ref medians
recompute to **325,575** step and **59,684** noisy; rogue runs `[0,3,4,12,13,14,19,20,27,29]` and
`[8,9]`), C4 `NO` verdicts (its "z up to 6.2 / 132 / 2.57" match my z computed in the analyzer's
`|Δmean|/√(se_i²+se_j²)` convention: 6.18 / 132.1 / 2.58), the perm table (invariance 0.000 and
min-mass 0.9677/0.9631/0.9695/0.9244/0.9473/0.9812/0.9937), the entire §4 letter-mechanism table
(all 7 arms' per-grid modal letters/counts and mean |Δvalue| / |Δletter| — identical), the
per-role-game distinct-order/modal-share table, the floors, the labeler stats
(`states=95200 flat=0.0022 edge=0.7118 entropy=2.213`), the eval `MaxRSS ≈ 1.16 GB`
(`1161564K`), the train-job histories (385450 "exit=0 then FAILED", 385598 `torch.cuda.OutOfMemoryError`
on 22 GiB L4; 390955 all exits 0), and `parse_fail=0`.

**Claims that fail (all minor; none alters the verdict):**

1. **§3 R1 med-Δ range "+1.7k to +17.2k"** — WRONG low end. The binding argmax-noisy med Δ is
   seed0 **+17,184**, seed1 **+6,966**, seed2 **+2,116** → range **+2.1k … +17.2k**. The `1,740`
   figure is the seed-2 *step-argmax n=1* row (report-only) pulled into the binding-variant range.
2. **§3 R1 "sampled-step … p ≤ 0.02"** — WRONG for seed 0: R1 sampled-step wins 21/30 at
   **p = 2.14e-02** (> 0.02); seeds 1–2 are 8.06e-03 / 2.61e-03 (≤ 0.02).
3. **§1/§6 "61 wave-7 runs"** — mis-stated total. 61 is the record count **per arm** (1+30+30);
   the protocol total is **427 records** (21 files), i.e. **61,488 decisions**. "4,320+ decisions"
   corresponds to a single 30-record arm, so the run/decision figures don't multiply out.
4. **§4 "every role in every noisy game emits one order value for all 36 weeks"** (RLCD) — overstated:
   `w7-rlcd-seed1-argmax-noisy` runs **24 and 26** have the retailer emitting `{30, 32}`; only
   **28/30** runs are exactly constant 30 (seed0 and seed2 are 30/30). Its own table already reports
   rlcd-seed1 as range 1–2.
5. **§1 untrained label-mass min "0.9977"** — the actual min is **0.99759** (→ 0.9976); the trained
   range (0.9292–0.9831) is correct.
6. **§4 SFT "modal order 34 at share 0.56–0.86"** — measured shares are **0.556 / 0.583 / 0.832**
   → **0.56–0.83**, not 0.86.
7. **§4 "target argmax at order 0 in 28–42%"** — measured **27.5% / 33.0% / 41.7%** (order 40:
   51.8% / 43.3% / 35.2%, matching "35–52%").
8. **§4 "or one step off it (P, N)"** — R = letter index 17; P = index 15 (**2** grid steps) and
   N = index 13 (**4** steps); not one step.
9. **§4 train top-1 "0.03 → 0.13–0.25"** — the start is **0.03125** for seeds 0/1 but **0.0** for
   seed 2 (`train_log.jsonl`); the end range (0.125–0.25) is right.
10. **§4 "full-vocab letter mass from 0.995"** — `train_log.jsonl` step-0 `mean_label_mass` is
    **0.9966 / 0.9967 / 0.9972** (~0.997), not 0.995; the end range (final `min_label_mass`
    0.9633–0.9727 ≈ "0.963–0.973") is right.

Nothing in this list changes the negative readout, the C1/C2/C3/C4 verdicts, the permutation-gate
failure, or the constant-order mechanism. Unverifiable here: §6's "wave-6 free-form arms with 58–188
rogue orders and 34–40% parse failures" (wave-6 artifacts not in this checkout).

---

## Residual uncertainty

- **Readout not re-executed.** No torch/GPU here and the Qwen3-0.6B snapshot is not on this box, so I
  verified the *recorded artifacts*, not the forward pass. A mis-resolved label id or a template
  drift in the cluster runtime would not be visible from the artifacts alone. Mitigating evidence:
  the RLCD argmax cost equals a constant-order policy **exactly** on 30/30 noisy paths, which is only
  possible if the harness played the recorded order grid faithfully.
- **`values`/`orders` trusted as recorded.** The permutation probe stores only picked *values*, not
  the raw logits; I derived letter indices from `grids` + `values`. The probe JSONs are self-consistent
  (each value is present in its grid; invariance and disagreement reproduce).
- **C3 / R1 not fully recomputed.** Out of the requested scope; I checked rogue lists and wins only.
- **Constant-order pricing** used `agent_bullwhip.engine.run_game` directly (identical `SimConfig`
  defaults, same demand constructors), not the eval harness — but the exact-identity result above ties
  the two together.
- **Sign-p display.** The report rounds several exact p ∈ (0.9999, 1) to `1.00e+00`; harmless but not
  literally exact.

---

## Bottom line

1. **The negative readout survives independent recomputation**: every per-arm stat (21/21) and every
   C1 paired number reproduces; all trained arms lose to their own untrained readout (≤12/30 wins,
   negative median reductions) on both the step-sampled and noisy n=30 arms; all six permutation gates
   fail at 0.000 invariance.
2. **The analyzer is correct** where it can be checked; the only deviations from the pre-registered
   §4 wording (both-reduction-variant test; `ceil(2n/3)` win scaling) are *stricter* and disclosed,
   and no gate application changes any verdict.
3. **The mechanism claim is confirmed and stronger than stated**: in argmax mode the RLCD arms are
   literally constant-order policies (order 34/30/26, 30/30 runs cost-exact), and the SFT arms are
   near-constant on order 34 — these sit high on a monotone cost curve (const-34 = 37,526 ≈ 3.9× the
   untrained 10,464), so the C1 failure is a dominant-order artifact, not a degraded control.
4. **Minor, non-verdict-affecting flags**: R1 and the C3 step table print the n=1 step-argmax rows
   without an explicit report-only caveat (the `n` column still shows 1); the report displays exact
   p ≈ 0.99999+ as `1.00e+00`; C2's "4×/2×" labels quote the code constants (12,800/6,400), not
   4×3,206/2×3,206; and `WAVE7.md` reproduces every core number but carries ~10 minor numeric slips
   (Item 6), none of which touches a gate.
5. **Nothing here flips the verdict**: the interface-alone hypothesis fails at 0.6B, and the failure
   is best explained by the readout collapsing to a (near-)constant large order and a letter-position
   answer, not by measurement error or gate misapplication.
