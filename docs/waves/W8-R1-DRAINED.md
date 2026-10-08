# WAVE8 R1 — array drained, read of 2026-10-08

Source: cluster read of `results/wave8_r1_*` (6 arms = 2 lrs × 3 seeds) + `results/wave8_val`.
All numbers below are copied from the arm dirs; nothing is inferred except where marked.

## 1. Arm status

| arm | markers | collapse reads per round | final read | G7 (min_mass ≥ 0.9) |
|---|---|---|---|---|
| r1 lr1e-5 seed0 | **DONE + STOPPED_collapse** | C, C, clean | medH 1.760, share 0.391, min_mass 4.0e-05 | **fail** (readout leaves the label set) |
| r1 lr1e-5 seed1 | STOPPED_collapse | C, C | medH 1.083, share 0.492, min_mass 0.925 | pass |
| r1 lr1e-5 seed2 | STOPPED_collapse | C, C | medH 0.896, share 0.633, min_mass 0.090 | **fail** |
| r1 lr5e-5 seed0 | DONE | C, clean, C | medH 0.926, share 0.742, min_mass 0.984 | pass |
| r1 lr5e-5 seed1 | STOPPED_collapse | C, C | medH 0.879, share 0.434, min_mass 0.998 | pass |
| r1 lr5e-5 seed2 | STOPPED_collapse | C, C | medH 1.059, share 0.246, min_mass 6.4e-03 | **fail** |

(`C` = collapse gate tripped. min_mass of a read is `min_mass` in `collapse_reads.jsonl`.)

## 2. Two findings

**F-A. The gate fires on 6/6 arms at round 1.** Round-1 modal-argmax share runs 0.51–1.00 against
the frozen `m_hi = 0.1487` (STAGING-READY §1.3) — 3.4× to 6.7× the bound. Every arm, both lrs, all
seeds. A gate that trips on every candidate at the first read is not ranking arms.

**F-B. R1 is stopped as a whole by its own §6.5 rule.** "All three seeds of the better lr tripping by
end of round 2 stops R1 as a whole (no further lr/seed runs, R3 decided)." lr1e-5: 3/3 tripped twice
by end of round 2. lr5e-5: seeds 1 and 2 tripped twice; seed0 tripped at round 1 and again at round 3.
So whichever lr the band calls "better", all three of its seeds tripped by the end of round 2. The
full protocol eval would select an lr from a rung the spec says is stopped.

## 3. The band cannot carry a selection

`results/wave8_val/` holds 2 files, both seed 0 (`lr1e-5` and `lr5e-5`). `select-lr` pools over the
seeds present, so it compares seed0 against seed0 — n=1 per lr. The submission guard in
`gpu-wave8-eval.sbatch` only warns when the band covers a single lr, so a 1-seed-per-lr band passes it
silently and still writes `lr_selection.json` with a `winner`. The eval's seed loop also hard-fails on
seeds 1 and 2 (`FATAL: missing train output ..._seed1_round3/state.json`), so `W8_EVAL_SEEDS="0"` is
required regardless.

The two band entries each fail a different gate: lr1e-5 seed0 is the only arm of the six whose *final*
read is clean (round 3) but it fails G7 on min_mass; lr5e-5 seed0 passes G7 but its final read is
collapsed. Neither arm is a pass, so there is nothing to select between.

## 4. Not yet known: edge vs mid-letter signature

WAVE8 §6.6 branches on the collapse signature: letter Z share > 5 % or letter A share > 2× the oracle's
(A = order 0, `m* = 0.1270`) → grid/teacher story, run R3. Pile-up at a middle letter → letter-lock =
objective/lr, run R2. **The signature is not on disk**: `collapse_reads.jsonl` rows (written by the
trainer, `decision_rounds.py:354-366`) carry `median_entropy`, `modal_argmax_share`, `I`, `min_mass`
and no letter identity. Only `wave8_gates.py collapse-final` writes the histogram
(`argmax_histogram`, `top_letter_share` = letter Z, `zero_letter_share` = letter A).

So the next read is one gate pass over the six final adapters (round 3 where it exists, else round 2),
not the protocol eval. See §5.

## 5. Stop-marker semantics (needs a ruling)

`run_plan` (`decision_rounds.py:320-377`) writes `STOPPED_collapse.json` and returns 0 when the gate
trips at two consecutive reads (line 360), and writes `DONE` (line 376) only after the whole round loop
plus the validate stage. **It never reads `STOPPED_collapse.json`**, so a resume silently continues
past the stop. That is why `lr1e-5 seed0` carries both markers today: stopped after round 2, resumed,
round 3 clean, `DONE`.

Consequence: the §6.5 rung rule is not computable from the markers, because "tripped by end of round 2"
and "completed" both match that arm. Two consistent options:

* **A (recommended).** The marker is terminal: a resume needs an explicit `--override-stop` and writes
  `resumed_after_stop.json` alongside. The rung rule then reads markers.
* **B.** The marker stays advisory, and the rung rule reads `collapse_reads.jsonl` (count arms with two
  consecutive trips by end of round 2) instead of the markers.

## 6. Sequencing note

`gpu-wave8-eval.sbatch` runs the untrained control (3 modes) before any arm, then hard-fails on the
first missing `state.json`. Any eval run today therefore needs `W8_EVAL_SEEDS="0"`, and its
`lr_selection.json` must be recorded as "1 seed per lr, rung stopped as a whole" — not as a selection.

## 7. First S_nom read (job in flight)

`results/wave8/r1-lr1e-5-seed0-S_nom.json` (1024 states; the round reads in §1 are the 256-state
trainer subsample, so small differences are expected, same verdict):

* `collapsed=False`, reasons `[]` — medH 1.604 (round read 1.760), share 0.327 (0.391), I 0.662.
* `min_mass = 3.17e-04` → G7 fail, independently of the trainer read.
* `zero_letter_share` (letter A, order 0) = **0.000**; `top_letter_share` (letter Z) = **0.000** → no
  edge pile-up at either end, so the §4.3 R3 trigger is not met by this arm.
* Argmax histogram concentrated on three middle letters: R (44 units) 0.327, V (94) 0.276, D (6)
  0.194 — 0.797 of all states on three options, and the cheapest option A (0 units) is never chosen.
* Permutation probe `invariance_rate = 0.000`, `pass=False` → the gate table's "fixed-ordinal-scale
  specialist": the argmax value changes under a shuffled grid, so the arm is not reading the offered
  values.

Combined reading: a fixed small set of middle letters, indifferent to the option values offered
(0.000 invariance) and to the cheap end (A never chosen). That is the §6.6 row "R1 fails, collapse
clean, perm fail → value-reading not learned in budget", whose action is R2 with `--permute-p 1.0` on
seed 0 only. The five stopped arms still need their reads: they tripped, and a tripped arm piling on
A or Z would be an edge signature from a different mechanism.

Incidental bracket on the frozen entropy bar, from the reads so far: `H_lo ∈ (1.059, 1.604]` nats
(lr5e-5 seed2 round 2 tripped at 1.059; lr1e-5 seed0 does not trip at 1.604).

## 8. Band record and the paired comparison (both login-node, CPU only)

`select-lr` on `results/wave8_val` (2 files, seed 0 each): winner **lr5e-5**, pooled means
`lr1e-5 = 90,448.1`, `lr5e-5 = 16,035.3`, `seeds_per_lr` = single seed 0 for both. Written to
`$B/w8-band-record.json`, deliberately not `results/wave8/lr_selection.json`.

Paired over the 30 shared runs (same states, same run seeds):

* **lr5e-5 cheaper in 30/30**, median per-run ratio 5.602, one-sided sign test **p = 9.3e-10**,
  ratio range 3.80–10.46.
* So the arms differ decisively. But with one seed per lr this is an **arm** comparison: the lr
  effect and the seed effect are the same object until the stopped arms have costs. It is not yet
  evidence about the learning rate.

Per-run structure (realized cost across 30 different demand paths):

| arm | mean | sd | CV | min | max | spread |
|---|---|---|---|---|---|---|
| lr1e-5 seed0 | 90,448 | 3,075 | 0.034 | 85,597 | 95,689 | 1.12× |
| lr5e-5 seed0 | 16,035 | 5,083 | 0.317 | 8,812 | 23,726 | 2.69× |

lr1e-5 seed0's realized cost moves by 12 % across 30 different demand realizations. A policy whose
cost is nearly independent of the demand path is not responding to the state — that is the cost-side
fingerprint of the state-blindness the readout gates already report (low `I`, round-1 trip on 3/3
lr1e-5 seeds), from an independent measurement. lr5e-5 seed0 looks bimodal (20 runs < 18k, 10 runs
≥ 18k), worth a look when more seeds exist.

No interpretable absolute level yet: the band runs `--pattern curriculum`, the eval's control runs
`step`/`noisy`, so there is **no untrained counterpart** for 90,448 or 16,035. One control run with
`--pattern curriculum --mode argmax --runs 30 --seed-base 9500` (no adapter) supplies it.

## 9. All six S_nom reads + rung verdict

Reads of `results/wave8/r1-<lr>-seed<S>-S_nom.json` (1024 states each), final adapter (round 3 where
it exists, else round 2 — all resolved via `state.json.checkpoint_dir`, per the P0 table of job
406209):

| arm (adapter) | collapsed | medH | share | I | min_mass | G7 | modal | top-3 coverage |
|---|---|---|---|---|---|---|---|---|
| lr1e-5 s0 (r3) | **no** | 1.604 | 0.327 | 0.662 | 3.2e-04 | fail | R (44) | 0.80 |
| lr1e-5 s1 (r2) | yes | 1.114 | 0.522 | 0.167 | 0.929 | pass | T (64) | 0.99 |
| lr1e-5 s2 (r2) | yes | 0.753 | 0.761 | 0.422 | 0.054 | fail | T (64) | 0.96 |
| lr5e-5 s0 (r3) | yes | 1.066 | 0.727 | 0.635 | 0.981 | pass | E (8) | 0.87 |
| lr5e-5 s1 (r2) | yes | 0.851 | 0.312 | 1.016 | 0.996 | pass | V (94) | 0.83 |
| lr5e-5 s2 (r2) | yes | 0.697 | 0.186 | 1.804 | 1.0e-03 | fail | I (16) | 0.43 |

Facts that decide the rung:

* **A = 0.000 and Z = 0.000 on all six arms.** Across 6,144 state reads (6 × 1024) the cheapest
  option (order 0, value 0) and the top option (value 200) are never argmax. The §4.3 edge trigger
  (Z > 5 %, or A > 2× the oracle's 0.127) is not met by any arm — **R3 is off the table by its own
  definition**, now as a six-arm statement.
* **No SUPPORTING candidate.** 5/6 are collapsed at the final read; the one clean arm (lr1e-5 s0)
  fails G7 (`min_mass` 3.2e-04 — the readout leaves the label set) and has perm 0.000.
* **The lock is per-seed, not per-model.** Modal letters differ across seeds (R/T/T for lr1e-5,
  E/V/I for lr5e-5), all mid-grid values 8–94 with top-3 coverage 0.43–0.99. Not "the model prefers
  one letter" — each run converges to its own small mid-grid set.
* Cost and collapse disagree (from §8): the only collapse-clean arm is the 5.6× more expensive one;
  the collapsed arm (lr5e-5 s0) is the cheap one and holds label mass (`min_mass` 0.981).

Verdict per §6.6: **row 3 — collapse tripped mid-letter → letter-lock = objective/lr, not grid →
run R2 (SFT warm start, ~14.5 GPU-h); if R2+RL also locks → INCONCLUSIVE-negative at 0.6B and the
ladder.** R3 (§4.3) does not apply.

Two things remain open: (1) the marker ruling (terminal vs advisory — a resume past a stop needs
both arms of this decision), and (2) whether to spend ~1.7 GPU-h completing R1 (resume the four
stopped arms to round 3 + validate + the untrained curriculum-band control) so G1 is decided on 3
seeds before R2's 14.5 GPU-h goes anywhere. Per-arm permutation probes for arms 2–6 are on disk
(`results/eval/w8-r1-*-perm.json`) for the report.

## 10. External-reference audit (Unsloth "decision model" guide, 2026-10-08)

Reference: unsloth.ai — train a Clef/Jev-class decision model (0.8B in ~42 min to 73-81% on
classification benchmarks): **pure SFT, no RL** — LoRA r=16, lr 2e-4 (head lr 1e-4), 1-2 full
epochs, CE vs gold answers, then post-hoc calibration (temperature scaling, ECE). Head reads each
option's hidden state and scores it; the model never writes text.

Audit vs this kit — three real gaps, none of which overturn the §9 verdict:

* **A full supervised epoch has never run at decision scale.** R1 = RL rounds from a fresh LoRA
  (§4.1: "R1 has no SFT stage"). The only SFT ever executed was wave-7: 120 of 2,827 steps/epoch
  (4.2 %), lr 1e-5, final train_top1 0.22–0.25 — the spec's own §1 row 6 calls that
  under-trained. So "R1 locks at 0.6B" is confounded with "RL from cold start at 0.6B"; the
  guide's result is evidence about the method (SFT stabilizes), not about our cost ceiling.
* **The readout is the zero-weight token-logit readout; a per-option head was never tried.**
  `decision_readout.py` reads the next-token distribution over label token ids; the skill files
  the Clef-style head (score each option from its own hidden state) as "optional". Perm invariance
  0.000 (readout ignores the offered values) is exactly what a per-option head is built to avoid —
  but R2-SFT's `p_perm = 0.5` (half canonical, half shuffled grids) may fix the shortcut without
  new weights. Defer the head; it is confound-fix #2 behind SFT evidence.
* **lr regime and calibration note.** Our SFT lrs are RL-tuned (1e-5) or top out at 1e-4; the
  guide's working SFT point is 2e-4 at r=16, 2 epochs. Post-hoc scaling was tested on
  under-trained/RLC heads and hurt; re-test on a fully-trained SFT adapter before keeping that
  ruling. Data side we are ahead: soft targets over the cost vector (`p*(a) ∝ exp(-J/tau)`) are
  richer than their one-hot golds.

Does not change: R3 stays dead (edge signature absent by measurement — the guide has no choice-set
geometry content), and raw-accuracy numbers do not transfer (their golds are near-unambiguous;
our oracle modal share is 0.127, so even a perfect policy must spread mass).

Sharpened R2 plan (supersedes §9's open-item framing):
1. Run **R2-SFT first**: one full epoch, lr sweep extended to {1e-5, 1e-4, 2e-4} (+~1 GPU-h),
   3 seeds; read collapse-final (modals + perm + min_mass) + band paired vs the untrained
   curriculum control before scheduling R2+RL.
2. Only if R2-SFT gates trip does R2+RL (3 × ~0.7 GPU-h) fire. The rung's verdict is then
   attributable, not blended.
3. Retest temperature scaling post-hoc on the fullest SFT adapter before shipping the "scaling
   hurts" rule.

