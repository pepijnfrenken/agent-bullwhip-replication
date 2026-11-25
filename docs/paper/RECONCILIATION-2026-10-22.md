# Revision reconciliation, draft 3 to draft 4

Date: 22 October 2026. Companion to `paper.tex` (draft 4) and to the machine-readable
extract `docs/verify/W8-MANIFEST.json`.

Purpose: every fact that changed between the third and fourth supplied drafts is listed here
with the artifact it was re-derived from. No number below was taken from an earlier draft of
the paper, from memory, or from a chat summary. This note is the supplement the manuscript
refers to; it is deliberately not repeated in the paper body.

Legend: **old** = claim in draft 3; **new** = claim in draft 4; **source** = artifact and field
read this revision.

## 1. Wave-7 design description

- old: "six recipe arms" (six arm-seed rows described as six independent recipes).
- new: two recipe arms (soft-target SFT, exact-gradient RLCD) times three seeds each = six
  arm-seed rows.
- source: `results/eval/` contains `w7-sft-seed{0,1,2}-argmax-noisy.*` and
  `w7-rlcd-seed{0,1,2}-argmax-noisy.*`, six files per mode, two recipe prefixes.
- why it matters: draft 3's "33 planned tasks" total was a residue of the six-recipe reading.
  Correct total is 24 planned tasks (Table 1 row count).

## 2. 8B cost generation label

- old: 8B costs described as final-evaluation means.
- new: 8B costs 35,289 / 12,444 (lr 5e-5) and 71,855 / 22,083 (lr 1e-5) are
  curriculum-validation band means used for arm selection; no 8B final-eval generation exists.
- source: `docs/waves/WAVE8_WORD8B_RESULT.md` (column header "Scoreboard (val cost)", per-run
  label "Final val (30 runs, grid w8_g26b, argmax)"); absence of any `w8-r1-8b*` file under
  `results/eval/`; the only 8B file in `results/wave8_val/` is
  `w8-val-r1-8b-lr1e-5-seed0.summary.json` (mean 99,618.3), which is a *separate, earlier
  TU/e L4 validation run* on a different lane and is not one of the four selection legs.
- note: the four selection legs ran on Snellius (28414248, 28418615) and Modal
  (ap-mmFrW6iXGxGWou9pzdRuss, ap-aqowSMnlxmrP0tkBzeK2jv). Modal's archival step crashed on a
  ZIP timestamp bug (1980 epoch), so those two legs are document-level provenance only.

## 3. Backlog dynamics

- old: "bounded backorders".
- new: no backlog cap.
- source: `agent_bullwhip/engine.py` (backlog accumulates without a cap; cost is linear in
  unfilled orders at weight 2.0).

## 4. 4B permutation reads

- old: permutation rate 0.000 for every arm including 4B.
- new: seed 0 = 24/512 = 0.046875, seed 1 = 20/512 = 0.0390625, both `pass=false` against
  gate 0.90; untrained control and every wave-7 arm 0/512. Seed 0 per-shuffle disagreement
  counts vs canonical: [0, 464, 452, 449].
- source: `results/eval/w8-r1-4b-seed0-perm.json`, `results/eval/w8-r1-4b-seed1-perm.json`.
- wording consequence: "no arm preserved the value" is false and was replaced by "no tested
  arm met the 90% criterion".

## 5. Wave-8 0.6B stopping count

- old: "collapse tripped on six of six tasks; five stopped before final evaluation; one
  completed".
- new: all six tripped at round 1, the stop-loss persisted on five; two were later resumed
  past a stop (one accidental, one under written override), and the completed read fails G7
  at min_mass 3.2e-04.
- source: `results/wave8_logs/wave8-train_407006_*.log` and `wave8-train_406726_*.log` (16
  deduplicated collapse-check reads), `STOPPED_collapse.json` markers,
  `docs/waves/W8-R1-DRAINED.md` (stop table, accidental-resume defect, completion override).
- provenance warning: the six `results/wave8_r1_lr*` arm directories and their two band files
  were removed from the archive by the repository's own results slim-down. Every wave-8 0.6B
  number in the paper is therefore log-sourced, and the paper says so in the reproducibility
  section. Band record survives at `/scratch-shared/<cluster-account>/bullwhip/w8-band-record.json`
  (lr5e-5 16,035, lr1e-5 90,448, seed 0 each).

## 6. 1.7B seed-1 output mass

- old: 5e-9.
- new: 7.33e-09 in the final noisy-argmax generation; 7.85e-09 in the 1,024-state final read;
  8.93e-09 in the 256-state round read. All binding failures.
- source: `results/eval/w8-r1-1.7b-seed1-argmax-noisy.summary.json` (`min_mass`),
  `results/wave8/r1-1.7b-seed1-S_nom.json`, round read JSON.

## 7. Thresholds now stated to gate-file precision

| Rung | H_lo | H_hi | I_hi0 | m_hi | source |
|---|---|---|---|---|---|
| 0.6B | 1.393 | 2.022 | 0.0970 | 0.149 | `wave8-preflight_404131.log` freeze line (no F3 block in `gates.json`) |
| 1.7B | 9.6527e-05 | 1.6313 | 1.4858 | 0.1487 | `results/wave8/gates_1.7b.json` |
| 4B | 0.038026 | 1.910513 | 0.676551 | 0.1487 | `results/wave8/gates_4b.json` |
| 8B | 0.000503 | 1.881214 | 1.033805 | 0.1487 | `results/wave8/gates_8b.json` |

Two corrections follow. First, the 1.7B H_lo printed as "0.000" in draft 3 was display
rounding of a small positive number; a literal zero would make the strict inequality
unsatisfiable. Second, the 0.6B cutoff is the largest of the four (1.393 nats), so 0.6B reads
trip on median entropy as well as modal share, which is why "modal share 0.51 to 1.00" alone
was an incomplete description of the evidence.

## 8. 4B collapse reads, per checkpoint

| Read | H | I | share | min_mass | collapsed |
|---|---|---|---|---|---|
| seed0 R1 | 2.0863 | 0.8714 | 0.1836 | 0.999996 | no |
| seed0 R2 | 1.5563 | 0.7063 | 0.4688 | 1.000000 | no |
| seed0 R3 | 1.7783 | 0.3459 | 0.7461 | 0.999994 | no |
| seed0 final S1024 | 1.7887 | 0.5287 | 0.6514 | 0.999983 | no |
| seed1 R1 | 0.5765 | 1.6662 | 0.1953 | 0.999813 | no |
| seed1 R2 | 1.7547 | 0.4540 | 0.4102 | 0.999996 | no |
| seed1 R3 | 1.7455 | 0.0739 | 0.8398 | 0.999996 | no |
| seed1 final S1024 | 1.7239 | 0.3719 | 0.7939 | 0.999996 | no |

Source (corrected 2026-10-23): `docs/verify/W8-MANIFEST.json` → `collapse_4b`. The per-run paths
originally cited here (`results/wave8_r1-4b_lr5e-5_seed{0,1}/round*/collapse_round{n}.json`,
`results/wave8/r1-4b-seed{0,1}-S_nom.json`) are **not present** in the working tree or in any
committed revision; the values in the table above are archived in the manifest and were re-read
from it, unchanged. Likewise `gates_1.7b.json` and `gates_4b.json` appear in no committed revision,
so the 1.7B and 4B F3 thresholds in §8 rest on this note's record only — the paper's supplement S1
states that limitation. No read is below H_lo (minimum 0.5765), and the
single read above H_hi (seed0 R1, 2.086) has I = 0.871 > I_hi0 = 0.677, so neither branch can
fire. High modal share alone is not collapse under the registered rule.

## 9. Ledger totals

| Class | Count | Rows |
|---|---|---|
| Planned | 24 | all rows of Table 1 |
| Attempted, trained checkpoint | 22 | 24 minus 2 unattempted 1.7B 5e-5 seeds |
| Matched final evaluation | 11 | 6 wave-7, 3 (1.7B 1e-5), 2 (4B) |
| No matched final evaluation | 13 | 6 wave-8 0.6B, 1 stopped 1.7B 5e-5, 2 unattempted, 4 8B |
| Persisted stop-loss marker | 6 | 5 wave-8 0.6B (lr1e-5 s0, s1, s2; lr5e-5 s1, s2) + 1 stopped 1.7B 5e-5 seed 0 |

Note on the last row: six distinct tasks carry a `STOPPED_collapse` marker. The 0.6B task
lr5e-5 seed 0 carries `DONE` only (its reads are C, not-collapsed, C, so it was never
stopped). Two of the six were subsequently resumed past the stop: lr1e-5 seed 0 by an
unattended bare-array resubmission, and lr1e-5 seed 1 under a written completion override. Draft 3's "33 planned / 21 trained / 8 without
evaluation" is retracted in full.

## 10. Selection-rule reading

`docs/waves/WAVE8_SPEC.md` section 6.3, verbatim: "A seed passes iff all three hold; the arm
needs >= 2/3 seeds." The text does not define how a planned seed that never reaches evaluation
enters the fraction. Draft 4 therefore prints both the evaluated-denominator and
planned-denominator readings and names five disjoint failure classes (unattempted, stopped,
trained-without-evaluation, evaluated-failing, evaluated-passing) instead of choosing one
denominator after seeing results. Both readings agree on 4B (2/2 pass), which is the only
positive rung.

## 11. G7 explanation

Draft 3 stated that "an individual letter probability cannot exceed 1/26"; that is false. One
label can carry probability near 1. What is bounded by 1/26 is the *smallest of the 26*
individual probabilities, and G7 does not bound that either: G7 bounds the *sum* of the 26
label probabilities, which ranges from near 0 to near 1. The parenthesis was deleted and the
correct statement substituted.

## Round 6, 2026-11-22: external audit of commit 156738a

A second-model audit (GPT-6.2-Astra, running against the committed tree) reproduced every
headline recomputation and then found six claims the artifacts do not support. Each was verified
against the file before it was changed. What moved, and the field it was re-derived from:

1. **October 4B paired statistic, withdrawn.** The abstract and Results said the October verdict
   recorded "30/30 wins per seed". `docs/waves/WAVE8_WORD8B_RESULT.md` contains the two costs and
   21,578 and no win count, no per-path reduction and no sign test. The 4B verdict is now carried
   by the November rebuild (9,346 and 17,958 against 21,578; 30/30 and 25/30; p 9.3e-10 and
   1.6e-4), and the October figures are reported as recorded with no paired claim attached.
   Source of the error: this assistant, in the previous round.
2. **1.7B cohort labels.** Medians 21,468 / 43,660 / 71,270 and "0/30 on every seed" carried no
   October qualifier, and supplement S3 tagged those rows "per-run JSON" for files that were never
   committed. They now read as recorded figures from the rebuild memo's published-figures column,
   with no per-run file and no manifest entry, and the recomputable November row (7/30, 0/30,
   0/30 at 2.41x, 6.87x, 8.25x) sits beside them.
3. **Appendix 8B sentence.** The provenance paragraph still read "two seeds, both passing G1"
   against the corrected Results. It now states the band-level reading and that those legs are not
   a G1 pass, with the reason (curriculum pattern against a noisy control).
4. **Evaluation policy, corrected.** The readout paragraph said the noisy-argmax arm samples the
   action from the policy. `results/eval/w8-r1-4b-seed0-argmax-noisy.jsonl` carries
   `readout.composition: argmax`, and `decision_eval.py` puts the noise in the demand draw. The
   paragraph now says 30 noisy demand paths with a deterministic modal-letter policy, and names the
   sampled-step arm as the separate stochastic generation.
5. **Ledger counts, recounted.** "11 with a matched paired final evaluation, 13 without" predates
   the November read of the October 8B checkpoint. With that row moved to evaluated, the 24 rows
   split 12 and 12, and the table's caption dates the count. Wave-8 R2 work (SFT and RL, job
   `418889`, evaluated but never committed) is named as outside the R2 ledger rather than left for
   a reader to find in a log.
6. **Freeze chronology, dated.** The 2026-10-07 spec blob contains no 8B rung. The 8B extension is
   the A2 amendment, commit `6152424` at 2026-10-21T08:17:07Z. The paper now says the freeze covers
   three rungs, that the 8B rung joins by dated amendment, and that the marker semantics are a
   ruling recorded after the first stops rather than a pre-registration.

Also changed in this round: "162 batches per round" is 162 per epoch and 170 optimizer steps per
round (`train_log.jsonl` `n_batches`); S7 names the real gate fields `median_entropy` and
`modal_argmax_share` instead of claiming nulls and quoting the console; the seed-spread sentence
prints the November pooled mean 13,151.53; the arXiv metadata abstract has resolved references and
the license recommendation no longer claims CC BY while `LICENSE` grants MIT.
