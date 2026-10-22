# DRAFT - Post-tuned LLM decision agents on a simulated bullwhip game
Status: skeleton draft v0.1 (2026-10-22). Working title, not final.
Scope decision to make before sending: this paper covers the local-model training ladder
(0.6B / 1.7B / 4B / 8B). Waves 1-5 (the formula-beats-agent result) are referenced, not
re-litigated, unless we expand scope (decision below).

---

## 0. Correspondence plan (PINO, this drives section polish)

**Who to contact:** the original authors are Long, Simchi-Levi, Zhu, Su, du Pin Calmon &
Calmon (Harvard / MIT / Kempner Institute / Georgia Tech). Suggested first contacts:
Carol Xuan Long (Harvard, first author, most likely to answer) and Huangyuan Su
(Harvard/Kempner, next-gen author). David Simchi-Levi is somebody you reach after the
first reply lands, not before. Find emails on the arXiv landing page or their lab pages,
never guessed - if an address is not public, use the arXiv correspondence link.

**What we send before any paper is live:** a 3-paragraph note, not the draft.
P1: who we are + one-line result (full pre-registered sweep; recipe reproduces at 4B,
goes negative at 8B under the same protocol; verdict rule did the work).
P2: the one figure/table that matters (rung ladder: cost vs floor at 0.6B/1.7B/4B/8B).
P3: ask - are we reading your protocol as intended; could we get 30 min to walk the
delta table (schema, sim constants, exact-expected-cost gradient vs sampled-K,
frozen-anchor discipline)? Offer full artifact package.

**Where the draft can overreach and hurt the contact:**
- Never claim "we refuted" the paper. Our claim is protocol-level: "under OUR pre-registered
  recipe adaptation, trained-8B does not beat untrained-8B." Their v4 (2026-09-21) may
  already differ from v1. Check their v4 changelog BEFORE the note goes out.
- The tuned-4B PASS is on OUR artifact stack. Their recipe used sampled-K cost gradient;
  we use exact expected cost. That is OUR delta, realize it in the note.

---

## 1. Introduction (skeleton - claim ladder)

Post-training LLM agents for operational decision problems is the paper-era default fix for
agent unreliability [REF 2605.17036]. The open question is not whether it can work once; it
is whether the result survives pre-registered controls and scaling.

We ran a claim ladder on a public Beer-Game-style decision environment. Every rung used
the same pre-registered protocol: F0 synthetic-regret gate, F1 tokenizer check, F2 label
integrity, F3 collapse stopping rule, permutation-invariance gate, triple baseline
(walk-forward floor / untuned formula / oracle).

Headline claim: "Trained > untrained at 4B, but trained-8B under the same recipe is
3.0x worse than our 4B result and worse than its own untrained baseline. The tuned-model
claim is rung-dependent, not scale-free."

The negative rungs are the point. A tuned 0.6B agent made things worse across all six
training arms (WAVE7). A tuned 1.7B closed negative. A tuned 4B passed cleanly (2/2).
A tuned 8B failed with the SAME pre-registered verdict rule that had two prior negatives
and one prior pass - the rule, not our judgment call, carried the verdict each time.

[NOTE: insert the exact 4B G1 PASS numbers + floor-ratio tier before circulating]

---

## 2. Environment (condensed from B in PAPER_PLAN; see hpc-umbrella/ OPERATOR.md)

- Engine: four-echelon Beer-Game-sim, grid g26b (frozen letter-grid 36..200, revised after
  F0 failure on g26, see WAVE8_SPEC 3.1 amendment log - REPRODUCIBLE, both grids committed).
- Labels: label/policy separation, prompt-sha-equal test (F2), no-leak audit trails.
- Cost readout: exact expected cost (not sampled-K as in the original paper).

Open question for the referee-check: our sim constants differ from the original paper
(schema-level {schema, sim constants, gradient}); we document the exact delta table in
Appendix A - the note and the paper BOTH need this table.

---

## 3. Gates + verdict protocol (C in PAPER_PLAN; this section is the contribution)

Protocol = commit THEN run. Every rung had its gates frozen before compute started, and
the spec documents every amendment with evidence (never silently).

Core gates:
- F0 synthetic-regret: can the model's readout express enough ordinal signal to be worth
  training? (g26 grid failed this at 1,000 states; g26b revised BEFORE training.)
- F1 tokenizer: teacher = student encoding match.
- F2 label integrity: no prompt leakage between label and policy data.
- F3 collapse: rolling mode-collapse / letter-lock detector; halts training.
- Permutation invariance: never passed (0.010-0.047 vs 0.90 threshold) - documented as a
  NEGATIVE finding: the models are fixed-ordinal-grid specialists, not invariant parsers.
- Verdict rule: pooled validation mean over the pre-registered lr arm x seed grid selects
  the arm. Never a post-hoc pick. The 8B seed-variance result exists BECAUSE the rule
  refused to cherry-pick a lucky seed (35,289 vs 12,444 at the same lr = 2.8x spread
  pre-registered as a finding, not a bug).

---

## 4. Results - the rung ladder

| rung | recipe | val cost (pooled) | vs own floor tier | verdict |
|---|---|---|---|---|
| 0.6B | soft-target SFT + RLCD, 3 seeds x 6 arms | C1 fail 0-12/30 on all arms, perm gate 0.000, letter-lock | - | NEG training |
| 0.6B | R1 (closed-choice) | closed-choice killed rogue tail 0 rogue vs 7-188 wave6; stable untrained control 11,569 | - | interface-positive, training-null |
| 1.7B | R1 recipe | 5e-5 arm: STOPPED_collapse (collapsed=true, medH 6.3e-05, I 2.1e-06); 1e-5 arm pooled 48,311 (47,125 / 36,734 / 61,073) | worse than untrained-4B (21,578) | NEG (collapse + cost) |
| 4B | R1 recipe, 2 seeds | 8,315 / 7,612 (pooled 7,964) | 1.42-1.56x floor | PASS (G1) |
| 8B | R1 recipe, full 2x2 (2 lr x 2 seeds) | best-arm pooled 23,867 (5e-5) | 3.0x trained-4B, worse than untrained-8B (21,578) | NEG |

Untrained reference line: 0.6B 11,569; 4B 21,578; 8B 21,578 (same untrained-4B checkpoint
reference used by the 8B comparison).

[1.7B VERIFIED 2026-10-22 from cluster artifacts:
lr 5e-5 arm stopped by the collapse gate (STOPPED_collapse.json, collapsed=true,
median_entropy 6.3e-05); lr 1e-5 arm completed 3 seeds but pooled 48,311 mean cost -
worse than the untrained-4B baseline 21,578. 1.7B rung = NEGATIVE, two failure modes:
distribution collapse at 5e-5, high cost without collapse at 1e-5. Note the 1e-5 seed1
min_mass 5e-9 line still needs a stability read before shipping.]

The inversion: cost does NOT monotonically improve with scale along the recipe. Between
4B and 8B the tuned advantage disappeared entirely (compressed by seed variance at the
same lr arm: 12,444 best seed vs 35,289 worst seed). This is the kernel finding: the
original paper's 4B claim did not pre-register cross-scale robustness, and our sweep shows
the tuned advantage is not a scale-stable object.

[TO VERIFY before release: exact 1.7B rung verdict numbers - currently only the memory
line '1.7B closed negative'. Pull the job readout into the repo as VERIFY-1.7B doc.]
[GAP: the whole 4B floor-ratio sentence needs the floor.json artifact re-cited line by line.]

---

## 5. Analysis - what survives contact with the gates

1. Interface >> training at small scale: the biggest win in our work was the READOUT, not
   the gradient. Closed-choice letter-logit readout at 0.6B eliminated 100% of rogue orders
   (0 vs 7-188 per free-form arm). That gain did not need training to land.
2. The permutation gate never passed at any rung. Whatever the trained model is doing,
   it is not learning an order-scale-invariant policy. Training sharpened a fixed ordinal
   grid (fixed-ordinal-scale specialist hypothesis), which caps cross-grid transfer.
3. Seed variance grows with model size (0.6B tight, 4B tighter, 8B armd = 2.8x spread).
   Pre-registered pooling was the ONLY thing that made the 8B negative survivable
   without an argument. Post-hoc seed-picking would have produced a positive 8B claim
   (12,444 beats trained-4B). This is the paper's methodological punchline and the single
   most-transferable finding for anyone replicating tuned-agent claims.
4. Hardware confound honesty: 0.6B/4B ran on TU/e Umbrella (L4); 8B arms split between
   Snellius A100 (seed 0) and Modal A100-80GB (seed 1). We state tier per rung in the
   ledger. Same hardware family per rung kept; cross-rung hardware is not held identical,
   which limitation G6 covers explicitly.

---

## 6. RELATED work (draft list; add DOIs before send)
- Long, Simchi-Levi, Zhu, Su, Calmon & Calmon, arXiv:2605.17036 (the reference pole).
- RLCD / representation-free RL distillation line - [fill 2 refs].
- Decision-native readouts (Clef-style fixed-choice readouts) - [fill 2 refs].
- Sim scale-ladder discipline: [fill 1-2 refs on scaling-locality in RL].

---

## 7. Limitations (be blunt - every one of these is already known to the contact)
1. Simulated environment; external validity is exactly the original paper's problem too.
2. 2 seeds on 4B and 8B; 3 seeds on 0.6B. The rung verdict is conditional on the
   pre-registered seed count not more.
3. One oracle.
4. Our recipe adaptation uses an exact-expected-cost gradient, theirs uses sampled-K.
   Our 4B PASS and 8B NEGATIVE are both inside "our" recipe family - not the original
   recipe byte-for-byte.
5. Cross-rung hardware is not identical.

---

## 8. Artifacts page
- repo @ 88c31c9 + the kit SHA per rung (audit trail)
- W8 final doc: docs/waves/WAVE8_WORD8B_RESULT.md
- WAVE8_SPEC amendments log
- [INSERT HF/Codalab artifact release decision]

---

## 9. TODO BEFORE SEND (blocking)
1. Pull the 1.7B rung readout from results/ into a VERIFY-1.7B doc (the draft currently
   references it from memory only). [DO NOT SHIP TO AUTHORS WITHOUT THIS]
2. Prepare the 4B floor-ratio trace (results/*/floor.json) so section D is cite-able.
3. Delta table Appendix A (OURS vs ORIGINAL: schema, constants, gradient, anchor).
4. Pull the ORIGINAL paper's v4 (2026-09-21) changelog and re-check no claim here was
   already updated by them (esp. "should stay tuned at larger scale" if they cache it).
5. Write the 3-paragraph contact note (see section 0) and freeze the rung-figure PNG
   for it. Send that, not the PDF.
6. Pick ONE figure to lead: the rung ladder accuracy/cost plot (NOT the training curves).
