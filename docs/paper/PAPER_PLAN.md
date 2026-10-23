# PAPER PLAN — Post-tuned LLM decision agents on a simulated bullwhip game
(not the final title; working title for the repo)

## Reference pole
Original claim we test: Long, Simchi-Levi, Zhu, Su, Calmon & Calmon,
"Reliability and Effectiveness of Autonomous AI Agents in Supply Chain Management",
arXiv:2605.17036 (v4 2026-09-21, v1 2026-05-16) - GRPO post-training of a 4B LLM on the
four-tier beer-game-style decision environment, claiming trained > untrained on cost.
Our WAVE8_SPEC 0 documents the borrowed recipe shape and where ours differs (schema,
sim constants, exact-expected-cost gradient vs sampled-K, frozen-anchor discipline).

## Story spine (holds for either outcome)
1. Claim ladder, not one point: 0.6B negative -> 1.7B negative -> 4B PASS (2/2 seeds,
   1.42-1.56x of floor / 1.53-1.81x of untuned formula / 1.66-1.81x of oracle) -> 8B documented
   (5e-5 arm negative 35,289; 1e-5 arm + seed variance pending on A100).
2. Methodology as contribution: pre-registered gates (F0-F3), collapse stopping rule,
   permutation-invariance gate, floor/untuned/oracle triple baseline, frozen-anchor KL
   discipline, paired-path comparison (trained vs own untrained, never cross-sim absolute).
3. Confound of the rung: same repo, same recipe shape, hardware tier differs (L4 vs A100)
   and lr arm selection interact - "scale alone is not a knob you can turn after tuning"
   (either direction) is the paper's honest kernel.
4. Audit trail: every run = sbatch id + committed kit sha + collapse-check json + val jsonl,
   all reproducible from the repo @ 88c31c9.

## Sections
A. Intro: post-training LLMs for decision tasks, why claim verification matters vs vibes.
B. Environment: engine + grid g26b, curriculum, label/policy separation, no-leak tests.
C. Gates: F0 synthetic-regret, F1 tokenizer, F2 label ids, F3 collapse
   (m*=0.127 at 8B; 4B m* similar family), permutation gate.
D. Results: rung table with G1 verdict per rung; 8B lr-arm table; untrained readouts.
E. Analysis: effect size vs floor tiers (strong <=4x, headline <=2x, parity <=1.25x);
   what "learning" means when perm gate never passes (0.010-0.047 vs 0.90):
   fixed-ordinal-scale specialist, not general invariance.
F. Related: the reference pole + RLCD/GRPO line + decision-native readouts.
G. Limitations: simulator, 2 seeds cap on 4B, single hardware family per rung, one oracle.
H. Artifacts: repo, gates, ledger.

## Publication targets (write-ready, after 8B lands)
- arXiv preprint (primary)
- TU/e research page / thesis-lite angle (IE department fits bullwhip studies)
- X/LinkedIn thread from the paper's figures (process-as-content works as promotion too)
