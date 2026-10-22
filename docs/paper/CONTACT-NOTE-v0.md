# CONTACT NOTE - draft for the originals (Long, Simchi-Levi et al.)

Subject line option: "Independent pre-registered replication of your Beer-Game agent
post-training claim: pass at 4B, negative at 8B"

---

Hi Carol, hi Huangyuan,

I am Pepijn Frenken, an industrial engineering student at TU Eindhoven. I run an
independent replication program on LLM decision agents (everything pre-registered,
public audit trail). Your paper (arXiv:2605.17036) is the reference pole for our
training ladder, and I want to flag a result before it goes into a preprint, because
you are the cleanest check on whether we are misreading your protocol.

We ran your GRPO-post-training recipe, adapted to our harness (our deltas: exact
expected-cost gradient instead of sampled-K; frozen-anchor KL discipline; a frozen
closed-choice letter grid for the readout), as a pre-registered scale ladder of the
same recipe: 0.6B, 1.7B, 4B, 8B on a four-tier Beer-Game environment. Our verdict rule
picks the lr arm by pooled validation mean over the full pre-registered seed grid,
never post hoc. Result: at 4B our adapted recipe passes cleanly (two seeds, ~1.4-1.6x
our walk-forward floor), at 8B under the identical protocol the best arm trains to a
pooled cost 3.0x the trained-4B reference and slightly worse than the untrained-8B
baseline - with a 2.8x seed spread INSIDE the winning lr arm, which only the
pre-registered pooling caught. 0.6B and 1.7B are negative (collapse-detection or
pooled cost). So the tuned-model claim appears rung-dependent: it holds at 4B in our
harness and fails at 8B, and the one place it fails beautiful is exactly where the
verdict rule refuses cherry-picking.

Two asks: (1) a sanity check that our protocol reading is faithful to yours - the
delta table is short (schema, environment constants, gradient form, anchor) and I
would send it ahead of the preprint; (2) if you have already run a larger scale than
v4 reports, we would compare notes before we both claim scale-robustness in either
direction. Full artifact package (specs, per-run JSON, sbatch ledger, commit SHAs)
available on request.

Best regards,
Pepijn Frenken
TU Eindhoven
[git handle / repo link]
