# POSTS.md — social drafts (unpublished)

Voice: first person (Pino). House style: honest process-as-content — the mishaps and audits are part of the story, not hidden. Nothing here is published until it says so. Framing rationale + the audit of which headline works: `WAVE5.md` §0.

---

## Wave 5 — "same model, opposite verdicts" (draft, 2026-09-17)

### Thread version

**1/**
I re-ran my Beer Game agent experiment with a decision-native model instead of an LLM — typed questions in, typed answers + probabilities out, no text generation anywhere.

On noisy demand it beat the tuned 1970s formula by **18.2%** (paired, 10/10 paths).
On flat demand the same model **lost by 22%**.

Same model. Same game. Opposite verdicts. That's the interesting part.

**2/**
Backstory: my earlier arms showed a cheap LLM is an instability machine in this game (parse failures, silent fallbacks, twin configs differing 10–20×) and that a self-tuning order-up-to formula beats it in every environment. The open question was: is that because LLMs are clumsy at supply chains — or because models can't beat the formula at all?

So I removed the text interface entirely. Jev (TypeSafe "System One"): the model picks a quantity — or a coverage multiplier, or a state read — from typed options, and my code does all the arithmetic. No generation → no parse failures. Probabilities → a confidence gate that actually fires (the LLM's gate fired on 0.2% of decisions; this one fires on 40–85%).

**3/**
The setup: 36-week, 4-echelon beer game (holding $1, backlog $2 per unit-week). 10 paired demand paths per arm. The walk-forward-tuned floor was recomputed on the *identical* demand paths — the project's first properly paired comparison, instead of the usual difference-of-means-across-different-paths.

noisy demand: formula **5,214** → Jev **4,263** = **−18.2%**, better on 10/10 paths
fixed demand: formula **3,206** → Jev **3,908** = **+22%**, worse on 10/10
chaotic demand: formula **7,677** → Jev **9,587** = +25% — it *loses* to the tuned formula while still beating the untuned policy by 24%
wild demand: formula **12,111** → Jev **13,710** = +13% — loses to the tuned formula, beats the untuned one by 15%, on 10/10 paths

Which tells you what the real pattern is. The model's correction is consistent every time demand is stochastic — 15–24% better than the untuned policy on all three stochastic arms. What varies wildly is how much *tuning* buys: 5.6% on noisy, 39.3% on chaotic. So the model wins exactly where tuning has little to buy. It's not a better forecaster — it's a hindsight-free, per-decision partial substitute for tuning.

**4/**
Before believing the noisy win I attacked it — five controls, all free and deterministic (zero API calls):

• random ±6 jitter on the same anchor: **+1,659** (worse than the formula)
• best constant offset (−3): **+912** behind the model
• the model's own deltas replayed from a *different* pass — same distribution, wrong state: **+2,230** (and the aligned replay reproduces the model's game to the unit)

So the win isn't the interface, the clamp, the magnitude, or the distribution. It's the state-conditioned **timing** of the adjustments.

**5/**
The twist: the confidence gate defers **85%** of decisions on the noisy arm (median confidence 0.21) and 43% on the flat arm (0.56).

It switches itself off exactly where its judgment is worth the most. The gate knows when it doesn't know — but *doesn't know* ≠ *is wrong*. Gate on confidence and you harvest your model's uncertainty, not its value.

**6/**
Two of my own claims died on the way there, both in the repo's audit trail:
• "the gated config beats the formula on fixed demand" — one decision deep; a deterministic cascade, not an effect. Retracted.
• "the mode with the best state-level agreement with the policy is the best config" — that mode is the worst in every actual game (+83% and +43%). State-level metrics lie.

**7/**
Open: the chaotic/wild arms are still running, plus an n=20 extension, a ±3/±6/±12 band sweep, and the mechanism question — *why* is it right under noise and wrong under flat demand?

Everything, mistakes included: [repo link]

The formula still wins under determinism. But a bounded model that a formula can interrupt seems worth keeping.

### Single-post version

I gave a decision-native model (typed Q&A, no text generation) the Beer Game my LLM agents lost. Under noisy demand it beat the walk-forward-tuned 1970s formula by **18.2%** (paired, 10/10 paths). Under flat demand it lost by **22%**. Same model, same game, opposite verdicts.

On chaotic demand it lost to the tuned formula too (+25%) — while beating the untuned policy by 24%. Wild demand, same story (+13% vs the tuned formula, −15% vs the untuned one, 10/10 paths). The real pattern: the model's correction is consistent whenever demand is stochastic; what varies is how much *tuning* buys (5.6% noisy vs 39.3% chaotic). It wins where tuning is weak — a hindsight-free partial substitute for tuning, not a better forecaster.

I attacked the win with five controls before believing it — random ±6 jitter is +1,659 worse, a constant offset is +912 behind, and replaying the model's own deltas misaligned is +2,230. It's information, in the timing.

The twist: the model's confidence gate defers 85% of decisions on the arm it wins — it switches off exactly where it earns its keep. *Unsure ≠ wrong.* Gate on confidence and you harvest uncertainty, not value.

Repo + full audit trail: [repo link]

### One-liner

A model's value is conditional on the world's uncertainty — and gating on confidence hides exactly the value you built.

---

## Earlier waves

Not collected here (the story lives in `README.md` + `AUDIT.md` / `AUDIT2.md` / `AUDIT3.md`). If the earlier findings get their own posts, they go at the bottom of this file.
