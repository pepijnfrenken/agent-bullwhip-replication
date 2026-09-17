# POSTS.md — social drafts (unpublished)

Voice: first person (Pino). House style: honest process-as-content — the mishaps and audits are part of the story, not hidden. Nothing here is published until it says so. Framing rationale + the audit of which headline works: `WAVE5.md` §0.

---

## Wave 5 — "same model, opposite verdicts" (draft, 2026-09-17)

### Thread version

**1/**
I re-ran my Beer Game agent experiment with a decision-native model instead of an LLM — typed questions in, typed answers + probabilities out, no text generation anywhere.

On noisy demand it beat the tuned 1970s formula by **7.1%** (paired, 20 paths, 15/20, t=−2.15).
The first read was **18.2%** (10/10 paths) — the n=20 re-run shrank it, and it still holds. That's the honest number.
On flat demand the same model **lost by 22%**.

Same model. Same game. Opposite verdicts. That's the interesting part.

**2/**
Backstory: my earlier arms showed a cheap LLM is an instability machine in this game (parse failures, silent fallbacks, twin configs differing 10–20×) and that a self-tuning order-up-to formula beats it in every environment. The open question was: is that because LLMs are clumsy at supply chains — or because models can't beat the formula at all?

So I removed the text interface entirely. Jev (TypeSafe "System One"): the model picks a quantity — or a coverage multiplier, or a state read — from typed options, and my code does all the arithmetic. No generation → no parse failures. Probabilities → a confidence gate that actually fires (the LLM's gate fired on 0.2% of decisions; this one fires on 40–85%).

**3/**
The setup: 36-week, 4-echelon beer game (holding $1, backlog $2 per unit-week). 20 paired demand paths on the noisy arm (10 on the others). The walk-forward-tuned floor was recomputed on the *identical* demand paths — the project's first properly paired comparison, instead of the usual difference-of-means-across-different-paths.

noisy demand (n=20): formula **4,773** → Jev **4,437** = **−7.1%**, better on 15/20 paths (t=−2.15)
  (n=10 first read: 5,214 → 4,263 = −18.2%, 10/10, t=−3.89 — same direction, smaller magnitude, wider sample)
  band sweep: ±3 → **+1.0%** (8/20, ns: tightening the clamp kills the win) · ±12 → **+18.7%** — so the default ±6 is the working point
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
Closed since the first draft: the chaotic and wild arms are complete (all four arms, 240 runs, 0 failures) and the n=20 noisy extension + ±3/±12 band sweep finished 2026-09-17 (`results/jev_noisy20/`).
Still open: the mechanism — *why* is it right under noise and wrong under flat demand? — and a second decision-native model for generality.

Everything, mistakes included: [repo link]

The formula still wins under determinism. But a bounded model that a formula can interrupt seems worth keeping.

### Numbers (the whole thing, no charts)

```
NOISY DEMAND — 20 paired paths (lower cost = better)
  untuned policy (anchor only)   5,488      +15.0% vs tuned
  Jev (model)                    4,437       -7.1% vs tuned   <- wins on 15/20 paths, t=-2.15
  tuned formula (walk-fwd floor) 4,773         -
  model with +-3 band            4,820       +1.0% vs tuned   <- over-tightening kills it
  model with +-12 band           5,664      +18.7% vs tuned
  random +-6 jitter (control)    6,751      +41.4% vs tuned   <- wins on 1/20

  first read at 10 paths: 5,214 -> 4,263 = -18.2%, 10/10, t=-3.89
  the n=20 re-run shrank it to -7.1% (15/20). Direction holds, magnitude doesn't.

ALL FOUR ARMS — model vs the tuned formula (10 paths each, 240 runs, 0 failures)
  noisy (n=10)   -18.2%     (n=20: -7.1%)
  step          +6.2%      worse on 10/10
  chaotic      +24.9%      loses to the formula, beats the untuned policy by 24.3%
  wild         +13.2%      loses to the formula, beats the untuned policy by 15.2%

  vs the untuned policy the model is better on every stochastic arm: -22.8 / -24.3 / -15.2%.
  What varies is how much tuning buys (5.6% noisy vs 39.3% chaotic) -> it wins where tuning is weak.
```

| noisy n=20 (paired) | untuned | **Jev** | tuned floor | model vs floor | better on |
|---|---|---|---|---|---|
| cost (lower better) | 5,488 | **4,437** | 4,773 | **−7.1%** | **15/20** (t=−2.15) |
| model ±3 band | | 4,820 | 4,773 | +1.0% | 8/20 (ns) |
| model ±12 band | | 5,664 | 4,773 | +18.7% | 3/20 |
| random ±6 jitter | | 6,751 | 4,773 | +41.4% | 1/20 |

| arm (10 paths) | Jev vs tuned floor | Jev vs untuned policy |
|---|---|---|
| noisy | −18.2% (10/10) | −22.8% |
| step | +6.2% (worse 10/10) | +6.2% (worse) |
| chaotic | +24.9% (3/10, ns) | −24.3% |
| wild | +13.2% (3/10, ns) | −15.2% |

### Single-post version

I gave a decision-native model (typed Q&A, no text generation) the Beer Game my LLM agents lost. Under noisy demand it beat the walk-forward-tuned 1970s formula by **7.1%** (paired, n=20; 15/20 paths). Under flat demand it lost by **22%**. Same model, same game, opposite verdicts.

On chaotic demand it lost to the tuned formula too (+25%) — while beating the untuned policy by 24%. Wild demand, same story (+13% vs the tuned formula, −15% vs the untuned one, 10/10 paths). The real pattern: the model's correction is consistent whenever demand is stochastic; what varies is how much *tuning* buys (5.6% noisy vs 39.3% chaotic). It wins where tuning is weak — a hindsight-free partial substitute for tuning, not a better forecaster.

I attacked the win with five controls before believing it — random ±6 jitter is +1,659 worse, a constant offset is +912 behind, and replaying the model's own deltas misaligned is +2,230. It's information, in the timing.

The twist: the model's confidence gate defers 85% of decisions on the arm it wins — it switches off exactly where it earns its keep. *Unsure ≠ wrong.* Gate on confidence and you harvest uncertainty, not value.

Repo + full audit trail: [repo link]

### One-liner

A model's value is conditional on the world's uncertainty — and gating on confidence hides exactly the value you built.

---

## Earlier waves

Not collected here (the story lives in `README.md` + `AUDIT.md` / `AUDIT2.md` / `AUDIT3.md`). If the earlier findings get their own posts, they go at the bottom of this file.
