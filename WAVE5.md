# WAVE5.md — a decision-native model plays the Beer Game (post draft)

One place for everything wave 5: the story, the numbers, the audit trail, and the open items.
Companion: `AUDIT4.md` (the mistakes and the controls, in full). Data: `results/jev_protocol/`,
`results/jev_controls_*/`, `results/walkforward_paths.json`.

**Status:** step, noisy and chaotic arms complete (10 passes each, interleaved, seeded, paired). Wild running (partial) — **no claim here rests on a partial arm.**

---

## 0. Framing audit — which headline works best

### A. "The claim we retracted three audits ago returns — corrected" *(backstory, not headline)*
- **For:** it *is* the corrected form of the same shape (agent beats formula under noise); strong narrative pull; the project's audits give it credibility.
- **Against:** the old claim was about a *text LLM*; this is a different model class with a ±6 clamp — not the same treatment. Leading with vindication invites re-litigating the LLM arm, which the reader hasn't seen.
- **Verdict:** opening paragraph. Never the headline.

### B. "A bounded model correction is a hindsight-free substitute for tuning — and it wins exactly where tuning has nothing to buy" *(recommended headline)*
- **For:** across the three complete arms the model's bounded correction improves on the untuned policy by **18–24% on every stochastic arm** (noisy −22.8%, chaotic −24.3%) and **harms by +6.2% on flat demand** — while per-path tuning's own gain ranges from **5.6% (noisy) to 39.4% (chaotic)**. Hence the crossover: the model **beats the tuned floor on noisy** (−18.2%, 10/10 paths, t=−3.89) and **loses on chaotic** (+25%, 3/10, not significant) — not because the model changed, but because tuning has far more to buy on chaotic. Sharper moral: a bounded, state-conditioned correction recovers a large share of what a hindsight-free parameter search finds, per decision, with no search.
- **Against:** three arms, one model, n=10 paths, wild partial. A reviewer attacks: "it's a tuning substitute that loses to tuning — say that."
- **Verdict:** headline — with the loss in the same sentence as the win. "It wins where tuning is weak" is more defensible than "it wins under noise" and it survives the chaotic arm.

### C. "How to tell whether an agent adds information: jitter / offset / delta-replay" *(methods section, reusable)*
- **For:** the control suite is reusable and it's what makes B credible: uniform ±6 jitter (+1,659 on noisy), best constant offset (−3: +912 behind the model on noisy, +2,288 behind on chaotic), the model's own deltas misaligned (+2,230) — and the *aligned* replay reproduces the model's game **to the unit** (3,783 = 3,783), verifying the machinery. Rule of thumb: *a model's intervention is information only if the same intervention, randomised or misaligned, loses its value.*
- **Against:** methods posts have a smaller audience; needs B for stakes.
- **Verdict:** act 3 of the post, its own section in the repo writeup.

**Recommended structure:** open with A in three sentences → B as the thesis → prove B with the arm-crossing numbers → make B credible with C → close on the open mechanism. Keep the retraction list visible: that's the house style, and it's what makes the numbers believable.

---

## 1. Why this arm exists

The LLM arms failed for *executional* reasons, not intellectual ones: parse failures, silent mirror fallbacks (13.9% mirror-after-error in the tool arm), byte-identical twins differing 10-20×, and a confidence gate that fired on **0.2%** of decisions (3/1,440). "Gated LLM" was an empty treatment because there was nothing to gate.

If the failure is the interface (text in, text out, parse the text), the clean test removes the text: typed question in, typed answer + probability out. **Jev** (TypeSafe, "System One"): a decision-native model. No generation → no parse failures; probabilities → a confidence signal with resolution.

## 2. Setup

- **Modes:** `choice_grid` (pick a quantity), `choice_mult` (pick a coverage multiplier κ; order = max(0, κ·q̂ − IP), arithmetic in code), `expectation` (probability-weighted), `reads` (auxiliary state reads composed in code).
- **Structure:** structured JSON state payload; all arithmetic stays deterministic in code. The model supplies judgment, the formula supplies arithmetic — the project thesis run forward on a decision-native model.
- **Gate + clamp:** confidence < τ → order := the deterministic anchor; a ±6 margin clamp is always applied; any failure → anchor fallback.
- **Controls (free, deterministic — 0 API calls):** `anchor_only` (the wrapper alone), `gate_always`/`gate_never` (τ endpoints), `jitter_only` (seeded uniform ±6), `offset_only` (±3/±6), `delta_replay` (the model's own deltas, misaligned).

## 3. Results

### 3.1 Cross-arm table (means; arms are paired per path, 10 paths each)

| arm | untuned (wrapper) | Jev, bounded (model only) | Jev vs untuned | tuning's own gain (wf − untuned) | Jev vs walk-forward floor | verdict |
|---|---|---|---|---|---|---|
| step (fixed) | 3,681 | 3,908 | **+6.2%** | −475 (−12.9%) | +702 (0/10) | loses |
| noisy | 5,521 | **4,263** | **−22.8%** | −307 (−5.6%) | **−951 (10/10, t=−3.89)** | **wins** |
| chaotic | 12,659 | **9,587** | **−24.3%** (9/10) | −4,982 (−39.4%) | +1,910 (3/10, ns) | loses to tuning |
| wild | 16,159 | 13,268 *(partial 3/10)* | −17.9% *(partial)* | −4,048 *(partial)* | +4,957 (0/3) *(partial)* | pending |

Read it as: **the model's correction is consistent (≈−18–24% over the untuned policy on stochastic arms, +6% harm when demand is flat), while tuning's gain is arm-dependent. So the model wins exactly where tuning has little to buy.** Share of tuning's gain recovered by the model: noisy 410% (beats tuning), chaotic 62%, wild ~71% (partial).

### 3.2 Fixed demand (step, 36 weeks, n=10)

- tuned formula (θ=3.0, λ=0.35): **3,148**; walk-forward grid floor 3,206
- wrapper alone (`anchor_only`, 0 API calls): **3,681** (CV 0.000)
- `gate_always` (τ=1.0): **3,681** — gate 144/144, **0 violations**
- `jev_grid_gated` (τ=0.5): **3,549** — *retracted, one-decision-deep (AUDIT4 §1.1)*
- `gate_never` (τ=0.0): **3,908** (+227 vs wrapper) · jitter: 4,972 · `reads`: 6,731

### 3.3 The noisy arm — the win

- walk-forward floor **5,214** · untuned 5,521 · full-path oracle 4,544
- **`gate_never` 4,263 → −951 (−18.2%), 10/10 paths, t=−3.89**; −1,258 vs untuned (10/10); below the oracle (−281, 7/10, t=−1.51 — *not* significant: "below, not proven below")
- controls on the same paths: uniform ±6 jitter **6,873** (+1,659) · best constant offset (−3) **5,175** (+912 behind the model) · **delta-replay misaligned 6,493** (+2,230; aligned pass = exact replica 3,783 = 3,783)
- raw deltas: +6: 37.6%, −6: 22.6%, 0: 12.4% — the model orders *up* more than down; not a bias

### 3.4 The gate points the wrong way

| arm | deferral rate | confidence median | model value vs wrapper |
|---|---|---|---|
| step | **42.7%** | 0.56 | +227 (hurts) |
| noisy | **85.2%** | 0.21 | −1,258 (helps) |

The gate closes where the model's judgment is most valuable. The LLM arm's problem was a gate with nothing to gate; Jev's is a gate that works perfectly and points the wrong way. **Unsure is not the same as wrong.**

## 4. What the control suite establishes

Same ±6 budget, same anchor, on every stochastic arm the model beats every control:

- noisy: model 4,263 vs jitter 6,873 · offsets 5,175-6,900 · misaligned replay 6,493
- chaotic: model 9,587 vs jitter 12,552 · offsets 12,223-12,416
- wild (partial): model 13,268 vs jitter 16,028 · offsets 15,088-17,115

**The win is the state-conditioned timing of the adjustments — not the interface, the magnitude, the distribution, or the clamp.**

## 5. Pending (do not claim yet)

1. Wild arm (running, 3/10 passes).
2. Noisy extension to n=20 paths.
3. Band sweep ±3/±6/±12 (is the pattern band geometry? — partially answered: offsets of the same size don't reproduce it).
4. Mechanism: what correction does the model make that tuning finds and the ±6 band can't? Decision-by-decision characterisation pending.
5. A second decision-native model, for generality.

## 6. Compute

- Jev: ~974 prompt + 213 completion tokens per decision (no generation), ~0.4-2 s/call.
- The complete floor computation — 4 patterns × 10 paths, walk-forward tuning + oracle — **3.2 CPU-seconds, $0**.
- All controls and replays free (deterministic, 0 API calls).

## 7. Limitations

- n=10 paths per arm; single model; ±6 band is a design choice (sweep pending); wild partial.
- The tuned floor is half in-sample (pre-existing caveat).
- `reads` — the mode we picked from state-level metrics — is the worst in every game (6,731 / 7,883 / 17,061). State-level agreement does not predict game-level cost.
- Wild is sample-brittle in the LLM arms (AUDIT3); the same caution applies to wave 5's wild arm.
