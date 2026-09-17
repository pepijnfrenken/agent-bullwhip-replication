# WAVE5.md — a decision-native model plays the Beer Game

One place for everything wave 5: the story, the numbers, the audit trail, and the open items.
Companion: `AUDIT4.md` (the mistakes and the controls, in full). Data: `results/jev_protocol/`,
`results/jev_controls_*/`, `results/walkforward_paths.json`.

**Status:** all four arms complete — 240 runs, 0 failures (interleaved, seeded, paired). The n=20 noisy extension + ±3/±12 band sweep completed 2026-09-17: the model beats the tuned floor by **−7.1% (15/20 paths, t=−2.15)** vs −18.2% (10/10, t=−3.89) at n=10 — direction holds, magnitude shrank; ±3 loses to the floor (+1.0%, 8/20) and ±12 is much worse (+18.7%). Data: `results/jev_noisy20/`.

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
| step (fixed) | 3,681 | 3,908 | **+6.2%** | −12.9% | +21.9% (0/10) | loses |
| noisy | 5,521 | **4,263** | **−22.8%** | −5.6% | **−18.2% (10/10, t=−3.89)** | **wins** |
| chaotic | 12,659 | **9,587** | **−24.3%** (9/10) | −39.3% | +24.9% (3/10, ns) | loses to tuning |
| wild | 16,159 | **13,710** | **−15.2%** (10/10, t=−4.29) | −25.1% | +13.2% (3/10, ns) | loses to tuning |

Read it as: **the model's correction is consistent (≈−15 to −24% over the untuned policy on stochastic arms, +6% harm when demand is flat), while tuning's gain is arm-dependent (−5.6% to −39.3%). So the model wins exactly where tuning has little to buy.** Share of tuning's gain recovered by the model: noisy 410% (beats tuning), chaotic 62%, wild 61%.

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

1. ~~Noisy extension to n=20 + band sweep~~ — **done 2026-09-17**: model 4,436.6 (CV 0.144) vs tuned floor 4,773.4 (CV 0.094) = **−7.1%, 15/20, t=−2.15**; ±3 4,819.9 (+1.0%, 8/20, ns); ±12 5,663.9 (+18.7%); anchor-only 5,487.9 (+15.0%); jitter control 6,750.9 (+41.4%, 1/20). Paired per-path stats: `results/jev_noisy20/paired_stats.json`; numbers (no charts): `(draft held locally, not in the repo)` § Numbers.
2. Mechanism: what correction does the model make that tuning finds and the ±6 band can't? Decision-by-decision characterisation pending.
3. A second decision-native model, for generality.

## 6. Compute

- Jev: ~974 prompt + 213 completion tokens per decision (no generation), ~0.4-2 s/call.
- The complete floor computation — 4 patterns × 10 paths, walk-forward tuning + oracle — **3.2 CPU-seconds, $0**.
- All controls and replays free (deterministic, 0 API calls).

## 7. Limitations

- n=10 paths per arm; single model; ±6 band is a design choice (sweep pending); wild partial.
- The tuned floor is half in-sample (pre-existing caveat).
- `reads` — the mode we picked from state-level metrics — is the worst in every game (6,731 / 7,883 / 17,061). State-level agreement does not predict game-level cost.
- Wild is sample-brittle in the LLM arms (AUDIT3); the same caution applies to wave 5's wild arm.
