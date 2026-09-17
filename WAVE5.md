# WAVE5.md — a decision-native model plays the Beer Game (post draft)

One place for everything wave 5: the story, the numbers, the audit trail, and the open items.
Companion: `AUDIT4.md` (the mistakes and the controls, in full). Data: `results/jev_protocol/`,
`results/jev_controls_noisy/`, `results/walkforward_paths.json`.

**Status:** step + noisy arms complete (10 passes each, interleaved, seeded, paired). Chaotic + wild
running — **no claim here rests on a partial arm.** Plain-text tables; markdown version goes to the repo.

---

## 0. Framing audit — which headline works best

Three candidate headlines, audited against what the data actually supports.

### A. "The claim we retracted three audits ago returns — corrected" *(backstory, not headline)*
- **For:** it *is* the corrected form of the same shape (agent beats formula under noise); strong narrative pull; the project's audits give it credibility.
- **Against:** the old claim was about a *text LLM*; this is a different model class with a ±6 clamp — not the same treatment. And leading with vindication invites re-litigating the LLM arm, which the reader hasn't seen. A reviewer attacks: "you found a bounded perturbation and dressed it as a comeback."
- **Verdict:** use as the opening paragraph. Never the headline.

### B. "A model's value is conditional on the world's uncertainty — and a confidence gate is the wrong instrument" *(recommended headline)*
- **For:** it is what the two completed arms show, quantitatively and in opposite directions: **−18.2% (noisy, 10/10 paths) vs +21.9% (fixed demand, 0/10)**. It has a clean mechanism moral: the gate defers **42.7% → 85.2%**, i.e. it switches the model off exactly where the model earns its keep. And it generalises past this game: *confidence tells you when the model doesn't know — it does not tell you when the model is wrong; those are different questions and only the second one should gate.*
- **Against:** rests on two arms (chaotic/wild pending); n=10 paths. A reviewer attacks: "which arm will the third one look like?"
- **Verdict:** headline. State the n and the pending arms in the same breath — the discipline is the credibility.

### C. "How to tell whether an agent adds information: jitter / offset / delta-replay" *(methods section, reusable)*
- **For:** the control suite is genuinely reusable and it's what makes B credible: uniform ±6 jitter (+1,659), best constant offset (−3: +912), the model's own deltas misaligned (+2,230) — and the aligned replay reproduces the model's game **to the unit** (3,783 = 3,783), which verifies the machinery. Rule of thumb it produces: *a model's intervention is information only if the same intervention, misaligned or randomised, loses its value.*
- **Against:** methods posts have a smaller audience; needs B to supply the stakes.
- **Verdict:** act 3 of the post, and its own section in the repo writeup.

**Recommended structure:** open with A in three sentences → B as the thesis → prove B with the arm-crossing numbers → make B credible with C → close on the open mechanism. And keep the retraction list visible: that's the house style, and it's the thing that makes the noisy number believable.

---

## 1. Why this arm exists

The LLM arms failed for *executional* reasons, not intellectual ones: parse failures, silent mirror fallbacks (13.9% mirror-after-error in the tool arm), byte-identical twins differing 10-20×, and a confidence gate that fired on **0.2%** of decisions (3/1,440). "Gated LLM" was an empty treatment because there was nothing to gate.

If the failure is the interface (text in, text out, parse the text), the clean test removes the text: typed question in, typed answer + probability out. **Jev** (TypeSafe, "System One"): a decision-native model. No generation → no parse failures; probabilities → a confidence signal with resolution.

## 2. Setup

- **Modes:** `choice_grid` (pick a quantity), `choice_mult` (pick a coverage multiplier κ; order = max(0, κ·q̂ − IP), arithmetic in code), `expectation` (probability-weighted), `reads` (auxiliary state reads composed in code).
- **Structure:** structured JSON state payload; all arithmetic stays deterministic in code. The model supplies judgment, the formula supplies arithmetic — the project thesis run forward on a decision-native model.
- **Gate + clamp:** confidence < τ → order := the deterministic anchor; a ±6 margin clamp is always applied; any failure → anchor fallback.
- **Controls (free, deterministic — no API calls):** `anchor_only` (the wrapper alone), `gate_always`/`gate_never` (τ endpoints), `jitter_only` (seeded uniform ±6), `offset_only` (±3/±6), `delta_replay` (the model's own deltas, misaligned).

## 3. Results

### 3.1 Fixed demand (step, 36 weeks, seed 42, n=10)

- tuned formula (θ=3.0, λ=0.35): **3,148** — the bar; walk-forward grid floor 3,206
- wrapper alone (`anchor_only`, 0 API calls): **3,681** (CV 0.000)
- `gate_always` (τ=1.0): **3,681** — gate 144/144, **0 violations**
- `jev_grid_gated` (τ=0.5): **3,549** — *one-decision-deep; retracted, see AUDIT4 §1.1*
- `gate_never` (τ=0.0, model only): **3,908** (+227 vs the wrapper, 4/10 paths better)
- uniform ±6 jitter: **4,972** · `reads`: **6,731**

### 3.2 Noisy demand (n=10 paths, paired, walk-forward floor on the identical paths)

- walk-forward floor **5,214** · untuned 5,521 · full-path oracle 4,544
- **`gate_never` (bounded model judgment): 4,263 → −951 vs the floor = −18.2%, better on 10/10 paths, t=−3.89**; −1,258 vs untuned (10/10); trends below the oracle (−281, 7/10, t=−1.51 — *not* significant, quote it as "below, not proven below")
- `jev_grid_gated`: **5,495** (+281 vs the floor — a wash; the gate deferred 85.2% of decisions and threw the win away)
- uniform ±6 jitter: 6,873 (+1,659) · `reads`: 7,883 (+2,670) · best constant offset (−3): 5,175 (+912 behind the model)
- **delta-replay** (own deltas, wrong pass): 6,493 (+2,230); aligned pass = exact replica (3,783 = 3,783)

### 3.3 The gate points the wrong way

| arm | gate deferral | confidence median | model value |
|---|---|---|---|
| step (fixed demand) | **42.7%** | 0.56 | −227 (hurts) |
| noisy | **85.2%** | 0.21 | **+1,258 (helps, vs the wrapper)** |

The calibrated gate closes where the model's judgment is most valuable. The LLM arm's problem was a gate with nothing to gate on; Jev's problem is a gate that works perfectly and defers in the wrong direction. **Unsure is not the same as wrong.**

## 4. What the control suite establishes

Same ±6 budget, same anchor: uniform random ±6 → +1,659; best scalar offset → +912; the model's own deltas misaligned → +2,230; aligned → the model's exact game. **The win is the state-conditioned timing of the adjustments — not the interface, the magnitude, the distribution, or the clamp.** Raw delta histogram on noisy: +6: 37.6%, −6: 22.6%, 0: 12.4% — the model orders *up* more than down; it is not a downward bias.

## 5. Pending (do not claim yet)

1. Chaotic + wild arms (running, partial).
2. Noisy extension to n=20 paths.
3. Band sweep ±3/±6/±12 on noisy (is the conditional pattern band geometry?).
4. Mechanism: *why* right under noise, wrong under flat demand? (damping? trend-aware protection intervals?) — decision-by-decision characterisation pending.
5. A second decision-native model, for generality.

## 6. Compute

- Jev: ~974 prompt + 213 completion tokens per decision (no generation), ~0.5-2 s/call.
- The complete floor computation — 4 patterns × 10 paths, walk-forward tuning + oracle — is **3.2 CPU-seconds, $0**.
- Fixed-demand controls and replays are free (deterministic, 0 API calls).

## 7. Limitations (carried, unchanged where noted)

- n=10 paths per arm; single model; the ±6 band is a design choice (sweep pending).
- The tuned floor is half in-sample (pre-existing caveat, unchanged).
- `reads` — the config we picked from state-level metrics — is the worst in every game. State-level agreement does not predict game-level cost.
- Wild is sample-brittle in the LLM arms (AUDIT3); expect the same caution for wave 5's wild arm.
