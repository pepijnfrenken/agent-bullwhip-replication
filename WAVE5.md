# Wave 5 — a decision-native model plays the Beer Game

**Draft. Status: fixed-demand arm complete (numbers final); protocol arm (step / noisy / chaotic / wild × 10 passes, interleaved, seeded) running — placeholders marked ⟨TBD⟩.**

---

## Why a fifth arm existed at all

The LLM arms failed for *executional* reasons, not intellectual ones:

- parse failures and silent mirror fallbacks (13.9% mirror-after-error in the tool arm),
- byte-identical twin configs differing 10-20×,
- and a **confidence gate that fired on 0.2% of decisions (3/1,440)** — there was nothing to gate, so "gated LLM" was an empty treatment.

If the failure is the interface (text in, text out, parse the text), then the clean test is a model with **no text**: typed question in, typed answer + probability out. Enter **Jev** (TypeSafe): a decision-native "System One" model. No generation → no parse failures. Probabilities → a confidence signal with actual resolution.

## Setup

- Four modes: `choice_grid` (pick an order quantity), `choice_mult` (pick a coverage multiplier κ; order = max(0, κ·q̂ − IP) computed in code), `expectation` (probability-weighted), `reads` (auxiliary state reads composed in code).
- The **state payload is structured JSON**, and **all arithmetic stays deterministic in code** — the project thesis run forward: the model supplies judgment, the formula supplies arithmetic.
- **Gate:** confidence < τ → order := the deterministic anchor. A ±6 margin clamp is always applied. Any failure → anchor fallback.
- Controls invented for this arm (the methodological contribution):
  1. `anchor_only` — the wrapper with **zero API calls** (isolates the formula from the model).
  2. `gate_always` / `gate_never` — the two endpoints of the gate (τ=1.0 / τ=0.0).
  3. **`jitter_only` — seeded pseudo-random ±6 perturbation of the same anchor.** The decisive control: *random actions of the same magnitude as the model's*. If random jitter tracks the model's jitter in cost, the model added no information.

## Fixed demand (step pattern, 36 weeks, seed 42) — final

Mean total supply-chain cost, lower is better. Deterministic demand → CV≈0 by construction.

| config | mean cost | vs untuned floor | notes |
|---|---|---|---|
| **tuned formula (θ=3.0, λ=0.35)** | **3,148** | −14% | the bar that matters (repo's own measurement; half in-sample) |
| untuned formula (3.0, 0.5) = `anchor_only` | 3,681 | — | the wrapper alone, 0 API calls |
| `gate_always` (τ=1.0) | 3,681 | — | gate 144/144, **0 violations** → the gate provably defers to the formula |
| **`jev_grid_gated` (τ=0.5, mixture)** | **3,543** | **−3.8%** | ⚠️ one-decision-deep — see audit below |
| `jev_gate_never` (τ=0.0, model only, clamped) | 4,164 | +13% | 131-137/144 decisions differ from the anchor |
| `jitter_only` — **random** ±6, 3 seeds | 4,972 / 5,010 / 5,345 (≈**5,109**) | **+39%** | same ±6 budget as the model |

## The audit that killed our own headline

The gated config looked like it beat the formula (−138). We ran the wrapper ablation first: `anchor_only` scored **exactly 3,681** — so it wasn't the forecast implementation. Then the trace audit: the mixture differs from *its own anchor* in **1 of 144 decisions** (a single −6). In a closed-loop deterministic game that one decision cascades → the entire −138 is **one decision deep**. It will not survive demand seeds or patterns. Not a result. *(Also: we first "found" 10 deviating weeks by diffing the gated run's order sequence against a different run's — invalid, because different orders change every later week's state. The audit is within-run: each decision trace stores its own anchor.)*

## What actually survives

1. **Even a purpose-built decision model does not beat tuned 1970s math.** This is a *stronger* form of the thesis: in the LLM arms you could argue the model was simply bad at the task; Jev is the strongest available alternative (typed decisions, probabilities, no text interface) and it still loses to a formula that tunes itself from 18 weeks of history.
2. **It degrades gracefully where the LLM exploded.** No parse failures (structural), no mirror fallbacks, no 10^194 blowups, no 107-unit spikes. Worst case is bounded by the clamp.
3. **Its confidence gate is usable.** 40-43% of decisions deferred to the formula, **zero gate violations** (verified per decision). Compare: the LLM's gate fired 0.2%. The gate now does what it says.
4. **Its deviations are informative — and that is measurable.** Same ±6 budget: the model's own deviations cost **+13%**, random deviations cost **+39%**. The model's judgment carries real signal (~3× less harmful than noise); it just doesn't carry more signal than the tuned formula.
5. **You don't need RL to make an agent safe.** The paper's fix is GRPO post-training. The cheap inference-time equivalent, demonstrated here: **bound the action space (clamp) + let a formula interrupt the model (gate)**. A decision model you can interrupt is a decision model you can ship.

## Protocol arm (step / noisy / chaotic / wild × 10 passes, interleaved, seeded)

⟨TBD — table + CV columns when the run lands; it is the one that decides how much of the above is protocol-grade.⟩

## Compute

⟨TBD — token/wall-clock totals from the run record.⟩ Fixed-demand arm: 864 decisions, ~1.0M tokens (~1.2k/decision), ~0.54 s/call, **zero completion generation** (typed answers only).

## Limitations

- Fixed-demand arm is n=3 runs, one demand seed; the protocol arm is the one that counts.
- Jev is not deterministic across calls (same state → different choices at 0.09-0.34 confidence), so variance is a first-class metric, not noise.
- The tuned floor (3,148) is scored half in-sample (README's own caveat, unchanged).
- The clamp ±6 is a modeling choice; the jitter control bounds how much the ±6 budget itself can be worth.

---

*Repo: `~/projects/agent-bullwhip-replication` (commits `38ff517`→`06b56fc`). Data: `results/jev_smoke/`, `results/jev_controls/`, `results/jev_jitter/`, `results/jev_protocol/` ⟨TBD⟩. Scripts: `probe_jev.py`, `compare_jev_vs_text.py`, `analyze_jev_traces.py`, `agent_bullwhip/jev_agent.py`.*
