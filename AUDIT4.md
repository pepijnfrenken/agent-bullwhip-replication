# AUDIT4 — the decision-native model arm (wave 5)

Scope: every claim made for the Jev arm. Same rules as `AUDIT.md` / `AUDIT2.md` / `AUDIT3.md` — our mistakes first, then what survived them and the exact support.

Status: **step and noisy arms complete (10 passes each, interleaved, seeded).** Chaotic and wild are running (partial). No claim below rests on a partial arm.

---

## 0. Summary

| # | what | verdict |
|---|---|---|
| 1.1 | "the gated config beats the formula on fixed demand (−3.8%)" | **RETRACTED** — one-decision-deep |
| 1.2 | "`reads` is the best config" (from 6-state agreement) | **RETRACTED** — worst config in every game |
| 1.3 | cross-run order-sequence diffing (our analysis method) | **INVALID** — closed loop; rewritten within-run |
| 2 | noisy arm: "bounded model judgment beats the walk-forward floor by 18.2%" | **SURVIVED 5 controls + paired design** |
| 3 | "the confidence gate defers exactly where the model is valuable" | **SURVIVED** (42.7% step vs 85.2% noisy) |
| 4 | "model value is conditional on demand uncertainty" | **REVISED** — the correction is consistent on *every* stochastic arm (noisy −22.8%, chaotic −24.3%, wild −17.9% partial vs the untuned policy) and only flips sign when demand is flat (+6.2%). What is conditional is whether it beats *tuning* (noisy yes, chaotic no). See §3.2 |

---

## 1. Our mistakes

### 1.1 The fixed-demand "win" was one decision deep — retracted

`jev_grid_gated` scored 3,543-3,575 on step demand vs the untuned floor's 3,681: a stable-looking −3.6%. Trace audit: the mixture's order sequence differs from **its own anchor in 1 of 144 decisions** (a single −6). The game is deterministic and closed-loop, so that one deviation cascades into the whole −138.

- **Why we nearly published it:** five runs, tight spread, CV 0.009, all below the floor — it *looked* like a consistent effect.
- **What killed it:** the within-run trace (each decision stores its own anchor). The consistency was run-to-run stability of the same single deviation, not a repeated contribution.
- **Consequence:** on fixed demand the honest statement is `gated ≈ the formula` (one deviation deep, sign unstable across paths).

### 1.2 `reads` — state-level metrics do not predict game-level cost

On a 6-state comparison, `reads` had the **best** agreement with the policy anchor (mean |Δ| 7.3 vs the other modes' 9.5-15.2), and we called it "the config to take forward". In games it is the worst config in both completed arms:

- step: **6,731** vs anchor 3,681 (+83%)
- noisy: **7,883** vs anchor 5,521 (+43%), +2,670 vs the walk-forward floor (1/10 paths)

Modest state-level over-orders (e.g. 61 where the anchor says 39) compound through the closed loop into ~2× cost. Any wave-6 config selection must be game-level, never state-level.

### 1.3 Cross-run sequence diffing is invalid here (our method error)

Our first audit of the gated config diffed its order sequence against `anchor_only`'s sequence week by week and "found" 10 deviating weeks. **Invalid:** in a closed-loop game different orders change every later week's state, so the two sequences are different trajectories, not a treatment/control pair. The same mistake would have manufactured a fake effect either way.

- **Fix:** every decision trace now stores its own `anchor`, so audits are within-run. `analyze_jev_traces.py`.

---

## 2. Controls — what we attacked, and the result

All on the noisy arm (the claim), n=10 paths, paired (same demand path per pass across configs).

| control | rules out | result |
|---|---|---|
| `anchor_only` (this agent's anchor, 0 API calls) | "the win is my forecast/warm-start implementation" | **3,681** on step = the repo's floor exactly; identical sequence to `order_up_to` |
| `gate_always` (τ=1.0) | "the gate is hand-wavy" | 3,681; **144/144 gated, 0 violations** — the gate provably defers to the formula |
| `gate_never` (τ=0.0) = the model-only arm | — | **4,263** on noisy (−951 vs walk-forward floor) |
| uniform ±6 jitter (`jitter_only`, seeded) | "any bounded perturbation helps" | **6,873** — +1,659 vs the walk-forward floor |
| constant offsets −6 / −3 / +3 / +6 | "a scalar nudge explains the win" | 6,155 / 5,545 / 5,175 / 6,900 — **best offset is +912 behind the model** |
| **delta-replay** (the model's own deltas, replayed from another pass: same distribution, wrong state alignment) | "the delta distribution alone explains the win" | **6,493** (+2,230 vs the model) — and on its *aligned* pass it reproduces the model's game **exactly** (3,783 = 3,783), so the machinery is verified |
| grid-ceiling check (is the model just forced to −6?) | "clamp geometry, not judgment" | on noisy the deltas sit at **+6: 37.6%**, −6: 22.6%, 0: 12.4% — the model orders *up* more than down; state-conditioned, not a bias |
| same controls on chaotic + wild (free) | "the control conclusions are noisy-arm-specific" | chaotic: model 9,587 vs jitter 12,552, offsets 12,223-12,416; wild (partial): 13,268 vs 16,028 / 15,088-17,115 — **the model beats every control on every stochastic arm** |
| paired analysis + walk-forward floor on the identical paths | "different paths / untuned baseline" | **−951 (−18.2%), 10/10 paths, t=−3.89** |

**What the control suite establishes:** the model's win is not the interface, not the clamp geometry, not the magnitude, not the distribution — it is the **state-conditioned timing** of its adjustments. Same deltas misaligned cost +2,230; the best scalar nudge is +912 behind.

---

## 3. What survived (claims and their exact support)

1. **On noisy demand, bounded model judgment beats the walk-forward tuned floor.** `gate_never` 4,263 vs wf floor 5,214: −951, −18.2%, paired, 10/10 paths, t=−3.89. Also below the untuned floor by −1,258 (10/10) and trending below the **full-path oracle** (4,544; −281, 7/10, t=−1.51 — *not* significant; quote it as "below, not proven below").
2. **On fixed demand the same model loses** (+702 vs the wf floor, 0/10; +227 vs the untuned wrapper, 4/10). And on chaotic it improves on the untuned policy by −24.3% (9/10) but **still loses to the per-path-tuned floor** (+1,910, +25%, 3/10, not significant). The correct statement is not "it wins under noise": the model is a **consistent ≈18–24% bounded correction over the untuned policy on stochastic arms**, which beats *tuning* only where tuning has little to buy (tuning's own gain: 5.6% noisy vs 39.4% chaotic).
3. **The gate defers where the model is valuable.** Deferral 42.7% (step) → **85.2% (noisy)**; confidence median 0.56 → 0.21. So `grid_gated` ≈ the anchor on noisy (+281 vs wf floor, a wash) — the gate throws away the win it should harvest. *The gate knows when it doesn't know; it does not know when it's wrong.*
4. **First config in this project to beat a walk-forward floor on any demand arm** (the LLM configs only ever beat the untuned point).
5. **Cost profile:** ~974 prompt + 213 completion tokens per decision (no generation), ~0.5-2 s/call; the complete floor computation for all four patterns × 10 paths = **3.2 CPU-seconds, $0**.

---

## 4. Still open (do not claim these yet)

1. **n=10 paths.** Noisy extension to n=20 planned; the effect should be re-tested on fresh paths.
2. **Band sweep** ±3/±6/±12 for the model on noisy — the band is a design choice; the conditional pattern must not be a band artifact.
3. **Chaotic / wild arms** — running. The conditional-value claim currently rests on two arms (one win, one loss).
4. **Single model.** Jev only; a second decision-native model would test generality.
5. **Mechanism.** *Why* is the model right under noise and wrong under flat demand? Current evidence: state-conditioned deviations whose timing matters; the mechanism (damping? trend-aware protection intervals? over-ordering avoidance?) is not yet characterised decision-by-decision.
6. **The tuned floor is half in-sample** (pre-existing caveat, unchanged).
