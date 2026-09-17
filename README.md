# Agent Bullwhip — LLM Agents in the Beer Game, Audited

Replication + extension of **arXiv:2605.17036**, *Reliability and Effectiveness of Autonomous AI Agents in Supply Chain Management* (Long, Simchi-Levi, Zhu, Su, Calmon & Calmon; Harvard/MIT/Georgia Tech). LLM agents manage a four-echelon MIT Beer Game. The paper's question: are autonomous AI agents reliable supply-chain decision-makers? Its fix: GRPO RL post-training. Our question: does cheap inference-time structure work instead? Our answer took three audits to arrive at, because the first three answers we produced were wrong.

The mistakes are ours. They are documented in full in `AUDIT.md`, `AUDIT2.md`, `AUDIT3.md`; the surviving numbers below are the ones that survived.

---

## TL;DR — the surviving claim

> "A 1970s inventory formula that tunes itself from 18 weeks of history beats a state-of-the-art LLM agent in every environment — mild, chaotic, and wild demand — at 3-4 orders of magnitude less compute. The LLM's apparent edge was information leakage and an untuned baseline. The intelligence in the 'AI agent' was the formula all along."

Every contested word in that sentence, grounded:

- **"state-of-the-art LLM agent"** — the strongest free-API model this endpoint offered at run time, deepseek-v4-flash, wired into the paper's own agent framework. Not a GRPO-tuned frontier model; we make no claim about those. The paper's *reliability finding* does replicate on this endpoint: a plain LLM with no structure is an instability machine (baseline mean 38.8M / CV 1.06 on fixed demand; medians 281M–394M under noise and chaos; the paper's own agent CVs were 0.13–0.46 — qualitatively the same story, far worse here).
- **"1970s"** — order-up-to with exponential smoothing. A standard inventory policy class, decades older than the agents; the paper itself uses it as its deterministic contrast. Nothing new was invented.
- **"tunes itself"** — an 81-point (θ, λ) grid search on the **first 18 weeks** of each demand path; the winning parameters are deployed frozen. No oracle, no future knowledge, no API calls. "No hindsight" refers to *parameter choice*; the headline walk-forward cost column is scored over the full 36-week game and is therefore **half in-sample** — see the protocol section, where we do not paper over that.
- **"mild, chaotic, and wild demand"** — the noisy, chaotic, and wild demand arms (noisy = the demand step with AR(1) noise).
- **"3-4 orders of magnitude less compute"** — measured two ways, both from the run records: one LLM arm (10 passes) burned **~2.73M prompt tokens + ~177K completion tokens across 4,320 API calls, ~56–65 minutes of wall-clock** per environment. The complete walk-forward + oracle computation for **all three** environments is ~4,900 deterministic simulations, **~3 CPU-seconds, $0**. We do not measure FLOPs; by API tokens and wall-clock the formula side is 3+ orders of magnitude cheaper.
- **"apparent edge"** — the two retracted comparison wins (information leak; untuned baseline). Note: in the surviving table the verbal gate **still beats the untuned point in all three environments**. The wins that remain are against an untuned (θ, λ) — which is what tuning is for.
- **"the intelligence … was the formula all along"** — the wrapper-only ablation (LLM decision replaced by the policy, zero chat calls) scores **exactly 3,681** on fixed demand: identical to the policy alone. The LLM *inside* the wrapper **adds** +12–34% cost; it does not subtract. On stochastic demand no LLM config mean beats the self-tuned policy.
- **"wave 5 — a different model class"** — replace the text LLM with a decision-native model (typed questions in, typed answers + calibrated probabilities out, no generation) and the picture changes in exactly one place: on **noisy demand** its bounded, state-conditioned adjustments beat the walk-forward floor by **−18.2% (paired, 10/10 paths, t=−3.89)**; on fixed demand the same model **loses** by +22%. Under determinism the formula is still the win; under uncertainty a bounded model that a formula can interrupt is worth having. But its confidence gate defers **85%** of decisions on the arm where it wins — it switches itself off exactly where it earns its keep: *unsure is not the same as wrong*. `WAVE5.md` · `AUDIT4.md`

### The dead claims

1. ~~"A gated LLM is within 11% of optimal."~~ Three independent problems: the sequential run was window-confounded; the "floor" wasn't optimal (other settings score 3,148–3,206 vs 3,681); and the named confidence gate fired 0.2% of decisions — there was nothing to gate.
2. ~~"The LLM beats the formula under noise."~~ The LLM saw a same-week information leak (`incoming_now`); a formula with the same lookahead beat it by 37–40%. And the "formula" was an untuned a-priori point — a grid search beat the LLM once information was equal.
3. ~~"The crossover: the LLM catches up as chaos rises."~~ The chaotic/wild arms ran unseeded; each config played a *different* demand path. The pairing the crossover depended on did not exist.
4. Framing, not a dead claim: the headline is cost, not reliability — and neither side wins CV outright: the formula wins noisy and chaotic variance, the LLM wins wild variance (see CV table).

---

## The surviving result

All numbers: **10 passes × 4 configs, interleaved, seeded, leak-free; 40/40 runs completed (0 API failures); mean total supply-chain cost over 36 weeks** (holding $1, backlog $2 per unit-week; lower is better). CV = coefficient of variation across the 10 runs of each column's own seeded path set.

| environment | verbal gate (LLM) | gated KB (LLM) | untuned floor (3.0, 0.5) | walk-forward floor | oracle floor | on-disk source |
|---|---|---|---|---|---|---|
| **noisy** | 5,424 | 5,973 | 5,521 | **5,214** | 4,544 | `results/noisy_leakfree/` |
| **chaotic** | 9,156 | 11,525 | 12,659 | **7,678** | 6,926 | `results/chaotic_seeded/` |
| **wild** | 13,335 | 15,849 | 16,159 | **12,111** | 9,821 | `results/wild_seeded/` |

| CV (within-column) | verbal | gated | untuned | walk-forward |
|---|---|---|---|---|
| noisy | 0.199 | 0.203 | 0.136 | **0.131** |
| chaotic | 0.435 | 0.377 | 0.380 | **0.339** |
| wild | 0.429 | 0.406 | 0.437 | 0.603 |

Range (min–max) per row: walk-forward noisy 4,025–6,246, chaotic 3,848–12,545, wild 4,183–26,200; verbal gate 3,943–7,104 / 4,443–18,061 / 6,265–23,304.

**How to read this without fooling yourself:**

- **The columns are not path-paired.** Walk-forward uses seeds 1000–1009; the LLM arms use their own seeded paths. So no column-to-column significance test exists in this table. What the table *does* support is means, CVs, and ranges — and even those come from n=10.
- **Noisy is the weakest row:** 5,214 vs 5,424 (−3.9%), overlapping ranges, unpaired, n=10. Treat it as consistent-but-weak.
- **Chaotic is the strongest row:** −16% mean against the best LLM config, with the lowest CV (0.339) of any column in that row.
- **Wild is mean-better, not variance-better:** −9% mean, but walk-forward CV (0.603) is *worse* than every LLM column (0.406–0.437), and the top of its range (26,200) exceeds the verbal gate's (23,304). The wild row is a range, not a point.
- **The best LLM summary is the verbal gate, and it is a mixture, not pure model** — 21–29% of its decisions are silent mirror fallbacks (see below). Comparing a policy-and-model mixture to a tuned formula is the honest framing, and the mixture still loses on mean cost everywhere.
- The verbal gate beats the **untuned** point in all three environments (5,424 < 5,521; 9,156 < 12,659; 13,335 < 16,159); the gated config beats it in two of three. What beats the LLM is not "a formula" — it's *tuned* parameters.

**Paired evidence we do have (different comparator).** AUDIT3 reconstructed the exact demand paths the LLM actually played (from traces, pre-seed-fix arms) and scored the tuned floor on the *same* paths: tuned floor vs `kb_system_gated` — chaotic 7,175 vs 11,055 (paired permutation p = 0.015, 9/10 passes), wild 18,042 vs 28,054 (p = 0.004). Comparator is the gated config, not the verbal gate, and the wild levels reflect those arms' different demand draws — treat this as directional support for the table, not p-values for it.

### Fixed demand: the cleanest result in the repo

Deterministic demand step: `order_up_to` (3.0, 0.5) scores **3,681, CV 0.000**. The wrapper-only ablation — LLM decision swapped for the policy, zero chat calls, zero tokens — scores **exactly 3,681**. The LLM inside the wrapper **adds +12–34%** over the wrapper alone. That is the whole story in miniature: the wrapper is the entire win; the model riding inside costs.

Two caveats, in the open: 3,681 is the a-priori point (3.0, 0.5), not an optimal policy — a tuned (3.0, 0.35) scores 3,148; and "CV 0.000" is the deterministic engine, not a reliability property of anything.

### Baseline: plain LLM, no structure

| environment | statistic |
|---|---|
| fixed (step) | mean 38.8M, CV 1.06 |
| noisy | median 281M |
| chaotic | median 394M (leak-free arm; means ran to ~2×10^21) |
| wild | median 11.6M, blowups to ~1.8×10^194 |

Medians, because blowup runs destroy the means. Provenance: the noisy/chaotic/wild medians are AUDIT3-certified distribution stats from the leak-free arms (which include the pre-seed-fix chaotic/wild runs). AUDIT3 rules the *qualitative* claim — plain LLM = instability machine — clean either way: the seeded re-runs give chaotic baseline mean 1.8×10^12 and wild mean 21.1M (max 91M). The wild median alone is ~1,000× the walk-forward floor.

### The verbal gate — the better LLM config, honestly labeled

`kb_pointer_verbal` fires when the model **orders 0 while holding backlog while its stated reasoning says "cover the backlog"** — the *confidently wrong* failure mode — and overrides to the order-up-to anchor. It fires on ~3–4% of decisions and is the cheapest LLM config in every environment.

What it actually is: **a mixture.** 21–29% of its decisions are silent mirror fallbacks — parse/format failures defaulting to `mirror` (order = last incoming; `agents.py`, `fallback: str = "mirror"`), which is a *different policy* from the anchor it also contains, and the config's real weakness (mirror cascades can blow up: AUDIT2 documents an 11,522 run). The gate is a keyword regex, so its firing rate is a heuristic property, not evidence about model reasoning. This is why we never call it "pure model." (Distinct from the confidence gate — fired 3 of 1,440 decisions — whose inertness killed dead claim #1.)

### What the paper's fix is, and why we didn't run it

The paper's remedy — GRPO RL post-training — is expensive, model-specific, and out of reach for most teams. We tested cheap inference-time structure instead, and then forced that structure through three adversarial audits of our own design. The paper's *finding* (agents are unreliable) replicates here; its *mechanism* (reliability can be RL-trained in) is untested on this endpoint.

---

## The story — why this table took three audits

The numbers above were each preceded by a headline that was wrong. The audits are the content; details in `AUDIT*.md`. Each section is tagged with the (now-dead) protocol it ran under.

### 1. The naive win *(sequential, leak on, untuned floor, n=30)*

First run: `kb_system_gated` 4,086 / CV 0.089, "within 11% of optimal". Exciting. One hour of auditing killed it.

### 2. Audit #1 — window drift *(same protocol; drift only)*

Identical prompts, one hour apart, differed by **16×**. The sequential matrix ranked configs across different serving-load windows; endpoint non-stationarity, not agent quality, produced the ranking. Fix: an interleaved round-robin runner — every config plays the same window.

What interleaving did *not* fix: the gated-vs-ungated twin gap reproduced under interleaving (`kb_system_gated` 4,113 vs `kb_system_introspect` 55,098, no pass overlap) with the gate firing 0.2% of the time. We have no mechanism for that. It is the open confound (config-correlated completions; exchangeability p ≈ 0.001) — which is why the two LLM columns' relative ranking is provisional.

### 3. Audit #2 — the leak and the untuned baseline *(interleaved, leak on, untuned floor)*

"LLM beats formula under noise" died twice, independently:

- **The leak.** Upstream tiers saw `incoming_now` — the tier-below's *same-week* order, moments after it was computed. One-week lookahead. A formula handed the same lookahead scored 2,920–3,042 — **37–40% below the LLM**, three times the size of the headline effect. Fix: upstream tiers see only last week's information; verified **0 leaked weeks across 5,752 upstream traces**. The retailer still sees current-week customer demand — shared with the formula (AUDIT3 measured its value to the floor at ~7%: 4,701 → 4,369), disclosed, not hidden.
- **The untuned baseline.** The comparison point (3.0, 0.5) was never tuned. Fix: tuned floors became the honest baseline — and they beat the LLM.

### 4. Audit #3 — the seeding bug *(chaotic/wild arms, unseeded)*

The crossover experiment ran **unseeded** — each config per pass drew independent demand, silently destroying the pairing, the interleaving control, and the tuned baselines (tuned on demand the LLM never saw). The "crossover" was an artifact of unshared randomness. Fix: every pattern seeded, one shared demand per pass, regression tests locking the contract. **53/53 tests pass** (unit tests — they bless the code, not the ten-pass means).

### 5. The walk-forward objection

"You tuned the formula with hindsight!" Fair. So the formula now tunes itself and the protocol section below states exactly what that means — including the in-sample half we don't hide.

---

## The walk-forward protocol, exactly as run

1. Simulate the **first 18 weeks** of each seeded demand path.
2. Grid-search (θ, λ) — 9 θ values × 9 λ values = **81 points** — choosing the parameters with the lowest cost on those 18 weeks. Deterministic; ~3 CPU-seconds for all three environments in `walkforward_floor.py` (measured).
3. Deploy those parameters frozen for the rest of the game.

**Honest scoring note.** `walkforward_floor.py` tunes on weeks 1–18, then scores the **full 36-week game**. The headline walk-forward column therefore includes the tuning half — 50% in-sample optimism in the number. We measured the effect instead of hiding it, with a held-out check:

- A fresh 18-week game on the **held-out demand suffix** (weeks 19–36; fresh state — call it what it is: a suffix game, not the tail of the same trajectory): walk-forward vs untuned (3.0, 0.5) — noisy 4,540 vs 5,918 (10/10 passes), chaotic 5,954 vs 8,675 (9/10), wild 8,978 vs 9,558 (7/10) → **26/30 wins**. The tuning is not merely in-sample overfitting.
- Scope of this check, stated narrowly: it answers "is the *tuning* robust out-of-sample?" It does **not** compare walk-forward to the LLM columns — those are full-game and not path-paired. The claim "walk-forward beats the LLM" rests on the full-game means in the table, in-sample half included.

**Headroom.** Walk-forward sits 11–23% above the oracle (full-path-tuned: 4,544 / 6,926 / 9,821); every LLM config sits 19–66% above it. Tuning buys 6–39% (over untuned); the LLM's distance to the floor is not closed by any amount of it.

## Audit history (one line each)

| What died | Cause | Fix / status |
|---|---|---|
| "within 11% of optimal" | window drift (16×) + floor not optimal + confidence gate inert (0.2%) | interleaved runner; floor disclosed; twin gap = open confound |
| "LLM beats formula under noise" | `incoming_now` leak (lookahead worth 37–40%) + untuned baseline | upstream t−1 only; 0/5,752 leaked weeks; tuned-floor baselines |
| "LLM catches up as chaos rises" | chaotic/wild arms unseeded — different demand per config | seed every pattern, one demand per pass, regression tests |
| (disclosure) | retailer sees demand[t]; 3,681 isn't optimal | documented; 3,148–3,206 settings exist; wrapper-only = 3,681 exactly |
| (wave 5) "gated Jev beats the formula on fixed demand" | one-decision-deep — 1/144 decisions differ from its own anchor, deterministic cascade | retracted; within-run trace audit (`AUDIT4.md` §1.1) |
| (wave 5) "`reads` is the best config" | state-level agreement ≠ game-level cost | 6,731 step / 7,883 noisy vs anchor 3,681 / 5,521 (`AUDIT4.md` §1.2) |
| (wave 5, our method) cross-run order-sequence diffing | closed loop: different orders ⇒ different later states ⇒ two trajectories, not a treatment/control pair | invalid; audits are within-run, every decision trace stores its own anchor (`AUDIT4.md` §1.3) |
| (wave 5) "bounded model judgment beats the walk-forward floor by 18.2% under noise" | attacked with paired paths, a path-matched walk-forward floor, jitter/offset/delta-replay controls | **survived** (`AUDIT4.md` §2) |

## Method

- **Engine:** deterministic 4-echelon beer game per the paper's §5.1 operational model, canonical beer-game cost constants (holding 1, backlog 2). The paper's exact constants are private; this is a protocol-specified replication, not bit-exact.
- **Information set, one box:** retailer sees current-week customer demand; every upstream tier sees last week's downstream order only; the formula sees the same `incoming_last` its tier sees. Enforced by tests (`tests/test_no_leak.py`). The retailer's current-week view is the one disclosed deviation from the paper's t−1 timing; it is shared by both sides of the comparison.
- **Agents:** deepseek-v4-flash (free endpoint), four roles; wrappers: knowledge-base playbooks, introspection + confidence traces, low-confidence gating to the anchor, verbal-consistency gate. Cache-buster nonce on every API call (defeats serving-level caching).
- **Runner:** interleaved round-robin (same window, same seeded demand, one pass at a time); truncate-on-start with a run UUID; KB text snapshotted per run; full per-call traces.
- **Deterministic baselines:** `mirror` (order = incoming); `order_up_to` (order-up-to with exponential smoothing; θ = safety-stock multiplier, λ = smoothing weight).
- **Metrics computed per run** (in the summaries): total cost, CV, Ψ/Φ bullwhip amplification, p95/p99 backlog, instruction-failure rate. They are not load-bearing for the surviving claim, which is cost and CV; that is stated rather than implied.
- **Wave 1 configs** (`voting5`, `budget`, `prompt_weighted`, `anchor`, `combined`) ran only under the pre-audit sequential protocol; they are not part of the surviving table. Tombstoned, not vanished.

## Limitations (the ones we could not fix)

1. **n=10, unpaired columns.** The LLM arms and the walk-forward script run different seeded paths; the table cannot be significance-tested column-to-column. AUDIT3's demand-matched reconstruction is the paired test that exists; it is directional support (different comparator), not a p-value for this table.
2. **The headline walk-forward cost is half in-sample.** The held-out suffix check (26/30 vs untuned) shows the tuning isn't overfit, but it does not re-serve the LLM comparison; that claim rests on the full-game means.
3. **The verbal column is a mixture** — 21–29% mirror fallbacks + 3–4% regex gate overrides + model orders. It is the cheapest LLM config, and it is not "the model alone."
4. **Wild is sample-brittle.** Tuned floors swung 11k–22k across demand samples (AUDIT3); walk-forward CV 0.603 is the worst in its row. Read the wild row as a range.
5. **One cheap free-API model.** deepseek-v4-flash only; GRPO-tuned frontier models untested; no human baseline, so the paper's 67%-vs-humans headline is not replicated.
6. **Twin gap has no mechanism.** Byte-identical twin configs differ 10–20× with a 0.2% gate; the LLM columns should not be treated as clean treatments until that is explained.
7. **Retailer lookahead residual** — shared, ~7% of floor cost to the holder, documented; a deviation from the paper's timing, not a demonstrated LLM-side bias.
8. **Compute is quoted as engineering cost** (tokens + wall-clock), not scientific FLOPs.

## Tool-calling arm — preliminary result (TBC with a better harness + model)

> **Status: PRELIMINARY / TO BE CONTINUED.** One full game (n=1, 144 decisions), one model (Qwen/Qwen3.8-Flash with tool access), real engine, full per-decision trace. The result below is directional — the taxonomy is the finding, not the cost point. A stronger, more consistent tool-calling model + a harness with better error recovery is the planned continuation (see the TBC list at the end).

**Setup.** `ToolAgent` may call a `run_python` tool to compute each week's order; the tool result (stdout or error) comes back, the agent decides, and the decision is recorded alongside what the deterministic floor would have ordered for the same state. 144 decisions (4 roles × 36 weeks) on the fixed step demand. Trace: `results/toolagent_traces/run0.jsonl` (one JSON per decision: ctx, code, exec result, floor comparison, error class).

**Headline.** The tool agent **cannot beat the 1970s order-up-to formula**: game cost **6,948 vs 3,681 floor (+89%)**. Match-to-floor 61/144 (**42.4%**); when it deviates it over-orders 37.5% and under-orders 20.1% of decisions (mean |Δ| 7.96 units, max over +117, max under −177).

**The failure taxonomy (the actual finding, audit-verified from the trace).**

| failure class | rate | detail |
|---|---|---|
| **Mirror-after-error** | **13.9%** (20/144) | tool error occurred AND the returned order equals the previous week's order (the measurable version of the silent-fallback anti-pattern). No parse-fail records exist in the trace (0/144). |
| **Broken code** | **22.9%** (33/144) | undefined names (`outstanding_sum`, `sim`, `sqrt` not imported, `demand_fn` never defined), TypeError 1, IndexError 2. |
| **Recovery vs give-up** | 54% / 46% | of the 35 broken-code decisions: 19 issued another tool call (recovered), 16 gave up → mirrored. |
| **Error is catastrophic** | 51.4% → 12.1% | when code runs clean the model matches the floor 51.4% (57/111); when a tool error occurs, match drops to 12.1% (4/33) and mean |Δ| roughly doubles. |
| **Code-length drop on error** | 1,178 → 765 chars | clean decisions carry ~1,178 chars of code; error decisions ~765 — the model writes less and stops reasoning. |
| **Knows the right family** | high | base-stock formula in 95/144, forecast/mean in 128/144, simulation/brute-force in 109/144 — the strategy is right, the *execution* is what botches. |

> **Audit note (2026-08-30):** the earlier draft's "43.8% silent mirror fallback (63/144)" was **not supported** by the trace — there are 0 parse-fail records and no explicit fallback flag. The measurable form is **20/144 mirror-after-error (13.9%)** + the 12.1%-match-on-error collapse. The 6,948 game cost comes from the run summary (`results/toolagent_cc/*.summary.json`), not from per-decision fields in run0.jsonl. This section states only what the trace supports.

**Interpretation.** This is the same story as the main result, extended to tool use: the model knows the right *family* of solutions (order-up-to, base-stock, simulation), but execution failures (undefined names, wrong references, stopping after an error) make it worse than the formula — and the mirror-after-error pattern (13.9% of decisions) hides part of that failure inside the cost number. The intelligence is still in the deterministic layer; the tool-enabled model adds cost, not capability.

### To be continued (TBC)

This arm is intentionally published as **preliminary**. The planned continuation, with a better harness and a more consistent tool-calling model:

1. **Better model** — a model with reliable tool-calling (fewer undefined-name errors; better recovery after tool errors) instead of Qwen3.8-Flash.
2. **Better harness** — the harness should: (a) catch faulty tool calls **mid-flight** and feed the error back for a retry loop, (b) gate/override on repeated failure (the deterministic floor as the fallback when the tool path degrades), (c) detect the silent-mirror failure mode and surface it, not hide it, (d) checkpoint per decision so a mid-game outage doesn't restart the whole game.
3. **More games** — n≥3 games (or more) on the step pattern to confirm the taxonomy is stable, plus one noisy-demand arm to see whether tool use helps when the world isn't deterministic.
4. **A tool-error-aware writeup** — the 51%→10% match-drop and reasoning collapse quantified across runs; whether recovery attempts (54%) ever actually fix the decision; whether gating on tool-error rate beats the fixed verbal gate.

The trace collector and analyzers (`collect_tool_traces.py`, `analyze_toolagent.py`, `analyze_tool_traces.py`) are in the repo and reproduce everything above from `results/toolagent_traces/run0.jsonl`.

## Wave 5 — a decision-native model (step + noisy arms complete)

The LLM arms failed for **executional** reasons: parse failures, silent mirror fallbacks (13.9% mirror-after-error), twins differing 10–20×, and a confidence gate that fired on **0.2%** of decisions — "gated LLM" was an empty treatment because there was nothing to gate. Wave 5 removes the text interface entirely: **Jev** (TypeSafe, "System One") — typed question in, typed answer + probability out. No generation, so no parse failures; probabilities, so the gate has resolution.

Full story, numbers and open items: **`WAVE5.md`**. The mistakes and every confound control: **`AUDIT4.md`**.

| demand arm | wrapper alone | walk-forward floor | Jev, model only | gated Jev (τ=0.5) | random ±6 jitter |
|---|---|---|---|---|---|
| step (fixed) | 3,681 | 3,206 | **3,908** (+227, 4/10) | 3,549 † | 4,972 |
| noisy | 5,521 | 5,214 | **4,263 (−951, 10/10 paths, t=−3.89)** | 5,495 (+281 ≈ wash) | 6,873 (+1,659) |
| chaotic | 12,659 | 7,677 | **9,587** (−24.3% vs the wrapper, 9/10; +1,910 vs the tuned floor, 3/10, ns) | 12,621 | 12,552 |
| wild | 16,159 | 12,111 | **13,710** (−15.2% vs the wrapper, **10/10**, t=−4.29; +1,598 vs the tuned floor, 3/10, ns) | 16,178 | 16,028 |

† retracted: the fixed-demand "win" is one-decision-deep (1 of 144 decisions differ from its own anchor; deterministic cascade). Details in `AUDIT4.md` §1.1.

- **First paired comparison in the project:** the walk-forward floor was recomputed on the *identical* demand paths (seeds 1000+i) as the Jev arms, so every delta above is a per-path paired statistic, not a difference of means across unpaired columns (`walkforward_paths.py`, `paired_analysis.py`).
- **The control suite (all free, deterministic, 0 API calls)** — `anchor_only` (wrapper alone), `gate_always`/`gate_never` (τ endpoints: the gate provably defers to the formula, 144/144, 0 violations), `jitter_only` (seeded uniform ±6), `offset_only` (±3/±6), `delta_replay` (the model's own deltas, replayed from another pass). Result on noisy: the model's win is **not** its interface (+0, controlled), **not** the clamp geometry (random ±6: +1,659), **not** a scalar nudge (best constant offset: +912), **not** its delta distribution (misaligned replay: +2,230; aligned replay reproduces the model's game exactly) — it is the **state-conditioned timing** of the adjustments.
- **The pattern is not "it wins under noise" — it is "it wins where tuning is weak".** All four arms complete (240 runs, 0 failures): the model's correction is consistent across stochastic arms (−22.8% noisy, −24.3% chaotic, −15.2% wild vs the untuned policy) while per-path tuning's own gain ranges 5.6% → 39.3%. So it beats the tuned floor on noisy and loses on chaotic/wild *without the model changing* — it is a hindsight-free, per-decision partial substitute for tuning.
- **The gate points the wrong way.** Deferral 42.7% on step → **85.2% on noisy** (confidence median 0.56 → 0.21). The gate closes exactly where the model's judgment is most valuable.
- **`reads`, the mode with the best state-level agreement with the policy (mean |Δ| 7.3), is the worst in every game** (6,731 step / 7,883 noisy). State-level agreement does not predict game-level cost.
- **Still running / open:** a noisy extension to n=20 paths + a ±3/±12 band sweep (running), and the decision-level mechanism ("what correction does the model make that tuning finds?").

### Wave 5 — to be continued

1. Noisy extension to n=20 paths + ±3/±12 band sweep (running).
2. Mechanism characterisation: which correction does the model make that tuning finds and the ±6 band can't?
3. A second decision-native model, for generality.

```bash
# env: FREEINFERENCE_API_KEY set (model defaults to deepseek-v4-flash)
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

# the audit-surviving protocol — interleaved, seeded, one demand per pass.
# The noisy arm ran with --seed 7 (CHANGES.md): same seed = same demand paths,
# but LLM costs will differ run-to-run across endpoint stochasticity — that
# variance is exactly what CV measures, so expect a *distribution*, not the table.
.venv/bin/python -m agent_bullwhip.interleaved_runner \
  --configs kb_pointer_verbal,kb_system_gated,order_up_to,baseline \
  --runs 10 --horizon 36 --pattern noisy --seed 7 --outdir results/noisy_repro

# walk-forward floor (no LLM, no API): reproduces the walk-forward + oracle columns
.venv/bin/python walkforward_floor.py

# tuned-floor sweep:
.venv/bin/python tuned_floor.py

# audit regression suite: 53 tests
.venv/bin/python -m pytest tests/ -q
```

Do **not** use the sequential runner (`agent_bullwhip.runner`) for claims — that is the protocol Audit #1 deleted. It remains for smoke tests and archaeology.

**Data provenance.** The surviving sets are `results/noisy_leakfree/`, `results/chaotic_seeded/`, `results/wild_seeded/` — filenames match the table. Earlier directories (`interleaved_val2`, `wrapper_only`, `chaotic_leakfree`, `wild_leakfree`, the flat `*.summary.json` files) are archaeology; the audits explain exactly which parts of them are compromised.

## Links

- Paper: https://arxiv.org/abs/2605.17036 (DOI 10.48550/arXiv.2605.17036)
- Audits: `AUDIT.md` · `AUDIT2.md` · `AUDIT3.md` — the mistakes, in full
- Change log: `CHANGES.md`
- Surviving data: `results/noisy_leakfree/` · `results/chaotic_seeded/` · `results/wild_seeded/`
- Wave 5 (decision-native model): `WAVE5.md` · `AUDIT4.md` — data `results/jev_protocol/` · `results/jev_controls_noisy/` · `results/walkforward_paths.json`
-  (unpublished): ``