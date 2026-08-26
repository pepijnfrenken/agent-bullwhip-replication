# Confound & Edge-Case Audit #2 — Agent Bullwhip Replication (post-interleaving)

Auditor: skeptical re-derivation from source + on-disk results. Read-only; simulations in /tmp only.
All numbers below recomputed from the result files on disk on 2026-08-26 ~13:00 UTC.

Result sets audited:
- `results/interleaved_val2/deepseek-v4-flash-interleaved-dc2d55b0.jsonl` (+summary) — fixed demand, 10 passes × 8 configs
- `results/noisy_interleaved/deepseek-v4-flash-interleaved-5c99f622.jsonl` (+summary) — noisy demand, 10 passes × 4 configs
- `results/wrapper_only/deepseek-v4-flash-interleaved-32993b1b.jsonl` (+summary) — wrapper-only ablation, 10 passes × 5 configs

Simulation cross-check: my reproduction of the order-up-to floor on the 10 noisy demand paths
(`make_demand(36,"noisy",seed=1000+i)`, engine.py:95-107) gives per-run costs
[4635, 5996, 6397, 5634, 6681, 6211, 4327, 4922, 5375, 5035] — **byte-identical** to the on-disk
`order_up_to` per-pass costs. The engine/sweep/statistics pipeline below is validated against the
observed data.

---

## Verdict table

| # | Threat | Verdict | Key evidence |
|---|--------|---------|--------------|
| C1 | Noisy headline (LLM beats formula) is confounded by untuned floor hyperparameters | **HIGH** | tuned floor (θ=3.5, λ=0.2) = 4,701 mean, *below* the LLM's 4,849; OOS ≈ parity; see §1 |
| C2 | Comparison unfair: LLM sees `incoming_now`, floor doesn't | **HIGH** | leaked floor (3.0, 0.5) = 3,042, best leaked 2,920 — 37–40% below the LLM (§3) |
| C3 | Noisy result statistically marginal at n=10 | **HIGH** | sign p=0.0215, perm p=0.044, Wilcoxon p=0.084, bootstrap 95% CI [−1,083, +251] includes 0 (§1) |
| A | "Gating tames the agent 10–22×" — ratio real, mechanism unexplained | **HIGH** | 9.6–21.9× per pass reproduced, but gate fired **3×** and clamp bound **2×** across 1,440 decisions; identical prompts draw non-exchangeable completions (perm p ≈ 0.001) (§5) |
| B | Wrapper-only = floor exactly | **NONE** (verified) | monkeypatch replaces `decide`; 0 chat calls, 0 tokens, 50/50 runs = 3,681 exactly — but the inference "LLM contributes 0%" is directionally wrong: LLM config costs **more** (§6) |
| — | Interleaving fairness | **NONE** | shuffle is seeded and reproduced; every config once per pass; positions vary; noisy run used seed 7 (undocumented) (§7) |
| — | Data integrity | **NONE** | 80/80, 40/40, 50/50; run indices 0–9; zero error stubs; token/call sums match summaries exactly (§8) |
| — | Cache-buster nonce | **LOW** | injected live (client.py:58); `prompt_hash_dups_total=0` is now guaranteed by construction, not evidence of stochastic serving (§7) |
| — | Verbal gate blowup + 34% failure fallback | **HIGH** | noisy pass 5 = 11,522 (+85% vs floor 6,211) from a mirror-fallback cascade; 461/1,440 (32%) of noisy verbal decisions are *fallbacks*, never the gate (§4) |
| — | Same-week `incoming_now` leak | **HIGH** (systematic, pro-LLM) | engine.py:132-142 → prompt literal agents.py:239/323; upstream tiers read the tier-below's *same-week* order at decision time (§3) |
| — | Residual minute-scale window/config correlation | **HIGH** (open) | non-exchangeable completions at byte-identical prompts; mechanism UNVERIFIED; 10-pass interleaving slashed but didn't kill drift (§5) |

---

## 1. Is the noisy result statistically real? (checklist item 1 + 2)

### 1a. The headline, recomputed from the jsonl

| config (noisy) | mean | popCV | per-run costs |
|---|---|---|---|
| kb_system_gated | 4,849.3 | 0.243 | [3659, 4480, 6003, 4407, 6225, 5414, 3327, 3801, **7019**, 4158] |
| kb_pointer_verbal | 5,421.2 | 0.397 | [4265, 3969, 4252, 4767, 6639, **11522**, 4726, 4772, 4272, 5028] |
| order_up_to (floor) | 5,521.3 | 0.136 | [4635, 5996, 6397, 5634, 6681, 6211, 4327, 4922, 5375, 5035] |
| baseline | 5.63e11 | 2.62 | two runs ≥ 6.8e11 dominate; median 122.7M, min 3.9M, max 4.95e12 |

Paired per-pass deltas `kb_system_gated − floor`: **−976, −1516, −394, −1227, −456, −797, −1000, −1121, +1644, −877** → 9 wins / 10 losses-to-floor, mean diff **−672 (−12.2%)**.

Exact tests (n=10, paired):
- Sign test (two-sided binomial): **p = 0.0215**
- Exact permutation test on the mean delta (40k sign flips): **p = 0.0444**
- Exact Wilcoxon signed-rank (full sign-flip enumeration): **p = 0.0840**
- Paired bootstrap 95% CI of mean diff: **[−1,083, +251]**, 98.5% of resamples negative → **includes 0**

**Verdict: the 4,849 vs 5,521 difference is borderline.** Two of three tests pass 0.05; the classic non-parametric test for paired data (Wilcoxon) does not; the bootstrap CI brackets zero. The result is entirely carried by the one +1,644 counterrun (pass 8) — drop it and p goes to ~0.001, but you can't drop it. At n=10, "the gated LLM beats the floor on noisy demand" is a **suggestive effect, not established**. (The task brief says "8/10" wins; the on-disk data gives **9/10** — beware of stale counts in the writeup.)

`kb_pointer_verbal` under noise: 8/10 wins, mean −100 (−1.8%), perm p=0.91 → **no effect at all**; the +85% pass (11,522) erases it.

### 1b. THE #1 CHECK — is the formula optimally tuned? **NO — a tuned floor beats the LLM**

Grid sweep of the order-up-to floor (θ ∈ {1.5…6} × λ ∈ {0.1…0.9}, 72 points, 10 seeded noisy demand paths, simulator validated above):

| policy | mean cost | CV | vs LLM 4,849 |
|---|---|---|---|
| **study floor (θ=3.0, λ=0.5)** | 5,521.3 | 0.136 | +13.9% worse |
| **best tuned floor (θ=3.5, λ=0.2)** | **4,701.1** | **0.092** | **−3.1% (beats the LLM)** |
| best at θ=3 (λ=0.3) | 4,749.5 | 0.096 | −2.1% |
| per-run ex-post best over grid | 4,544.0 | — | −6.3% |

In-sample tuning on the same 10 paths overfits slightly, so: **out-of-sample split-half check** (tune on 5 runs, score the other 5):

| split | tuned (θ, λ) | OOS mean |
|---|---|---|
| train {0–4}, test {5–9} | (3.5, 0.2) | 4,615.8 |
| train {5–9}, test {0–4} | (3.0, 0.3) | 4,950.2 |
| train evens, test odds | (3.5, 0.2) | 4,970.4 |

OOS tuned-floor mean ≈ **4,846 — statistical parity with the LLM (4,849)**; one split worse, two better.

**Claim C collapses as stated.** "Gated LLM beats the deterministic formula by ~12%" is true only against the *untuned* a-priori (3.0, 0.5). A 1-minute deterministic grid search reaches parity out-of-sample and beats it in-sample, with lower CV (0.092 vs 0.243). The honest statement is: *the LLM + wrappers beats one arbitrarily-chosen deterministic policy; it ties or loses to a tuned one.* The headline number is an artifact of benchmarking against an untuned hyperparameter point, exactly the failure AUDIT.md §1 flagged for fixed demand, repeated under noise.

### 1c. CV tradeoff

Against the study floor: yes, "LLM wins mean, formula wins consistency" (CV 0.243 vs 0.136) is the honest read of *these two configs*. Against the **tuned** floor the tradeoff disappears: (3.5, 0.2) has **both** a lower mean (4,701) and lower CV (0.092) than the LLM. The "LLM wins mean, formula wins consistency" framing quietly omits that the formula side never got its λ/θ tuned.

---

## 2. Is the noisy demand generator fair? (checklist item 3)

`make_demand` noisy (engine.py:95-107): `d_t = step(t) + round(prev_noise)`, `prev_noise = 0.5·prev_noise + U{−2,…,2}`, clipped [0,20].

Verified fair-process properties:
- **Seeds per-run and shared across configs within a pass** — verified: the runner calls `make_demand(36,"noisy",seed=1000+i)` per config in pass i (interleaved_runner.py:146-148), and the floor's on-disk per-run costs exactly equal my seeded reproduction → identical demand paths within each pass. ✓ Fair comparison.
- **Noise is mild.** Observed demand over the 10 paths: mean 6.7–8.2, sd ≈ 1.88, max |deviation from step| = 4, mean |dev| = 1.36. The [0,20] clip **never binds** (observed range 1–11). This is a 20%-CV wobble around the same step-4→8 structure — a tame robustness arm, not a stress test. The step still dominates (baseline chaos under noise is driven by the same amplification dynamics).
- **No structural tilt toward adaptive agents in the generator itself** — the noise is symmetric AR(1); nothing rewards "response to noise" per se. The tilt toward the LLM comes from information, not process (see §3).

---

## 3. Same-week information leak — systematic pro-LLM asymmetry (checklist item 10)

Still present and still in the prompt: `build_prompt`/`build_user_prompt` render `Incoming order from your customer this week: {incoming_now}` (agents.py:239, 323), and the engine sets `incoming_now` for wholesaler/distributor/factory to the **current-week order of the tier below**, computed moments earlier in the sequential decision loop (engine.py:132, 142) — i.e., a perfect one-week lookahead the deterministic floor does not have (`OrderUpToAgent` reads only `incoming_last`, agents.py:41-43). The floor can't see the value the leak grants; the LLM can.

Quantified with a deterministic "leaked floor" (same order-up-to formula, forecast advances on `incoming_now`):

| variant | best mean (θ,λ) | (3.0,0.5) mean |
|---|---|---|
| no leak (the floor as run) | 4,701.1 | 5,521.3 |
| retailer-only leak | 4,369.0 | 4,562.7 |
| upstream-only leak | 3,320.3 | 3,410.7 |
| **full leak** | **2,920.0** | **3,041.8** |

A deterministic formula given **the same information the LLM sees** beats the gated LLM by **37–40% — three times the size of the headline −12% effect** — even *untuned*. The upstream leak (tiers reading the tier-below's same-week order) carries ~3/4 of that gain.

**Verdict: the noisy comparison is not "LLM vs formula"; it is "LLM-with-lookahead vs formula-blind."** This is a second, independent reason claim C is confounded (first: untuned hyperparameters). Note this also degrades the fixed-demand comparison in the same direction, but the fixed step is perfectly predictable (arrives 4 weeks early), so the leak's *differential* value is small there and large under noise — precisely why noise "reveals" an LLM advantage that is really lookahead.

---

## 4. Verbal gate blowup (checklist item 4)

Noisy pass 5, `kb_pointer_verbal` = 11,522 vs floor 6,211 (+85%). Trace dig:
- Retailer: **15 of 21 decisions are format failures** → silent `_fallback(ctx)` = **mirror** (fallback default "mirror", agents.py:151-156; counted in `failures` but invisible unless you read the jsonl). The mirror is a pure chase policy; under noise it overshoots (raw orders 35, 36, 55, **95**, 72 made it through as valid parses).
- Backlog tower: retailer max 72 → wholesaler 77 → distributor 112 → factory 124.
- Factory t=8: order 0 with backlog → the **confidence gate** fired (`gated=True, gated_verbal=False`), replacing the order with the *stateful anchor's* forecast → **0 → 227**. Under a backlog tower the anchor's own state has inflated, so even the "safe" path amplifies once the state is poisoned. (The verbal gate — `gated_verbal` — fired 6 times elsewhere in this pass, e.g. distributor/factory overrides.)

Aggregate: noisy verbal failure rate **461/1,440 = 32%** (fixed: 496/1,440 = 34%) — a third of the config's decisions are mirror fallbacks, not gate overrides, not model decisions. The reported mean/CV for `kb_pointer_verbal` is an unlabeled ⅓-fallback mixture (the same integrity issue AUDIT.md §8.4 flagged, now quantified and *larger*). The higher CV (0.397 noisy / 0.101 fixed) is not a tradeoff, it is the failure-mode: one blowup pass per 10.

---

## 5. Claim A — "gating tames the agent" (checklist item 8, 10)

### 5a. Interleaving works for drift… mostly
Per-pass costs, fixed demand (rows = configs, cols = passes 0–9):

```
order_up_to           3681 3681 3681 … (CV 0)
kb_system_gated       3.4k 4.3k 4.2k 4.4k 4.0k 4.5k 3.8k 4.3k 4.3k 3.9k   CV 0.073
kb_pointer_verbal     4.2k 3.9k 5.0k 4.5k 5.2k 4.7k 3.9k 4.4k 4.4k 3.7k   CV 0.101
kb_pointer_gated      4.8k 4.8k 5.5k 5.5k 4.3k 4.5k 4.6k 6.1k 5.1k 4.3k   CV 0.116
kb_introspect_gated   4.1k 3.7k 5.7k 4.0k 4.9k 7.0k 4.4k 5.0k 5.4k 4.1k   CV 0.194
kb_system_introspect 75.1k 59.8k 52.8k 42.0k 54.0k 47.2k 48.8k 79.6k 48.0k 43.8k CV 0.221
kb_general           52.8k 66.8k 178.6k 43.6k 771k 117.7k 1046k 59.9k 113k 147k CV 1.28
baseline              7.8M 104M 23.6M 0.86M 92M 88M 3.8M 2.2M 0.99M 65M   CV 1.06
```

The gated configs are tight across passes (CV 0.07–0.19) and there is **no pass-level shared drift** among LLM configs (corr(system_gated, system_introspect) = −0.31, corr(kb_general, baseline) = +0.04). Day-scale window drift — the AUDIT.md #1 confound — is genuinely killed.

### 5b. But the mechanism claim does not survive the traces
`kb_system_gated` vs its byte-identical-twin `kb_system_introspect` (differ only in `conf_threshold=0.5` + `anchor_margin=6`): **4,113 vs 55,098 mean, no pass overlap** (max gated 4,464 < min introspect 42,968), ratio 9.6–21.9× per pass (mean 13.6×). This **reproduces the old between-window 16× difference under interleaving** — the config effect is real and stable.

But the named mechanisms are inert:
- Confidence gate fired **3 decisions total** in all 1,440 gated decisions (0.2%; runs 2/3/4, all distributors, weeks 5/9/5), noisy: 2.
- Anchor clamp changed **2 decisions** (order_used ≠ order). The raw orders were already within ±6 of the anchor everywhere.
- Trajectories diverge at week 1–2 on **byte-identical prompts**; the factory's week-0 order is 6 in 9/10 gated passes vs 16 in 7/10 introspect passes. **Permutation test of exchangeability of week-0 completions between the two configs: p = 0.00095.**

So the 13× "gated win" cannot be produced by 5 wrapper interventions; identical prompts draw non-exchangeable completions that are *config-correlated* (or position-in-pass-correlated — config positions within a pass are config-specific, e.g. system_introspect mean position 3.2 vs system_gated 4.2, and the noisy run shows the opposite position bias with the same direction of effect). Mechanism UNVERIFIED — endpoint serving bias, temperature-sampler autocorrelation, or something else; read-only I cannot probe it. What is verifiable: **the interleaved design's core assumption (exchangeable draws given identical prompts) is violated at the p≈0.001 level by this endpoint**, which means claim A's causal reading ("the gate protects against bad windows") is unsupported, and the residual risk is that the *ranking* of same-prompt configs is endpoint-correlated rather than config-driven.

What survives claim A intact: interleaved, `kb_system_gated` robustly beats every other LLM config on fixed demand (0/10 per-pass losses vs pointer_gated; 2/10 vs introspect_gated: passes 1 and 3, both by <600 cost), and its own CV is the tightest. The wrapper-only ablation says what that's worth: the pure anchor scores 3,681 — **the LLM adds +432 mean cost (+11.7%) on fixed demand**, it does not subtract. "Gating tames the agent" should read "the kb_system_gated *config* (and the deterministic anchor it contains) tames the agent; the gate itself fires 0.2% of the time."

---

## 6. Wrapper-only correctness (checklist item 5) — verified, but the claim's wording is wrong

- The `--wrapper-only` branch (interleaved_runner.py:127-132) replaces `decide` with `lambda ctx, _o=orig, _a=a: _a.anchor(ctx)` for every LLMAgent that has an anchor. **Not a no-op**: the wrapper run made **0 chat calls, 0 tokens, `prompt_hash_dups_calls=0`**, and every config's per-run cost is exactly 3,681 (50/50 runs) = the order-up-to floor. `decide` was truly replaced; the anchor produced the order. ✓
- What the ablation shows on fixed demand: anchor alone = 3,681; `kb_system_gated` (LLM + anchor + gate) = 4,113; `kb_pointer_gated` = 4,948; `kb_introspect_gated` = 4,841. So **the LLM plus the wrappers is 12–34% WORSE than the wrapper alone on fixed demand**. Claim B as worded ("the LLM contributes 0% beyond the anchor") is directionally wrong — the correct statement is *the LLM contributes cost, not value, on fixed demand; the anchored wrapper is the entire win, and it would win more without the LLM inside it*. (The variance dimension is different: the LLM configs' CVs are nonzero but small; on fixed demand consistency is nearly free because the demand is deterministic.)
- Caveat: the ablation replaces `decide` but leaves the gate/clamp code path untouched *because it is never reached* — exactly right for "what does the wrapper alone do", and the audit confirms it means what it says.

## 7. Interleaving mechanics, nonce, collisions (checklist items 6, 7)

- **Shuffle**: seeded `rng.shuffle(configs)` per pass (interleaved_runner.py:137-142). Reproduction with `random.Random(42)` matches the val2 and wrapper_only jsonls **exactly**; the noisy run used **seed 7** (only seed ≤30 that reproduces), not the default 42, and it is **undocumented** — no launch script for the interleaved runs exists in the repo (the .sh files on disk drive the *old* runner). Provenance gap: the exact CLI for all three sets is inferable only from the data.
- Positions vary per config per pass; every config runs exactly once per pass; run indices 0–9 for all sets; no duplicates, no appended contamination (truncate-on-start, interleaved_runner.py:110-113, plus run_uuid in tag) — verified: 80/80, 40/40, 50/50 records, zero error stubs.
- **Cache-buster**: injected into the last user message on every chat call (client.py:48-55) — live code, not dead: the per-trace `_nonce` field and `prompt_hash_dups_calls` (= 10,080 / 4,320 = exactly the number of chat calls) confirm. It defeats byte-exact prompt caching, as claimed. **But** `prompt_hash_dups_total = 0` is now *guaranteed by construction* (the fingerprint is taken *after* nonce injection, client.py:59-63), so the collision metric no longer measures anything; and a *semantic* cache keyed on content (nonce stripped) would still fire — endpoint behavior UNVERIFIED. The week-0-order-determinism evidence that motivated the nonce (27/30 draws of 6) is gone from this data (week-0 orders now vary, e.g. {4,6,8} across passes) — the nonce did its job.

## 8. Data integrity & accounting (checklist item 10)

Per-config counts: 10/10 everywhere; `failures` tallies per config (fixed): kb_pointer_verbal **496**, kb_pointer_gated **459**, kb_introspect_gated **193**, all others 0 (including kb_system_gated, kb_system_introspect, baseline, kb_general). Token sums from records exactly equal summary totals (val2: 8,526,271 + 449,108; noisy: 2,718,870 + 182,749). No accounting bug.

**Baseline under noise**: mean **5.63e11**, popCV 2.62, median 122.7M — this is *not* "10.6M" as the brief states; the on-disk summary and my recomputation agree (5.63e11). It is **~14,500× higher** than the fixed-demand baseline (38.8M), not lower: unstriated LLM agents under variance are catastrophically unstable, with two runs ≥ 6.8e11. The qualitative "LLM = chaos machine" claim is *stronger* under noise, and survives interleaving. CV 2.62 vs fixed 1.06 supports "variance makes the defect worse", but n=10 with two dominant runs makes the exact mean meaningless (report the median too).

**order_up_to CV under noise (0.136)**: pure demand variance — the floor is deterministic given the demand path, and each pass sees a different seeded path. Verified by simulation: re-running the deterministic policy on the 10 seeded paths byte-reproduces the observed costs, so the CV is entirely path-to-path variance, zero agent variance. Correct reading.

**Let-down check — claim A "gated vs ungated":** the "ungated" side of the 9.6–21.9× ratio is `kb_system_introspect`, which has *no* gate and *no* clamp but byte-identical prompts to `kb_system_gated`. Since the gate/clamp fired ≤5 times, that ratio measures the *config-prompt twin gap* (§5b), not "gating protects against bad windows".

---

## Top 5 remaining issues, ranked

1. **The noisy headline is an untuned-benchmark artifact (HIGH).** A seen-the-same-noise (θ,λ) sweep undershoots the LLM by 3% in-sample and ties it out-of-sample; a leaked variant beats it by 37%. Claim C should be withdrawn or rewritten as "untuned deterministic policy" vs "LLM wrapper" — and the leak removed (see fix 2) before any rerun.
2. **Same-week `incoming_now` leak still in the prompt (HIGH).** It systematically gifts the LLM (and only the LLM) current-week downstream information worth 37–40% under noise. Fix: drop `incoming_now` from non-retailer prompts entirely (or gate it behind a config flag) to match the paper's through-t-1 timing; re-run the noisy arm both ways to decompose lookahead vs policy quality.
3. **Non-exchangeable completions at byte-identical prompts (HIGH, open).** Week-0 draws for the gated vs introspect twins differ with p≈0.001 (9/10 draws of 6 vs 7/10 draws of 16); the 13× twin gap has no mechanistic explanation from the wrappers (5 interventions). Either the endpoint's serving correlates with config/position-in-pass — in which case *all* same-prompt config rankings in this study are suspect — or the model's completion distribution is bimodal with long autocorrelation. Audit action: increase passes (n=10 is thin), randomize more aggressively, log serving node/id if available, and re-run the twin pair as a deliberate A/B to close this.
4. **`kb_pointer_verbal` integrity: ~⅓ of decisions are silent mirror fallbacks (HIGH).** 496/1,440 (fixed), 461/1,440 (noisy); the blowup pass is a fallback cascade, and the reported mean/CV mix gate overrides with deterministic fallbacks. Fix: flag failure-fallback decisions in traces (`order_used` + `is_fallback`), report a fallback-free mean, and lower the parse-failure rate (better format enforcement or `parse_introspection` anchoring).
5. **n=10 + no multiple-comparison control (MEDIUM).** The headline fails Wilcoxon (p=0.084) and its bootstrap CI includes zero; per-pass sign/permutation pass 0.05 only barely. Every pairwise claim in the writeup should carry a paired test; with 12 configs across two arms, Holm-correct. Also: **document the interleaved runs' seeds** (42, 42, 7) and CLI in the repo — they currently exist only in the data.

---

## What survives this audit

- **Interleaving beats day-scale drift**: configs share passes, no pass-level shared shock among LLM configs, gated-config CVs are tight, ratios reproduce across all 10 passes. The AUDIT.md #1 confound is appropriately retired for the new sets.
- **The deterministic engine and floors are exact**: my seeded simulation byte-reproduces every on-disk floor cost (fixed and noisy), including CV 0.136 under noise = pure demand-path variance.
- **Wrapper-only = pure anchor, verified** (0 calls, 0 tokens, 3,681 exact): the anchored wrapper, not the LLM, is what contains cost on fixed demand — the LLM inside the wrapper costs +12–34% mean.
- **Data integrity of the new sets**: 80/80/50/40 complete, no stubs, truncation + run-UUID worked, token/usage accounting exact.
- **Qualitative instability finding, strengthened**: unstriated baseline under noise (median 122.7M, worst 4.95e12) is worse than the fixed arm, not better; "structure > raw LLM judgment on this endpoint" remains the right general direction.
- **`kb_system_gated` robustly ≤ every other LLM config on fixed demand** (interleaved) and beats the *untuned* floor under noise — but both the "gating" mechanism and the "beats the formula" headline are compromised as detailed above.

## 10-line summary

1. The noisy headline (−12.2%, 9/10 passes) is borderline at n=10: sign p=0.0215, permutation p=0.044, Wilcoxon p=0.084, bootstrap CI of the mean diff [−1,083, +251] includes zero.
2. The #1 check fails the claim: a 72-point (θ,λ) sweep of the same floor on the same seeded noise gives (3.5, 0.2) → 4,701 mean, below the LLM's 4,849 in-sample, ≈ parity out-of-sample (split-half OOS ≈ 4,846). The formula was untuned, not beaten.
3. The comparison is also information-unfair: the LLM prompt leaks the current week's downstream orders (`incoming_now`, engine.py:132/142 → agents.py:239/323); a deterministic floor given the same lookahead scores 2,920–3,042, 37–40% under the LLM.
4. Noise is mild (sd≈1.88, max dev 4, clip [0,20] never binds) but fair across configs: seeds 1000+i are per-pass and shared; the floor's noisy CV 0.136 is pure path variance (byte-reproduced).
5. Wrapper-only is verified real (0 calls, 0 tokens, 3,681 exact), but claim B is directionally wrong: the LLM config costs *more* than the wrapper alone on fixed demand (+11.7% for system_gated, up to +34% for pointer_gated).
6. The 9.6–21.9× "gating" ratio is real and interleaved-stable, but the gate fired 3× and the clamp bound 2× in 1,440 decisions — the mechanism is unexplained, and identical prompts draw config-correlated completions (week-0 factory order 6 vs 16, exchangeability p≈0.001): residual endpoint/config correlation is the open confound.
7. The verbal gate's noisy blowup (11,522) is a fallback cascade: 32–34% of its decisions are silent mirror fallbacks, and one bad pass eats its entire edge (noisy mean vs floor p=0.91).
8. Baseline under noise is 5.63e11 mean / 122.7M median (the "10.6M" in the brief is wrong) — worse chaos than fixed, strengthening the qualitative instability claim.
9. Interleaving fairness, data integrity, truncation, run-UUIDs, token accounting, KB snapshots, and cache-buster injection all verify clean; `prompt_hash_dups_total=0` is now vacuously guaranteed by the nonce rather than evidence.
10. What stands: interleaved config rankings for the gated family, wrapper-only = floor, "LLM = chaos without structure", and the engine/floors — all numeric claims about the LLM's *positive* contribution under fixed or noisy demand must be re-run with the leak removed, the floor tuned, and n > 10 before they are publishable.