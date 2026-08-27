# Agent Bullwhip — LLM Agents in the Beer Game, Audited

Replication + extension of **arXiv:2605.17036**, *Reliability and Effectiveness of Autonomous AI Agents in Supply Chain Management* (Long, Simchi-Levi, Zhu, Su, Calmon & Calmon; Harvard/MIT/Georgia Tech). LLM agents manage a four-echelon MIT Beer Game. We test whether cheap inference-time structure tames the paper's "agent bullwhip" without its GRPO RL post-training. We then audited our own results three times. Each audit retired a headline. The surviving claim is below, with what died and why.

Authors of the audits are the authors of this repository: the mistakes are ours, documented in `AUDIT.md`, `AUDIT2.md`, `AUDIT3.md`.

---

## TL;DR — the surviving claim, precisely

> **A textbook inventory formula that tunes itself from 18 weeks of history beats the LLM agent in every environment — mild, chaotic, and wild demand — at roughly 3–4 orders of magnitude less compute. Two of the LLM's wins were information leakage and an untuned baseline. In the configurations that actually work, the deterministic wrapper — not the model's reasoning — carries the cost.**

What each contested word means, in this repo:

- **"the LLM agent"** = deepseek-v4-flash on a free API endpoint, wired into the paper's own agent framework. We are not claiming anything about GRPO-tuned frontier models. The paper's reliability finding *does* replicate on this endpoint: a plain LLM with no structure is an instability machine (baseline means 38.8M fixed / medians 281M–394M under noise and chaos, blowups to ~10^194 in the wild arm; paper reported CV 0.13–0.46 for the same *agent*, ours is an order of magnitude worse).
- **"1970s"** = order-up-to with exponential smoothing, the standard inventory policy class the paper itself uses as its deterministic contrast. Nothing new.
- **"tunes itself"** = a deterministic 72-point (θ,λ) grid search on the first 18 weeks of each demand path, deployed frozen. No oracle, no future knowledge, no API calls.
- **"3–4 orders of magnitude less compute"** = one LLM arm (10 passes × 4 configs) burns ~2.73M prompt tokens + ~177K completion tokens across 4,320 API calls, ~1 hour of wall-clock per environment (measured in the summaries). The complete walk-forward + oracle computation for **all three** environments is ~4,400 deterministic simulations, ~3 CPU-seconds, $0. We measure "compute" as API tokens + wall time; by either, the formula side is orders of magnitude cheaper.
- **"the deterministic wrapper carries the cost"** = wrapper-only ablation (LLM decision replaced by the policy, zero chat calls): exactly **3,681**, identical to the policy alone. The LLM *inside* the wrapper **adds** +12–34% cost on fixed demand; it does not subtract. On noisy/chaotic/wild demand the LLM adds even more, and the tuned formula beats it outright (below).

## What died (the dead claims)

1. ~~"Gated LLM is within 11% of optimal."~~ Three independent problems: the sequential run was window-confounded; the "floor" wasn't optimal (other (θ,λ) settings score 3,148–3,206 vs 3,681); and the named gate fired 0.2% of decisions — nothing was being gated.
2. ~~"The LLM beats the formula under noise."~~ The LLM saw a same-week information leak (`incoming_now`) the formula didn't; a formula with the same lookahead beat it by 37–40%. And the "formula" was an untuned a-priori point — a 1-minute sweep beat the LLM once information was equal.
3. ~~"The crossover: the LLM catches up as chaos rises."~~ The chaotic/wild arms ran unseeded — every config played a *different* demand path. The pairing the crossover depended on didn't exist.
4. ~~"The reliability metric is something the LLM wins on."~~ We only report cost here; coefficient-of-variation columns appear in the table below and the formula wins those too wherever paths are comparable.

## The surviving result

All numbers: **10 passes × 4 configs, interleaved, seeded, leak-free, 40/40 runs completed (0 API failures), mean total supply-chain cost** (holding $1, backlog $2 per unit-week). Cost is lower-is-better. CV = coefficient of variation across the 10 runs.

| environment | verbal gate (LLM) | gated KB (LLM) | untuned floor (3.0, 0.5) | walk-forward floor | oracle floor | on-disk source |
|---|---|---|---|---|---|---|
| **noisy** | 5,424 | 5,973 | 5,521 | **5,214** | 4,544 | `results/noisy_leakfree/` |
| **chaotic** | 9,156 | 11,525 | 12,659 | **7,678** | 6,926 | `results/chaotic_seeded/` |
| **wild** | 13,335 | 15,849 | 16,159 | **12,111** | 9,821 | `results/wild_seeded/` |

| CV (within-column) | verbal | gated | untuned | walk-forward |
|---|---|---|---|---|
| noisy | 0.199 | 0.203 | 0.136 | 0.131 |
| chaotic | 0.435 | 0.377 | 0.380 | 0.339 |
| wild | 0.429 | 0.406 | 0.437 | 0.603 |

Reading this table like a skeptic:

- **The noisy row is the weakest evidence.** Walk-forward beats the best LLM config by 210 cost (5,214 vs 5,424, −3.9%) at n=10, and the columns come from *different seeded paths* (walk-forward seeds 1000–1009; LLM arm seeds set per run), so no paired test is possible across columns from the on-disk summaries. Treat noisy as consistent-but-weak. AUDIT3's demand-matched reconstruction — floors scored on the identical demand paths the LLM actually played — is the paired evidence, and it agrees: tuned floor beats the gated LLM on chaotic (7,175 vs 11,055, permutation p=0.015, 9/10 passes) and wild (18,042 vs 28,054, p=0.004).
- **Chaotic and wild are the decisive rows:** walk-forward wins by 16% and 9% over the best LLM config respectively, with lower CV in chaotic and similar CV in wild.
- **The LLM's *only* wins are against the untuned point** — the verbal gate beats (3.0, 0.5) in all three environments (5,424 < 5,521; 9,156 < 12,659; 13,335 < 16,159). What beats the LLM is not "a formula" — it's *tuned* parameters.
- Values are **means**; spread is large. Walk-forward range by environment: noisy 4,025–6,246; chaotic 3,848–12,545; wild 4,183–26,200 (CV 0.60 — the wild arm is high-variance even for the formula). The verbal gate ranges: 3,943–7,104 / 4,443–18,061 / 6,265–23,304.

### Fixed demand: the cleanest result in the repo

Deterministic demand step: the policy scores **3,681, CV 0.000**. The wrapper-only ablation — LLM decision swapped for the policy, zero chat calls, zero tokens — scores **exactly 3,681**. The LLM inside the wrapper adds **+12–34%** over the wrapper alone. That is the whole story in miniature: the wrapper is the entire win, and the model riding inside costs.

Two caveats, stated plainly: 3,681 is the a-priori point (3.0, 0.5), not the optimal policy — a tuned (3.0, 0.35) scores 3,148. And 3,681 is *the same number* for wrapper and no-wrapper because the anchor is deterministic; "CV 0.000" is a deterministic engine, not a reliability win.

### The baseline: plain LLM, no structure

| environment | statistic | source |
|---|---|---|
| fixed (step) | mean 38.8M, CV 1.06 | interleaved fixed-demand arm |
| noisy | median 281M | leak-free arm |
| chaotic | median 394M | leak-free arm |
| wild | median 11.6M; blowups to ~1.8×10^194 | leak-free arm |

Medians, because a handful of blowup runs make means meaningless (chaotic means ran to ~10^21). The blowups are certified by AUDIT3 as real instability, not seed pathology: the **seeded** re-run shows chaotic baseline mean 1.8×10^12 and wild mean 21.1M (max 91M). "LLM without structure = instability machine" holds under every protocol we ran; the wild median alone is ~1,000× the walk-forward floor.

### The verbal gate — best of the two LLM configs, and it's a mixture

`kb_pointer_verbal` fires when the model **orders 0 while holding backlog while its stated reasoning says "cover the backlog"** — the *confidently wrong* failure mode — and overrides to the order-up-to anchor. It fires on ~3–4% of decisions and is the cheapest LLM config in every environment.

The honest label is *mixture, not pure LLM*: **21–29% of its decisions are silent mirror fallbacks** — parse or format failures defaulting to `mirror` (order = last incoming), per `agents.py` (`fallback: str = "mirror"`). That is a different policy from the anchor, and it's the config's real weakness (it can cascade: AUDIT2 documents an 11,522 mirror-cascade blowup). A keyword regex triggers the gate, so the firing rate is a heuristic property, not evidence about model reasoning.

### What the paper's fix would look like, and why we didn't do it

The paper's remedy (GRPO RL post-training) is expensive, model-specific, and out of reach for most teams. We test cheaper structure instead — and the structure had to survive being audited. The paper's *finding* replicates; its *mechanism* (that the agent can be taught reliability) is one we cannot test on this endpoint, and we make no claim about it.

---

## The story — why the table took three audits to earn

Each section is tagged with the protocol it came from. Anything pre-fix is dead protocol — kept because the audit trail is the point.

### 1. The naive win *(sequential, leak on, untuned floor, n=30)*

`kb_system_gated`: **4,086, CV 0.089**, "within 11% of optimal". Exciting headline. One hour of auditing killed it.

### 2. Audit #1 — window drift *(same protocol)*

Identical prompts one hour apart gave results differing by **16×**. The sequential matrix compared configs across different serving-load windows; endpoint non-stationarity, not agent quality, produced the ranking.

**Fix:** an interleaved round-robin runner — every config plays the same window, one pass at a time. This killed day-scale ranking noise, which is all we claimed it did. Audit #2 later showed the *twin gap itself* — `kb_system_gated` 4,113 vs its byte-identical-untwin `kb_system_introspect` 55,098, interleaved, no pass overlap — is **stable**, and the named gate fires 0.2% of the time. We do not have a mechanism for that. It is the open confound (config-correlated completions, exchangeability p≈0.001), not a window artifact.

### 3. Audit #2 — the leak and the untuned baseline *(interleaved, leak on, untuned floor)*

The claim "the LLM beats the formula under noise" died twice, independently:

**(a) The leak.** Upstream tiers saw `incoming_now` — the tier-below's *same-week* order, computed moments earlier in the sequential loop. One-week lookahead. A deterministic formula handed the same lookahead scored **2,920–3,042 — 37–40% under the LLM — three times the size of the headline effect**. The formula was fighting one-armed.

**Fix:** upstream tiers see only last week's information; verified 0 leaked weeks across 5,752 upstream traces (all leak-free arms). Residual: **the retailer still sees current-week customer demand** — AUDIT3 quantified it at ~7% of floor cost (4,701 → 4,369), it favors the lookahead holder, and the walk-forward floor still wins with it in place. Not small; documented.

**(b) The untuned baseline.** The comparison point was the a-priori (3.0, 0.5), never tuned. A 72-point sweep takes ~1 minute.

**Fix:** tuned floors became the honest baseline — and they beat the LLM.

### 4. Audit #3 — the seeding bug *(chaotic/wild arms, unseeded)*

The crossover experiment — "the LLM catches up as chaos rises" — ran **unseeded**. Each config in each pass drew independent random demand. The interleaving control was silently destroyed, per-pass pairing meant nothing, and the tuned baselines had been tuned on demand the LLM never saw. The crossover was an artifact of unshared randomness.

**Fix:** every pattern seeded, one shared demand per pass across all configs, regression tests locking the contract. 48/48 tests pass — unit tests, which bless the code, not the 10-pass means.

### 5. The walk-forward objection

"You tuned the formula with hindsight!" Fair. So the formula now tunes itself: grid-search (θ,λ) on the **first 18 weeks only**, deploy frozen. It still beats the LLM in every environment. The protocol, exactly as implemented in `walkforward_floor.py`, is described below — including the part we don't paper over: the reported walk-forward cost is scored over the **full 36-week game**, so half of it is in-sample. We checked the unseen half separately; the verdict survives (below).

---

## The walk-forward protocol, exactly as run

1. Simulate the first 18 weeks of a seeded demand path.
2. Grid-search (θ, λ) ∈ {1.5…6.0} × {0.1…0.9} — 72 points, deterministic, ~1 minute of CPU — choosing the (θ,λ) with lowest cost on those 18 weeks.
3. Deploy those parameters frozen for the rest of the game.

**Honest scoring note.** `walkforward_floor.py` tunes on weeks 1–18, then scores the **full 36-week game** — the headline walk-forward cost includes the tuning half. That is in-sample optimism on half the reported number. We did not hide it; we measured it. The unseen-half-only check (fresh 18-week game on weeks 19–36):

| environment | second-half, walk-forward | second-half, untuned (3.0, 0.5) | walk-forward wins |
|---|---|---|---|
| noisy | 4,540 | 5,918 | 10/10 passes |
| chaotic | 5,954 | 8,675 | 9/10 |
| wild | 8,978 | 9,558 | 7/10 |

(These are half-length games; don't compare them to the full-game numbers above.) The tuning is not just in-sample overfitting: on the truly unseen half, the self-tuned policy beats the untuned point in 26 of 30 passes.

**Headroom.** Walk-forward sits 11–23% above the oracle (full-path-tuned floor: 4,544 / 6,926 / 9,821); the LLM configs sit 19–66% above it. Tuning headroom is real (6–39% over untuned) but bounded; the LLM's distance to the floor is not closed by tuning at all.

---

## Audit history (brief)

| What died | Cause | Fix / status |
|---|---|---|
| "within 11% of optimal" | window drift (16× between identical prompts) + floor not actually optimal + gate inert (0.2%) | interleaved runner; disclosed floor; open confound: twin gap stable, mechanism unknown |
| "LLM beats formula under noise" | `incoming_now` leak (1-week lookahead; leaked floor −37–40%) + untuned baseline | upstream sees t−1 only; 0/5,752 leaked weeks; tuned-floor baselines |
| "LLM catches up as chaos rises" | chaotic/wild arms unseeded — different demand per config | seed every pattern, one demand per pass, regression tests |
| (residual) | retailer still sees demand[t] | documented, ~7% of floor cost, favors LLM; floor still wins |
| (disclosure) | 3,681 called "floor" isn't optimal | 3,148–3,206 settings exist; wrapper-only = 3,681 exactly |

Full text: `AUDIT.md` (drift, circular baselines), `AUDIT2.md` (leak, untuned floor, inert gate, mirror cascades), `AUDIT3.md` (seeding, demand reconstruction, strawman-formula checks, residual quantification).

---

## Method

- **Engine:** deterministic 4-echelon beer game per the paper's §5.1 operational model; canonical beer-game cost constants (holding 1, backlog 2 per unit-week; the paper's exact constants are private).
- **Information set, one box:** retailer sees current-week customer demand; every upstream tier sees last week's downstream order only; the formula sees the same `incoming_last` any tier sees. This is the leak-free contract the tests enforce.
- **Agents:** deepseek-v4-flash (free endpoint) in four roles; wrappers include knowledge-base playbooks, introspection + confidence traces, low-confidence gating to the order-up-to anchor, and the verbal-consistency gate. Cache-buster nonce on every call (defeats serving-level caching).
- **Runner:** interleaved round-robin (same window, same seeded demand per pass); every run truncated-on-start with a run UUID; KB text snapshotted per run; full per-call traces.
- **Deterministic baselines:** `mirror` (order = incoming) and `order_up_to` (order-up-to with exponential smoothing; parameters θ, λ — θ is the safety-stock multiplier, λ the smoothing weight).
- **Metrics:** total cost (headline), CV across runs, Ψ/Φ bullwhip amplification, p95/p99 backlog, instruction-failure rate. The prompt-hash collision metric is omitted: the nonce is injected before the hash, so zero collisions are guaranteed by construction — it measured nothing.
- **Wave 1 configs** (voting5, budget, prompt_weighted, anchor, combined) were run under the pre-audit sequential protocol only; they are not part of the surviving table. Tombstoned, not vanished.

## Reproduce

Order: the surviving protocol first.

```bash
# env: FREEINFERENCE_API_KEY set (model defaults to deepseek-v4-flash)
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

# interleaved, seeded pass matching the noisy arm (the audit-surviving protocol):
.venv/bin/python -m agent_bullwhip.interleaved_runner \
  --configs kb_pointer_verbal,kb_system_gated,order_up_to,baseline \
  --runs 10 --horizon 36 --pattern noisy --seed 7 --outdir results/noisy_repro

# same for chaotic / wild (seeded; one demand per pass)
# walk-forward floor (no LLM, no API) — reproduces the walk-forward + oracle columns:
.venv/bin/python walkforward_floor.py

# tuned-floor sweep:
.venv/bin/python tuned_floor.py

# audit regression suite: 48 tests
.venv/bin/python -m pytest tests/ -q
```

Do **not** use the sequential runner (`agent_bullwhip.runner`) for claims — that is the protocol Audit #1 deleted. It exists for smoke tests and archaeology.

Data: `results/noisy_leakfree/`, `results/chaotic_seeded/`, `results/wild_seeded/` are the surviving sets (their filenames match the table). Earlier directories (`interleaved_val2`, `wrapper_only`, `chaotic_leakfree`, `wild_leakfree`, the flat `*.summary.json` files) are archaeology — the audits explain exactly which parts of them are compromised.

## Limitations (the ones we could not fix, sorted by how much they can hurt)

1. **n=10, no paired test across columns.** The LLM arms and the walk-forward script use different seeds, so the surviving table cannot be significance-tested column-to-column. AUDIT3's demand-matched reconstruction (identical paths) is the paired test that exists, and it agrees — but for noisy it is weak (see table notes).
2. **Walk-forward headline cost includes its tuning half.** The second-half-only check survives (26/30 passes), but the headline number is not a clean OOS statistic.
3. **Retailer lookahead residual, ~7% of floor cost, pro-lookahead.** Upstream is clean; the last tier isn't. The floor wins with the residual *in the LLM's favor*; closing it should widen the gap.
4. **Verbal column is a mixture.** 21–29% mirror fallbacks + 3–4% gate overrides + keyword regex = a heuristic, not a pure measure of model behavior.
5. **One cheap model, one endpoint.** deepseek-v4-flash on a free API. GRPO-tuned frontier models are untested; the paper's human-comparison headline (agents beat human teams by up to 67%) is not replicated because we have no human baseline.
6. **Wild is sample-brittle.** Tuned floors swung 11k–22k across demand samples in AUDIT3; the wild row should be read as range, not point.
7. **The twin gap has no mechanism.** Gated-vs-ungated twin configs differ 10–20× with the gate firing 0.2% of the time. We know it is stable, not why.

## Links

- Paper: https://arxiv.org/abs/2605.17036 (DOI 10.48550/arXiv.2605.17036)
- Audits: `AUDIT.md` · `AUDIT2.md` · `AUDIT3.md` — the mistakes, in full
- Change log: `CHANGES.md`
- Surviving data: `results/noisy_leakfree/` · `results/chaotic_seeded/` · `results/wild_seeded/`
- Social drafts (unpublished): `POSTS.md`