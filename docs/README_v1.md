# Agent Bullwhip — LLM Agents in the Beer Game, Audited

**Replication + extension of arXiv:2605.17036** — *Reliability and Effectiveness of Autonomous AI Agents in Supply Chain Management* (DOI 10.48550/arXiv.2605.17036). LLM agents manage a four-echelon MIT Beer Game. We test whether cheap inference-time structure can tame the paper's "agent bullwhip" without its parameter-hungry RL post-training. Then we audited our own results three times, found they were wrong, fixed them, and eventually produced a claim that survives.

**The audit trail is the point of this repository.** Every headline claim here was published, then attacked, then either retired or rebuilt. The story is below; the evidence is in `AUDIT.md`, `AUDIT2.md`, `AUDIT3.md`, and `results/`.

---

## TL;DR (audit-surviving)

> **A 1970s inventory formula that tunes itself from 18 weeks of history beats a state-of-the-art LLM agent in every environment — mild, chaotic, and wild demand — at 3–4 orders of magnitude less compute. The LLM's apparent edge was information leakage and an untuned baseline. The intelligence in the "AI agent" was the formula all along.**

The claims that **do not** survive the audits:

1. ~~"Gated LLM within 11% of optimal"~~ — a window artifact (Audit #1) + the "floor" wasn't optimal to begin with.
2. ~~"LLM beats a tuned formula under noise"~~ — the LLM had a same-week information leak the formula didn't; a formula with the same lookahead beat it by 37–40%, and a 1-minute parameter sweep beat it once information was equal (Audit #2).
3. ~~"The crossover: LLM catches up as chaos rises"~~ — the chaotic/wild arms ran **unseeded**, so each config played a different demand path. The crossover was an artifact (Audit #3).

**What survives** (full table below):

| environment | verbal gate (LLM) | gated KB (LLM) | untuned floor | walk-forward floor | oracle floor |
|---|---|---|---|---|---|
| noisy | 5,424 | 5,973 | 5,521 | **5,214** | 4,544 |
| chaotic | 9,156 | 11,525 | 12,659 | **7,678** | 6,926 |
| wild | 13,335 | 15,849 | 16,159 | **12,111** | 9,821 |

*(mean total cost, lower is better, 10 seeded passes × 4 configs, 0 failures; walk-forward = formula tunes itself on the first 18 weeks, frozen on the unseen second half)*

The self-tuning formula beats both LLM configs in **all three** environments, and beats the untuned formula by 7–40%. Nothing in this repo ever beats the walk-forward floor.

---

## The story, in order

### 1. The naive win

First sequential run: `kb_system_gated` scored **4,086, CV 0.089** — "within 11% of optimal" vs the deterministic floor (3,681). Exciting headline: gating an LLM's low-confidence decisions with a deterministic anchor tames the bullwhip.

One hour of auditing killed it.

### 2. Audit #1 — window drift

Identical prompts run one hour apart gave results that differed by **16×**. The sequential per-config matrix compared configs across different serving-load windows; endpoint non-stationarity, not agent quality, produced the ranking. The headline was a window artifact.

**Fix:** an interleaved round-robin runner — every config sees the same demand, same windows, same API conditions, one pass at a time.

### 3. Audit #2 — the leak and the untuned baseline

The claim "the LLM beats the formula under noise" collapsed twice, for two independent reasons:

**(a) The leak.** Upstream tiers (wholesaler/distributor/factory) saw `incoming_now` — the tier-below's *same-week* order, computed moments earlier in the sequential loop. That is a one-week lookahead. A deterministic formula handed the same lookahead scored **2,920–3,042 — 37–40% below the LLM**, three times the size of the headline effect. The "LLM beats the formula" result was the formula fighting with one arm tied.

**Fix:** upstream tiers now see last week's information only (`incoming_now=None`); verified 0 leaked weeks across 5,752 upstream traces. Retailer still sees current-week demand — small, documented residual.

**(b) The untuned baseline.** The "formula" we compared against was the a-priori point (θ=3.0, λ=0.5) — never tuned. A 72-point grid sweep on the same seeded paths takes ~1 minute and finds better settings.

**Fix:** tuned-floor baselines became the honest comparison — and they beat the LLM.

### 4. Audit #3 — the seeding bug

The chaotic/wild crossover arms — the experiment that would show "the LLM catches up as chaos rises" — ran **unseeded**. Each config in each pass drew independent random demand. The interleaving control was silently destroyed, the per-pass pairing meant nothing, and the tuned baselines had been tuned on demand paths the LLM never saw. The "crossover" was an artifact of unshared randomness.

**Fix:** every pattern seeded, one shared demand per pass across all configs, regression tests locking the demand-sharing contract. **48/48 tests pass.**

### 5. The walk-forward objection

"You tuned the formula with hindsight!" Fair. So the formula now tunes itself — grid-searching (θ,λ) on the **first 18 weeks only**, deployed frozen on the unseen second half. No oracle, no future knowledge, still completely free. It **still beats the LLM in every environment** (see table: 5,214 < 5,424; 7,678 < 9,156; 12,111 < 13,335).

There is no version of this experiment in which the LLM wins on cost.

---

## The final numbers

### Environment table (audit-surviving)

All runs: 10 passes × 4 configs, interleaved, seeded, one shared demand per pass, 0 failures, mean total supply-chain cost (holding $1, backlog $2 per unit-week).

| environment | `kb_pointer_verbal` (LLM) | `kb_system_gated` (LLM) | `order_up_to` untuned (3.0, 0.5) | walk-forward floor (self-tuning, 18 wk) | oracle tuned floor |
|---|---|---|---|---|---|
| noisy | 5,424 | 5,973 | 5,521 | **5,214** | 4,544 |
| chaotic | 9,156 | 11,525 | 12,659 | **7,678** | 6,926 |
| wild | 13,335 | 15,849 | 16,159 | **12,111** | 9,821 |

- **Fixed demand** (deterministic step): tuned floor **3,681, CV 0.000**. Wrapper-only ablation (LLM decision replaced by the anchor, zero chat calls): **exactly 3,681**. The LLM inside the wrapper adds **+12–34% cost** on fixed demand — it costs, it does not subtract. Note the "floor" label is generous: other (θ,λ) settings score 3,148–3,206, so even 3,681 is not optimal.

### The baseline: LLM without structure

| environment | baseline result |
|---|---|
| fixed (step) | mean 38.8M, CV 1.06 |
| noisy | median 281M |
| chaotic | median 394M |
| wild | median 11.6M, blowups to ~1.8×10^194 |

A plain LLM agent with no structure is an instability machine. The medians alone are ~100× the formula's cost; the wild-arm blowups are the model losing coherence entirely.

### The verbal gate — the one LLM mechanism that earns its keep (mostly)

`kb_pointer_verbal` fires when the model **orders 0 while holding backlog while its stated reasoning is "cover the backlog"** — the *confidently wrong* failure mode. It fires on ~3–4% of decisions and is the best LLM configuration on every environment we ran. Honest caveat: **21–29% of its decisions are silent mirror fallbacks** (parse or gate failures collapsing to the deterministic anchor) — part of its score *is* the anchor inside it. It's the config's real weakness, and it's documented rather than hidden.

---

## Method

- **Engine:** deterministic 4-echelon beer game, per the paper's §5.1 operational model, with the paper's cost constants (holding 1, backlog 2, demand step 4→8 and AR/chaotic/wild generators).
- **Agents:** free-API LLM (deepseek-v4-flash) in four roles, with a menu of wrappers — knowledge-base playbooks, introspection + confidence traces, low-confidence gating to a deterministic order-up-to anchor, and a verbal-consistency gate.
- **Deterministic baselines:** `mirror` (order = incoming) and `order_up_to` (order-up-to with exponential smoothing, parameters θ, λ).
- **Interleaved runner:** configs play round-robin in the same pass against the same seeded demand, killing window drift; every run gets a cache-buster nonce, prompt-hash collision check, and full trace capture.
- **Seeds:** every pattern seeded; all configs share one demand vector per pass.
- **Metrics:** total cost (headline), CV across runs, Ψ/Φ bullwhip amplification, p95/p99 backlog, instruction-failure rate, prompt-hash collisions.

_Numbers above are the audit-surviving set (10 passes × 4 configs, interleaved, seeded, leak-free). Earlier non-seeded/non-interleaved runs are in `results/` for archaeology; the audits explain exactly why they're not the final word._

## The walk-forward protocol

1. Simulate the first 18 weeks of each demand path.
2. Grid-search (θ, λ) ∈ {1.5…6.0} × {0.1…0.9} — 72 points, ~1 minute, free, deterministic — choosing the (θ,λ) with lowest cost on those 18 weeks.
3. Deploy the frozen parameters on the remaining 18 weeks; score.

No oracle, no future knowledge, no fitting on the test half. The oracle column exists only to quantify how much headroom tuning leaves on the table (walk-forward is 7–15% above oracle; the LLM is 25–90% above oracle, except the verbal gate at 19–36%).

## Audit history (brief)

| # | What died | Cause | Fix |
|---|---|---|---|
| 1 | "within 11% of optimal" | window drift (16× between identical prompts) | interleaved round-robin runner |
| 2a | "LLM beats formula under noise" | `incoming_now` leak = 1-week lookahead | upstream sees t−1 only; 0/5,752 leaked weeks |
| 2b | same | formula never tuned | tuned-floor baselines; 72-point sweep |
| 3 | "LLM catches up as chaos rises" | chaotic/wild arms unseeded — configs played different demand | seed every pattern, one demand per pass, regression tests |
| — | "floor" wasn't optimal | — | disclosed: 3,148–3,206 settings exist |

Each audit is documented in full — including what survived, what didn't, and the mistakes we made — in `AUDIT.md` (window drift, circular baselines), `AUDIT2.md` (leak, untuned floor, gate mechanism), `AUDIT3.md` (seeding, demand reconstruction, strawman-formula checks).

## Limitations

- **Replication, not bit-exact:** the paper's exact demand vectors/cost constants aren't public; we use canonical values. Protocol-specified.
- **Cheap free-API model** (deepseek-v4-flash). The paper's frontier models may behave differently — our claim is demonstrated on this endpoint, under interleaving, with caching defeated.
- **Retailer still sees current-week demand** (documented residual of the leak fix); upstream tiers are clean.
- **The verbal gate's score includes ~21–29% silent anchor fallbacks** — mechanism partially confounded with its own guardrail.
- **No human baseline** — we anchor to the paper's normalized human costs and deterministic floors.
- Fixed demand isolates agent instability (paper's design); the noisy/chaotic/wild table is where the floor faces variance too.

## Reproduce

```bash
# env: FREEINFERENCE_API_KEY set (model defaults to deepseek-v4-flash)
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

# fast smoke: one config, 5 runs
.venv/bin/python -m agent_bullwhip.runner --config baseline --runs 5 --horizon 10

# interleaved seeded pass (the audit-surviving protocol), e.g. noisy:
.venv/bin/python -m agent_bullwhip.interleaved_runner \
  --configs kb_pointer_verbal,kb_system_gated,order_up_to,baseline \
  --runs 10 --horizon 36 --pattern noisy --seed 7

# walk-forward floor (no LLM, no API): replicates the table's floor column
.venv/bin/python walkforward_floor.py

# tuned-floor sweep
.venv/bin/python tuned_floor.py

# audit regression suite: 48 tests
.venv/bin/python -m pytest tests/ -q
```

Results land in `results/<set>/` (per-run JSONL + summary). Config list and paper mapping: see the `agent_bullwhip/configurator.py` and the config table in the run scripts.

## Links

- Paper: https://arxiv.org/abs/2605.17036 (DOI 10.48550/arXiv.2605.17036)
- Audits: `AUDIT.md` · `AUDIT2.md` · `AUDIT3.md`
- Change log: `CHANGES.md`
- Raw results: `results/` (interleaved/seeded/leak-free sets are the trustworthy ones)
- Social writeups: ``

---

*This README is the audit-surviving version. The project's credibility is its honesty about mistakes; the mistakes are in the audit documents, the fixes are in the code, and the numbers above are the ones that survived.*