# Confound & Edge-Case Audit #3 — Crossover Experiment (chaotic + wild demand)

Auditor: skeptical re-derivation from source + on-disk results. Read-only; simulations in /tmp.
All numbers below recomputed from `results/{noisy,chaotic,wild}_leakfree/` on 2026-08-26 ~22:30 UTC plus
local sims (scripts in /tmp/audit3_*.py). Result sets: `noisy_leakfree` (10 passes × 4 configs, seeded),
`chaotic_leakfree` and `wild_leakfree` (10 passes × 4 configs, **unseeded — see §1**).

---

## Executive summary

**The crossover experiment is invalid as run and the crossover does not exist.**

1. **The #1 finding of this audit: chaotic/wild ran on unseeded, per-config demand.** `interleaved_runner.py:113-115`
   calls `make_demand(args.horizon, args.pattern)` with **no seed** for `chaotic`/`wild` (seeding exists only for
   `noisy`). Each config in each pass drew an independent entropy-seeded demand. Verified from the traces:
   within a pass, `kb_pointer_verbal` and `kb_system_gated` saw **different demand in 31–36 of 36 weeks**
   (value mismatches on ~24 shared weeks after excluding format-failure weeks), and **0 of 10** runs matched the
   seeded reproduction (1000+i) that `tuned_floor_chaos.py` tuned the "fair baseline" on. The on-disk chaotic
   floor costs are statistically indistinguishable from the *untuned* floor's cost distribution (KS p = 0.954) —
   they are 10 unseeded draws, not the seeded benchmark paths. Consequences: (a) the interleaving control is
   destroyed — there is no shared environment per pass; (b) every per-pass "paired" comparison in the brief's
   plan is impossible from the seeds; (c) the tuned floors (7,420 / 11,002) were tuned on demand paths the
   LLM **never saw**.
2. **Demand-matched (rescued) tests invert the headline.** Reconstructing each pass's demand from the
   `kb_system_gated` traces (36/36 weeks present in every run) and scoring floors/policies on the *same*
   demand: the **tuned floor beats the gated LLM in both arms, significantly** — chaotic in-sample (4.0,0.1)
   = 7,175 vs 11,055 (paired perm p = 0.015, 9/10 per-pass losses), LOOCV 7,611 (p = 0.019); wild in-sample
   18,042 vs 28,054 (perm p = 0.004), LOOCV 19,614 (p = 0.004). Even the **untuned** floor ties the gated LLM
   on matched demand (chaotic diff −15, p ≈ 1; wild the floor wins by 812, p = 0.50).
3. **The OOS tuned-floor check (the key check) exonerates in-sample tuning but indicts the experiment.**
   Chaotic: split-half OOS 7,121 / 7,341 ≈ in-sample 7,175; a 2,000/5,000-draw OOS tune gives 8,434/8,440 —
   the floor's edge is real and stable out-of-sample. Wild: the tuned floor is strongly sample-brittle
   (authors' 11,002 on their seeds vs 18,042 on the gated paths vs 15,144 on 2k draws — a ±60% spread from
   the environment's own variance), but **every OOS estimate ≥ 15k still beats the gated LLM by 8–13k**.
4. **Strawman-formula check fails the narrative.** A 141-candidate family of causal adaptive policies
   (rolling-window order-up-to + regime-detector with EWMA reset) beats the gated LLM on matched demands:
   chaotic LOOCV 7,884 vs 11,055 (perm p = 0.037), wild LOOCV 17,984 vs 28,054 (p = 0.004) — and the gains
   come from **tuning, not adaptation** (adaptive ≈ tuned-fixed within 3–8%). "LLM catches up as chaos rises"
   is really "tuned algorithms beat one untuned hyperparameter point."
5. **Distribution-level (the only comparisons the on-disk data legitimately support):** `kb_pointer_verbal`
   ties the environment's tuned floor (chaotic 8,746 vs 8,434, Welch p = 0.83; wild 14,059 vs 15,144,
   p = 0.75) and beats only the *untuned* floor (chaotic rank-perm p = 0.023; wild NS by variance).
   `kb_system_gated` loses to the tuned floor (chaotic Welch p = 0.11 / paired p = 0.019; wild
   p = 0.004–0.009). **No config ever significantly beats the tuned floor.**
6. The brief's interim read ("verbal 6,692 may edge the tuned floor 7,420") does not survive the complete
   run: final verbal mean 8,746 > 7,420; on matched demands the tuned floor wins by 3.4–3.9k.

---

## Verdict table

| # | Threat | Verdict | Key evidence |
|---|--------|---------|--------------|
| 1 | Chaotic/wild demand **unseeded per config** (interleaving destroyed, tuned floors off-sample) | **HIGH** (fatal, new) | interleaved_runner.py:113-115 (`make_demand(horizon, pattern)` w/o seed); trace-reconstructed within-pass demand differs on 31–36/36 weeks; 0/10 match seeds 1000+i; disk floor costs ≈ untuned-floor distribution (KS p=0.954) |
| 2 | Statistical reality of "LLM ≈ tuned floor" | **HIGH** | distribution-level: verbal ties tuned floor (p = 0.83/0.75), gated loses (p = 0.11/0.004); demand-matched: tuned floor beats gated p = 0.015–0.019 (chaotic), 0.004 (wild), 9/10 passes; n=10 not the issue — the design is |
| 3 | Tuned floor computed fairly? (OOS check) | **LOW** (no in-sample artifact) / **HIGH** (wrong demand sample) | chaotic OOS = in-sample (7.1–7.3k vs 7.2k); wild sample-brittle (11.0k–21.7k across samples) but every OOS estimate still beats gated LLM |
| 4 | Generator fairness / strawman formula | **HIGH** (crossover = "LLM beats one dumb formula") | tuned fixed + adaptive policies all beat the gated LLM on matched demands (p = 0.004–0.037); adaptive ≈ tuned-fixed |
| 5 | Same-week leak still closed? | **NONE** (closed) with **MEDIUM** residuals | 0/1,940–1,945 upstream trace weeks carry `incoming_now`; engine.py:174-180 + tests/test_no_leak.py; residuals: retailer still sees demand[t] (AUDIT2 quantified ≈ −7% floor cost), and kb_system_gated's prompt renders literal "None" upstream (agents.py:326) |
| 6 | Verbal-gate / fallback integrity | **HIGH** (persists) | kb_pointer_verbal = 21–29% silent mirror fallbacks (324/412/298 of 1,440 decisions); conf-gate fires 128–135× on verbal config, 1–8× on gated (inert); verbal gate 34–70× |
| 7 | Window drift / week-0 exchangeability | **HIGH** (open, unchanged) | week-0 factory order = 4 in 7–8/10 verbal runs vs 0–1/10 gated in all arms (Fisher p = 0.0008 / <1e-4 / 0.0013); same-state completions are config-correlated |
| 8 | Data integrity & accounting | **NONE** (records/tokens) / **LOW** (missing artifact) | 40/40/arm, runs 0–9, zero error stubs, token sums byte-equal summaries, calls = 4,320; **wild has no summary.json** on disk; shuffle seed 7 inferred for all arms (undocumented) |
| 9 | Baseline under chaos | **NONE** (qual claim survives) | medians 393.9M (chaotic), 11.6M (wild), 280.7M (noisy); means dominated by blowups (chaotic to 2.03e21, wild to 1.8e193) |
| 10 | Anything else (mirror amplification, clips, "None" prompt) | **MEDIUM** | mirror-fallback cascades drive the verbal tail (max orders 857–1,125 in the worst wild runs); chaotic clip [1,40] never binds at top, wild [1,60] binds 3.9% top / 6.9–12.5% at 1 |

---

## 1. The unseeded-demand bug — full evidence

**Code path.** `interleaved_runner.py:113-115`:
```python
if args.pattern == "noisy":
    demand = make_demand(args.horizon, "noisy", seed=1000 + i)
else:
    demand = make_demand(args.horizon, args.pattern)
```
`make_demand` for `chaotic`/`wild` (engine.py:110-145) calls `random.Random(seed)` with `seed=None` →
OS-entropy seeding → a *different* path per call. The runner calls it once per (pass, config), so all four
configs in a pass drew independent demands. `noisy` is the only pattern that gets a seed, so the noisy arm is
correct (verified: within-pass demand identical, byte-equal to the seeded reproduction — see §8).

**Trace-level proof** (retailer `incoming_now` recovered from `traces[role].ctx`, which the introspect configs
log every decision; format-failure weeks append no trace and are excluded):

| arm | within-pass value mismatches (weeks where both configs have traces) | runs matching seeds 1000+i |
|-----|-------------------------------------------------------------------|----------------------------|
| noisy (validation) | 0/30–36 all runs | 10/10 (only weeks with a verbal trace; gated 36/36) |
| chaotic | 20–27 of 22–28 shared weeks, 31–36/36 raw | **0/10** |
| wild | 19–31 of 19–31 shared weeks, 29–36/36 raw | **0/10** |

The chaotic gated demands also *look* like chaotic draws (means 4.4–13.6, max 23) but are not the seeded set
(seeded means 4.7–15.1). The on-disk `order_up_to` costs in chaotic (n=10, mean 14,646) are a draw from the
**untuned** floor's chaotic cost distribution (10,000-sim mean 14,641, sd 5,736; two-sample KS D=0.155,
p = 0.954) — i.e., the floor itself also ran on unseeded draws, its "10-pass variance" is pure demand luck.

**Consequences, concrete:**
- The brief's checklist item 1 ("recompute the tuned floor per pass locally — seeds are 1000+i") is
  impossible for chaotic/wild: no seed was used. I rescued pairing instead by reconstructing the gated
  config's demand from traces (36/36 weeks present in all 20 runs).
- The tuned floors in the brief (chaotic (4.5,0.1) → 7,420 CV 0.396; wild (5.0,0.1) → 11,002 CV 0.617)
  **reproduce exactly** on seeds 1000..1009 (verified; tuned_floor_chaos.py is correct) — they are honest
  numbers for a demand sample the LLM never faced. Comparing 8,746 (verbal, unseeded draws) to 7,420
  (tuned on seeds) is a two-sample comparison, not a paired one, and it does not survive as a win.
- Config-vs-config rankings inside chaotic/wild (verbal < gated) are between-independent-demand comparisons
  — no better than comparing across arms.
- Baseline, floor, and LLM means all absorb independent environment draws; with chaotic path means ranging
  4.4–13.6 and wild 2.7–38.0, run-level cost is dominated by which demand you drew.

## 2. Statistical reality (checklist 1) — the crossover is dead

### 2a. Paired, demand-matched (rescued; n=10 paired per-pass)

Floors and policies scored on each pass's *actual* demand (from gated traces). `kb_system_gated` vs:

| baseline (chaotic, same demand) | mean cost | mean diff | sign p | perm p | Wilcoxon p | boot 95% CI diff | wins |
|---|---|---|---|---|---|---|---|
| untuned floor (3.0,0.5) | 11,070 | **−15** | 1.00 | 0.97 | 1.00 | [−955, +895] | 5/10 |
| tuned floor in-sample (4.0,0.1) | 7,175 | **+3,880** | 0.0215 | **0.0153** | 0.0195 | [+1,551, +6,308] | 1/10 |
| tuned floor LOOCV (tune 9, score 1) | 7,611 | **+3,444** | 0.0215 | **0.0191** | 0.0195 | [+1,297, +5,635] | 1/10 |
| best adaptive in-sample (rolling-MA/detector, 141 cand.) | 7,320 | +3,736 | 0.0215 | 0.0114 | 0.0137 | [+1,627, +5,886] | 1/10 |
| best adaptive LOOCV | 7,884 | +3,171 | 0.0215 | **0.0367** | 0.0273 | [+869, +5,410] | 1/10 |

| baseline (wild, same demand) | mean cost | mean diff | sign p | perm p | Wilcoxon p | boot 95% CI diff | wins |
|---|---|---|---|---|---|---|---|
| untuned floor (3.0,0.5) | 27,242 | **+812** (LLM worse) | 0.75 | 0.50 | 0.70 | [−857, +2,830] | 4/10 |
| tuned floor in-sample (2.5,0.3) | 18,042 | **+10,012** | 0.0215 | **0.0038** | 0.0039 | [+5,694, +14,518] | 1/10 |
| tuned floor LOOCV | 19,614 | **+8,440** | 0.0215 | **0.0038** | 0.0039 | [+4,478, +12,921] | 1/10 |
| best adaptive in-sample | 17,610 | +10,444 | 0.0215 | 0.0038 | 0.0039 | [+5,072, +16,432] | 1/10 |
| best adaptive LOOCV | 17,984 | +10,070 | 0.0215 | **0.0038** | 0.0039 | [+4,975, +15,610] | 1/10 |

`kb_pointer_verbal` cannot be paired the same way: its retailer traces are 18–31/36 weeks (format-failure
weeks lack a trace and the demand cannot be recovered without on-hand/receipt series). Its comparisons are
distribution-level (§2b).

### 2b. Distribution-level (10 LLM draws vs large floor simulations; two independent samples from the same generator)

| comparison | LLM mean | floor mean | Welch p | exact rank-perm p |
|---|---|---|---|---|
| chaotic verbal vs tuned-floor OOS (2k draws, (3.0,0.2)) | 8,746 | 8,434 | 0.83 | 0.89 |
| chaotic verbal vs untuned-floor dist (10k) | 8,746 | 14,670 | <1e-4 | 0.023 (disk 10v10) |
| chaotic gated vs tuned-floor OOS | 11,055 | 8,434 | 0.11 | 0.17 |
| chaotic gated vs untuned-floor dist | 11,055 | 14,670 | 0.028 | 0.14 (disk 10v10) |
| wild verbal vs tuned-floor OOS | 14,059 | 15,144 | 0.75 | 0.98 |
| wild verbal vs untuned-floor dist (10k) | 14,059 | 20,778 | 0.029 (rank, 10k) | 0.123 (disk 10v10) |
| wild gated vs tuned-floor OOS | 28,054 | 15,144 | **0.009** | **0.004** |
| wild gated vs untuned-floor dist | 28,054 | 20,778 | 0.32 | 0.39 (disk 10v10) |

(An earlier pooled-label permutation test I ran gave ~0.0005–0.0006 for gated-vs-tuned; that test is invalid
here — unequal variances (LLM σ≈5.0k vs tuned-floor σ≈2.5k) and n=2000/10 break label exchangeability under
the null. The Welch and exact rank-permutation values above are the defensible ones.)

**Verdict on the crossover claims:**
- Claim A ("verbal beats untuned in chaotic, approaches/beats tuned floor"): the "beats untuned" half is
  statistically true at the distribution level (chaotic rank-perm p = 0.023 vs the disk floor; vs the 10k
  untuned distribution clearly). The "edges the tuned floor" half is false: verbal ties it (p = 0.83/0.89
  chaotic — identical means within noise — and p = 0.75/0.98 wild), and the paired rescue shows the *gated*
  LLM (the config the claim was actually made about in the interim read) loses to the tuned floor 9/10
  passes with p = 0.015–0.019.
- Claim B ("tuned floor per environment is the fair baseline"): fair in kind (it is a 1-minute grid search,
  stable OOS in chaotic), but its *value* is sample-brittle in wild (11.0k–21.7k across 10-path samples) —
  "the fair baseline" needs ≫10 paths to be well-defined. The 7,420 / 11,002 numbers themselves are honest
  but refer to demands the LLM never saw.
- Claim C ("crossover exists"): **untrue as stated.** The tuned formula's edge over the LLM *grows* with
  chaos (noisy +1,272, p = 0.008; chaotic +3,880, p = 0.015; wild +10,012, p = 0.004, all vs the gated
  config on matched demand). The most favorable reading of the whole dataset is "the verbal-gated LLM
  approximately ties a tuned 1970s forecasting formula on chaotic/wild demand at 3–4 orders of magnitude
  more compute, while being 21–29% fallback decisions" — that is not a crossover.

## 3. Is the tuned floor computed fairly? (checklist 2; THE key check)

| arm | in-sample (81-grid) | LOOCV | split-half OOS (evens→odds / odds→evens) | large-sample OOS (4k→4k, coarse grid / 2k→2k) |
|---|---|---|---|---|
| chaotic | 7,175 (4.0,0.1) | 7,611 | 7,341 / 7,121 | 8,440 / 8,434 |
| wild | 18,042 (2.5,0.3) | 19,614 | 21,702 / 19,601 | 15,241 / 15,144 |

- **No in-sample tuning artifact.** Chaotic OOS ≈ in-sample (7.1–8.4k band); the tuned floor's win over the
  gated LLM (11,055) holds out-of-sample with 9/10 paired losses and CI [+1,297, +5,635] (LOOCV).
- **Wild is sample-brittle but still wins.** The 10-path tuned floor bounces 11.0k (authors' seeds) ↔ 18.0k
  (gated paths) ↔ 19.6–21.7k (split-half) — the environment's own variance makes any n=10 "tuned floor" a
  noisy number. But every OOS estimate (15.1–21.7k) sits 8–13k below the gated LLM's 28,054, and the
  split-half OOS versions of the authors' own seeds-tuned model were not even run (their script tunes and
  scores the same 10 paths — `tuned_floor_chaos.py:29-37` prints only in-sample). Extending `oos_floor.py`
  to chaotic/wild as the brief asked gives the split-half numbers in the table.
- **Tuning headroom vs LLM edge:** the tuned-vs-untuned gain is −38% chaotic, −27% wild, −15% noisy —
  larger than any LLM advantage anywhere in the data. The formula side was never "given its best shot"
  in the on-disk arms (they ran (3.0,0.5)); that single fact explains most of the apparent crossover.

## 4. Generator fairness / strawman check (checklist 3)

(a) **Regime-switching is partially learnable; the fixed floor was the wrong null.** The simple adaptive
family I ran (rolling-window order-up-to with window w ∈ {2..12}, and an EWMA-with-regime-detector that
resets its forecast when recent mean deviates >30–80%) — 141 candidates, tuned on 9 passes, scored on the
held-out pass — beats the gated LLM on matched demands in **both** arms (chaotic LOOCV 7,884 vs 11,055,
perm p = 0.037; wild 17,984 vs 28,054, p = 0.004). A few lines of deterministic code, no LLM.

(b) **The gains are from tuning, not from adaptation.** Best adaptive (7,320 in-sample chaotic; 17,610
wild) ≈ tuned *fixed* floor (7,175 / 18,042) within 3–8%. The regime-detector buys almost nothing over a
tuned static policy on a 36-week horizon — consistent with the tuned floor being an exponentially-smoothed
mean-tracking policy that already adapts at λ = 0.1–0.3. The brief's premise ("exponential smoothing
genuinely can't track wild") is technically true but irrelevant: the order-up-to policy does not need to
track the level to score well, and whatever tracking matters, λ∈{0.1,0.2} provides it.

(c) Conclusion: the crossover as framed is **"LLM ≈ one dumb formula; loses to a tuned formula and to
tuned algorithms."** If the intent is "LLM beats a strawman", the check is passed; if it is "LLM beats
algorithms", it fails in both new environments.

## 5. Same-week leak — closed, with two residuals (checklist 4)

- **Closed and verified at three levels.** Engine: upstream roles get `incoming_now = None`
  (engine.py:174-180, AUDIT2 fix); test: tests/test_no_leak.py passes (upstream `incoming_now is None`;
  prompt says "not yet known"); runtime: **0 of 1,940 (chaotic) / 1,945 (wild) / 1,867 (noisy) upstream
  trace weeks carry a non-None `incoming_now`** across all LLM configs.
- Residual 1 (MEDIUM): the **retailer** still sees current-week customer demand `demand[t]` (engine.py:176).
  AUDIT2's leak quantification put the retailer-only component at ~7% of floor cost (4,701 → 4,369). The
  paper's own timing (docstring: "period-t order is placed on info through t-1") excludes it. Small,
  pro-LLM, unresolved.
- Residual 2 (bug, LOW): `build_user_prompt` (used only by `kb_system_gated`, the system-placement
  variant) renders `- Incoming order from your customer this week: {ctx['incoming_now']}` (agents.py:326) —
  for upstream tiers that is the literal string "None". `build_prompt` (used by `kb_pointer_verbal`) renders
  "not yet known (you decide on last week's information)". No information leaks, but the two configs were
  given different *phrasings* of the same information — an unplanned config difference beyond the gate/clamp.

## 6. Verbal-gate / fallback integrity under chaos (checklist 5)

| arm | kb_pointer_verbal: format-failure fallbacks | of 1,440 decisions | verbal gate firings | conf-gate firings | kb_system_gated conf-gate firings |
|---|---|---|---|---|---|
| noisy | 412 | 28.6% | 34 | 132 | 4 |
| chaotic | 324 | 22.5% | 46 | 128 | 1 |
| wild | 298 | 20.7% | 70 | 135 | 8 |

- **≈1/4 of kb_pointer_verbal's decisions are silent deterministic mirror fallbacks** (parse failure →
  `_fallback(ctx)` = mirror, agents.py:151-154; no trace record). Its reported mean/CV are unlabeled
  fallback mixtures — the AUDIT2 §4 integrity issue persists at the same rate. The verbal config's
  chaotic/wild means are not LLM-only numbers.
- **Mirror fallbacks amplify:** wild verbal runs 4/7/8 have max orders 1,125 / 842 / 857 and are exactly
  the worst-cost runs (29,936 / 30,513 / 26,890) — the tail is fallback cascades, and the high CVs
  (0.49 chaotic, 0.72 wild) are failure modes, not tradeoffs. (The gated config has 0 failures; its large
  orders up to 1,301 are genuine LLM orders.)
- **kb_system_gated's confidence gate is inert** (1–8 firings across 1,440·10 decisions, 0.1–0.6%); the
  config difference vs `kb_pointer_verbal` is not explained by gating (same conclusion as AUDIT2 §5b).

## 7. Window drift and week-0 exchangeability (checklist 6)

- **Interleaving is moot here**: with per-config demand, there is no shared pass environment to drift
  jointly — the AUDIT.md #1 confound was removed by design and then reintroduced by the missing seed in a
  different form (environmental non-comparability instead of temporal drift).
- **Week-0 order exchangeability is violated at p ≤ 0.0013 in all three arms.** At week 0 every factory
  has identical state and identical information (demand unknown); the factory ordered **4 in 7–8/10
  `kb_pointer_verbal` passes but 0–1/10 `kb_system_gated` passes** (exact Fisher two-sided p = 0.0008 noisy,
  <1e-4 chaotic, 0.0013 wild). Identical states draw config-correlated completions from this endpoint —
  AUDIT2 §5b confirmed again, and since chaotic/wild config rankings are between-demand anyway, the
  "verbal beats gated" ordering in these arms is doubly uninterpretable as a policy effect. Mechanism
  UNVERIFIED (endpoint non-exchangeability; cannot probe read-only).
- Per-pass config positions within the shuffle are mildly imbalanced (baseline mean position 2.0 of 3 vs
  LLM configs 1.2–1.5) — immaterial given unseeded demand.

## 8. Data integrity & accounting (checklist 7)

- **40/40 records per arm; zero error stubs; run indices 0–9; each config exactly once per pass.**
- Token accounting: per-record sums byte-equal the summary files (noisy: 2,731,940+177,506; chaotic:
  2,734,230+176,507), `prompt_hash_dups_calls` = 4,320 = chat calls; `prompt_hash_dups_total = 0` is
  vacuous (nonce-injected, AUDIT2 §7).
- **`results/wild_leakfree/` has no summary.json** (only the jsonl; it is complete). The chaotic/noisy
  summaries match their jsonls exactly.
- **Seeds: `noisy` seeded 1000+i per pass (verified byte-exact); `chaotic`/`wild` unseeded (§1). Shuffle
  seed = 7 inferred for all three arms** (the only seed ≤400 reproducing every pass's config order) — still
  undocumented in the summaries/launch scripts (AUDIT2 §7 provenance gap persists).
- The brief's stated numbers are stale in places: "interim read (chaotic, ~3.5 passes): verbal ~6,692 vs
  order_up_to ~10,688" vs completed 8,746 / 14,646; and the brief's "measured tuned noise floor 4,701
  beats LLM 5,424–5,973" is confirmed exactly (4,701.1 vs 5,423.5 / 5,972.7).

## 9. Baseline / environment sanity (checklist 8)

| arm | baseline mean | baseline median | baseline min–max |
|---|---|---|---|
| noisy | 3.02e12 | 280.7M | 230k – 3.02e13 |
| chaotic | 2.03e20 | 393.9M | 309k – 2.03e21 |
| wild | 1.84e193 | 11.6M | 272k – 1.84e194 |

The qualitative "ungated LLM = instability machine" claim survives and strengthens (wild worst-case
1.8e194). Chaotic median ~394M > noisy median ~281M, consistent with "more chaos → worse LLM" — but every
number is dominated by 1–2 blowup runs, so medians are the only honest summary (also true of the floors:
chaotic untuned floor mean 14,646 with CV 0.35 is pure demand-path variance).

## 10. Anything else (checklist 9)

1. **Clips.** Chaotic [1,40] never binds at the top in the observed paths (max 23, P95 18); wild [1,60]
   binds top on 14/360 weeks (3.9%). The **lower** clip binds frequently: 25/360 (6.9%) chaotic, 45/360
   (12.5%) wild weeks sit at demand = 1 — the shock machinery's negative tail is truncated, so "wild" is
   less wild than the unclipped process. Both clips bind in expectation; the generators are bounded, not
   heavy-tailed.
2. **Cost function under huge orders.** Cost = holding·OH + 2·backlog; huge orders show up as backlog
   surges upstream (e.g. chaotic order_up_to run 0: the factory orders 188 at week 4 and the distributor
   places 148 at week 35 into a stale regime). The mirror fallback amplifies by chasing the tier-below's
   (possibly LLM-inflated) last order. No overflow in the cost path (Python ints); the baseline's 9.2e18+
   orders are real but contained within the simulator.
3. **`kb_system_gated` vs `kb_pointer_verbal`.** Gated is consistently worse than verbal in mean under
   noisy (5,973 vs 5,424), chaotic (11,055 vs 8,746), wild (28,054 vs 14,059; unpaired rank-perm p = 0.019)
   — but the comparison is between independent demand draws, and week-0 serving correlation (p ≤ 0.0013)
   means no policy-free causal reading. The confidence gate cannot be the mechanism (1–8 firings).
4. **The reshaped noisy arm re-confirms AUDIT2's leak quantification from the other direction.** With the
   leak removed, the gated LLM no longer beats even the *untuned* floor (5,973 vs 5,521, +451, CI
   [+55, +915], perm p = 0.063 — the "win" flipped to a loss), and loses to the tuned floor significantly
   (+1,272, p = 0.008). Verbal: −98 vs untuned (p = 0.67), +722 vs tuned (p = 0.041).

---

## Top 5 remaining issues, ranked (with fixes)

1. **The crossover arms ran unseeded (HIGH — kills the experiment's core comparison).** interleaved_runner.py
   seeds only `noisy`. Fix: `make_demand(horizon, pattern, seed=1000+i)` for every pattern (store the seed
   per record + in the summary), and add a test asserting within-pass demand equality across configs
   (extend tests/test_no_leak.py or a new test_demand_sharing.py).
2. **`kb_pointer_verbal` integrity: 21–29% silent mirror fallbacks (HIGH, unchanged from AUDIT2 §4).** The
   config's means/CVs are unlabeled mixtures; under chaos its tail runs are fallback cascades. Fix: record
   `order_used`/`is_fallback` per decision, report a fallback-free mean, and enforce the ORDER format
   (or retry once on parse failure) to cut the 20–29% failure rate.
3. **Week-0/config-correlated completions (HIGH, open).** p ≤ 0.0013 in all three arms; identical states
   draw config-correlated outputs, so any same-arm config ranking (verbal vs gated) lacks a causal read.
   Fix: rotate configs' *positions* with the same seed per pass and re-run twin pairs as deliberate A/Bs
   (AUDIT2 recommendation, still open); log serving node/id per call if the endpoint exposes it; raise n.
4. **The formula side still never got its best shot in the on-disk arms (HIGH, design).** Both new arms ran
   the untuned (3.0,0.5) floor while the claim was benchmarked against a *differently-sampled* tuned floor.
   Fix: run the tuned (θ,λ) per environment as a first-class config in the interleaved matrix (it costs
   nothing — it is deterministic), and report LLM vs tuned vs untuned in one table.
5. **Wild's "fair baseline" is undefined at n=10 (MEDIUM/HIGH).** Tuned-floor estimates over 10-path samples
   span 11.0k–21.7k. Fix: use ≳30 passes for wild (its demand variance is the dominant term), and report
   the tuned floor's own sampling distribution (bootstrap over paths), not a point value.

## What survives this audit

- **The leak fix is real and verified** end-to-end (code, test, and 100% of upstream trace weeks).
- **The noisy arm is methodologically sound** (seeded, shared, reproducible) and its headline numbers are
  exactly as the brief states (tuned floor 4,701 < verbal 5,424 < gated 5,973; all paired tests on the
  same demand).
- **The tuned floors are honest numbers** (reproduced exactly on seeds 1000..1009) and robust out-of-sample
  in chaotic; in-sample tuning is *not* the artifact this time.
- **"Formula family tunes up and wins" is robust: tuned fixed and tuned adaptive policies beat the gated
  LLM significantly in chaotic and wild on matched demands** (p = 0.004–0.037, 9/10 passes) and tie-or-beat
  the verbal config at the distribution level.
- **The empirical engine mechanics remain exact** (my seeded floor reproductions byte-match on-disk costs
  for noisy; the tuned script's numbers reproduce).
- **Qualitative claims survive:** the untuned baseline under chaotic/wild demand is catastrophic
  (medians 394M / 11.6M, blowups to 2e21 / 1.8e194) — "structure > raw LLM judgment on this endpoint"
  holds even as demand gets harder.

## 10-line summary

1. **Fatal new finding:** `interleaved_runner.py:113-115` seeds only `noisy`; the chaotic and wild arms ran
   on unseeded, per-config demand — within-pass demand differs 31–36/36 weeks, 0/10 runs match the seeds
   (1000+i) the tuned floors were computed on, and the on-disk chaotic floor is a draw from the *untuned*
   floor distribution (KS p=0.95). The interleaving control and all planned per-pass pairings are void.
2. The crossover is dead: on demand-matched (trace-reconstructed) comparisons, the tuned floor beats
   `kb_system_gated` 9/10 passes in chaotic (7,175 in-sample / 7,611 LOOCV vs 11,055; perm p=0.015/0.019)
   and wild (18,042/19,614 vs 28,054; p=0.004); even the untuned floor ties gated on matched demand.
3. The key OOS check exonerates in-sample tuning: chaotic split-half OOS 7,121–7,341 ≈ in-sample 7,175;
   wild is sample-brittle (11.0k–21.7k across 10-path samples) but every OOS estimate still betters the
   gated LLM by 8–13k.
4. The strawman check fails the narrative: 141 causal adaptive policies (rolling-window/regime-detector)
   also beat the gated LLM on matched demand (LOOCV 7,884 chaotic p=0.037, 17,984 wild p=0.004) — and the
   gain is tuning, not adaptation (adaptive ≈ tuned-fixed within 3–8%).
5. Distribution-level (the only legitimate on-disk comparisons): verbal ties the tuned floor (chaotic
   p=0.83, wild p=0.75) and beats only the *untuned* floor (chaotic rank-perm p=0.023); gated loses to the
   tuned floor (wild p=0.004–0.009, chaotic paired p=0.019). No config ever significantly beats the tuned
   floor — "LLM wins/ties on unfittable demand" is unsupported.
6. The leak is closed and verified (0 upstream leak-weeks in 5,752 across all arms; test + code agree) —
   residuals: retailer still sees demand[t] (~7% pro-LLM by AUDIT2's quantification), and `kb_system_gated`
   renders literal "None" upstream (agents.py:326) while `kb_pointer_verbal` says "not yet known".
7. `kb_pointer_verbal` remains 21–29% silent mirror fallbacks (324/412/298 of 1,440), its high chaotic/wild
   CVs are fallback cascades (worst wild runs have max orders 857–1,125), and the gated config's confidence
   gate fires 1–8 times total (inert) — yet gated is nominally worse than verbal everywhere; unpaired +
   serving-correlated, so no causal read.
8. Week-0 exchangeability is violated at Fisher p ≤ 0.0013 in all three arms (factory orders 4 in 7–8/10
   verbal passes, 0–1/10 gated at byte-identical state) — the AUDIT2 open confound persists and contaminates
   any same-arm config ranking.
9. Integrity is otherwise clean: 40/40 per arm, zero error stubs, runs 0–9, token sums byte-equal
   summaries, calls=4,320; shuffle seed 7 inferred for all arms (undocumented); wild's summary.json is
   missing (jsonl complete); baseline medians 394M (chaotic) / 11.6M (wild) / 281M (noisy) keep the
   "LLM = instability machine" claim alive.
10. The authors' own numbers are confirmed where they stand alone (tuned 4,701/7,420/11,002 reproduce
    exactly on seeds 1000..1009; noisy arm is properly seeded) — the failure is the unseeded crossover
    design plus the tuned-vs-untuned benchmark mismatch, not fabrication. Re-run chaotic/wild with
    `seed=1000+i` per pass, add the tuned floor as a first-class config, and log per-call serving identity
    before any crossover claim can be taken seriously.

All scripts: /tmp/audit3_{helpers,adaptive,noisy,dist,block5,exact2,reconcile,integrity}.py. No repo files modified.