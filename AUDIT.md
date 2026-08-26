# Confound & Validity Audit — Agent Bullwhip Replication

Auditor: skeptical re-derivation from source + on-disk results. Read-only; no source/results modified.
All file:line citations refer to this repo's tree as of 2026-08-26 ~06:45 UTC. Note: the results
directory was **live during the audit** (`kb_pointer_verbal.jsonl` appeared mid-audit with 2 error
stubs; `baseline.jsonl` grew to 32 lines). Any number below was recomputed from the files on disk.

---

## Verdict table

| # | Threat | Verdict | Key evidence |
|---|--------|---------|--------------|
| 1 | Deterministic demand makes the floor trivially strong | **MEDIUM** | fixed path is the paper's design (defensible), but floor isn't optimal; robustness arm has no data |
| 2 | Verbal gate (kb_pointer_verbal) unrun / leaning on anchor | **HIGH** | no results on disk; regex over-triggers; gate is a thin veneer over the floor |
| 3 | Confidence gate doesn't predict quality / fallback does the work | **HIGH** | gate fires 0.3% in the winner, cost-neutral in counterfactual, corr(conf,dev) = −0.18 |
| 4 | Cache/collision: CV possibly deflated, check never ran | **HIGH** | identical prompts in one window served identically; `prompt_hash_dups` absent from all summaries |
| 5 | Statistics: n=30, no significance, no multiple-comparison control | **HIGH** | mean differences between headline configs are NS (perm p=0.37); ≥120 implicit tests |
| 6 | "Within 11% of optimal" claim | **HIGH** | floor is not optimal (3,148 < 3,681); comparison is circular; benchmark is a hybrid of the floor |
| 7 | General-KB beats domain-KB claim | **HIGH** | cross-window comparison + KB text changed mid-matrix |
| 8 | Anything else (drift, engine leak, parse, data integrity) | **HIGH** | endpoint non-stationarity is the single biggest confound in the whole study |

---

## 1. Deterministic-demand validity — MEDIUM

**What's right:** A single fixed demand path across all 30 runs is exactly how the paper isolates
run-to-run *agent* instability; `engine.py:74-108` implements the step (4→8 at t≥4) and the
deterministic floor (verified by re-running: `order_up_to` = 3,681, `mirror` = 13,920 — matches
summaries exactly; 41 tests pass).

**What's wrong:**

1. **The "floor" is not an optimum.** Dense sweep of (θ, λ) on the same demand gives
   `(3.0, 0.4) → 3,206` and `(3.0, 0.35) → 3,148`, i.e. 13–14.5% below the 3,681 "floor"
   the study benchmarks against. Even the authors' own `gap_analysis.py` "ground truth"
   reference has a smoothing bug (`gap_analysis.py:29`: `forecast = λ·x + (1−λ)·x = x` — the
   exponential smoothing collapses to the raw last value), so both the floor label and the
   gap-mining reference are wrong in the same direction. "Within 11% of the deterministic
   optimum" (README) is really "within ~30% of the best policy found in a 1-minute sweep".
2. **The scenario is the easiest possible for the wrapper, not the LLM.** With demand fixed and
   the model clamped to the anchor's policy (see §3), a near-optimal deterministic outcome is
   guaranteed by construction. The robustness arm that would test this (`noisy_arm.sh`,
   `make_demand(..., pattern="noisy")` in `engine.py:95-107`, seeded per run) has **no result
   files on disk**. The README maps it honestly as "in progress", so the reliability-advantage
   claim under variance is UNVERIFIED, not wrong.

## 2. The verbal gate (`kb_pointer_verbal`) — HIGH

**No results exist.** `results/` contains no `*kb_pointer_verbal*` summary; only a
`kb_pointer_verbal.jsonl` that appeared during this audit with exactly 2 lines, both
`"error": "LLM call failed after 6 attempts"`. `postmortem_verbal.py` and `mine_traces_gate.py`
both hard-code `results/deepseek-v4-flash-kb_pointer_verbal.jsonl` and would crash on it. The
README "Status" checkbox "[x] Wave 2e verbal-consistency gate" is **false**.

**Even if run, the design is a veneer.** In `agents.py:163-178`, the gate fires when
`order == 0 and backlog > 0` **and** the reasoning matches
`cover (the )?backlog|backlog (is|of|first)|order-?up-?to|not over-?react|pipelin|outstanding`
(`agents.py:165`). "pipelin" and "outstanding" match nearly any coherent reasoning, so the
effective rule is: *order 0 with outstanding backlog → replace with the full order-up-to floor*.
Simulating the exact gate on the three ungated introspect configs' real traces:

| config | actual mean | verbal-gate sim | triggers/run | triggers where floor=0 |
|--------|-------------|-----------------|--------------|------------------------|
| kb_introspect | 24,199 | 23,306 (−4%) | 10.4 | 8.7 (84%) |
| kb_system_introspect | 64,929 | 65,845 (+1%) | 4.2 | 3.1 (74%) |
| kb_pointer_introspect | 35,198 | 35,260 (±0%) | 8.2 | 6.2 (76%) |

The gate is cost-neutral on real data; ~80% of its triggers are **no-ops** (the floor itself
orders 0 for that state — the model was right). The "trace-driven fix for the dominant failure
mode" narrative is not supported by the data it would operate on. And the override replaces the
model's decision with a *fresh* `OrderUpToAgent` (`agents.py:169-172`), i.e. the deterministic
policy — the "fix" is the wrapper, not the model.

## 3. The confidence gate — HIGH

**The winner barely uses it.** In `kb_system_gated` (the headline 4,086 / CV 0.089), the
confidence gate fired **13 of 4,320 decisions (0.30%)**, on only 11 of 30 runs; the run-level
correlation between #gated and cost is **+0.13**. The `anchor_margin=6` clamp changed **0 of
4,307** non-gated decisions (also 0/3,184 in `kb_introspect_gated`, 0/2,877 in `kb_pointer_gated`)
— the model's raw orders were always within ±6 of the anchor, so the clamp was inert in every
gated window.

**Low confidence does not predict bad decisions.** On ungated `kb_introspect` traces:
corr(self-reported confidence, |order − floor|) = **−0.18**; low-conf (<0.5) decisions deviate
18.4 vs 14.1 units — weak, and low-conf decisions are actually *less* likely to deviate from the
floor (67% vs 75%). Counterfactual: replacing every conf<0.5 decision with the order-up-to anchor
changes cost 24,199 → 24,299 — **cost-neutral** (7.6 decisions/run replaced).

**Yet the twin config "improves" 16×.** `kb_system_gated` (4,086 ± 364) and `kb_system_introspect`
(64,929 ± 20,902, run 57 min earlier, *identical prompts*) differ by 16×, while the config
difference (gate 0.3% usage + inert clamp) explains ~0–4% of it. The remaining explanation is
the run window, not the config (see §8). README Key Finding 2 — "**The confidence gate is the
cost killer**" — is **unsupported**: the gate is demonstrably inactive where the win happened.

**Failure fallback masks decisions.** For `*_gated` configs, any response without an `ORDER:` line
is counted as a failure and silently replaced by the mirror order (`agents.py:140-142`).
Failure rates: kb_pointer_gated **26.4%**, kb_introspect_gated **19.4%**, kb_general_introspect
**34.7%**, kb_system_gated 0%. So ~1/4 of the "winner runner-up's" decisions were *not LLM
decisions at all* — the reported mean/CV are unlabeled mixtures. No trace entry is written for
failure-fallback decisions, so they're invisible unless you know to look. Note the docstring
contradiction: `introspect.py:7-9` claims a classic first-integer fallback "so the KB upgrade
never breaks the baseline measurement", but the agent path has **no** classic fallback.

## 4. Cache / collision check — HIGH

- The collision metric was **never computed on any on-disk result**: `prompt_hash_dups_total` /
  `prompt_hash_dups_runs` (runner.py:152-153) are **absent from every summary JSON** (verified
  key-by-key on `kb_system_gated.summary.json`). The check was added to `runner.py` after the
  matrix was produced; the summaries predate it.
- As coded, the check is per-run (`client_mod.prompt_hashes.clear()`, runner.py:101) and counts
  within-run duplicates only — the cross-run case that would deflate CV (the *same* week-1 prompt
  hashed 30× across runs) is **never examined**. Within-run identical prompts are also legitimate
  (early weeks have identical states), so even the within-run counter conflates natural state
  recurrence with cache hits.
There is direct evidence of identical-prompt determinism in at least one window: factory week-0
order was `6` in **27 of 30** `kb_system_gated` runs (4 in the other 3) while the same prompt in
the adjacent `kb_system_introspect` window drew a spread over 7 values
(`{16×13, 8×8, 32×4, 12×2, 24×1, 6×1, 4×1}`). That is either
  cache-served completion or temperature-ignored serving — either way, the CV measured in that
  window is not comparable to CVs from windows where serving was stochastic. CV 0.089 is
  therefore suspect as an *agent-instability* estimate. (Mechanism UNVERIFIED — cannot probe the
  endpoint read-only.)

## 5. Statistical rigor — HIGH

n=30. Bootstrap 95% CIs (40k resamples, from per-run costs):

| config | mean | 95% CI | CV | 95% CI CV |
|--------|------|--------|-----|-----------|
| kb_system_gated | 4,086 | [3,958, 4,216] | 0.089 | [0.063, 0.110] |
| kb_pointer_gated | 4,284 | [3,941, 4,685] | 0.250 | [0.158, 0.336] |
| kb_introspect_gated | 4,778 | [4,316, 5,304] | 0.290 | [0.194, 0.343] |
| anchor | 5,728 | [5,335, 6,120] | 0.193 | [0.139, 0.236] |

Two-sided permutation tests (40k):

| comparison | mean diff p | CV diff p |
|-----------|-------------|-----------|
| kb_system_gated vs kb_pointer_gated | **0.37** (NS) | **1e-4** (sig) |
| kb_system_gated vs kb_introspect_gated | 0.008 | 0.001 |
| kb_system_gated vs anchor | <1e-4 | 0.001 |

- The **mean** difference between the two headline configs (4,086 vs 4,284) is **not
  significant** — yet the README presents them as a ranked frontier.
- The CV difference 0.089 vs 0.25 survives naive 0.05 but only ~half the nominal pairwise set
  survives a Bonferroni correction (0.05/120 = 4.2e-4): 1e-4 yes, 0.001 no.
- **No multiple-comparison control anywhere.** 16 deepseek LLM configs ⇒ ≥120 implicit pairwise
  comparisons; the README's "pointer > system > inline", "general > domain", "gated ≈ floor"
  claims are all unadjusted selections from that space.
- Baseline tails make even huge mean gaps NS: `budget` (15,565) vs `baseline` (29.2M) Welch
  p = 0.27; `voting5` vs `baseline` p = 0.45. The "99.9% cost reduction" for budget is not
  supported at n=30; the paper's own lever replications are weaker than claimed.
- The protocol (`protocol.md`) pre-registered effect gates H2/H3/H4 — that's good practice — but
  no test was ever performed; CV comparisons are presented point-estimate only.

## 6. "Within 11% of optimal" — HIGH

4,086 / 3,681 = 1.110 — arithmetic correct, claim misleading:

1. The floor is **not** the optimum (3,148–3,206 achievable; §1). "Within 11% of optimum" should
   read "within 11% of one reasonable deterministic policy" (and +30% of the best policy swept).
2. The comparison is **circular**: `kb_system_gated` is structurally clamped/gated onto that
   policy's own order values (anchor fallback `agents.py:150`, gate overrides to floor
   `agents.py:169-172`, ±6 clamp `agents.py:202-205`). You are comparing a hybrid policy to its
   deterministic component.
3. The right benchmark per the replication's own framing is the paper's numbers / human teams —
   which are reported **normalized** in the paper and are not reproduced here (README admits "No
   human baseline"). Absent that, "close to the floor" is only an internal sanity metric; and
   since the floor is a *mean-cost* bound while the study's headline is *CV*, the "within 11%"
   framing quietly drops the dimension the study is actually about.

## 7. General-KB vs domain-KB claim — HIGH

`kb_general` 124,372 vs `kb` 2,427,502 (19×). This is not interpretable:

1. **Different KB text, mid-study.** `PROMPT_KB.md` gained kb-3b ("in-transit is NOT available
   this week", commit fa538d3, **2026-08-25 15:28**). `kb` ran **11:00–11:32** and
   `kb_introspect_gated` ran **13:40–14:56** (v1 KB); `kb_system` **straddles the change** (started
   14:56, finished 15:42 — the KB file was replaced mid-run at 15:28, so its 30 runs mix KB v1 and
   v2); `kb_general` ran **20:24–21:02** using `GENERAL_KB.md` created **17:44** the same day.
   Three different prompt bodies AND three different run windows.
2. **Different windows.** Given adjacent-window drift of ≥16× (§8), a 19× ratio proves nothing
   about KB transferability.
3. **Format-failure asymmetry is real but points the other way:** under introspection the general
   KB collapses — `kb_general_introspect` failure rate 34.7%, `kb_general_gated` **100%** (mean
   13,920 = exact mirror cost, CV 0, max_order 8 — a fully degenerate fallback that any writeup
   would cite as a *result*). Claiming a general-KB win while its introspect/gated arms
   disintegrate requires an explanation that the paper doesn't give.

## 8. Other findings (in rough severity order)

1. **Endpoint non-stationarity — the study's #1 confound (HIGH).** Identical prompts, temperature
   0.7, same model alias, windows minutes to hours apart, produce statistically disjoint output
   distributions: `kb_system_introspect` (15:42–16:50) mean 64,929 ± 20,902 vs `kb_system_gated`
   (16:49–17:47) 4,086 ± 364 (perm p≈1e-4, Welch t = 15.9 ≈ 16 σ); factory week-0 draws
   {16,8,32,…} spread vs {6×27, 4×3}; `kb_general_gated`'s window (22:19–00:48) had a **100%**
   format-failure rate while
   `kb_general_introspect`'s (21:02–22:19) had 34.7%. Every config ran in its own window ⇒ every
   cross-config comparison in the README is a between-window comparison. The magnitudes claimed
   (16× gate win, 19× general-KB win) are of the same order as pure window-to-window drift.
   Mechanism UNVERIFIED (cache vs backend aliasing on FreeInference) — but the statistical
   incompatibility of adjacent windows is verified from the data.
2. **KB/protocol drift not logged.** `protocol.md` §7 "Changes log" still reads "(none yet —
   pre-registered)" though the KB changed mid-matrix. Results for `kb`, `kb_introspect`,
   `kb_introspect_gated` were produced with KB text that no longer exists on disk and were never
   re-run, and `kb_system`'s 30 runs span both KB versions (see §7) — any downstream re-analysis
   silently uses the *current* KB to explain past runs.
3. **Same-week information leak to upstream tiers (MEDIUM).** The prompt shows
   `Incoming order from your customer this week: {incoming_now}` (`agents.py:239`, engine
   `engine.py:132-141`), where the wholesaler/distributor/factory see the *current week's* order of
   the tier below because the engine processes roles sequentially. The paper (and the engine's own
   docstring) specify orders are placed on information through t−1. LLM agents thus get one week of
   lookahead the deterministic floor and the paper's model don't get — systematically *helping* the
   LLM toward the floor.
4. **`parse_order` = first integer anywhere (`client.py:103-107`).** Prose like "the 2-week lead
   time means I order 8" parses as 2. Non-introspect configs report failure_rate 0% (every
   response contains *an* integer) — "instruction-following rate" as reported measures nothing
   about following the order format. Responses aren't stored for non-introspect configs, so the
   parse error rate is UNVERIFIED (and unverifiable) for baseline/kb/kb_pointer/kb_general.
5. **Data integrity (MEDIUM).** `runner.py` appends to jsonl in append mode (`open(..., "a")`)
   without truncation ⇒ invocations mix: `baseline.jsonl` now has **32** lines (runs 0–1 duplicated
   as API-error stubs, total_cost None) while its summary reports 30 completed / 0 failed.
   `mirror.jsonl` and `order_up_to.jsonl` are 30 **error stubs** (`'MirrorAgent' object has no
   attribute 'failures'` — the exact bug commit ada7b2d fixed) with the real per-run data lost;
   only the aggregate summaries (13,920 / 3,681) survive. Any jsonl-driven reanalysis is
   corrupted for those configs.
6. **Protocol deviation:** `protocol.md` §2 fixed "retries 3"; `client.py:38` uses `retries: int =
   6`. Unlogged.
7. **"Repeated sampling doesn't help" nuance:** voting5 CV 4.55 vs baseline 4.97 fails the
   pre-registered 10% gate (H2 rejected — correctly reported), but voting5 mean is 8.6M vs 29.2M
   — sampling helps cost a lot; "doesn't help" is only true for stability and is statistically
   unsupported either way (perm p≈0.5).
8. **`combined` is not cherry-picked** (it was pre-registered as config F, `protocol.md` §3) — but
   it's statistically identical to `anchor` (perm p=0.47), so the premium "structural combo"
   framing in README is unsupported.
9. Token accounting checks out (prompt ≈1,233 tok/call for KB+introspect; completion ≈59 — no
   accounting bug found).
10. OFF-BY-ONE, seeds, initialization: none found. Demand step at t≥4 is consistent between
    protocol, engine, and README; noisy seeds are reproducible and identical across configs for
    the same run index (verified).

---

## Top 5 confounds that would change the conclusions if fixed

1. **Run-window / endpoint non-stationarity (HIGH).** Every cross-config claim — the gated
   "16× win", "general KB beats domain KB", the whole ranking table — is a between-window
   comparison against drift of the same magnitude. Fix: interleave configs per run (one run of
   each config per round-robin), log per-call model ID/timestamp, and re-check the twins.
2. **The winner's gain is the deterministic wrapper, not the LLM (HIGH).** Conf gate 0.3% +
   inert clamp explain none of the observed win; counterfactual gate is cost-neutral; the
   "within 11% of optimum" config is the floor's own policy wearing an LLM costume. Fix: report
   an ablation column "wrapper only (no LLM)" — which, from the counterfactuals, will reproduce
   most of the 4,086.
3. **KB + protocol drift mid-matrix (HIGH).** kb-3b added at 15:28 while kb-family results from
   11:00–14:56 remain on disk; changes log empty. Fix: freeze KB files per run (snapshot
   content into the jsonl), log every change, re-run affected configs.
4. **CV deflation by identical-prompt serving + dead collision check (HIGH/MEDIUM).** Week-0/early
   prompts repeat verbatim across runs (fixed demand); one window served them deterministically,
   and `prompt_hash_dups` was never actually recorded. Fix: run the check across runs (not
   cleared per run), vary a harmless prompt nonce per run to defeat serving-level caching, and
   compare week-1-order entropy across configs as a serving-mode sanity metric.
5. **No statistical inference (HIGH).** Headline mean gaps are NS; CV gaps mostly don't survive
   correction; 120+ implicit comparisons. Fix: permutation tests on cost AND CV, Bonferroni/Holm,
   report bootstrap CIs for every row, and demote any claim failing correction.

## Concrete fixes (one line each)

- Interleave configs round-robin per run; log model ID + timestamp per call, not per config.
- Delete/mark `kb_pointer_verbal` claims: no results exist; also shrink its regex to
  `cover (the )?backlog|order-?up-?to` and require the *override* to beat the floor before firing.
- Report "wrapper-only" ablation (LLM outputs ignored; anchor/clamp/gate applied to nothing) as a
  column; expect it to land ≈ 4,000 — that's the honest headline.
- Snapshot the exact prompt (incl. KB text + hash) into every jsonl line; freeze KBs per wave.
- Make the collision check cross-run: hash across all runs, and put a per-run nonce in the prompt
  to defeat serving-level caching before trusting any CV.
- Replace point estimates with permutation CIs for mean and CV; Holm-correct across the config
  matrix; mark NS comparisons as NS in the README table.
- Add the classic-parse fallback the parser docstring promises (or delete the docstring); count
  failure-fallback decisions in the failure rate *and* flag them in results.
- Store raw model responses for every call (even non-introspect) so parse-order semantics are
  auditable; change `parse_order` to anchor at the `ORDER:` label when present.
- Rerun `kb`, `kb_introspect`, `kb_introspect_gated` under the current KB or version-stamp them
  as KB-v1 results.
- Label the "floor" as "one tuned policy" and cite the 3,148–3,206 sweep; drop "optimal".
- Repair jsonl side of mirror/order_up_to (rerun) and switch runner to truncate-on-start with a
  run-UUID per invocation.
- Document the same-week `incoming_now` leak explicitly or remove `incoming_now` from upstream
  roles' prompts to match the paper's t−1 timing.

## What survives the audit

- **Engine mechanics** (deterministic state update, 2-week lag, cost accounting; floors verify:
  mirror 13,920, order_up_to 3,681; 41/41 tests pass; no off-by-one in demand timing found).
- **The paper's reliability concern replicates qualitatively:** plain deepseek-v4-flash agents on
  this endpoint are catastrophically unstable (baseline cost CV ≈ 4.97 vs the paper's 0.13–0.46;
  max order 22.8M) — if anything worse than the paper.
- **Wrapper-level mechanism, not config-level:** clamping ordering to a deterministic ES anchor
  (the `anchor` config, 5,728/CV 0.19) and the budget cap (15,565) tame the chaos *within their
  own windows*, and the clamp demonstrably binds when the model is in the chaotic regime — the
  general direction "structure > raw LLM judgment" for this endpoint is plausible and worth
  re-testing under interleaving.
- **Protocol pre-registration exists** (H2/H3/H4 gates; voting5 correctly fails its gate; CRUD
  was improved along the way), though the changes log was abandoned.
- The deterministic baselines and the noisy-arm *script* are correct; only their results are
  missing.

## 10-line summary

1. Every cross-config conclusion in the README is a between-window comparison; adjacent windows
   on identical prompts differ by ≥16× (64,929 vs 4,086), which exceeds the effects being
   claimed.
2. The headline winner's "mechanisms" are inert: confidence gate 0.3% of decisions, clamp 0
   changes, gate cost-neutral in counterfactual — the win is the window, not the gate.
3. The verbal gate (kb_pointer_verbal) has no results on disk and its regex touches ~every
   plausible trace; simulated, it's cost-neutral and ~80% no-op.
4. "Within 11% of optimal" is circular (the config is the floor's own policy) and the floor is
   not optimal (3,148–3,206 beats 3,681).
5. KB content changed mid-matrix (kb-3b, 15:28) with results from earlier KB versions left on
   disk and the protocol change-log empty.
6. The prompt-hash collision check never ran (keys absent from every summary) and cannot see the
   cross-run cache that would deflate CV; one window shows identical-prompt deterministic
   serving.
7. n=30: headline mean gap (4,086 vs 4,284) is NS (perm p=0.37); no multiple-comparison control
   across 16 configs.
8. Up to 26% of decisions in some configs are silent mirror-fallbacks (failure rate), with no
   trace entry — reported means/CVs are unlabeled mixtures.
9. Data integrity: append-mode jsonl mixes invocations (baseline 32 lines), mirror/order_up_to
   jsonl are all-error stubs, and the results dir was written to during this audit.
10. What survives: the engine, the deterministic floors, the qualitative claim that plain LLM
    agents are instability machines on this endpoint, and "structure/anchoring tames the chaos" —
    all of which need interleaved re-runs before any number can be trusted.