# README Iteration Log — agent-bullwhip-replication

Process: wrote v1 → grok CLI critique (v1) → v2 → grok CLI critique (v2) → v3 (final, deployed to `README.md`). The grok CLI was run headless (`grok -p "<critique prompt + pasted README>"`, stdout capture). All final numbers were independently re-verified against the on-disk summaries, `AUDIT*.md`, and by re-running `walkforward_floor.py` and an extended per-seed analysis before v3 shipped.

Versions: `docs/README_v1.md`, `docs/README_v2.md`, `docs/README_v3.md`, deployed final `README.md`.

---

## Round 1 — critique of v1

Prompt: "Act as a brutal technical editor. Critique this README for clarity, honesty, impact, and structure…" (v1 pasted).

### What the critic said (grok, verbatim highlights)

> "This wants to be two documents: a detective story about three dead claims, and a methods+results note for the one that lived… A skeptical reader hits a slogan, a table with no uncertainty, then 2,000 words of autopsy, then the same table again."

> "**'state-of-the-art LLM agent.'** The agent is `deepseek-v4-flash` on a free endpoint… That word is a lie. Write 'this LLM agent' or 'a cheap free-API agent.'"

> "**'3–4 orders of magnitude less compute.'** …Put dollars, tokens, or wall-clock here or kill the phrase."

> "**'The intelligence in the "AI agent" was the formula all along.'** Overfit. The verbal config is ~70–79% model decisions plus 21–29% *mirror* fallbacks…"

> "**'small, documented residual'** (retailer still sees demand[t])… Your noisy walk-forward edge is 3.9%. The residual is *larger than the surviving noisy win* and it is pro-LLM. 'Small' is indefensible."

> "**Walk-forward scores the half it trained on.** `walkforward_floor.py` tunes on weeks 0–17, then `run_floor(d, th, la)` scores the **full 36-week path**. Half the reported cost is in-sample."

> "**'silent mirror fallbacks … collapsing to the deterministic anchor.'** False. `fallback: str = "mirror"` (`agents.py`). Mirror is chase-the-incoming… The gate overrides to a fresh `OrderUpToAgent`. The parse failures override to **mirror**. Those are opposite policies."

> "**"beats the untuned formula by 7–40%."** …The 7% is invented. The range is 6–39%, and the noisy end is a rounding error at n=10."

> "**Wild 'median 11.6M, blowups to ~1.8×10^194' is from `wild_leakfree`, the unseeded arm Audit #3 invalidated.**"

> "**10^194 | Baseline table | Trash. Unseeded.**" and "**CV omitted.** 'The paper's claim was reliability. You switched the metric to cost after the formula won cost.'"

### Accepted and fixed in v2 (with verification)

1. **Structure inverted:** surviving claim → dead claims → surviving table with CVs/uncertainty → story (protocol-tagged) → walk-forward protocol → audit history → limitations → reproduce. First reproduce command is now the interleaved runner, sequential runner demoted to "do not use for claims."
2. **Pull-quote definitions added**: each contested word ("LLM agent", "1970s", "tunes itself", "orders of magnitude", "wrapper carries the cost") mapped to concrete repo facts; compute grounded in actual token counts (2.73M prompt + 177K completion tokens, 4,320 calls, ~1 h wall per arm) and the formula's ~3 CPU-seconds.
3. **CV columns added** to the surviving table (on-disk summaries: verbal 0.199/0.435/0.429 etc.; walk-forward CVs computed by re-running the simulation).
4. **Mirror-vs-anchor conflation fixed** after verifying `agents.py:68` (`fallback: str = "mirror"`): parse failures ride `mirror`, the gate overrides to the anchor; these are opposite policies. v1 had laundered both as "the anchor."
5. **Walk-forward scoring disclosed**: tuned on weeks 1–18, scored on the full 36-week game — half in-sample. Added a held-out suffix check (see below).
6. **"7–40%" arithmetic corrected** after checking: walk-forward vs untuned = 5.6% / 39.3% / 25.0%.
7. **Leaked-floor evidence moved next to dead claim #2** (2,920–3,042, 37–40%).
8. **Retailer residual re-labeled**: quantified ~7% of floor cost (4,701 → 4,369, AUDIT3 §5); "small" removed.
9. **16× window-drift story amended with the twin-gap finding** (AUDIT2: 4,113 vs 55,098 *interleaved*, gate fires 0.2%, mechanism unknown) — the fix that actually happened.
10. **Wave 1 tombstoned** (pre-audit protocol only); prompt-hash metric dropped (vacuous by construction — nonce precedes hash); "published" softened to "written down"; paper authors restored; Ψ/Φ kept but marked non-load-bearing.
11. **Twin-gap open confound**, n=10/unpaired-columns, and wild sample-brittleness added to limitations.

### Rejected (and why)

1. **Collapsing the mandated headline claim to "a boring table."** The pull quote is a named deliverable of the task with fixed wording; the honesty burden moved into the definition block under it. Rejected softening the quote itself; accepted every factual correction the critic demanded.
2. **Deleting the walk-forward narrative section.** The task's story arc explicitly requires the walk-forward step as narrative; kept it, trimmed, with the honest-scale note moved to the protocol section.
3. **"Wild 10^194 → trash."** The medians and blowup bounds are AUDIT3-certified distribution stats and are mandated numbers ("wild median 11.6M, blowups to 1.8e194"); the seeded re-run cross-check (chaotic mean 1.8e12, wild mean 21.1M / max 91M) was added so the provenance is explicit instead of the number being dropped.
4. **"Put matched-path numbers in the main table."** The surviving table's means are mandated; the AUDIT3 matched-path reconstruction is a different comparator (tuned floor vs `kb_system_gated`, on pre-seed-fix arms). It stays as a precisely-labeled footnote, not the front table.

---

## Round 2 — critique of v2

Prompt: same role; explicitly asked to check quote-vs-table consistency, walk-forward accuracy, internal contradictions, and "what would a skeptical reader attack now."

### What the critic said (grok, verbatim highlights)

> "v2 is a better document… The surviving claim is still a tighter headline than the evidence… A skeptic does not need the audits now; they can attack the **unpaired means, the wrong OOS comparison, and the collapsed 'formula.'**"

> "**'Beats the LLM agent in every environment.'** Noisy is **5,214 vs 5,424 (−3.9%)**, n=10, **different seeds, unpaired**, ranges **4,025–6,246 vs 3,943–7,104**. You later write 'weakest evidence' and 'no paired test.' The quote does not. That is the overclaim."

> "**Pull quote merges two policies:** Walk-forward: per-path 72-point fit on weeks 1–18. Wrapper-only (Fixed demand): frozen (3.0, 0.5), explicitly not tuned… Those are not one object."

> "**Arithmetic in the compute bullet.** 4,320 = 3 × 10 × 36 × 4. The 4-config / 4,320 story does not add up."

> "**'Similar CV in wild' is false.** Walk-forward **0.603** vs verbal **0.429**, gated **0.406**, untuned **0.437**. That is worse, not similar."

> "**The unseen-half check does not support the claim.** …It shows tuning beats the a-priori point on held-out demand (26/30). It does **not** show the self-tuned policy beats verbal/gated out of sample."

> "**Retailer lookahead is the worst of these.** …If the contract is shared, it is **not** an LLM thumb on the scale… You are claiming fairness and bias at once."

> "**Repro snippet will not reproduce the table** (`--seed 7`, `noisy_repro`)… changing the seed is a tell."

> "Delete 'decisive,' 'similar CV,' 'mild,' '1970s,' and the third copy of the autopsy. Fix 4,320 / 1 minute / seed 7. Resolve retailer bias vs shared info — one of those sentences is false."

### Accepted and fixed in v3 (with verification)

1. **Mandated quote restored verbatim** (v2 had softened "1970s"/"state-of-the-art"): kept the task-specified sentence and moved every hedge into a bullet-by-bullet grounding block. "Mild" mapped explicitly to the noisy arm.
2. **Compute arithmetic fixed**: 4,320 calls = 3 API-bound configs × 10 passes × 36 weeks × 4 tiers (`order_up_to` is deterministic, 0 calls); simulation count corrected to ~4,900 after recounting (3 envs × 10 seeds × (81 + 1 + 81)); "3 CPU-seconds" retained as measured, "1 minute" removed from the protocol (it referred to a different script and conflicted). Grid relabeled **81 points (9 θ × 9 λ)** — the audits' "72-point" label does not match either script's grid; the script is the source of truth.
3. **"Decisive" demoted**: chaotic = strongest row (16% mean, lowest CV); wild = mean-better (−9%) but walk-forward CV 0.603 *worse* than all LLM columns — stated as such, with overlapping top-of-range called out.
4. **Dead-claims list #4 rewritten**: neither side wins CV outright (formula: noisy/chaotic; LLM: wild). "Formula wins those too" deleted.
5. **AUDIT3 footnote de-coupled**: labeled as a different comparator (tuned floor vs gated config) on pre-seed-fix arms, directional support only, not p-values for this table.
6. **Suffix check scoped narrowly**: proves tuning isn't overfit vs the untuned point (26/30); no longer implied to answer the LLM-vs-formula OOS question; reworded to "held-out suffix game (fresh state), not the tail of the same trajectory."
7. **Retailer residual contradiction resolved in favor of the shared-information fact**: the retailer's current-week view is shared with the formula (AUDIT3 measured its value to the floor at ~7%); not an LLM-side thumb; "closing it should widen the gap" removed.
8. **Repro seed honesty**: `--seed 7` is the noisy arm's actual seed (CHANGES.md); text now states same-seed ⇒ same demand paths, but LLM costs differ run-to-run by design (CV), so expect a distribution, not the table.
9. **Twin-gap limitation upgraded**: LLM columns' relative ranking marked provisional until the no-mechanism twin gap is explained.
10. **"gate" disambiguated** (confidence gate 0.2% vs verbal gate 3–4%); Ψ/Φ/p95/p99 explicitly "computed, not load-bearing for the surviving claim."
11. **Triplication trimmed**: audit-history table cut to one line per death; story sections slimmed to the numbers the list can't carry.

### Rejected (and why)

1. **"Delete '1970s'… and the third copy of the autopsy."** Both are mandated content: the claim sentence is fixed wording; the story arc (naive win → audit → fix → walk-forward) is the task's core deliverable ("the mishap → audit → fix structure IS the content"). The autopsy list was shortened to one line per audit instead.
2. **"Quote limited to what Limitations 1–2 still allow"** — same mandate reason; the grounding block carries the caveats.
3. **"Paired AUDIT3 numbers should be the honest front table"** — different comparator and different arms; it stays a footnote rather than displacing the mandated means table.
4. **"Drop the OOM from the quote"** — compute is now grounded (tokens + wall-clock, both measured; explicitly "not FLOPs"); the phrase survives as an engineering-cost claim, which the critic conceded is fair.

---

## Post-critique self-audit (v3-only)

Before deploying v3, every number in the text was checked programmatically against the summaries, `AUDIT*.md`, and fresh runs:

- Table means/CVs/on-disk paths: all present and matching (`noisy_leakfree`, `chaotic_seeded`, `wild_seeded` summaries; walk-forward/oracle from a re-run of `walkforward_floor.py` + an extended per-seed analysis).
- **Caught and fixed a fabrication risk in both v2 and v3 drafts:** the held-out suffix table listed untuned (3.0, 0.5) means of 6,689 / 8,912 / 11,383 that no computation had produced. Recomputed: **5,918 / 8,675 / 9,558** (wins 10/10, 9/10, 7/10 unchanged — 26/30). Both archive and final corrected; this log records it.
- Walk-forward CVs (0.131 / 0.339 / 0.603), second-half CVs and win counts: recomputed from the engine.
- Mandated pull quote: byte-verified present verbatim.
- Leftover-overclaim scan for v1/v2 phrases ("7–40%", "favors the LLM", "in the LLM's favor", "decisive", old suffix numbers): clean (only "37–40%" leak claim and corrected content remain).
- `pytest tests/ -q` → 48 passed (unchanged; no code touched).
- `cmp README.md docs/README_v3.md` → identical.

## Result

- `README.md` (v3, final); `docs/README_v1.md`, `docs/README_v2.md`, `docs/README_v3.md`, this log.
- Scope respected: no code or test files modified (`git status` deltas are README/docs only).
- The two grok critiques are archived at `/tmp/grok_critique_v1.txt` and `/tmp/grok_critique_v2.txt` (full text) alongside this digest.