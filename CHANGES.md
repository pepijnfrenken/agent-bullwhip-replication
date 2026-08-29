# Changes & Audit Log

## 2026-08-29 — Publish cleanup + GLM parsing fix hardened

- **GLM order parsing fixed and tested.** `parse_order_prefer_label` (added 2026-08-28, commit `accc0f9`) had two bugs: (1) `\bORDER` matched "order 7 units" inside a THINKING block, silently grabbing the wrong number instead of the `ORDER:` label; (2) `ORDER -3` parsed as `3` because the optional `-` separator consumed the sign. Fixed the regex to require the label at line start and to not treat `-` as a separator. Added `tests/test_label_parser.py` (5 tests) covering THINKING-pollution, negative orders, and all separator variants. **53 tests pass.**
- **Repo hygiene.** Archived 19 one-off audit scripts (check_*, mine_traces*, diag_*, probe_*, postmortem_*, wave2d.sh) into `scratch/` with a README explaining their role; they are not part of the reproduce protocol. Kept the result-producing scripts in the root (`final_noisy.py`, `final_val2.py`, `tuned_floor_chaos.py`, `noisy_arm.sh`, `walkforward_floor.py`, `tuned_floor.py`). Moved `test_chaos_demand.py` into `tests/` (chaotic/wild demand regression coverage).
- **Published state:** README v3, CHANGES.md, AUDIT 1-3, POSTS.md, docs/ — all on `origin/master`.

## 2026-08-26 — Leak-free noisy re-run (final clean dataset)

- Re-ran the noisy-demand arm **with the leak fix in place** (`results/noisy_leakfree/`, interleaved, seed 7, 10 passes × 4 configs, 40/40, 0 failures).
- **Final leak-free verdict:** tuned floor (θ=3.5, λ=0.2) = 4,701 (OOS 4,616–4,970) beats **kb_system_gated 5,973** and **kb_pointer_verbal 5,424** under noise. The tuned formula wins by 10–20% on cost with lower CV (0.092 vs 0.20). `kb_pointer_verbal` ties the *untuned* floor (5,424 vs 5,521, p=0.75) — but that's the deterministic anchor inside it, not model judgment (40% of its decisions are mirror fallbacks).
- Added `oos_floor.py`: out-of-sample split-half tuned-floor check (4,616–4,970 across three splits).
- README TL;DR updated to the final audit-surviving story.

## 2026-08-26 — Post-AUDIT2 fixes

### 1. Same-week information leak removed (AUDIT2 §3, HIGH)
- **Problem:** upstream tiers (wholesaler/distributor/factory) saw `incoming_now` = the tier-below's *current-week* order, computed moments earlier in the sequential decision loop (`engine.py:132,142`). This is a one-week lookahead the deterministic floor doesn't get — it systematically advantaged the LLM. A deterministic floor given the same lookahead scored 2,920–3,042 (37–40% below the LLM), dwarfing the headline −12% "LLM wins under noise" effect.
- **Fix:** `engine.py` now passes `incoming_now=None` for all upstream tiers; only the retailer sees the current-week exogenous customer demand. Prompt builders render "not yet known (you decide on last week's information)" for upstream tiers (`agents.py`).
- **Tests:** `tests/test_no_leak.py` (2 tests) — verifies upstream ctx has `incoming_now is None` and the rendered prompt hides the value. **43 tests pass.**

### 2. Tuned-floor baseline (AUDIT2 §1, HIGH)
- **Problem:** the "LLM beats the formula by 12% under noise" was against the *untuned* a-priori (3.0, 0.5). A 72-point (θ,λ) sweep on the same seeded noisy paths finds (3.5, 0.2) → **4,701**, *below* the LLM's 4,849 in-sample; split-half out-of-sample ≈ parity (4,846).
- **Fix/added:** `tuned_floor.py` — reproducible (θ,λ) sweep + leaked-floor documentation. The honest baseline is now the tuned floor, not the fixed (3.0, 0.5).

### 3. Seeds / CLI provenance documented (AUDIT2 §7, MEDIUM)
- Interleaved runs used seeds: fixed `interleaved_val2` = 42, `wrapper_only` = 42, `noisy_interleaved` = **7** (the only seed ≤30 that reproduces; previously undocumented).
- CLI: `python -m agent_bullwhip.interleaved_runner --configs ... --runs 10 --horizon 36 --outdir results/<set> --seed N [--pattern noisy] [--wrapper-only]`.

## 2026-08-26 — Original AUDIT.md fixes (pre-interleaving)

- **Window drift:** sequential per-config runs confounded by endpoint non-stationarity (16× between adjacent windows on identical prompts). Fixed by the interleaved round-robin runner (`interleaved_runner.py`).
- **Wrapper-only ablation:** added `--wrapper-only` to isolate the deterministic wrapper from the LLM.
- **Cache-buster nonce:** per-call random token in the last user message to defeat serving-level caching.
- **Cross-run prompt-hash check:** moved from per-run to cross-run; now vacuously 0 by construction (nonce).
- **Data integrity:** truncate-on-start + run-UUID (no append-mode mixing); reran mirror/order_up_to.
- **KB snapshotting:** KB text snapshotted into each run record so re-analysis uses the injected text.
- **Per-call metadata:** timestamp + model + nonce + run index logged on every trace.
