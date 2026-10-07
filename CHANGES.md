# Changes & Audit Log

## 2026-10-07 — WAVE6-FIX implemented (mission bullwhip-impl; code + tests + staged kit, NOT run)

`waves/WAVE6-FIX_SPEC.md` is implemented behind explicit flags; every wave-6 default is preserved
(`tests/test_w6fix_contract.py` pins `collect_step` bit-identical to the pre-change code on goldens
recorded before the edit). No GPU/cluster run happened; no wave 1–7 number or verdict was touched.

- `agent_bullwhip/contract.py` (+ `qwen_chat.py`): paper `order_quantity` contract text, chat-template
  render (verified equal to `apply_chat_template(enable_thinking=False)` in an external venv), strict parser.
- `grpo_env.py`: `prompt_builder` hook; `combine_parse_aware` (A2 advantage). `grpo_train.py`: flags
  `--prompt-format --max-new --parse-aware-reward --fmt-weight --dump-completions-every`; per-step
  telemetry (`contract_parse_rate`, `cost_rows_frac`, `fmt_adv_share`, `demand_sha`, …); completion dumps.
- `scripts/w6fix_analysis.py`: V-gates, P25/P50/P100, H1–H5, rules R-A..R-D, paired cost windows.
- Kit: `hpc-umbrella/campaign-w6fix.tsv`, `gpu-w6fix.sbatch` (`--array=0-5`, `W6FIX_STEPS=4` pilot knob),
  `config.sh` `CAMPAIGN_W6FIX_TSV`, `verify_pull.py` EXPECT, OPERATOR.md section.
- Tests 326/15 → 379/16 (new: contract, analysis, sbatch). Deviations/decisions: spec changes log.

## 2026-10-07 — Specs for the two agreed next experiments land in the repo (spec-only; nothing run)

Both specs were written 2026-10-07 (mission `bullwhip-plans`; sources kept at `~/bullwhip-plans/`)
and are now in-repo. Pre-registration drafts: no code changed, no run started, no numbers touched.

- `waves/WAVE6-FIX_SPEC.md` — contract/format micro-probe on the wave-6 free-form lane: **A0** as-is
  reference vs **A1** the paper's own `order_quantity` JSON answer contract applied through the chat
  template vs **A2** plus a parse-aware reward; parse-rate gates decide whether the wave-6 null is a
  training result or a request/parse artifact (≈14–24 GPU-h).
- `waves/WAVE8_SPEC.md` — Clef-style decision-model arm: corrected 26-letter choice set `w8_g26`,
  **R1** outcome-scored RL / **R2** soft-target SFT warm start / **R3** grid variants, computed
  collapse gate, permutation gate, size ladder 0.6B → 1.7B → 4B, plus the GGUF CPU step (≈30–45 GPU-h).
- `waves/SPECS-2026-10-07-README.md` — index for the two: inputs, dependencies, intended order, and
  the note that they are independent except for WAVE8's G3 comparator.
- Implementation follows as mission `bullwhip-impl`; output is meant to be staged by Pino to the
  cluster (the hub has no route to it), so the specs stage exact sbatch/operator commands.

## 2026-10-07 — Repo cleanup + navigation (mission bullwhip-cleanup)

Structural reorganization only — no numbers, verdicts, claims, code behavior, or dependencies
changed; nothing deleted. Full audit of what moved where, the inventory decisions, and the
untracked-binary table: `CLEANUP-REPORT.md`.

- **Navigation docs (new).** `docs/README.md` = how to navigate/run everything;
  `docs/STATE.md` = wave table 1–7, current verdicts, OPEN ISSUES (incl. the wave-6 free-form
  lane validity caveats and the agreed-but-not-started Clef-style arm). Root `README.md` is now
  the short entry point (project statement + binding protocol rules + links); the pre-cleanup
  full text with the surviving tables is archived at `docs/archive/README_v4-2026-10-07.md`
  (old versions v1–v3 + iteration log also under `docs/archive/`).
- **Docs taxonomy.** Wave/spec docs → `docs/waves/`; audits → `docs/audits/`; fixes →
  `docs/fixes/`; verifications → `docs/verify/`; wave-6 mission logs → `docs/missions/`.
- **Scripts taxonomy.** Live tooling → `scripts/`; legacy one-offs (final_*.py, wave2*.sh,
  probe_*/trace analyzers of past waves) → `scripts/archive/`. All references updated; move map
  in `CLEANUP-REPORT.md`.
- **Binary hygiene.** Checkpoint dirs under `results/wave7_{sft,rlcd}_seed*` untracked
  (`git rm --cached`), kept on disk, `.gitignore` extended so they do not come back; summary in
  `docs/ARTIFACTS.md` (sha256, size, how to regenerate).
- **Test parity.** `uv run python -m pytest -q` before = 326 passed / 15 skipped; after =
  identical (captures in the mission dir). Note: bare `uv run pytest` fails at collection on
  this poetry-style pyproject (`agent_bullwhip` not importable) — use `python -m pytest`.
- **Reported, not fixed.** `scratch/` (the 2026-08-29 archive of 19 one-off scripts) was
  removed by commit `99a6c6e` the same day; those scripts survive only in git history.

## 2026-10-07 — Results transfer = git (rule); rsync demoted

Pino: results travel through this repo in both directions, not rsync — a must. New kit script
`hpc-umbrella/push_results.sh` (`-n` preview; arg + missing-path guards; `git add -f` + commit +
push). `OPERATOR.md` Step 6 rewritten around it (was `pull_results.sh`), prereqs line + quota row +
wave-7 §5 updated; `README.md` item 7, `config.sh` comments, the `W7-READY.md` handoff and the
`verify_pull.py` usage line follow. `pull_results.sh` stays only as an emergency bulk mirror of
scratch. Historical notes (`FIXES*`, `VERIFY*`, `00-PROBE`) left as written.

## 2026-09-30 — Pilot readout (351229) + batched-generation padding fix

`gpu-pilot` 351229 **COMPLETED** on Umbrella (L4). All three parts ran: A (the minimal
optimizer step the desktop stalled on), B (`bench_rollout.py` → `results/grpo_bench/bench_351229.json`),
C (4-step GRPO pilot → `results/grpo_a0_pilot_351229/`, checkpoint + state.json + train_log.jsonl).
`train_log.jsonl`'s `elapsed_s` is **cumulative since run start**: 4 steps in 369.5 s ≈
**92 s/step** at G=8, horizon=36, `rollout_batch=16` → 300-step configs ≈ 7.7 h, inside the
18 h array window (final go/no-go on the bench table).

**Bug found in the pilot log:** hundreds of `right-padding was detected` warnings. Nothing in
the repo set `padding_side`, so the batched `generate()` right-padded its prompts. HF's warning
is real here: right-padded batches generate from position-shifted sequences, so the batched arm
would not reproduce the per-prompt (`rollout_batch=1`) path waves 1–5 ran — D1's "batching is
semantically safe" claim would not hold *exactly*. **Fixed:** `_Policy.rollout` left-pads for
generation and restores the previous side in `finally` (training tokenization unchanged).
Bench timings are unaffected (padding side changes positions, not compute); the fix is
re-verified on the cluster by campaign task 0, which re-runs the pilot config under it.

**Bench readout** (351229, L4, G=8 H=36, 2 repeats, real model):

| rollout_batch | step_wall_s | generate_wall_s | µs/decision | tok/s | peak VRAM |
|---|---|---|---|---|---|
| 1 (sequential) | 243.6 | 191.6 | 166 293.5 | 48.1 | 5.29 GB |
| 4 | 113.9 | 61.7 | 53 551.0 | 149.4 | 5.32 GB |
| **16 (default)** | **95.0** | **30.4** | **26 355.1** | **303.5** | 6.19 GB |
| 32 | 93.3 | 31.0 | 26 919.0 | 297.2 | 6.14 GB |

`speedup_vs_baseline`: {4: 2.14, 16: 2.56, 32: 2.61}. Two findings to keep: batch 32 buys
nothing over 16 (within noise; pinned default stays 16), and at batch 16 the **train step is
now the dominant cost** (64.6 s of 95.0 s — the generate share fell 191.6 → 30.4 s, 6.3×), so
further speedup is training-side work, not rollout dispatch. VRAM 6.2 GB of the L4's 24 GB.
Part A: two optimizer steps in 0.77 s, peak 5.99 GB — the desktop stall does not reproduce on
the module stack (torch 2.1.2 + transformers 4.55.4 + TORCHDYNAMO_DISABLE=1).
**Decision:** 13 × 300-step configs ≈ 7.9 h (a0-style) to ≈ 16 h (group16) → the Umbrella
array fits; Snellius stays backup.

## 2026-09-30 — Wave 6 unparked: Batched rollouts, compile hardening, resumable runs, Umbrella kit

Wave 6 was parked on 2026-09-21 (a compile-path stall on the desktop, `PARKED.md`). This entry
is the unpark: the compute route becomes **TU/e Umbrella** (free institutional Slurm; D6 below),
and the three code blockers are fixed. Suite: 102 → **243 tests, all green**. No model weights
were downloaded, no GPU was used, nothing trained — hub CPU work only.

- **D1 — batched rollouts (`agent_bullwhip/grpo_env.py`, `grpo_train.py`).** Rollout collection
  used to submit one prompt per `generate()` call (1 152 calls per step at G=8, horizon 36). It now
  drives all causally independent decisions of a week — the 4 roles × G episodes — through one
  call per `--rollout-batch` prompts: **72 calls per step** at the default (`--rollout-batch 16`;
  16× fewer), or **36** with the whole week in one call (`<=0`; 32× fewer). Why it is
  semantically safe: a week-t decision reads state through t−1 only (engine's AUDIT2 no-lookahead
  rule) and episodes share no state. `--rollout-batch 1` restores the per-decision path for small
  VRAM; `<=0` is unbounded.
- **STEER-01 refactor — one week loop, not two.** The batched driver first mirrored
  `engine.run_game`; the operator ruled for a shared implementation instead, so the numbered week
  steps now live in **`engine.GameStepper`** (`observations()` / `advance()` / `log()`), `run_game`
  is a thin synchronous adapter over it, and `grpo_env` drives the same stepper. Record assembly
  is shared too (`grpo_env.episode_result_from_log`). Guarded from both sides:
  `tests/fixtures/engine_golden.json` + `tests/test_engine_golden.py` (108 cases — 3 demand
  patterns × seeds 1–3 × horizons 12/36 × 3 config variants × 2 action rules — pinning per-week
  orders, costs and totals for the waves 1–5 settings by **exact equality**), plus the batching
  driver-equivalence and same-week-independence tests. Decision record:
  `~/missions/grpo-umbrella/DECISION-REQUEST-engine-mirror.md`.
- **D2 — compile-path hardening.** `grpo_train.harden_compile_path()` sets `TORCHDYNAMO_DISABLE=1`
  *before* `import torch` and clears `generation_config.compile` after model load (setdefault, so
  an explicit operator override wins), and prints the resolved knobs so a pilot log proves the
  state it ran in. `git grep torch.compile` across `*.py`: no hits.
- **D3 — walltime-safe checkpoint/resume.** `--save-every N`, atomic writes (adapter dir via
  `.tmp` + rename, then `state.json` via tmp + rename, so a SIGKILL can never leave a half-written
  checkpoint), `--resume` continuing from `state.json` (adapter + tokenizer + AdamW moments on the
  torch path; step index and RNG state always), and a **SIGTERM handler** that checkpoints and
  exits at the next step boundary — Slurm's pre-walltime SIGTERM is what makes an 18 h job
  survivable. Hub-verified with stub runs (SIGTERM at step 2 → `state.json` step 2 → resume ran
  3–5, no `.tmp` debris).
- **D4 — Umbrella run kit (`hpc-umbrella/` in this repo, vendored 2026-09-30; originally outside at `~/hpc-umbrella/`).** New: `probe.sh` (login-node
  probe → paste-back file), `config.sh` (the one pins file: `--print` / `--pins` / `--check`;
  auto-detects the L4 partition + feature from `sinfo` and fails on any unresolvable pin, including
  sbatch-header/pin drift), `stage.sh` (staging + the two environment tracks), `pull_results.sh`
  (VPN-side rsync off expiring scratch), `requirements-umbrella.txt`, `OPERATOR.md` (run order +
  failure playbook + the Snellius decision arithmetic), and rewritten `cpu-harness.sbatch` /
  `gpu-pilot.sbatch` / `gpu-campaign.sbatch` (all sourced from `config.sh`, with
  `--signal=B:TERM@600` forwarding SIGTERM to python).
- **Kit vendored into the repo (2026-09-30).** The run kit moved from `~/hpc-umbrella/` to
  `hpc-umbrella/` here, so one public `git clone` on the cluster brings code + kit; `PARKED.md`
  is tracked now too (it was cited by `CHANGES.md` + `grpo_train.py` while untracked).
- **First real Slurm submit (2026-09-30, evening).** The pilot exposed what hub-side checks
  cannot: Slurm runs batch scripts from its own spool copy (`/var/spool/slurmd/jobNNN/slurm_script`),
  so `KIT_DIR` from `$BASH_SOURCE` resolved to the spool dir and the job died in 2 s
  (`config.sh: No such file or directory`). Fixed in all three sbatch files: `KIT_DIR` now comes
  from `SLURM_SUBMIT_DIR` (manual-run fallback + loud FATAL otherwise); OPERATOR.md notes the
  submit-from-kit-dir rule.
- **TRACK A version pins settled empirically (2026-09-30).** The first staging run's import check
  caught the module torch 2.1.2 ↔ modern transformers clash: transformers 4.56+ calls
  `torch.utils._pytree.register_pytree_node` (exists only on torch >=2.2) and dies at import.
  Local bisect against torch 2.1.2: 4.55.4 is the newest that imports AND ships Qwen3
  (>=4.51 required). Full rehearsal on the exact set (real Qwen3-0.6B load, LoRA wrap, 3
  optimizer steps: loss 6.2345 -> 6.1563) passed. `requirements-umbrella.txt` TRACK A now pins
  transformers==4.55.4 / peft==0.21.1 / accelerate==1.15.0.
- **Fresh-clone + version-proofing fixes (2026-09-30, night).** The first cpu-harness run after
  re-staging (351112) went fully green — node env sanity, pytest 242 passed, all 40 walk-forward
  floors printed — then died writing `results/walkforward_paths.json`: `results/` is gitignored,
  so fresh clones lack it. Fixed: the script now creates its output dir (verified in a clean dir:
  exit 0, 40 records, step pass 0 matches the cluster print exactly). Pre-flight also caught the
  pilot's part A using the transformers-5.x-only `dtype=` kwarg (hard TypeError on 4.55.4);
  gpu-pilot.sbatch now falls back to `torch_dtype=` with the fallback verified on 4.55.4.
- **cpu-harness green end-to-end + batching check made real (2026-09-30, night).** Run 351155
  completed exit 0 — env sanity, pytest 242, floors + `walkforward_paths.json` write, stub
  dry-run, SIGTERM checkpoint/resume — all on the cluster. Its dry-run had a vacuous batching
  demo (the loop never passed `--rollout-batch`; all three chunks printed identical counts);
  now wired through with exact expected counts `{1: 144, 16: 18, 4: 36, 0: 18}` (week-level
  dispatch of 8 decision-prompts/week at G=2,T=6) as a tripwire — verified torch-free locally.
- **`bench_rollout.py` (repo root).** The pilot's machine-readable numbers: per-step wall-clock,
  µs per `generate()` and per decision, generated tokens, tokens/s, peak VRAM, and
  batched-vs-sequential speedup, written to JSON. `--stub` runs torch-free (verified on the hub).
- **D6 — compute route (spec changes log).** Training this wave runs on **Umbrella** (free,
  institutional), superseding the desktop-only D3; the 4B paper-comparable rung stays parked.
  Modal stays out.

## 2026-09-21 — Wave 6 opened: GRPO on our harness (domain-neutral), spec + env wire

- **Decisions (Pino):** D1 TRL-first/hand-rolled-fallback · D2 ladder = Qwen3-0.6B →
  LFM2.5-1.2B CPU arm → 4B parked · **D3 Modal OUT — all-desktop, $0** · D4 desktop smoke
  first · D5 build the env layer domain-neutral so the recipe generalizes to
  financial-services decision problems (cash buffers, settlement lags, shortfall costs).
- **`WAVE6_SPEC.md`** — pre-registered: paper recipe (§6–7), ladder, success gates
  (CV ↓ ≥25% + mean ↓ ≥20% vs untrained same model = "replicated"; below = partial/null),
  collapse detector, no-step-demand-training rule, compute accounting. Changes after this
  commit get logged in the spec's changes section.
- **`agent_bullwhip/grpo_env.py`** — `DecisionRecord`/`EpisodeResult` over the existing
  leak-free engine + wave 1–5 prompt family (`build_prompt`); system-level return-to-go
  rewards r_{k,t} = −Σ_{τ≥t} c_τ; paper curricula; group-normalized advantage per
  (agent, week) across G episodes (paper Eq. 25, two-pass explicit).
- **`agent_bullwhip/grpo_trl.py`** — rollout aggregation + TRL `reward_fn` glue with a
  hard-fail cursor guard (row-order desync can never train silently); torch imports lazy.
- **Tests (11 new, suite 87 pass):** env shapes, leak-guard on prompts, advantage sign
  (cheap policy reinforced / blowup discouraged), return-to-go geometry pinned (late cost
  propagates to all earlier positions, early cost contained), curricula resample+bounds,
  floor sanity (anchor 3,681 exactly at 36wk on step).
- **Not yet done:** `grpo_train.py` (TRL loop, desktop), vLLM/llama.cpp serving, evals.

## 2026-09-17 — Wave 5: decision-native model arm (Jev, TypeSafe)

- **Agent:** `agent_bullwhip/jev_agent.py` — typed-question decision agent ("System One"): modes `choice_grid` / `choice_mult` / `expectation` / `reads`; structured JSON state payload; all arithmetic in code; confidence gate (τ) + ±`margin` clamp; anchor fallback on any failure.
- **Controls (free, deterministic, 0 API calls):** `anchor_only` (wrapper alone), `gate_always`/`gate_never` (τ endpoints), `jitter_only` (seeded uniform ±margin), `offset_only` (constant ±k), `delta_replay` (the model's own deltas, misaligned). Configs in `runner.CONFIGS`.
- **Analysis tooling:** `walkforward_paths.py` (per-path walk-forward / untuned / oracle floors on the protocol seeds 1000+i), `paired_analysis.py` (per-path paired deltas + t-stats), `analyze_jev_traces.py` (within-run audits — every decision trace stores its own `anchor`; cross-run sequence diffing is invalid in a closed loop), `extract_deltas.py`, `compare_jev_vs_text.py`, `probe_jev.py`.
- **Result (all four arms, 10 passes, 240 runs, 0 failures):** model-only (bounded judgment) improves on the untuned policy by −22.8% / −24.3% / −15.2% (noisy / chaotic / wild) and harms +6.2% on step; it beats the walk-forward floor **only on noisy** (−18.2%, 10/10 paths, t=−3.89) and loses on chaotic/wild (+24.9% / +13.2%, 3/10 each, ns) — i.e. it wins where tuning has little to buy. Gate deferral 42.7% (step) → 85.2% (noisy): the gate closes where the model is valuable. `reads` (best state-level agreement) is worst in every game.
- **Files:** `WAVE5.md` (story + framing audit), `AUDIT4.md` (mistakes + controls in full). Data: `results/jev_protocol/`, `results/jev_controls_*/`, `results/walkforward_paths.json` (results/ is gitignored).
- **Complete (2026-09-17): n=20 noisy extension + ±3/±12 band sweep** (`results/jev_noisy20/`, 120 calls, 0 failures). Paired, per-path: model **4,436.6** vs tuned floor **4,773.4** = **−7.1%, 15/20 paths, t=−2.15** (n=10 read: −18.2%, 10/10, t=−3.89 — magnitude shrank with the bigger sample, direction holds). Band sweep: ±3 **+1.0%** (8/20, ns), ±12 **+18.7%** → default ±6 is the working point. Controls at n=20: anchor-only +15.0%, jitter +41.4% (1/20). Model CV 0.144 vs floor 0.094.

## 2026-08-29 — Publish cleanup + GLM parsing fix hardened

- **GLM order parsing fixed and tested.** `parse_order_prefer_label` (added 2026-08-28, commit `accc0f9`) had two bugs: (1) `\bORDER` matched "order 7 units" inside a THINKING block, silently grabbing the wrong number instead of the `ORDER:` label; (2) `ORDER -3` parsed as `3` because the optional `-` separator consumed the sign. Fixed the regex to require the label at line start and to not treat `-` as a separator. Added `tests/test_label_parser.py` (5 tests) covering THINKING-pollution, negative orders, and all separator variants. **53 tests pass.**
- **Repo hygiene.** Archived 19 one-off audit scripts (check_*, mine_traces*, diag_*, probe_*, postmortem_*, wave2d.sh) into `scratch/` with a README explaining their role; they are not part of the reproduce protocol. Kept the result-producing scripts in the root (`final_noisy.py`, `final_val2.py`, `tuned_floor_chaos.py`, `noisy_arm.sh`, `walkforward_floor.py`, `tuned_floor.py`). Moved `test_chaos_demand.py` into `tests/` (chaotic/wild demand regression coverage).
- **Published state:** README v3, CHANGES.md, AUDIT 1-3, docs/ — all on `origin/master`.

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
