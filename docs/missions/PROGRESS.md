# PROGRESS — Wave 6: grpo_train.py refactor, tests, dry-run (hub side)

Mission state: **COMPLETE** (2026-09-21, hub CPU). Refactor done, 15 new tests,
suite green (102), 3-step stub dry-run logged and verified. No model weights
downloaded, no GPU, no Modal (D3 respected).

## What changed and why

`../../agent_bullwhip/grpo_train.py` was a one-pass draft with dead code and no test
coverage. Refactored to a torch-free, policy-agnostic loop:

1. **Dead code removed** — `collect_step` had four unused fragments (`weekly_prompts`,
   `make_pol`, `batch_buf`, the two-phase `pol` closure) around the actually-used
   `pol_simple` path. Now a single documented per-decision rollout path (`pol`),
   with a comment on why decisions are not cross-agent batched (engine drives the
   loop synchronously; 4x single-prompt calls are cheap at 0.6B; revisit only on
   a profile). Unused `import math` dropped; `re` hoisted to module level.
2. **`SimConfigHorizon` shim deleted** — it added nothing over
   `SimConfig(horizon=...)` (a plain dataclass). `collect_step` now builds
   `BeerGameEpisode(SimConfig(horizon=horizon))` directly.
3. **`policy.model.load_adapter if False else None` leftover removed** from the
   resume path. Also fixed a real resume bug found while there: `--resume` assigns
   a `PeftModel` to `self.model`, and `grpo_step`'s lazy wrap would then call
   `get_peft_model()` on the already-wrapped model → stacked double LoRA. Now
   guarded with `isinstance(self.model, PeftModel)`.
4. **KL reference fixed (decision below).**
5. **`PolicyLike` protocol introduced** (duck-typed: `rollout(list[str]) ->
   (seqs, texts)`, `grpo_step(...) -> float`). `collect_step` is typed against it;
   tests and the dry-run driver inject `_StubPolicy`. The torch-backed `_Policy`
   conforms to it and keeps every torch/peft/transformers import in method scope,
   so the module imports and runs on a torch-less hub.
6. **Loop made drivable without `main()`** — the step loop moved into
   `run_loop(policy, args, log_path, ckpt_path, start_step, rng, has_torch)`.
   `main()` parses args, short-circuits `--steps 0` (rc 0, no dir/log), builds the
   real policy, handles resume, then calls `run_loop`. Checkpoint saves are gated
   by a test-controllable `has_torch` guard (default `_torch_available()`).

## KL reference decision (the paper's `pi_ref`, Eq. 25)

**Problem:** the draft computed the reference as
`model.get_base_model()(input_ids=...)` inside `no_grad`. In PEFT, LoRA layers are
injected *inside* the base model's linears — `get_base_model()` unwraps the
`PeftModel` but the adapters stay attached and active, so that forward produces
**post-LoRA logits**: the KL penalty would compare the policy to itself and
silently collapse.

**Decision:** use `model.disable_adapter()` — a `PeftModel` context manager that
temporarily disables adapter layers so the forward is the frozen base (pre-LoRA).
Verified the API exists in peft main (`src/peft/peft_model.py`) and v0.11+
package docs; issue huggingface/peft#367 confirms it existed by ~peft 0.4/0.5.
Comment in code pins the requirement: **peft>=0.5.0** on the desktop venv. The
reference is computed with `disable_adapter()` + eval mode + `no_grad` (restores
train mode after). If an old peft lacks the API, `grpo_step` **raises** instead of
silently using a wrong reference — a wrong KL target is worse than a hard stop.

## Tests (before → after)

- Before: **87 passed** (baseline, verified pre-change).
- After: **102 passed** (87 + 15 new in `../../tests/test_grpo_train.py`,
  `pytest tests/ -q`, 8.7s). Style follows `../../tests/test_grpo_env.py` (stub
  policies, no model calls, deterministic).

New coverage: collect_step shapes (G*4*T records) + per-(agent, week)
standardized advantages (mean 0, std ≈ 1); seed reproducibility (same seed →
identical prompts/completions/advantages and decision counts; different seed →
different signal); curriculum alternation (poisson vs truncnorm, same seed →
different demand-driven advantage patterns, completions identical since the stub
emits the same text); `extract_int` (digits, junk-wrapped, signed, negative-looking
junk, empty); `--steps 0` short-circuit (rc 0, no dir/log); arg parsing defaults;
`run_loop` JSONL one line per step with expected keys + curriculum rotation;
checkpoint saving skipped with `has_torch=False` and performed (model + tokenizer)
with `has_torch=True`; `_torch_available()` false on the hub.

## Dry-run proof (hub CPU, stub policy)

Driver `/tmp/grpo_dryrun_driver.py` (StubPolicy, horizon 6, G 2, 3 steps,
curricula rotation, `has_torch=False`). Output `../../results/grpo_dryrun/train_log.jsonl`
(results/ is gitignored; no checkpoint dir created — save correctly gated off):

```json
{"step": 0, "curriculum": "poisson", "loss": 0.123, "mean_adv": 0.0, "min_adv": -0.9999996015937842, "max_adv": 0.9999996015937842, "elapsed_s": 0.0}
{"step": 1, "curriculum": "truncnorm", "loss": 0.123, "mean_adv": 0.0, "min_adv": -0.9999986111130401, "max_adv": 0.9999986111130401, "elapsed_s": 0.0}
{"step": 2, "curriculum": "poisson", "loss": 0.123, "mean_adv": 0.0, "min_adv": -0.9999983333361111, "max_adv": 0.9999983333361111, "elapsed_s": 0.0}
```

Evidence: 3/3 lines written; curriculum alternates poisson/truncnorm/poisson;
advantages group-standardized (mean 0, |min|/|max| ≈ 1) — Eq. 25 normalization
observable through the whole loop with a stub policy; per-step loss recorded.
Engine/episode mechanics (costs, prompts) already covered by test_grpo_env.py.

## Known limitations (next-mission scope)

- **Torch path unexecuted on the hub** (no torch/peft here): `_Policy.rollout`,
  `grpo_step` (incl. the `disable_adapter()` KL reference) are verified against
  the peft/transformers API contract only — the desktop 0.6B smoke is the first
  real execution. `disable_adapter()` requires peft>=0.5.0 (pin at desktop venv
  install).
- `grpo_step` accepts `clip=0.2` but the PPO-style clip is a no-op with a single
  inner epoch (ratio ≡ 1 pre-update); kept as a documented placeholder for the
  trust-region guard, consistent with the draft.
- Reference forward recomputed per micro-batch (base is frozen → could be cached
  across steps; not done — 0.6B pilot doesn't need it).
- `collect_step` issues rollouts one decision at a time (4x more generate calls
  than an ideal weekly batch); documented in-code, revisit on a profile.
- No bug found in `../../agent_bullwhip/grpo_env.py` / `../../agent_bullwhip/grpo_trl.py` (frozen as required); nothing to
  report upstream.
- `--steps 0` prints `steps<=0: nothing to do` and exits rc 0 without touching
  `--out` (deliberate; tested).

## Commit

`[verified]` commit contains exactly `../../agent_bullwhip/grpo_train.py` +
`../../tests/test_grpo_train.py`. Not pushed. This file stays untracked.

## Launch command for the NEXT mission (desktop 0.6B smoke)

Inside the wave6 venv on the desktop (torch 2.5.1+cu121, peft>=0.5, transformers,
trl; model weights cached, no download), from the repo root:

```bash
python -m agent_bullwhip.grpo_train --model Qwen/Qwen3-0.6B \
    --steps 30 --group 8 --horizon 36 --out results/grpo_a0_smoke --resume
```

Smoke = 30 steps (~1h-ish on the 3060 Ti at G=8, 0.6B); then eval the checkpoint
on the MIT step protocol (never trained on step demand — WAVE6_SPEC §3.4). Resume
is safe (checkpoint + JSONL step count), so the box powering off at night is fine.
---

# PROGRESS — 2026-09-30: Slurm-ready GRPO path + Umbrella kit (P1–P3, hub CPU)

Mission: `grpo-umbrella` (runtime-logged at `~/missions/grpo-umbrella`). Full summary:
`~/missions/grpo-umbrella/SUMMARY.md`. No GPU, no cluster ops, no weights downloaded, nothing trained.

## P1 — batched rollouts (D1), commit 68b1298
- `collect_step` used to submit one prompt per `generate()` (1 152 calls/step at G=8, T=36). Now the
  4 same-week roles × G episodes go through one call per `--rollout-batch` prompts: **72 calls/step**
  at the default 16 (36 with the whole week in one call; 16×/32× fewer). Safety argument: a week-t decision reads state through t−1 only
  (engine AUDIT2 no-lookahead) and episodes share no state. `--rollout-batch 1` = per-decision path.
- `extract_int` moved to `grpo_env` (env-side parse: text → order).
- 21 new tests. Dry-run (stub, 3 steps): 144 → 18 calls, JSONL identical at chunk 1/16/0.

## STEER-01 — one week loop (operator decision), commit 26b0ebd
- Golden capture FIRST: `../../tests/fixtures/engine_golden.json` (108 cases) + `../../tests/test_engine_golden.py`
  asserting **exact equality** of per-week orders/costs/totals → the waves 1–5 comparability guard.
- Then the refactor: week steps → `engine.GameStepper`; `run_game` = thin adapter; `grpo_env`
  drives the same stepper; the ~80-line mirror deleted; record assembly shared.
- Suite 134 → 243 green; dry-run outputs unchanged.

## P2 — compile hardening + walltime-safe resume (D2/D3), commit 66d4658
- `harden_compile_path()`: `TORCHDYNAMO_DISABLE=1` before `import torch`, `generation_config.compile
  = False` after load, resolved knobs printed. `git grep torch.compile`: 0 hits.
- `--save-every`, atomic tmp+rename for adapter and `state.json`, `--resume` from `state.json`
  (adapter+tokenizer+AdamW on the torch path; step+RNG always), SIGTERM handler → checkpoint +
  clean exit. Proof run: SIGTERM at step 2 → resume ran 3..5, no `.tmp` debris.

## P3 — Umbrella kit (`~/hpc-umbrella/`, outside the repo) + `../../scripts/bench_rollout.py` (commit e3f38cf)
- `../../hpc-umbrella/config.sh` (single pins file; `--print`/`--pins`/`--check`; auto-detects the L4 partition+feature,
  fails on sbatch-header drift), `../../hpc-umbrella/probe.sh`, `../../hpc-umbrella/stage.sh` (two env tracks), `../../hpc-umbrella/pull_results.sh`,
  `../../hpc-umbrella/requirements-umbrella.txt`, `../../hpc-umbrella/OPERATOR.md` (run order + failure playbook + Snellius arithmetic).
- sbatch set rewritten to source `../../hpc-umbrella/config.sh`, with `--signal=B:TERM@600` forwarded to python.
- `../../scripts/bench_rollout.py` (repo): per-step wall-clock, µs/`generate()`, tokens/s, peak VRAM, speedup JSON;
  `--stub` verified torch-free on the hub.
- Verified off-cluster: `bash -n` on all scripts, `config.sh --print/--pins`, `--check` fails fast,
  `stage.sh --dry-run` end-to-end, `../../hpc-umbrella/pull_results.sh` reachability preflight, `../../hpc-umbrella/probe.sh` in a sandbox
  `$HOME` (degrades gracefully, writes the paste-back file).

## Not done here / next
- The torch path still has never executed anywhere: the pilot's step A is its first real run.
- Pins marked `assumed` in `../../hpc-umbrella/config.sh` need the probe output (module names, walltime caps, L4 feature).
- 4B/Snellius rung stays parked (WAVE6_SPEC D6).
