# WAVE6_SPEC.md — GRPO post-training, on our harness (DRAFT — awaiting approval)

**Status:** pre-registration draft. Nothing below has been run. Per repo rules this file is
committed only when Pino approves it, and changes after that get logged in a changes section.

**Headline question (decided with Pino):** the **size/cost axis IS the result** — does the
paper's GRPO reliability fix replicate on our harness, and how does the effect scale across
model sizes (0.6B → 1.2B → 4B) and serving costs (GPU API → local CPU)?

---

## 1. What we're testing

The paper (arXiv:2605.17036 §6–7) claims GRPO post-training turns a volatile general LLM into
a reliable inventory agent: Qwen-3 4B mean cost 1,585 → 952 (−40%), CV 26% → 13% (−50%), max
2,847 → 1,353 (−52%) over 30 runs on the MIT step demand. No code or checkpoints are released.
Our repo currently says their *mechanism* is "untested on this endpoint" (README §limitations).
Wave 6 closes that hole, then extends it along two axes the paper does not have:

- **Scale:** does the reliability effect survive at 0.6B / 1.2B / 4B?
- **Cost:** a trained tiny model served on CPU (llama.cpp, $0/run) vs our tuned floors —
  the paper's "expensive, out of reach" line rewritten as "one-time cost, then free."

## 2. The paper's recipe (build target, from §6–7)

- One shared LLM backbone across all 4 tiers. Centralized training, decentralized execution.
- G episodes (rollouts) per training step under the current policy.
- Reward: system-level cost, return-to-go attribution per agent-week
  (r_t = −Σ_{τ≥t} c_τ), group-normalized advantage (Eq. 25: normalize across the G episodes
  per (agent, week) position), KL penalty to frozen reference (β), gradient clipping, AdamW.
- Demand curriculum: Poisson(λ~U[5,20]) per episode; TruncNormal(μ~U[8,20], σ~U[2,6], [0,50]).
- Eval: MIT step (4,4,4,4,8…8), 30 runs, report mean cost / CV / max.
- **Not published:** G, lr, steps, β, max_tokens, LoRA-or-full. We choose and document ours.

## 3. Design

### 3.1 Environment wire (new: `agent_bullwhip/grpo_env.py`)

- One episode = 36 weeks × 4 tiers = 144 decisions via `engine.run_game`.
- Prompt = the exact leak-free ctx fields `agents.py` already builds (on_hand, backlog,
  outstanding, incoming_last, incoming_now[retailer only], last_orders) — same prompt family
  as waves 1–5, so the comparison stays inside our harness. Paper prompts (App. A.2) are
  reference, not adopted; we replicate the mechanism, not their scaffolding.
- Completion = the order quantity (short integer output, max_tokens small). Token logprobs
  captured per decision; joint episode loglik = Σ over agent-weeks (paper Eq. 25 factorizes
  exactly this way).
- Determinism: demand drawn from the paper's two curricula with logged seeds; eval always on
  our fixed protocol paths (step + noisy seeds 1000+i) for comparability with waves 1–5.

### 3.2 Training loop (the crux — decision point D1)

- **Option A (default):** TRL `GRPOTrainer` with a custom rollout source that runs
  `run_game` per generated episode. Pro: maintained, correct GRPO/KL/clip internals.
  Con: multi-turn custom-env friction is real (known TRL pain point).
- **Option B (fallback, ~200 LOC):** hand-rolled loop — sample G episodes, compute
  group-normalized return-to-go advantages per (agent, week), one AdamW step on
  Σ Adv·logπ − β·KL. With LoRA at small scale this is fully tractable and 100% auditable.
- Choose A first; if the custom-rollout integration burns >1 day, switch to B and document.

### 3.3 The ladder (decided: size/cost axis)

| rung | model | trains on | serving eval | purpose |
|---|---|---|---|---|
| A0 | Qwen3-0.6B | desktop 3060 Ti (QLoRA 4-bit) or Modal | CPU llama.cpp Q4 | pipeline shakedown, $0 |
| A1 | LFM2.5-1.2B (already cached) or Qwen3-1.7B | desktop or Modal | CPU llama.cpp | mid-rung + the CPU-economics arm |
| A2 | Qwen3-4B (paper's base) | Modal H100 (LoRA bf16) | vLLM | the paper-comparable headline |

- Ladder science prefers one family (Qwen3-0.6B/1.7B/4B). LFM2.5-1.2B is the *deployment*
  arm (edge/CPU story, already on the desktop disk). Decision point D2: keep both 1.2B-class
  arms (recommended, +1 small run) or family-only.
- Every rung evaluated identically: untrained baseline vs trained, 30 runs, MIT step +
  our noisy/chaotic/wild arms (10 paths paired, protocol seeds).

### 3.4 Controls & diagnostics (all pre-registered)

1. **Floors on the same paths:** untuned (3.0, 0.5), walk-forward tuned, oracle — the
   existing `walkforward_paths.py` machinery, zero new code.
2. **Collapse detector:** order-correlation-vs-floor + per-(agent,week) order variance.
   A trained model that "wins" by outputting a near-constant order is reported as such
   (distilled-formula world), not celebrated.
3. **Curriculum-transfer check:** eval on step demand only after training on the paper's
   Poisson/TruncNormal curricula — no step-demand training runs (that would be our old
   in-sample sin with extra steps).
4. **Leak guard:** prompts come from the post-AUDIT2 engine (upstream `incoming_now=None`);
   `test_no_leak.py` stays green.
5. **Seed discipline:** G rollout seeds, curriculum seeds, eval seeds all logged per step;
   final eval = protocol seeds 1000+i, paired.

## 4. Pre-registered success criteria

- **Replication (4B, MIT step):** direction required on all three paper metrics
  (mean ↓, CV ↓, max ↓). "Replicated" = CV drops ≥25% relative and mean drops ≥20% relative
  vs the same model untrained (half the paper's effect size — we use LoRA not full FT,
  smaller budget, different scaffolding). Below that: "partial/null," reported as such.
- **Scaling claim:** the ladder is described with per-rung effect sizes; we only claim
  "monotone-ish" if 0.6B < 1.2B/1.7B < 4B on the CV reduction.
- **CPU-economics claim:** tokens/s measured (not assumed), $0 marginal/run, and the
  trained-CPU arm's cost table lands within the WAVE5 framing (wins reported only vs the
  comparators it actually beats; tuned floor is the honest benchmark, stated up front).
- **Null-result clause:** if GRPO destabilizes or fails to improve at our budget, the wave
  publishes that honestly ("mechanism did not transfer at $X / N steps") — consistent with
  the repo's standards. A null at 0.6B with a positive at 4B is itself the scaling result.

## 5. Compute & budget

| phase | where | est. cost | est. wall-clock |
|---|---|---|---|
| env wire + unit tests | hub | $0 | 1–2 days of work |
| A0 smoke + pilot (0.6B) | desktop GPU, then CPU serve | $0 | 1 day (nights, resume-capable unit) |
| A1 mid rung | desktop (if it fits) else Modal | $0–30 | 1–2 days |
| A2 4B run | Modal H100 | ~$150–500 | 1–3 days |
| evals (all rungs) | hub CPU (engine is free) | $0 | hours |

Desktop guards (from the Sep 20 probe): **C: is at 98% (25GB free)** — all model downloads
land in the WSL ext4 (~681GB free), never /mnt/c; fresh venv (torch 2.5.1+cu121 already
works there; add trl/peft/transformers-compatible set); long runs in the existing
systemd+checkpoint pattern; the box powers off at night, so checkpoints are mandatory.

## 6. Deliverables

- `agent_bullwhip/grpo_env.py` + `grpo_train.py` + tests (engine-side tests stay green)
- `results/grpo_<rung>/` (JSONL, per-step reward curves, final eval summaries)
- `WAVE6.md` — results doc in repo voice (numbers, audits, dead claims as they die)
- README: wave-6 section replacing "mechanism untested on this endpoint"
- Compute-accounting table: formula (~3 CPU-s) vs API arms (2.73M tokens) vs trained arms
  (one-time GPU-hours + marginal $0 CPU serving) — the wave's signature table
- Audit pass (AUDIT6) before any external post material is drafted; post drafts stay out of
  the repo (standing rule)

## 7. Decision points for Pino (answer inline)

- **D1 — training loop:** TRL first with hand-rolled fallback (my rec), or straight hand-rolled?
- **D2 — ladder:** family-only (Qwen3 0.6/1.7/4) vs adding LFM2.5-1.2B as the CPU arm (my rec: add it)?
- **D3 — budget cap:** authorize Modal up to ~$300? ~$500? (hard stop recorded here before any run)
- **D4 — pilot order:** desktop 0.6B smoke first while Modal prep happens in parallel (my rec)?

## Decisions (2026-09-21, by Pino)

- **D1 — training loop:** TRL `GRPOTrainer` first; hand-rolled ~200-LOC fallback if the
  custom-rollout integration stalls >1 day.
- **D2 — ladder:** add **LFM2.5-1.2B as the CPU arm** (already in the desktop HF cache);
  Qwen3-1.7B optional. Ladder = Qwen3-0.6B → LFM2.5-1.2B → (parked) Qwen3-4B.
- **D3 — Modal is OUT.** No paid cloud compute this wave. All-desktop; see how far $0 gets.
  The 4B paper-comparable run is **parked** — unlocks only by an explicit new decision
  after desktop results are reviewed. The spec's Modal budget table stays as reference.
- **D4 — pilot order:** desktop Qwen3-0.6B smoke first; loop proven before any spend.
- **D5 — generalization constraint (added on Pino's ask):** the env/training layer is built
  domain-neutral (`DelayedInventoryDecisionEnv`) so the same GRPO recipe generalizes to
  **financial services** decision problems with the same geometry — inventory/cash buffers,
  settlement lags (T+2), unfilled-order backlogs, carry/shortfall costs, order-flow
  curricula. The beer game is instance #1; a finance env is instance #2 without touching
  the training loop. Mapping documented in `grpo_env.py`.

## Changes log

- 2026-09-21: decisions D1–D5 recorded (D3 supersedes the Modal track); draft written.
- 2026-09-30 — **D6 (compute route): this wave trains on TU/e Umbrella**, the free institutional
  Slurm cluster, superseding the desktop-only reading of D3 (Modal remains OUT). The desktop stall
  parked the wave (`PARKED.md`); the unpark fixes the three code blockers first — batched rollouts
  (D1 above / `--rollout-batch`, one `generate()` per week instead of per decision), the compile
  path disabled (`TORCHDYNAMO_DISABLE=1` + the generation-compile knob), and walltime-safe
  checkpoint/resume with a SIGTERM handler — and moves training to `tue.gpu3.q` (8× L4, 24 GB each).
  The **4B paper-comparable rung stays parked** (unlocks only by an explicit new decision after the
  0.6B results are reviewed). Nothing in §1–§5 of this spec (recipe, ladder, success criteria,
  pre-registered controls) changes; only where the compute comes from. The run kit lives in this
  repo at `hpc-umbrella/` (`OPERATOR.md`; vendored 2026-09-30 — the cluster needs one public
  `git clone`), and the run book is: probe → pin one file → stage → cpu-harness → gpu-pilot →
  campaign array.
- 2026-09-30 — **STEER-01 (internal refactor, no scope change):** the batched rollout driver shares
  the engine's week loop (`engine.GameStepper`) instead of mirroring it; `run_game` is a thin
  adapter over the same stepper. Behaviour is pinned by a new waves-1–5 golden fixture
  (`tests/fixtures/engine_golden.json`, 108 cases, exact equality on per-week orders/costs/totals),
  so the refactor cannot have moved any number the earlier waves depend on. Full reasoning:
  `~/missions/grpo-umbrella/DECISION-REQUEST-engine-mirror.md`.
