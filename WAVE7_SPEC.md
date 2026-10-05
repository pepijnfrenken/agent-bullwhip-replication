# WAVE7_SPEC.md — Decision-native policy (System-1 readout, outcome-trained) — approved 2026-10-05

**Status:** **approved 2026-10-05 (decisions recorded below).** Nothing below has run on the cluster
yet. Post-approval changes get logged in the changes section at the bottom.

**Lane:** wave 7 (decision-native). Wave 6 (free-form GRPO fidelity line) is **untouched** by this
spec; its eval and verdict stand as reported.

---

## 0. Why this lane, why now

Wave-6 evidence (all on disk, hub-verified):

- ~85% of GRPO training rollouts are unparseable free text; the reward has no parse-awareness
  (`reward-unparsed-findings.md`, 866f336), and the parse rate never improves over 300 steps.
- The a0_base eval aggregate is null by the §4 gates, tail-carried by digit-run rogues
  (`eval-a0-report.md`); the robust read (median, log-space) is positive but far from floors.
- Hub probes (2026-10-05), **corrected by AUDIT-1**: stock Qwen3-0.6B follows the option FORMAT
  (≥99.8% of next-token mass on option letters, one forward pass, no parse failures) but the
  zero-shot argmax is a **letter prior, not a state-driven judgment** — permuting the option→value
  map moves the chosen VALUE (ascending → 14; descending → 40; rotations/shuffles → 8..34), state
  counterfactuals (flooded/starved) barely move it, and an EMPTY state still yields `L`=22 at p=0.50
  (`probe_bias.py`). So: the interface fixes format/parse; **value-sensitivity is what training must
  put in**. The C1 control (untrained readout) is expected to be near-constant; the new
  permutation-invariance gate (AUDIT-1 risk 1) measures what training bought. A generic 79M
  System-1 model (Byrne-Jev) on our domain is flat noise zero-shot → domain training is ours to supply.

**Headline question:** can a decision-native policy (no generation; the answer is a probability
distribution over the allowed orders) trained primarily from **our own simulator's outcome labels**
beat (i) its own untrained readout, (ii) the classic floors, (iii) the wave-6 free-form arms —
paired on identical demand paths?

Precedents and references:
- Wave 5 already tested decision-native via the **hosted** Jev API (typed questions, probabilities,
  no parse failures; `WAVE5.md`), but hosted = untrainable and ToS-bound. Wave 7 makes the same
  interface **local, trainable, open**, on our existing base model.
- Cloudflare **Clef** (Apache-2.0): Qwen3.8-27B + a small "joint schema head" (4 layers, 1024 wide,
  span-routed option scoring); span-tracked record encoding; `/v1/systemone` wire protocol.
- Community **RLCD recipe** (AstroHan, Apache-2.0): LoRA r=8/α=16 on attention projections only
  (~5M trainable params) + letter-logit readout; trained with outcome-scored RL, noise on candidate
  logits + leave-one-out baseline; their `decision.py` ships with `Qwen/Qwen3-0.6B` as default model.
- The grammar-constrained probe on the existing wave-6 checkpoint stays queued as the **format-only
  control** (same model, constrained decoding; isolates format from training).

## 1. Decisions (draft — answer inline)

- **D1 — base model: `Qwen/Qwen3-0.6B`** (already cached on hub and staged on the cluster; fits the
  existing cluster env pins with zero changes; same rung as wave-6 A0, so comparisons are direct).
- **D2 — interface:** one forward pass. Prompt = system instruction + leak-free state (the wave-1–5
  field family) + question + options rendered as single-token letters **A..U → orders 0..40 step 2**
  (21 options). Read the last-position logits of the 21 label token ids, softmax. **Primary
  composition: argmax; secondary: expectation** over the grid (both reported). Invariants asserted
  per batch: labels are single tokens; label mass ≥ 0.99; option order fixed ascending.
- **D3 — labels (simulator is the teacher):** states sampled from the **wave-6 training curricula**
  (Poisson λ~U[5,20]; TruncNormal μ~U[8,20], σ~U[2,6]; resampled per episode — **never eval paths**),
  all 4 roles, all weeks, under a mixed behavior policy (floors + random + wave-6 checkpoint
  snapshots). Per (state, candidate): simulate return-to-go J under a fixed continuation policy;
  targets = softmax(−J/τ). τ set by a stated rule (matched to the candidate-J gap scale; ablated).
- **D4 — training:** LoRA r=8/α=16, attention projections only, base frozen. **Stage 1:** supervised
  soft-target cross-entropy. **Stage 2 (arm):** RLCD refinement — sample K actions from the policy,
  score sampled outcomes by simulation, update with a leave-one-out baseline over the K outcomes
  (public recipe shape). Seeds {0,1,2}.
- **D5 — eval:** the exact protocol arms as waves 1–6: **step n=30 (seeds 1000+i)**, noisy n=10
  (→ n=20 if cheap), paired vs floors and vs the wave-6 arms on identical paths. Extra diagnostics:
  counterfactual sensitivity (± demand, ± inventory state edits), grid regret (best-on-grid vs
  fine-grid optimum), argmax-vs-expectation, parse-fail asserted 0.
- **D6 — serving:** batched in-process torch readout for eval (no HTTP server needed); optional
  local `/v1/systemone` server for protocol-parity demos (drop-in for the wave-5 harness).
- **D7 — compute (cluster):** labeler on **`tue.cpu3.q`** (CPU array, deterministic from seeds — no
  file uploads needed); training + eval forward passes on **`tue.gpu.q -C l4`**. Same kit pins,
  same sbatch conventions (`SLURM_SUBMIT_DIR` lesson), same `pull_results.sh` + `verify_pull.py`.
- **D8 — scope guard:** new lane; wave-6 files/verdicts untouched; all gates pre-registered before
  the first cluster job. 4B/Jev-scale rungs out of scope. Joint-head variant is a post-readout
  stretch (only if readout quality caps out).

## 2. Reuse map (build vs take — the answer to "start from scratch or reuse")

**Reuse as-is (zero changes):**
- `engine.GameStepper` + `run_game` + demand curricula (`grpo_env`) + golden fixture + `test_no_leak`.
- Floors: `walkforward_paths.py`, `walkforward_floor.py`, `tuned_floor.py`, `oos_floor.py` +
  `paired_analysis.py`; analysis: `hpc-umbrella/eval-analyze.py` (its rogue columns will read zero —
  expected and visible proof the interface is fixed).
- `hpc-umbrella/` kit: `config.sh` pins, `stage.sh`, all sbatch conventions, `pull_results.sh`,
  `verify_pull.py`; mission runtime + `train_log.jsonl` logging shape.

**Adapt (same scaffolding, different inner loop):**
- `grpo_train.py` → `decision_train.py`: keep CLI / logging / checkpoint / SIGTERM-resume shape;
  replace generation rollouts with batched readout forwards; replace GRPO advantage updates with
  soft-target CE then RLCD.
- `grpo_env` episode machinery → state-snapshot sampler (reuses stepper + curricula).
- `runner.py` → decision eval mode, emitting the same JSONL schema so the analysis stack is unchanged.

**New:**
- `readout.py` (prompt render, option grids, label ids, mass checks), `labeler.py` (candidate
  simulation → J → targets; multiprocess; deterministic), `decision_train.py` (stages 1–2),
  `decision_eval.py`; optional `/v1/systemone` server.

**Reference-only (copy patterns with attribution; both Apache-2.0):** AstroHan `decision.py`,
Cloudflare `joint_schema_model.py`.

## 3. Design detail

- **State rendering:** identical field text to the wave-1–5 prompts (on_hand, backlog, outstanding,
  incoming_last, incoming_now[retailer only], last_orders) — leak-free engine, upstream
  `incoming_now=None`. One question per decision: "How many units should this stage order this
  week?"; options `A. order 0 units` … `U. order 40 units`.
- **Labeler:** for each state, J(s, a) = realized cost of the remainder of the episode when this
  decision is a and all later decisions (all roles) follow the continuation policy (primary:
  walk-forward tuned floor; sensitivity: naive-zero). Batch/parallel; expected ~0.2 s/state
  single-core; N≈50–100k states.
- **Targets:** p*(a) ∝ exp(−J(s,a)/τ). τ: set so the target entropy sits in a stated band; ablate
  τ ∈ {small, medium, large} on a validation slice (label-side, no cluster needed).
- **Trainer:** batch 32 (packed ~400-token sequences), lr/epochs logged; checkpoints every N steps
  with resume; validation = held-out states (NLL vs targets + top-1 agreement + regret).
- **RLCD arm:** from the SFT checkpoint: sample K=8 actions per state, simulate each, advantage =
  J − mean(J) over the K (LOO); one update per round; 2–3 rounds on a fresh 20k-state subset.
  Same logging; this is an arm, not a risk to the SFT result.
- **Eval:** batched readout on the fixed protocol; JSONL identical to waves 1–6 so `eval-analyze.py`
  and the paired machinery work unchanged; parse-fail==0 is an assertion, not a hope.

## 4. Pre-registered success criteria (draft numbers — set by Pino)

- **C1 (interface + training):** trained readout beats **its own untrained readout** on the paired
  step arm: median cost reduction ≥ 15% AND ≥20/30 wins (sign test p<0.02).
- **C2 (absolute level):** step-arm median ≤ 4× the walk-forward tuned floor (~13k) = strong tier;
  median ≤ 2× = headline. Report the actual number regardless.
- **C3 (vs wave-6):** beats the wave-6 a0_base trained arm on identical paths, paired (≥20/30 wins).
- **C4 (reproducibility):** 3 seeds agree within combined CI; parse-fail == 0; artifacts committed
  (adapters + logs + labels) and pulled/verified.
- **R1 (RLCD arm):** report SFT-vs-RLCD paired; adopt RLCD only if it wins beyond noise, else the
  simpler SFT line is the story.
- **Null clause:** if C1/C3 fail, the interface-alone hypothesis dies at 0.6B and gets reported as
  such (consistent with repo standards).

## 5. Traps & controls

- **Surface-statistic shortcut** (echoing digits from the state): counterfactual edits (±demand,
  ±inventory) + grid regret + held-out response-slope checks.
- **Labeler policy dependence:** targets depend on the continuation policy — two-policy agreement
  report on a validation slice; primary = wf_tuned floor.
- **Closed-loop distribution shift:** iterated relabeling is an optional round 2 (collect visited
  states → relabel → retrain); primary dataset = mixed behavior policies.
- **Tokenization/labels:** single-token assertions for A..U in Qwen3-0.6B; fixed option order;
  option-shuffle control arm (quantifies order sensitivity, known from the Byrne card).
- **Version drift hub↔cluster:** hub transformers 4.57.6 vs cluster 4.55.4 — chat-template
  `enable_thinking=False` and related API compat must be pinned/asserted on the cluster (audit
  item); fallback = pre-rendered prompt constant, no template API needed.
- **Eval demand hygiene:** labeler draws curricula paths only; eval paths never enter training —
  same guard as wave 6, plus a test.

## 6. Compute & budget

| phase | where | est. |
|---|---|---|
| labeler (N≈50–100k × 21 candidates) | cluster `tue.cpu3.q` array (deterministic from seeds) | hours, free |
| SFT + RLCD (3 seeds × 2 stages) | cluster `tue.gpu.q -C l4`, one L4 per leg | ~6–12 GPU-hours total, free |
| protocol evals (step 30 + noisy) | cluster GPU (batched forwards) | minutes |
| analysis + report | hub | hours |

No new spend: no paid clouds, no downloads beyond cache (Qwen3-0.6B already staged; labeler data is
generated on-cluster).

## 7. Deliverables

- `agent_bullwhip/decision_*.py` + tests (readout determinism, labeler, loss shapes).
- `results/wave7_<arm>/`: `train_log.jsonl`, adapters, eval JSONLs + summaries; pulled & verified.
- `WAVE7.md` results doc (repo voice) + audit pass before any post material (drafts never in repo).
- README: wave-7 section; this spec's changes log updated.

## 8. Decision points for Pino

- **D1/D2:** base + grid 0..40 step 2 + argmax-primary OK?
- **D4:** RLCD as a wave-7 arm (recommended) or supervised-only first?
- **C-gates:** C1/C2/C3 numbers — set them.
- **D7:** cluster-permission notes (queue limits, per-person caps?) for the session.

## Decisions (2026-10-05, by Pino)

- **Wave 7 approved** ("sounds very good"), commissioning the cluster move once the build is green.
  Drafted defaults stand; changes get logged below.
- **D1/D2:** `Qwen/Qwen3-0.6B`; option grid 0..40 step 2 (A..U); primary composition = argmax;
  expectation reported as secondary.
- **D4:** RLCD runs as the second-stage arm; the SFT line is the main result, R1 decides whether
  RLCD joins the story.
- **C-gates as drafted:** C1 ≥15% median reduction AND ≥20/30 paired wins vs the untrained readout
  (sign p<0.02); C2 ≤4× the wf_tuned floor (≈13k) = strong tier, ≤2× = headline; C3 ≥20/30 paired
  wins vs the wave-6 a0_base trained arm; C4 seed agreement + parse-fail==0 + artifacts committed.
- **D7 (cluster session):** standard kit flow (git pull → `stage.sh` → sbatch); labeler on
  `tue.cpu3.q`; train + evals on `tue.gpu.q -C l4`. The "pull" ping comes from the operator once
  build + audits are green, with exact commands.

## AUDIT-1 amendments (2026-10-05) — required before the first cluster job

The adversarial design audit `WAVE7-AUDIT-1.md` (complete; probes in `/tmp/w7audit/`) requires the
following changes. The build treats the audit as its fix-list and notes each fix in its commit
messages; this section amends §1–§7 by reference (the audit's checklist is the pre-flight gate).

1. **§0 premise corrected** (letter prior, not coherent readout; edit above). NEW GATE: after
   training, the argmax *value* is invariant under ≥3 option permutations on ≥90% of validation
   states; permuted-option augmentation in training; shuffle control arm retained.
2. **D2 interface:** renderer derives from `observations()` only (golden prompt-SHA at pinned
   revision `c1899de289a04d12100db370d81485cdf75e47ca`); `dtype=` forbidden on 4.55.4 →
   `torch_dtype` + TypeError fallback; left-padding + `logits_to_keep=1`; batch-vs-single equality
   test; mass ≥0.99 stays fatal.
3. **D3 labels:** E[J] over M≥16 common-random-number futures (not single-draw; 32% argmin
   disagreement); continuation forecast-replayed from week 0 (stateful `_forecast`; 47%); fix
   (θ,λ) once tuned out-of-sample and record the values here when frozen (current walkforward
   machinery self-tunes on the same path = leakage); grid edge-rate gate (widen or state-relative;
   42% of argmins at the edge); mask t ≥ H−lead flat states; τ chosen by closed-loop cost on
   curriculum validation, rule pre-registered.
4. **D4 RLCD:** advantage A = −(J−b)/std (J is a cost — spec draft had the sign wrong); baseline
   excludes self; score all 21 actions exactly (exact expectation gradient, zero variance);
   RLCD states from current-policy rollouts; explicit epochs/lr/KL-anchor to SFT.
5. **D5 eval:** step arm is n=1 (unseeded demand + deterministic argmax) → report as a single
   labeled cost; ADD a sampled-readout arm (seeded 1000+i) as the counterpart of wave-6's T=0.7;
   gates (C1/C3/C4) move to noisy n=30; regenerate `walkforward_paths.json` for 30 passes;
   C2 thresholds written: 12.8k = strong (4× wf_tuned), 6.4k = headline (2×).
6. **Reuse seams to fix:** eval-analyze adaptations (failures dict, tie handling — false p≈1.9e-9
   on identical policies, arm naming); paired_analysis mtime pattern; verify_pull EXPECT;
   save_checkpoint args + torch RNG + data cursor/epoch; `config.sh --check` CPU rule →
   `cpu-*.sbatch`; `HF_HUB_OFFLINE=1` in job env; stage.sh master fast-forward + EXPECTED_COMMIT +
   weight-revision pin.
7. **Mandatory relabel round** on learned-policy states before reading C1–C3 (closed-loop shift
   is 8's label flaw, no longer merely monitored).

## Changes log

- 2026-10-05: draft written (post-wave-6 a0 eval + reward findings; hub readout probes; Clef/RLCD
  references reviewed).
- 2026-10-05: approved by Pino; decisions recorded; status → approved.
- 2026-10-05: AUDIT-1 amendments recorded (above) after the adversarial design audit falsified the
  zero-shot premise (letter prior) and the step-n=30 assumption, and pinned the first-cluster-job
  crash causes. Spec §0 corrected inline; all other changes tracked in the amendments section.

### Wave-7 build changes log (implementation notes, 2026-10-05)

- Continuation floor for labels: FIXED (theta=3.5, lam=0.2), tuned
  out-of-sample on 60 curriculum episodes (seeds 4000..4059), validated on 60
  held-out (4060..4119): 8,766 / 9,066 mean cost vs untuned (3.0,0.5)
  15,544 / 15,997. No per-path walk-forward search (that leaks future demand —
  AUDIT-1 §1 fix 4).
- Labels: E[J] over M=16 CRN futures from each episode's OWN curriculum params
  (AUDIT-1 §1 fix 1); per-state tau = tau_scale * max(std_a(E[J]), 1e-3),
  tau_scale=0.5 default (ablate 0.25/0.5/1.0); late weeks t >= H-2 excluded;
  labeler seed 6000, eval band [1000,1030) rejected. Measured edge-argmin rate
  ~0.68 on a smoke run — grid truncation is a live risk (AUDIT-1 §1 fix 3):
  the sbatch prints a WARN above 0.9 and `--p-random` can lower extreme
  behavior states.
- Readout prompt hash fixture b96fb847... (pinned snapshot
  c1899de289a04d12100db370d81485cdf75e47ca); label ids resolved on the RENDERED
  continuation; left padding + logits_to_keep=1 (probed; falls back to
  logits[:, -1]); mass >= 0.99 fatal.
- RLCD arm: A = -(J - b)/std with LOO baseline excluding self; exact all-21
  expected-cost PG by default (--rlcd-sample-k K restores the sampled recipe);
  beta_kl=0.02 anchor to pi_sft; states collected by CURRENT-policy rollout +
  relabel (DAgger round 2) before RLCD.
- Eval: step-argmax is the single deterministic game (n=1, labeled as such —
  AUDIT-1 §5 P0); the n=30 distributional step arm samples from the readout
  (seeds 1000+i); inferential gates move to noisy n=30.
