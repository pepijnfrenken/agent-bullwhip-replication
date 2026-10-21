# WAVE8_SPEC.md — a decision-model arm in the Clef style: outcome-scored RL on a corrected choice set (DRAFT — pre-registration, nothing below has run)

**Status:** IMPLEMENTED 2026-10-07 (mission `bullwhip-impl`): code, tests, hub pre-flights F0/F1, staged cluster kit. **No GPU/cluster run has happened.**
**Pre-flight F0 FAILED for the frozen `w8_g26` and the grid was revised per §3.1 to `w8_g26b`** (numbers and procedure in the Changes log, 2026-10-07 unit B3; F0 on the revised grid passed on fresh seeds). The spec below is the unchanged pre-registration;
every deviation is logged at the bottom. Per repo protocol (root `README.md` §research protocol), changes after approval are logged there.

**Layout note.** Wave docs are cited by FILENAME and follow the post-cleanup layout: `WAVE6_SPEC.md`, `WAVE7.md`, `WAVE7_SPEC.md`,
`W7-READY.md` in `docs/waves/`; `WAVE7-AUDIT-1.md`/`-2.md` in `docs/audits/`; `VERIFY-W7-EVAL.md` in `docs/verify/`; `STATE.md`,
`ARTIFACTS.md` in `docs/`. Code/test/cluster paths (`agent_bullwhip/*`, `tests/*`, `scripts/*`, `hpc-umbrella/*`) are stable and
cited exactly, file:line at repo HEAD `9f04e59` (2026-10-07) — lines are hints, re-read before editing.

**Lane:** wave 8 = decision-native extension of wave 7. Wave-6/7 files and verdicts are untouched; wave-7 defaults (grid,
prompt, checkpoints) stay reproducible behind a `--grid w7_u21` default (§12).

---

## 0. Thesis and relation to the paper

The user's headline direction: *a decision-type model purely built for this scenario, proving specialist models work in this
field.* This wave tests the strongest honest version of that sentence we can afford.

* **The paper** (arXiv 2605.17036v3, §5–§6): GRPO post-training of an **autoregressive 4B LLM** (Qwen-3 4B) on the four-tier beer
  game, system-level cost reward, agent-level return-to-go attribution, demand curricula; mean cost 1,585 → 952 (−40 %), CV
  26 % → 13 %, max 2,847 → 1,353 over 30 runs on the MIT step demand. Decisions are generated text (Appendix A.2 JSON line).
* **This wave is an EXTENSION, not a replication.** The policy is a **decision-native readout**: no generation, one prefill
  pass, the answer is a probability distribution over a *caller-defined choice set* (the Cloudflare **Clef** interface — typed
  answers + probabilities over choices the caller defines; Qwen backbone post-trained; prefill-only scoring; RL service =
  rollouts scored in a sandbox + trainer; blog 2026-10-01, `context/clef-cloudflare-blog.md`). Our model is **0.6B** (ladder to
  1.7B/4B), our training data is a **synthetic simulator**, our compute is free institutional L4s.
* **Hard bound on every claim in this wave:** we do NOT claim "matches Clef". Clef is 9B/27B (`Qwen3.5-9B` / `Qwen3.8-27B` per the blog),
  rank-256 LoRA + a routing head, label-smoothed CE + Brier + RLCD, trained on internal synthetic data permuting schemas
  *and* "many years of labelled decisions". We borrow the interface and the recipe *shape*; none of the scale, data, or head.

### 0.1 What counts as SUPPORTING vs CONTRADICTING the paper (pre-registered)

The comparison object is "trained vs *its own untrained* version, same interface, identical paired paths" — the paper's own
design (untrained 4B vs post-trained 4B). Absolute costs are never compared with the paper's 1,585/952: our simulator constants
(holding 1 / backlog 2, lead 2; `engine.SimConfig`) differ from the paper's prompt (0.50 / 1.00, order lead 1, shipping 2;
Appendix A.2), and the cost horizon/aggregation is not published.

* **SUPPORTING (mechanism-level, bounded):** outcome-scored training of the decision readout (R1 or R2) passes **G1** (§6.3) on
  ≥ 2 of 3 seeds at ≥ 1 rung **and** reproduces the paper's *three-metric direction* on the paper-comparable arm
  (**step-sampled n=30**: same fixed demand, repeated stochastic runs = the paper's "30 identical runs"): mean ↓ ≥ 20 %,
  CV ↓ ≥ 25 %, max ↓ (the half-effect-size bar `WAVE6_SPEC.md` §4 pre-registered for the replication), with the collapse and
  permutation gates read (§6.2). Statement allowed: "post-training on realized cost improves a decision-native specialist
  in this simulator, in the paper's direction".
* **CONTRADICTING (bounded):** pre-flight gates F0–F3 clean **and** R1 + R2 fail G1 at **all three rungs** (0.6B, 1.7B, 4B).
  Statement allowed: "outcome-scored RL does not install state-conditioned control in a closed-choice readout at ≤ 4B, with this
  recipe/budget" — which *weighs against* the generality of "post-training fixes reliability" for decision-native readouts, but
  **cannot contradict the paper's autoregressive-4B result** (different policy class, different algorithm, different budget).
* **INCONCLUSIVE (default):** anything else — a pre-flight gate failed, a collapse/permutation gate read as artifact, only
  0.6B fails, or seeds disagree (C4-style). Reported as such, with the failing gate named. `WAVE6_SPEC.md` §4's rule carries over:
  "a null at 0.6B with a positive at 4B is itself the scaling result".

---

## 1. What wave 7 actually did (the baseline this wave must improve on) — verified, with corrections

Everything in this table was re-derived from artifacts on 2026-10-07 (HEAD `9f04e59`). Rows 5–6 correct the brief's summary.

| # | Claim | Verdict | Evidence / verify command |
|---|---|---|---|
| 1 | Candidate grid 0..40 step 2 (21 letters A..U); edge-argmin share 0.71 | **VERIFIED** | `results/wave7_labels/labels.manifest.json`: `edge_argmin_rate 0.7118` (round-1, 95,200 states, seed 6000, M=16), `edge_argmin_orders [0,40]`, `mean_target_entropy_nats 2.213`. Round-1 `labels.jsonl` itself is **not in this checkout** (only the manifest is tracked) — regenerate with `python -m agent_bullwhip.decision_labeler --seed 6000 --episodes 700 --out …` if the rows are needed. |
| 2 | Simulator-optimal action at the grid cap in 35–52 % of labeled states | **VERIFIED for the round-2 (policy-visited) states** | `python3 -c` over `results/wave7_labels/policy_labels_seed{0,1,2}.jsonl`: `argmin_idx==20` (order 40) = **0.518 / 0.433 / 0.352**; `argmin_idx==0` (order 0) = **0.275 / 0.330 / 0.417**; edge total 0.793 / 0.763 / 0.769 (manifest `edge_argmin_rate` 0.7928 for seed 0). So the *low* edge (order 0) is as heavy as the high one — see §2.2 for why that matters. |
| 3 | Trained arms collapsed to a constant letter (RLCD letter-lock) | **VERIFIED** | `WAVE7.md` §4; `results/eval/w7-rlcd-seed{0,1,2}-argmax-noisy.jsonl`: every role-game emits one order value (modal share 1.00; orders 34/30/26); permutation-invariance 0.000 on all 7 arms incl. the untrained control (`results/eval/w7-*-perm.json`). |
| 4 | Untrained readout is the strong baseline | **VERIFIED** | Untrained: step-argmax (n=1) 10,464; noisy-argmax n=30 median 11,569; step-sampled n=30 median 22,207 (`WAVE7.md` §2–§3; recomputed in `VERIFY-W7-EVAL.md` Item 1). Floors (30 passes, `results/wave7_floors.json`): step wf_tuned 3,206; noisy wf_tuned 5,346, untuned 5,423, oracle 4,594. The untrained pick is letter **E = order 8** in 474/512 states (`WAVE7.md` §4) — ≈ the demand mean; luck of the letter, not competence. |
| 5 | RLCD config: LoRA r=8/α=16 attention-only (~5M trainable), K sampled actions, LOO baseline, KL anchor to frozen `pi_sft`, 2–3 rounds on a 20k-state subset, frozen-anchor/resume discipline | **PARTLY WRONG as a description of what ran** | LoRA r=8, α=16, dropout 0.05, `q/k/v/o_proj` ✓ (`decision_train.py:498-501`) — but **2,293,760 trainable** params, not ~5M (computed from the snapshot `config.json`: 28 layers, hidden 1024, 16 q-heads, 8 kv-heads, head_dim 128; adapter file 9,204,512 B = 2.30 M × 4 B). The **sampled-K / LOO** estimator exists (`--rlcd-sample-k K`, `decision_train.py:368-370,731-742`) but is **not what ran**: the train sbatch passes no `--rlcd-sample-k`, so the default **exact all-21 expected-cost gradient** ran (`decision_train.py:743-745`, `hpc-umbrella/gpu-wave7-train.sbatch` stage 2). **One** round (round-2 relabel), `--steps 40 --batch 32` ⇒ 1,280 state-visits on 5,168 train states (`results/wave7_rlcd_seed0/train_log.jsonl` `start` event: `n_train 5168, n_batches 162`) = 0.25 epoch, not "2–3 rounds on 20k". `beta_kl 0.02` anchored to frozen `pi_sft` ✓ (`prepare_pi_sft`, `decision_train.py:187-221`: fresh start computes π_sft once and persists `pi_sft.json` with `labels_sha`; resume loads it and refuses if missing/mismatched — WAVE7-AUDIT-2 P1-4). Observed RLCD `loss_kl` ≈ 1.41 nats at step 39 against β = 0.02: the anchor was nominal. |
| 6 | Wave-7 "training null" | **CONFOUNDED BY UNDER-TRAINING** (new) | SFT: `--steps 120 --batch 32` of 2,827 batches/epoch (`results/wave7_sft_seed0/train_log.jsonl` `start`: `n_train 90440, n_batches 2827`) = 3,840 of 90,440 rows = **4.2 % of one epoch**, lr 1e-5; final `train_top1` 0.22–0.25 (seed 0 steps 118–119), `mean_target_entropy` 2.1–2.3. A null from a 4 %-epoch LoRA at lr 1e-5 is not evidence that the interface cannot be trained. |
| 7 | "Permuted-option augmentation was pre-registered but not implemented" (`WAVE7.md` §7 option 1) | **CONTRADICTED by the code and the run state** | `decision_train.py:372-373` `--permute-options` default **1**; `permute_grid_and_target` `:79-86`; applied in `sft_step` `:637-647`; `results/wave7_sft_seed0/state.json` has `"permute_options": 1`. SFT therefore trained **only** on shuffled grids (p = 1.0) and was evaluated on the canonical ascending grid; **`rlcd_step` renders only the canonical grid** (`:724`, no augmentation). The permutation gate still read 0.000. Wave 8 must not re-propose "implement augmentation" as new. |
| 8 | Wave-7 prompt/readout | as built | `decision_readout.py:43-44` (`ORDER_GRID`, `LABELS`), `:58-68` instruction ("21 possible orders labelled A..U"), `:129-161` body + chat template (`enable_thinking=False`), `:170-201` label-id resolution on the rendered continuation, `:297-342` `readout_batch` (left-pad, `logits_to_keep=1`, softmax over label logits, mass). `decision_eval.py:68-131` writes the wave-1–6 JSONL with `"traces": {r: []}` (`:125`) — the empty slot wave 8 fills (§8). |
| 9 | Community recipe provenance | **note** | `WAVE7_SPEC.md` §0 cites the AstroHan RLCD recipe ("noise on candidate logits + LOO baseline", `Qwen/Qwen3-0.6B` default). The local reference `/home/pino/bw6-eval/reference/astro_decision.py` is **inference-only** (`DecisionModel.distributions/predict/evaluate`); its adapter config targets **`Qwen/Qwen3.5-4B`**, r=8/α=16, attention + linear-attn projections. The trainer-side details ("noise", "LOO") are therefore **unverified from local sources**; wave 7 and this spec implement our own audited estimator (§5). That reference also documents the constraint that fixes our grid size: labels past `Z` sit "about nine logits below the best of A–Z" on an untrained model, so choice 27+ is unreachable (§3.2). |

---

## 2. Why the wave-7 grid failed — and the evidence for the new one

### 2.1 Two distinct defects, not one

1. **Truncation.** The wave-7 choice set 0..40/2 cannot contain the simulator optimum in a large fraction of states.
2. **Teacher bias.** Even a 0..160 grid has argmin at the *far edge* in 6–14 % of states (§2.2), and week-0 factory states (all-zero)
   have optimum 160. Cause: the label J(s,a) is "deviate once, then everyone plays the continuation floor (θ=3.5, λ=0.2)".
   λ = 0.2 is a slow forecaster (`decision_labeler.py:29-33,72-74`); a big catch-up order beats the laggy floor — i.e. the label
   rewards "order a lot once" because the *teacher* is slow, not because the optimum is huge. The wave-7 audit called this "one
   policy-improvement step" (`decision_labeler.py` header, fix 6); it improves on the floor only if the floor's own cost is
   improvable that way.

Defect 2 is not fixed by any choice-set change — it is why **pre-flight gate F1 (teacher check) exists** and why R1/R2 results
must be read against it (§6.1).

### 2.2 Hub probe (2026-10-07; read-only; Appendix A reproduces it)

Method: wave-7's own labeler functions (`decision_labeler._replay_to_week`, `make_continuation`, `score_candidates`) with
the *candidate set* widened to 0..160 step 2 (81 candidates), M=8 CRN futures (wave 7: 16), seed 6000, curricula
poisson/truncnorm, 240 random (episode, week<34, role) states per behaviour regime:

* **mixed**: wave-7 behaviour (75 % floor with θ∈[1.5,5], λ∈[0.2,0.8] / 25 % uniform-random order from 0..40), i.e. the round-1 state generator;
* **nominal**: floor θ=3.5, λ=0.2 only (the states a *competent* policy visits).

Sanity: the mean target entropy of this probe on the wave-7 grid is **2.19 nats (mixed)** vs the real labeler's manifest **2.213** — the probe reproduces the labeler.

| quantity | mixed | nominal |
|---|---:|---:|
| wide-grid (0..160) argmin: median / p75 / p90 / p99 | 36 / 80 / 138 / 160 | 16 / 28 / 50 / 160 |
| share argmin = 0 | 0.308 | 0.267 |
| share argmin ≥ 40 (outside wave-7 grid) | 0.475 | 0.150 |
| share argmin at the 160 wide edge | 0.062 | 0.017 |
| median # wide candidates within 1 % of min J | 9 | 6 |
| **wave-7 grid (21): mean J-regret vs wide grid** (relative to mean min-J) | 141.1 (**4.25 %**) | 67.0 (**1.28 %**) |
| wave-7 grid: mean target mass on the top letter (U = order 40) | **0.147** | 0.049 |
| **G26 (§3.1): mean J-regret** (relative) | 4.7 (**0.24 %**) | 0.9 (**0.09 %**) |
| G26: mean target mass on the top letter (Z = order 160) / on letter A (order 0) | 0.051 / 0.042 | 0.013 / 0.047 |
| G26: share target-argmax at Z | 0.071 | 0.021 |

(Preliminary; n = 240 per column. Regret is against the 81-point wide grid, so it is a lower bound on truncation loss beyond 160.)

**Three consequences that shape the design:**

1. **The edge-argmin share is the wrong gate.** `J` is flat near the optimum (median 6–9 wide candidates within 1 %), so "argmin sits
   at the cap" often costs ~0. The decision-relevant quantity is **regret of the choice set vs a much wider reference**, plus the
   **target mass** on the top letter. wave-7's `edge_argmin_rate` (0.71) mixed both edges and conflated them.
2. **Order 0 is a legitimate, heavy edge, not a truncation.** 27–31 % of states have optimum 0 (excess inventory/pipeline); a
   choice set must keep 0 and the "0-edge" must not be counted as truncation. Only the *top* edge is.
3. **The optimum scale is state-dependent** (median 16 nominal vs 36 mixed; p90 50 vs 138). A fixed absolute grid can only trade
   resolution for coverage — hence a **non-uniform** grid: step-2 resolution where the demand mean lives (0..30), geometric tail above.
   A state-relative grid (wave-5 `choice_mult`: `max(0, round(κ·F̂ − IP))`) is the structural fix and is held as R3a (§4.3).

---

## 3. Interface contract (Clef-style, made explicit)

| element | wave-8 definition |
|---|---|
| **Typed answer** | `order ∈ GRID` (non-negative integers), one decision per (role, week); no free text anywhere |
| **Choice set** | `GRID = w8_g26`, §3.1, **caller-defined and fixed by this spec**; carried in every output record (`readout.grid`) so a different caller grid is a different `grid_id` |
| **Readout** | one prefill pass; last-position logits of the 26 single-token labels `A..Z` (A=ordered ascending by value ⇒ **monotone letter→value**); softmax over the 26 label logits |
| **Decision composition** | argmax (primary; deterministic); sampled from π (secondary; seeded 1000+i, the wave-7 T=1 counterpart); expectation Σπ·value (diagnostic only) |
| **Probabilities** | the **full 26-vector** is logged per decision (§7.2), plus `mass` = full-vocabulary probability on the 26 label tokens |
| **Confidence** | `confidence = max_a π(a)` (same key name the Jev traces already use, `jev_agent.py:326`); also `entropy` over the 26 (nats) |
| **Validity assertions** | labels are single tokens on the *rendered continuation* (`resolve_label_ids`); `mass ≥ 0.99` **fatal for the untrained readout, recorded for trained adapters** (WAVE7_SPEC amendment A1 — SFT erodes it: 0.997 → 0.963–0.973); `parse_fail == 0` asserted per run (there is nothing to parse) |
| **What the model sees** | the leak-free engine ctx rendered with the wave-1–5 field text (`decision_readout.render_state`, `:98-126`) + a system instruction + the question + the options block, via the pinned chat template with `enable_thinking=False`; the instruction's option count is templated from `len(GRID)` (today hard-coded "21 possible orders labelled A..U", `:65`) |

### 3.1 The choice set `w8_g26` (frozen by this spec)

```
GRID = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30,   # A..P  — identical to wave-7 letters A..P (values 0..30)
        34, 38, 44, 52, 62, 76, 94, 116, 140, 160]                    # Q..Z  — geometric tail
LABELS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")          # 26 single tokens, ids 32..57 in the pinned tokenizer (A..U = 32..52, tests/test_decision_readout.py:150)
```

Why these values (each a constraint from §1–§2, none invented):

* **Must contain plausible optima.** Median nominal optimum 16, p75 28 → step-2 resolution through 30 (A..P). The wave-7 letters A..P keep
  their values, so the untrained letter-E = 8 prior (the strong baseline, §1 row 4) stays *the same decision* and C1-style comparisons
  remain interpretable.
* **Must not put target mass on the edge.** Tail Q..Z covers up to the wave-7 mixed p90 (138) and the 160 reference. Measured (§2.2): top-letter
  target mass 0.051 mixed / 0.013 nominal vs wave-7's 0.147 / 0.049; J-regret vs the 81-point reference 0.24 % / 0.09 % vs 4.25 % / 1.28 %.
* **≤ ~30–40 candidates for a cheap readout, and ≤ 26 hard cap.** A..Z are the only uppercase single tokens; the reference code documents
  that two-letter labels past Z sit ~9 logits below on an untrained model (`astro_decision.py` comment) so positions 27+ are unreachable without
  heavy training. 26 also keeps the labeler cost at 26/21 of wave 7.
* **Monotone letter→value.** Strictly increasing; asserted at import and in the golden prompt test. Duplicate values forbidden.
* **Evaluation reads the whole 0..160 range** but orders are *grid values only*; no order > 160 can occur (rogue-order columns read 0 by construction).

Alternatives evaluated in the same probe and rejected: uniform 0..80/2 (41 candidates, > 26 labels) regret 1.5 % mixed / 0.31 % nominal;
three other 26-point geometric grids (G26a/c: same regret within ±0.1 pp — the tail shape is not load-bearing; G26b, listed above, was lowest nominal regret).
Frozen now so the pre-flight gate F0 (§6.1) cannot be tuned after seeing R-arm results; if F0 fails at ≥ 1,000 states the grid is
revised **before any training** and the revision logged here.

### 3.2 Labels in the state generator are NOT widened

`BehaviorPolicy.act` draws random orders from `ORDER_GRID` (`decision_labeler.py:170`). If the grid constant is widened naively the
`p_random=0.25` behaviour would draw orders up to 160 and change the *state distribution*, not just the candidate set. **The behaviour
sampler must use a frozen `BEHAVIOR_RANDOM_GRID = list(range(0,41,2))`** so wave-8 states are comparable to wave-7's, and only the *scored candidates* change.

---

## 4. Arms (priority-ordered, minimal-scope-first)

All arms: base `Qwen/Qwen3-0.6B` @ `c1899de2…`, LoRA r=8/α=16/dropout 0.05 on `q/k/v/o_proj` (2,293,760 trainable), bf16, chat template with
`enable_thinking=False`, grid `w8_g26`, training seeds {0,1,2}, eval seeds 1000+i. **Control C0** = the untrained readout on `w8_g26` (no adapter) —
required in every gate.

### 4.1 R1 — outcome-scored RL from the instruct checkpoint, no labels (cheapest informative config; runs first)

* **What:** RLCD recipe (exact expected-cost gradient over all 26 candidates — the audited wave-7 estimator) starting from a **fresh LoRA on the
  untrained checkpoint**; no SFT stage; J(s,a) from the simulator (`decision_labeler.score_candidates`, M=16 CRN futures) only as the *reward oracle*, never as a
  supervised target.
* **Loop (per seed; 3 rounds):** collect states → relabel J over 26 candidates → train 170 steps (one epoch of ~5,440 states at batch 32) → next round starts
  from the previous adapter. Round-r states = 40 curriculum episodes of the **current** policy: 15 argmax rollouts + 15 *sampled* rollouts (T=1 from π) + 10
  wave-7 behaviour-mix episodes (so a near-constant π0 still visits informative states and coverage cannot collapse with the policy).
* **Cost:** ≈ 0.7 GPU-h/run (170×3 steps × ~4.7 s) + 3 collects (~1.5 min) + 3 relabels (~7 min each on 8 CPUs — measured basis §9) ≈ 1.1 h wall.
  Two lr values × 3 seeds = 6 runs ≈ 7 GPU-h.
* **What it buys:** the Clef-style claim in its purest form — *a typed readout trained only by realized-cost feedback*; no label-teacher
  semantics beyond J. If it works, R2 is unnecessary.
* **Skip when:** pre-flight F0–F3 not all passed (nothing is spent before they pass).

### 4.2 R2 — optional warm start: soft-target SFT on the simulator's cost distribution

* **What:** SFT (`decision_train --phase sft`) on labels whose targets are the **softmax over the whole 26-candidate cost vector**,
  `p*(a) ∝ exp(−(E[J_a] − min)/τ_s)`, `τ_s = 0.5·std_a(E[J])` (wave-7 rule, `decision_labeler.py:277-299`, `--tau-scale 0.5`) — **not the argmin**. This is
  the Clef-ish probabilistic target: the model is trained to represent *how bad each alternative is*, which is what makes adjacent choices partial-credit
  (Clef's RLCD "partial credit to adjacent ordinal choices"; here it comes from J for free).
* **Differences from wave-7 SFT (all forced by §1 rows 6–7):** new grid; **one full epoch** (2,827 steps) instead of 120; option-permutation mix `p_perm = 0.5`
  (half canonical, half shuffled — wave 7 used p = 1.0, so the canonical task was never seen in training and evaluated only there); lr chosen from {1e-5, 1e-4} on
  label-side validation NLL (seed 0, first 500 steps) — never on eval paths; checkpoints every 250 steps with the permutation + collapse gates run on each (§6.2).
  Optional `--brier-weight 1.0` (Clef's calibration loss: Brier on top of CE) only if the calibration read (§6.4) shows over-confidence — default 0.
* **Then:** the same R1 RL rounds starting from the SFT adapter, anchored to the **frozen π_sft** exactly as in wave 7 (`prepare_pi_sft`) — reported as
  **R2+RL**; the SFT-only adapter is reported as **R2-SFT**.
* **Cost:** labels 95,200 states × 26 candidates ≈ 4 min on 96 cores (wave-7: 186.7 s at 21 candidates, `labels.manifest.json` `wall_sec`, scaled 26/21) +
  SFT 3 seeds × 2,827 steps ≈ 3 × 3.7 GPU-h + lr sweep 1.3 GPU-h + RL rounds 3 × ~0.7 ≈ 14.5 GPU-h.
* **What it buys:** a better-conditioned start when R1's from-scratch signal is too weak to install value-reading; also the only arm that exercises the
  Clef "post-train on labelled decisions" step.
* **Skip when:** R1 passes G1 on ≥ 2/3 seeds (R2 adds cost without changing the verdict); or F1 (teacher check) failed (labels would inherit the teacher defect).

### 4.3 R3 — choice-set variants (only if the collapse gate trips *with an edge signature*)

Trigger (computed, §6.2): collapse gate tripped **and** the trained argmax mass piles on a grid edge — letter Z in > 5 % of the fixed state sample S, or letter A
(order 0) in > 2× the oracle's share on S. Pile-up at a *middle* letter is letter-lock, not an edge problem — R3 does not apply (the fix is objective/lr, see §11 risks).

* **R3a (preferred for top-edge pile): state-relative choice set** (wave-5 `choice_mult`, `jev_agent.py:19-23`): values `max(0, round(κ_j·F̂_t − IP_t))`, κ on a 26-point
  ladder, F̂ the ES forecast (θ,λ) = (3.5, 0.2), IP = on_hand + outstanding − backlog. Letter→value stays monotone **within a state**; ties at 0 are allowed (documented).
  Trained the R1 way.
* **R3b: re-centred grid** (more resolution at 4..20) if J-regret on S shows the loss concentrated below 20.
* **R3c: widened tail** (…, 190, 240) if top-letter target mass on policy-visited states > 6 %.
* **Cost:** each variant = one R1 pass (~7 GPU-h for 6 runs, or ~3.5 for one lr). **Skip when:** collapse gate not tripped.

### 4.4 L-arms — size ladder (§9)

Same R1 recipe (best lr by validation) at `Qwen/Qwen3-1.7B` and `Qwen/Qwen3-4B`; run regardless of the 0.6B outcome once pre-flight passed and the 0.6B run completed
without an infrastructure fault (a null at 0.6B + a positive at 4B is the scaling result).

---

## 5. Training recipe — wave 7 vs wave 8

Wave-7 values are exact flags/lines (`hpc-umbrella/gpu-wave7-train.sbatch`, `decision_train.py`). The right column is the full R1 command set.

| item | wave 7 RLCD (what ran) | wave 8 R1 | why it changes |
|---|---|---|---|
| base / revision | Qwen3-0.6B @ `c1899de2…` | same | comparability |
| adapter | r8/α16/drop 0.05, q/k/v/o | same (fresh init) | comparability; 2,293,760 trainable |
| start | SFT adapter (`--base-adapter <sft ckpt>`, required, `decision_train.py:385-386`) | **fresh LoRA on the untrained model** → new `--init fresh` (lora_B = 0 ⇒ step-0 policy = untrained readout) | R1 has no SFT stage |
| objective | exact all-21 PG: `loss = Σ π(a)·(J−mean J)/max(std J,1e-3) + β·KL(π‖π_anchor)` (`:743-753`, per-state z-scored J) | **same, 26 candidates** | comparability; zero-variance estimator (WAVE7-AUDIT-1 §2 fix 4) |
| KL anchor | frozen π_sft computed once from the SFT adapter, persisted `pi_sft.json`, resume loads it (`:187-221`) | frozen **π0 = the untrained readout**, computed with the adapter disabled (`model.disable_adapter()`) → new `--anchor base`. π0 is a pure function of (frozen base weights, state), so each round's anchor table (persisted as that round's `pi_sft.json` with its `labels_sha`) can never follow the trained policy | AUDIT-2 P1-4: the anchor must never follow the policy |
| β_KL | 0.02 | 0.02 | comparability (observed KL 1.41 nats ⇒ weak anchor; not retuned here) |
| lr | 1e-5 | **{1e-5 (wave-7), 5e-5}**; winner by validation closed-loop cost (below) | wave 7 is under-trained (§1 row 6); 1e-5 kept as the comparability twin |
| steps / batch | `--steps 40 --batch 32` on 5,168 states (0.25 epoch) | `--steps 170 --batch 32` per round × 3 rounds on ~5,440 fresh states each | an epoch per round; 3 rounds of on-policy states (the wave-7 spec's own "2–3 rounds") |
| states | one round of SFT-policy argmax rollouts, 40 episodes (`collect_policy_states`, `decision_eval.py:134-197`) | 40 episodes/round: 15 argmax + 15 sampled + 10 behaviour-mix | coverage when π ≈ constant |
| option permutation | **none in RLCD** (`rlcd_step` renders canonical only, `:724`) | `p_perm = 0.5` per sample, J permuted with the same permutation → new `--permute-p 0.5` | force value-reading; keep canonical performance |
| `--rlcd-sample-k` | 0 (exact) | 0 (exact) | same estimator; the sampled-K recipe is not re-run |
| precision / memory | bf16, `--grad-ckpt`, `--max-length 1024`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | same | 22 GiB L4 OOMs batch 32 without grad-ckpt (`WAVE7.md` §1 job 385598) |
| checkpoint/resume | versioned `checkpoint-<step>/`, `state.json` last, SIGTERM → exit 75 | same; **SIGTERM-resume of a multi-round job resumes at the round recorded in a new `round_state.json`** | the wave-7 sbatch skips finished rounds only via the relabel manifest |
| seeds | 0,1,2 | 0,1,2 | |

**lr selection rule (pre-registered):** after round 3, run the *curriculum-validation* closed loop — 30 curriculum episodes, argmax readout, seeds band **9500..9529**
(disjoint from eval 1000–1029, labeler 6000+, collect 7000+, floor tuning 4000–4119) — for both lr arms of every seed; the lr with the lower **pooled mean validation cost** is "the R1
arm" for the gates; the other is reported as sensitivity. Eval paths are never used to choose.

R2 recipe additions: §4.2. Both recipes inherit wave-7's engine-side guarantees: `engine.GameStepper` + golden fixture + `tests/test_no_leak.py`; eval demand never enters training
(labeler rejects seed band [1000,1030)).

---

## 6. Gates, decision rules, stop-loss

### 6.1 Pre-flight gates (CPU/hub; **all must pass before any GPU job**)

| gate | quantity | threshold | basis |
|---|---|---|---|
| **F0 grid** | on ≥ 1,000 states each of mixed and nominal regimes: mean relative J-regret of `w8_g26` vs the 0..200/2 reference; p90; mean target mass on letter Z; target-argmax share at Z | regret ≤ 0.5 % mean **and** ≤ 2 % p90, both regimes; Z-mass ≤ 0.06 mixed / ≤ 0.03 nominal; Z-argmax ≤ 0.08 / ≤ 0.03 | thresholds set with the §2.2 probe in view (passes: 0.24 % / 0.09 %, Z-mass 0.051 / 0.013); they sit between wave-7's measured defect (4.25 %/1.28 %, 0.147/0.049) and the new grid |
| **F1 teacher** | closed-loop **PI1** policy (each role plays `argmin_a E[J(s,a)]` on `w8_g26` each week; labeler M=16) vs the continuation floor (3.5, 0.2) on the same **30 curriculum paths, seeds band 9000..9029** (never eval paths) | PI1 mean cost ≤ floor mean cost (paired; wins reported). If PI1 > floor the teacher is defective ⇒ **stop**: no R-arm is run; spec revised (continuation policy / horizon) first | §2.1 defect 2: one-step improvement over a slow floor can *raise* closed-loop cost; if the teacher cannot beat its own continuation, no learner can be read as "capacity-limited". Cost ≈ 30 games × 136 decisions × ~0.55 s ≈ 37 core-min |
| **F2 interface** | all 26 labels single-token on the rendered continuation; ids contiguous; prompt SHA fixture regenerated for the new instruction text and recorded here; `w7_u21` default still reproduces the old SHA `b96fb847…` | asserts | `decision_readout.resolve_label_ids`, `PINNED_REVISION` |
| **F3 control stats** | on the fixed state sample **S_nom** (§6.2) compute and freeze, *before training*: untrained-readout entropy distribution, state-dependence `I0`, modal-letter share, oracle modal share `m*` | recorded in `results/wave8/gates.json` (committed) | §6.2 thresholds are functions of these |

### 6.2 Computed gates (thresholds derived from data, not invented)

**Fixed state sample S_nom:** 1,024 states = every 2nd state of 16 nominal-floor (θ=3.5, λ=0.2) curriculum episodes, seeds band **9100..9115**, weeks < 34, all 4 roles (16×34×4 = 2,176 → stride 2 → 1,088 → first 1,024). Same S for every arm and for the untrained control. A second sample S_pol (the arm's own visited states) is reported, not gated.

**Collapse gate.** Frozen from the untrained readout and the simulator oracle on S_nom:
* `H_lo` = 5th percentile, `H_hi` = 95th percentile of the per-state entropy (nats, over 26) of the **untrained** readout;
* `m_hi` = upper bound of the 95 % Wilson interval of `m*` = share of S_nom states whose **oracle argmin letter** is the most common oracle argmin letter (oracle labels: M=16 on S_nom, ≈ 10 core-min);
* `I_hi0` = 95th percentile over 1,000 bootstrap resamples of S_nom of the untrained readout's state-dependence `I = H(mean_s π_s) − mean_s H(π_s)` (mutual information between the state and the answer; ≈ 0 for a letter prior).

A trained readout is **collapsed** iff
`[median entropy < H_lo AND (modal-argmax-letter share > m_hi OR I ≤ I_hi0)]` **OR** `[median entropy > H_hi AND I ≤ I_hi0]`.
(Low entropy alone is *not* collapse — a confident state-dependent policy is the goal; diffuse-and-state-blind is the other failure.) The only fixed numbers are the
conventional 5th/95th percentile and 95 % level, named here. The gate runs (i) at every round end / 250-step SFT checkpoint on 256 states of S_nom (training stop-loss, §6.5) and (ii) on all 1,024 at final eval.
A collapsed arm's costs are reported but **cannot satisfy SUPPORTING** (§0.1).

**Permutation-invariance gate (kept from wave 7).** `scripts/probe_permutation_invariance.py` generalised to `--grid`: 512 round-policy states (grid-shuffle seed 20261006, 3 shuffled grids): the
argmax **value** unchanged in ≥ 0.90 of states. Wave 7 read 0.000 on every arm including the untrained control (`results/eval/w7-*-perm.json`). Consequence of a fail, defined now:
**the cost gates still decide the performance claim**, but the interface claim "the model reads the options" is withheld and the arm is labelled **"fixed-ordinal-scale specialist"** (letters are an output code, as in any fixed-schema classifier); it is then a
legitimate specialist model but *not* evidence of Clef-style caller-defined choice sets. A pass with a cost-gate fail is reported as "reads values, does not help".

### 6.3 Performance gates (evaluated on the final adapters; protocol §7)

Binding variant (wave-7 AUDIT-1 §5): **argmax / noisy n=30**. Step-argmax is a single deterministic game (n=1).

* **G1 (vs own untrained readout, paired, noisy-argmax n=30)** — wave-7 C1 verbatim: median cost reduction ≥ **15 %** in *both* variants (ratio of medians and median of per-path reductions) **and** ≥ **20/30** paired wins **and** one-sided sign-test p < **0.02**. A seed passes iff all three hold; the arm needs ≥ 2/3 seeds.
* **G2 (floor context, report + tiers)** on noisy-argmax n=30 medians vs `wf_tuned` noisy floor **5,346**: *strong* ≤ 4× (21,384), *headline* ≤ 2× (10,692), *floor-parity* ≤ 1.25× (6,683; not expected, listed so an unexpected result is unambiguous). The untrained readout is 11,569 = 2.16× — so the headline tier is within reach of letter luck; **G1, not G2, is the test of learning**. Also report vs the untuned (3.0, 0.5) formula (5,423) and the oracle floor (4,594), same paths.
* **G3 (vs wave 6, identical paths):** the wave-7 C3 construction — paired vs the wave-6 `a0_base` trained arm (step n=30: ≥ 20/30 wins, raw and rogue-clean subsets; noisy n=10 seeds 1000..1009; refs `results/local-eval/trained-{step,noisy}/qwen3-0.6b-trained-baseline.jsonl`). Reported with the wave-7 caveat (wave-6 is rogue-carried: median 325,575 step), so a G3 pass is "no rogue tail", never "cheaper policy". If WAVE6-FIX later yields a contract-fixed arm, add it as a second reference.
* **G4 (paper-direction triplet on the paper-comparable arm = step-sampled n=30, same demand, stochastic runs):** mean ↓ ≥ 20 %, CV ↓ ≥ 25 %, max ↓ vs the untrained *sampled* readout (untrained: median 22,207; CV 0.48).
* **G5 (reproducibility):** wave-7 C4 rule — every seed pair's mean gap within 2·√(SE_i²+SE_j²) on the binding variant; zero-variance arms (a constant policy) are flagged "mechanically disagreeing", not scored as agreement (wave-7 RLCD z up to 132 was exactly that).
* **G6:** collapse and permutation gates (§6.2) read for every arm.
* **G7 (parse):** `parse_fail == 0` per run (assertion), `min_mass` recorded and ≥ 0.9 for trained adapters (wave-7 observed 0.924–0.982; < 0.9 is flagged — the readout is leaving the label set).

### 6.4 Calibration read (report-only, extension to Clef's claim)

On the held-out label validation rows (5 %): reliability of `confidence = max π` against the indicator "chosen candidate's J is within 1 % of the row minimum" — 10 equal-width bins ECE, Brier, AUROC of confidence vs that indicator; plus mean target-vs-π NLL. No threshold: the point is to show whether the model's confidence carries information (Clef sells calibrated probabilities; a 0.6B readout may not have them). R2b (`--brier-weight 1.0`) is run only if ECE > 0.15 on R2-SFT.

### 6.5 Stop-loss (compute-saving; decided now)

* A run is **stopped** (and its slot reported as "stopped: <gate>") when the collapse gate trips at two consecutive checkpoint reads, or the permutation invariance is < 0.05 after round 2 **and** the collapse gate has tripped.
* R1 as a whole is stopped (no further lr/seed runs, R3 decided) if all three seeds of the better lr trip the collapse gate by the end of round 2.
* Total wave budget **≤ 70 GPU-h** (free L4s) at 0.6B + 1.7B + 4B incl. GPU/GGUF steps (§9); beyond that, stop and report.
* Non-finite loss/grad (F1 tripwire in the trainer): abort, `--resume` refused; rerun once from scratch, a second abort ends that arm.

### 6.6 Decision table

| outcome | reading | next |
|---|---|---|
| F0–F3 pass, R1 passes G1 (≥ 2/3 seeds), collapse clean | SUPPORTING candidate; check G4 + permutation | run ladder; build GGUF; publish "specialist decision model works at 0.6B" with the §10 bounds |
| R1 fails G1, collapse tripped with edge signature | grid/teacher story still live | R3 (§4.3) once |
| R1 fails G1, collapse tripped mid-letter | letter-lock = objective/lr, not grid | run R2 (warm start); if R2+RL also locks → INCONCLUSIVE-negative at 0.6B; go to the ladder |
| R1 fails, collapse clean, perm fail | value-reading not learned in budget | R2 with `p_perm 1.0` variant on seed 0 only; otherwise report |
| F1 fails | teacher defective | stop; revise labeler continuation (not a model result) |
| fails at all rungs, all pre-flight clean | CONTRADICTING (bounded, §0.1) | write the honest negative; STATE.md update |

---

## 7. Evaluation protocol and the JSONL schema extension

### 7.1 Protocol (existing analysis stack unchanged)

Per arm and seed, `python -m agent_bullwhip.decision_eval` (`decision_eval.py:200-274`) three arms exactly as wave 7, with `--grid w8_g26`:
**step-argmax n=1** (deterministic; labelled single cost), **step-sampled n=30** (seeds 1000+i; the paper-comparable CV/max read), **noisy-argmax n=30**
(seeds 1000+i; binding for G1/G3/G5; also reported on the first 10 paths for wave-6 pairing). This follows WAVE7-AUDIT-1 §5 — the brief's
"step n=30, noisy n=10→20" is **not** reproduced literally because step-argmax is one game; noisy runs at n=30 (wave 7) ⊇ n=10/20. Floors: reuse
`results/wave7_floors.json` (`python3 scripts/wave7_floors.py`, ~6 s) — floors do not depend on the policy. Gate readout: new `scripts/wave8_analysis.py`
(stdlib only, modelled on `scripts/wave7_analysis.py`), writes `results/wave8_report.md` + `.json`; C1/C3/C4 machinery is reused by import-free copy.

### 7.2 Schema extension — backwards compatible by construction

Every wave-1–7 JSONL row keeps all keys `run, config, model, tag, total_cost, orders, backlogs, failures, calls, tokens, prompt_hash_dups, traces, tool_traces`
(`runner.py:162-178`, `decision_eval.py:113-127`). Wave 8 only **fills the existing `traces` slot** (wave 7 wrote `{role: []}`) and adds one new top-level key. The Jev traces already use the
same pattern (`traces[role]` = list of per-decision dicts with `confidence`, `jev_agent.py:323-332`).

```jsonc
"readout": {                       // NEW top-level key (analyzers never read it)
  "schema": "w8-readout-1", "grid_id": "w8_g26",
  "grid": [0,2,...,160], "labels": ["A",...,"Z"],
  "composition": "argmax" | "sampled",
  "model": "Qwen/Qwen3-0.6B", "revision": "c1899de2…", "adapter": "<dir or null>",
  "prompt_sha": "<sha256 of the canonical rendered prompt>",
  "n_decisions": 144, "min_mass": 0.9xx
},
"traces": { "<role>": [            // 36 dicts per role, week order; wave 7 wrote []
  { "t": 7, "idx": 4, "order": 8,              // chosen letter index and its grid value
    "probs": [0.0123, …],                      // 26 floats, rounded to 6 dp, sums to 1 ± 1e-5 (softmax over the 26 label logits)
    "confidence": 0.61,                        // max(probs)
    "entropy": 1.37,                           // nats over the 26
    "mass": 0.9971,                            // full-vocab probability on the 26 label tokens
    "expect_order": 14.2,                      // Σ probs·grid (diagnostic)
    "ctx": {"on_hand": 0, "backlog": 3, "outstanding": 10, "incoming_last": 9, "incoming_now": null, "last_orders": [8,8,9]} }
] }
```

Compatibility argument (and the test that enforces it, `tests/test_w8_schema.py`): (1) the only analyzers on the path are `hpc-umbrella/eval-analyze.py` (reads `total_cost`, `orders`, `failures`; `load_arm` keys at `:34-46`),
`scripts/wave7_analysis.py`, `scripts/paired_analysis.py` — `grep -rn "traces" scripts/*.py hpc-umbrella/eval-analyze.py` → **no hits**; the only `traces` readers are `scripts/archive/*` Jev tools, which
guard on `"anchor" in traces[0]` and skip records without it (`analyze_jev_traces.py:40-42`) and `interleaved_runner.py:48` which reads agent objects, not JSONL. (2) The test loads one wave-7 record and one wave-8 record
through each analyzer's loader and asserts identical `total_cost/orders/failures`-derived statistics, plus that wave-7 files (empty traces) still load. (3) File size: 26 floats × 144 decisions × 30 runs ≈ 0.6–1 MB per eval JSONL (tracked, like wave 7's).
`summary.json` gains `grid_id, entropy_mean, confidence_mean, top_letter_share, zero_letter_share, min_mass` (additive keys).

---

## 8. Calibration/diagnostic outputs per arm

`results/eval/w8-<arm>-<mode>-<pattern>.{jsonl,summary.json}` (same directory and `-perm.json` convention as `results/eval/w7-*`), plus `results/wave8/gates.json` (F3 frozen stats and each arm's gate reads),
`results/wave8/<arm>-S_nom.json` (entropy, modal share, `I`, per-letter argmax histogram, calibration table).

---

## 9. Size ladder, compute, venue

### 9.1 Ladder (model IDs)

| rung | model | basis | notes |
|---|---|---|---|
| 1 | `Qwen/Qwen3-0.6B` @ `c1899de289a04d12100db370d81485cdf75e47ca` | wave 6/7 | all gates; R1, R2, R3 |
| 2 | `Qwen/Qwen3-1.7B` | `WAVE6_SPEC.md` §3.3 ("optional 1.2B-class") — same family, same tokenizer/labels/code path | revision pin to be fixed at staging (**unverified here**); R1 only; seeds 0,1,2 |
| 2b (optional) | `LFM2.5-1.2B` | `WAVE6_SPEC.md` D2 ("already in the desktop HF cache"); **exact HF id not recorded in the repo — confirm on the desktop** | hybrid conv/attention architecture: `decision_train._attach_adapter` target list (`q_proj,k_proj,v_proj,o_proj`, `:498-501`) will not match its layers; needs its own `target_modules` and label-token check. Skipped unless the CPU-economics story needs a 1.2B point |
| 3 | `Qwen/Qwen3-4B` | the paper's own reference model ("Qwen-3 4B", §5/§6, lines 232, 329 of the staged text); NB the community reference adapter is `Qwen/Qwen3.5-4B` — a different model | revision pinned at staging (**unverified**); R1 only; seeds 0,1; needs micro-batching (new `--micro-batch`, grad accumulation) because batch 32 × 420 tokens with grad-ckpt on a 22 GiB L4 is not demonstrated for 4B `[INFERENCE]` |

The ladder does not change the grid, prompt, gates or protocol; the F3 control stats are recomputed per rung (the untrained readout differs).

### 9.2 CPU/GGUF serving step ("the specialist runs for free" must be a real artifact)

For the best 0.6B arm (and rung 3 if it passes):

1. `merge_and_unload()` the adapter into the bf16 base, save HF dir; `python /home/pino/bw6-eval/llama.cpp/convert_hf_to_gguf.py <dir> --outtype f16`; `llama-quantize … Q8_0` and `Q4_K_M` (build present: `/home/pino/bw6-eval/llama.cpp/build/bin/{llama-server,llama-cli,llama-quantize}`; wave 6 precedent `models/Qwen3-0.6B-Q4_K_M.gguf` + LoRA GGUF).
2. Serve with `llama-server` and read the 26-vector via `/completion` (`n_predict:1`, `temperature:-1` ⇒ greedy, with token probabilities "via a simple softmax of the logits", `n_probs:64`; `/home/pino/bw6-eval/llama.cpp/tools/server/README.md:579-609`) — map `top_logprobs` entries whose token is `"A".."Z"` to the 26 probabilities; a label absent from the returned top-64 counts as 0, and the summed label mass is asserted within 0.01 of the torch `mass`. `llama-cpp-python` is **not installed** (checked in the jev venv). If the `n_probs` route proves unreliable, the implementer may use any other llama.cpp binding that returns last-position logits; the equivalence gate below is mandatory either way.
3. **Equivalence gate (pre-registered):** on 2,000 states (S_nom ∪ S_pol) the GGUF-f16 argmax equals the torch argmax in ≥ 99 %, top-1 probability within 0.02 (f16); Q8_0 ≥ 98 %; Q4_K_M reported (no gate).
4. Run the standard protocol (step-argmax n=1, noisy-argmax n=30) through the CPU server; costs must equal the torch run's within the argmax-flip tolerance. **Report measured** prefill latency per decision (prompt ≈ 420 tokens), decisions/s on the hub's 6 cores, and full-game wall time — *no number is quoted before it is measured*.
5. Artifacts: `results/wave8_gguf/` (manifest + sha256 in `docs/ARTIFACTS.md`; `.gguf`/adapters untracked per the repo rule), serving script `scripts/wave8_gguf_serve.py`.

### 9.3 Compute, venue, hub vs cluster split (measured bases cited; `[INFERENCE]` marked)

| phase | venue | work | est. |
|---|---|---|---|
| F0 / F1 / F3 / S_nom oracle | **hub CPU** (6 cores) | probe ≥ 1,000 states ×2 regimes ≈ 0.87 s/state/core at 81 candidates (measured) ⇒ ~2 h core-time for F0; F1 ≈ 37 core-min; oracle on S_nom ≈ 10 core-min | hub, hours; or `cpu-wave8-label.sbatch` on `tue.cpu3.q` |
| R2 round-1 labels (95,200 × 26) | `tue.cpu3.q`, 96 procs | wave-7: 186.7 s at 21 cand. ⇒ ≈ 4 min | CPU, free |
| R1 (6 runs) | `tue.gpu.q -C l4` array | 3 rounds × (collect ≈ 1.5 min + relabel on 8 CPUs ≈ 7 min + 170 steps × ~4.7 s ≈ 13 min) | ≈ 7 GPU-h total |
| R2 + RL (3 seeds + sweep) | L4 | §4.2 | ≈ 14.5 GPU-h |
| R3 (one pass if triggered) | L4 | | ≤ 7 GPU-h |
| Ladder 1.7B (3 seeds, R1) | L4 | 4.7 s × (1.7/0.6) ≈ 13 s/step `[INFERENCE]` × 510 steps ≈ 1.8 h/seed | ≈ 5.5 GPU-h |
| Ladder 4B (2 seeds, R1) | L4 (micro-batched) | 4.7 s × (4/0.6) ≈ 31 s/step `[INFERENCE]` × 510 ≈ 4.4 h/seed | ≈ 9 GPU-h |
| Ladder 8B (2 seeds, R1) — **A2 amendment** | Snellius `gpu_a100` burst lane (A2: amplifies the 1.7b→4b phase-change question; L4 24G cannot hold 8B LoRA train; submit cap boost tied to this rung only) | 8B bf16 + LoRA + micro-batch; A100 80G; measured smoke step TBD | ≈ 18 GPU-h × A100 (2 seeds) + eval ≈ 4 |
<!-- A2 pins: ModelId Qwen/Qwen3-8B, revision `b968826d9c46dd6066d109eabc6255188de91218` (snapshot pulled to Snellius 2026-10-21, 5 shards ~16 GB); venv torch 2.8.0+cu126 / transformers 4.57.6; kit overlay: `hpc-umbrella/config-snellius.sh` + `gpu-wave8-train-snl.sbatch` (no `-C l4`, `gpu_a100`, longer walltime); gates file `results/wave8/gates_8b.json` after a fresh F2/F3 preflight. -->
| Evals (all arms, 3 modes + perm + calibration) | L4 | wave-7 eval job: 21 runs + 7 probes in ~50 min | ≈ 3 GPU-h |
| GGUF + CPU serving | hub CPU | convert/quantize minutes; equivalence 2,000 states; full games | CPU, hours |
| **Total GPU** | | | **≈ 30–45 GPU-h** (≤ 70 cap), wall ≈ 1–2 days with the 8-L4 pool |

Basis: wave-7 per-step wall (RLCD 187 s / 40 steps = 4.7 s; SFT 561 s / 120 steps; `train_log.jsonl` `wall`), wave-7 labeler throughput, wave-7 eval job 398422 timings (`WAVE7.md` §1). Hub/cluster mechanics follow `hpc-umbrella/OPERATOR.md` §"Wave 7"
(`cpu-wave7-label.sbatch`, `gpu-wave7-train.sbatch`, `gpu-wave7-eval.sbatch` clones; `config.sh` pins; `SLURM_SUBMIT_DIR`; `HF_HUB_OFFLINE=1`; `EXPECTED_COMMIT`; results by `push_results.sh` + `verify_pull.py` EXPECT updated). Modal is **not** used (WAVE6_SPEC D3).

---

## 10. What this would — and would not — prove

**It would show (if SUPPORTING):** in this simulator, a small open model with a Clef-style typed-readout interface, trained only by realized cost, can beat its own untrained
version on paired paths, with zero parse failures, a logged probability for every choice, and a CPU-servable artifact; and the paper's three-metric direction (mean/CV/max) can be reproduced by a
decision-native specialist. With the ladder, whether that improves with size.

**It would NOT show:**
* that a specialist beats the 1970s tuned order-up-to floor — that formula stays the honest benchmark (the repo's surviving claim; floor-parity is listed as a tier, not an expectation);
* that the paper's autoregressive GRPO works or fails at any size — different policy class;
* anything about Clef's quality, scale, or "matches Clef" — different model, data, head and training;
* generalisation beyond this simulator/demand family; the eval demand (MIT step + seeded noisy) is a held-out *protocol*, not a held-out *environment*; the curriculum and engine are the teacher's own;
* that the model "understands inventory" — a permutation-gate fail would mean the letters are a fixed ordinal output code (§6.2);
* a CPU speed advantage over the formula (the formula costs ~3 CPU-s; the 3–4-orders-of-magnitude-cheaper claim is the formula's, not the model's).

---

## 11. Risks, stop-loss, honest-negative clause

| risk | handling |
|---|---|
| **Teacher bias** (§2.1): J is "improvement over a slow floor" | F1 pre-flight; any positive result is stated as "outperforms its untrained self under floor-continuation labels", never "optimal" |
| **Letter-lock again** (wave-7 RLCD) | permutation mix p=0.5, sampled/behaviour-mix states, frozen anchor π0, collapse gate computed from data, stop-loss at two consecutive trips; the objective is unchanged (comparability) so lock caused by per-state z-scored J on flat states remains possible → recorded diagnostic: per-round share of states with `std J` clamped at 1e-3 |
| **Under-training again** | one epoch/round, 3 rounds, lr sweep; budget stated; a null is explicitly "at ≈ 510 steps/seed (≈ 16 k state-visits)", not "RL fails" |
| **Top-letter/low-letter confound** | grid fixed before training (F0); R3 only on computed edge signature |
| **Selection leakage** | lr and checkpoint chosen on validation band 9500+, never eval 1000+; labeler rejects the eval band (`decision_labeler.py` fix 7) and a test pins S_nom/validation/collect bands disjoint |
| **bf16 mass erosion** | recorded not asserted for adapters (A1); flagged < 0.9 |
| **GGUF readout mismatch** | mandatory equivalence gate; no CPU claim without it |
| **4B memory/time** | micro-batching change; if it fails on L4, rung 3 is reported "not run: <reason>", not substituted |

**Honest-negative clause.** Every row of the §6.6 table is a publishable result as written. If R1/R2 fail at every rung with F0–F3 clean, the wave publishes "outcome-scored RL did not install
state-conditioned control in a closed-choice readout at ≤ 4B at this budget" with the gate reads; thresholds are never moved after seeing data — any change is logged in the changes log with the reason
and the data that prompted it. A INCONCLUSIVE outcome is reported as inconclusive, not rounded to support or contradiction.

---

## 12. Implementation change list (for the later implementer; nothing done here)

Defaults keep wave-7 behaviour (`--grid w7_u21`), so `tests/*` and wave-7 reproducibility stay green.

1. **`decision_readout.py`:** grid registry `GRIDS = {"w7_u21": …, "w8_g26": …}` with strict-increase assertion; `ORDER_GRID`/`LABELS` become the registry default; `INSTRUCTION` and `render_options` template the count; `ReadoutModel(grid_id=…)`; `readout_batch` additionally returns `entropy`, `confidence`; `compose` uses the active grid; new `grid_id` argument threaded to `render_prompt`. Regenerate the prompt-hash fixture only for `w8_g26`.
2. **`decision_labeler.py`:** `--grid`; `BEHAVIOR_RANDOM_GRID = list(range(0,41,2))` for `BehaviorPolicy.act` (`:170`); edge metrics report `top_edge_rate`, `zero_rate`, **target mass on the top letter**, and **regret vs a `--reference-grid 0..200/2`** instead of the combined `edge_argmin_rate` (`:424,496`); `pi_sft_key` gains a `src` component (§4.1).
3. **`decision_train.py`:** `--grid`; `--init {adapter,fresh}` (fresh LoRA, lifts the `--base-adapter` requirement at `:385-386`); `--anchor {adapter,base}` (base ⇒ `compute_pi_sft` under `model.disable_adapter()`, persisted per `labels_sha` with the round-1 table reused); `--permute-p P` (replaces the 0/1 `--permute-options`; applies to `sft_step` *and* `rlcd_step`, permuting `j_vals` with the grid); `--brier-weight`; `--micro-batch` (rung 3); per-round `round_state.json`; log `std_j_clamped_frac`.
4. **`decision_eval.py`:** `--grid`; fill `traces[role]` and the `readout` key (§7.2); `summary.json` additions; `--collect-mode {argmax,sampled}` for R1 rounds.
5. **Scripts (new, stdlib/torch as appropriate):** `scripts/wave8_teacher_check.py` (F1), `scripts/wave8_grid_probe.py` (F0, Appendix A productionised), `scripts/wave8_gates.py` (S_nom, F3, collapse, calibration), `scripts/wave8_analysis.py`, `scripts/wave8_gguf_serve.py`; generalise `scripts/probe_permutation_invariance.py` (`--grid`).
6. **Cluster:** `cpu-wave8-label.sbatch`, `gpu-wave8-train.sbatch` (R1 rounds loop + R2), `gpu-wave8-eval.sbatch` (clones of the wave-7 files; `tests/test_sbatch_contract.py` and `config.sh --check` header pins; `verify_pull.py` EXPECT entries for `wave8_*` dirs).
7. **Tests:** `tests/test_w8_grid.py` (monotone, 26 single tokens, ids contiguous, behaviour grid frozen, wave-7 default unchanged), `tests/test_w8_schema.py` (§7.2), `tests/test_w8_gates.py` (collapse gate on synthetic letter-lock / diffuse / healthy readouts; computed thresholds), RLCD permuted-J unit test (J permuted with the same permutation as letters), `--init fresh` ⇒ step-0 logits equal the untrained readout.
8. **Docs after the runs:** `docs/waves/WAVE8.md` (results, repo voice), `docs/verify/VERIFY-W8.md`, STATE.md wave table row, CHANGES.md entry, README wave line.

### Verification (independent recomputation, separate code)

Recompute from raw artifacts, not from `wave8_analysis.py`: (1) per-arm medians/wins/sign-p for G1 from `results/eval/w8-*.jsonl`; (2) F0 regret and Z-mass from a fresh probe with a different seed; (3) the collapse-gate inputs (`H_lo, H_hi, m_hi, I_hi0`) from the S_nom readout JSON and an independently regenerated oracle; (4) the permutation rate from `w8-*-perm.json`; (5) per-decision `probs` sum to 1 ± 1e-5 and `confidence == max(probs)`, `order == grid[idx]`; (6) GGUF equivalence; (7) the wave-7 numbers quoted in §1 (recompute commands are in the table rows); (8) run-count integrity: `n_runs`, `failed == 0`, `parse_fail == 0`.

---

## Appendix A — read-only grid probe (reproduces §2.2; run from `/tmp`, never write into the repo)

```python
import sys; sys.dont_write_bytecode = True
sys.path.insert(0, "/home/pino/projects/agent-bullwhip-replication")
import json, random, multiprocessing as mp
from agent_bullwhip import decision_labeler as L
from agent_bullwhip.engine import ROLES
WIDE = list(range(0, 162, 2))
def task(t):                               # (seed, ep, week, role, curriculum, p_floor, p_random, nominal?)
    seed, ep, week, role, cur, pf, pr, nominal = t
    beh = L.BehaviorConfig(pf, pr, (3.5, 3.5), (0.2, 0.2)) if nominal else L.BehaviorConfig(pf, pr)
    step, il, params = L._replay_to_week(seed, ep, week, cur, 36, beh)
    ctx = step.observations()[role]
    cont = L.make_continuation(L.CONTINUATION_THETA, L.CONTINUATION_LAM, il, week)
    rng = L._seeded_rng(f"{seed}:ep:{ep}:w:{week}:{role}")
    fut = [L.draw_future_demand(rng, cur, params, 36 - week) for _ in range(8)]
    J = L.score_candidates(step, week, role, WIDE, cont, fut, L.CONTINUATION_THETA, L.CONTINUATION_LAM, il)
    return dict(role=role, week=week, J=J, ctx={k: ctx[k] for k in ("on_hand","backlog","outstanding","incoming_last")})
# states: rnd=random.Random(1|2); ep=rnd.randrange(700); week=rnd.randrange(34); role=rnd.choice(ROLES); cur=["poisson","truncnorm"][ep%2]
# 240 states per regime; mixed = (0.75, 0.25), nominal = (1.0, 0.0) with fixed (theta, lam); mp.get_context("fork").Pool(6)
# analysis: regret = min(J[g_idx]) - min(J);  target p ∝ exp(-(J-min)/ (0.5*std_a J))  over the grid's candidates;  idx = value//2
```
(Use the `fork` start method; the default `forkserver` in this Python does not inherit `sys.path`. 480 states ≈ 70 s on 6 cores.)

## Appendix B — items not verified here, with the command that verifies each

| item | command / action |
|---|---|
| Round-1 `labels.jsonl` edge numbers (only the manifest is tracked) | regenerate: `python -m agent_bullwhip.decision_labeler --seed 6000 --episodes 700 --m-futures 16 --processes <N> --out /tmp/w7labels` (the wave-7 sbatch knobs) and compare `edge_argmin_rate` to 0.7118 |
| Paper A.2 braces in the JSON example | open arXiv 2605.17036v3 PDF, Appendix A.2 (the staged text lacks `{}`) — matters for WAVE6-FIX, noted here for the contract wording |
| `Qwen/Qwen3-1.7B`, `Qwen/Qwen3-4B` revisions; `LFM2.5-1.2B` HF id | `huggingface-cli` on the staging login node / desktop cache; pin in `config.sh` |
| 4B LoRA batch-32 fit on a 22 GiB L4 | 10-step smoke with `--micro-batch` before the full run |
| llama-server `n_probs` returning all 26 label tokens | the §9.2 equivalence gate |
| AstroHan trainer details ("noise on candidate logits", LOO) | not in `astro_decision.py`; wave 7's own estimator is what this spec uses |

## Changes log

- 2026-10-07: draft written (spec-only; waiting for repo cleanup to finish and Pino's approval).
- 2026-10-07 (implementation, mission `bullwhip-impl`) — unit B1 (grid registry, readout interface, trace schema); nothing run on GPU/cluster:
  1. **Grid tail is a frozen constant, not a formula.** §3.1 gives no closed form for Q..Z; a fit search over simple geometric forms (planning, 2026-10-07) did not reproduce it
     (closest `[34,38,44,52,62,74,90,108,132,160]`). `decision_readout.W8_G26_TAIL` carries the spec's constants; `make_w8_g26()` composes + validates (strictly increasing, A..Z labels);
     the artifact `results/wave8/grid_w8_g26.json` is emitted by `scripts/wave8_grid.py`. `GRIDS`/`get_grid`/`GridSpec` registry; wave-7 default and `INSTRUCTION` bytes unchanged.
  2. **F2 interface, verified (external venv `/home/pino/absolver/.venv`, transformers 5.14.1, offline, pinned snapshot):** all 26 labels single-token on the rendered continuation, ids 32..57 contiguous;
     `apply_chat_template` prompt SHA == pure-string renderer SHA == `5062a9d71c7372a01dd2dbd626e8b3960574292ef770626195b9fe39b59e58fc` (canonical ctx, `w8_g26`). The pure-string renderer also
     reproduces the wave-7 fixture `b96fb847…` on the hub (test `test_string_renderer_reproduces_wave7_prompt_fixture`), so the hub can pin both SHAs without transformers.
  3. **`pi_sft_key`** lives in `decision_train.py`, not the labeler (§12.2 mis-locates it); no `src` component was needed (round label files are merged with unique episode ids instead, B2).
  4. **Labeler:** `--grid`, `--score-reference`; `BEHAVIOR_RANDOM_GRID` frozen (stream-identical to wave 7, test). `edge_argmin_rate` kept for compatibility; new `top_edge_rate`, `zero_rate`,
     `target_mass_top_letter`, and (with `--score-reference`) `regret_vs_reference_{mean,p90}` = min_grid J − min_ref J, relative to the MEAN reference min-J (so p90 is the p90 of per-state regret / that constant).
  5. **Eval:** `traces`/`readout` filled only with `--traces 1` or any non-default `--grid` (wave-7 JSONL stays byte-identical by default). `probs` rounded to 6 dp as specified; `confidence` is `max` of the
     ROUNDED probs so `confidence == max(probs)` holds exactly; the per-vector sum error from 6-dp rounding is typically ~1e-6 but can reach 1.3e-5 in the worst case (§Verification item 5 tolerance 1e-5 may then
     fail on a vector — a rounding artifact, not a model fault). New `--pattern curriculum` (validation band) and `--collect-mode/--collect-offset`.
- 2026-10-07 (implementation) — unit B2 (training flags + round orchestrator); the torch policy path is UNOBSERVED (no peft in any local venv; the torch tests are gated and run in the cluster venv during `stage.sh`):
  1. **Flags** (`decision_train.py`): `--grid`, `--init {adapter,fresh}`, `--anchor {adapter,base,external}` (+ `--anchor-table`), `--permute-p`, `--brier-weight`, `--micro-batch`, `--val-every`, `--tag`. Wave-7 defaults reproduce wave-7 bytes
     (`--permute-options 1` ⇒ p=1.0 with the same `_seeded_rng(f"{seed}:perm:…")` shuffle; tests pin it). **RLCD permutes only when `--permute-p` is passed explicitly** (wave-7 RLCD never permuted although the default `--permute-options 1` was in effect).
  2. **`--anchor external` is a spec addition.** R2+RL rounds 2–3 train on NEW states, but `--anchor adapter` would freeze the policy *as loaded* (= the previous round's trained adapter), which would let the anchor follow the policy — the AUDIT-2 P1-4 defect. §4.2 says "anchored to the frozen π_sft exactly as in wave 7", so
     `decision_rounds anchor-table` computes π_sft from the SFT adapter (via `ReadoutModel`, eval mode) on each round's labels and `decision_train --anchor external --anchor-table` loads it (labels sha checked, persisted as `pi_sft.json` with `"anchor": "external"`). R1 uses `--anchor base` (adapter disabled), as specified.
  3. **Orchestrator** `agent_bullwhip/decision_rounds.py`: per-round stages collect(argmax 15 eps) → collect(sampled 15 eps, episode offset 15) → concat → relabel policy states (`--states-jsonl`) → behaviour-mix labels (10 episodes, `decision_labeler --episodes 10`) → merge (episode ids namespaced +10,000·k so `pi_sft_key` stays unique) → train (170 steps, `--permute-p 0.5`, `--resume`) → collapse read (`scripts/wave8_gates.py collapse-check`, 256 states of S_nom);
     after the last round the curriculum-validation eval (`--pattern curriculum --seed-base 9500 --runs 30`). State in `<arm_dir>/round_state.json`; SIGTERM forwarded to the child, exit 75; two consecutive collapse trips write `STOPPED_collapse.json` and exit 0. Seed bands: collect 7100+10s+r, sampled 7200+10s+r, label 6100+10s+r, behaviour 6200+10s+r (disjoint from every other band; test).
     Train out dirs are flat per round: `results/wave8_<arm>_<lrtag>_seed<S>_round<r>` (one `verify_pull` EXPECT entry each); working files in `results/wave8_<arm>_<lrtag>_seed<S>/round<r>/`.
  4. **Validation / selection details the spec left open:** R2-SFT lr sweep runs with a new `--val-every 100` (the first epoch boundary is step 2826, beyond the 500-step sweep); SFT collapse/permutation reads are post-hoc over the saved `checkpoint-*` (every 250 steps), not an in-run abort (SFT is 3.7 GPU-h/seed and the signal is read afterwards).
  5. **`std_j_clamped_frac`** is logged per RLCD step (share of rows whose J std < 1e-3, unbiased std as in the estimator).
- 2026-10-07 (implementation) — unit B3 (gates, scripts, hub pre-flights): measured on the hub CPU (6 cores), results committed under `results/wave8/`:
  1. **F0 on the frozen `w8_g26` FAILED** (1,000 states per regime, M=8, reference 0..200/2, state seed 1, label seed 6000; `results/wave8/f0_w8_g26_prereg.json`, wall 465.8 s on 6 procs):
     mixed — regret mean 0.236 % / p90 0.569 % (ok), **top-letter target mass 0.0684 (> 0.06)**, **target-argmax at Z 0.098 (> 0.08)**; nominal — regret mean 0.233 % / p90 0.177 % (ok), top mass 0.0218 (ok), **target-argmax at Z 0.032 (> 0.03)**.
     (For scale, the wave-7 grid on the same states: regret 2.13 % / 3.67 % mean, top-letter argmax 0.378 / 0.170.) The §2.2 preliminary probe (n=240 per regime) had read 0.051 / 0.071 and 0.013 / 0.021 — sampling noise at n=240 was larger than the margin to the thresholds.
  2. **Revision (explicitly allowed by §3.1 "if F0 fails at >= 1,000 states the grid is revised before any training and the revision logged"): `w8_g26b`.** Same letters A..P (0..30 step 2, the untrained letter-E = 8 prior is unchanged); the tail Q..Z is a GENERATED
     geometric ladder 30 -> 200: `v_k = 2*round(30*(200/30)^(k/10)/2)` = 36, 44, 54, 64, 78, 94, 114, 136, 166, 200 (`decision_readout.make_w8_g26b`). Selection procedure: the saved reference-grid J rows of the failing run were re-read for ONE pre-declared family
     (the same construction with top in {170, 180, 190, 200}); only top = 200 cleared the thresholds on that selection data (top 190 missed the mixed top-argmax bar: 0.082), so the choice was made on data that is NOT used for the gate. The F0 that counts is a fresh run on new states
     (state seed 3, label seed 6050): **PASS** — mixed regret 0.109 % / p90 0.340 %, top mass 0.0404, top argmax 0.058; nominal 0.064 % / 0.194 %, 0.0154, 0.021 (`results/wave8/f0.json`, wall 603.9 s). No threshold was moved. Caveat recorded: the reference grid ends at 200, so for a grid whose cap IS 200 the regret column cannot see truncation beyond 200;
     that dimension is carried by the top-letter mass / argmax-share gates (a state wanting > 200 lands on Z and counts there). The pre-registered grid stays in the registry as `w8_g26` (artifacts and the failing record are kept); everything staged defaults to `w8_g26b` (`decision_readout.W8_STAGED_GRID`).
     F2 for `w8_g26b` (external venv, pinned snapshot): 26 single-token labels, ids 32..57 contiguous, prompt SHA (`apply_chat_template` == string renderer) `55bdc663eee13e101be89e1ddf112e48fcafc62d4c8bf4950b39a2b49dafc422`.
  3. **F1 teacher check PASS** on `w8_g26b` (30 curriculum paths, seeds 9000..9029, M=16): PI1 mean cost 4,590.7 vs continuation-floor mean 8,541.2, PI1 cheaper on 30/30 paths (`results/wave8/f1.json`, wall 631 s). (An earlier F1 started on the failed grid was killed unfinished and is not used.)
  4. **Implementation details the spec left open:** F0 reference J is scored once per state (101 candidates) and every grid is read from it by value; relative regret is regret / MEAN reference min-J (so p90 is the p90 of per-state regret over that constant); nearest-rank percentiles; Wilson z=1.96; bootstrap B=1000, q=0.95, seed 20261007;
     S_nom = 16 nominal-floor episodes (`seed 9100+i`, `ep 0`, curriculum alternating, `BehaviorConfig(1.0, 0.0, (3.5,3.5), (0.2,0.2))`), weeks 0..33, roles in engine order, stride 2, first 1,024 — the labeler task gained optional `theta_range`/`lam_range` so the oracle replays exactly those states (asserted equal ctx);
     collapse reads at round ends use every 4th S_nom state (256) against the thresholds frozen on all 1,024; permutation probe default cases = `S_nom` (the same 512-state subsample for every arm and the control) instead of arm-specific round states (comparability; wave 7 also used one file for all arms).
  4b. **S_nom and its oracle (hub CPU, M=16, `w8_g26b`)**: 1,024 states (sha256 `f651fc58e5a34c36…`, `results/wave8/S_nom.jsonl`); the oracle's most common argmin letter is **A (order 0)** with share m* = 0.1270 (Wilson upper bound m_hi = 0.1487) — `results/wave8/S_nom_oracle.json`. F3 (untrained-readout entropy quantiles, state-dependence I0 / I_hi0) is NOT computed here (needs the model): it is frozen by `gpu-wave8-preflight.sbatch`.
  5. **G-gates** are computed in `decision_gates.py` (stdlib) and reproduce wave 7's own C1 on the real wave-7 files (test: wave-7 RLCD seed 0 fails C1 identically in both implementations).
- 2026-10-07 (implementation) — unit B4 (cluster kit, ladder, GGUF): `cpu-wave8-label.sbatch`, `gpu-wave8-preflight.sbatch`, `gpu-wave8-train.sbatch` (array; `W8_ARM=r1|r2|r2sweep`, `W8_RUNG`), `gpu-wave8-eval.sbatch`; `config.sh` pins for the ladder
  (`Qwen/Qwen3-1.7B` @ `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`, `Qwen/Qwen3-4B` @ `1cfa9a7208912126459214e8b04321603b3df60c`, both read from `huggingface.co/api/models/<id>` `sha`, 2026-10-07, lastModified 2025-07-26; this resolves Appendix B's "revision to be pinned"), `stage.sh --with-ladder`,
  `verify_pull.py` EXPECT for the 18 R1 round dirs. 4B micro-batch fit on a 22 GiB L4 and the whole torch `_Policy` path (fresh init, base anchor, permuted/micro-batched RLCD step) are UNOBSERVED here — covered by gated torch tests that run during `stage.sh` on the cluster venv.
  GGUF/CPU step: `scripts/wave8_gguf_serve.py` (convert, llama-server 26-vector readout, equivalence gate f16 >= 0.99 / Q8_0 >= 0.98 / Q4_K_M reported, protocol eval with MEASURED latency); exercised against a fake llama-server only — the real conversion/serving is untested until a trained adapter exists. `llama-cpp-python` is not used (plain HTTP).
- 2026-10-07 (implementation) — unit B5 (R3 sets, handoff doc): **R3a is NOT implemented** (deviation, smallest faithful alternative chosen). §4.3 wants a state-relative choice set `max(0, round(κ_j·F̂_t − IP_t))`, but a per-state grid changes the prompt, the label ids and the J-permutation on every decision, i.e. it threads a *new axis* through `decision_readout` → `decision_labeler` → `decision_train` → `decision_eval` and would invalidate the F0/F2/F3 gates frozen on a fixed grid; that is a spec-sized change, not an R3 variant. The two fixed-grid fallbacks the same section defines are in the registry instead: `w8_r3b` (re-centred: `0,2 | 4..20 | 24,30,40,60,90,140,200`) and `w8_r3c` (widened tail: `…,136,190,240`). Both are reachable with `--grid w8_r3b|w8_r3c` on every stage that already takes `--grid`; a triggered R3 decision table row therefore runs b/c and records "R3a: not run — needs per-state grids" rather than silently substituting.
  Handoff doc `waves/STAGING-READY.md` written (ordered commands, EXPECTED_COMMIT pin, expected wall-clock, F0/F1 reads, the W6FIX V0-V3/R-A..R-D and WAVE8 F0-F3/G1-G7 gate tables, push_results path lists, residual unobserved items). Suite at this revision: 481 passed / 20 skipped.
- 2026-10-08 (deviation, external-reference audit — unsloth.ai "Train your own Decision Model" guide; recorded in `W8-R1-DRAINED.md` §9-§10):
  1. **R2 is SFT-first, RL gated.** `gpu-wave8-train.sbatch` `W8_ARM=r2` now runs SFT + post-hoc `sft-gates` and stops unless `W8_R2_RL=1` (new knob, default 0). §4.2 ordered SFT then the RL rounds in one arm; the plan's own "SFT-only reported as R2-SFT" is now the default order. Rationale: the reference recipe for this model class is pure SFT (1-2 full epochs, no RL), and R1's letter-lock is confounded with RL-from-cold-start — the only supervised epoch ever run on this task was 4.2 % of one (wave-7, §1 row 6). RL rounds cost +~2.1 GPU-h and fire only if the SFT reads trip.
  2. **R2-SFT lr sweep extended to {1e-5, 1e-4, 2e-4}** (`W8_ARM=r2sweep`, array 0-2). The reference working point is 2e-4 at LoRA r=16; our r=8/attention-only is kept for wave-7/R1 comparability (capacity is not the binding constraint — epochs and lr are). Adds ~0.7 GPU-h to the sweep.
  3. **Untrained curriculum control** added to `gpu-wave8-eval.sbatch` (`results/wave8_val/w8-untrained-curriculum-argmax-curriculum.jsonl`: argmax, runs 30, seeds 9500..9529, no adapter) so every band number has an untrained counterpart (W8-R1-DRAINED §8 trap 3: band is `curriculum`, C0 control is `step`/`noisy`).
  4. **Head-vs-readout question deferred**, not decided: the zero-weight token-logit readout (`decision_readout.py`) is the only readout ever trained; a Clef-style per-option head remains the confound-fix behind R2-SFT evidence. `--permute-p 0.5` (half canonical, half shuffled grids) may remove the "fixed-ordinal-scale specialist" signature without new weights.
  5. **Marker ruling still open** (terminal vs advisory, W8-R1-DRAINED §9): not blocking R2-SFT. R1's formal G1 report can be completed anytime via the ~1.7 GPU-h completion run (4 stopped-arm resumes + control) — parked in favour of momentum.


### A1 (2026-10-19) — untrained-readout mass gate narrowed for the 1.7B/4B rungs

- **Why.** Preflight 448805 (rung 1.7B) tripped the `readout_batch(require_mass=0.99)`
  assert at S_nom prompt 11: bf16 mass 0.9632. A full fp32-CPU re-read of all 1,024
  S_nom states (cluster job 449217, `results_probe_fp32_17b.json`) shows it is not a
  precision artifact: fp32 MIN mass is 0.348 (prompt 763), 222/1,024 prompts below
  0.99, 84 below 0.95; prompt 11 reads LOWER in fp32 (0.873). The 0.6B rung passed the
  same gate at >= 0.998 (wave-7 AUDIT-1 3b); the 1.7B base simply does not concentrate
  its vocabulary mass on the candidate letters the way the 0.6B does.
- **External precedent (checked):** the wave-7 A1 switched TRAINED adapters to
  record-not-assert for exactly this quantity (mass erosion observed 0.963-0.973 on
  SFT); AnyJev's decision readout treats answer mass as a REPORTED statistic with an
  exclusion rule, a fatal gate only when mass is ~0 (prompt-format fault); the
  bonzi-1.7B Jev recipe ships median label_mass 0.941 without an assert. Nothing in
  the external decision-model literature asserts 0.99 candidate-letter mass on
  untrained bases above 1B.
- **What stays fatal.** F0 (grid) and F1 (teacher) gates are untouched and remain
  fatal for all rungs. `decision_readout.readout_batch`'s 0.99 default still applies
  to the TRAINED readout of every rung (G7 threshold >= 0.9 recorded). The wave-7
  `untrained readout: assert` rule stays for the 0.6B rung.
- **What changed.** `wave8_gates.control` computes the untrained readout with
  `require_mass=0.0` and RECORDS `min_mass_untrained` in the F2 stats (already
  committed there) and the F3 block; the assert is gone for the untrained path of
  rungs whose preflight so declares. Decision authority for "F3 frozen" remains with
  `wave8_gates.report`, which prints the recorded min mass.
