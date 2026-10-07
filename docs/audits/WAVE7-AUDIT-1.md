# WAVE7-AUDIT-1 — adversarial audit of ../waves/WAVE7_SPEC.md + jevstyle_readout.py (design pass 1)

Status: COMPLETE (session 2, 2026-10-05). Read-only; CPU only. Probes/outputs in `/tmp/w7audit/`.
Severity: P0 = voids a pre-registered gate or premise. P1 = wrong training signal, silent corruption, or first-job crash. P2 = breaks a reuse claim or costs a cluster cycle. P3 = hygiene.

## 1. Labeler design (§1 D3, §3, §5)

Verdict: **sound in principle**, **unsound as specced**. Simulator Q-values under a fixed policy are a valid target: one policy-improvement step over the continuation. Evidence: `labeler_probe.py` (stdlib, imports only `engine`). Setup: 160 states from 40 Poisson-curriculum episodes; behavior = floor + U{−6..6}; continuation = `OrderUpToAgent(3.0,0.5)` logic. Output: `labeler_probe.out`. Ranked:

1. **[P1] Hindsight labels.** §3's J is the "realized cost of the remainder of the episode", i.e. one draw of future demand the policy cannot see. Compared against E[J] over M=16 resampled futures with common random numbers (CRN): argmin differs in **50/156** non-flat states (mean |Δ| 5.1 units, max 40; target TV median 0.11, mean 0.17). Fix: E[J] over M≥16 CRN futures from the episode's curriculum parameters. Measured cost: 0.22 s/state, vs 13 ms single-path. That fits the spec's own 0.2 s budget.
2. **[P1] The continuation is stateful.** `OrderUpToAgent._forecast` persists across weeks (`agents.py:47,51`). Fresh agents at the branch point disagree with warm ones on the argmin in **73/156** states, and §3 does not say which to use. Fix: replay each role's forecast from week 0, with an equality test against an unbranched run.
3. **[P1] The grid truncates the optimum.** E[J] argmin = 40 (top of grid) in **68/160** states, and = 0 in 32/160. The 0..40 grid fits step demand (4→8), not the curricula: λ~U[5,20], μ≤20 (`grpo_env.py:90,103`). D3's random/wave-6 behaviors push backlog past 500 (§3b). Fix: widen the grid or make it state-relative; gate on the edge-argmin rate.
4. **[P1] "Walk-forward tuned floor" is undefined on curricula and leaks.** It grid-searches (θ,λ) on `demand[:18]` *of the same path* (`../../scripts/walkforward_paths.py:31-38,46`). Fix: fix (θ,λ) once, tuned on held-out curriculum episodes, and write the values into the spec.
5. **[P2] The τ rule cannot be global.** The spread of E[J] across candidates has median 400 at t<12 vs 120 at t≥24, against mean J of 11,467 vs 1,165. Per-state τ = std_a E[J] gives median normalized entropy 0.87. Choose τ by closed-loop cost on curriculum validation episodes, and pre-register the rule.
6. **[P2] Flat/terminal states.** 12/160 realized labels (4/160 under E[J]) are flat across all 21 actions. The E[J]-flat ones are mostly factory at t=35, where the order lands after the horizon (`engine.py:240`). The week is in the prompt, so end-of-horizon dumping is learnable. Fix: mask t ≥ H − lead.
7. **[P2] Same-week co-decisions unspecified.** J needs the other three roles' week-t orders, which come from either the behavior or the continuation policy; the spec must pick one.
8. **[P2] Closed-loop shift (§5, only partly handled).** The labels are Q^floor: "deviate once, then everyone plays floor". At eval, all four roles play the learned policy. The two-policy agreement report detects this; it does not fix it. Make one relabel round on learned-policy states mandatory (it is "optional" in §5).

## 2. RLCD formulation (§1 D4, §3)

The on-disk reference has no recipe. `astro_decision.py` is inference-only (`:87-168`). No README or card exists in `jevsrc/`. The adapter config targets **Qwen3.5-4B** (`astro_adapter_config.json:6`), with r=8, α=16, q/k/v/o (`:20,31,34`). "Defaults to Qwen3-0.6B" holds only for the inference script (`astro_decision.py:20`). Training claims (noise on logits, LOO): **INCONCLUSIVE**.

- **[P1] Sign error.** §3 writes "advantage = J − mean(J)", but J is a cost, so ascent on A·∇log π favours expensive actions. It must be A = −(J − b).
- **[P3] Not LOO:** the baseline includes the sample itself. Use b_i = mean_{j≠i} J_j.
- **[P2] Scale.** Raw J spreads differ about 3× by week (§1.5), so early weeks dominate the gradient. Normalize per state by std_K (as in `grpo_env.py:325-343`).
- **[P1] Wrong estimator.** All 21 actions can be scored for 13 ms (single path) or 0.22 s (M=16). That gives the exact gradient of Σ_a π(a)E[J(a)] with zero variance. Sampling K=8 with replacement from a peaked SFT policy instead produces duplicates and zero advantages.
- **[P1] The on-policy state distribution is unspecified.** "Fresh 20k-state subset" does not say the states come from rolling out the *current* policy. Without that, RLCD differs from SFT only in its loss, and R1 measures noise.
- **[P2] The update is under-specified.** "One update per round, 2–3 rounds" reads as 2–3 optimizer steps. State epochs, lr, and a KL/entropy anchor to SFT; mode-seeking PG plus argmax eval risks collapse.

## 3. Compatibility

### 3a. Tokenizer + 4.55.4 API

- **A..U single tokens — VERIFIED on 4.57.6 (run, `tokcheck.py`).** Ids A=32…U=52. For every label L, `tok(prompt+L)` = `tok(prompt)`+[id(L)]. No BOS is added. Space variants are separate ids (" A"=362). `padding_side` defaults to **right**.
- **4.55.4 tokenizer — VERIFIED by construction, not run:** the ids come from the snapshot's `tokenizer.json`. **[P2]** `stage.sh:101` has no revision pin; pin `c1899de289a04d12100db370d81485cdf75e47ca`, as the reference does (`astro_decision.py:21`).
- **`enable_thinking` — VERIFIED (static).** `apply_chat_template(**kwargs)` (`tokenization_utils_base.py:1518-1534`) forwards to Jinja (`utils/chat_template_utils.py:374-384`). The render ends `assistant\n<think>\n\n</think>\n\n` (run on the hub). If the kwarg were dropped, the read position would land on `<think>`; the D2 mass assert catches that, so keep it fatal.
- **`logits_to_keep` — VERIFIED (static).** It exists in 4.55.4: `modeling_qwen3.py:456,494`.
- **`dtype=` — FALSIFIED for 4.55.4 (static trace, not run).** `from_pretrained` pops only `torch_dtype` (`modeling_utils.py:4622`). The config has no `dtype` attribute (`configuration_utils.py:221,279`), so the kwarg survives `return_unused_kwargs` (`auto_factory.py:549`) into `model_kwargs` (`modeling_utils.py:4853`). It then reaches `cls(config, **model_kwargs)` (`:4999`) and `Qwen3ForCausalLM.__init__(self, config)` (`modeling_qwen3.py:429`), which raises **TypeError**. **[P1]** The seed (`jevstyle_readout.py:47`) and the reference (`astro_decision.py:106`) both pass `dtype=`. Use `grpo_train.py:206-210`'s `torch_dtype` + TypeError fallback.
- 4.55.4 declares `torch>=2.1` (`dependency_versions_table.py:95`), so torch 2.1.2 is in range. **[P2]** Numerics differ (fp32 CPU vs bf16 L4), and near-tie argmaxes will flip. Run every compared arm on one stack.

### 3b. Re-run + bias probe

**VERIFIED** reproduction: retailer_w6 → `H`=14, p=0.292; factory_w12 → `H`=14, p=0.584; letter mass 0.9987/0.9990.

**FALSIFIED: §0's "coherent order distributions zero-shot on real game states".** Source: `probe_bias.py`, full run, output in `probe_bias.out`.

- **[P0] The argmax follows the letter, not the value.** Holding the retailer_w6 state fixed and changing only the option map: ascending `H`=14, descending `A`=40, rot7 `A`=14, shuffle1 `A`=8, shuffle2 `L`=34. Factory gives `H,H,H,L,L` (14/26/28/32/34). Across all 10 permuted readouts the argmax letter is in {A,H,L}.
- **[P0] No response to state, and the wrong sign.** Flooded (oh=80, demand 4) → 14. Starved (backlog 80, demand 30) → 8. **Empty material → `L`=22, p=0.50.**
- **[P1] Engine-reachable states rendered in repo field text.** In 8/8 states the argmax is `H`=14 or `L`=22, for on_hand 0–1695 and backlog 0–533.
- Mass ≥0.998 everywhere, including the empty prompt. The D2 mass invariant shows the format is followed; it shows nothing about the decision. The C1 control is therefore a near-constant letter-prior policy.

### 3c. Seed code review (`jevstyle_readout.py`)

- **[P1] Both probe states are unreachable in the engine.** `advance` sets `s=min(oh+r, inc+b)` (`engine.py:233-234`) and subtracts `s` from both on-hand and backlog (`:248-249`). One of the two is therefore always 0, so oh=12 with b=3 (`:26-27`) cannot occur. Factory o=0 with recent orders [8,9,10] (`:36,39`) is also impossible: with the 2-week lag (`engine.py:32,240`), o ≥ 19.
- **[P1] Field text is not the repo's.** This contradicts §3's "identical field text". Differences: a "Your stage:" line; "ordered, not yet received" vs `agents.py:342`; "Incoming order last week" vs `:343`; 3 recent orders vs the engine's `[-6:]` (`engine.py:210`); no goal sentence (`:337`). `build_prompt` itself ends with "Reply with ONLY an integer" (`agents.py:377`), so readout.py needs its own renderer, golden-tested against `observations()`.
- **[P2] Batching.** `logits[0,-1]` (`:60`) is single-sequence. With the default right padding, a batched `[:, -1]` reads pad positions. Use left padding (as `astro_decision.py:101` does), `logits_to_keep=1` (full fp32 logits are ≈220 MB at T=361), a no-truncation assert, and a batch-vs-single equality test.

## 4. Reuse map (§2)

"Reuse as-is" is **FALSIFIED** for the analysis seam, pull verification, and resume.

- **[P0] `../../hpc-umbrella/eval-analyze.py` on deterministic arms.** The step demand is unseeded (`runner.py:144`) and argmax is deterministic, so all 30 step records are identical. Consequences:
  - `paired_deltas` reports p≈1.9e-9 (`eval-analyze.py:172-194`). Ties count as "not worse", so even identical policies come out "significant".
  - `cost_stats` gives CV=0 (`:50-59`), and the cell-std frac<1 is 1.00 (`:97,120`).
  - The collapse flag fires whenever every role uses ≤4 distinct orders (`:252-259`).
- **[P2] `../../hpc-umbrella/eval-analyze.py` assumptions.** It needs a `failures` dict (`:167`) plus `run`, `orders`, `total_cost`. It assumes exactly two arms named untrained/trained (`:215-216`), prints a GRPO parse line (`:323-324`), and accepts only step|noisy (`:217`). Floor keys: `wf_tuned` (step) vs `tuned` (noisy).
- **[P2] `../../scripts/paired_analysis.py`.** It infers the pattern from **file mtime order** (`:31-33`). Its references are `walkforward_paths.json`, which has 10 passes (`../../scripts/walkforward_paths.py:20`). There are no wave-6 refs, so C3 cannot be computed with it.
- **[P2] `../../agent_bullwhip/runner.py`.** It reads `.failures/.calls/traces` (`:170-175`). A missing attribute is caught by a broad `except` and retried 5× with sleeps (`:181-188`), then logged as `total_cost: None`. `run_game` calls the policy once per decision (`engine.py:287-290`); batching needs `run_episodes_batched` (`grpo_env.py:229`) plus a JSONL adapter.
- **[P2] `../../hpc-umbrella/verify_pull.py`** hard-codes the 13 wave-6 runs (`:17-31`).
- **[P2] Resume (`../../agent_bullwhip/grpo_train.py`).**
  - `save_checkpoint` reads GRPO-only args (`:827-834`).
  - Only `random.Random` is saved (`:824`); torch RNG is not (`:220-223`), so RLCD sampling is not reproducible across a resume.
  - There is no data cursor, epoch, or scheduler state, so SFT re-sees or skips data after a SIGTERM unless the shuffle is derived from (seed, step).
  - `_Policy` hard-codes LoRA r=16/α=32 on q,v (`:211-212`), against the spec's r=8/α=16, and calls `eval()` permanently (`:219`).
  - Reusable as-is: atomic versioned checkpoints, `is_trainable=True` + zero-trainable guard (`:1016-1025`), the finite check, the SIGTERM path, and log dedupe.
- **[P3]** The tokenizer load lacks `local_files_only` (`:191`). Compute nodes have no egress (`OPERATOR.md:208`).

## 5. Eval comparability

- **[P0] The step arm is n=1.** "Step n=30 (seeds 1000+i)" (D5) is wrong: `runner.py:141-144` seeds only noisy. Under argmax, C1, C3 and C4's CI degenerate. Fix: report step as one labeled cost. Add a sampled-readout arm (`torch.Generator` seeded 1000+i) as the counterpart of wave 6's T=0.7 runs. Put the inferential gates on **noisy n=30**, and regenerate `walkforward_paths.json` for 30 passes.
- **C2 — VERIFIED.** The step wf_tuned floor is 3,206 (`floors_summary.json`, `eval-a0-report.md:93`). So 4× = 12.8k (the "~13k" is the threshold) and 2× = 6.4k. Write these numbers into the spec.
- **Demand hygiene — VERIFIED by construction.** The labeler draws only from `sample_curriculum_demand` (`grpo_env.py:84-111`). Keep the §5 test, and keep seeds disjoint from 1000–1029.
- **[P2]** `test_no_leak` must cover the new renderer, which bypasses `build_prompt`. Upstream `incoming_now=None` (`engine.py:209`).
- **[P3]** The serving stack differs from wave 6 (llama.cpp Q4, T=0.7; `eval-a0-report.md:4-12`). The even-only 0..40 action space changes order-variance metrics.

## 6. Cluster plan (§6, D7)

- **VERIFIED:** `tue.cpu3.q` (`config.sh:70`); `tue.gpu.q` with `-C l4` (`:72,75`); pins torch 2.1.2 / transformers 4.55.4 / peft 0.21.1 (`:65`; `requirements-umbrella.txt:28-29`).
- **[P2] `config.sh --check` fails on a labeler sbatch.** Only `cpu-harness.sbatch` may use the CPU partition (`config.sh:299-309`). Extend the rule to `cpu-*.sbatch`.
- **Walltime is feasible.** `WALLTIME_CPU=01:00:00` (`:79`). 100k states × 0.22 s ≈ 6.1 core-hours, which fits as ≥8 one-core array tasks.
- **[P2] Staging.**
  - `../../hpc-umbrella/stage.sh` fast-forwards **master** (`:63-64`), so wave-7 code must be merged first.
  - Set `EXPECTED_COMMIT` (`config.sh:54`); the job-side gate is `gpu-campaign.sbatch:62-65`.
  - The weights revision is unpinned (`stage.sh:101`).
- **[P2] `gpu-campaign.sbatch` hard-codes `grpo_train` and the GRPO TSV** (`:77,94-100`). Write new files, copying the `SLURM_SUBMIT_DIR` block (`:38-45`), the commit stamp (`:56-65`), and the TERM loop (`:102-117`).
- **[P3]** No `HF_HUB_OFFLINE` in `bw_load_env` (`config.sh:187-191`).
- **[P3]** Scratch is wiped after 14 days (`:34-35`). Assert determinism with a SHA-256 of a label slice, hub vs cluster; on the hub, CPython 3.14 and 3.11 give identical probe output (run).

## Top-10 correctness risks (ranked, with fixes)

1. **[P0] The zero-shot premise is false** (letter prior, §3b). Rewrite §0. Gate: after training, the argmax *value* is invariant under ≥3 option permutations on ≥90% of validation states. Augment training with permuted options; keep the shuffle control arm.
2. **[P0] Step n=30 is one deterministic game.** Report step as a single cost; add a sampled arm seeded 1000+i; gate on noisy n=30.
3. **[P1] Hindsight labels** (32% argmin disagreement). Use E[J] over M≥16 CRN futures.
4. **[P1] Cold-start continuation** (47% argmin disagreement). Replay the forecast state; add an equality test.
5. **[P1] Grid truncation** (42% of argmins at 40). Widen or use a state-relative grid; gate on the edge rate.
6. **[P1] RLCD ascends cost and lacks on-policy states.** Use A = −(J−b)/std, exact 21-action scoring, and current-policy rollouts.
7. **[P1] The "wf-tuned" continuation leaks and is undefined on curricula.** Fix (θ,λ) out of sample, in the spec.
8. **[P1] Q^floor labels vs closed-loop eval.** Make one relabel round mandatory before reading C1–C3.
9. **[P1] The first cluster job crashes or mis-renders.** `dtype=` TypeError; hand-typed unreachable states; unpinned snapshot. Use the `torch_dtype` fallback, render from `observations()`, pin `c1899de2…`, and add a prompt-hash test.
10. **[P2] The reuse seam breaks silently.** Spurious collapse/significance in `eval-analyze`; mtime patterns in `paired_analysis`; the runner's broad-except retries; `verify_pull` names; the `--check` partition rule; `save_checkpoint` args/RNG/cursor. Adapt and test each.

## Minimum safe first cluster job — checklist

- [ ] Spec amended for risks 1–8 and approved; code on `master`; `EXPECTED_COMMIT` set.
- [ ] `../../hpc-umbrella/verify_pull.py`:
  - [ ] renders from `observations()` only; golden prompt SHA at revision `c1899de2…`;
  - [ ] in-context label-id assert; mass ≥0.99 fatal;
  - [ ] left padding + `logits_to_keep=1`; batch-vs-single equality; no truncation.
- [ ] Model load: `torch_dtype` with TypeError fallback, `local_files_only=True`, pinned revision (also in `../../hpc-umbrella/stage.sh`).
- [ ] `labeler.py`:
  - [ ] branch-copy equals `run_game`; forecast-replay test;
  - [ ] CRN E[J]; flat-rate and edge-rate report; fixed (θ,λ);
  - [ ] seeds disjoint from 1000–1029; curricula-only test; hub label-slice SHA.
- [ ] `test_no_leak` extended; `test_engine_golden` green.
- [ ] sbatch passes `config.sh --check` (CPU rule extended), with the `SLURM_SUBMIT_DIR` block, TERM loop, `HF_HUB_OFFLINE=1`, and walltime within pins.
- [ ] `decision_train` checkpoints the data cursor/epoch and torch RNG; hub resume drill gives an identical loss trace.
- [ ] Eval and analysis:
  - [ ] eval JSONL has `failures/calls/traces`;
  - [ ] analysis labels step as n=1;
  - [ ] `verify_pull` EXPECT updated;
  - [ ] C2 written as 12.8k/6.4k.
- [ ] **Job 1 = CPU labeler smoke** (1k states), SHA matches the hub. **Job 2 = GPU readout smoke:** 10 noisy paths; parse-fail 0; mass ≥0.99; argmax agreement with the hub on 50 fixed prompts (near-ties may flip).

## INCONCLUSIVE

- **transformers 4.55.4 at runtime** (dtype TypeError, template, tokenizer): traced statically only. Installs are forbidden and `clvenv` is incomplete.
- **RLCD recipe training details:** no training code or card on disk. The HF card and Clef were not fetched.
- **Labeler magnitudes:** come from a stand-in behavior (floor + noise), the untuned floor, and Poisson only. The spec's random/wave-6 behaviors likely raise the edge rate [INFERENCE]. Not run on the cluster.
- **Cluster:** per-core speed, GPU-hours, and queue caps cannot be measured from the hub.
- **bf16/L4 readout numerics:** not run.
