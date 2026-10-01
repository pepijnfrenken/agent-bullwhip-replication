# AUDIT-RUNTIME-2026-10-01 — GRPO sweep 351673 (Umbrella L4): runtime failures

Scope: read-only audit of `agent_bullwhip/grpo_{train,env}.py`, `hpc-umbrella/` and the 13 pulled logs. No repo edits, no commits.

Library citations come from pinned sources (transformers v4.55.4, peft v0.21.1, torch v2.1.2), fetched and checked by sha256 against the tags. The fetched `generation/utils.py:2617,3643` and `peft_model.py:1012` match the traceback frames line-for-line.

CPU probes live in `/tmp/bwaudit/` and use the repo's env code plus the cached Qwen3-0.6B `tokenizer.json` (no torch). Baseline `pytest tests/ -q`: 243 passed; no test touches a torch path. Line numbers are for HEAD `51d3d5a`.

## 1. Verdict

**What killed a2/a3/a5/a8/a10:** a non-finite value in the LoRA policy under fp16. It surfaced one step late, in `torch.multinomial`'s input check during the next `generate()`. Confidence: **high** for the chain below; **medium** for which fp16 op overflowed first (nothing logs it).

1. **The objective is unbounded.**
   - `loss_pg = -(a*seq_logp).mean()` (`grpo_train.py:266`) has no ratio and no clip; `clip=0.2` (`:226`) is never read.
   - The scored string is `str(parsed_order)` (`grpo_env.py:188-189`), not the sampled text.
   - Rows the policy rates at e^-20…e^-100 per token produce the logged −16.8…−49.5. For a<0 the gradient does not vanish as p→0.
2. **The update geometry amplifies it.**
   - `opt.step()` runs once per micro-batch of 4 (`:235-295`): 288 AdamW steps per training step at G=8, H=36.
   - Each micro-batch is one episode-week and carries one advantage (288/288 measured). 97.6% of consecutive steps share the advantage sign; the median longest same-sign run is 76.
   - Adam normalises step size, so `clip_grad_norm_(…,0.5)` does not bound how far the weights move.
3. **fp16 runs with no guard.**
   - The model is loaded fp16 (`:139-141`), but Qwen3-0.6B ships `"torch_dtype": "bfloat16"`. There is no GradScaler and no finite check.
   - The first inf/NaN becomes all-NaN gradients inside `clip_grad_norm_` (`clip_grad.py:69-82`).
   - AdamW then NaNs all 2,293,760 LoRA params and both moment buffers, and the next `generate` fails `max()<INF & min()>=0` (`Distributions.cpp:609-612`).

**The array ran pre-fix code.** The traceback frames (`grpo_train.py` 156/319/481/547/553) match `653247f^` exactly; HEAD has 165/330/492/558/564. That is the right-padding flood.

**The "completed" runs learned nothing usable.** A CPU replay of a *constant-order* policy reproduces the logged advantage min/max to <1e-9 at every printed step from:
- large constant order: a0 step 20 on, a1 from 10, a6 from 60, a11 at 10;
- order in 0..~7: a4 from 60, a7 from 130.

Orders stopped depending on state. The flat loss is β·(KL term) alone.

**Safe to rerun as-is? No.** HEAD fixes only the padding during generation. It still:
- trains the unbounded fp16 objective with no guard;
- scores 69% of completions from a pad position;
- crashes on every `--resume` that has an adapter (F5);
- loses the walltime checkpoint (F10).

## 2. Findings

| ID | Sev | Symptom | Mechanism | Evidence (file:line) | Minimal fix (sketch, not applied) |
|---|---|---|---|---|---|
| F1 | P0 | multinomial `inf/nan` crash, 5/13 tasks | fp16 overflow → non-finite loss/grad → `clip_grad_norm_` multiplies every grad by NaN/0 → AdamW NaNs all LoRA params and moments → next generate | `grpo_train.py:139-141` (fp16), `:291-295` (no check); torch `clip_grad.py:63-82`; `Distributions.cpp:609-612`; Qwen3 `config.json` bf16 | Load in `torch.bfloat16` (L4 = sm_89 supports it). `gn = clip_grad_norm_(…)`; if `!isfinite(gn) or !isfinite(loss)`, skip `opt.step`, count it and log it (§3.2) |
| F2 | P0 | losses of −16.8/−27/−49.5; PG trains on text never sampled | completion recorded as `str(parsed int)` and re-tokenised; the sampled token ids are thrown away | `grpo_env.py:188-189`; `grpo_train.py:172-174` → `:330-331` (`_, texts`); `:240` | Carry the generated ids (or the exact decoded text, minus pads) through `DecisionRecord` and score those. Keep the parsed order as a separate field |
| F3 | P0 | divergence within 4-33 steps | 288-576 AdamW steps per training step; one advantage per micro-batch, so no in-batch baseline; long same-sign runs; rollouts go stale against a policy that has already moved hundreds of steps | `grpo_train.py:235-295`; `grpo_env.py:70-71` (same reward for all K); `grpo_train.py:341-345` (episode-major order); `probe_pad.py`, `probe_runs.py` | Accumulate gradients over the whole step (loss/n_micro) and call `opt.step()` once. If several steps are needed, shuffle rows and store old logprobs for a PPO ratio |
| F4 | P0 | no objective-level trust region | `clip` is unused. "KL" = `exp(s)(s−r)` is minimised at s=r−1 and its gradient ∝e^s vanishes when s≪0 | `grpo_train.py:226`, `:263-266`, `:288` | Ratio clip with stored old logprobs, plus per-token k3 KL: `exp(r_t−s_t)−(r_t−s_t)−1` |
| F5 | P0 | resubmitting a2/a5/a8 dies at startup | `PeftModel.from_pretrained(…)` has `is_trainable=False` → adapter frozen → `AdamW([])` raises `ValueError("optimizer got an empty parameter list")` | `grpo_train.py:548-551`, `:200-201`; peft `peft_model.py:423,496`, `tuners_utils.py:2064-2065`; torch `optimizer.py:260-261` | `is_trainable=True`; assert trainable count > 0 after resume |
| F6 | P0 | stale code ran the sweep with nothing in the log saying so | no commit stamp in the log; `stage.sh` swallows fetch/pull failures | traceback lines vs `git show 653247f^`; `stage.sh:54-55`; `gpu-campaign.sbatch:62` (header has no commit) | Echo `git rev-parse HEAD` and `git status --porcelain` in the sbatch. Abort unless HEAD == `EXPECTED_COMMIT` from `config.sh` |
| F7 | P0 | collapse to a constant order goes undetected for 280 steps | no entropy, completion-diversity, parse or order metrics are logged | `grpo_train.py:497-506` | Log the metrics listed in §3.4 |
| F8 | P1 | wrong conditional when scoring | prompts are right-padded in `grpo_step`, so the first completion token is read from logits at a PAD position for 790/1152 rows (retailer 288/288, pad width ≤14). Positions = `arange`, which shifts RoPE by the pad width | `grpo_train.py:239,244-261`; `modeling_qwen3.py:371-378`; `tokenization_utils_base.py:1389` | Build each row as a contiguous `[prompt|completion]` and pad only at the end, or left-pad and pass `position_ids = attn.cumsum(-1)-1` |
| F9 | P1 | sampled distribution ≠ scored distribution | `generate()` without an explicit config inherits `top_k=20, top_p=0.95` from the model's `generation_config.json`; scoring uses the full softmax | utils `:1729-1734,1779`; `generation_config.json`; `grpo_train.py:165-169` | Pass `top_k=0, top_p=1.0` explicitly (or score under the same warpers) |
| F10 | P1 | walltime SIGTERM loses the checkpoint | trapped TERM makes bash `wait` return at once, the script `exit`s while python is still finishing its step, and Slurm reaps the job | `gpu-campaign.sbatch:73-81`; local probe: script exited 4 ms after forwarding, child alive, checkpoint written 3 s later | `while kill -0 $PY_PID 2>/dev/null; do wait $PY_PID \|\| RC=$?; done` |
| F11 | P1 | silent restart from a fresh LoRA | `rmtree(ckpt)` → `os.replace` leaves a window with no adapter; `optimizer.pt` is written non-atomically *after* the rename; resume quietly falls back to "step/RNG only" | `grpo_train.py:422-428`, `:546-556` | Versioned `checkpoint-<step>/`, write `state.json` last, abort when `adapter_saved` is true but the dir is missing |
| F12 | P1 | `a11_seed2` is a replicate, not a seed variant | no `--seed`; loop RNG fixed at `Random(12345)`; torch never seeded | `grpo_train.py:354-371`, `:485`; `campaign.tsv:14` | Add `--seed`, use it for the loop RNG and `torch.manual_seed`, record it in `state.json` |
| F13 | P1 | a checkpoint can be saved with NaN weights | `save_checkpoint` never checks finiteness; a NaN on a step where `(step+1)%10==0` gets persisted | `grpo_train.py:402-444`, `:508-512` | Check `isfinite` on all trainable params before saving; refuse and abort otherwise |
| F17 | P1 | weak, demand-dominated advantage | each of the G episodes in a step draws its *own* demand process (new λ/μ), so a per-(agent, week) z-score largely measures demand luck. With constant actions it measures *only* demand luck (§3.4) | `grpo_train.py:326-327`; `grpo_env.py:268-286` | Share one demand path across the G episodes of a step (common random numbers), so the group differs only in policy samples |
| F14 | P2 | the "large order" attractor | `extract_int` has no upper bound (8 tokens allow up to 99,999,999); the engine only floors at 0 | `grpo_env.py:171-172`; `engine.py:220` | Clamp to a documented max and log how often it clamps |
| F15 | P2 | the bench is confounded | `bench_rollout` trains the shared policy (288 AdamW steps per repeat) between chunk sizes | `bench_rollout.py:138-140` | `--skip-train-step`, or a fresh policy per chunk size |
| F16 | P2 | dead reference | `_PromptCache` exists only in a comment; prompts are re-tokenised every micro-batch (perf only) | `grpo_train.py:117-119` | Delete the comment, or implement the cache |

Probes: `/tmp/bwaudit/probe_{const,timeline,pad,runs}.py`, `batch.sh` + `child.py`.

## 3. Suspect list verdicts

### 3.1 NaN propagation: VERIFIED (mechanism); INCONCLUSIVE (first op)

- **Dtypes.** The base model is fp16. peft upcasts the LoRA params to fp32 (`mapping_func.py:110` `autocast_adapter_dtype=True`; `tuners_utils.py:628-641`), so the AdamW state is fp32. The LoRA output is cast back to fp16 (`lora/layer.py:1100,1113`), so any q/v value above 65504 becomes `inf` there.
- **inf turns into NaN.** Qwen3 RMSNorm (`modeling_qwen3.py:58-63`) and `q_norm`/`k_norm` (`:198-199`) map one `inf` to NaN for the whole token: variance=inf, so `x·rsqrt(inf)` = inf·0.
- **Bounds.**
  - s ≤ 0, so `seq_logp.exp()` ≤ 1 (`:288`) cannot overflow.
  - Since r ≤ 0, `e^s(s−r) ≥ e^s·s ≥ −1/e`, so β·KL ≥ −0.037 even at β=0.1.
  - |a| ≤ √(G−1). The logs show it: min_adv −2.6457 = −√7 at G=8, −3.8728 = −√15 at G=16.
  - The head gradient ∂loss/∂s_i = −a_i/4 with ∂s/∂logit ∈ [−1,1], so the loss algebra cannot overflow.
  - **[INFERENCE, medium]** The first inf therefore comes from activations driven by grown LoRA weights (the v_proj → residual/MLP path, or the q/v cast above).
- **`clip_grad_norm_` cannot rescue** (`clip_grad.py:63-82`, `error_if_nonfinite=False` by default):
  - norm=NaN → coefficient NaN → every grad becomes NaN;
  - norm=inf → coefficient 0 → inf grads become inf·0 = NaN.

  The right guard checks the returned norm: `if not torch.isfinite(gn) or not torch.isfinite(loss): zero_grad(); skip opt.step()`.
- **Poisoning.** NaN grads make `exp_avg`, `exp_avg_sq` and the params NaN permanently. The remaining micro-batches of that step return NaN loss, and the next rollout's first `generate` crashes. This matches each death being in the rollout right after the bad step.
- **Death steps** from `train_log.jsonl` sizes at 189 B/record (a0: 56747/300): a10 → step 4, a3 → 9, a5 → 15, a2 → 33, a8 → ~146.

### 3.2 No NaN tripwire: VERIFIED

There is no `isfinite`/`isnan`/`manual_seed` anywhere in `agent_bullwhip/`. stdout prints every 10 steps (`:505`); `train_log.jsonl` (not pulled) has every step, with `NaN` written by `json.dumps`.

**Minimal guard:**
- **(a) `grpo_step`:** per micro-batch, check `isfinite(loss) & isfinite(gn)`. On failure, skip the step, count it, keep the first event's micro index, loss_pg, loss_kl, `s.min()` and `logits.abs().amax()`, and return a dict.
- **(b) `run_loop`:** log those fields every step. On the first event, save `checkpoint-prenan/` (still finite because the step was skipped), append `{"event":"nonfinite","step":…}`, set `"aborted":"nonfinite"` in `state.json`, and exit 3.
- **(c) `save_checkpoint`:** refuse non-finite params (F13).

**Abort, don't skip.** While F2-F4 stand, skipping only postpones the divergence (a8 spent 70+ steps in the violent regime), and silently skipped updates make sweep arms incomparable. A skip budget is acceptable only after bf16 plus F2-F4.

**Naive resubmit:**
- a2/a5/a8 checkpoints (steps 29/9/139) are almost certainly finite. Later rollouts ran on them, and a NaN in any q/v LoRA tensor reaches every token.
- The resubmit still dies deterministically at startup (F5).
- With F5 patched, a2 (−8.96 at step 30) and a8 (−18.6 at step 140) resume inside the divergent regime, so re-divergence is expected [INFERENCE].
- a3/a10 have no checkpoint and restart at step 0.

**Smallest safe change: do not resume.** After F1-F5 the algorithm differs, so every config restarts at 0.

### 3.3 Padding (generation + scoring): VERIFIED

- **Deployment.** The pre-fix frames prove the clone lacked `653247f`. HEAD's fix does suppress the warning:
  - The trigger is `batch_size>1 and sum(inputs[:, -1]==pad)>0` (`utils.py:2394-2402`).
  - Real-tokenizer probe on a 16-prompt chunk: right-padded, 15/16 rows end in pad; left-padded, 0/16. Qwen adds no BOS/EOS (`add_bos_token:false`).
  - The warning counts equal the call counts (21600=300×72, 31200=300×104, 10800=300×36).
  - **To verify:** `git -C $REPO_DIR log -1 --format='%H %s'` and `grep -n 'padding_side = "left"' agent_bullwhip/grpo_train.py`.
  - **Likely route [INFERENCE]:** `stage.sh:54-55` swallows pull failures.
- **Deployed right-padded generation.** A short row's first token is sampled from the last column, a `<|endoftext|>` pad at position 1 (`utils.py:604-605`). That is a pad-conditioned first token for ~15/16 rows per call.
- **Scoring (deployed and HEAD).**
  - `tok(ps, padding=True)` (`:239`) right-pads, because `padding_side="right"` is the default (`tokenization_utils_base.py:1389`; Qwen3 sets nothing, and HEAD restores it at `:171`).
  - The mask span (`:258-260`) selects the correct target ids. But the first target is predicted from index `plen-1`, which is a pad for 790/1152 rows: the retailer in 288/288 micro-batches (its prompt is ~10 tokens shorter), mean pad 3.3, max 14.
  - Positions default to `arange` (`modeling_qwen3.py:377-378`), so the completion sits w positions further from the prompt than during generation.
- **Net.** `seq_logp` scores `str(order)` (F2), in a shifted context for 69% of rows (F8), under the full softmax rather than the top-k/top-p sampler (F9). It does not score the generated tokens.

### 3.4 Flat loss: VERIFIED

- **Derivation.** Each (agent, week) group has Σa=0 (`grpo_env.py:281-285`), so loss_pg is the within-group covariance −(1/N)Σaᵢ(sᵢ−s̄). If the policy emits one string with p≈1, then s≈0 and loss ≈ −β·r̄. That value is fixed by the base model's log-prob of the collapsed string.
- **Data.** The plateau scales with β, giving two attractors:
  - a0 0.0309/0.02 and a8 (steps 40-50) 0.155/0.1 → r̄ ≈ −1.55;
  - a7 0.039/0.005 and a4 0.153/0.02 → r̄ ≈ −7.7.
- **Replay (`probe_timeline.py`).** The constant-order families match at the steps listed in §1. The large family is invariant for K ≳ 136-200 and the small one for K ∈ 0..~7, so K cannot be identified. a0 step 299 ≡ a6 step 299 and a0 step 20 ≡ a7 step 20 bit-for-bit (same seed → same demands → identical actions).
- **Why it stays stuck.** The softmax is saturated (∇s≈0), advantages are pure demand luck (F17), and sampling from p≈1 explores nothing.
- **Interpretation.** a0/a1/a6/a11 learned "always order ≥~200"; a4/a7 learned "always order ~0". None is a usable policy or ablation arm.
- **Should be logged per step:**
  - loss_pg and loss_kl separately;
  - mean/min s;
  - first-token entropy;
  - unique completions and unique orders;
  - parse-failure rate;
  - order mean/std/max;
  - fraction of rows with s > −0.01 and with s < −10;
  - pre-clip grad norm, `n_nonfinite`, LoRA norm;
  - mean episode cost.

### 3.5 Violent negatives: VERIFIED (mechanism); INCONCLUSIVE (knobs)

- **Source.** The KL term is ≥ −0.037, so −49.5 is loss_pg. That needs mean |s| ≳ 49.5/2.65 ≈ 19 nats/token on many rows, correlated with a. Top-k/top-p samples cannot be that improbable.
- **[INFERENCE, high]** Those rows are the F2 mismatch: no digit → scored "0"; or `' 12'`→`['Ġ','1','2']` and `'12\n'`→`['1','2','Ċ']` re-scored as `['1','2']` (tokenizer probe). F8's pad-conditioned first token adds to it.
- **Timing.** The spikes sit on attractor transitions: a4 goes L(10-20) → −16.8 → S; a7 goes L(20) → −49.5 at 40 (already the S family) → S.
- **Unbounded push.** ∂(−a·s)/∂logit = −a(1−p), which tends to |a| as p→0, so improbable tokens keep being pushed down at full force: the logit growth that precedes overflow.
- **Clamps: none.** No ratio clip, no advantage clip beyond √(G−1), no logp floor, no order cap, constant lr with no warmup, and `clip_grad_norm_` does not bound Adam's step.
- **Per-step drift ≈ n_opt_steps·lr:**

  | Run | AdamW steps × lr | Outcome |
  |---|---|---|
  | a10 | 288 × 2e-5 | died at step 4 |
  | a2 | 576 × 1e-5 | died at 33 |
  | a4 | 416 × 1e-5 | survived |
  | a3 | 192 × 1e-5 | died at 9 |
  | a1 | 144 × 1e-5 | survived |

  Not monotone, with n=1 per config and unseeded sampling. lr is the only knob with a direct mechanism. β does not protect: a8 (β=0.1) died while a7 (β=0.005) recovered, as F4 predicts.

### 3.6 Sampler crash surface: VERIFIED

- **Path.** NaN weights → NaN logits, which pass through the warpers → NaN softmax → multinomial raises.
- **Cheap guard.** Catch it at the source (§3.2). Optionally run an `isfinite` sweep over the 2.3M trainable params before each rollout.
- **Retry is unsafe.** With NaN weights every retry fails; with overflowing logits a retry samples the same broken model. `remove_invalid_values=True` (`InfNanRemoveLogitsProcessor`) would hide the failure and sample garbage.
- **Exit 1 is the correct fail-fast.** Array tasks are independent. The defect is losing the pre-NaN state and the first bad step.

### 3.7 Other long-run hazards (most likely to recur first)

1. **F5 resume crash.** Certain on any resubmit or walltime split.
2. **F6 stale deploy.** Already happened.
3. **F10 + F11.** Lost or silently fresh adapter at walltime. a2 runs ~195 s/step, ≈16 h for 300 steps against an 18 h limit.
4. **F12.** Certain mislabel; corrupts the variance estimate.
5. **F13.** Needs the NaN to land on a save step (~1 in 10).
6. **Append-only train log.** A resume re-appends the steps after the last checkpoint, so dedupe by last occurrence.
7. **Memory: FALSIFIED.** MaxRSS 1.87 GB after 300 steps vs 1.86 GB after 4; bench peak VRAM 5.3-6.2/23 GB; the loss is detached (`:296`).
8. **F16.** Perf only.
9. **Eval scripts.** `walkforward_paths.py` is CPU-only, deterministic and never loads an adapter; no script evaluates GRPO checkpoints yet. `bench_rollout.py`: see F15.

## 4. Minimum safe rerun checklist

**Code:**
1. bf16 load, with `assert torch.cuda.is_bf16_supported()` (F1).
2. Finite guard + abort + pre-NaN checkpoint + finiteness check before save (F1, F13).
3. Score the generated token ids; keep the parsed order separately (F2).
4. One `opt.step()` per training step via accumulation, or shuffled minibatches with stored old logprobs and a real ratio clip; per-token k3 KL (F3, F4).
5. Contiguous `[prompt|completion]` scoring rows or explicit `position_ids`; explicit `top_k=0, top_p=1.0` (F8, F9).
6. Resume with `is_trainable=True`, assert trainable > 0, versioned checkpoints, `state.json` last, abort when the adapter dir is missing (F5, F11).
7. `--seed` (loop RNG + torch), one shared demand path per group, an order cap (F12, F17, F14).
8. The §3.4 metrics.

**Deploy:**

9. Re-stage. The sbatch prints HEAD and the dirty flag and exits unless HEAD == `EXPECTED_COMMIT` (F6).
10. Wait loop in the sbatch (F10).

**Gate:** a 30-step GPU job each for a0 and a10. Pass only if:
- every loss is finite and `n_nonfinite` is 0;
- loss_pg is not ≡0 and first-token entropy is > 0;
- each step has more than one unique order;
- there are zero padding warnings;
- `scancel --signal=TERM` at around step 12 checkpoints and then resumes with the same step, optimizer loaded and finite loss.

**Config:** re-choose the lr grid after F3, because AdamW steps per training step drop from 288 to 1. Keep `SAVE_EVERY=10`.

**Outputs:**
- **Invalid** for the sweep: all 13 checkpoints and loss curves, including a9/a11, which are still running the same code (kill them).
- **Usable:** the logs as failure evidence, `bench_351229.json` for generation throughput only, and the `walkforward_paths.py` floors.

## 5. Honesty note

**Not verifiable here** (no GPU, no cluster, torch absent by design):
- **(a) The first fp16 op to overflow.** No norms or maxima were logged.
- **(b) NaN in `grpo_step` vs only in `generate`.** Run `tail -n 3 results/grpo_{a2_group16,a3_horizon24,a5_poisson_only,a8_kl_high,a10_lr_high}/train_log.jsonl` on scratch. A last record with `"loss": NaN` means the NaN arose in `grpo_step`. A finite last loss means either a NaN in the last micro-batch's grads, or finite weights whose logits overflow at generation.
- **(c) The actual completions and orders.** Never logged. The constant-order claim rests on advantage extremes matching to <1e-9 at 30+ printed steps per run; the exact K is unknown.
- **(d) F5.** Derived from peft/torch source, not executed.
- **(e) Slurm reaping python after the batch script exits.** Standard behaviour, but this cluster's `proctrack` config was not checked. The bash half was reproduced locally.
- **(f) The exact deployed commit.** Any commit from `eda9c54` to `653247f^` gives the same frames.
- **(g) CUDA fp16 `log_softmax` accumulation.**

**First rerun must log:**
- git HEAD and the dirty flag;
- the §3.4 metrics plus `max|logit|`;
- three sampled rows per step (prompt hash, generated text, parsed order, scored ids);
- the torch seed;
- the result of the resume drill.
