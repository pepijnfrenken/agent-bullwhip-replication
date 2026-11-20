# Closing the 8B hole: run proposal (2026-11-20)

Status: **proposal only, nothing submitted.** Pino approved the path ("close the hole"); the lane,
cost and stop conditions below are what needs his go.

## What the hole is, decomposed

The abstract says 8B is not evaluable within size for two reasons. They are separable, and they have
very different prices:

- **A. No matched untrained-8B control evaluation.** Pure inference on the untrained checkpoint. No
  adapter needed.
- **B. No final-eval generation at 8B.** Needs a trained 8B adapter to evaluate.

## What actually survives (read this session)

| artifact | evidence | state |
|---|---|---|
| `results/wave8/gates_8b.json` | 1,497 bytes, in tree, committed | present |
| 8B lr1e-5 seed1 checkpoints | Modal volume `wave8-hf-8b`, `results/wave8_r1-8b_lr1e-5_seed1_round{1,2,3}/checkpoint-{9..169}` | **present** |
| 8B lr1e-5 seed1 adapter | `checkpoint-169/adapter_model.safetensors`, 30,709,192 bytes | **present** |
| 8B lr1e-5 seed1 band val | `w8-val-r1-8b-lr1e-5-seed1.summary.json`, `runs=30`, mean 22,083.37, min 10,401, max 40,466, `min_mass` 0.99205 | present |
| 8B lr1e-5 seed1 run state | `round_state.json`, `DONE`, `collapse_reads.jsonl` | present |
| 8B lr5e-5 seed1 archive | `leg_5e-5_seed1.zip` | **22 bytes: the crashed archival, empty** |
| other two arms (5e-5 s0, 1e-5 s0) | nowhere in tree, cluster home, scratch, or volume | **gone** |

Consequence: **one of the four 8B arms is evaluable today with inference alone.** The other three are
not, and no further inference will recover them.

The 30.7 MB adapter is the important line. `docs/waves/WAVE8_WORD8B_RESULT.md` line 51 states that no
checkpoint files were persisted to the Volume. That is true for the 5e-5 leg (its zip is 22 bytes) and
false for the 1e-5 seed-1 leg, which persisted three rounds of checkpoints plus its band table. The
paper itself does not claim otherwise, but S5 should record that one arm's weights survive.

## A warning that shapes the expected outcome

The 8B gate file's own frozen untrained readout carries `min_mass = 0.22706`, and its untrained median
entropy is 1.2947. Compare the 4B gate file, `min_mass = 0.99761`. The untrained 8B does not commit to
the 26-letter action set anywhere near as cleanly as the 4B does.

So Part 1 may produce a control that **fails G7's 0.99 mass threshold**, exactly as the untrained 1.7B
did (0.60945 on the band). If that happens the answer is not "rerun until clean", it is that the
untrained 8B has no usable readout, which is itself the within-size finding and would replace a
refusal with a mechanism.

## Part 1: matched untrained-8B control (free)

- **What:** Qwen3-8B at `b968826d9c46dd6066d109eabc6255188de91218`, no adapter.
  (i) 30-path curriculum band, argmax, seeds 9500..9529: the metric the paper's 8B band means are on.
  (ii) 30-path noisy-argmax generation: the metric G1 names.
- **Lane:** TU/e Umbrella, `tue.gpu.q`, `--gres=gpu:2`, `--mem=16G`, 8 h. Two GPUs because the
  verdict doc records 8B inference fitting sharded at 7.4 + 8.9 GiB; one 22 GiB L4 at batch 1 may fit
  but need not be risked. 8B bf16 weights are ~16 GB.
- **Cost:** zero. Two jobs, roughly 1 to 2 GPU-hours each, behind whatever queue the L4s have.
- **Stop condition:** stop and report if `min_mass < 0.99` on either generation, or if loading needs
  more than 2 GPUs. Never shrink the batch, never switch to a quantised readout: either would change
  the readout and make the control incomparable to the trained arms.

## Part 2: final-eval generation for the one surviving arm (free)

- **What:** download `checkpoint-169` (30.7 MB adapter) from the volume, evaluate it with the same
  30-path noisy-argmax generation Part 1 uses, against the Part 1 control.
- **Lane:** same L4 lane, one job, adapter + base model.
- **Cost:** zero, under an hour.
- **Stop condition:** stop if `min_mass < 0.99` for the adapter read, or if the adapter does not load
  against the pinned base revision. Report rather than substitute a different checkpoint: round 3
  step 169 is the arm that produced val 22,083, and any other checkpoint is a different arm.
- **What it yields:** a genuine G1 read for 8B lr1e-5/seed1, and the band comparison for that arm
  against a matched control for the first time.

## Part 3: the other three arms (paid, deferred)

- **Why:** their adapters are gone; only retraining recovers them.
- **Shape:** 3 arms (5e-5 s0, 5e-5 s1, 1e-5 s0) × 3 rounds × 170 steps at the same recipe.
- **Lane:** Modal A100-80GB, same pinned stack (torch 2.1.2, transformers 4.55.4, peft 0.21.1),
  `gates_8b.json` frozen as-is.
- **Cost estimate:** 4 to 8 GPU-hours per arm on an A100-80G, so 12 to 24 GPU-hours, roughly **$40 to
  $80** at Modal's on-demand A100 rate. Estimate, not a quote; the previous legs ran inside a 12 h cap
  and one was killed by a budget QOS on the cluster, which is why this would be Modal again.
- **Risk:** the recorded failure modes are CUDA OOM at the r2 backward on 24 GB cards, a training-loop
  stall with `use_reentrant=False` gradient checkpointing, and one budget kill. An A100-80G avoids the
  first; the stall and the budget remain.
- **Stop condition:** abandon the rung if any arm stalls the loop twice, or if spend reaches the
  approved ceiling. Report the rung as partially recoverable rather than spending past the ceiling.

## What each outcome does to the paper

| outcome | 8B row becomes |
|---|---|
| Part 1 control is mass-clean, Part 2 runs | band-level within-size read with a matched control, plus a G1 read for one arm. Abstract stops saying "no matched untrained-8B control evaluation" |
| Part 1 control fails mass | the untrained 8B has no usable closed-choice readout; the row becomes a mechanism, not a gap |
| Parts 1 and 2 only, Part 3 declined | one arm evaluable, three band-only; the row is "partially recovered", stated arm by arm |
| Part 3 funded and lands | full 2×2 with matched control, same shape as the 4B row |

## Recommendation

Run Parts 1 and 2 now. They are free, they take an afternoon of queue time, and either outcome is
better than the current refusal: one gives a real 8B verdict, the other gives a reason. Hold Part 3
until those results exist, because if the untrained 8B control fails the mass gate, retraining three
more arms buys nothing the paper can use.

Nothing here is submitted. The lane, the spend ceiling and the go are Pino's.

## Submission record (Parts 1 and 2, approved 2026-11-20)

Pino approved Parts 1 and 2 only. Nothing from Part 3 was submitted and no Modal spend was incurred.

| item | value |
|---|---|
| job | `780799` `bw8-eval`, `tue.gpu.q`, `--gres=gpu:1`, `--mem=64G`, 8 h cap |
| export | `EXPECTED_COMMIT=0bcd9df`, `W8_RUNG=8b`, `W8_EVAL_SEEDS=1`, `W8_LR=1e-5`, `W8_EVAL_RUNS=30` |
| base model | `Qwen/Qwen3-8B@b968826d9c46dd6066d109eabc6255188de91218`, 16 GB fetched to the scratch HF cache |
| adapter | `results/wave8_r1-8b_lr1e-5_seed1_round3/checkpoint-169/adapter_model.safetensors`, 30,709,192 bytes, staged from the Modal volume; `state.json` records lr 1e-5, seed 1, step 169, saved 2026-10-22T14:06:37Z |
| log | `/scratch-shared/20234428/bullwhip/logs/wave8-eval_780799.log` |

The job runs the untrained 8B control first (three generations plus the 30-path curriculum band), then
the trained arm for seed 1. Part 1 and Part 2 therefore land in one job and share the same demand
paths, which is what makes the pairing valid.

`decision_eval` sets `require_mass = 0.0` on the main path and records `min_mass` in every summary
rather than aborting on it (`agent_bullwhip/decision_eval.py:339`). A low mass on the untrained 8B
control is therefore an outcome to read, not a crash to retry, exactly as the stop condition intends.

The 8B adapter's `state.json` records `labels_sha = a9ce6cf5...`, against `02097b5a...` for the 1.7B
arms and `e7db0006...` for the 4B arms. That value is model-dependent by construction (it differs
between the 1.7B and 4B arms on the same labels file, `results/wave8/S_nom.jsonl`), so it is not
evidence of a label mismatch.

### Job 780799 failed in five seconds: the 8B gate file was never in git

`780799` exited 1 after 5 s with `FATAL: results/wave8/gates_8b.json missing (F3 not frozen for
this rung)`. The file exists on the local disk (1,497 bytes, mtime 2026-10-21 15:04) and the paper's
8B row is derived from it, but `results/` is gitignored (`.gitignore:4`) and it had never been
force-added, unlike `S_nom.jsonl`, `f0.json` and the other wave-8 inputs. The cluster checkout only
sees what was pushed, so it has never held the 8B gate freeze.

This is a retention failure on the 8B row itself, not a job configuration bug. Fixed by force-adding
the file at `3f71c90` and pulling the cluster to that commit; both copies now hash
`b6194407e03adb35d10d4c54edd0a33c14a9b01d4772a551e4202147db8b1f90`. The file's own content confirms
the freeze: `Qwen/Qwen3-8B@b968826d`, `H_lo` 0.0005025333805207596, `m_hi` 0.14874833921755284,
`min_mass` 0.22705760598182678.

Resubmitted as `780926` with the same knobs and `EXPECTED_COMMIT=3f71c90`.
