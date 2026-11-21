# FSDP on the L4 lane: scope for the 8B retrain (2026-11-21)

Question: can the three lost 8B legs be retrained on the cluster with FSDP, for free, without
changing the recipe?

## Why the previous cluster attempt failed, and why this is a different failure

Amendment W8-L4MP (commit `21ad077`, smoke job `460423`) sharded the model with
`device_map="auto"` from accelerate. Two tasks stalled ~90 min at ~0% GPU utilisation in the first
training step with memory frozen, and the diagnosis on record is a training-loop stall between
gradient checkpointing (`use_reentrant=False`) and accelerate's dispatch hooks.

FSDP is not that mechanism. It shards parameters, gradients and optimizer state across ranks and
runs a real backward through the sharded graph, with activation recompute handled by the FSDP wrap
rather than by hooks bolted onto a dispatched model. The failure mode above does not carry over, but
nothing here is tested yet, which is why the first step is a two-step probe and not a leg.

## What makes this tractable

`agent_bullwhip/decision_train.py` is 984 lines and contains no `generate()` call. Training is
teacher-forced over label rows: `sft_step` and `rlcd_step` forward the batch and call backward.
Rollouts live in separate scripts (`grpo_env.py`, `decision_rounds.py`) that load a saved adapter in
their own process. Generating under FSDP is the hard part of most such ports and it is not on this
path.

## Memory, from the model itself

Qwen3-8B in bf16 is ~16 GB. LoRA r=8 on attention projections only (q, k, v, o; `decision_train.py`
line 627) trains roughly 20 M parameters, so AdamW state is a few hundred MB total. On 4×L4 that is
~4 GB of weights per rank with ~18 GB left for activations at micro-batch 1. Grad checkpointing may
not even be needed, which would keep us away from the one combination that has bitten this project.

## The six touchpoints

1. `_Policy.__init__` (line 635): process group from the SLURM environment, rank to local GPU, FSDP
   wrap after `_attach_adapter`, `use_orig_params=True` (mandatory with PEFT: only LoRA params are
   trainable and they must stay individually visible after flattening), auto-wrap on
   `Qwen3DecoderLayer`, `device_id=local_rank`, `sync_module_states=True`.
2. The `.to(self.device)` guard at line 695 already skips when `hf_device_map` exists; it needs the
   same skip when wrapped, otherwise it raises on a sharded model.
3. `_ensure_optimizer` (line 745): `requires_grad` filtering still works under `use_orig_params`,
   but must be checked on a wrapped model rather than assumed.
4. `save_checkpoint` (line 174) calls `model.save_pretrained`, which cannot run on sharded weights.
   Needs `FSDP.summon_full_params` on rank 0 for the adapter only, about 31 MB gathered, and a
   sharded-to-full transition for `optimizer.pt`.
5. Micro-batch accumulation (line 112 `split_microbatches`, 8 micro-batches per 32-row step) needs
   `model.no_sync()` on all but the last micro-batch, or every micro-batch pays a full all-reduce
   and the leg crawls.
6. The sbatch launch shape: FSDP needs one process per GPU, where the current script runs one
   process and uses `--processes ${SLURM_CPUS_PER_TASK}` for CPU rollout workers. This must be
   env-gated to the 8B rung so 0.6B, 1.7B and 4B are untouched.

Estimate: 150 to 200 lines across `decision_train.py` and one new launch path, plus the probe cycle.
One day of work, not a research project.

## Step 1, the port (free, half a day to a day)

Write the six touchpoints above behind `W8B_FSDP=1`, off by default, so every existing rung keeps its
current code path. Nothing is launched until the flag is on and the probe below passes.

## Step 2, the probe (free, about 15 minutes of queue)

Two-step smoke on **2×L4**, the smallest thing that can distinguish a working port from another
stall. It tests the port, so it cannot come before it:

- 2 optimizer steps, micro-batch 1, batch 2, no grad checkpointing, `W8B_FSDP=1`
- **pass criteria**: both steps complete; loss finite on both; LoRA `lora_A`/`lora_B` gradients
  non-zero and finite after step 1; the two steps do not produce identical loss; wall time per step
  under 90 s
- **stop conditions**: any stall over 10 min, any OOM, or zero LoRA gradients (which would mean the
  port runs but trains nothing, the exact silent failure this project has already been bitten by)
- cost: zero, about 15 minutes including queue

A probe that only checks that the process exits is worthless here. The gradient assertion is the
gate.

## Step 3, one leg

4×L4, the same recipe as October (3 rounds × 170 steps, batch 32, micro-batch 4, lr 1e-5 seed 1
first, because that is the arm whose G1 already exists and would give a direct sanity comparison
against 22,083 on the band). Estimated 5 to 7 h wall against 1h28m on an A100-80G, because L4 bf16
throughput is roughly 0.39× an A100 and FSDP communicates over PCIe with no NVLink.

Stop: if the round-1 validation band mean lands outside 2× of 22,083, stop and report rather than
grinding through three rounds on a port that is producing a different model. If the leg completes,
run seeds 0 and 1 for the 5e-5 arm and seed 0 for 1e-5.

## What this does not buy, and must not be described as

**A re-execution, not a recovery.** FSDP changes reduction order, so a retrained 5e-5 arm will not
reproduce the October band means 35,289 / 12,444 exactly. The paper would report new arms trained on
the same recipe with a different sharding, dated and disclosed, not the October arms restored.

**The selection question reopens.** October selection used the A100 band means. If the retrained
5e-5 arm's band mean lands worse than the retrained 1e-5 arm, the frozen rule's choice is contradicted
by the new run. That does not license a switch: it gets reported as a selection instability, and it
is a stop condition, not an invitation to re-pick.

**Queue time is the real price.** Three legs at 5 to 7 h each is 15 to 21 GPU-hours on a lane whose
12 L4s are frequently fully allocated, as they were for all of last night. Free in euros, not free in
days.

## Recommendation

Run Step 0 now. It costs nothing and it answers the only question that matters: whether FSDP on this
stack trains at all, with gradient evidence rather than an exit code. Everything downstream is
worthless if that probe fails, and the port should not be written before it passes.
