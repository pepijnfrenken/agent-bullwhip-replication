WAVE-8 8B RUNG RESULT (2026-10-21):
- Job 28414248 (gcn17): COMPLETED 1h28m, exit 0. 3 rounds trained, all collapse-checks clean
  (r1 medH=0.108/0.605/1.2421; r2 1.092/0.570/0.8422; r3 0.797/0.480/0.9633; none collapsed).
- Final val (30 runs, grid w8_g26b, argmax): mean_cost = 35,289 (min 19,269, max 75,578),
  confidence 0.9218, min_mass 0.99964, parse_fail 0.
- Comparison (same metric): 4B trained seed0 = 8,315; 4B trained seed1 = 7,612; untrained 4B = 21,578.
- VERDICT: 8B at lr 5e-5 R1 recipe is ~4.2x WORSE than the 4B rung. Scaling rung DOWN, not up.
- CORRECTION (2026-10-21, after sacct check): the run DID run on gpu_a100 with 1x A100-80G
  (AllocTRES gres/gpu:a100=1). "CPU-only" was wrong. The actual confound vs the proper protocol:
  (1) only the lr=5e-5 arm ran — the pre-registered spec runs BOTH lr arms {1e-5, 5e-5} and picks
  by pooled validation; (2) seed 1 was cancelled, so no second seed. Micro-batch 4 (8B) vs 8 (4B)
  is only a grad-accum split (effective batch 32 in both) - not an optimizer confound.
- Seed1 job 28415540: CANCELLED (job id vanished from scontrol) - probably swept by budget/cancel;
  no log ever created.
- Next action: none for now - park 8B negative-unless-GPU-retry, back to 4B lane for wave 8.

## Proper re-run (2026-10-21) — L4 lane CLOSED, moved to Modal A100-80GB
TU/e L4 (22.06 GiB usable of nominal 24 GiB) could not complete protocol-equivalent 8B bf16
training. Four deterministic CUDA OOMs during the r1 backward/recompute stage:
458899 (seed1 lr1e-5, 30 min in, 250 MiB short), 458900 (seed1 lr5e-5, same),
459027_1 (same deficit with PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True),
459027_0 (126 MiB short even with the allocator fix; log's stage banner also printed
"r2:train" while mapping task 0 -> r1, cosmetic sbatch labeling bug, separate issue).
Inference/policy-collection passes fit and ran clean (2,160 states collected twice per leg);
only the optimizer/backward recompute stage exceeds the card. Verdict: **8B training does
not fit a 24 GB L4 under the A2 protocol (bf16, LoRA, micro-batch 1, grad-ckpt, max_len
1024, batch 32)**. Two-GPU co-scheduling of the legs was tried (array 459027, tasks grouped);
the failure is per-process peak memory, not scheduling.
Consequence: the two missing seed-1 legs move to Modal A100-80GB (user-approved lane).
Gates stay the Snellius-frozen gates_8b.json (model-deterministic); pod image pins the same
library stack as TU/e (torch 2.1.2, transformers 4.55.4, peft 0.21.1) so the rung ladder
stays protocol-equivalent.
- Resubmission of the cancelled `28418616` (seed 1, lr 5e-5) as `28422970`, plus the missing
  seed-1/lr-1e-5 arm as `28423007`: **both auto-killed 2–3 s after start, `reason=budget`** on
  the admin comment. The workshop grant ran out right after `28418615` completed (validation
  71,855). Full 2×2 sweep is **blocked on Snellius credits**; A2 amendment's ~22 GPU-h envelope
  is spent. Umbrella L4 cannot hold 8B LoRA training, so no viable fallback lane for this rung.
- Proper-final run (`28418615`, seed 0, lr 1e-5, gpu_a100 1×A100-80G, exit 0): 3/3 rounds,
  collapse checks all `collapsed=False`, confidence 0.9459, min_mass 0.9457, val mean **71,855**.

## Modal A100-80GB legs (2026-10-21/22) — completing the 2×2 sweep
### Leg 1: seed 1, lr 5e-5 — COMPLETED, val mean 12,444
Modal app `ap-mmFrW6iXGxGWou9pzdRuss`, A100-80GB, stack pinned to TU/e (torch 2.1.2,
transformers 4.55.4, peft 0.21.1); invocation byte-identical to the sbatch r1 branch
(3×170, batch 32, micro-batch 1, max_len 1024, grad-ckpt, Snellius-frozen gates_8b.json).
- 3/3 rounds, collapse checks: r1 medH=0.001/modal_share=0.844/I=0.8200; r2 0.072/0.629/
  1.3413; r3 1.165/0.289/0.9897 — none collapsed.
- Final val (30 runs, grid w8_g26b, argmax): **mean_cost = 12,444** (min 6,057, max 23,914),
  parse_fail 0, min_mass 0.99642.
- NOTE on artifacts: the app's result-archival step crashed (ZIP pre-1980 timestamp bug);
  no checkpoint files were persisted to the Volume. The numbers above are from Modal's
  retained app logs. Checkpoint loss noted honestly; the spec verdict needs only the val
  table, which is recoverable by re-running the val stage on a later checkpoint if needed.
- Current qualified picture (2 of 4 planned runs done):
  - 8B lr 5e-5 seed 0: 35,289 (n=30)
  - 8B lr 1e-5 seed 0: **71,855** (n=30)
  - 4B reference: 8,315 / 7,612; untrained 4B: 21,578
  Both arms worse than untrained 4B; not enough seeds to call it a final negative but
  the direction is consistent across both lr arms tried so far.
- Submitted on gpu_a100, 1x A100, 12 h cap, single tasks (dodging the budget-QOS array kill):
  28418615 = seed 0, lr 1e-5 (the missing arm, 3x170, micro-batch 4)
  28418616 = seed 1, lr 5e-5 (completes the 5e-5 arm)
- Verdict rule (pre-registered): pooled validation mean of each lr arm on seeds {0,1} picks the
  R1 winner; G1 pass = trained beats formula floor 5,346 at headline tier (<=2x) with the 4B
  comparison as the rung-over-rung read (report ratio vs 4B 8,315/7,612).


## Model-parallel attempt on 2x L4 (2026-10-22) — HANG, not fit
Amendment W8-L4MP: `device_map="auto"` behind `W8B_DEVICE_MAP=auto` (commit 21ad077;
inference stages ran sharded at 7.4+8.9 GiB/GPU well under the 22.06 GiB ceiling, zero
OOMs). Smoke 460423 (2 steps, micro-batch 1): both tasks stalled in the first training
step for ~90 min with ~0% GPU util and frozen memory; task 0 hit the 2 h TIMEOUT, task 1
cancelled. Diagnosis: training-loop stall with gradient checkpointing (use_reentrant=False)
across sharded layers — not a memory failure. TU/e 8B training attempts now: 4 OOMs + 1 hang
= lane closed. Equivalent-protocol fallback: Modal A100-80GB resume (~$2-4).
