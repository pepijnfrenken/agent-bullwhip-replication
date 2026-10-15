# Wave-8 consolidation — repo cleanup + 1.7B go/no-go (2026-10-15)

This block is the user-facing one-page summary. Full verdicts live in the referenced docs.
Run everything from the login node after `git pull --ff-only`, from the repo root.

## Part 1 — what the failures actually were (all three already diagnosed)

| arm | job | outcome | verdict doc |
|---|---|---|---|
| r2 SFT only | (earlier waves) | 1.5-2.9x WORSE than untrained on all 3 seeds; G1/G2/G4 all fail | W8-R2-SFT-VERDICT |
| r2 SFT+RL (RLCD) | 419129 | G1 FAIL. seed0 passes (+20.9%), seed2 marginal (+10.6%), seed1 catastrophic (median 27,638 = 94.7% WORSE than untrained 14,194, 0/30 runs beat untrained). Pooled sign test shows no detectable improvement (48/90, p≈0.6) | W8-RL-VERDICT |
| r2 re-submits 429581/432642 | both | correct no-ops (skip-glob name mismatch) — zero GPU used | W8-PULL-BLOCK |
| 1.7B pre-flight | 432656 | crash, HF download attempted on a compute node that has no egress | this file, Part 3 |

## Part 2 — repo cleanup (do this once now, keeps future pulls small)

Root cause of the recent pain: `results/` is gitignored normally, but wave-8 keeps
force-adding entire per-round label/state jsonl data (~250 of the 373 MB tracked at
HEAD). Do the following:

1. Slim the tracked result tree (keep all verdict-grade evidence; drop redundant
   full history): 
 reproduce later. Keep `states_argmax.jsonl` (5.7 MB each), `collapse_*.json`,
 `train_log.jsonl`, `pi_sft.json`, `state.json`, `round_state.json`, `DONE`,
 `anchor_table.json`, manifests, and the wave8_val band files + wave8_logs.
 Delete from tracking: `round_labels.jsonl` (~7 MB x 9), `policy_labels.jsonl`,
 `behavior_labels.npz`/`.jsonl`, `states.jsonl` (non-argmax), and wave7's
 `policy_states_seed*.jsonl` if no longer referenced (they cost 23 MB).
 
 Expected new tracked size: ~90-110 MB instead of 373 MB.

2. Change `push_results.sh` default: replace the current unconditional exclude with
   a keep-list model (argmax states + collapse + train logs + gates + val bands),
   and use `git rm --cached` (not `git reset --hard`) for anything already tracked
   that we now wish to drop. NEVER use `git reset --hard` inside this repo again —
   on shared repos it strips the worktree.

3. Checkpoint policy going forward (write into the wave-8 block):
   - Checkpoints stay on scratch (`$B/checkpoints/wave8_r2*.ckpt`), referenced by
     path in the verdict doc, never in git.
   - If a checkpoint becomes needed for a paper figure, zip on the cluster, upload
     to the S3-compatible mirror used by the release agent, and reference the URL
     from the verdict doc.

## Part 3 — the 1.7B go/no-go

### What the 1.7B needs before it can even start
The 1.7B ladder pre-flight (job 432656) FAILED because the compute node tried to
reach huggingface.co for a weight snapshot the login node should have cached.
Concretely: '$B/hf/models--Qwen--Qwen3-1.7B' does NOT actually exist on scratch
(stage.sh --with-ladder was cut off by the Stage-B1 getcwd bug on the 14th).
Resubmit = run stage B1 weight download by hand on the login node first, verify the
directory exists with the expected files, then rerun wave8-next.sh which will
detect gates_1.7b.json missing and rerun the pre-flight sbatch.

### Investment that is already sunk (do NOT rescind this)
- 27.7 GPU-h of wave-8 GPU spend (of 70).
- The 0.6B record is complete: R1 done earlier; r2 SFT arm harmful; r2+RL ran clean
  and is unreadable as "helping"; all 3 collapse guards silent (they never fire).
- The r2 verdicts are USEFUL, not wasted: they kill a whole hypothesis branch
  cheaply. Nobody should spend 30 more GPU-h on r2.

### What "moving to a bigger model" actually buys
The entire premise of the wave-8 scaling ladder (W8-NEXT-SUBMITS §3-4) is: does the
4.7 s/step process that failed/did-not-fire at 0.6B change命运 when the model scales?
0.6B r2 answers "the r2 recipe fails at 0.6B". Scaling tests whether that failure is
_size-specific or recipe-specific.

### The honest go/no-go
Recommendation: PROCEED with the 1.7B ladder, but at REDUCED scope:

1. Fix B1 first (weight download on login node, `stage.sh --with-ladder` by hand,
   verify snapshot dir). Cost: 0 GPU-h, ~20 min of download.
2. Pre-flight gate (F2 prompt_sha_equal + F3 frozen) — rerun, ~1 GPU-h.
3. Stage C 1.7B R1 at 1 LR (5e-5), one task first (--array=0), ~1.7 GPU-h for seed0,
   then 3 seeds ~5.5 GPU-h only if s/step and stability read clean.
   STOP before the two-lr variant unless the one-lr read is ambiguous (
   W8-NEXT-SUBMITS §3.4's own rule).

Expected total additional spend: ~8-10.5 GPU-h to a clean 1.7B R1 readout,
total wave-8 ~37-39 GPU-h, well inside the 60-submit/70-cap.

### What NOT to do
- Do not resubmit r2 at 0.6B or 1.7B (Stage D is on HOLD for a reason: arm-dir
  collision, 31 GPU-h for 3 seeds, already documented).
- Do not rescue seed1 of the 0.6B r2+RL arm with a lower LR — it is a reproducible
  recipe property (bad grad_norm tail, KL drift), not seed noise; the cheaper
  experiment is the 1.7B R1 ladder rung, which tests the same question.
- Do not push more big label/state jsonl blobs through git.
