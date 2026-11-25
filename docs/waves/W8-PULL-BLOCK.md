# Wave-8 results pull, step by step (v3 — after the 429581 no-op diagnosis, 2026-10-15)

## What the logs showed (already read on the hub)

* The R2+RL rounds **already ran on 2026-10-12** (job 419129, submitted from the SFT-verdict tree
  before `wave8-next.sh` existed): all three seeds completed 3 rounds of real RL with collapse
  reads and the curriculum-validation band. Everything A2 wanted already exists on the cluster.
* 429581 and 432642 (2026-10-15) were **correct no-ops**: the resume logic found the SFT at step
  2827 and the RL rounds already had round state, so they exited in seconds. My skip-check glob
  (`wave8_r2rl_seed*`) missed the actual arm naming (`wave8_r2rl_lr0.0002_seedN`), so the array
  resubmitted twice for nothing. Bookkeeping bug in the script, zero wasted GPU (both tasks no-op'd).
* The 1.7B preflight (432656) **crashed in the HF download path** (weights not reachable on the
  compute node — needs the B1 snapshot verified before a resubmit).

## What to run on the login node now (one block, ships everything back)

```bash
R=/scratch-shared/<cluster-account>/bullwhip/agent-bullwhip-replication
B=/scratch-shared/<cluster-account>/bullwhip
cd "$R" && git pull --ff-only

bash hpc-umbrella/push_results.sh -m "wave8: r2rl arms + val bands (job 419129) + preflight 432656" \
     results/wave8_r2rl_lr0.0002_seed0 \
     results/wave8_r2rl_lr0.0002_seed0_round1 \
     results/wave8_r2rl_lr0.0002_seed0_round2 \
     results/wave8_r2rl_lr0.0002_seed0_round3 \
     results/wave8_r2rl_lr0.0002_seed1 \
     results/wave8_r2rl_lr0.0002_seed1_round1 \
     results/wave8_r2rl_lr0.0002_seed1_round2 \
     results/wave8_r2rl_lr0.0002_seed1_round3 \
     results/wave8_r2rl_lr0.0002_seed2 \
     results/wave8_r2rl_lr0.0002_seed2_round1 \
     results/wave8_r2rl_lr0.0002_seed2_round2 \
     results/wave8_r2rl_lr0.0002_seed2_round3 \
     results/wave8_r2sft_seed0/state.json \
     results/wave8_r2sft_seed1/state.json \
     results/wave8_r2sft_seed2/state.json \
     results/wave8_val/w8-val-r2rl-lr0.0002-seed0.jsonl \
     results/wave8_val/w8-val-r2rl-lr0.0002-seed1.jsonl \
     results/wave8_val/w8-val-r2rl-lr0.0002-seed2.jsonl \
     results/wave8_logs/wave8-train_419129_0.log \
     results/wave8_logs/wave8-train_419129_1.log \
     results/wave8_logs/wave8-train_419129_2.log \
     results/wave8_logs/wave8-preflight_432656.log

git log --oneline -1    # paste this back to me
```

## What I do on the hub once this lands

1. Paired G1 read on the three band files vs the untrained control (the exact statistic in
   `W8-R2-SFT-VERDICT.md` section 11).
2. Per-round collapse rows -> the section 6.6 R2+RL verdict ("also locks -> INCONCLUSIVE-negative
   at 0.6B, go to the ladder" or the alternative).
3. Diagnose the 1.7B preflight crash and patch wave8-next.sh's skip-glob so the next rerun is
   genuinely idempotent.

## Do NOT run now

* Anything that submits jobs (`wave8-next.sh`, sbatch). The R2+RL class is complete; the 1.7B
  preflight needs the B1 weight investigation first.
* A second eval (`gpu-wave8-eval.sbatch`). The band files from 419129 hold the answer; an eval job
  would re-derive the same numbers.
