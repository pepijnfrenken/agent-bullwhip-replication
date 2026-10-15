# W8 1.7B start block (2026-10-15)

One command starts the 1.7B ladder. All knobs verified against this tree
(`config.sh` MODEL_REVISION_17B=70d244cc..., `gpu-wave8-preflight.sbatch` W8_RUNG=1.7b reads
`gates_1.7b.json`, `gpu-wave8-train.sbatch` W8_ARM default r1 / W8_SEEDS default 3 and
refuses to run without F0/F1 + rung F3).

`hpc-umbrella/start-1.7b.sh` is an idempotent stage machine. Run it, and re-run it after
each stage finishes:

```bash
cd /scratch-shared/20234428/bullwhip/agent-bullwhip-replication
git pull --ff-only          # bring HEAD to the hub commit (7d611af or later), then:
bash hpc-umbrella/start-1.7b.sh
```

What each rerun does, depending on where you are:

1. **B1** - verifies (or downloads on this login node) the Qwen3-1.7B snapshot into
   `$B/hf`, revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` (config.sh pin), and refuses
   to continue without a weight blob > 2.5 GB. This is the fix for preflight job 432656:
   the crash was the HF download path from a compute node (no egress); staging now happens
   on the login node before any sbatch.
2. **B2** - `sbatch --export=ALL,W8_RUNG=1.7b kit/gpu-wave8-preflight.sbatch`
   (~1 GPU-h; job-name bw8-preflight). Produces `results/wave8/gates_1.7b.json`
   (F2 tokenizer prompt_sha + F3 frozen) plus `S_nom_control-1.7b.json`.
3. **C** - when `gates_1.7b.json` shows F2 pass + F3 frozen, submits
   `sbatch --export=ALL,W8_RUNG=1.7b,W8_ARM=r1,W8_LRS=5e-5,W8_SEEDS=3 --array=0 kit/gpu-wave8-train.sbatch`
   (seed 0 only, ~1.7 GPU-h). When seed0 completes cleanly, rerun this script in a commit
   with the remaining seeds.

Built-in guard: stops everything when `bw8-*` GPU-h since 2026-10-07 reaches 60 (wave cap
70; GPU-h count printed each run).
