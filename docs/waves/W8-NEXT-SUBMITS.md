# W8 next submits - operator sheet (2026-10-12)

Companion to `docs/waves/W8-R2-SFT-VERDICT.md`. Everything here is verified against the kit on
this tree (`hpc-umbrella/gpu-wave8-{train,eval,preflight}.sbatch`, `config.sh`) - the knob names,
required settings and failure modes below are read from those files, not assumed.

## 0. Submit boilerplate

The cluster is not reachable from the hub (VPN-only), so every submit is run by hand on the login
node. Three rules that apply to every command below:

* `EXPECTED_COMMIT` **must be exported** (`gpu-wave8-train.sbatch` fails with
  `FATAL: EXPECTED_COMMIT not exported`). It must equal the repo HEAD, and the script compares
  `REPO_HEAD` against it and refuses to run stale code. Compute it on the cluster, never paste a SHA
  from elsewhere. Dirty files are printed (`dirty_files=N`) but do not block.
* `SBATCH` directives are read before the shell runs, so knobs are passed by environment, not edited
  into the files: `sbatch --export=ALL ...`.
* Do not write `VAR=$(...) sbatch --export=ALL,EXPECTED_COMMIT=$VAR` - in that form `$VAR` expands to
  the old (empty) value. Export first, then submit.

```bash
B=/scratch-shared/$USER/bullwhip; R=$B/agent-bullwhip-replication; K=$B/kit
export EXPECTED_COMMIT=$(git -C "$R" rev-parse HEAD); echo "EXPECTED_COMMIT=$EXPECTED_COMMIT"
```

## Stage A - finish the 0.6B record (~3 GPU-h, no code change)

### A1. Protocol eval for the r2 SFT arm (first, ~40 min)

`W8_EVAL_ARMS=r2sft` reads `results/wave8_r2sft_seed${SEED}` and needs `W8_R2_LR`, whose leading
`lr` is stripped and float-checked. `lr2e-4` is the value the three seeds ran (SubmitLine) and the
value `select-sft-lr` records, so pass the tag form.

```bash
cd $K
export W8_RUNG=0.6b W8_EVAL_ARMS="r2sft" W8_R2_LR=lr2e-4 W8_EVAL_SEEDS="0 1 2"
sbatch --export=ALL gpu-wave8-eval.sbatch
```

Produces, per seed: `results/eval/w8-r2sft-seed<S>-{step-argmax,step-sampled,noisy-argmax}.jsonl`
(+ `.summary.json`), `w8-r2sft-seed<S>-perm.json`, and `results/wave8/r2sft-seed<S>-S_nom.json`
(the per-letter argmax histogram + calibration that the WAVE8 §6.6 edge check needs, and which no
r2sft read has produced yet).

Expected: roughly the cost of the untrained control in job 405888 (3.1 s + 131 s + 129 s + ~2 min
perm + ~1 min collapse per arm set) x 3 seeds. Watch for `min_mass >= 0.9` (G7) - the r1 arms
failed G7 on `min_mass` at 4.0e-05 and 6.4e-03, so this is a real possibility here too.

### A2. R2+RL rounds (~2.4-3.6 GPU-h, expected to stop itself)

```bash
cd $K
export W8_ARM=r2 W8_R2_LR=lr2e-4 W8_R2_RL=1
sbatch --export=ALL --array=0-2 gpu-wave8-train.sbatch
```

**Why this is cheap and safe.** The `r2` branch runs the SFT stage first, but `decision_rounds sft`
always passes `--resume`, and the training loop is `for step in range(start_step, args.steps)` with
`start_step` = last completed step + 1 = 2827 and `args.steps` = 2827, so the loop body is empty. The
SFT stage costs one label load (~11.8 s for 95k rows) plus the sft-gates re-read, then the rounds
start. **Abort condition:** if the log prints new `"step": 0, "phase": "sft"` records, the resume did
not take; `scancel` the array immediately and report, because that would be 13.5 GPU-h of re-training.

**Expected outcome:** `STOPPED_collapse.json` in each arm dir after round 2. WAVE8 §6.5 stops a run
when the collapse gate trips at two consecutive round reads, and the SFT init already trips the
identical read (256 S_nom states, same frozen F3) at all 12 of its checkpoints. This turns the §6.6
row "if R2+RL also locks -> INCONCLUSIVE-negative at 0.6B" into a measured result instead of an
inference. Note `run_plan` writes the stop marker but never reads it, so **do not resubmit these
tasks** - a resume would continue past the stop (W8-R1-DRAINED.md §5).

## Stage B - ladder prerequisites (~1 GPU-h + a download)

### B1. Ladder weights (login node, has egress)

```bash
cd $R                                                                          # or the kit dir
bash hpc-umbrella/stage.sh --with-ladder
```

Downloads `Qwen/Qwen3-1.7B` @ `70d244cc...` and `Qwen/Qwen3-4B` @ `1cfa9a72...` into `$HF_HOME`.
Compute nodes are offline (`HF_HUB_OFFLINE=1`), so this must happen first. 4B is ~8.06 GB.

### B2. Per-rung pre-flight (mandatory; without it the train sbatch refuses to start)

```bash
cd $K
export W8_RUNG=1.7b
sbatch --export=ALL gpu-wave8-preflight.sbatch
```

`gpu-wave8-train.sbatch` refuses unless `RUNG_GATES` (= `results/wave8/gates_1.7b.json` for 1.7b)
holds F2 with `prompt_sha_equal` **and** a frozen F3. Both are rung-specific: F2 is the tokenizer /
chat-template prompt-SHA check and F3 is the untrained readout's entropy and `I` thresholds, and the
untrained 1.7B readout is not the 0.6B one. Walltime 1 h.

Verify before spending GPU:

```bash
python3 -c "import json;g=json.load(open('$R/results/wave8/gates_1.7b.json'));\
print(sorted(g), (g.get('F2') or {}).get('prompt_sha_equal'), 'F3' in g)"
```

## Stage C - ladder, R1 recipe at 1.7B (~5.5 GPU-h)

```bash
cd $K
export W8_RUNG=1.7b W8_ARM=r1 W8_LRS=5e-5 W8_SEEDS=3
sbatch --export=ALL --array=0-2 gpu-wave8-train.sbatch
```

One lr (`5e-5`, the 0.6B validation winner per W8-R1-DRAINED.md §4), three seeds, the R1 recipe
unchanged, WAVE8 §4.4. This is the submit that decides "capacity" versus "recipe": a null at 0.6B
plus a positive at 1.7B is the scaling result, and §4.4 licenses it now that the 0.6B run completed
with no infrastructure fault.

Cost basis: §9.3 estimates 4.7 s/step scaled by 1.7/0.6 = 13 s/step x 510 steps = 1.8 h/seed, so
~5.5 GPU-h for three. **Verify, do not trust it:** submit one task first (`--array=0`), read the
first 50 `wall` deltas from `train_log.jsonl`, and extrapolate before releasing the other two. The
0.6B SFT measured 5.2 s/step against the spec's 4.7 s/step basis, so the inference is optimistic by
~10 %.

Two-lr variant (6 tasks, ~11 GPU-h) only if the one-lr read comes back ambiguous. Add `1e-5` to
`W8_LRS` and use `--array=0-5`; task index maps to `lr = W8_LRS[idx / W8_SEEDS]`, seed = `idx % W8_SEEDS`.

## Stage D - HOLD: 1.7B R2-SFT, needs fixes first

Do not submit a 1.7B R2-SFT run on this tree. Two concrete defects, both verified in the kit:

1. **Arm-dir collision.** `gpu-wave8-train.sbatch` r2 branch sets
   `SFT_DIR="results/wave8_r2sft_seed${SEED}"` with no `${RUNG_SUFFIX}`, and the eval's r2sft case is
   `D="results/wave8_r2sft_seed${SEED}"` likewise. A 1.7B run would therefore read the 0.6B
   `state.json`, find `start_step` = 2827 >= `--steps`, no-op the loop, exit 0, and the eval would
   then report the **0.6B adapter under a 1.7B arm name**. The r2 sweep dir has the same problem
   (`results/wave8_r2sft_lrsweep_${TAG}`). Fix both sides together:
   `SFT_DIR="results/wave8_r2sft${RUNG_SUFFIX}_seed${SEED}"` in both files, and rename the sweep dir
   in a way that still matches the `wave8_r2sft_lrsweep_*` discovery glob in `wave8_analysis.py`.
2. **Walltime.** 2,827 steps at ~13 s/step is ~10.2 h against `#SBATCH --time=08:00:00`, so each seed
   needs two windows via the exit-75 + resubmit path (`--signal=B:TERM@600`, state.json written
   atomically). Three seeds at full epoch is ~31 GPU-h, which does not fit the remaining budget.

Budget position: WAVE8 §6.5 caps the wave at 70 GPU-h. Roughly spent so far: R1 six arms ~5-7, r2
sweep ~2, r2 SFT 13.5, pre-flight ~1, plus reads and evals. Check it before Stage C:

```bash
sacct -S 2026-10-01 -X -P -o JobName,ElapsedRaw | awk -F'|' '{s+=$2} END {printf "%.1f GPU-h since Oct 1\n", s/3600}'
```

If Stage C is ambiguous and the 1.7B R2-SFT is still wanted, the defensible reduced form is **one
seed, full epoch** (the epoch is the confound fix; a fractional epoch would reintroduce it), ~10
GPU-h over two windows, reported as a capacity probe, not a gate.

## Order

1. A1 eval (40 min, completes G1/G4/G6/G7 + the S_nom histogram for the r2sft arm).
2. A2 R2+RL rounds (expected stop at round 2; documents the §6.6 row).
3. B1 weights, B2 pre-flight at 1.7b.
4. C one-lr R1 ladder at 1.7B after the s/step check.
5. D only behind the two fixes above.
