# Wave-8 operator block v2 (2026-10-14) — A2 + ladder, git-pull first

Companion to `W8-NEXT-SUBMITS.md` (its Stage A1 note is obsolete: **A1 already ran** — job 418889
read in `W8-R2-SFT-VERDICT.md` §10, arm is 1.5x to 2.9x WORSE than its own untrained readout,
G1/G2/G4 fail on all three seeds, G7 `min_mass` 0.466 to 0.629 vs a 0.9 bar. Do not resubmit it).
This block is for order and verification conventions only; the submit commands live in the script
below. The spec, the gates and the 70 GPU-h cap are unchanged.

## The one command to paste (cluster login node)

```bash
bash /scratch-shared/<cluster-account>/bullwhip/agent-bullwhip-replication/hpc-umbrella/wave8-next.sh
```

(The cluster user is hardcoded — `<cluster-account>`; `$TUE_USER` is not set in a plain login shell, which
is why the export form failed. `BULLWHIP_ROOT` still overrides if you ever need to point the
script elsewhere.)

The script is idempotent: each stage is skipped when its evidence file already exists on the
cluster, so pasting it twice never double-spends. It exits (status 1) at the first failed gate
instead of submitting further, and every sbatch it makes is behind a hard `sacct` GPU-h read
(stop by 60 of the 70 GPU-h cap). The A2 guard makes the `resume-did-not-take` hang-up stop the
script by hand instead of silently burning 13.5 GPU-h.

## Stages, in the script and in order

1. `EXPECTED_COMMIT` is pinned **from the cluster HEAD after `git pull --ff-only`** on `master`
   (never pasted from the hub; W8-NEXT-SUBMITS §0). The script then verifies local master == origin/master.
2. `stage.sh` runs from `$R/hpc-umbrella`, never from inside `$B/kit` (stage.sh replaces the kit
   dir). `config.sh --check` and a `kit == repo` diff gate follow.
3. **Stage A2 guard:** greps the newest `wave8-train_*.log` for new `"step": 0, "phase": "sft"`
   records. Any hit stops the script with the `squeue`/`sacct` commands to run by hand — that
   signature with a seed state past 2,827 is the 13.5 GPU-h SFT-retrain failure, and the array
   must be `scancel`-ed before anything else is submitted.
4. **Stage A2:** `W8_ARM=r2 W8_R2_LR=lr2e-4 W8_R2_RL=1 --array=0-2` (skipped when
   `wave8_r2rl_seed*/round_state.json` exists). Expected per seed: `STOPPED_collapse.json` by
   round 2 (~1.4 GPU-h total for the class); a run without the stop marker is the interesting
   result. Never resubmit a task after the stop marker (W8-R1-DRAINED §5: a resume continues past it).
5. **Stage B:** `stage.sh --with-ladder` (weights are downloaded on the login node only), then the
   per-rung pre-flight `W8_RUNG=1.7b sbatch gpu-wave8-preflight.sbatch`. The script exits right
   after with "NEXT: rerun this script once the pre-flight COMPLETES". On rerun it verifies
   `results/wave8/gates_1.7b.json` has `F2.prompt_sha_equal true` and `F3` frozen before going on,
   and dies if the file exists but fails those two.
6. **Stage C:** ladder R1 at 1.7B, `W8_LRS=5e-5 W8_SEEDS=3`, one task first (`--array=0`), with the
   50-`wall`-delta extrapolation inline. The script exits; a rerun (after task 0 has ~50 steps)
   submits `--array=0-2` if the extrapolated per-seed wall is inside the 8 h window and the
   GPU-h budget read is fine.

## Pull-side, after the jobs drain (hub)

```bash
cd ~/projects/agent-bullwhip-replication && git pull --ff-only
mkdir -p results/wave8_logs; cp /scratch-shared/<cluster-account>/bullwhip/logs/wave8-*.log results/wave8_logs/ # cluster side, then:
bash hpc-umbrella/push_results.sh -m "wave8: A2 + ladder submits (idempotent wave8-next.sh)" \
     results/wave8_r2rl_seed* results/wave8_logs results/wave8/gates_1.7b.json
python3 hpc-umbrella/verify_pull.py ~/projects/agent-bullwhip-replication --size
```

The reads I make on pull (no interpretation needed from your side): `STOPPED_collapse.json`
markers, round-2 collapse rows in `train_log.jsonl`, the 1.7B pre-flight F2/F3, and the first-50
timing. Ping me after pulling: the next-step verdict (ladder ambiguous-armline vs INCONCLUSIVE) is
mine to compute, not yours.

## Deviations from W8-NEXT-SUBMITS.md that this block pins

* **A1 removed from the checklist** (already ran, verdict negative).
* The A2 guard has its own stop; in that sheet the check was a note, not a blocker.
* The 1.7B ladder run keeps the one-lr shape (`W8_LRS=5e-5` only); the two-lr variant is gated
  behind "one-lr read comes back ambiguous", which is a hub-side read (mine), not a login-node one.
* Stage D (1.7B R2-SFT) stays HOLD, unchanged: the arm-dir collision
  (`SFT_DIR` with no `${RUNG_SUFFIX}` in both the train and eval sbatch) and the walltime math are
  still unresolved on this tree.
