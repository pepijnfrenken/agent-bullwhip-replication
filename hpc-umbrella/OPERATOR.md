# OPERATOR — bullwhip Wave-6 GRPO on TU/e Umbrella

Everything you run, in order, with the expected output at each step and what to do when it
breaks. Pins live in **`config.sh`** (one file); `config.sh --check` fails if anything in it is
unresolvable on the cluster. Facts behind the pins: `00-PROBE-AND-SETUP.md`.

**Prerequisites:** TU/e account, **VPN** (hpc.tue.nl is campus/VPN only), a clone of this repo
(the run kit lives at `hpc-umbrella/` inside it), and a hub session open for paste-back. `rsync` +
`ssh` are needed only for `pull_results.sh` and the no-egress fallback.

**Submit rule (learned the hard way):** always `cd` into the kit dir before `sbatch` — Slurm
executes batch scripts from its own spool copy (`/var/spool/slurmd/jobNNN/slurm_script`), so the
scripts locate `config.sh` via `SLURM_SUBMIT_DIR`, not via their own path. Submitting from
elsewhere now fails loudly (`FATAL: config.sh not found`) instead of a mystery 2-second exit.

**Where things live (all on scratch, all disposable):**

| Path | What |
|---|---|
| `/scratch-shared/$USER/bullwhip/kit` | this kit (staged copy) — submit sbatch from here |
| `/scratch-shared/$USER/bullwhip/agent-bullwhip-replication` | the repo clone (results land inside it) |
| `/scratch-shared/$USER/bullwhip/{hf,pipcache,torchinductor,venv,tmp}` | caches, venv, scratch |
| `/scratch-shared/$USER/bullwhip/logs` | Slurm job logs |

Scratch is **deleted 14 days after the last write**; nothing is backed up. Pull early, pull often.

---

## Step 0 — get the kit onto the cluster (git clone)

The kit lives at `hpc-umbrella/` **inside this repo** — one clone brings code + kit, and the repo
is **public**, so no credentials are needed on the cluster. On the login node:

```bash
git clone https://github.com/pepijnfrenken/agent-bullwhip-replication.git ~/bullwhip-repo
cd ~/bullwhip-repo/hpc-umbrella
```

Refresh later from the same checkout: `cd ~/bullwhip-repo && git pull --ff-only`.

**No login-node egress?** (probe §8 tells you): upload from the laptop instead — `rsync -av` the
kit and a repo checkout, or drag/drop in Open OnDemand (one connection at a time — SCC asks; no
parallel scp sessions) — then run everything from that upload directory.

## Step 1 — probe (login node, ~1 min)

```bash
bash probe.sh          # from the kit dir (Step 0 leaves you in it)
```

Expected: a readable report in the terminal **and** at `~/umbrella-probe-<date>.txt`.

Then **paste the whole file back to the hub session**. The hub reconciles exactly the pins marked
`assumed` in `config.sh --pins`:

| Pin | Where in the probe | What it settles |
|---|---|---|
| `MODULE_UMBRELLA`, `MODULE_PYTORCH`, `MODULE_PYTHON` | §6 `module avail` / `module spider PyTorch` | exact module names (the 2.1.2-CUDA-12.1.1 guess) |
| `CPU_PARTITION`, `GPU_PARTITION` (+`_ALT`) | §1, §4 | partition names actually offered |
| `L4_CONSTRAINT` | §2, §3 (the feature token on L4 nodes) | the `--constraint` value; also needed on the sbatch headers |
| `WALLTIME_PILOT`, `WALLTIME_CAMPAIGN` | §5 `MaxTime` (**not** the `sinfo` default) | whether the 18 h campaign fits or must be split |
| scratch path + quota | §7 `myquota` | path sanity for the 8 TiB scratch |
| login-node egress yes/no | §8 | git clone + HF download here, or upload from the laptop |
| `GPU_PARTITION_AVOID` sanity | §1, §4 | confirm gpuA is the A30 MIG pool (never target it) |

**Resolved automatically:** the L4 feature name and the partition that owns L4 nodes are
auto-detected from `sinfo` in `config.sh`, so a wrong `assumed` pin there is often corrected by
itself — `config.sh --check` prints what it resolved and fails if nothing matches.

**Needs your one-file edit:** module names (`MODULE_*`), the walltime pins if the partition caps
below 18 h, and — only if the probe shows something unexpected — the partition names. Then edit
`config.sh` **and the matching `#SBATCH` header lines** (Slurm cannot interpolate shell variables;
`config.sh --check` fails on any header/pin mismatch, so you cannot forget).

## Step 2 — stage (login node, ~10 min)

```bash
bash stage.sh            # idempotent; --dry-run to preview (from the kit dir)
bash config.sh --check                        # must end "RESULT: 0 fail / N warn"
```

What it does: pools the kit to scratch, clones/updates the repo, builds the venv, installs the
requirements track, downloads `Qwen/Qwen3-0.6B`, then runs the checks.

Expected tail: `config.sh --check` with `0 fail`; pytest green (one hub-only assertion is
deselected on a torch-bearing host); a printed `torch / transformers / peft / cuda` line.

**Two environment tracks** (pins in `requirements-umbrella.txt`, selected by `ENV_MODE`):

- **A — `ENV_MODE=module` (recommended).** Umbrella's PyTorch 2.1.2 + a `--system-site-packages`
  venv with a matched-era transformers/peft. No multi-GB downloads, CUDA matched to the node
  driver, and it is the *control arm* for the parked stall (which is suspected to be a
  transformers-5.x-on-torch-2.5.1 mismatch).
- **B — `ENV_MODE=pip`.** The desktop-proven matrix verbatim (torch 2.5.1+cu121, transformers
  5.17.0, peft 0.21.0, trl 1.13.0). The *treatment arm*: if the stall reproduces here and not on
  A, the version pairing is confirmed. Switch with `ENV_MODE=pip bash stage.sh`.

If `peft` lacks `disable_adapter()`, the KL reference in `grpo_step` raises immediately with a
named error — install `peft>=0.11` (track A already requires it).

## Step 3 — CPU harness (free, immediate)

```bash
cd /scratch-shared/$USER/bullwhip/kit && sbatch cpu-harness.sbatch && squeue --me
```

Expected (`logs/cpu-harness_<job>.log`): pytest all green; walk-forward floors printing the
anchor **3 681.0** on step demand; then a torch-free dry-run table

```
rollout_batch= 1  generate_calls=144 for 3 steps
rollout_batch=16  generate_calls= 18
rollout_batch= 0  generate_calls= 18
```

That 144 → 18 is the batching lever visible on a CPU node. It also proves the checkpoint/resume
and SIGTERM paths run on the cluster interpreter.

## Step 4 — GPU pilot (the unblock + the Snellius numbers)

```bash
sbatch gpu-pilot.sbatch            # ~2 h walltime; check the queue with: squeue --me
tail -f /scratch-shared/$USER/bullwhip/logs/gpu-pilot_<job>.log
```

Three parts, each useful on its own:

- **A — minimal optimizer step.** Prints two losses, `A_seconds`, `A_PEAK_VRAM_GB`. This is the
  step the desktop never completed; if it finishes here, the `TORCHDYNAMO_DISABLE=1` + no-compile
  path is what unblocked it.
- **B — `bench_rollout.py`** → `results/grpo_bench/bench_<job>.json`: per-step wall-clock, µs per
  `generate()` call and per decision, tokens/s, peak VRAM for `rollout_batch ∈ {1,4,16,32}`, plus
  `speedup_vs_baseline` (1 = sequential). Also printed to the log, and `cat`-ed by the sbatch.
- **C — a 4-step GRPO pilot** (`results/grpo_a0_pilot_<job>/`): `train_log.jsonl`,
  `checkpoint/`, `state.json`. `--save-every 1` so a kill costs at most one step.

**How to read the metrics file (this decides Snellius):**

```bash
python3 - <<'PY'
import json; d = json.load(open("results/grpo_bench/bench_<job>.json"))
print(d["gpu"], d["torch"], d["transformers"], d["notes"])
for r in d["runs"]:
    print(r["rollout_batch"], r["step_wall_s"], r["us_per_decision"], r["tokens_per_s"], r["peak_vram_gb"])
print("speedup vs sequential:", d["speedup_vs_baseline"])
PY
```

Decision arithmetic (13 configs × 300 steps = 3 900 steps):

| If `step_wall_s` at the chosen batch is… | Then |
|---|---|
| ≤ 200 s | 300 steps ≈ 17 h → **the 18 h array fits**, submit as-is |
| 200–600 s | split: `STEPS_CAMPAIGN=150`, two submissions, `--resume` continues |
| > 600 s, or `peak_vram_gb` near the L4's 24 GB | **escalate to Snellius** — take `speedup_vs_baseline`, `tokens_per_s`, `peak_vram_gb` and the per-step wall-clock with you |

Also record: does the batched arm actually beat sequential (`speedup_vs_baseline["16"] > 1`), and
is VRAM under control. If batching gives no speedup, the bottleneck is not the rollout dispatch —
say so rather than assuming batching worked.

## Step 5 — campaign array

```bash
cd /scratch-shared/$USER/bullwhip/kit
sbatch --array=0-12 gpu-campaign.sbatch        # 13 pre-registered configs from campaign.tsv
squeue --me ; sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS
```

One task = one config (`results/grpo_<name>/`). To A/B the batching knob, run a second array with
`ROLLOUT_BATCH=1` (edit the pin, re-`--check`, resubmit) — that is the cheap ablation.

Walltime: each run checkpoints every 25 steps and resumes, so a kill is not lost work. Resubmit the
same command to continue. `logs/campaign_<arrayjob>_<task>.log` per task.

## Step 6 — pull results (laptop/hub, VPN)

```bash
bash pull_results.sh --dry-run        # see what would come over
bash pull_results.sh --dest ~/umbrella-results
```

Pulls `results/`, `logs/` and the bench JSONs into `umbrella-results/`. Do this after the pilot and
after each campaign wave: scratch expires 14 days after the last write, with no backup.

**Paste back to the hub after each step:** the probe file (step 1), the `config.sh --check` tail +
`ls` of the venv (step 2), the cpu-harness tail (step 3), the bench JSON + the pilot tail (step 4),
`sacct` + the per-config `train_log.jsonl` tails (step 5).

---

## Failure playbook

| Symptom | Where it shows | What to do |
|---|---|---|
| Job sits in `PD` (queue wait) | `squeue --me`, `sinfo -p <part>` | fairshare; nothing to fix. Run the CPU harness meanwhile; lower `--time` and `--mem` if you over-asked (they lower priority) |
| `module load … failed` / unknown module | job log line 1-5 | `module spider PyTorch`; fix `MODULE_*` in `config.sh` **and the sbatch headers**, then `config.sh --check` |
| `invalid constraint` / job rejected | sbatch error at submit | the L4 feature name is wrong: take it from `sinfo -N -o "%20N %60f"` (probe §2/§3), update `L4_CONSTRAINT` + headers, `--check` |
| Job lands on an A30 MIG slice (≈6 GB) | `nvidia-smi` in the log | you are on `tue.gpu1.q`; `--partition`/`--constraint` mismatch — fix the pins and resubmit |
| CUDA init failure / driver mismatch | step A of the pilot | `ENV_MODE=module` (module torch is built for this cluster) instead of the pip torch wheel |
| **Compile stall** (GPU 100 %, ~80 W, no output for minutes, `compile_worker` processes) | `squeue` shows running, log silent; `ps` shows `compile_worker` | already guarded: `TORCHDYNAMO_DISABLE=1` is set by the sbatch env and by `harden_compile_path()` before `torch` import. If it still stalls, that is a *result*: capture `py-spy dump --pid <pid>` and the resolved knob line (`compile-path knobs:` in the log) and stop — the version pairing or a newer transformers compile path is the cause |
| OOM (CUDA out of memory) | log traceback | lower `--rollout-batch` (16 → 4 → 1) and/or `--group`; 24 GB on an L4 is plenty at 0.6B, so OOM means something else (a second process on the same GPU) |
| Walltime kill | last log line `{"event": "sigterm", "step": N}` | normal: resubmit the same command; `--resume` continues from `state.json`. If the last line is *not* a sigterm record, check `logs/` for OOM / node failure |
| `OSError: no internet` / HF download fails on a compute node | pilot log | expected: compute nodes have no egress. Weights and pip must be fetched in `stage.sh` on the login node; the job runs offline from `$HF_HOME` |
| `myquota` near the limit / `Disk quota exceeded` | job log or `myquota` | scratch is 8 TiB but expires; clean `$BULLWHIP_ROOT/hf` or `pipcache`, or `pull_results.sh` then delete old `results/` |
| `peft lacks disable_adapter()` | traceback in step C | install `peft>=0.11` in the venv (`pip install 'peft>=0.11'`) |
| `state.json` exists but the adapter does not | `--resume` prints "step/RNG only" | expected when the previous run had no torch save (dry runs); a real run always writes both. Delete `results/grpo_<name>/` to restart a config cleanly |

## What the numbers are for

The pilot exists to answer one question with evidence: **can 13 × 300-step 0.6B GRPO runs be done
on Umbrella for €0, or must the 4B/Snellius escalation happen first?** The answer is
`bench_<job>.json` (per-step wall-clock, µs per `generate()`, batched-vs-sequential speedup,
tokens/s, peak VRAM) plus the pilot's `train_log.jsonl`. Nothing else in this kit is allowed to be
the reason for that decision.
