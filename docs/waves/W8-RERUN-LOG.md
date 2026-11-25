# Wave-8 rebuild log, starting 2026-11-19

Purpose: an append-only record of what was submitted, on which lane, with which commit, and what
was predicted before each read. Entries are added before the result is known; nothing is backfilled.

Approved: path A from `docs/waves/W8-RERUN-PROPOSAL.md` (rebuild 1.7B and 4B), Pino, 2026-11-19.

## Pre-registered predictions, written before the first submission

1. **Gate re-freeze.** The 1.7B and 4B F3 control statistics come back identical to
   `RECONCILIATION-2026-10-22.md` section 7, within print precision:
   - 1.7B: `H_lo` 9.6527e-05, `H_hi` 1.6313, `I_hi0` 1.4858, `m_hi` 0.1487
   - 4B: `H_lo` 0.038026, `H_hi` 1.910513, `I_hi0` 0.676551, `m_hi` 0.1487
   A mismatch means seed pinning drifted on the state sample or the readout, and the run stops.
2. **4B cost.** If the 4B verdict reproduces, the newly measured median will land within a few
   percent of the October figure (8,315 seed 0, 7,612 seed 1) but is NOT expected to match it
   exactly, because the RNG path through training is not seeded end to end. The verdict rule is the
   frozen G1 criterion, not agreement with the old number.
3. **1.7B direction.** The 1.7B rung remains adverse (median cost above its matched untrained
   control on every seed). If it flips, the October row is withdrawn and the disagreement is
   reported as a reproducibility failure.

## Environment, stated before the run

- Cluster checkout `edb3b547337c16928b285edda936d5a1b6ab340f`, pinned via `EXPECTED_COMMIT` on every
  submit.
- Venv rebuilt on the cluster: torch 2.1.2, transformers 4.55.4, peft 0.21.1 (the pinned stack).
- Weights resolved to the pinned revisions: 0.6B `c1899de2`, 1.7B `70d244cc`, 4B `1cfa9a72`. The
  `refs/main` pointers in the cache were written manually to those shas so that code paths asking
  for `main` cannot silently resolve elsewhere.
- Known-red suite at this commit, recorded rather than ignored: 5 failed / 2 errors out of the
  torch subset. `test_eval_mass_assert_only_for_untrained` fails because commit `2009f20`
  (WAVE8-A1, 2026-10-19) deliberately changed the eval CLI from assert to record for both branches;
  the test was never updated. The remaining four `TestSftTorch` failures and two errors are under
  investigation; none of them is on the train or eval CLI path that the runs use.
- Retention: `scripts/results_guard.sh` verified before the first submit; cron job `426a66b96848`
  every 15 minutes; first snapshot `bullwhip-20261119T110636Z-76a655b.tar.gz`, 69 MB, sha256 in
  `~/backups/bullwhip/SHA256SUMS`. See `docs/waves/RESULTS-RETENTION.md`.

## Submissions

| UTC time | job | rung | what | state |
|---|---|---|---|---|
| 2026-11-19T12:33Z | 764195 | 1.7b | `W8_RUNG=1.7b gpu-wave8-preflight.sbatch` (F2+F3 freeze) | submitted |
| 2026-11-19T12:33Z | 764196 | 4b | `W8_RUNG=4b gpu-wave8-preflight.sbatch` (F2+F3 freeze) | submitted |

## Reads

(appended as artifacts are read; each entry names the file and the field)

## Dependency chain submitted (2026-11-19T12:52Z)

The L4 lane is saturated (the only 8-GPU L4 node reports `AllocTRES=gres/gpu=8`; Slurm estimated
2026-11-20T13:24Z for a start, and the VPN profile expires 2026-11-20T02:55Z, both before that).
So the pipeline is expressed as Slurm dependencies and needs nothing from this side:

| job | name | depends on | command |
|---|---|---|---|
| 764195 | bw8-preflight 1.7b | - | `W8_RUNG=1.7b gpu-wave8-preflight.sbatch` |
| 764196 | bw8-preflight 4b | - | `W8_RUNG=4b gpu-wave8-preflight.sbatch` |
| 764263 | bw8-gatecheck | afterok:764195:764196 | `cpu-wave8-gatecheck.sbatch` |
| 764264 | bw8-train 1.7b (3 tasks) | afterok:764263 | `W8_RUNG=1.7b W8_LRS=1e-5 W8_SEEDS=3 --array=0-2 gpu-wave8-train.sbatch` |
| 764265 | bw8-train 4b (2 tasks) | afterok:764263 | `W8_RUNG=4b W8_LRS=5e-5 W8_SEEDS=2 W8_MICRO_BATCH=8 --array=0-1 gpu-wave8-train.sbatch` |
| 764269 | bw8-eval 1.7b | afterok:764264 | `W8_RUNG=1.7b W8_EVAL_SEEDS="0 1 2" W8_LR=1e-5 gpu-wave8-eval.sbatch` |
| 764270 | bw8-eval 4b | afterok:764265 | `W8_RUNG=4b W8_EVAL_SEEDS="0 1" W8_LR=5e-5 gpu-wave8-eval.sbatch` |

`EXPECTED_COMMIT=6b2fe7dc93d36b4519b57796dfa25e27311ed29d` on every submit.

`cpu-wave8-gatecheck.sbatch` runs `scripts/verify_rung_gates.py`, which compares the freshly frozen
`gates_1.7b.json` and `gates_4b.json` against the predictions registered at the top of this file at
1e-3 relative tolerance and exits non-zero on any deviation. Because training depends `afterok` on
it, a drifted gate cancels the training jobs instead of letting them run. The verifier was tested
both ways before submission: a hand-constructed matching pair returns `PREDICTION HOLDS` (rc 0) and
a 5% perturbation of `H_lo` returns `MISMATCH — do not train` (rc 1).

Watcher: cron job `9f0ebfccfb0d` every 20 minutes, silent unless a state changes or a job fails.
Retention guard: cron job `426a66b96848` every 15 minutes.

## Staging defects found and fixed (2026-11-19)

Staging took four attempts. Three failures were kit defects, not user error, and they are now
fixed in `hpc-umbrella/stage.sh` rather than worked around by hand:

1. **Repo detection.** `if [ -d "${REPO_DIR}/.git" ]` was true on a purged directory shell, so
   staging took the "update" branch and died at `remote set-url` with "not a git repository"
   (attempts 1 and 2, exit 1 and 128). Now it asks git: `git -C "${REPO_DIR}" rev-parse --git-dir`,
   and clears the shell before cloning when that fails.
2. **Venv detection.** `if [ ! -d "${VENV_DIR}" ]` was false on the empty `venv/` left by the purge,
   so staging said "venv exists, reusing" and died on the missing `bin/activate` (attempt 3,
   exit 1). Now the test is `[ ! -x "${VENV_DIR}/bin/activate" ]`.
3. **Offline flag and interpreter for the weight download.** `bw_load_env` exports
   `HF_HUB_OFFLINE=1` for compute nodes, and the staging download inherited it, so
   `snapshot_download` refused with "couldn't connect" while `curl https://huggingface.co` returned
   200 from the same login node (attempt 5, exit 127 after the venv-python error). The download also
   ran outside the venv, so it depended on the system python carrying `huggingface_hub`. It is now
   the `bw_download` helper: subshell, venv activated, `HF_HUB_OFFLINE=0`, repository id and
   revision passed as argv.

Verified after the fix: `bash -n` clean, `stage.sh --dry-run --with-ladder` prints the three
`bw_download` calls with the pinned revisions, and a real re-stage on the login node reuses the
existing venv and repo instead of destroying them.

Weight revisions now in the cache, each with a manual `refs/main` pointing at the pin so no code
path can resolve "main" elsewhere: 0.6B `c1899de2`, 1.7B `70d244cc`, 4B `1cfa9a72` (13 GB total
under `/scratch-shared/<cluster-account>/bullwhip/hf`).

## First attempt failed: the rung never reached the job (2026-11-19T14:48Z)

Raised by Pino ("some of the waves ran"). Two jobs ran, neither was the rung that was ordered.

**764195 (ordered 1.7b) COMPLETED, but ran as 0.6b.** Its first log line is
`repo ... HEAD=9db73fe rung=0.6b model=Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca`.
It therefore re-froze the 0.6B control statistics and wrote `results/wave8/gates.json`, not
`gates_1.7b.json`. Values written: `H_lo` 1.3933294132789242, `H_hi` 2.021894960269872, `I_hi0`
0.09699194743853656, `m_hi` 0.14874833921755284. `git status` is clean and `git diff
results/wave8/gates.json` is empty, so the rewrite is byte-identical to the committed file. Two facts
follow: the accidental run did no damage, and the 0.6B F3 freeze reproduces exactly across a month
and a rebuilt environment.

**764196 (ordered 4b) FAILED, exit 1.** It ran as 0.6b too, found the F2 block already frozen in
`gates.json`, and refused: `F2 already set — frozen before training; pass --force to overwrite`. The
kit's own protection stopped it from overwriting a frozen gate.

**Consequence:** gatecheck 764263 inherited `DependencyNeverSatisfied`, and 764264, 764265, 764269,
764270 could never start. All five were cancelled with `scancel` at 2026-11-19T15:0xZ.

**Cause, established by a two-way probe, not by inference.** The documented invocation
`W8_RUNG=1.7b sbatch gpu-wave8-preflight.sbatch` does not put `W8_RUNG` in the job environment on
this cluster; `sbatch --export=ALL,W8_RUNG=...` does. Probe: `envprobe2`, two CPU jobs reading
`${W8_RUNG:-UNSET}`. With `--export=ALL,W8_RUNG=probeA,PROBE_X=yes` the job printed
`W8_RUNG=[probeA] extra=[yes]`; with the prefix form it printed `W8_RUNG=[UNSET] extra=[UNSET]`.
(An earlier probe was invalid: its heredoc expanded `${W8_RUNG:-UNSET}` on the login node, so the
script contained the literal word. Recorded because a broken probe is not evidence.)

**Fixes, committed:**
- `hpc-umbrella/w8-submit.sh` builds `--export=ALL,...` for every stage, refuses to submit without an
  explicit rung, exports `EXPECTED_COMMIT`, and takes seeds colon-separated (`--seeds 0:1`) so they
  survive `--export`.
- `gpu-wave8-preflight.sbatch`, `gpu-wave8-train.sbatch`, `gpu-wave8-eval.sbatch` now print a loud
  warning when `W8_RUNG` is absent, instead of looking like a deliberate 0.6b run.
- `gpu-wave8-eval.sbatch` accepts a colon-separated `W8_EVAL_SEEDS`.

## Resubmitted chain (2026-11-19T15:0xZ)

`EXPECTED_COMMIT=3e1be14d85f262846d465d7be3136cbae988eaa9`, all knobs via `--export=ALL`.

| job | name | depends on |
|---|---|---|
| 766006 | bw8-preflight 1.7b | - (RUNNING on tue-gpuc001; log line `rung=1.7b model=Qwen/Qwen3-1.7B@70d244cc`) |
| 766007 | bw8-preflight 4b | - |
| 766014 | bw8-gatecheck | afterok:766006:766007 |
| 766015 | bw8-train 1.7b (0-2) | afterok:766014 |
| 766016 | bw8-train 4b (0-1) | afterok:766014 |
| 766017 | bw8-eval 1.7b | afterok:766015 |
| 766018 | bw8-eval 4b | afterok:766016 |

The lesson for the paper's provenance section: a submit form that silently degrades to the default
rung would have produced four weeks of "ladder" data at one model size. It did not, because the first
log line names the rung and the gate files are per-rung.

## Prediction 1 checked: the 1.7B gate reproduces exactly (2026-11-19)

`766006` completed (1m27s, exit 0) and wrote `results/wave8/gates_1.7b.json` (1495 bytes) for
`Qwen/Qwen3-1.7B@70d244cc`. Read from the file, field by field:

| field | registered (RECONCILIATION §7) | read from file | rel diff |
|---|---|---|---|
| `H_lo` | 9.6527e-05 | 9.65265603466344e-05 | 4.6e-06 |
| `H_hi` | 1.6313 | 1.6312969325077367 | 1.9e-06 |
| `I_hi0` | 1.4858 | 1.4857733402975195 | 1.8e-05 |
| `m_hi` | 0.1487 | 0.14874833921755284 | 3.3e-04 |

`scripts/verify_rung_gates.py --gates results/wave8/gates_1.7b.json --rung 1.7b` returns
`RESULT: PREDICTION HOLDS` (rc 0). `F2` reports `contiguous True`, `prompt_sha_equal True`.
Also read but not registered as a prediction: `m_star` 0.126953125, `H_median_untrained`
0.6148521091212052.

Meaning: the F3 control statistics for 1.7B are reproducible across a month, a rebuilt venv and a
freshly downloaded checkpoint. The October gate values were not an artefact of that session, and the
state sample plus readout are deterministic given the pins.

## Second failure and fix: the pin moved under the queued jobs (2026-11-19T15:2xZ)

`766007` (4b preflight) failed in 5 seconds:
`FATAL: EXPECTED_COMMIT=3e1be14... but repo HEAD=73a2bbc... — refusing to run stale code`.
`73a2bbc` was the retention guard's own `auto-mirror` commit, pushed to the same branch. So the guard
that protects the data was invalidating the pin that protects the protocol. My design conflict.

Fixed by narrowing the gate: the pinned commit must be an ancestor of HEAD, and every path changed
since it must sit under `results/` or `docs/`. Any other change still refuses. Applied to
`gpu-wave8-preflight.sbatch`, `gpu-wave8-train.sbatch`, `gpu-wave8-eval.sbatch`.

## Chain in flight (2026-11-19T15:3xZ), HEAD `ff88dc7`

| job | name | depends on | state at submit |
|---|---|---|---|
| 766006 | bw8-preflight 1.7b | - | COMPLETED (gate file read above) |
| 766026 | bw8-preflight 4b | - | PENDING (Resources) |
| 766027 | bw8-gatecheck | afterok:766006:766026 | PENDING (Dependency) |
| 766028 | bw8-train 1.7b (0-2) | afterok:766027 | PENDING (Dependency) |
| 766029 | bw8-train 4b (0-1) | afterok:766027 | PENDING (Dependency) |
| 766030 | bw8-eval 1.7b | afterok:766028 | PENDING (Dependency) |
| 766031 | bw8-eval 4b | afterok:766029 | PENDING (Dependency) |

Superseded and cancelled: 766014, 766015, 766016, 766017, 766018.

## Prediction 1 confirmed on both rungs; gatecheck passed (2026-11-19T16:2xZ)

`766026` (4b preflight) COMPLETED, 2m41s, exit 0, writing `results/wave8/gates_4b.json` for
`Qwen/Qwen3-4B@1cfa9a72`. `766027` (gatecheck) then COMPLETED in 4 seconds, exit 0, with
`overall ok: True` across all 8 checks in `results/wave8/gate_verification.json`:

| rung | field | registered | read from file | rel diff |
|---|---|---|---|---|
| 1.7b | H_lo | 9.6527e-05 | 9.65265603466344e-05 | 4.6e-06 |
| 1.7b | H_hi | 1.6313 | 1.6312969325077367 | 1.9e-06 |
| 1.7b | I_hi0 | 1.4858 | 1.4857733402975195 | 1.8e-05 |
| 1.7b | m_hi | 0.1487 | 0.14874833921755284 | 3.3e-04 |
| 4b | H_lo | 0.038026 | 0.038025627538467104 | 9.8e-06 |
| 4b | H_hi | 1.910513 | 1.9105125259947648 | 2.5e-07 |
| 4b | I_hi0 | 0.676551 | 0.6765505195974482 | 7.1e-07 |
| 4b | m_hi | 0.1487 | 0.14874833921755284 | 3.3e-04 |

Additional reads from `gates_4b.json`, not registered as predictions: `m_star` 0.126953125,
`H_median_untrained` 1.4007411841055655, untrained readout `min_mass` 0.997614860534668.

What this buys the rebuild: both arrowheads of the protocol, the state sample and the readout, are
deterministic across a month, a rebuilt venv and freshly downloaded checkpoints. The October gate
values were real, so the verdicts they gated are not suspect on instrument grounds.

Training is now queued and this is the last gate before numbers exist: `766028` (1.7b, 3 tasks) and
`766029` (4b, 2 tasks) are PENDING on Resources, then `766030` and `766031` eval.
