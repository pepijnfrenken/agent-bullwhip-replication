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
