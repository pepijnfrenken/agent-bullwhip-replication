# Wave-8 re-run proposal, 2026-11-19

Written after the archive check in `docs/verify/CLAIMS-AUDIT-2026-11-19.md`. The project scratch
root holds one file, `.keep`. The 4B and 1.7B per-run data no longer exists anywhere reachable, and
the paper's positive row cannot be substantiated from what survives.

This is a proposal. Nothing is submitted until Pino picks.

## What has to be rebuilt

| Rung | Evidence lost | Can it be rebuilt | Lane |
|---|---|---|---|
| 4B (2 seeds, lr 5e-5) | trained adapters, final-eval JSONL, perm JSON, gates_4b.json | yes, fully | Umbrella L4 |
| 1.7B (3 seeds, lr 1e-5) | same, plus gates_1.7b.json | yes, fully | Umbrella L4 |
| 0.6B wave-7 | intact, 328 tracked files | n/a | none |
| 0.6B wave-8 | logs + manifest only, already declared log-sourced | reproduce only if cheap | Umbrella L4 |
| 8B | trained adapters never persisted to the Modal volume, val-band means only | control eval yes, trained generation needs a retrain | Modal A100 |

## Steps, in order

1. Re-freeze the 1.7B and 4B gates with `hpc-umbrella/gpu-wave8-preflight.sbatch`, using the same
   state seeds as October. Prediction, registered before the run: the written values come back
   identical to `RECONCILIATION-2026-10-22.md` section 7, that is 1.7B H_lo 9.6527e-05, H_hi
   1.6313, I_hi0 1.4858, m_hi 0.1487 and 4B H_lo 0.038026, H_hi 1.910513, I_hi0 0.676551, m_hi
   0.1487. A mismatch means seed pinning drifted and is itself a finding.
2. Rebuild the label sets with `cpu-wave8-label.sbatch` (deterministic from demand seeds and the
   tuned floor). Prediction: byte-identical to the October label files, checked by checksum against
   any surviving copy in `results/wave7_labels/` if the seeds overlap.
3. Train: 1.7B lr 1e-5 seeds 0,1,2 and 4B lr 5e-5 seeds 0,1, three rounds of 170 steps. `sacct`
   records for October: 4B legs between 01:00 and 04:32 wall, 1.7B legs around 01:15 to 01:45.
4. Evaluate each trained arm plus its matched untrained control: noisy-argmax n=30, permutation
   grid, collapse-final read, per `gpu-wave8-eval.sbatch`. One evaluation job took 01:30 in October.
5. Pull back and commit after every leg with `hpc-umbrella/push_results.sh`. This is the step that
   was skipped in October and it is the reason this proposal exists.

## Cost

Zero euros on the Umbrella lane, which is up: `sinfo -p tue.gpu.q` reports L4 nodes mixed and up,
and this account's home is writable. Budget 15 to 25 L4 GPU-hours across 1.7B and 4B, so one to two
days of wall clock with queue waits. No Modal spend unless the 8B control is included, which would
add roughly $2-4 by the estimate already recorded in `WAVE8_WORD8B_RESULT.md` line 75.

## Stop and verdict rules, registered before the first submission

- Gate re-freeze mismatch on either rung: stop, report, do not train. No thresholds get adjusted
  after seeing data.
- 4B re-run: the frozen G1 criterion decides, unchanged. If both seeds clear it, the row stands as
  pass and the numbers become the new published values, which may differ from 8,315 and 7,612. If
  either seed fails, the paper's 4B row is rewritten to what the fresh data says and the paper ships
  as a methods and protocol contribution. Either outcome is a result.
- 1.7B re-run: the October verdict was adverse on all three seeds, and that verdict is already
  independent of the lost data being favourable. If the fresh run disagrees, the disagreement is
  reported as a reproducibility failure and the October row is withdrawn.
- No leg is reported from a validation band as if it were a final evaluation. The 8B mistake is not
  repeated.

## The retention fix, mandatory in the same mission

Every leg ends with `push_results.sh` and a commit, and a weekly `tar` of `results/` copied off the
cluster to this machine. Evidence that exists only on `/scratch-shared` has an unbounded expiry date
and a zero-day notice.
