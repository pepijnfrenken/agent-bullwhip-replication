# Results retention: what broke, and the rule now in force

Written 2026-11-19, during the wave-8 rebuild.

## What broke

The wave-8 1.7B and 4B legs were trained and evaluated in October. Their per-run JSONL was written
to `/scratch-shared/20234428/bullwhip/agent-bullwhip-replication/results/`, which per
`hpc-umbrella/config.sh` line 35 is deleted 14 days after the last write and is never backed up. No
`push_results.sh` run carried those files into git. The local machine, the local git history, the
cluster home directory, Snellius and the hub were all checked on 2026-11-19 and none of them held a
copy. Scratch held exactly one file, `.keep`.

Result: the paper's only positive row lost its evidence, and the row had to be rebuilt from
scratch. Details in `docs/verify/CLAIMS-AUDIT-2026-11-19.md`.

## The rule

Any run artifact that a sentence in the paper will depend on must exist in at least three places
before the next job is submitted:

1. the cluster working tree (where the job writes it),
2. git, pushed to `github.com:pepijnfrenken/agent-bullwhip-replication` (`push_results.sh`),
3. a local snapshot on the hub machine, checksummed (`scripts/results_guard.sh`).

Anything that exists only in 1 is already lost. The 14-day clock is the deadline, and it starts at
the last write, not the last read.

## The mechanism

`scripts/results_guard.sh` runs one cycle, or loops:

- `ssh` to the login node, copy the wave-8 Slurm logs into `results/wave8_logs/`, then run
  `hpc-umbrella/push_results.sh` with the weights excluded
  (`--exclude-glob 'results/**/*.safetensors'` and friends) so git takes readouts, not adapters.
- fetch the pushed commits into a dedicated mirror clone, `~/bullwhip-archive`, which is never
  edited. The working checkout is untouched, so paper edits cannot race the mirror.
- on a new HEAD, write `~/backups/bullwhip/bullwhip-<ts>-<sha>.tar.gz` plus a line in `SHA256SUMS`,
  and prune to the newest 8. Checkpoint directories are excluded from the tar.
- compare the cluster HEAD and eval file count against the mirror and log `DRIFT` at exit code 4
  when they disagree.

Bounds: `INTERVAL` 300 s, `MAX_HOURS` 36, stop file
`~/.hermes/cache/scratch/bullwhip/guard.stop`. Log: `~/.hermes/cache/scratch/bullwhip/guard.log`.

Durable schedule: cron job `426a66b96848`, `bullwhip_guard_cron.sh`, every 15 minutes, silent unless
a snapshot lands or drift appears. `--once` is the cron mode; `--status` prints state without
changing anything.

## Verification, done before the first job of the rebuild

- `bash scripts/results_guard.sh --once` cloned the mirror and wrote
  `bullwhip-20261119T110636Z-76a655b.tar.gz`, 69 MB, sha256 recorded in `SHA256SUMS`.
- `cronjob_manage list` shows job `426a66b96848` scheduled, next run 2026-11-19T11:22Z.
- `git ls-remote origin HEAD` matched the pushed commit, so channel 2 is live from both machines
  (the login node authenticates to GitHub as `pepijnfrenken`).

## Second copy on the cluster, deliberately

`/vast.mnt/home/20234428/agent-bullwhip-replication` survived the scratch purge because home is not
scratch and is not auto-deleted. It is a checkout at `21ad077` with 356 MB of results from the
wave-6/7 era. It is not a substitute for git and it is not a source for anything the paper claims,
but it is a fourth copy for the rebuild and it is why this failure was recoverable at all for the
earlier waves.

## Rules for the next run, in the operator checklist

- Submit one leg at a time, not an array whose later tasks nobody watches.
- After every leg, run `push_results.sh` from the cluster with the job id in the message. The guard
  does this every 15 minutes, but the operator does it once per leg so the commit names the job.
- A leg is not finished until its commit is visible in `git log` from a different machine.
- Before submitting anything that trains a new rung, run `results_guard.sh --status` and confirm the
  mirror head equals the cluster head.
