# Wave-8 results pull, step by step (v2 — first run after A2 + ladder submits)

**Run everything from the cluster login node.** You are `20234428`; the repo is
`/scratch-shared/20234428/bullwhip/agent-bullwhip-replication` (= `$R` below). Do NOT run
`wave8-next.sh` again unless I say so: it submits jobs. Right now we only ship back what the
jobs have produced.

```bash
R=/scratch-shared/20234428/bullwhip/agent-bullwhip-replication
B=/scratch-shared/20234428/bullwhip
cd "$R"
```

## 1. Check what exists (read-only)

```bash
squeue --me                                    # is job 429581 (A2) still running?
sacct -j 429581 -X -o JobID,State,Elapsed,End
echo "--- A2 stop markers (STOPPED_collapse.json = the expected outcome)"
ls -la results/wave8_r2rl_seed*/STOPPED_collapse.json 2>/dev/null || echo "none yet"
ls results/wave8_r2rl_seed*/
echo "--- anything wave-8 submitted today (preflight/eval/train)"
sacct -S 2026-10-14 -X -P -o JobID,JobName%24,State,Elapsed | grep -E 'bw8|wave8' || echo none
```

If a job is still RUNNING: wait for it to finish before pushing (a partial push shows as
INCOMPLETE in verify_pull and just gets re-read next cycle). If everything is COMPLETED, continue.

## 2. Copy the Slurm logs into the repo (results/ does not see them by itself)

```bash
mkdir -p results/wave8_logs
cp -v "$B"/logs/wave8-*.log results/wave8_logs/ 2>/dev/null | tail -20
ls results/wave8_logs/
```

## 3. Push over git (the transfer channel, operator sheet section 4)

```bash
cd "$R"
git status --short                             # should be clean apart from results/ noise
bash hpc-umbrella/push_results.sh -m "wave8: A2 R2+RL results (job 429581)" \
     results/wave8_r2rl_seed* results/wave8_logs || true
# Only if the 1.7B preflight already produced its gates file (stage B2):
bash hpc-umbrella/push_results.sh -m "wave8: 1.7B preflight gates (F2/F3)" \
     results/wave8/gates_1.7b.json 2>/dev/null \
     || echo "(no preflight gates yet - fine, that stage runs after B1 downloads weights)"
```

## 4. Confirm it is on origin (do not skip)

```bash
git log --oneline -3
git status -sb | head -1                       # must NOT say 'ahead'
```

Paste me back the full output of steps 1-4. On the hub I then run the gate reads myself:
verify_pull.py, the STOPPED_collapse markers, the round-2 collapse rows in train_log.jsonl,
and (if present) the 1.7B F2/F3 read. The verdict and the next submit are mine to produce; you
never need to interpret a gate.
