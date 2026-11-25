#!/bin/bash
# diagnose-429581.sh — run on the login node, paste the whole output back.
set -u
R=/scratch-shared/<cluster-account>/bullwhip/agent-bullwhip-replication
B=/scratch-shared/<cluster-account>/bullwhip
cd "$R" || exit 1

echo "=== 1. sacct, exact per-task exit ==="
sacct -j 429581 -X -P -o JobID,State,ExitCode,Elapsed,End

echo "=== 2. the three job logs, tail -40 each ==="
for f in $(basename -s .log $(ls -1 "$B"/logs/wave8-train_429581_*.log 2>/dev/null)); do
  echo "--- ${f#wave8-train_} ---"
  tail -40 "$B/logs/${f}.log"
done

echo "=== 3. what the r2 arm dirs look like right now ==="
ls results/wave8_r2sft_seed0/ 2>/dev/null | head
python3 - <<'PY'
import json, glob, pathlib
for p in sorted(glob.glob("/scratch-shared/<cluster-account>/bullwhip/agent-bullwhip-replication/results/wave8_r2sft_seed*/state.json")):
    d = json.load(open(p))
    print(p.split("results/")[-1], "->", {k: d.get(k) for k in ("start_step", "step", "checkpoint_dir") if k in d})
PY

echo "=== 4. GitHub auth sanity (push_results needs a working push) ==="
git ls-remote origin refs/heads/master | head -1
