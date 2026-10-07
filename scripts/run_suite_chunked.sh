#!/bin/bash
# Chunked exact-stack suite: one test file per python process.
# Rationale: the monolithic run gets SIGKILLed (exit 137) on this 15 GB hub when
# agents hold their own RSS — each module-scoped _Policy fixture loads the 0.6B
# model (~2.4 GB fp32) and several coexist in one process. Chunking bounds peak
# memory to a single model instance.
cd /home/pino/projects/agent-bullwhip-replication || exit 1
PY=/home/pino/.hermes/cache/scratch/w7s/bin/python
export HF_HUB_OFFLINE=1
FAILED=""
for f in tests/test_*.py; do
  echo "=== $f ==="
  if ! timeout 400 "$PY" -m pytest "$f" -q; then
    FAILED="$FAILED $f"
  fi
done
echo "=== SUMMARY failed:${FAILED:- none} ==="
