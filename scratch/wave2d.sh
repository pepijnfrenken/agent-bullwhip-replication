#!/usr/bin/env bash
# Wave 2d: generalizable (domain-agnostic) KB configs — 30 runs each.
set -uo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
RUNNER="agent_bullwhip.runner"
MODEL="deepseek-v4-flash"
CONFIGS=(kb_general kb_general_introspect kb_general_gated)
RUNS=30
HORIZON=36
for config in "${CONFIGS[@]}"; do
  tag="${MODEL}-${config}"
  if [ -f "results/${tag}.summary.json" ] && grep -q '"completed": 30' "results/${tag}.summary.json"; then
    echo "SKIP $tag (already complete)"
    continue
  fi
  echo "=== RUN $tag ==="
  $PY -m "$RUNNER" --config "$config" --runs "$RUNS" --horizon "$HORIZON" --model "$MODEL"
done
echo "=== WAVE2D DONE ==="
