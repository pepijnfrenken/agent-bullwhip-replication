#!/usr/bin/env bash
# Run the Wave 2 matrix: introspective / KB configs (30 runs each, deepseek).
set -uo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
RUNNER="agent_bullwhip.runner"
MODEL="deepseek-v4-flash"
CONFIGS=(kb introspect kb_introspect kb_introspect_gated kb_system kb_system_introspect kb_system_gated)
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
echo "=== WAVE2 DONE ==="
