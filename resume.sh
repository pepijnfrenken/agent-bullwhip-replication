#!/usr/bin/env bash
# Resume the agent bullwhip matrix: skip configs with existing completed summaries.
set -uo pipefail
cd "$(dirname "$0")"

PY=.venv/bin/python
RUNNER="agent_bullwhip.runner"
MODELS=("deepseek-v4-flash" "qwen3.6-35b")
CONFIGS=(baseline budget prompt_weighted voting5 anchor combined mirror order_up_to)
RUNS=30
HORIZON=36

for model in "${MODELS[@]}"; do
  for config in "${CONFIGS[@]}"; do
    tag="${model}-${config}"
    summary="results/${tag}.summary.json"
    if [ -f "$summary" ] && grep -q '"completed": 30' "$summary"; then
      echo "SKIP $tag (already complete)"
      continue
    fi
    echo "=== RUN $tag ==="
    $PY -m "$RUNNER" --config "$config" --runs "$RUNS" --horizon "$HORIZON" --model "$model"
  done
done

echo "=== RESUME DONE ==="
