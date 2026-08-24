#!/usr/bin/env bash
# Agent Bullwhip — full experimental matrix
# Wave 1: the paper's own levers (model selection x2, policy/guardrail, prompt, voting negative-check)
# Wave 2: our extensions (anchor, combined)
# Deterministic baselines: mirror, order_up_to
set -euo pipefail
cd "$(dirname "$0")"

PY=.venv/bin/python
RUNNER="agent_bullwhip.runner"
MODELS=("deepseek-v4-flash" "qwen3.6-35b")
CONFIGS=(baseline budget prompt_weighted voting5 anchor combined mirror order_up_to)
RUNS=30
HORIZON=36

for model in "${MODELS[@]}"; do
  for config in "${CONFIGS[@]}"; do
    echo "=== $model / $config ==="
    $PY -m "$RUNNER" --config "$config" --runs "$RUNS" --horizon "$HORIZON" --model "$model"
  done
done

echo "=== ALL DONE ==="
