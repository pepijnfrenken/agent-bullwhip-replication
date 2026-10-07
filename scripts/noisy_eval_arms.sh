#!/usr/bin/env bash
# Launch both noisy arms AFTER the step arms complete (one runner per server).
#   results/local-eval/untrained-noisy  (8081)   --pattern noisy --runs 10
#   results/local-eval/trained-noisy    (8082)   --pattern noisy --runs 10
# Usage (from repo root):
#   bash noisy_eval_arms.sh
set -uo pipefail
cd "$(dirname "$0")/.."

run_arm() {
  local port="$1" model="$2" outdir="$3" label="$4"
  FREEINFERENCE_BASE="http://127.0.0.1:${port}" FREEINFERENCE_RATE_LIMIT_DELAY=0 \
  BULLWHIP_COMPLETIONS_API=1 FREEINFERENCE_MODEL="$model" \
  uv run --no-project --with requests python3 -m agent_bullwhip.runner \
    --config baseline --runs 10 --pattern noisy --outdir "$outdir" \
    > "results/local-eval/${label}.log" 2>&1
  echo "arm $label exit=$?"
}

run_arm 8081 qwen3-0.6b-untrained results/local-eval/untrained-noisy untrained-noisy &
run_arm 8082 qwen3-0.6b-trained results/local-eval/trained-noisy trained-noisy &
wait
echo "=== NOISY ARMS DONE ==="