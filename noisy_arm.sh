#!/usr/bin/env bash
# Run the noisy-demand robustness arm (Wave 3): does the reliability advantage
# survive when the deterministic floor itself faces variance?
# Run AFTER the verbal-gate ablation finishes (FreeInference concurrency = 2).
set -uo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
MODEL="deepseek-v4-flash"
# key configs: baseline (chaos), kb_system_gated (best), kb_pointer_verbal (new gate),
# order_up_to (deterministic floor — now has CV>0 under noise)
CONFIGS=(baseline kb_system_gated kb_pointer_verbal order_up_to)
RUNS=30
HORIZON=36
for config in "${CONFIGS[@]}"; do
  tag="${MODEL}-${config}-noisy"
  if [ -f "results/${tag}.summary.json" ] && grep -q '"completed": 30' "results/${tag}.summary.json"; then
    echo "SKIP $tag (already complete)"
    continue
  fi
  echo "=== RUN $tag ==="
  $PY -m agent_bullwhip.runner --config "$config" --runs "$RUNS" --horizon "$HORIZON" \
      --model "$MODEL" --pattern noisy
done
echo "=== NOISY ARM DONE ==="
