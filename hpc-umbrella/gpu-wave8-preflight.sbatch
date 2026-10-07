#!/bin/bash
# =============================================================================
# gpu-wave8-preflight.sbatch — WAVE8 pre-flight F2 + F3 on the cluster (docs/waves/WAVE8_SPEC.md §6.1).
#   F2 interface: all 26 labels single-token on the rendered continuation, ids contiguous, the real
#      apply_chat_template prompt SHA equals the pure-string renderer SHA (scripts/wave8_grid.py).
#   F3 control stats: the UNTRAINED readout on the fixed sample S_nom (1,024 states, committed by
#      the hub: results/wave8/S_nom.jsonl + S_nom_oracle.json) -> thresholds frozen into the
#      rung's gates json BEFORE any training. Refuses to overwrite an existing F3.
# Needs (arrive via git): results/wave8/{S_nom.jsonl,S_nom_oracle.json,gates.json with F0,F1 passing}.
#
#   Submit:  cd /scratch-shared/$USER/bullwhip/kit && sbatch gpu-wave8-preflight.sbatch
#            W8_RUNG=1.7b sbatch gpu-wave8-preflight.sbatch   # per-rung F2/F3 (the untrained readout differs)
#   Results: results/wave8/gates.json (F2, F3)  [rung files: gates_1.7b.json / gates_4b.json]
#            results/wave8/S_nom_control[-<rung>].json
# =============================================================================
#SBATCH --job-name=bw8-preflight
#SBATCH --partition=tue.gpu.q
#SBATCH --gres=gpu:1
#SBATCH --constraint=l4
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=/scratch-shared/%u/bullwhip/logs/wave8-preflight_%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=umbrella-cluster@pepijnfrenken.nl

set -euo pipefail

KIT_DIR="${SLURM_SUBMIT_DIR:-}"
if [ -z "${KIT_DIR}" ] || [ ! -r "${KIT_DIR}/config.sh" ]; then
  KIT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
if [ ! -r "${KIT_DIR}/config.sh" ]; then
  echo "FATAL: config.sh not found in KIT_DIR=${KIT_DIR} — submit from the kit dir (cd <kit> && sbatch <file>)" >&2
  exit 1
fi
# shellcheck source=config.sh
. "${KIT_DIR}/config.sh"

bw_load_env
export HF_HUB_OFFLINE=1
mkdir -p "${LOG_DIR}"
cd "${REPO_DIR}"
bw_w8_rung

REPO_HEAD="$(git -C "${REPO_DIR}" rev-parse HEAD 2>/dev/null || echo unknown)"
echo "repo ${REPO_DIR} HEAD=${REPO_HEAD} rung=${W8_RUNG:-0.6b} model=${RUNG_MODEL}@${RUNG_REV}"
if [ -n "${EXPECTED_COMMIT:-}" ] && [ "${REPO_HEAD}" != "${EXPECTED_COMMIT}" ]; then
  echo "FATAL: EXPECTED_COMMIT=${EXPECTED_COMMIT} but repo HEAD=${REPO_HEAD} — refusing to run stale code" >&2
  exit 1
fi

for f in results/wave8/S_nom.jsonl results/wave8/S_nom_oracle.json results/wave8/gates.json; do
  [ -s "${f}" ] || { echo "FATAL: missing ${f} — git pull the hub's results/wave8 first" >&2; exit 1; }
done
python3 - <<'PY'
import json, sys
g = json.load(open("results/wave8/gates.json"))
bad = [k for k in ("F0", "F1") if not ((g.get(k) or {}).get("verdict") or {}).get("pass")]
if bad:
    sys.exit(f"FATAL: pre-flight {bad} not passing in results/wave8/gates.json — no GPU job is run (W8 §6.1)")
print("F0, F1 passing:", {k: g[k]["verdict"]["pass"] for k in ("F0", "F1")})
PY

RUNG_TAG="${W8_RUNG:-0.6b}"
GRID_OUT="results/wave8/grid_w8_g26b.json"
[ "${RUNG_TAG}" != "0.6b" ] && GRID_OUT="results/wave8/grid_w8_g26b${RUNG_SUFFIX}.json"
echo "=== F2: tokenizer check (${RUNG_MODEL}) ==="
python3 -u scripts/wave8_grid.py --tokenizer-check --model "${RUNG_MODEL}" --revision "${RUNG_REV}" --out "${GRID_OUT}"

echo "=== F3: freeze control stats on S_nom (untrained readout) ==="
python3 -u scripts/wave8_gates.py control --model "${RUNG_MODEL}" --revision "${RUNG_REV}" \
  --gates "${RUNG_GATES}" --out "results/wave8/S_nom_control${RUNG_SUFFIX}.json" ${W8_FORCE_F3:+--force}

echo "=== gate report ==="
python3 scripts/wave8_gates.py report --gates "${RUNG_GATES}"
echo "=== wave8-preflight done at $(date -Is) ==="
