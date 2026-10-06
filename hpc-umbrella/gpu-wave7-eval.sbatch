#!/bin/bash
# =============================================================================
# gpu-wave7-eval.sbatch — wave-7 protocol evals (batched readout, one L4).
#
# The wave-7 result arms (WAVE7_SPEC D5 + AUDIT-1 fixes, binding): step-argmax
# (n=1, single labeled game), step-sampled (n=30, seeds 1000+i), noisy-argmax
# (n=30) for the untrained readout (the C1 control) and every trained adapter
# (sft_seed{0,1,2} checkpoint-119, rlcd_seed{0,1,2} checkpoint-39), plus the
# AUDIT-1 amendment-1 permutation-invariance gate probe per adapter.
#
# 21 eval runs + 7 probes; minutes each on an L4 (model load dominates).
# Outputs (REPO_DIR-relative):
#   results/eval/w7-<arm>-<mode>-<pattern>.jsonl + .summary.json
#   results/eval/w7-<arm>-perm.json            (permutation gate: >=0.90)
# Commit them with the wave-7 force-add flow (results/ is gitignored).
#
#   Submit:  cd /scratch-shared/$USER/bullwhip/kit && sbatch gpu-wave7-eval.sbatch
#   Needs:   results/wave7_{sft,rlcd}_seed*/checkpoint-*  (gpu-wave7-train.sbatch)
# =============================================================================
#SBATCH --job-name=bw7-eval
#SBATCH --partition=tue.gpu.q
#SBATCH --gres=gpu:1
#SBATCH --constraint=l4
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=/scratch-shared/%u/bullwhip/logs/wave7-eval_%j.log
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
export HF_HUB_OFFLINE=1            # compute nodes have no egress
mkdir -p "${LOG_DIR}"
cd "${REPO_DIR}"

REPO_HEAD="$(git -C "${REPO_DIR}" rev-parse HEAD 2>/dev/null || echo unknown)"
echo "repo ${REPO_DIR} HEAD=${REPO_HEAD}"
if [ -n "${EXPECTED_COMMIT:-}" ] && [ "${REPO_HEAD}" != "${EXPECTED_COMMIT}" ]; then
  echo "FATAL: EXPECTED_COMMIT=${EXPECTED_COMMIT} but repo HEAD=${REPO_HEAD} — refusing to run stale code" >&2
  exit 1
fi

OUT="results/eval"
mkdir -p "${OUT}"
FAILED=0

ev() {  # ev <tag> <model-name> <pattern> <mode> [adapter]
  local tag="$1" mname="$2" pattern="$3" mode="$4" adapter="${5:-}"
  local aargs=()
  [ -n "${adapter}" ] && aargs=(--adapter "${adapter}")
  echo "--- eval ${tag} (${pattern}/${mode}) at $(date -Is)"
  if python3 -u -m agent_bullwhip.decision_eval \
      --model "${MODEL_ID}" --outdir "${OUT}" \
      --tag "${tag}" --model-name "${mname}" \
      --pattern "${pattern}" --mode "${mode}" \
      ${aargs[@]+"${aargs[@]}"}; then
    echo "--- ok ${tag}"
  else
    echo "--- STEP FAILED: ${tag} rc=$?" >&2
    FAILED=1
  fi
}

perm() {  # perm <arm> <adapter|-> <gate>
  local arm="$1" adapter="$2" gate="$3"
  local aargs=()
  [ "${adapter}" != "-" ] && aargs=(--adapter "${adapter}")
  echo "--- perm probe ${arm} (gate=${gate}) at $(date -Is)"
  if python3 -u probe_permutation_invariance.py \
      --model "${MODEL_ID}" \
      --cases results/wave7_labels/policy_states_seed0.jsonl \
      --n 512 --perms 3 --gate "${gate}" \
      ${aargs[@]+"${aargs[@]}"} \
      --out "${OUT}/w7-${arm}-perm.json"; then
    echo "--- ok perm ${arm}"
  else
    echo "--- PROBE ${arm}: gate failed or errored (recorded in the JSON; continuing)" >&2
  fi
}

# --- untrained control (C1 reference; mass >= 0.99 stays fatal here) ---------
ev w7-untrained-argmax-step  qwen3-0.6b-untrained step  argmax
ev w7-untrained-sampled-step qwen3-0.6b-untrained step  sampled
ev w7-untrained-argmax-noisy qwen3-0.6b-untrained noisy argmax
perm untrained - 0                 # control: low invariance is the expected contrast

# --- trained adapters: every seed, both stages -------------------------------
for SEED in 0 1 2; do
  for ARM in sft rlcd; do
    if [ "${ARM}" = sft ]; then
      ADP="results/wave7_sft_seed${SEED}/checkpoint-119"
    else
      ADP="results/wave7_rlcd_seed${SEED}/checkpoint-39"
    fi
    if [ ! -d "${ADP}" ]; then
      echo "FATAL: missing adapter ${ADP} — run gpu-wave7-train.sbatch first" >&2
      exit 1
    fi
    for PM in "argmax step" "sampled step" "argmax noisy"; do
      set -- ${PM}
      ev "w7-${ARM}-seed${SEED}-$1-$2" "w7-${ARM}-seed${SEED}" "$2" "$1" "${ADP}"
    done
    perm "${ARM}-seed${SEED}" "${ADP}" 0.90
  done
done

echo "--- artifacts ---"
ls -la "${OUT}" | tail -30
echo "=== wave7-eval done at $(date -Is) failed=${FAILED} ==="
exit "${FAILED}"
