#!/usr/bin/env bash
# =============================================================================
# config-snellius.sh -- Snellius burst-lane overlay (A2 amendment, WAVE8 SS9.3).
#
# Sources the TU/e kit config, then overrides everything cluster-specific.
# The ONLY legal use: rung-8b (Qwen3-8B) on the gpu_a100 partition. Snellius
# credits are a temporary SURF workshop grant -- burst lane, never mainline; the
# wave-8 TU/e submit cap (60 GPU-h) is NOT spent here and is not raised by this.
#
# Snellius facts settled 2026-10-21:
#   - module stacks only carry torch 2.1.2 (too old for transformers 4.57, which
#     needs >=2.6) -> PURE-PIP venv at $VENV_DIR: torch 2.8.0+cu126 (self-hosted
#     CUDA runtime, no module needed), transformers 4.57.6, accelerate 1.10.1.
#   - Qwen/Qwen3-8B snapshot b968826d9c46dd6066d109eabc6255188de91218 fully in
#     $HF_HOME (5 shards, ~16 GB) -- compute nodes run offline.
#   - gpu_a100: 4x A100-80G/node, 72 CPU, shared nodes, max walltime 5 days.
# =============================================================================
set -u
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/config.sh"

# --- account / filesystem ------------------------------------------------------
SCRATCH_ROOT="/home/scur0659/bullwhip8b"
BULLWHIP_ROOT="/home/scur0659/bullwhip8b"
REPO_DIR="${BULLWHIP_ROOT}/agent-bullwhip-replication"
LOG_DIR="${BULLWHIP_ROOT}/logs"
RESULTS_DIR="${REPO_DIR}/results"
HF_HOME="${BULLWHIP_ROOT}/hf-cache"
PIP_CACHE_DIR="${BULLWHIP_ROOT}/pipcache"
TORCHINDUCTOR_CACHE_DIR="${BULLWHIP_ROOT}/torchinductor"
VENV_DIR="${BULLWHIP_ROOT}/venv"
BBOX_TMP="${BULLWHIP_ROOT}/tmp"
export HF_HOME PIP_CACHE_DIR TORCHINDUCTOR_CACHE_DIR

# --- environment: pure-pip, NO modules -----------------------------------------
# bw_load_env's module path would try Umbrella/2024 (TU/e-only). Instead of
# loading modules this replacement only activates the pip venv (torch 2.8.0
# cu126 wheel carries its own CUDA runtime).
bw_load_env() {
  # shellcheck disable=SC1091
  . "${VENV_DIR}/bin/activate"
  export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
  export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
  export TOKENIZERS_PARALLELISM=false
  export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
  mkdir -p "$LOG_DIR" "$HF_HOME" "$PIP_CACHE_DIR" "$BBOX_TMP" 2>/dev/null || true
}

# --- rung pins ------------------------------------------------------------------
ENV_MODE="pip-frozen"   # marker only; bw_load_env above is module-free

# --- partitions -----------------------------------------------------------------
GPU_PARTITION="gpu_a100"
GPU_GRES="gpu:1"
L4_CONSTRAINT=""
CPU_PARTITION="gpu_a100"
WALLTIME_GPU_CAP="5-00:00:00"

# --- ladder rung 4 (A2): Qwen3-8B ------------------------------------------------
MODEL_ID_8B="Qwen/Qwen3-8B"
MODEL_REVISION_8B="b968826d9c46dd6066d109eabc6255188de91218"
WALLTIME_W8_TRAIN_8B="20:00:00"

bw_w8_rung() {
  case "${W8_RUNG:-0.6b}" in
    0.6b) RUNG_MODEL="${MODEL_ID}";     RUNG_REV="${MODEL_REVISION}";     RUNG_SUFFIX="";      RUNG_GATES="results/wave8/gates.json" ;;
    1.7b) RUNG_MODEL="${MODEL_ID_17B}"; RUNG_REV="${MODEL_REVISION_17B}"; RUNG_SUFFIX="-1.7b"; RUNG_GATES="results/wave8/gates_1.7b.json" ;;
    4b)   RUNG_MODEL="${MODEL_ID_4B}";  RUNG_REV="${MODEL_REVISION_4B}";  RUNG_SUFFIX="-4b";   RUNG_GATES="results/wave8/gates_4b.json" ;;
    8b)   RUNG_MODEL="${MODEL_ID_8B}";  RUNG_REV="${MODEL_REVISION_8B}";  RUNG_SUFFIX="-8b";   RUNG_GATES="results/wave8/gates_8b.json" ;;
    *) echo "FATAL: W8_RUNG must be 0.6b|1.7b|4b|8b (got '${W8_RUNG:-}')" >&2; return 2 ;;
  esac
}
