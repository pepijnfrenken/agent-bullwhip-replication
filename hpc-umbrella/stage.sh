#!/usr/bin/env bash
# =============================================================================
# hpc-umbrella/stage.sh — login-node staging for the bullwhip Wave-6 Umbrella run.
#
# Does, in order (everything idempotent — safe to re-run):
#   1. pool the kit into $BULLWHIP_ROOT/kit            (sbatch must be submitted from scratch)
#   2. clone / update the repo at $REPO_DIR
#   3. create the venv + install the requirements track (see requirements-umbrella.txt)
#   4. download Qwen/Qwen3-0.6B into $HF_HOME           (needs login-node egress)
#   5. verify: config.sh --check + a torch-free pytest run + one GPU-free import check
#
# Usage:   bash stage.sh [--dry-run] [--skip-weights] [--skip-venv] [-h]
#   --dry-run        print every action, change nothing (works off-cluster too)
#   --skip-weights   do not download the model
#   --skip-venv      reuse the existing venv
# =============================================================================
set -euo pipefail

KIT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=config.sh
. "${KIT_DIR}/config.sh"

ARGS=("$@")          # kept for the re-exec into the staged copy
DRY=0; SKIP_WEIGHTS=0; SKIP_VENV=0
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY=1 ;;
    --skip-weights) SKIP_WEIGHTS=1 ;;
    --skip-venv) SKIP_VENV=1 ;;
    -h|--help) sed -n '2,18p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

say() { printf '[stage] %s\n' "$*"; }
doit() { if [ "$DRY" = 1 ]; then printf '  would: %s\n' "$*"; else eval "$*"; fi; }

[ "$DRY" = 1 ] || bw_load_env

say "kit -> ${BULLWHIP_ROOT}/kit"
doit "mkdir -p '${BULLWHIP_ROOT}/kit' '${LOG_DIR}' '${HF_HOME}' '${PIP_CACHE_DIR}'"
if [ "$DRY" = 0 ] && [ "$(cd "${KIT_DIR}" && pwd)" != "$(cd "${BULLWHIP_ROOT}/kit" && pwd 2>/dev/null || echo x)" ]; then
  # rsync keeps campaign.tsv + sbatch + config.sh in sync with this checkout; the
  # .sbatch files are always submitted from the staged copy so SLURM_SUBMIT_DIR is stable.
  doit "rsync -a --delete '${KIT_DIR}/' '${BULLWHIP_ROOT}/kit/'"
  say "re-exec from the staged copy so the run uses scratch, not home"
  exec bash "${BULLWHIP_ROOT}/kit/stage.sh" "${ARGS[@]}"
fi

say "repo -> ${REPO_DIR}"
if [ -d "${REPO_DIR}/.git" ]; then
  # no-egress tolerant: a failed fetch/pull must not abort staging on the cluster
  doit "git -C '${REPO_DIR}' fetch --all --prune || echo '[stage] fetch failed (no egress?) — using the existing checkout'"
  doit "git -C '${REPO_DIR}' checkout master && git -C '${REPO_DIR}' pull --ff-only || echo '[stage] pull failed — using the existing checkout'"
else
  doit "git clone '${REPO_URL}' '${REPO_DIR}'"
fi

if [ "$SKIP_VENV" = 0 ]; then
  say "venv -> ${VENV_DIR} (ENV_MODE=${ENV_MODE})"
  if [ ! -d "${VENV_DIR}" ]; then
    if [ "$ENV_MODE" = "module" ]; then
      # --system-site-packages so the module's torch/CUDA bindings are visible
      doit "python3 -m venv --system-site-packages '${VENV_DIR}'"
    else
      doit "python3 -m venv '${VENV_DIR}'"
    fi
  else
    say "venv exists — reusing (delete ${VENV_DIR} to rebuild)"
  fi

  # extract the requirement block for this track from the single requirements file
  REQ_IN="${KIT_DIR}/requirements-umbrella.txt"
  REQ_OUT="${BULLWHIP_ROOT}/requirements-$( [ "$ENV_MODE" = module ] && echo A || echo B ).txt"
  TRACK="$( [ "$ENV_MODE" = module ] && echo "TRACK A" || echo "TRACK B" )"
  doit "sed -n '/>>>BEGIN ${TRACK}/,/<<<END ${TRACK}/p' '${REQ_IN}' | grep -v '>>>BEGIN\\|<<<END' > '${REQ_OUT}'"
  say "installing ${TRACK} pins -> ${REQ_OUT}"
  doit ". '${VENV_DIR}/bin/activate' && python3 -m pip install --upgrade pip"
  if [ "$ENV_MODE" = "pip" ]; then
    doit ". '${VENV_DIR}/bin/activate' && python3 -m pip install 'torch==2.5.1' --index-url https://download.pytorch.org/whl/cu121"
  fi
  doit ". '${VENV_DIR}/bin/activate' && python3 -m pip install -r '${REQ_OUT}'"
else
  say "--skip-venv: leaving ${VENV_DIR} untouched"
fi

if [ "$SKIP_WEIGHTS" = 0 ]; then
  say "weights ${MODEL_ID} -> ${HF_HOME}"
  doit "HF_HOME='${HF_HOME}' python3 -c \"from huggingface_hub import snapshot_download as d; print(d('${MODEL_ID}'))\" || echo '[stage] HF download failed (no egress?) — upload weights from the laptop, see OPERATOR.md'"
  say "if this fails: no login-node egress — upload from the laptop instead:"
  say "  rsync -av ~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B ${USER}@hpc.tue.nl:${HF_HOME}/hub/"
else
  say "--skip-weights: not downloading"
fi

say "verify"
doit "bash '${BULLWHIP_ROOT}/kit/config.sh' --check || true"
doit "cd '${REPO_DIR}' && python3 -m pytest tests/ -q -x -k 'not torch_available' || true"
doit "cd '${REPO_DIR}' && python3 -c \"import torch,transformers,peft;print('torch',torch.__version__,'transformers',transformers.__version__,'peft',peft.__version__,'cuda',torch.cuda.is_available())\" || echo 'python stack incomplete (expected until the module stack is right)'"

say "next:"
say "  cd ${BULLWHIP_ROOT}/kit && sbatch cpu-harness.sbatch      # free, immediate"
say "  cd ${BULLWHIP_ROOT}/kit && sbatch gpu-pilot.sbatch        # the unblock + Snellius numbers"
say "  read OPERATOR.md for the full order and the failure playbook"
