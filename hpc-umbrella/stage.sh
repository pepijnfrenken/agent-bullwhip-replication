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
# Usage:   bash stage.sh [--dry-run] [--skip-weights] [--skip-venv] [--with-ladder] [-h]
#   --dry-run        print every action, change nothing (works off-cluster too)
#   --skip-weights   do not download the model
#   --skip-venv      reuse the existing venv
#   --with-ladder    also download Qwen3-1.7B and Qwen3-4B (WAVE8 size ladder; 4B ~ 8 GB)
# =============================================================================
set -euo pipefail

KIT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=config.sh
. "${KIT_DIR}/config.sh"

ARGS=("$@")          # kept for the re-exec into the staged copy
DRY=0; SKIP_WEIGHTS=0; SKIP_VENV=0; WITH_LADDER=0
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY=1 ;;
    --skip-weights) SKIP_WEIGHTS=1 ;;
    --skip-venv) SKIP_VENV=1 ;;
    --with-ladder) WITH_LADDER=1 ;;
    -h|--help) sed -n '2,19p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

say() { printf '[stage] %s\n' "$*"; }
doit() { if [ "$DRY" = 1 ]; then printf '  would: %s\n' "$*"; else printf '  + %s\n' "$*"; eval "$*"; fi; }

# Non-interactive hardening: a credential / host-key prompt on the login node must
# FAIL FAST with a readable error instead of freezing the staging run. If git says
# "Permission denied (publickey)" here, fix the checkout key with: ssh -T git@github.com
export GIT_TERMINAL_PROMPT=0 GIT_ASKPASS=/bin/true SSH_ASKPASS=/bin/true
export GIT_SSH_COMMAND="${GIT_SSH_COMMAND:-ssh -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new}"

[ "$DRY" = 1 ] || bw_load_env

say "kit -> ${BULLWHIP_ROOT}/kit"
doit "mkdir -p '${BULLWHIP_ROOT}/kit' '${LOG_DIR}' '${HF_HOME}' '${PIP_CACHE_DIR}'"
if [ "$DRY" = 0 ] && [ "$(cd "${KIT_DIR}" && pwd)" != "$(cd "${BULLWHIP_ROOT}/kit" && pwd 2>/dev/null || echo x)" ]; then
  # tar-mirror keeps campaign.tsv + sbatch + config.sh in sync with this checkout
  # (the login node has NO rsync, so we mirror manually: rm -rf + fresh copy, where
  # rsync --delete used to prune removed files). The .sbatch files are always
  # submitted from the staged copy so SLURM_SUBMIT_DIR is stable.
  # NEVER run this from inside the kit dir: rm -rf kit would delete the caller's CWD,
  # and every later command in that shell dies with 'getcwd: No such file or directory'.
  if case "$PWD" in "${BULLWHIP_ROOT}/kit"|"${BULLWHIP_ROOT}/kit"/*) true;; *) false;; esac; then
    say "FATAL: your shell is inside '${BULLWHIP_ROOT}/kit', which staging replaces (rm -rf).
  cd out first (e.g. 'cd \"${BULLWHIP_ROOT}\"'), then re-run stage.sh"
    exit 1
  fi
  doit "rm -rf '${BULLWHIP_ROOT}/kit' && mkdir -p '${BULLWHIP_ROOT}/kit' && tar -C '${KIT_DIR}' -cf - . | tar -C '${BULLWHIP_ROOT}/kit' -xf -"
  say "re-exec from the staged copy so the run uses scratch, not home"
  exec bash "${BULLWHIP_ROOT}/kit/stage.sh" "${ARGS[@]}"
fi

say "repo -> ${REPO_DIR}"
if [ -d "${REPO_DIR}/.git" ]; then
  # F6: a failed fetch/pull must FAIL staging — the sweep once ran pre-fix code
  # with nothing in any log saying so (stage.sh swallowed the pull failure and
  # the array ran a stale checkout). The HEAD + dirty stamp goes into the stage
  # log, and an EXPECTED_COMMIT pin (config.sh, empty = log only) becomes a hard
  # gate: staging refuses to serve any other commit.
  # Private repo (2026-10-01): pin the clone's origin to config.sh's URL so an existing
  # clone created over HTTPS (or any drifted origin) self-heals to SSH — HTTPS would need
  # a token for a private repo; the login node's key is already authorized.
  doit "git -C '${REPO_DIR}' remote set-url origin '${REPO_URL}'"
  doit "git -C '${REPO_DIR}' fetch --all --prune || { echo '[stage] FATAL: git fetch failed (no egress?) — refusing to stage a possibly-stale checkout' >&2; exit 1; }"
  doit "git -C '${REPO_DIR}' checkout master"
  doit "git -C '${REPO_DIR}' pull --ff-only || { echo '[stage] FATAL: git pull failed — refusing to stage a possibly-stale checkout' >&2; exit 1; }"
  doit "echo \"[stage] repo HEAD=\$(git -C '${REPO_DIR}' rev-parse HEAD) dirty_files=\$(git -C '${REPO_DIR}' status --porcelain | wc -l)\""
  doit "if [ -n \"${EXPECTED_COMMIT:-}\" ] && [ \"\$(git -C '${REPO_DIR}' rev-parse HEAD)\" != \"${EXPECTED_COMMIT:-}\" ]; then echo '[stage] FATAL: EXPECTED_COMMIT=${EXPECTED_COMMIT} but HEAD is not that commit — refusing to stage' >&2; exit 1; fi"
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
  doit ". '${VENV_DIR}/bin/activate' && python3 -m pip install --no-input --upgrade pip"
  if [ "$ENV_MODE" = "pip" ]; then
    doit ". '${VENV_DIR}/bin/activate' && python3 -m pip install --no-input 'torch==2.5.1' --index-url https://download.pytorch.org/whl/cu121"
  fi
  doit ". '${VENV_DIR}/bin/activate' && python3 -m pip install --no-input -r '${REQ_OUT}'"
else
  say "--skip-venv: leaving ${VENV_DIR} untouched"
fi

if [ "$SKIP_WEIGHTS" = 0 ]; then
  say "weights ${MODEL_ID} -> ${HF_HOME}"
  doit "HF_HOME='${HF_HOME}' python3 -c \"from huggingface_hub import snapshot_download as d; print(d('${MODEL_ID}', revision='${MODEL_REVISION}'))\" || echo '[stage] HF download failed (no egress?) — upload weights from the laptop, see OPERATOR.md'"
  say "if this fails: no login-node egress — upload from the laptop instead:"
  say "  tar -C ~/.cache/huggingface/hub -cf - models--Qwen--Qwen3-0.6B | ssh ${USER}@hpc.tue.nl 'mkdir -p ${HF_HOME}/hub && tar -C ${HF_HOME}/hub -xf -'"
else
  say "--skip-weights: not downloading"
fi

if [ "$WITH_LADDER" = 1 ]; then
  for pair in "${MODEL_ID_17B}:${MODEL_REVISION_17B}" "${MODEL_ID_4B}:${MODEL_REVISION_4B}"; do
    say "ladder weights ${pair%%:*} @ ${pair##*:} -> ${HF_HOME}"
    doit "HF_HOME='${HF_HOME}' python3 -c \"from huggingface_hub import snapshot_download as d; print(d('${pair%%:*}', revision='${pair##*:}'))\" || echo '[stage] ladder download failed (no egress?)'"
  done
fi

say "verify"
doit "bash '${BULLWHIP_ROOT}/kit/config.sh' --check || true"
# full run, no -x: with -x a stale test hides every failure behind it (observed 2026-10-08: one
# real failure masked a second one for hours). Non-blocking — a red suite warns loudly, it does
# not stop staging, so a GPU job is never lost to a test-only problem.
doit "cd '${REPO_DIR}' && python3 -m pytest tests/ -q -k 'not torch_available' || echo '[stage] WARNING: pytest reported failures above — read them before trusting a run'"
doit "cd '${REPO_DIR}' && python3 -c \"import torch,transformers,peft;print('torch',torch.__version__,'transformers',transformers.__version__,'peft',peft.__version__,'cuda',torch.cuda.is_available())\" || echo 'python stack incomplete (expected until the module stack is right)'"

say "next:"
say "  cd ${BULLWHIP_ROOT}/kit && sbatch cpu-harness.sbatch      # free, immediate"
say "  cd ${BULLWHIP_ROOT}/kit && sbatch gpu-pilot.sbatch        # the unblock + Snellius numbers"
say "  read OPERATOR.md for the full order and the failure playbook"
