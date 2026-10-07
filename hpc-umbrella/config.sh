#!/usr/bin/env bash
# =============================================================================
# hpc-umbrella/config.sh — THE pins file for the bullwhip Wave-6 Umbrella run.
#
# Every probe/sbatch/stage script sources this file; nothing else in the kit
# hardcodes a partition, module, path, walltime or run-shape knob. Values marked
# "probe:" were reconciled against the login-node probe of 2026-09-30 (facts and
# rulings in 00-PROBE-AND-SETUP.md -> "Login-node probe results"); the module pins
# were load-tested the same day. `bash config.sh --check` verifies everything
# against the host on the cluster.
#
# Slurm limitation worth knowing: #SBATCH directives are read before the shell
# runs, so an sbatch file cannot interpolate ${GPU_PARTITION}. The sbatch headers
# therefore repeat these pins literally, and `config.sh --check` re-reads the
# whole sbatch set and fails if a header disagrees with the pin below — the
# "one file" property is enforced, not just documented.
#
# Modes:
#   source config.sh          set the variables (+ best-effort auto-detect)
#   bash config.sh --print    print resolved values, no checks (works anywhere)
#   bash config.sh --check    verify against this host; exit 1 if unresolved
#   bash config.sh --pins     print only the ambiguity table (paste vs probe.sh)
# =============================================================================
set -u

KIT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
USER="${USER:-$(id -un)}"
# Your TU/e (SCC) username (login-node probe 2026-09-30: 20234428). The emergency
# pull_results.sh mirror runs from the laptop and needs it explicitly. Set once here,
# never edit the scripts.
TUE_USER="${TUE_USER:-20234428}"
SSH_HOST="${SSH_HOST:-hpc.tue.nl}"

# --- filesystem --------------------------------------------------------------
# scratch-shared is 8 TiB but DELETED 14 days after last write; home is 200 GiB
# and NOT backed up. Results land in scratch, then travel off via git (push_results.sh
# from the cluster checkout; 'git pull --ff-only' on the hub).
SCRATCH_ROOT="/scratch-shared/${TUE_USER}"
BULLWHIP_ROOT="${BULLWHIP_ROOT:-${SCRATCH_ROOT}/bullwhip}"
REPO_DIR="${BULLWHIP_ROOT}/agent-bullwhip-replication"
LOG_DIR="${BULLWHIP_ROOT}/logs"
RESULTS_DIR="${REPO_DIR}/results"
HF_HOME="${BULLWHIP_ROOT}/hf"
PIP_CACHE_DIR="${BULLWHIP_ROOT}/pipcache"
TORCHINDUCTOR_CACHE_DIR="${BULLWHIP_ROOT}/torchinductor"
VENV_DIR="${BULLWHIP_ROOT}/venv"
BBOX_TMP="${BULLWHIP_ROOT}/tmp"
REPO_URL="git@github.com:pepijnfrenken/agent-bullwhip-replication.git"   # private repo → SSH (login-node key is authorized)
MODEL_ID="Qwen/Qwen3-0.6B"
# AUDIT-1 3a P2: pin the hub snapshot the wave-7 readout numbers were verified
# on (tokenizer/template bytes come from the snapshot). stage.sh downloads this
# revision; readout/train pass it to from_pretrained and the prompt-hash
# fixture (b96fb847...) asserts the same template bytes.
MODEL_REVISION="c1899de289a04d12100db370d81485cdf75e47ca"
CAMPAIGN_TSV="${KIT_DIR}/campaign.tsv"

# --- WAVE8 (docs/waves/WAVE8_SPEC.md) --------------------------------------------------
# Size ladder (WAVE8 §9.1). Revisions are the `sha` of https://huggingface.co/api/models/<id>
# read 2026-10-07 (lastModified 2025-07-26 for both). stage.sh --with-ladder downloads them
# (4B ~ 8.06 GB: usedStorage 8,056,404,654 B); compute nodes are offline (HF_HUB_OFFLINE=1).
MODEL_ID_17B="Qwen/Qwen3-1.7B"
MODEL_REVISION_17B="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
MODEL_ID_4B="Qwen/Qwen3-4B"
MODEL_REVISION_4B="1cfa9a7208912126459214e8b04321603b3df60c"
WALLTIME_W8_TRAIN="08:00:00"      # R1 ~1.1 h/run; R2 = SFT ~3.7 h + 3 RL rounds (W8 §4.2)

# W8_RUNG selects the model: 0.6b (default) | 1.7b | 4b. Sets RUNG_MODEL / RUNG_REV /
# RUNG_SUFFIX (arm-name suffix) / RUNG_GATES (the rung's frozen F2/F3 file).
bw_w8_rung() {
  case "${W8_RUNG:-0.6b}" in
    0.6b) RUNG_MODEL="${MODEL_ID}";     RUNG_REV="${MODEL_REVISION}";     RUNG_SUFFIX="";      RUNG_GATES="results/wave8/gates.json" ;;
    1.7b) RUNG_MODEL="${MODEL_ID_17B}"; RUNG_REV="${MODEL_REVISION_17B}"; RUNG_SUFFIX="-1.7b"; RUNG_GATES="results/wave8/gates_1.7b.json" ;;
    4b)   RUNG_MODEL="${MODEL_ID_4B}";  RUNG_REV="${MODEL_REVISION_4B}";  RUNG_SUFFIX="-4b";   RUNG_GATES="results/wave8/gates_4b.json" ;;
    *) echo "FATAL: W8_RUNG must be 0.6b|1.7b|4b (got '${W8_RUNG:-}')" >&2; return 2 ;;
  esac
}
CAMPAIGN_W6FIX_TSV="${KIT_DIR}/campaign-w6fix.tsv"   # WAVE6-FIX probe rows (gpu-w6fix.sbatch)

# F6: pin the exact commit staging must serve. Empty (default) = stage.sh logs
# HEAD + the dirty flag only; a full SHA makes stage.sh FAIL until the repo
# checkout sits at that commit. Set this after the fixes land and before the
# rerun sweep, so a stale deploy can never run silently again.
EXPECTED_COMMIT="${EXPECTED_COMMIT:-}"

# --- environment strategy ----------------------------------------------------
# module (recommended): Umbrella's PyTorch 2.1.2 + a --system-site-packages venv
# layered with peft/trl/pytest. Free, and it is an independent test of the parked
# version-pairing hypothesis (desktop was torch 2.5.1 + transformers 5.x).
# pip: torch 2.5.1+cu121 (the desktop-proven matrix) into a clean venv; needs the
# wheel downloads and a driver new enough for cu121.
ENV_MODE="${ENV_MODE:-module}"
# settled on the login node 2026-09-30 (load-tested: py 3.11.3, torch 2.1.2, built-cuda 12.1)
MODULE_UMBRELLA="Umbrella/2024"
MODULE_PYTORCH="PyTorch/2.1.2-foss-2023a-CUDA-12.1.1"
MODULE_PYTHON="Python/3.11.3-GCCcore-12.3.0"

# --- partitions --------------------------------------------------------------
# probe: `sinfo -o "%P %a %D %l %G"` + node features
CPU_PARTITION="tue.cpu3.q"            # 128 threads/node
CPU_PARTITION_BIG="tue.default.q"     # 1056 threads/node
GPU_PARTITION="tue.gpu.q"             # superset: gpua(A30-MIG)+gpub(4xL4)+gpuc(8xL4) — with -C l4 => 12x L4 (confirmed via scontrol 10-01)
GPU_PARTITION_ALT="tue.gpu3.q"        # gpuc: 8x L4 (fallback when the superset must be avoided)
GPU_PARTITION_AVOID="tue.gpu1.q"      # gpuA: 8x A30 MIG 1g.6gb (~6 GB, ~230 GB/s)
L4_CONSTRAINT="l4"                    # probe: node FEATURE name for an L4 node
GPU_GRES="gpu:1"

# --- walltimes (probe: `scontrol show partition <p>` MaxTime / `sinfo -o "%P %l"`)
WALLTIME_CPU="01:00:00"
WALLTIME_PILOT="02:00:00"
WALLTIME_CAMPAIGN="14:00:00"
# Slurm sends SIGTERM to the batch shell this many seconds before the walltime
# kill; the sbatch traps it and forwards it to python, whose handler checkpoints
# and exits (see grpo_train --save-every / SIGTERM). Also: `sbatch --signal=B:TERM@N`.
TERM_GRACE_SECONDS=600

# --- run shape (must match campaign.tsv for the campaign array) --------------
GROUP=8
HORIZON=36
CURRICULA="poisson,truncnorm"
LR="1e-5"
BETA_KL="0.02"
ROLLOUT_BATCH=16          # D1 batching knob: prompts per generate() call
SAVE_EVERY=10             # D3: checkpoint every N completed steps (Slurm kills cost <= N steps)
STEPS_PILOT=4
STEPS_CAMPAIGN=300
BENCH_BATCHES="1,4,16,32" # ../scripts/bench_rollout.py chunk sizes (1 = the sequential baseline)

# --- pins that only the probe can settle (name -> current best guess) --------
# Printed by --pins / checked by --check. Keep this list in sync with the
# assignments above: it is what the operator reconciles after pasting the probe.
bw_pin_table() {
  printf '%-22s %-44s %s\n' "PIN" "CURRENT VALUE" "SOURCE"
  printf '%-22s %-44s %s\n' \
    "MODULE_UMBRELLA"   "$MODULE_UMBRELLA"   "probe 09-30: 2024 chosen (2026 tree has no torch yet)" \
    "MODULE_PYTORCH"    "$MODULE_PYTORCH"    "probe 09-30 ok: loads -> py 3.11.3, torch 2.1.2, cu 12.1" \
    "MODULE_PYTHON"     "$MODULE_PYTHON"     "probe 09-30 ok (pip track)" \
    "CPU_PARTITION"     "$CPU_PARTITION"     "probe 09-30 ok" \
    "GPU_PARTITION"     "$GPU_PARTITION"     "probe 09-30 ok (gpuC: 8x L4)" \
    "GPU_PARTITION_ALT" "$GPU_PARTITION_ALT" "probe 09-30 ok (gpuB: 4x L4)" \
    "L4_CONSTRAINT"     "$L4_CONSTRAINT"     "probe 09-30 ok (feature token on gpuB+gpuC)" \
    "WALLTIME_PILOT"    "$WALLTIME_PILOT"    "probe 09-30 ok (DefaultTime 2h)" \
    "WALLTIME_CAMPAIGN" "$WALLTIME_CAMPAIGN" "probe 09-30 ok (MaxTime 5d)" \
    "ENV_MODE"          "$ENV_MODE"          "recommended default" \
    "TUE_USER"          "$TUE_USER"          "probe 09-30: 20234428 (laptop-side pulls)" \
    "ROLLOUT_BATCH"     "$ROLLOUT_BATCH"     "repo default; the pilot measures 1 vs N" \
    "SAVE_EVERY"        "$SAVE_EVERY"        "checkpoint cadence for the campaign" \
    "MODEL_REVISION_17B" "$MODEL_REVISION_17B" "HF api sha 2026-10-07 (W8 ladder rung 2)" \
    "MODEL_REVISION_4B" "$MODEL_REVISION_4B"   "HF api sha 2026-10-07 (W8 ladder rung 3)"
}

# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
bw_have() { command -v "$1" >/dev/null 2>&1; }

bw_module_cmd() {
  # `module` is a shell function defined by the site init; a non-interactive
  # batch shell may not have it. Try the usual init scripts once.
  if command -v module >/dev/null 2>&1; then return 0; fi
  local f
  for f in /etc/profile.d/modules.sh /etc/profile.d/lmod.sh /usr/share/lmod/lmod/init/bash \
           /usr/share/Modules/init/bash; do
    if [ -r "$f" ]; then
      # shellcheck disable=SC1090
      . "$f" && break
    fi
  done
  command -v module >/dev/null 2>&1
}

bw_autodetect() {
  # Best-effort only, and only where a Slurm client exists (login node). NEVER
  # overrides a pin that is already set — settled pins (probe-reconciled) are
  # canonical and `--check` validates them against the host; set a pin to "" to
  # have this fill it from sinfo. Never guesses a *module name* (too many
  # variants) — that stays a pin + --check, and the software modules are
  # hierarchical (the Umbrella gateway must be loaded before PyTorch/Python
  # appear in `module avail` at all).
  bw_have sinfo || return 0
  local feats cand part cap
  if [ -z "${L4_CONSTRAINT}" ]; then
    feats="$(sinfo -N -h -o '%N %f' 2>/dev/null || true)"
    if [ -n "$feats" ]; then
      cand="$(printf '%s\n' "$feats" | awk '{n = split($0, f, " "); for (i = 2; i <= n; i++) {m = split(f[i], a, ","); for (j = 1; j <= m; j++) if (tolower(a[j]) ~ /l4/) {print a[j]; exit}}}')"
      [ -n "$cand" ] && L4_CONSTRAINT="$cand"
    fi
  fi
  if [ -z "${GPU_PARTITION}" ]; then
    part="$(sinfo -h -o '%P %f' 2>/dev/null \
      | awk -v c="$L4_CONSTRAINT" 'index(tolower($0), tolower(c)) {gsub(/\*$/, "", $1); print $1; exit}')"
    [ -n "$part" ] && GPU_PARTITION="$part"
  fi
  # walltime cap of the chosen partition (the *default* is what sinfo reports)
  cap="$(sinfo -h -p "$GPU_PARTITION" -o '%l' 2>/dev/null | head -1 || true)"
  [ -n "$cap" ] && WALLTIME_GPU_CAP="$cap"
  return 0
}

bw_load_env() {
  # module stack + venv + cache env. Everything a compute node needs, in one place.
  bw_module_cmd || true
  if bw_have module; then
    module purge >/dev/null 2>&1 || true
    module load "$MODULE_UMBRELLA" >/dev/null 2>&1 \
      || echo "WARN: module load ${MODULE_UMBRELLA} failed (check the probe output)"
    if [ "$ENV_MODE" = "module" ]; then
      module load "$MODULE_PYTORCH" >/dev/null 2>&1 \
        || echo "WARN: module load ${MODULE_PYTORCH} failed (ENV_MODE=module)"
    else
      module load "$MODULE_PYTHON" >/dev/null 2>&1 \
        || echo "WARN: module load ${MODULE_PYTHON} failed (ENV_MODE=pip)"
    fi
  fi
  if [ -f "${VENV_DIR}/bin/activate" ]; then
    # shellcheck disable=SC1090
    . "${VENV_DIR}/bin/activate"
  fi
  export HF_HOME PIP_CACHE_DIR TORCHINDUCTOR_CACHE_DIR
  export TOKENIZERS_PARALLELISM=false
  # compute nodes have no egress (OPERATOR.md): never let transformers/hf hit
  # the network — any missing-cache error surfaces loudly instead of a silent
  # download stall (AUDIT-1 §6 P3)
  export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
  export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}" 
  # compile-path hardening (D2): belt to the code's braces, so even a stale
  # checkout or a helper process inherits the disabled compile path.
  export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
  mkdir -p "$LOG_DIR" "$HF_HOME" "$PIP_CACHE_DIR" "$BBOX_TMP" 2>/dev/null || true
}

# --------------------------------------------------------------------------- #
# --check: every pin verified against this host; non-zero if anything is unresolved
# --------------------------------------------------------------------------- #
bw_check() {
  local fails=0 warns=0
  say()  { printf '%-9s %s\n' "$1" "$2"; }
  ok()   { say "ok" "$1"; }
  warn() { say "WARN" "$1"; warns=$((warns + 1)); }
  bad()  { say "FAIL" "$1"; fails=$((fails + 1)); }

  echo "== bullwhip/Umbrella config check =="
  bw_pin_table
  echo

  if ! bw_have sinfo; then
    bad "sinfo not found: this is not a Slurm login node (off-cluster check is expected to fail)"
    echo; echo "RESULT: ${fails} fail / ${warns} warn"
    return 1
  fi

  bw_autodetect
  echo "-- auto-detected --"
  ok "GPU_PARTITION (pins win; autodetect fills only empty pins): ${GPU_PARTITION}"
  ok "L4_CONSTRAINT (pins win; autodetect fills only empty pins): ${L4_CONSTRAINT}"
  say "info" "GPU partition default walltime: ${WALLTIME_GPU_CAP:-unknown}"
  echo

  local parts
  parts="$(sinfo -h -o '%P' 2>/dev/null | tr -d '*' || true)"
  for p in "$CPU_PARTITION" "$GPU_PARTITION" "$GPU_PARTITION_ALT"; do
    if printf '%s\n' "$parts" | grep -qx "$p"; then ok "partition exists: $p"
    else bad "partition not visible in sinfo: $p"; fi
  done
  if [ "$GPU_PARTITION" = "$GPU_PARTITION_AVOID" ]; then
    bad "GPU_PARTITION resolves to ${GPU_PARTITION_AVOID} (A30 MIG 1g.6gb: ~6 GB, slower than the desktop 3060 Ti) — see 00-PROBE-AND-SETUP.md"
  fi

  # L4 feature must actually appear on nodes of the chosen GPU partition
  local l4nodes
  l4nodes="$(sinfo -N -h -o '%N %P %f' 2>/dev/null \
    | awk -v c="$L4_CONSTRAINT" -v p="${GPU_PARTITION}" \
      'index(tolower($0), tolower(c)) && index(tolower($2), tolower(p)) {n++} END {print n + 0}')"
  if [ "${l4nodes:-0}" -gt 0 ]; then
    ok "${l4nodes} node(s) in ${GPU_PARTITION} carry the '${L4_CONSTRAINT}' feature"
  else
    bad "no node in ${GPU_PARTITION} advertises the '${L4_CONSTRAINT}' feature — read `sinfo -N -o \"%20N %60f\"` and fix L4_CONSTRAINT (this is the pin the probe exists to settle)"
  fi

  # walltime cap
  local cap_s
  cap_s="$(sinfo -h -o "%P %l" 2>/dev/null | awk -v p="$GPU_PARTITION" 'gsub(/\*$/, "", $1) == p {print $2; exit}')"
  if [ -n "$cap_s" ]; then
    local need=0 have=0
    need=$(bw_secs "$WALLTIME_CAMPAIGN"); have=$(bw_secs "$cap_s")
    if [ "$have" -ge "$need" ]; then ok "campaign walltime ${WALLTIME_CAMPAIGN} fits the ${cap_s} partition default"
    else bad "campaign walltime ${WALLTIME_CAMPAIGN} exceeds the partition walltime ${cap_s} — lower STEPS_CAMPAIGN or WALLTIME_CAMPAIGN (the run resumes, so splitting is fine)"; fi
  else
    warn "could not read the partition walltime (check MaxTime with: scontrol show partition ${GPU_PARTITION})"
  fi

  # modules
  if [ "$ENV_MODE" = "module" ]; then
    if bw_module_cmd; then
      local m
      # Umbrella's software is hierarchical: the gateway module must be loaded
      # before the PyTorch/Python names appear in `module avail` at all.
      module load "$MODULE_UMBRELLA" >/dev/null 2>&1 || true
      for m in "$MODULE_UMBRELLA" "$MODULE_PYTORCH"; do
        if module -t avail "$m" 2>&1 | sed 's/ *(.*)//' | grep -qx "$m"; then ok "module available: $m"
        else bad "module not found: $m (run: module spider PyTorch)"; fi
      done
      if module load "$MODULE_PYTORCH" >/dev/null 2>&1; then ok "module loads: ${MODULE_PYTORCH}"
      else bad "module load ${MODULE_PYTORCH} failed"; fi
      m="$MODULE_PYTHON"
      if module -t avail "$m" 2>&1 | sed 's/ *(.*)//' | grep -qx "$m"; then ok "module available: $m"
      else warn "module not found: $m (needed for ENV_MODE=pip only; check with: module spider Python)"; fi
    else
      bad "no 'module' command on this host — cannot verify the module stack"
    fi
  fi

  # python + weights + venv + scratch + internet
  if bw_have python3; then ok "python3: $(python3 -V 2>&1)"; else bad "python3 not on PATH"; fi
  if [ -d "$SCRATCH_ROOT" ]; then ok "scratch exists: ${SCRATCH_ROOT}"
  else bad "scratch path missing: ${SCRATCH_ROOT} (check myquota / the real scratch mount)"; fi
  if [ -d "$REPO_DIR/.git" ]; then ok "repo staged: ${REPO_DIR}"
  else warn "repo not staged yet — run: bash stage.sh"; fi
  if [ -f "${VENV_DIR}/bin/activate" ]; then ok "venv: ${VENV_DIR}"
  else warn "venv missing — run: bash stage.sh (ENV_MODE=${ENV_MODE})"; fi
  if [ -d "${HF_HOME}/hub" ]; then
    local nw
    nw=$(find "${HF_HOME}/hub" -maxdepth 3 -name '*Qwen3-0.6B*' 2>/dev/null | wc -l)
    if [ "$nw" -gt 0 ]; then ok "weights present under ${HF_HOME} (Qwen3-0.6B)"
    else warn "no Qwen/Qwen3-0.6B under ${HF_HOME} — run: bash stage.sh"; fi
  else
    warn "no HF cache yet at ${HF_HOME} — run: bash stage.sh"
  fi
  if bw_have curl; then
    if curl -sI --max-time 10 https://github.com >/dev/null 2>&1; then ok "internet from this node (github reachable)"
    else warn "no internet from this node: stage.sh must run from a login node with egress"; fi
  fi

  # sbatch headers must agree with the pins (Slurm cannot interpolate them)
  local f hdr
  for f in "${KIT_DIR}"/*.sbatch; do
    [ -e "$f" ] || continue
    hdr="$(grep -E '^#SBATCH --partition=' "$f" | head -1 | cut -d= -f2 | tr -d ' \r')"
    case "$(basename "$f")" in
      cpu-*.sbatch) [ "$hdr" = "$CPU_PARTITION" ] \
        && ok "header matches pins: $(basename "$f") (${hdr})" \
        || bad "header/pin mismatch in $(basename "$f"): --partition=${hdr} vs CPU_PARTITION=${CPU_PARTITION}" ;;
      *) [ "$hdr" = "$GPU_PARTITION" ] \
        && ok "header matches pins: $(basename "$f") (${hdr})" \
        || bad "header/pin mismatch in $(basename "$f"): --partition=${hdr} vs GPU_PARTITION=${GPU_PARTITION}" ;;
    esac
    if grep -qE "^#SBATCH --constraint=.*${GPU_PARTITION_AVOID}|^#SBATCH --partition=.*${GPU_PARTITION_AVOID}" "$f"; then
      bad "$(basename "$f") targets the A30-MIG partition ${GPU_PARTITION_AVOID}"
    fi
  done

  echo
  echo "RESULT: ${fails} fail / ${warns} warn"
  [ "$fails" -eq 0 ]
}

bw_secs() {
  # HH:MM:SS or D-HH:MM:SS -> seconds
  local t="$1" d=0
  case "$t" in *-*) d="${t%%-*}"; t="${t#*-}";; esac
  awk -v d="$d" -v t="$t" 'BEGIN{n=split(t,a,":"); s=a[n]+60*a[n-1]+3600*a[n-2]; print d*86400+s}'
}

bw_print() {
  echo "KIT_DIR=$KIT_DIR"
  bw_pin_table
  echo
  echo "BULLWHIP_ROOT=$BULLWHIP_ROOT"
  echo "REPO_DIR=$REPO_DIR"
  echo "LOG_DIR=$LOG_DIR"
  echo "RESULTS_DIR=$RESULTS_DIR"
  echo "HF_HOME=$HF_HOME"
  echo "VENV_DIR=$VENV_DIR"
  echo "REPO_URL=$REPO_URL"
}

# --------------------------------------------------------------------------- #
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
  case "${1:---help}" in
    --print) bw_print; exit 0 ;;
    --pins)  bw_pin_table; exit 0 ;;
    --check) bw_check; exit $? ;;
    --help|-h) sed -n '2,22p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "usage: bash config.sh [--print|--pins|--check]" >&2; exit 2 ;;
  esac
else
  bw_autodetect || true
  # export the filesystem env so any sourcing script sees it, but never create
  # directories off-cluster (SCRATCH_ROOT does not exist on the hub).
  if [ -d "$SCRATCH_ROOT" ]; then
    export HF_HOME PIP_CACHE_DIR TORCHINDUCTOR_CACHE_DIR
  fi
fi
