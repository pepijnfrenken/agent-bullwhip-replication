#!/usr/bin/env bash
# =============================================================================
# scripts/results_guard.sh — retention guard: cluster -> GitHub -> local mirror -> snapshots
#
# Why this exists: the wave-8 4B/1.7B per-run data was written once to
# /scratch-shared and never pulled back. Scratch is deleted 14 days after the
# last write, so the evidence for the paper's only positive rung no longer
# exists anywhere. This script makes that class of loss impossible to repeat by
# copying results off the cluster on a timer, through two independent channels:
#
#   1. cluster -> git (push_results.sh, the documented provenance channel)
#   2. GitHub -> local mirror clone + a tar snapshot with SHA256SUMS
#
# It never touches the working checkout (~/projects/...), so paper edits and the
# mirror cannot collide. Run it in a loop while a mission is active, or --once
# from cron.
#
# Usage:
#   bash scripts/results_guard.sh            # loop: every INTERVAL until MAX_HOURS
#   bash scripts/results_guard.sh --once     # one cycle (cron mode)
#   bash scripts/results_guard.sh --status   # print state, change nothing
#
# Bounds (override by env):
#   INTERVAL=300      seconds between cycles
#   MAX_HOURS=36      loop stops after this many hours
#   STOP_FILE         touch it to stop the loop early
#   CLUSTER_HOST=tue-cluster
#   SNAP_KEEP=8       number of local snapshots retained
#
# Exit: 0 ok, 3 cluster unreachable (logged; loop keeps trying), 4 drift found.
# =============================================================================
set -uo pipefail

CLUSTER_HOST="${CLUSTER_HOST:-tue-cluster}"
CLUSTER_ROOT="${CLUSTER_ROOT:-/scratch-shared/${TUE_USER:-${CLUSTER_USER:?export TUE_USER or CLUSTER_USER}}/bullwhip}"
CLUSTER_REPO="${CLUSTER_REPO:-${CLUSTER_ROOT}/agent-bullwhip-replication}"
ARCHIVE="${ARCHIVE:-$HOME/bullwhip-archive}"
BACKUP_DIR="${BACKUP_DIR:-$HOME/backups/bullwhip}"
STATE_DIR="${STATE_DIR:-$HOME/.hermes/cache/scratch/bullwhip}"
LOG="${LOG:-${STATE_DIR}/guard.log}"
STOP_FILE="${STOP_FILE:-${STATE_DIR}/guard.stop}"
LAST_SHA_FILE="${STATE_DIR}/guard.last_sha"
INTERVAL="${INTERVAL:-300}"
MAX_HOURS="${MAX_HOURS:-36}"
SNAP_KEEP="${SNAP_KEEP:-8}"
REPO_URL="${REPO_URL:-git@github.com:pepijnfrenken/agent-bullwhip-replication.git}"

MODE="loop"
case "${1:-}" in
  --once)   MODE="once" ;;
  --status) MODE="status" ;;
  -h|--help) sed -n '2,32p' "${BASH_SOURCE[0]}"; exit 0 ;;
  "")       ;;
  *) echo "unknown option: $1" >&2; exit 2 ;;
esac

mkdir -p "${STATE_DIR}" "${BACKUP_DIR}"
log() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "${LOG}"; }

# --- 1. cluster -> git -------------------------------------------------------
cluster_push() {
  ssh -o BatchMode=yes -o ConnectTimeout=25 "${CLUSTER_HOST}" "bash -s" <<'EOS' 2>&1 | tail -4
set -uo pipefail
R=$CLUSTER_ROOT/agent-bullwhip-replication
B=$CLUSTER_ROOT
[ -d "$R/.git" ] || { echo "[guard] no cluster checkout at $R"; exit 0; }
cd "$R" || exit 0
# Stay in step with origin so the mirror never reports a false drift and the
# push below cannot be rejected as non-fast-forward. Job output lives in
# untracked/ignored paths, so a fast-forward never touches a running write.
git fetch --quiet origin 2>/dev/null || echo "[guard] cluster: fetch failed"
git merge --ff-only --quiet origin/master 2>/dev/null \
  || echo "[guard] cluster: cannot fast-forward (local commits present)"
# Slurm logs live outside the repo; copy the wave-8 ones in so they travel too.
mkdir -p results/wave8_logs
cp -p "$B"/logs/wave8-*.log results/wave8_logs/ 2>/dev/null || true
cp -p "$B"/logs/cpu-wave8-label_*.log results/wave8_logs/ 2>/dev/null || true
bash hpc-umbrella/push_results.sh \
  --exclude-glob 'results/**/*.safetensors' \
  --exclude-glob 'results/**/*.bin' \
  --exclude-glob 'results/**/*.pt' \
  --exclude-glob 'results/**/*.npz' \
  -m "auto-mirror $(date -u +%FT%TZ) from $(hostname -s)" results docs/verify docs/waves 2>&1 | tail -3
EOS
}

# --- 2. GitHub -> local mirror ----------------------------------------------
mirror_update() {
  local fresh=0
  if [ ! -d "${ARCHIVE}/.git" ]; then
    log "mirror: cloning ${REPO_URL} -> ${ARCHIVE}"
    git clone --quiet "${REPO_URL}" "${ARCHIVE}" || return 4
    fresh=1
  fi
  git -C "${ARCHIVE}" fetch --quiet --prune origin '+refs/heads/*:refs/remotes/origin/*' || return 4
  local before after
  before="$(git -C "${ARCHIVE}" rev-parse HEAD 2>/dev/null || echo none)"
  git -C "${ARCHIVE}" reset --hard --quiet origin/master || return 4
  after="$(git -C "${ARCHIVE}" rev-parse HEAD 2>/dev/null || echo none)"
  MIRROR_BEFORE="${before}"; MIRROR_AFTER="${after}"
  [ "${fresh}" -eq 1 ] && return 0
  [ "${before}" != "${after}" ]
}

# --- 3. snapshot + checksums -------------------------------------------------
snapshot() {
  local ts sha target
  ts="$(date -u +%Y%m%dT%H%M%SZ)"
  sha="$(git -C "${ARCHIVE}" rev-parse --short HEAD)"
  target="${BACKUP_DIR}/bullwhip-${ts}-${sha}.tar.gz"
  tar -C "${ARCHIVE}" --exclude='*/checkpoint-*' -czf "${target}" results docs/verify docs/waves 2>/dev/null || return 5
  ( cd "${BACKUP_DIR}" && sha256sum "$(basename "${target}")" >> SHA256SUMS )
  log "snapshot: $(basename "${target}") $(du -h "${target}" | cut -f1) sha256 recorded"
  # keep the newest SNAP_KEEP, delete the rest
  ls -1t "${BACKUP_DIR}"/bullwhip-*.tar.gz 2>/dev/null | tail -n +$((SNAP_KEEP + 1)) | while read -r f; do
    rm -f "${f}" && log "snapshot: pruned $(basename "${f}")"
  done
  echo "${sha}" > "${LAST_SHA_FILE}"
}

# --- 4. drift check ----------------------------------------------------------
drift_check() {
  local c_files a_files c_head a_head
  c_head="$(ssh -o BatchMode=yes -o ConnectTimeout=25 "${CLUSTER_HOST}" "git -C ${CLUSTER_REPO} rev-parse --short HEAD 2>/dev/null" 2>/dev/null || echo unreachable)"
  a_head="$(git -C "${ARCHIVE}" rev-parse --short HEAD 2>/dev/null || echo none)"
  c_files="$(ssh -o BatchMode=yes -o ConnectTimeout=25 "${CLUSTER_HOST}" "find ${CLUSTER_REPO}/results/eval -type f 2>/dev/null | wc -l" 2>/dev/null || echo 0)"
  a_files="$(find "${ARCHIVE}/results/eval" -type f 2>/dev/null | wc -l)"
  log "state: cluster_head=${c_head} mirror_head=${a_head} cluster_eval_files=${c_files} mirror_eval_files=${a_files}"
  if [ "${c_head}" != "unreachable" ] && [ "${c_head}" != "${a_head}" ]; then
    log "DRIFT: cluster is at ${c_head}, mirror at ${a_head}"
    return 4
  fi
  return 0
}

cycle() {
  local out rc=0
  out="$(cluster_push)" || true
  printf '%s\n' "${out}" | sed '/^$/d' | grep -v '^\*\*' | while read -r l; do log "push: ${l}"; done
  case "${out}" in
    *"no cluster checkout"*) log "cycle: cluster checkout absent, nothing to push" ;;
  esac
  ssh -o BatchMode=yes -o ConnectTimeout=15 "${CLUSTER_HOST}" true 2>/dev/null || return 3

  if mirror_update; then
    log "mirror: updated ${MIRROR_BEFORE:0:9} -> ${MIRROR_AFTER:0:9}"
    snapshot || rc=5
  else
    log "mirror: already current at $(git -C "${ARCHIVE}" rev-parse --short HEAD 2>/dev/null)"
  fi
  drift_check || rc=$?
  return "${rc}"
}

case "${MODE}" in
  status)
    echo "archive=${ARCHIVE}"
    [ -d "${ARCHIVE}/.git" ] && git -C "${ARCHIVE}" log --oneline -1
    echo "snapshots:"; ls -1t "${BACKUP_DIR}"/bullwhip-*.tar.gz 2>/dev/null | head -5
    echo "log tail:"; tail -5 "${LOG}" 2>/dev/null
    exit 0 ;;
  once)
    cycle; echo "exit=$?"; exit $? ;;
esac

# loop mode
START=$(date +%s)
log "guard: start (interval=${INTERVAL}s max=${MAX_HOURS}h stop=${STOP_FILE})"
while :; do
  [ -f "${STOP_FILE}" ] && { log "guard: stop file present, exiting"; break; }
  elapsed=$(( $(date +%s) - START ))
  [ "${elapsed}" -ge $(( MAX_HOURS * 3600 )) ] && { log "guard: max hours reached, exiting"; break; }
  cycle; log "cycle: rc=$? (elapsed ${elapsed}s)"
  sleep "${INTERVAL}"
done
log "guard: end"
