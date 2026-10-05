#!/usr/bin/env bash
# =============================================================================
# hpc-umbrella/pull_results.sh — laptop/hub-side pull of results back off scratch
# (rsync when both ends have it; tar-over-ssh when the cluster has no rsync).
#
# WHY THIS EXISTS: /scratch-shared is deleted 14 days after the last write and has
# NO backup; home (200 GiB) has no backup either. Nothing on Umbrella is durable.
# Mirror results out after the pilot and after EVERY campaign wave.
#
# Run from the laptop/hub ON THE VPN (hpc.tue.nl is campus/VPN only):
#     bash pull_results.sh --dry-run          # show what would come over
#     bash pull_results.sh                    # -> ./umbrella-results/
#     bash pull_results.sh --dest ~/results-umbrella --user p.frenken
#
# Do not run parallel scp/rsync sessions — SCC explicitly asks for one at a time.
# =============================================================================
set -euo pipefail

KIT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=config.sh
. "${KIT_DIR}/config.sh"

DRY=0; DEST="${KIT_DIR}/umbrella-results"
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY=1 ;;
    --dest) DEST="$2"; shift ;;
    --user) TUE_USER="$2"; shift ;;
    --host) SSH_HOST="$2"; shift ;;
    -h|--help) sed -n '2,16p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

REMOTE="${TUE_USER}@${SSH_HOST}"
RUNROOT="/scratch-shared/${TUE_USER}/bullwhip"
REPO="${RUNROOT}/agent-bullwhip-replication"
OPTS=(-avz --partial --human-readable --info=progress2 --timeout=120)
if [ "$DRY" = 1 ]; then OPTS+=(-n); fi   # not an && list: set -e would abort on it

if ! ssh -o BatchMode=yes -o ConnectTimeout=8 "${REMOTE}" true 2>/dev/null; then
  echo "[pull] cannot reach ${REMOTE} non-interactively." >&2
  echo "       - on the TU/e VPN? (hpc.tue.nl is campus/VPN only)" >&2
  echo "       - key-based login working? test: ssh ${REMOTE} true" >&2
  echo "       - or pass --user <tue-username> (default TUE_USER=${TUE_USER})" >&2
  exit 2
fi

# transport: rsync when both ends have it; otherwise tar-over-ssh (the cluster
# login node has no rsync).
HAVE_RSYNC=0
if command -v rsync >/dev/null 2>&1 \
   && ssh -o BatchMode=yes -o ConnectTimeout=8 "${REMOTE}" 'command -v rsync >/dev/null 2>&1' 2>/dev/null; then
  HAVE_RSYNC=1
fi
if [ "$HAVE_RSYNC" = 1 ]; then
  echo "[pull] transport: rsync"
else
  echo "[pull] transport: tar-over-ssh (no rsync locally or on ${REMOTE})"
fi

# pull <remote-path> <local-dir> — one directory at a time (SCC asks for single sessions)
pull() {
  local rp="$1" ld="$2"
  if [ "$DRY" = 1 ] && [ "$HAVE_RSYNC" = 0 ]; then
    echo "[pull] (dry) tar ${REMOTE}:${rp} -> ${ld}"
    return 0
  fi
  mkdir -p "$ld"
  if [ "$HAVE_RSYNC" = 1 ]; then
    rsync "${OPTS[@]}" "${REMOTE}:${rp}/" "${ld}/"
  else
    ssh "${REMOTE}" "tar -C '${rp}' -cf - ." | tar -C "${ld}" -xf -
  fi
}

echo "[pull] ${REMOTE}:${REPO}/results/  ->  ${DEST}/results/"
pull "${REPO}/results" "${DEST}/results"

# job logs are the evidence when a run misbehaves (SIGTERM checkpoints, OOM kills, module errors)
echo "[pull] ${REMOTE}:${RUNROOT}/logs/  ->  ${DEST}/logs/"
pull "${RUNROOT}/logs" "${DEST}/logs"

# the bench JSONs are small and are the Snellius-decision numbers — keep them separate
# (the tar fallback brings the whole grpo_bench/ dir; it is small by design)
echo "[pull] bench metrics"
pull "${REPO}/results/grpo_bench" "${DEST}/bench"

cat <<EOF

[pull] done -> ${DEST}
  bench JSONs : ${DEST}/bench/bench_<jobid>.json   (per-step wall-clock, µs/generate, tokens/s, peak VRAM)
  pilot runs  : ${DEST}/results/grpo_a0_pilot_<jobid>/
  campaign    : ${DEST}/results/grpo_<config>/     (train_log.jsonl + checkpoint/ + state.json)

Reminders
  * scratch expires 14 days after the last write — pull after the pilot and after each campaign wave
  * a run that was killed by walltime leaves {"event": "sigterm"} as the last line of train_log.jsonl;
    resubmitting the same sbatch command resumes from state.json (no lost work)
  * if the pull reports "No such file" (rsync) or a tar error, check the path with: ssh ${REMOTE} 'ls -la ${RUNROOT}'
EOF
