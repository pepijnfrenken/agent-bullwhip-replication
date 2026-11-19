#!/usr/bin/env bash
# =============================================================================
# hpc-umbrella/push_results.sh — send result artifacts UP from the cluster via git
#
# The rule: results travel through this repo in both directions. `results/` is
# gitignored, so result paths need `git add -f`; this script forces the add,
# commits, and pushes in one step.
#
# Run on the CLUSTER (login node, inside the repo checkout):
#     bash hpc-umbrella/push_results.sh -m "wave-7 eval results (job 398422)" \
#          results/eval results/wave7_logs/wave7-eval_398422.log
#     bash hpc-umbrella/push_results.sh -n -m "msg" results/eval    # preview only
#     bash hpc-umbrella/push_results.sh -m "auto" \
#          --exclude-glob 'results/**/*.safetensors' results            # skip big weights
#
# Checkpoint dirs (`results/**/checkpoint-*/**`) are always excluded. Use
# --exclude-glob (repeatable) for anything else that must not enter git.
#
# Then on the hub:  cd ~/projects/agent-bullwhip-replication && git pull --ff-only
#
# Slurm logs live OUTSIDE the repo ($BULLWHIP_ROOT/logs); copy the ones that
# matter into results/<lane>_logs/ first, then pass them here.
# =============================================================================
set -euo pipefail

MSG=""
DRY=0
PATHS=()
EXCLUDES=()
while [ $# -gt 0 ]; do
  case "$1" in
    -m|--message) MSG="${2:-}"; shift 2 ;;
    -n|--dry-run) DRY=1; shift ;;
    --exclude-glob) EXCLUDES+=(":(exclude,glob)${2:-}"); shift 2 ;;
    -h|--help) sed -n '2,18p' "${BASH_SOURCE[0]}"; exit 0 ;;
    --) shift; PATHS+=("$@"); break ;;
    -*) echo "[push] unknown flag: $1 (try --help)" >&2; exit 2 ;;
    *) PATHS+=("$1"); shift ;;
  esac
done

[ -n "${MSG}" ] || { echo "[push] FATAL: -m '<message>' is required" >&2; exit 2; }
[ "${#PATHS[@]}" -gt 0 ] || { echo "[push] FATAL: give at least one path to add" >&2; exit 2; }

git rev-parse --is-inside-work-tree >/dev/null 2>&1 \
  || { echo "[push] FATAL: not inside a git work tree (run from the repo checkout)" >&2; exit 2; }
ROOT="$(git rev-parse --show-toplevel)"; cd "${ROOT}"

missing=0
for p in "${PATHS[@]}"; do
  if [ ! -e "${p}" ]; then echo "[push] missing path: ${p}" >&2; missing=1; fi
done
[ "${missing}" -eq 0 ] || { echo "[push] refusing: fix the paths above" >&2; exit 2; }

if [ "${DRY}" -eq 1 ]; then
  echo "[push] dry run — would force-add (results/ is gitignored):"
  git add -f -n -- "${PATHS[@]}" ":(exclude,glob)results/**/checkpoint-*/**" ${EXCLUDES[@]+"${EXCLUDES[@]}"}
  echo "[push] would commit: ${MSG}"
  echo "[push] would run: git push"
  exit 0
fi

git add -f -- "${PATHS[@]}" ":(exclude,glob)results/**/checkpoint-*/**" ${EXCLUDES[@]+"${EXCLUDES[@]}"}
if git diff --cached --quiet; then
  echo "[push] nothing staged — the given content is already committed; nothing to push"
  exit 0
fi

# Blob-size guard: GitHub rejects any object over 100 MiB; check before commit, not after.
BIG=0
for sha in $(git diff --cached --name-only | xargs -r git ls-files -s -- 2>/dev/null | awk '{print $2}' | sort -u); do
  sz=$(git cat-file -s "${sha}" 2>/dev/null || echo 0)
  if [ "${sz}" -gt 104857600 ]; then
    echo "[push] refusing: staged blob over 100 MiB (object ${sha}, ${sz} bytes)" >&2
    BIG=1
  fi
done
[ "${BIG}" -eq 0 ] || { echo "[push] drop the oversized file from the path list and retry" >&2; exit 1; }

git commit -m "${MSG}"
git push
echo "[push] done. Hub side: git -C ~/projects/agent-bullwhip-replication pull --ff-only"
