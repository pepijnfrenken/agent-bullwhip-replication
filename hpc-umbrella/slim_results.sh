#!/usr/bin/env bash
# =============================================================================
# hpc-umbrella/slim_results.sh — one-shot cleanup: slim the tracked results tree
#
# What it does (idempotent; safe to re-run):
#   1. Untracks heavy ROUND-COLLECTION artifacts that are NOT verdict evidence
#      (labels jsonl/npz, non-argmax states). Files stay on cluster disk; only
#      git tracking is removed (--cached, never --hard).
#   2. Leaves untouched: states_argmax.jsonl, collapse_*, train_log.jsonl,
#      pi_sft.json, state.json, round_state.json, DONE, anchor_table.json,
#      manifests, wave8_val, wave8_logs, wave7 floors/labels manifests.
#   3. Commits + pushes. Hub side: git pull --ff-only gets ~250 MB lighter HEAD
#      (history keeps the old blobs until a rewrite; new pulls are small).
# =============================================================================
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

LIST=()
while IFS= read -r f; do
  case "$f" in
    */round_labels.jsonl) LIST+=("$f");;
    */policy_labels.jsonl) LIST+=("$f");;
    */behavior_labels.jsonl) LIST+=("$f");;
    */behavior_labels.npz) LIST+=("$f");;
    */policy_labels.npz) LIST+=("$f");;
    */round*/states.jsonl) LIST+=("$f");;  # per-round sampled full-states (argmax copies stay)
    results/wave7_labels/policy_states_seed*.jsonl) LIST+=("$f");;
  esac
done < <(git ls-files results/)

if [ ${#LIST[@]} -eq 0 ]; then
  echo "[slim] nothing to untrack — already slim"
else
  echo "[slim] untracking ${#LIST[@]} files (~250 MB) — disk copies stay put"
  xargs -0 -r git rm -q --cached -- < <(printf '%s\0' "${LIST[@]}")
fi

if git diff --cached --quiet; then
  echo "[slim] nothing changed"
else
  git commit -q -m "results: untrack round-collection labels/states (verdict evidence stays; checkpoints never enter git)"
  git push origin master
  echo "[slim] pushed. Hub: git -C ~/projects/agent-bullwhip-replication pull --ff-only"
fi

echo "[slim] tracked results now:"
git ls-files -z results/ | xargs -0 -r du -ch 2>/dev/null | tail -1
