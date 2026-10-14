#!/bin/bash
# =============================================================================
# wave8-next.sh — wave-8 next-submit operator script (docs/waves/W8-OPERATOR-BLOCK.md)
#
# Idempotent: every stage is skipped when its evidence file already exists on
# this cluster, so pasting it twice never double-spends. Stops (exit 1) at the
# first stage that fails its gate instead of continuing into the GPU budget.
#
# Run from the login node:
#   export BULLWHIP_ROOT=/scratch-shared/$TUE_USER/bullwhip
#   bash "$BULLWHIP_ROOT/agent-bullwhip-replication/hpc-umbrella/wave8-next.sh"
#
# The hub pull for the results side is documented in that same block (§4).
# =============================================================================

set -euo pipefail

B="${BULLWHIP_ROOT:-/scratch-shared/$TUE_USER/bullwhip}"
R="$B/agent-bullwhip-replication"
K="$B/kit"
CAP_GH=60          # stop submitting when the wave has spent more than this of the 70 GPU-h cap

die() { echo "STOP: $*" >&2; exit 1; }
note(){ echo "=== $* ==="; }
have(){ [ -e "$1" ]; }

# GPU-h accounting: wave 8 ONLY. WAVE8 §6.5 caps the wave at 70 GPU-h, not the user's whole
# account (my first cut summed every job since Oct 1 = 212.9 GPU-h, which counted wave-6 and
# anything else on the username — that number is irrelevant to this cap).
# bw8-* job names, GPU partitions only (bw8-label runs on tue.cpu3.q with no GPU), since Oct 7
# (the wave's first cluster job: the 0.6B pre-flight).
gh_hash() {
  sacct -S 2026-10-07 -X -P -n -o JobName%24,JobID,Partition,ElapsedRaw 2>/dev/null \
    | awk -F'[,|]' '$1 ~ /^bw8-(train|preflight|eval)/ && $3 ~ /[.]?gpu[.]?/ {print $4}' \
    || true
}
gh_total() {
  gh_hash | awk -F'[-:]' 'NF==4 {s+=$1*86400+$2*3600+$3*60+$4} NF==3 {s+=$1*3600+$2*60+$3} NF==1 {s+=$1} END {printf "%.1f", s/3600}'
}
budget_block() {
  local used; used="$(gh_total)"
  note "wave-8 GPU-h (bw8-* on GPU partitions, since Oct 7): ${used}"
  awk -v u="$used" -v c="$CAP_GH" 'BEGIN { exit !(u+0 <= c+0) }' \
    || die "budget: wave-8 has spent ${used} GPU-h, cap for new submits is ${CAP_GH} — nothing below was submitted"
}

# -----------------------------------------------------------------------------
note "stage 0: pin the cluster repo to origin/master"
git -C "$R" fetch --all --prune
git -C "$R" checkout master
git -C "$R" pull --ff-only
export EXPECTED_COMMIT="$(git -C "$R" rev-parse HEAD)"
echo "EXPECTED_COMMIT=$EXPECTED_COMMIT"
stale="$(git -C "$R" rev-list --count "master..origin/master" 2>/dev/null || echo 0)"
[ "$stale" = "0" ] || die "local master is behind origin/master by $stale commits (pull did not land)"

note "stage 0b: stage the kit (from OUTSIDE the kit dir — stage.sh replaces it)"
bash "$R/hpc-umbrella/stage.sh"
bash "$K/config.sh" --check | tail -1
diff -q "$K/gpu-wave8-train.sbatch" "$R/hpc-umbrella/gpu-wave8-train.sbatch" \
  || die "kit != repo after staging — do not submit"

# -----------------------------------------------------------------------------
note "stage A2 guard: was any 13.5 GPU-h SFT retrain started by a bad resume?"
if compgen -G "$B/logs/wave8-train_*.log" >/dev/null; then
  hit="$(grep -l '"step": 0, "phase": "sft"' "$B"/logs/wave8-train_*.log 2>/dev/null || true)"
  for f in $hit; do
    joblog="${f##*/}"; jid="${joblog#wave8-train_}"; jid="${jid%.log}"
    echo "resume-signature found in $f (phase sft starting from step 0) — verify whether it"
    echo "happened AFTER seed state.json already reached 2,827: that is the 13.5 GPU-h retrain."
    echo "  squeue --me; sacct -j ${jid%%_*} -X -o JobID,State,ElapsedRaw"
    bash -n "$0"; python3 - "$R" <<'PY'
import json, sys, pathlib
root = pathlib.Path(sys.argv[1])
for p in sorted(root.glob("results/wave8_r2sft_seed*/state.json")):
    d = json.load(open(p))
    print(f"  {p.parent.name}: state={json.dumps({k: d.get(k) for k in ('start_step','step','checkpoint_dir') if k in d})[:120]}")
PY
    die "A2 guard tripped on $f — resolve by hand (scancel if the array is alive), then rerun this script"
  done
fi

note "stage A2: R2+RL rounds (W8_ARM=r2 W8_R2_RL=1 W8_R2_LR=lr2e-4, --array=0-2)"
if compgen -G "$R/results/wave8_r2rl_seed*/round_state.json" >/dev/null; then
  ls -d "$R"/results/wave8_r2rl_seed* 2>/dev/null || true
  note "stage A2: SKIPPED — round_state.json already present (done)"
else
  budget_block
  cd "$K"
  export W8_ARM=r2 W8_R2_LR=lr2e-4 W8_R2_RL=1
  sbatch --export=ALL --array=0-2 gpu-wave8-train.sbatch
  note "stage A2: submitted (expected: STOPPED_collapse.json per seed by round 2, ~1.4 GPU-h)"
  note "        a run WITHOUT the stop marker is the interesting result; do NOT resubmit stopped tasks"
fi

# -----------------------------------------------------------------------------
note "stage B1: ladder weights (login node has egress; compute nodes are offline)"
if compgen -G "$B/hf/models--Qwen--Qwen3-1.7B" >/dev/null; then
  note "stage B1: SKIPPED — 1.7B snapshot already downloaded"
else
  bash "$R/hpc-umbrella/stage.sh" --with-ladder
fi

note "stage B2: 1.7B pre-flight (F2 + F3), mandatory before the ladder"
if have "$R/results/wave8/gates_1.7b.json"; then
  ok="$(python3 -c "import json;g=json.load(open('$R/results/wave8/gates_1.7b.json'));print(int((g.get('F2') or {}).get('prompt_sha_equal') is True), int('F3' in g))")"
  if [ "$ok" = "1 1" ]; then
    note "stage B2: SKIPPED — gates_1.7b.json has F2 pass + F3 frozen"
  else
    die "gates_1.7b.json exists but fails (F2pass/F3 = $ok) — read it, do not re-freeze without reason"
  fi
else
  budget_block
  cd "$K"
  export W8_RUNG=1.7b
  sbatch --export=ALL gpu-wave8-preflight.sbatch
  note "stage B2: submitted (walltime 1 h) — verify after with the block's python read; STOP here for today"
  echo "NEXT: rerun this script once gpu-wave8-preflight job COMPLETES; stage C runs only then"
  exit 0
fi

# -----------------------------------------------------------------------------
note "stage C: ladder R1 at 1.7B, ONE task first (--array=0) to measure s/step"
if compgen -G "$R/results/wave8_r1-1.7b*_seed*/train_log.jsonl" >/dev/null; then
  note "stage C: evidence present — check seeds: $(ls -d "$R"/results/wave8_r1-1.7b* 2>/dev/null)"
  seeds="$(ls -d "$R"/results/wave8_r1-1.7b* 2>/dev/null | wc -l)"
  if [ "$seeds" -ge 3 ]; then
    note "stage C: SKIPPED — ladder already has $seeds arm dirs"
  else
    budget_block
    cd "$K"
    export W8_RUNG=1.7b W8_ARM=r1 W8_LRS=5e-5 W8_SEEDS=3
    sbatch --export=ALL --array=0-2 gpu-wave8-train.sbatch
    note "stage C: remaining tasks submitted"
  fi
else
  budget_block
  cd "$K"
  export W8_RUNG=1.7b W8_ARM=r1 W8_LRS=5e-5 W8_SEEDS=3
  sbatch --export=ALL --array=0 gpu-wave8-train.sbatch
  note "stage C: task 0 submitted — after it prints 50 wall deltas (~1.8 h expected), EXTROPOLATE:"
  note "  python3 - \"$R\" <<'PY'"
  note "import json,glob,statistics"
  note "p=sorted(glob.glob(sys.argv[1]+'/results/wave8_r1-1.7b*_seed0/train_log.jsonl'))[-1]"
  note "w=[r['wall'] for r in map(json.loads, open(p))][-50:]"
  note "print('s/step', statistics.median(w), '-> full seed h', 510*statistics.median(w)/3600)"
  note "PY"
  note "if under 8 h, rerun this script and it will submit --array=0-2"
  exit 0
fi
note "all staged stages done — pull results side per W8-OPERATOR-BLOCK.md §4 when jobs drain"
