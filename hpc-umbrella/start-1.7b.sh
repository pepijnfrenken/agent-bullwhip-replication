#!/usr/bin/env bash
# =============================================================================
# hpc-umbrella/start-1.7b.sh — kick off the wave-8 1.7B ladder (run on the
# CLUSTER LOGIN NODE): bash hpc-umbrella/start-1.7b.sh
#
# Idempotent stage machine (safe to re-run after each stage completes):
#   B1   verify/complete the Qwen3-1.7B snapshot in $B/hf (login-node egress)
#   B2   submit the 1.7B pre-flight (F2 tokenizer + F3 frozen gates, ~1 GPU-h)
#   C    when gates_1.7b.json passes F2+F3, submit 1.7B r1 seed0 (--array=0)
#        at lr 5e-5. Rerun after it finishes for the remaining seeds.
# Stage D (1.7B r2) is intentionally NOT here: on HOLD (arm-dir collision bug,
# ~31 GPU-h does not fit in the 70-h wave cap).
# Budget guard: refuses to submit when bw8-* GPU-h since 2026-10-07 is >= 60.
# =============================================================================
set -euo pipefail

B="${BULLWHIP_ROOT:-/scratch-shared/20234428/bullwhip}"
R="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"    # the repo checkout on the cluster
cd "${R}"
KIT="${B}/kit"
export HF_HOME="${B}/hf"
# sbatch jobs refuse to run stale code: every submit exports the CURRENT repo HEAD. Compute it HERE,
# never paste a SHA from elsewhere (W8-NEXT-SUBMITS.md rule).
export EXPECTED_COMMIT="$(git -C "${R}" rev-parse HEAD)"


# ---- budget guard: bw8-* GPU hours since 2026-10-07 --------------------------
GPH=""
if command -v sacct >/dev/null 2>&1; then
  GPH="$(sacct -S 2026-10-07 -P -X -o JobName,Elapsed,State,Partition 2>/dev/null | awk -F"|" '
    $1 ~ /^bw8-/ {
      n = split($2, t, "[-:]");
      if (n == 3) s = t[1]*3600 + t[2]*60 + t[3];
      else if (n == 4) s = t[1]*86400 + t[2]*3600 + t[3]*60 + t[4];
      else s = 0;
      if ($4 ~ /gpu|GPU/) tot += s;
    }
    END { printf "%.1f", tot/3600 }')"
  echo "[1.7b] wave-8 GPU-h spent: ${GPH:-unknown} (submit cap 60, wave cap 70)"
  if [ -n "${GPH}" ] && awk "BEGIN{exit !(${GPH} >= 60)}"; then
    echo "[1.7b] STOP: at or over the 60 GPU-h submit cap - nothing submitted." >&2
    exit 1
  fi
else
  echo "[1.7b] sacct unavailable - no budget check possible" >&2
fi

# ---- B1: ladder weights, LOGIN node (compute nodes have no egress) -----------
SNAP="${HF_HOME}/models--Qwen--Qwen3-1.7B"
say_b1() { echo "[1.7b] B1: ${*}"; }
say_b1 "checking snapshot under ${HF_HOME}"
if python3 - "$SNAP" <<'PY'
import os, sys
snap = sys.argv[1]
bad = []
if not os.path.isdir(snap):
    print("no snapshot dir"); sys.exit(1)
refs = os.path.join(snap, "refs")
rev_ok = False
if os.path.isdir(refs):
    for fn in os.listdir(refs):
        with open(os.path.join(refs, fn)) as fh:
            if fh.read().strip() == "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e":
                rev_ok = True
if not rev_ok:
    print("no pinned revision ref"); sys.exit(1)
sprites = os.path.join(snap, "snapshots")
blobs = os.path.join(snap, "blobs")
big = [f for f in os.listdir(blobs)] if os.path.isdir(blobs) else []
has_big = False
for f in big:
    p = os.path.join(blobs, f)
    if os.path.isfile(p) and os.path.getsize(p) > 2_500_000_000:
        has_big = True
if not has_big:
    print("snapshot present but no large weight blob above 2.5 GB - incomplete")
    sys.exit(1)
print("snapshot OK (pinned revision + large weight blob present)")
PY
then
  say_b1 "reusing the 1.7B snapshot already on disk"
else
  say_b1 "downloading Qwen/Qwen3-1.7B revision 70d244cc on the login node"
  python3 - "$HF_HOME" <<'PY'
import sys
from huggingface_hub import snapshot_download
p = snapshot_download("Qwen/Qwen3-1.7B",
                      revision="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
                      cache_dir=sys.argv[1])
print("snapshot at", p)
PY
  say_b1 "verifying the weight blob (must exceed 2.5 GB)"
  python3 - "$HF_HOME" <<'PY'
import os, sys, glob
hf = sys.argv[1]
pats = [os.path.join(hf, "models--Qwen--Qwen3-1.7B", "blobs", "*")]
big = [f for pat in pats for f in glob.glob(pat) if os.path.isfile(f) and os.path.getsize(f) > 2_500_000_000]
if not big:
    print("FATAL: no weight blob above 2.5 GB - download incomplete, rerun this script.")
    sys.exit(1)
for f in big:
    print("weight blob:", os.path.basename(f), round(os.path.getsize(f) / 1e9, 2), "GB")
PY
fi

GATES="results/wave8/gates_1.7b.json"

# ---- B2: pre-flight (F2 tokenizer prompt sha + F3 frozen) --------------------
preflight_ok() {
  [ -s "$GATES" ] || return 1
  python3 - "$GATES" <<'PY'
import json, sys
g = json.load(open(sys.argv[1]))
f2 = (g.get("F2") or {}).get("verdict") or {}
f3 = (g.get("F3") or {}).get("verdict") or {}
sys.exit(0 if (f2.get("pass") and f3.get("pass")) else 1)
PY
}
if preflight_ok; then
  echo "[1.7b] gates_1.7b.json F2+F3 PASS - pre-flight already done"
else
  if [ -s "$GATES" ]; then
    echo "[1.7b] gates_1.7b.json exists but F2/F3 did not pass - read it before resubmitting:" >&2
    python3 - "$GATES" <<'PY'
import json, sys
g = json.load(open(sys.argv[1]))
for k in ("F2", "F3"):
    print(k, json.dumps(g.get(k), indent=1))
PY
    exit 1
  fi
  if squeue --me -h -n bw8-preflight 2>/dev/null | grep -q .; then
    echo "[1.7b] pre-flight already queued or running - rerun this script when it completes."
    exit 0
  fi
  if [ ! -d "${B}" ]; then
    echo "[1.7b] FATAL: ${B} unreadable from here - run on a cluster login node." >&2
    exit 1
  fi
  cd "${B}"   # never run stage.sh while standing inside the kit dir (getcwd bug)
  bash "${B}/agent-bullwhip-replication/hpc-umbrella/stage.sh" --skip-ladder 2>/dev/null \
    || bash "${B}/agent-bullwhip-replication/hpc-umbrella/stage.sh"
  cd "${OLDPWD:-$HOME}"
  echo "[1.7b] B2: submitting the 1.7B pre-flight (job ~1 GPU-h)"
  sbatch --export=ALL,W8_RUNG=1.7b "${KIT}/gpu-wave8-preflight.sbatch"
  echo "[1.7b] pre-flight submitted. Rerun this script when it completes; check: squeue --me"
  exit 0
fi

# ---- C: 1.7B ladder, seed 0 first, seeds 1-2 only after it reads clean --------
seed0_clean() {
  # "done" = the arm dir train.sbatch leaves behind has its DONE marker; missing = not started
  python3 - <<'PY'
import glob, os, sys
cand = sorted(glob.glob("results/wave8_r1*1.7b_seed0"))
sys.exit(0 if cand and os.path.isfile(os.path.join(cand[-1], "DONE")) else 1)
PY
}
if [ -z "$(squeue --me -h -n bw8-train 2>/dev/null)" ] && seed0_clean; then
  echo "[1.7b] C: seed0 done and clean - submitting the remaining seeds (array 1-2)"
  sbatch --export=ALL,W8_RUNG=1.7b,W8_ARM=r1,W8_LRS=5e-5,W8_SEEDS=3 \
        --array=1-2 "${KIT}/gpu-wave8-train.sbatch"
  echo "[1.7b] seeds 1-2 submitted."
else
  if squeue --me -h -n bw8-train 2>/dev/null | grep -q .; then
    echo "[1.7b] a bw8-train job is already queued or running - rerun when it completes."
    exit 0
  fi
  echo "[1.7b] C: submitting 1.7B r1 SEED0 ONLY (lr 5e-5, --array=0) - measure s/step first"
  sbatch --export=ALL,W8_RUNG=1.7b,W8_ARM=r1,W8_LRS=5e-5,W8_SEEDS=3 \
        --array=0 "${KIT}/gpu-wave8-train.sbatch"
  echo "[1.7b] seed0 submitted. When it completes cleanly, rerun this script for seeds 1-2."
fi
exit 0
