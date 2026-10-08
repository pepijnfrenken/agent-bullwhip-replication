#!/usr/bin/env bash
# =============================================================================
# diagnose-wave8.sh — one-command failure bundle for a wave-8 job ("bw8-*") on Umbrella.
#
# WHY: the cluster is VPN-only, so the laptop cannot read /scratch-shared. This script
# turns "the job failed" into a single paste-back bundle: exit codes, the log lines that
# matter, and an inventory of every prerequisite the eval checks — enough to name the
# cause without guessing.
#
# Run it on the CLUSTER (login node, where scratch is mounted):
#     bash hpc-umbrella/diagnose-wave8.sh                 # newest wave8 job of any kind
#     bash hpc-umbrella/diagnose-wave8.sh eval            # newest bw8-eval
#     bash hpc-umbrella/diagnose-wave8.sh 12345678        # one job id
#
# Or from the LAPTOP on the VPN (no cluster checkout needed):
#     ssh <TUE_USER>@hpc.tue.nl 'bash -s' < hpc-umbrella/diagnose-wave8.sh eval > wave8-diag.txt 2>&1
#
# Overridable for testing: BULLWHIP_ROOT, REPO_DIR, LOG_DIR.
# Never exits non-zero on a missing file — it prints what it found.
# =============================================================================
set -uo pipefail

SEL="${1:-auto}"                      # auto | eval | train | preflight | <jobid>
BULLWHIP_ROOT="${BULLWHIP_ROOT:-/scratch-shared/${USER}/bullwhip}"
# config.sh (if present) is the single source of pins; it may override the two below.
KIT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd || echo "$BULLWHIP_ROOT/kit")"
if [ -r "${KIT_DIR}/config.sh" ]; then
  # shellcheck disable=SC1090
  . "${KIT_DIR}/config.sh" 2>/dev/null || echo "[warn] config.sh present but failed to source"
fi
REPO_DIR="${REPO_DIR_OVERRIDE:-${REPO_DIR:-${BULLWHIP_ROOT}/agent-bullwhip-replication}}"
LOG_DIR="${LOG_DIR:-${BULLWHIP_ROOT}/logs}"
TS="$(date +%Y%m%d-%H%M%S)"
OUT="${HOME}/wave8-diag-${TS}.txt"

hr() { printf '\n===== %s =====\n' "$*"; }
have() { command -v "$1" >/dev/null 2>&1; }

{
hr "0. context"
echo "date        : $(date -Is)"
echo "host        : $(hostname) ($(whoami))"
echo "kit_dir     : ${KIT_DIR}$( [ -r "${KIT_DIR}/config.sh" ] && echo "" || echo "  <-- no config.sh here (test/remote mode)")"
echo "repo_dir    : ${REPO_DIR}$( [ -d "${REPO_DIR}" ] && echo "" || echo "  <-- MISSING")"
echo "log_dir     : ${LOG_DIR}$( [ -d "${LOG_DIR}" ] && echo "" || echo "  <-- MISSING")"
echo "selector    : ${SEL}"
have sacct && echo "sacct       : yes" || echo "sacct       : NO (not on a login node? run this on the cluster)"
# TU/e slurm may not support --me; prefer it, fall back to -u $USER.
sacct_me() { sacct --me "$@" 2>/dev/null || sacct -u "${USER}" "$@" 2>/dev/null; }
squeue_me() { squeue --me "$@" 2>/dev/null || squeue -u "${USER}" "$@" 2>/dev/null; }

# -----------------------------------------------------------------------------
hr "1. job records (sacct, last 4 days)"
if have sacct; then
  sacct --me --starttime=now-4days --parsable2 \
    --format=JobID,JobName%18,State,ExitCode,Elapsed,Timelimit,MaxRSS,NodeList,End 2>/dev/null \
    | awk -F'|' 'NR==1 || /bw8|wave8/'
else
  echo "(no sacct)"
fi
have squeue && { echo; echo "--- squeue --me ---"; squeue_me -o "%.12i %.18j %.10T %.10M %.6D %R"; }

# -----------------------------------------------------------------------------
hr "2. resolve the job to inspect"
JOBID=""
case "${SEL}" in
  auto|eval|train|preflight)
    [ "${SEL}" = "auto" ] && PAT='bw8|wave8' || PAT="bw8-${SEL}|wave8-${SEL}"
    if have sacct; then
      JOBID="$(sacct_me --starttime=now-4days --parsable2 --format=JobID,JobName%30,State,End \
        | grep -E "${PAT}" | sort -t'|' -k5 | tail -1 | cut -d'|' -f1)"
    fi
    ;;
  *) JOBID="${SEL}" ;;
esac
echo "job id      : ${JOBID:-<none found — pass one explicitly: bash $0 <jobid>}"
if [ -n "${JOBID}" ] && have sacct; then
  echo
  echo "--- sacct -j ${JOBID} (incl. array/batch steps) ---"
  sacct -j "${JOBID}" --parsable2 \
    --format=JobID,JobName%18,State,ExitCode,Elapsed,Timelimit,MaxRSS,ReqMem,NodeList,Start,End,DerivedExitCode 2>/dev/null
  echo
  echo "--- reason (scontrol) ---"
  have scontrol && scontrol show job "${JOBID}" 2>/dev/null | grep -iE "JobState|Reason|ExitCode|RunTime|TimeLimit|Command" || echo "(job no longer in slurm memory — use the log)"
fi

# -----------------------------------------------------------------------------
hr "3. logs present"
if [ -d "${LOG_DIR}" ]; then
  ls -lt --time-style=+%Y-%m-%dT%H:%M "${LOG_DIR}" 2>/dev/null | grep -iE "wave8|bw8" | head -20
  echo "--- newest 5 wave8 logs by mtime ---"
  ls -t "${LOG_DIR}"/*wave8* 2>/dev/null | head -5
else
  echo "(no ${LOG_DIR})"
fi

# -----------------------------------------------------------------------------
hr "4. what the log says (newest matching log)"
LOGF=""
if [ -n "${JOBID}" ]; then
  LOGF="$(ls -t "${LOG_DIR}"/*_${JOBID}*.log "${LOG_DIR}"/*${JOBID}*.log 2>/dev/null | head -1 || true)"
fi
[ -z "${LOGF}" ] && LOGF="$(ls -t "${LOG_DIR}"/*wave8* 2>/dev/null | head -1 || true)"
echo "log file    : ${LOGF:-<none>}"
if [ -n "${LOGF}" ] && [ -r "${LOGF}" ]; then
  echo "size / mtime: $(stat -c '%s bytes, %y' "${LOGF}" 2>/dev/null)"
  echo
  echo "--- first 25 lines (env, HEAD, rung, gates) ---"
  head -25 "${LOGF}"
  echo
  echo "--- every error-shaped line, with 2 lines of context ---"
  grep -n -iE "FATAL|Traceback|refus|missing |STEP FAILED|error:|Error|CUDA|out of memory|OOM|Killed|TIME LIMIT|CANCELLED|No such file|ModuleNotFound|ImportError|KeyError|ValueError|RuntimeError|AssertionError|permission denied|not found|Exception" "${LOGF}" \
    | head -60 || echo "(no error-shaped lines)"
  echo
  echo "--- per-step outcome lines only ---"
  grep -nE "^--- (ok|eval|perm|collapse|STEP FAILED)" "${LOGF}" | head -60 || true
  echo
  echo "--- last 60 lines ---"
  tail -60 "${LOGF}"
else
  echo "(no readable log — if the job died before the first line, check: sbatch submitted from elsewhere (SLURM_SUBMIT_DIR), or --output path not writable)"
fi

# -----------------------------------------------------------------------------
hr "5. prerequisite inventory (what the failed job needed)"
if [ -d "${REPO_DIR}" ] && cd "${REPO_DIR}" 2>/dev/null; then
  REPO_HEAD="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
  DIRTY="$(git status --porcelain 2>/dev/null | wc -l)"; DIRTY="${DIRTY//[^0-9]/}"
  echo "repo HEAD   : ${REPO_HEAD}  (dirty_files=${DIRTY:-0})"

  echo
  echo "--- gates (F0/F1 on the hub; F2/F3 frozen per rung on the cluster) ---"
  for G in results/wave8/gates.json results/wave8/gates_1.7b.json results/wave8/gates_4b.json; do
    if [ -s "${G}" ]; then
      python3 -c "
import json,sys
d=json.load(open('${G}'))
ks=[k for k in d if isinstance(d[k],dict)]
flags=[]
for k in ks:
    v=d[k].get('verdict')
    if isinstance(v,dict) and 'pass' in v: flags.append(f'{k}={\"pass\" if v[\"pass\"] else \"FAIL\"}')
    elif k in ('F0','F1','F2','F3'): flags.append(f'{k}=?')
print('    ${G}: ' + (', '.join(flags) if flags else f'keys={sorted(ks)[:8]}'))
" 2>/dev/null || echo "    ${G}: present, schema unexpected"
    else
      echo "    ${G}: MISSING/empty   <-- eval refuses with 'F3 not frozen'"
    fi
  done
  ls -la results/wave8/ 2>/dev/null | head -15 || echo "    results/wave8/: missing"

  echo
  echo "--- inputs the probes read ---"
  for f in results/wave8/S_nom.jsonl results/wave8/S_nom_oracle.json results/wave8_labels/labels.jsonl; do
    if [ -s "$f" ]; then printf '    %-42s %6s lines\n' "$f" "$(wc -l < "$f")"; else printf '    %-42s MISSING\n' "$f"; fi
  done

  echo
  echo "--- lr selection input (eval runs select-lr unless W8_LR is set) ---"
  if [ -d results/wave8_val ]; then
    echo "    results/wave8_val/ entries: $(ls results/wave8_val 2>/dev/null | wc -l)"
    ls -d results/wave8_val/*/ 2>/dev/null | head -12 | sed 's/^/      /'
    echo "    (select-lr --family r1<SUF> needs dirs named results/wave8_val/r1*_<lr>_seed*_round*)"
  else
    echo "    results/wave8_val/: MISSING  <-- select-lr will fail (KeyError) unless W8_LR is passed"
  fi

  echo
  echo "--- train outputs the eval will look for (state.json + the adapter it names) ---"
  shopt -s nullglob
  found=0
  for D in results/wave8_r1*_round3 results/wave8_r2*_round3 results/wave8_r2sft_seed* results/wave8_r3*_round3; do
    found=1
    if [ -s "${D}/state.json" ]; then
      CD="$(python3 -c "import json;print(json.load(open('${D}/state.json')).get('checkpoint_dir','<no checkpoint_dir key>'))" 2>/dev/null || echo '<unreadable state.json>')"
      if [ -d "${D}/${CD}" ]; then ADP_OK="adapter OK"; else ADP_OK="ADAPTER DIR MISSING"; fi
      STOP=""; [ -f "${D}/STOPPED_collapse.json" ] && STOP="  [STOPPED_collapse.json present]"
      printf '    %-52s checkpoint_dir=%-18s %s%s\n' "${D}" "${CD}" "${ADP_OK}" "${STOP}"
    else
      printf '    %-52s NO state.json  <-- run the train stage for this (arm,lr,seed)\n' "${D}"
    fi
  done
  [ "${found}" = 0 ] && echo "    (no results/wave8_*_round3 dirs at all — R1 never completed a round 3)"
  shopt -u nullglob

  echo
  echo "--- what the failed eval job did produce (results/eval, newest first) ---"
  ls -lt --time-style=+%Y-%m-%dT%H:%M results/eval 2>/dev/null | head -16 || echo "    results/eval/: missing (nothing ran)"
  echo "    summary files: $(ls results/eval/*.summary.json 2>/dev/null | wc -l)"
  echo
  echo "--- wave8 outputs already written ---"
  ls -lt --time-style=+%Y-%m-%dT%H:%M results/wave8/ 2>/dev/null | head -12

  echo
  echo "--- HF cache check (HF_HUB_OFFLINE=1: a missing pinned revision is fatal) ---"
  HFM="${BULLWHIP_ROOT}/hf/hub"
  if [ -d "${HFM}" ]; then
    if ls -d "${HFM}"/models--* >/dev/null 2>&1; then ls -d "${HFM}"/models--* 2>/dev/null | head -10
    else echo "    ${HFM}: exists but no models--* dirs (nothing pulled — HF_HUB_OFFLINE=1 will fail)"; fi
  else echo "    ${HFM}: MISSING"; fi
else
  echo "(no repo at ${REPO_DIR})"
fi

# -----------------------------------------------------------------------------
hr "6. reading the result (map log evidence -> cause)"
cat <<'EOF'
  "FATAL: ... missing (F3 not frozen for this rung)"  -> run gpu-wave8-preflight.sbatch for that W8_RUNG
  "FATAL: missing train output .../state.json"        -> the (arm,lr,seed) dir the eval expects does not exist.
                                                        Most common: eval guessed a DIFFERENT lr tag than the train
                                                        job wrote (check dir names in section 5 vs the log line
                                                        "r1 winner lr tag: ..."), or that seed never reached round 3.
                                                        Fix: pass W8_LR=<the lr you actually trained> (and
                                                        W8_EVAL_SEEDS to the seeds that exist).
  "FATAL: EXPECTED_COMMIT=... but repo HEAD=..."      -> stale checkout on the cluster; git pull, then resubmit.
  "FATAL: config.sh not found in KIT_DIR"             -> submitted from outside the kit dir; cd <kit> && sbatch.
  Traceback + KeyError 'winner'                        -> results/wave8_val/ missing or named differently.
  Traceback + KeyError 'checkpoint_dir'                 -> state.json from an older/failed train schema; re-run the train stage.
  "FATAL: unknown arm" (exit 2)                         -> W8_EVAL_ARMS typo; valid: r1 r2sft r2rl r3a r3b r3c.
  exit 1 with "--- STEP FAILED:" lines                   -> the eval RAN; read the first failing step's traceback
                                                         (usually CUDA OOM, a missing adapter, or a bad --grid).
  State CANCELLED/DUE TO TIME LIMIT                      -> 4 h walltime hit; the per-step lines show how far it got.
  exit 75                                                -> Slurm SIGTERM (train only): resubmit the same array task.
EOF

hr "7. bundle"
BUNDLE="${HOME}/wave8-diag-${TS}.tar.gz"
TMPD="$(mktemp -d)"
{ [ -n "${LOGF}" ] && [ -r "${LOGF}" ] && cp "${LOGF}" "${TMPD}/" 2>/dev/null
  ls -t "${LOG_DIR}"/*wave8* 2>/dev/null | head -3 | xargs -r -I{} cp {} "${TMPD}/" 2>/dev/null
  cp "${OUT}" "${TMPD}/" 2>/dev/null; } || true
tar -czf "${BUNDLE}" -C "${TMPD}" . 2>/dev/null && echo "bundle: ${BUNDLE}" || echo "bundle failed"
rm -rf "${TMPD}"
echo
echo "Paste-back options:"
echo "  1) this text file: ${OUT}"
echo "  2) commit the log into git:  cd ${REPO_DIR} && mkdir -p results/wave8_logs && cp \"${LOGF}\" results/wave8_logs/ && bash hpc-umbrella/push_results.sh -m \"wave-8 diag (job ${JOBID})\" results/wave8_logs"
} 2>&1 | tee "${OUT}"
