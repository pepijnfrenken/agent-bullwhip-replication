#!/usr/bin/env bash
# =============================================================================
# hpc-umbrella/probe.sh — login-node probe for the bullwhip Wave-6 Umbrella run.
#
# Run ON THE LOGIN NODE (VPN / campus network), then paste the whole file back to
# the hub session. Everything it prints exists to settle one config.sh pin:
# partition names, the L4 node FEATURE, module names, walltime caps, quotas,
# whether the login node has egress for git/HF, and the GPU you get in a job.
#
# Coreutils + module + Slurm only. Writes ~/umbrella-probe-$(date +%F).txt and
# also prints it, so `bash probe.sh | tee` is unnecessary.
#
# Usage:  bash probe.sh
# =============================================================================
set -u

: "${HOME:=/tmp}"
mkdir -p "${HOME}" 2>/dev/null || true
OUT="${HOME}/umbrella-probe-$(date +%F).txt"
# tee must be able to create the file, or the whole probe would print into a broken
# pipe and leave nothing behind (found the hard way with a sandbox $HOME).
if ! : > "${OUT}" 2>/dev/null; then
  OUT="$(pwd)/umbrella-probe-$(date +%F).txt"
  echo "note: \$HOME is not writable; writing the paste-back file to ${OUT}" >&2
fi
KIT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# tee everything to the file and to the terminal (no set -e: a probe must keep going)
exec > >(tee "${OUT}") 2>&1

sec() { printf '\n===== %s =====\n' "$*"; }
# run a command, echo it first (so the paste-back is self-documenting)
run() { printf '\n$ %s\n' "$*"; "$@" || printf '[exit %s]\n' "$?"; }

printf '=============================================================\n'
printf 'bullwhip/Umbrella probe  host=%s  user=%s  date=%s\n' "$(hostname)" "${USER}" "$(date -Is)"
printf 'file: %s\n' "$OUT"
printf '=============================================================\n'

sec "0. kit pins this probe must settle (from config.sh)"
if [ -r "${KIT_DIR}/config.sh" ]; then
  # shellcheck disable=SC1090
  . "${KIT_DIR}/config.sh"
  bw_pin_table
else
  echo "config.sh not found next to probe.sh — probing blind (copy the kit together)"
fi

sec "1. partitions: name, availability, nodes, default walltime, gres"
run sinfo -o "%22P %6a %5D %12l %12T %30G"
run sinfo -s

sec "2. nodes + FEATURES (this is where the L4 constraint name comes from)"
run sinfo -N -o "%20N %4c %10m %6t %70f"

sec "3. feature vocabulary (unique tokens; find the L4 spelling here)"
printf '\n$ sinfo -N -h -o "%%f" | tr "," "\\n" | sort -u\n'
sinfo -N -h -o '%f' 2>/dev/null | tr ',' '\n' | sort -u | sed '/^$/d' || true

sec "4. GPU pools: which partition has what, and what is free right now"
for p in $(sinfo -h -o '%P' 2>/dev/null | tr -d '*' | sort -u); do
  case "$p" in
    *gpu*) run sinfo -p "$p" -o "%22P %6a %5D %10t %20G %12l" ;;
  esac
done
run sinfo -t idle,mix,alloc -o "%22P %5D %10t %30G"

sec "5. walltime CAPS (scontrol is authoritative; sinfo %l is only the default)"
for p in $(sinfo -h -o '%P' 2>/dev/null | tr -d '*' | sort -u); do
  printf '\n$ scontrol show partition %s | grep -E "PartitionName|MaxTime|DefaultTime|MaxNodes|TRES"\n' "$p"
  scontrol show partition "$p" 2>/dev/null | grep -E 'PartitionName|MaxTime|DefaultTime|MaxNodes|TRES' || echo '[not permitted / not found]'
done

sec "6. modules: Umbrella stack, PyTorch builds, Python, CUDA"
run bash -c 'module avail 2>&1 | head -60'
run bash -c 'module -t avail PyTorch 2>&1 | head -40'
run bash -c 'module -t avail Python 2>&1 | head -40'
run bash -c 'module -t avail CUDA 2>&1 | head -20'
run bash -c 'module spider PyTorch 2>&1 | head -80'
run bash -c 'module -t avail Umbrella 2>&1 | head -20'
run bash -c 'which python3; python3 -V; python3 -c "import sys; print(sys.executable)"'

sec "7. quota + filesystem (home 200 GiB no backup; scratch-shared expires 14d)"
run myquota
run df -h /home /scratch-shared 2>/dev/null
run bash -c 'echo "TMPDIR=${TMPDIR:-unset}"; ls -ld /scratch-shared/$USER 2>/dev/null || echo "no scratch-shared dir yet"'
run bash -c 'ls /dataset 2>/dev/null | head -20 || echo "no /dataset"'
run bash -c 'ls ~/.cache/huggingface 2>/dev/null || echo "no HF cache in home"'

sec "8. egress from the login node (needed for git clone + HF download)"
run curl -sI --max-time 15 https://github.com
run curl -sI --max-time 15 https://huggingface.co
run bash -c 'command -v git && git --version'
run bash -c 'command -v curl && command -v rsync && command -v wget'

sec "9. GPU on this node (login nodes usually have none; informative either way)"
run bash -c 'command -v nvidia-smi && nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || echo "no nvidia-smi here (expected on a login node)"'

sec "10. my queue + fairshare view"
run squeue --me
run bash -c 'sinfo -o "%P %D %t %N" | head -20'

sec "11. what to paste back"
cat <<'EOF'
Paste the WHOLE file back to the hub session (it is also at ~/umbrella-probe-<date>.txt).
The hub then reconciles exactly these config.sh pins from section 3/5/6/7/8:
  MODULE_UMBRELLA, MODULE_PYTORCH, MODULE_PYTHON   <- section 6
  CPU_PARTITION, GPU_PARTITION (+ ALT)             <- sections 1/4
  L4_CONSTRAINT                                    <- sections 2/3 (the feature token for an L4 node)
  WALLTIME_PILOT, WALLTIME_CAMPAIGN                <- section 5 (MaxTime, not default)
  scratch path + quota sanity                      <- section 7
  login-node egress (yes/no)                       <- section 8 (decides scp vs git clone in stage.sh)
EOF
printf '\n=== probe complete: %s ===\n' "$(date -Is)"
if [ ! -s "${OUT}" ]; then
  echo "WARNING: ${OUT} is empty — tee could not write it; the terminal output above is all there is" >&2
fi
