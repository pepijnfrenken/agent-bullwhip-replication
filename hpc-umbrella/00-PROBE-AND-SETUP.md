# Umbrella — probe first, then run (verified against supercomputing.tue.nl docs, 2026-09-23)

> **Automated now (2026-09-30):** `probe.sh` runs the probe below and writes a paste-back file;
> `config.sh` is the single pins file (`--print`, `--check`, `--pins`); `stage.sh` does Steps 2-3;
> `gpu-pilot.sbatch` / `gpu-campaign.sbatch` / `cpu-harness.sbatch` source those pins.
> Read `OPERATOR.md` for the run order; this file stays as the verified-facts reference.

## Access
- Web: https://hpc.tue.nl (Open OnDemand) — files, shell, job composer in the browser
- SSH: `ssh <tue-username>@hpc.tue.nl`
- **Campus network or VPN only.** From home you must be on the TU/e VPN.

## Step 1 — probe (run these on the login node, paste the output back)

```bash
sinfo                                   # which partitions you can see
sinfo -N -o "%20N %4c %10m %60f"        # node FEATURES - you need these to target L4 vs a30
sinfo -p tue.gpu.q -o "%P %a %D %t %G"  # GPU pool: what's idle right now
myquota                                 # home 200 GiB / 1M files, scratch
module avail PyTorch 2>&1 | head -30    # which torch builds exist
module spider PyTorch                   # full list incl. CUDA variants
python3 -V
curl -sI https://github.com | head -1   # does the LOGIN node have internet? (needed for git clone + HF download)
```

Decision rule from the output:
- **Target `tue.gpu2.q` (node gpuB: 4x L4) or `tue.gpu3.q` (node gpuC: 8x L4).**
- **Avoid `tue.gpu1.q`** — node gpuA is 8x A30 **MIG 1g.6gb** slices: ~6 GB VRAM and roughly a
  quarter of an A30's bandwidth (~230 GB/s) — *worse than Pino's desktop 3060 Ti* (8 GB, 448 GB/s).

## Step 2 — get the code and weights on
If the login node has internet (see probe):
```bash
mkdir -p /scratch-shared/$USER/bullwhip && cd /scratch-shared/$USER/bullwhip
git clone https://github.com/pepijnfrenken/agent-bullwhip-replication.git
cd agent-bullwhip-replication
export HF_HOME=/scratch-shared/$USER/hf
python3 -m pip install --user huggingface_hub
python3 -c "from huggingface_hub import snapshot_download as d; d('Qwen/Qwen3-0.6B')"
```
If not, upload from the laptop with `scp -r`, or drag/drop in Open OnDemand.

## Step 3 — environment
Two options. **Try the module stack first** — Umbrella ships PyTorch 2.1.2 (foss-2023a, CUDA 12.1.1),
which is a *different* pairing from the desktop's torch 2.5.1 + transformers 5.17.0. Since the parked
bug is suspected to be exactly that version mismatch, the module stack is a free test of the hypothesis.

```bash
module load Umbrella/2024
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

If `peft`/`trl` aren't in the module env, layer them into a venv on top (don't fight the module):
```bash
python3 -m venv --system-site-packages ~/venvs/bullwhip
source ~/venvs/bullwhip/bin/activate
pip install peft trl datasets pytest
```

## Storage rules that will bite you
| FS | Quota | Expiry | Backup |
|---|---|---|---|
| Home `/home/<user>` | 200 GiB, 1M files | none | **none** |
| Scratch-shared `/scratch-shared/<user>` | 8 TiB, 3M files | **deleted 14 days after last modify** | **none** |
| Scratch-node `$TMPDIR` | node-dependent, NVMe | removed at job end / irregularly | none |

- Keep home under ~100 GB; check with `myquota`. SCC's own guidance.
- **Nothing is backed up.** `results/` must be copied back to the laptop/hub before it matters, and
  scratch expires on 14 days idle — so mirror results out as jobs finish.
- Datasets: `/dataset` has common public ML datasets already staged.
- Don't run parallel `scp` sessions (SCC explicitly asks).

## Rules
- Slurm Fairshare; job arrays supported (`--array`) — the campaign should be an array, one config per task.
- Generic `tue.gpu.q` is "freely available to everyone with a TU/e account" — no project needed.
- If you also got `mcs.gpu.q`: 7-day walltime, **max 2 GPUs simultaneously per user**, and you must
  specify cores/RAM explicitly.
