# Wave-8 8B seed-1 legs on Modal A100-80GB (L4 replacement lane)
# Protocol notes: identical invocation to hpc-umbrella kit sbatch (r1 branch):
#   python -m agent_bullwhip.decision_rounds run --arm r1 --seed 1 --lr <lr> \
#     --model Qwen/Qwen3-8B --revision b968826d9c46dd6066d109eabc6255188de91218 \
#     --rung-suffix -8b --rounds 3 --steps-per-round 170 --batch 32 --micro-batch 1 \
#     --max-length 1024 --grad-ckpt 1 --save-every 10 \
#     --processes <cpus> --gates results/wave8/gates_8b.json
# Library stack pinned to TU/e: torch 2.1.2, transformers 4.55.4, peft 0.21.1.
# Gates file = Snellius-frozen gates_8b.json (already in repo, tracked) - model-deterministic.
import modal

app = modal.App("wave8-8b-modal")
REVISION = "b968826d9c46dd6066d109eabc6255188de91218"

REPO_LOCAL = "/home/pino/projects/agent-bullwhip-replication"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.1.2",
        "transformers==4.55.4",
        "peft==0.21.1",
        "accelerate",
        "safetensors",
        "numpy<2",   # torch 2.1.2 era
        "pandas>=2.0",
        "requests>=2.31",
    )
    # HF model cache baked as layer from a one-time volume-backed build is too complex here;
    # instead download at cold start into a Modal Volume, then reuse.
    .add_local_dir(REPO_LOCAL, remote_path="/repo")
)

volume = modal.Volume.from_name("wave8-hf-8b", create_if_missing=True)

# The hub checkout is baked into the image at /repo (code, gates, S_nom, grids);
# training writes land in /hf_home (Volume) since image layers are read-only at runtime.

MINUTES = 60
@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={"/hf_home": volume},
    timeout=6 * MINUTES * 60,
    env={
        "HF_HOME": "/hf_home/hf",
        "TOKENIZERS_PARALLELISM": "false",
        "TORCHDYNAMO_DISABLE": "1",
        "PYTHONUNBUFFERED": "1",
    },
    cpu=8,
    memory=16384,
)
def snapshot_dl():
    """One-time (idempotent): fetch the pinned 8B snapshot into the volume, offline-resolve after."""
    import os
    from huggingface_hub import snapshot_download
    p = snapshot_download(
        "Qwen/Qwen3-8B",
        revision=REVISION,
        local_dir=None,
        cache_dir=os.environ["HF_HOME"] + "/hub",
    )
    print("snapshot at", p)


@app.function(
    image=image,
    volumes={"/hf_home": volume},
    timeout=2 * MINUTES,
)
def seed_volume():
    """Copy the gates file into the volume (run once; idempotent)."""
    import shutil, os
    os.makedirs("/hf_home/results", exist_ok=True)
    src = "/repo/results/wave8/gates_8b.json"
    dst = "/hf_home/gates_8b.json"
    shutil.copyfile(src, dst)
    print("copied", src, "->", dst)

@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={"/hf_home": volume},
    timeout=4 * MINUTES * 60,
    env={
        "HF_HOME": "/hf_home/hf",
        "TOKENIZERS_PARALLELISM": "false",
        "TORCHDYNAMO_DISABLE": "1",
        "PYTHONUNBUFFERED": "1",
    },
    cpu=8,
    memory=24576,
)
def run_leg(lr: str, seed: int, cpus: int = 8, resume: bool = False):
    """Identical invocation to gpu-wave8-train.sbatch r1 branch, 8B rung."""
    import subprocess, os, sys, shutil
    cwd = "/repo"
    cmd = [
        sys.executable, "-m", "agent_bullwhip.decision_rounds", "run",
        "--arm", "r1", "--seed", str(seed), "--lr", lr,
        "--model", "Qwen/Qwen3-8B", "--revision", REVISION,
        "--rung-suffix=-8b",
        "--rounds", "3", "--steps-per-round", "170",
        "--batch", "32", "--micro-batch", "1",
        "--max-length", "1024", "--grad-ckpt", "1",
        "--save-every", "10",
        "--processes", str(cpus),
        "--gates", "/hf_home/gates_8b.json",
        "--results", "/hf_home/results",
    ]
    if resume:
        cmd.append("--resume")
    print("CMD:", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=cwd, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log = proc.stdout[-8000:]
    print(log, flush=True)
    if proc.returncode != 0:
        raise RuntimeError(f"leg failed lr={lr} seed={seed} rc={proc.returncode}")
    # Results are already written directly into the Volume (writable, persistent).
    # Tar instead of zip to dodge archive pre-1980 mtime issues entirely.
    out = f"/hf_home/leg_{lr}_seed{seed}.tar.gz"
    if os.path.isdir("/hf_home/results"):
        subprocess.run(["tar", "czf", out, "-C", "/hf_home", "results"], check=True)
        print("archived to", out, flush=True)
