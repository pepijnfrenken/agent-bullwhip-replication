#!/usr/bin/env bash
# eval-cpu-setup.sh — set up the wave-6 CPU eval environment ON THE HUB (idempotent).
#
# Builds llama.cpp (CPU-only), fetches the official Qwen3-0.6B Q4_K_M GGUF (the SAME
# weights file serves both arms; the trained arm adds --lora at serve time). Log:
# ~/bw6-eval/setup.log.  Run from the repo:  bash hpc-umbrella/eval-cpu-setup.sh
set -euo pipefail
ROOT="$HOME/bw6-eval"
mkdir -p "$ROOT"
exec > >(tee -a "$ROOT/setup.log") 2>&1
echo "=== bw6 eval-cpu setup $(date -u +%FT%TZ) ==="

# 0) cmake for the build (hub has gcc/make but no cmake; get it via uv venv)
if [ ! -x "$ROOT/toolenv/bin/cmake" ]; then
  uv venv "$ROOT/toolenv"
  uv pip install --python "$ROOT/toolenv/bin/python" cmake
fi
CMAKE="$ROOT/toolenv/bin/cmake"
"$CMAKE" --version | head -1

# 1) llama.cpp (shallow) + CPU build of server/cli/quantize
if [ ! -d "$ROOT/llama.cpp/.git" ]; then
  git clone --depth 1 https://github.com/ggml-org/llama.cpp "$ROOT/llama.cpp"
fi
cd "$ROOT/llama.cpp"
if [ ! -x "$ROOT/llama.cpp/build/bin/llama-server" ]; then
  "$CMAKE" -B build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DLLAMA_CURL=OFF
  "$CMAKE" --build build -j"$(nproc)" --target llama-server llama-cli llama-quantize
fi
"$ROOT/llama.cpp/build/bin/llama-server" --version 2>&1 | head -2 || true

# 2) base GGUF for both arms (official build; no conversion needed for the base)
mkdir -p "$ROOT/models"
if [ ! -s "$ROOT/models/Qwen3-0.6B-Q4_K_M.gguf" ]; then
  hf download Qwen/Qwen3-0.6B-GGUF Qwen3-0.6B-Q4_K_M.gguf --local-dir "$ROOT/models"
fi
ls -la "$ROOT/models"
echo "=== DONE $(date -u +%FT%TZ) ==="
