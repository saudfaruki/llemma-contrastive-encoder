#!/usr/bin/env bash
# One-command 4090 training bootstrap: setup -> 4-bit smoke -> full train.
set -euo pipefail
cd "$(dirname "$0")"

echo "[run] phase2 4090 training bootstrap"
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[run] FATAL: nvidia-smi not found — this entry point is for a CUDA/4090 box." >&2
  echo "[run] On Apple Silicon use 'make smoke' / 'make train' instead." >&2
  exit 1
fi
nvidia-smi -L || true

make setup-cuda    # venv + requirements-cuda.txt + download Llemma (resume-safe)
make smoke-cuda    # 4-bit load + short overfit; aborts the run on failure
make train-cuda    # full run, configs/train_4090.yaml
echo "[run] done."
