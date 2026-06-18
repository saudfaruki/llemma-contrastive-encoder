# Phase 2 — Contrastive Encoder (Llemma-7B + LoRA + SupCon)

Trains a decoder-as-encoder (Llemma-7B, LoRA, masked-mean pooled, supervised
contrastive) over Lean proof pairs. Two targets: Apple-Silicon/MPS for
smoke/dev, CUDA/4090 for the real run.

## 4090 quickstart (the real training run)

Prerequisites: NVIDIA driver + CUDA, Python 3.10+, git.

```bash
git clone <this-repo> && cd phase2-train
./run.sh
```

`run.sh` creates a venv, installs `requirements-cuda.txt` (incl. bitsandbytes),
downloads Llemma (resume-safe), runs the 4-bit smoke test, and — only if the
smoke passes — starts the full run from `configs/train_4090.yaml`.

Run steps individually if you prefer:
`make setup-cuda` · `make smoke-cuda` · `make train-cuda`.

Tuning knobs live in `configs/train_4090.yaml` — `batch_size` (8 fits 24 GB in
nf4; raise if headroom) and `epochs` (3) are the main ones.

> **Caveat:** the 4-bit/bitsandbytes path is implemented against the standard
> QLoRA recipe but was developed on a Mac with no GPU, so its first real
> verification is `make smoke-cuda` on the box. If it fails at model load,
> the likely culprit is the `bitsandbytes` pin vs the box's CUDA toolkit
> (see `requirements-cuda.txt`).

## Apple-Silicon/MPS (smoke + dev only)

`make setup` · `make smoke-tiny` (no download) · `make smoke` (real Llemma) ·
`make test` (unit suite). MPS is too slow for a full run — it exists to validate
the pipeline, not to train.

## Windows (NVIDIA / RTX 4090, CUDA)

A Windows desktop with an NVIDIA GPU. Self-contained PowerShell bootstrap; it does
**not** use `make` or `run.sh` (those remain the macOS/Linux paths and are untouched).

Prerequisites: NVIDIA driver (CUDA 12.x or 13.x), Python 3.10+ on `PATH`, git.

```powershell
git clone <this-repo>; cd phase2-train
powershell -ExecutionPolicy Bypass -File .\run_win.ps1
```

`run_win.ps1` creates `.venv`, installs `torch` (CUDA 13.0 wheel) +
`requirements-cuda-win.txt` (incl. bitsandbytes), downloads Llemma (resume-safe),
runs the 4-bit smoke gate, and — only if it passes — starts the full run from
`configs/train_4090.yaml`. The trained adapter lands in `runs\<timestamp>\final\`.
Tuning knobs (`batch_size`, `epochs`) are the same `configs/train_4090.yaml` as Linux.

Windows-specific notes (none of these change the macOS/Linux code path):
- **Run training as a module** — `python -m src.train --config configs\train_4090.yaml`,
  not `python src\train.py`. Running the script directly puts `src\` on `sys.path`, so
  torch's internal `import queue` picks up `src/queue.py` and hits a circular import.
- **Xet disabled** — `run_win.ps1` sets `HF_HUB_DISABLE_XET=1`; the Xet transfer backend
  can stall on Llemma's large fp32 shards here. The classic downloader is reliable and
  resumes from partials.
- **Orphaned safetensors index** — Llemma-7B ships `model.safetensors.index.json` with no
  safetensors shards (weights are fp32 `.bin`, ~27 GB); the script removes it so the loader
  uses the `.bin` files.
- **Shared `src/` portability fixes** — UTF-8 file reads (`src/data.py`, `src/train.py`) and
  cloned queue snapshots (`src/queue.py`). These are behaviour-preserving on macOS and
  required for real training (incl. `use_queue`) anywhere.

## Layout

`src/` model/data/loss/queue/train · `scripts/` smoke + download ·
`configs/` run configs · `tests/` pytest suite · `docs/superpowers/` spec + plan.
