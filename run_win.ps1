#requires -Version 5.1
<#
  Phase 2 — Windows / NVIDIA CUDA (RTX 4090) one-command training bootstrap.

  Windows counterpart to run.sh: venv -> deps -> model download -> 4-bit smoke
  gate -> full train. Self-contained (no `make`). Does NOT touch the macOS/MPS
  path (Makefile / run.sh / requirements.txt are left exactly as-is).

  Usage (from the repo root):
      powershell -ExecutionPolicy Bypass -File .\run_win.ps1
#>
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "[run-win] Phase 2 Windows/CUDA training bootstrap"

if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
    Write-Error "nvidia-smi not found - this entry point needs an NVIDIA CUDA GPU. On a Mac use 'make smoke' / 'make train'."
    exit 1
}
nvidia-smi -L

# hf_xet (the Xet transfer backend) can stall on Llemma's large fp32 shards on
# this box; force the classic resumable downloader. Quiet the Windows symlink note.
$env:HF_HUB_DISABLE_XET = "1"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"

$py = ".\.venv\Scripts\python.exe"

# 1) venv
if (-not (Test-Path $py)) {
    Write-Host "[run-win] creating .venv"
    python -m venv .venv
}
& $py -m pip install -U pip

# 2) deps: torch from the CUDA 13.0 index FIRST (PyPI's Windows wheel is CPU-only),
#    then the rest from PyPI.
Write-Host "[run-win] installing torch (CUDA 13.0 build)"
& $py -m pip install "torch==2.12.0" --index-url https://download.pytorch.org/whl/cu130
Write-Host "[run-win] installing remaining deps"
& $py -m pip install -r requirements-cuda-win.txt

# quick CUDA sanity
& $py -c "import torch; assert torch.cuda.is_available(), 'CUDA not available to torch'; print('[run-win] CUDA OK:', torch.cuda.get_device_name(0))"

# 3) download Llemma-7B (~27 GB fp32 .bin; resume-safe, xet disabled)
Write-Host "[run-win] downloading model (resume-safe; this is the long part)"
& $py scripts\download_model.py

# 3b) remove orphaned safetensors index if present (Llemma ships one with no
#     shards, which can stop the loader from falling back to the .bin weights).
$hub = Join-Path $env:USERPROFILE ".cache\huggingface\hub"
if (Test-Path $hub) {
    Get-ChildItem -Path $hub -Recurse -Filter "model.safetensors.index.json" -ErrorAction SilentlyContinue | ForEach-Object {
        if (-not (Get-ChildItem -Path $_.Directory.FullName -Filter "*.safetensors" -ErrorAction SilentlyContinue)) {
            Remove-Item $_.FullName -Force
            Write-Host "[run-win] removed orphaned safetensors index: $($_.FullName)"
        }
    }
}

# 4) 4-bit smoke gate (aborts before training on failure)
Write-Host "[run-win] 4-bit smoke test (gate)"
& $py scripts\smoke.py --mode real --device cuda
if ($LASTEXITCODE -ne 0) { Write-Error "[run-win] smoke failed; aborting before training."; exit 2 }

# 5) full training. Invoke as a MODULE: this puts the repo root (not src\) on
#    sys.path, so torch's internal `import queue` resolves to the stdlib instead
#    of src\queue.py (which would cause a partially-initialized-torch circular import).
Write-Host "[run-win] starting full training (configs\train_4090.yaml)"
& $py -m src.train --config configs\train_4090.yaml
if ($LASTEXITCODE -ne 0) { Write-Error "[run-win] training failed."; exit 3 }

Write-Host "[run-win] done. Trained adapter under runs\<timestamp>\final\"
