# 4090 Training Handoff — Design Spec

**Date:** 2026-06-01
**Goal:** Make the repo a "clone it, run one command, training starts" handoff for a bare NVIDIA 4090 (CUDA/Linux) box. Includes implementing the currently-stubbed 4-bit QLoRA path, wiring CUDA as a device, and pushing the repo to a private GitHub remote.

**Status of the code today:** The MPS pipeline is validated end-to-end by the real smoke test (load → LoRA → SupCon → separation → queue → checkpoint). This spec covers only the additional work to run a real training run on CUDA.

---

## Critical reality check

1. **The 4-bit path is a stub.** `src/model.py` has a guard (`quantize=="4bit" and not torch.cuda.is_available()` → raise) but `_load_backbone` never applies a `BitsAndBytesConfig` — it loads bf16 and `.to(device)`. The 4-bit QLoRA loading must be **implemented**, not just configured.
2. **`cuda` is not a valid device yet.** `train.py`'s `--device` allows only `mps|cpu`, and `require_mps` is MPS-specific. CUDA must be wired in.
3. **None of the CUDA code can be tested on this Mac** (no GPU). It is written against the standard QLoRA recipe; the **first real verification is `make smoke-cuda` on the 4090.** This is why the entry point smoke-tests before the long run. This limitation is intentional and documented in the README.

---

## Architecture

A thin `run.sh` bootstrap backed by Makefile targets. The repo already centers on a Makefile; each step stays independently runnable for remote debugging, while `./run.sh` gives the one-command UX.

```
git clone <repo> && cd phase2-train && ./run.sh
  └─ run.sh:
       1. detect GPU (nvidia-smi present) — else exit with a clear message
       2. make setup       → create .venv, pip install -r requirements-cuda.txt,
                              python scripts/download_model.py (resume-safe)
       3. make smoke-cuda   → scripts/smoke.py --mode real --device cuda  (4-bit, ~2 min)
                              run.sh ABORTS here if the smoke fails
       4. make train        → python -m src.train --config configs/train_4090.yaml
```

## Components

### Create

**`requirements-cuda.txt`** — CUDA dependency set (the existing `requirements.txt` stays the MPS set, untouched). Mirrors the MPS pins but: `torch==2.12.0` (resolves to CUDA wheels on Linux), adds `bitsandbytes==0.48.0`, keeps `accelerate==1.13.0` (needed for `device_map`). Header notes the bitsandbytes/torch pairing may need adjustment to match the box's CUDA toolkit.

**`configs/train_4090.yaml`** — the real-run config:
```yaml
backbone_id: EleutherAI/llemma_7b
device: cuda
quantize: 4bit
max_length: 1024
batch_size: 8          # 7B in nf4 (~4 GB) + grad-ckpt fits comfortably on 24 GB; main tuning knob
lr: 1.0e-4
weight_decay: 0.01
epochs: 3              # tuning knob
temperature: 0.1
use_queue: true
queue_size: 4096
grad_checkpointing: true
warmup_ratio: 0.1
seed: 0
log_every: 10
ckpt_every: 200
out_dir: runs
```

**`scripts/download_model.py`** — pre-fetch Llemma with the resume/retry lessons baked in (`HF_HUB_DOWNLOAD_TIMEOUT=30`, retry loop, prints final path). Idempotent: a no-op if weights are already cached. Llemma is public (downloaded unauthenticated during smoke), so no token is required; the script accepts `HF_TOKEN` from env if present for higher rate limits.

**`run.sh`** — the bootstrap entry point (`set -euo pipefail`, the 4-step flow above, clear logging).

**`README.md`** — the handoff doc: prerequisites (NVIDIA driver + CUDA, Python 3.10+, git), the `clone → ./run.sh` quickstart, the tuning knobs (`batch_size`, `epochs`), the 4-bit-untested-until-the-box caveat, and how to run steps individually.

### Modify

**`src/model.py`** — implement real 4-bit loading in `_load_backbone`:
- When `cfg.quantize == "4bit"` and CUDA available, load with
  ```python
  BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                     bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
  ```
  passed as `quantization_config`, with `device_map={"": 0}`. Do **not** `.to(device)` afterward (bitsandbytes manages placement).
- Else keep the current bf16 + `.to(device)` path unchanged (MPS untouched).
- In `_apply_lora`, when 4-bit, call `prepare_model_for_kbit_training(base)` before `get_peft_model`.

**`src/train.py`** — add `cuda` to the `--device` choices; generalize `require_mps` → a device guard that validates `cuda`/`mps` availability and falls through for `cpu`; ensure `set_seed` seeds CUDA.

**`scripts/smoke.py`** — add a `--device` arg (auto-detect: cuda if available else mps). In real mode on CUDA, build the `ModelConfig` with `quantize="4bit"` so the smoke exercises the actual QLoRA path. The overfit/timing/batch-ceiling/queue checks already take `device` as a parameter.

**`Makefile`** — add `setup`, `smoke-cuda`, `train` targets (MPS `smoke`/`smoke-tiny` targets stay).

**`.gitignore`** — also ignore `.pytest_cache/`.

## Data flow

Unchanged from the validated MPS pipeline: `ProofPairDataset` → `build_dataloader` (2×B views, num_workers tunable on CUDA) → `Encoder.project` (4-bit backbone + LoRA → masked-mean pool → fp32 heads) → `supcon_loss` (+ optional queue) → AdamW on LoRA + head params → checkpoint (LoRA adapter + `heads.pt`). Only the backbone *loading* and device change for CUDA.

## Error handling

- `run.sh` uses `set -euo pipefail`; the smoke step is a hard gate — a non-zero smoke exit aborts before the long run.
- `train.py` already dumps batch + raises on non-finite loss; unchanged.
- `make setup` is re-runnable; the model download is resume-safe and idempotent.

## Testing / verification

- **On this Mac:** the MPS smoke (`make smoke`) and the unit suite (`pytest`, 31 tests) must still pass after the model.py/train.py edits — i.e., the CUDA changes must not regress the MPS path. This is the only verification possible here.
- **On the 4090 (first real verification):** `make smoke-cuda` exercises the 4-bit load + a short overfit; only after it passes does the full run proceed.
- The 4-bit branch is guarded so it never runs on MPS/CPU; the MPS smoke continues to use the bf16 path.

## Git / push

Build everything above, then a single clean initial commit (the repo has zero commits today; `.gitignore` already excludes `.venv/`, `runs/`, caches — so weights and venv are not pushed; the 1.4 MB data jsonl is included). Then install `gh` via Homebrew, the user runs `gh auth login`, and the repo is created **private** and pushed.

## Out of scope

- The full training run itself (happens on the 4090, by the user).
- Multi-GPU / distributed training.
- Any change to the SupCon/queue/data logic, which is already validated.
