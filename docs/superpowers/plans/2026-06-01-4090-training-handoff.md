# 4090 Training Handoff Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the validated MPS repo into a "clone + `./run.sh`" handoff for a bare CUDA/4090 box — implementing the stubbed 4-bit QLoRA path, wiring `cuda` as a device, and pushing to a private GitHub remote.

**Architecture:** A thin `run.sh` bootstrap calls CUDA-specific Makefile targets (`setup-cuda` → `smoke-cuda` → `train-cuda`). The 4-bit path is implemented in `src/model.py` with `BitsAndBytesConfig`; CUDA is wired through `train.py` and `smoke.py`. All CUDA-only code is verified by mocked unit tests on the Mac plus the on-box `smoke-cuda` gate. The existing MPS targets and bf16 path are left untouched.

**Tech Stack:** Python 3.13 (Mac) / 3.10+ (box), PyTorch, transformers 5.9, peft 0.19, bitsandbytes (CUDA), pytest, Make, bash.

**Verification reality:** No CUDA on the dev Mac. CUDA logic is unit-tested by mocking `AutoModel.from_pretrained` / `torch.cuda` / `snapshot_download`. The first true end-to-end verification is `make smoke-cuda` on the 4090. Every task below either runs green on the Mac or is explicitly marked box-only.

---

## File Structure

**Create**
- `requirements-cuda.txt` — CUDA dependency set (separate from the MPS `requirements.txt`)
- `configs/train_4090.yaml` — the real-run config (4-bit, cuda, batch/epochs)
- `scripts/download_model.py` — resume-safe, idempotent Llemma prefetch
- `run.sh` — bootstrap entry point
- `README.md` — handoff doc
- `tests/test_model_quant.py`, `tests/test_train_device.py`, `tests/test_smoke_device.py`, `tests/test_download_model.py`, `tests/test_train_4090_config.py`, `tests/test_bootstrap.py` — tests

**Modify**
- `src/model.py` — implement 4-bit loading in `_load_backbone`; `prepare_model_for_kbit_training` in `_apply_lora`
- `src/train.py` — add `cuda` device; `require_mps` → `require_device`; seed CUDA in `set_seed`
- `scripts/smoke.py` — `pick_device` prefers cuda; `--device cuda`; 4-bit in real mode; quant-aware dtype check
- `Makefile` — add `setup-cuda`, `smoke-cuda`, `train-cuda`
- `.gitignore` — add `.pytest_cache/`

---

### Task 1: CUDA requirements file

**Files:**
- Create: `requirements-cuda.txt`

This is a config file (TDD-exempt); it is consumed by `setup-cuda` and can only be installed on the box.

- [ ] **Step 1: Create `requirements-cuda.txt`**

```
# Phase 2 contrastive-encoder training — CUDA / 4090 (Linux) target.
# Companion to requirements.txt (which is the Apple-Silicon/MPS set).
# On Linux, torch resolves to the CUDA wheel automatically.
# NOTE: the bitsandbytes <-> torch <-> CUDA-toolkit pairing is the one thing
# that cannot be verified from the Mac. If `make smoke-cuda` fails at model
# load, pin bitsandbytes to the version matching the box's CUDA (see its README).
torch==2.12.0
transformers==5.9.0
peft==0.19.1
accelerate==1.13.0
bitsandbytes==0.48.0
huggingface_hub==1.17.0
numpy==2.4.6
PyYAML==6.0.3
tqdm==4.67.3
matplotlib==3.10.9
pytest==9.0.3
```

- [ ] **Step 2: Verify it parses as a requirements file**

Run: `.venv/bin/python -c "from pip._internal.req import parse_requirements" 2>/dev/null; grep -c '==' requirements-cuda.txt`
Expected: prints `11` (11 pinned packages).

- [ ] **Step 3: Commit**

```bash
git add requirements-cuda.txt
git commit -m "feat: add CUDA/4090 requirements set (adds bitsandbytes)"
```

---

### Task 2: Implement 4-bit QLoRA loading in model.py

**Files:**
- Modify: `src/model.py` (`_load_backbone` ~lines 54-61; `_apply_lora` ~lines 107-124)
- Test: `tests/test_model_quant.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_model_quant.py
import torch
import transformers
import pytest
from src.model import ModelConfig, _load_backbone, Encoder


class _FakeCfg:
    use_cache = True


class _FakeModel:
    def __init__(self):
        self.config = _FakeCfg()

    def to(self, *a, **k):
        return self


def _patch_automodel(monkeypatch, captured):
    def fake_from_pretrained(backbone_id, **kwargs):
        captured["backbone_id"] = backbone_id
        captured.update(kwargs)
        return _FakeModel()
    monkeypatch.setattr(transformers.AutoModel, "from_pretrained",
                        staticmethod(fake_from_pretrained))


def test_load_backbone_4bit_uses_nf4_double_quant_bf16(monkeypatch):
    captured = {}
    _patch_automodel(monkeypatch, captured)
    _load_backbone(ModelConfig(quantize="4bit"), "cuda")
    qc = captured["quantization_config"]
    assert qc.load_in_4bit is True
    assert qc.bnb_4bit_quant_type == "nf4"
    assert qc.bnb_4bit_use_double_quant is True
    assert qc.bnb_4bit_compute_dtype == torch.bfloat16
    assert captured["device_map"] == {"": 0}


def test_load_backbone_none_has_no_quant_config(monkeypatch):
    captured = {}
    _patch_automodel(monkeypatch, captured)
    _load_backbone(ModelConfig(quantize="none"), "cpu")
    assert "quantization_config" not in captured


def test_encoder_4bit_guard_still_raises_off_cuda():
    # 4-bit is CUDA-only; Encoder must refuse it when CUDA is absent.
    with pytest.raises(ValueError):
        Encoder(ModelConfig(quantize="4bit"), device="cpu")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_model_quant.py -v`
Expected: `test_load_backbone_4bit_uses_nf4_double_quant_bf16` FAILS with `KeyError: 'quantization_config'` (current `_load_backbone` ignores quantize). The other two PASS already (the guard exists; the `none` path is unchanged) — that is fine; only the 4-bit test must fail first.

- [ ] **Step 3: Implement 4-bit loading in `_load_backbone`**

Replace the body of `_load_backbone` (currently lines 54-61):

```python
def _load_backbone(cfg: ModelConfig, device):
    from transformers import AutoModel
    if cfg.quantize == "4bit":
        from transformers import BitsAndBytesConfig
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        try:  # transformers >=5 uses `dtype`; older uses `torch_dtype`
            m = AutoModel.from_pretrained(cfg.backbone_id, quantization_config=bnb,
                                          device_map={"": 0}, dtype=torch.bfloat16)
        except TypeError:
            m = AutoModel.from_pretrained(cfg.backbone_id, quantization_config=bnb,
                                          device_map={"": 0}, torch_dtype=torch.bfloat16)
        m.config.use_cache = False
        return m  # bitsandbytes placed the weights; do NOT call .to(device)
    try:
        m = AutoModel.from_pretrained(cfg.backbone_id, dtype=torch.bfloat16)
    except TypeError:
        m = AutoModel.from_pretrained(cfg.backbone_id, torch_dtype=torch.bfloat16)
    m.config.use_cache = False
    return m.to(device)
```

- [ ] **Step 4: Add `prepare_model_for_kbit_training` in `_apply_lora`**

In `_apply_lora` (the `@staticmethod` starting ~line 107), insert immediately after `from peft import LoraConfig, get_peft_model, TaskType`:

```python
        if cfg.quantize == "4bit":
            from peft import prepare_model_for_kbit_training
            backbone = prepare_model_for_kbit_training(
                backbone, use_gradient_checkpointing=cfg.grad_checkpointing)
```

(The rest of `_apply_lora` is unchanged; `backbone` is then passed to `get_peft_model(backbone, lconf)` as before.)

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_model_quant.py -v`
Expected: all 3 PASS.

- [ ] **Step 6: Commit**

```bash
git add src/model.py tests/test_model_quant.py
git commit -m "feat: implement 4-bit QLoRA backbone loading (nf4 + double-quant)"
```

---

### Task 3: Wire CUDA device into train.py

**Files:**
- Modify: `src/train.py` (`set_seed` ~lines 44-49; `require_mps` ~lines 52-55; `--device` choices ~line 135; call site ~line 141)
- Test: `tests/test_train_device.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_train_device.py
import pytest
import src.train as train_mod
from src.train import require_device


def test_require_device_cuda_unavailable_raises(monkeypatch):
    monkeypatch.setattr(train_mod.torch.cuda, "is_available", lambda: False)
    with pytest.raises(SystemExit):
        require_device("cuda")


def test_require_device_cpu_never_raises():
    require_device("cpu")  # returns None, no raise


def test_set_seed_seeds_cuda_when_available(monkeypatch):
    calls = {}
    monkeypatch.setattr(train_mod.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(train_mod.torch.cuda, "manual_seed_all",
                        lambda s: calls.__setitem__("seed", s))
    train_mod.set_seed(7)
    assert calls["seed"] == 7
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_train_device.py -v`
Expected: `ImportError: cannot import name 'require_device'` (collection error → all fail).

- [ ] **Step 3: Implement — rename/generalize the guard and seed CUDA**

Replace `require_mps` (lines 52-55) with:

```python
def require_device(device: str):
    if device == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("FATAL: device=mps but MPS is unavailable. "
                         "Use a Mac with MPS, or pass --device cpu/cuda.")
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("FATAL: device=cuda but torch.cuda.is_available() is False. "
                         "Check the NVIDIA driver and that torch is the CUDA build.")
```

In `set_seed` (lines 44-49), add after the MPS seeding line:

```python
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
```

In `main`, change the `--device` argument (line 135) to:

```python
    p.add_argument("--device", type=str, choices=["mps", "cpu", "cuda"])
```

And change the call site (line 141) from `require_mps(cfg["device"])` to:

```python
    require_device(cfg["device"])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_train_device.py -v`
Expected: all 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/train.py tests/test_train_device.py
git commit -m "feat: support cuda device in train.py (require_device + cuda seeding)"
```

---

### Task 4: CUDA + 4-bit in smoke.py

**Files:**
- Modify: `scripts/smoke.py` (`pick_device` lines 126-129; new helper; `run_real` config build line 336 + dtype check lines 342-344; `main` device arg/guard lines 396-408)
- Test: `tests/test_smoke_device.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_smoke_device.py
import importlib.util
import pathlib
import sys

_SMOKE = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "smoke.py"


def _load_smoke():
    spec = importlib.util.spec_from_file_location("smoke_mod", _SMOKE)
    m = importlib.util.module_from_spec(spec)
    sys.modules["smoke_mod"] = m
    spec.loader.exec_module(m)
    return m


def test_quant_for_device_4bit_on_cuda():
    m = _load_smoke()
    assert m.quant_for_device("cuda") == "4bit"
    assert m.quant_for_device("mps") == "none"
    assert m.quant_for_device("cpu") == "none"


def test_pick_device_prefers_cuda(monkeypatch):
    m = _load_smoke()
    monkeypatch.setattr(m.torch.cuda, "is_available", lambda: True)
    assert m.pick_device(None) == "cuda"


def test_pick_device_explicit_wins(monkeypatch):
    m = _load_smoke()
    monkeypatch.setattr(m.torch.cuda, "is_available", lambda: True)
    assert m.pick_device("cpu") == "cpu"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_smoke_device.py -v`
Expected: `test_quant_for_device_4bit_on_cuda` FAILS (`AttributeError: module ... has no attribute 'quant_for_device'`); `test_pick_device_prefers_cuda` FAILS (current `pick_device` ignores cuda).

- [ ] **Step 3: Implement — `pick_device`, `quant_for_device`, real-mode 4-bit, guard**

Replace `pick_device` (lines 126-129) with:

```python
def pick_device(preferred: str = None) -> str:
    if preferred:
        return preferred
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def quant_for_device(device: str) -> str:
    # 4-bit QLoRA is the CUDA lever; MPS/CPU stay full bf16.
    return "4bit" if device == "cuda" else "none"
```

In `run_real`, replace the config build (line 336) with:

```python
        cfg = ModelConfig(max_length=1024, quantize=quant_for_device(device))
```

Replace the dtype check block (lines 342-345) with:

```python
        if cfg.quantize == "4bit":
            is_4bit = any(p.dtype == torch.uint8 for p in model.backbone.parameters())
            ck("backbone is 4-bit quantized", is_4bit, "bitsandbytes nf4")
        else:
            bb_dtype = next(model.backbone.parameters()).dtype
            ck("backbone is bf16", bb_dtype == torch.bfloat16, f"dtype={bb_dtype}")
        head_dtype = model.embed_linear.weight.dtype
        ck("heads are fp32", head_dtype == torch.float32, f"dtype={head_dtype}")
```

(Delete the old standalone `bb_dtype`/`head_dtype` lines 342-345 that this replaces.)

In `main`, change the `--device` arg (line 397) to:

```python
    p.add_argument("--device", choices=["mps", "cpu", "cuda"], default=None)
```

And replace the real-mode guard (lines 406-407) with:

```python
        if device == "cuda" and not torch.cuda.is_available():
            raise SystemExit("FATAL: --device cuda but CUDA unavailable.")
        if device == "mps" and not torch.backends.mps.is_available():
            raise SystemExit("FATAL: --device mps but MPS unavailable.")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_smoke_device.py -v`
Expected: all 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/smoke.py tests/test_smoke_device.py
git commit -m "feat: smoke.py runs 4-bit on cuda (pick_device + quant_for_device)"
```

---

### Task 5: Resume-safe model download script

**Files:**
- Create: `scripts/download_model.py`
- Test: `tests/test_download_model.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_download_model.py
import importlib.util
import pathlib
import sys

_DL = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "download_model.py"


def _load_dl():
    spec = importlib.util.spec_from_file_location("dl_mod", _DL)
    m = importlib.util.module_from_spec(spec)
    sys.modules["dl_mod"] = m
    spec.loader.exec_module(m)
    return m


def test_fetch_returns_path_on_success(monkeypatch):
    m = _load_dl()
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda mid: "/cache/llemma")
    assert m.fetch() == "/cache/llemma"


def test_fetch_retries_then_succeeds(monkeypatch):
    m = _load_dl()
    import huggingface_hub
    calls = {"n": 0}

    def flaky(mid):
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("peer closed connection")
        return "/cache/llemma"

    monkeypatch.setattr(huggingface_hub, "snapshot_download", flaky)
    monkeypatch.setattr(m.time, "sleep", lambda s: None)
    assert m.fetch(sleep_s=0) == "/cache/llemma"
    assert calls["n"] == 3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_download_model.py -v`
Expected: FAIL — `FileNotFoundError`/`ModuleNotFoundError` loading the not-yet-created `download_model.py`.

- [ ] **Step 3: Create `scripts/download_model.py`**

```python
"""Resume-safe, idempotent Llemma prefetch for the 4090 bootstrap.

Bakes in the download lessons learned on the dev Mac:
- HF_HUB_DOWNLOAD_TIMEOUT so a stalled socket times out (instead of deadlocking).
- A retry loop so transient connection drops self-heal — snapshot_download resumes
  from the on-disk partial each attempt. Already-cached weights make this a no-op.
Llemma is public, so no token is required; HF_TOKEN is used if present (rate limits).
"""
import os
import time

MODEL_ID = "EleutherAI/llemma_7b"


def fetch(model_id: str = MODEL_ID, max_attempts: int = 300, sleep_s: int = 5) -> str:
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")
    from huggingface_hub import snapshot_download
    for attempt in range(1, max_attempts + 1):
        try:
            path = snapshot_download(model_id)
            print(f"SNAPSHOT_OK {path}", flush=True)
            return path
        except Exception as e:  # noqa: BLE001 — any drop is retryable; resume from partial
            print(f"[attempt {attempt}] download dropped: {e}; resuming in {sleep_s}s",
                  flush=True)
            time.sleep(sleep_s)
    raise SystemExit(f"FATAL: {model_id} did not finish after {max_attempts} attempts")


if __name__ == "__main__":
    fetch()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_download_model.py -v`
Expected: both PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/download_model.py tests/test_download_model.py
git commit -m "feat: resume-safe idempotent model download script"
```

---

### Task 6: 4090 training config

**Files:**
- Create: `configs/train_4090.yaml`
- Test: `tests/test_train_4090_config.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_train_4090_config.py
import argparse
import pathlib
from src.train import load_config

_CFG = pathlib.Path(__file__).resolve().parents[1] / "configs" / "train_4090.yaml"


def test_train_4090_config_merges_over_defaults():
    args = argparse.Namespace(config=str(_CFG))
    cfg = load_config(args)
    assert cfg["device"] == "cuda"
    assert cfg["quantize"] == "4bit"
    assert cfg["max_length"] == 1024
    assert cfg["use_queue"] is True
    assert cfg["batch_size"] == 8
    assert cfg["epochs"] == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_train_4090_config.py -v`
Expected: FAIL — `load_config` finds no file (`os.path.exists` false), so `cfg["device"]` stays `"mps"` → assertion error.

- [ ] **Step 3: Create `configs/train_4090.yaml`**

```yaml
# Real training run on a CUDA / 4090 (24 GB) box. Used by `make train-cuda`.
# batch_size and epochs are the main tuning knobs.
backbone_id: EleutherAI/llemma_7b
device: cuda
quantize: 4bit
max_length: 1024
batch_size: 8
lr: 1.0e-4
weight_decay: 0.01
epochs: 3
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

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_train_4090_config.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add configs/train_4090.yaml tests/test_train_4090_config.py
git commit -m "feat: add 4090 training config (4-bit, cuda, batch 8, 3 epochs)"
```

---

### Task 7: Bootstrap entry point + Makefile CUDA targets

**Files:**
- Create: `run.sh`
- Modify: `Makefile` (`.PHONY` line 5; append targets)
- Test: `tests/test_bootstrap.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_bootstrap.py
import pathlib
import subprocess

_ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_run_sh_syntax_valid():
    r = subprocess.run(["bash", "-n", str(_ROOT / "run.sh")])
    assert r.returncode == 0


def test_run_sh_steps_in_order():
    t = (_ROOT / "run.sh").read_text()
    assert t.index("setup-cuda") < t.index("smoke-cuda") < t.index("train-cuda")


def test_run_sh_requires_nvidia_smi():
    assert "nvidia-smi" in (_ROOT / "run.sh").read_text()


def test_makefile_has_cuda_targets():
    mk = (_ROOT / "Makefile").read_text()
    for tgt in ("setup-cuda:", "smoke-cuda:", "train-cuda:"):
        assert tgt in mk
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_bootstrap.py -v`
Expected: FAIL — `run.sh` does not exist (`bash -n` errors / `read_text` raises) and the Makefile lacks the targets.

- [ ] **Step 3: Create `run.sh`**

```bash
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
```

- [ ] **Step 4: Append CUDA targets to `Makefile`**

Change the `.PHONY` line (line 5) to:

```makefile
.PHONY: setup smoke-tiny smoke train test clean setup-cuda smoke-cuda train-cuda
```

Append at the end of the file:

```makefile

# ---- CUDA / 4090 path (used by ./run.sh) ----
setup-cuda:
	python3 -m venv .venv
	$(PIP) install -U pip
	$(PIP) install -r requirements-cuda.txt
	$(PY) scripts/download_model.py

smoke-cuda:
	$(PY) scripts/smoke.py --mode real --device cuda

train-cuda:
	$(PY) src/train.py --config configs/train_4090.yaml
```

- [ ] **Step 5: Make `run.sh` executable and run tests**

Run: `chmod +x run.sh && .venv/bin/python -m pytest tests/test_bootstrap.py -v`
Expected: all 4 PASS.

- [ ] **Step 6: Commit**

```bash
git add run.sh Makefile tests/test_bootstrap.py
git commit -m "feat: run.sh bootstrap + Makefile cuda targets"
```

---

### Task 8: README + .gitignore

**Files:**
- Create: `README.md`
- Modify: `.gitignore`

Docs/config (TDD-exempt).

- [ ] **Step 1: Add `.pytest_cache/` to `.gitignore`**

Append a line `.pytest_cache/` to `.gitignore`.

- [ ] **Step 2: Create `README.md`**

```markdown
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

## Layout

`src/` model/data/loss/queue/train · `scripts/` smoke + download ·
`configs/` run configs · `tests/` pytest suite · `docs/superpowers/` spec + plan.
```

- [ ] **Step 3: Commit**

```bash
git add README.md .gitignore
git commit -m "docs: README handoff guide; ignore .pytest_cache"
```

---

### Task 9: MPS regression gate (Mac verification)

**Files:** none (verification only)

This is the only on-Mac end-to-end check that the CUDA changes did not regress the MPS path.

- [ ] **Step 1: Run the full unit suite**

Run: `.venv/bin/python -m pytest tests/ -v`
Expected: all tests PASS (the original 31 + the new quant/device/download/config/bootstrap tests). Zero failures.

- [ ] **Step 2: Run the tiny MPS smoke (no download)**

Run: `PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python scripts/smoke.py --mode tiny`
Expected: `[smoke] 10/10 checks passed.` (the bf16/MPS path is untouched by the 4-bit branch).

- [ ] **Step 3: If anything fails, fix before proceeding**

Do not continue to the push until both are green. A failure here means a CUDA edit leaked into the MPS path — fix it and re-run.

---

### Task 10: Initial commit + private GitHub push

**Files:** none (git/gh operations)

The repo currently has only the spec/plan/feature commits from the tasks above; this task pushes everything to a new private remote. Requires one interactive user action (`gh auth login`).

- [ ] **Step 1: Confirm clean tree and review history**

Run: `git status -s && git log --oneline | head -15`
Expected: working tree clean (all task commits in); no untracked source files. The `.venv/`, `runs/`, and HF cache are gitignored.

- [ ] **Step 2: Sanity-check what will be pushed (no weights/venv)**

Run: `git ls-files | grep -E "\.venv/|runs/|\.incomplete" | head; echo "exit=$?"`
Expected: prints nothing (grep exit 1) — confirming no venv/weights are tracked. `git ls-files | wc -l` should be a few dozen source files.

- [ ] **Step 3: Install the GitHub CLI**

Run: `brew install gh`
Expected: `gh` installs; `gh --version` prints a version.

- [ ] **Step 4: USER ACTION — authenticate**

Ask the user to run (interactive browser login):
`gh auth login`
Then verify: `gh auth status`
Expected: "Logged in to github.com as <user>".

- [ ] **Step 5: Create the private repo and push**

Run: `gh repo create phase2-train --private --source=. --remote=origin --push`
Expected: creates the private repo, adds `origin`, pushes the default branch. Prints the repo URL.

- [ ] **Step 6: Verify the remote**

Run: `git remote -v && git log origin/$(git branch --show-current) --oneline | head -3`
Expected: `origin` points to the new GitHub URL; remote log matches local.

---

## Notes for the implementer

- **bitsandbytes is not installed on the Mac and must not be imported at module load** in `src/model.py` — the imports are inside the `quantize == "4bit"` branch precisely so the MPS path and the unit tests never need it. Keep them there.
- `BitsAndBytesConfig` itself imports fine without bitsandbytes (it is a transformers dataclass); only `from_pretrained(..., quantization_config=...)` needs the library, which is why Task 2's tests mock `from_pretrained`.
- Do not re-run the real MPS smoke (`make smoke`) as part of this plan — it is slow and already validated; the tiny smoke + unit suite are the regression gate.
