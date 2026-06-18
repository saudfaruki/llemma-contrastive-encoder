PY := .venv/bin/python
PIP := .venv/bin/pip
MPS := PYTORCH_ENABLE_MPS_FALLBACK=1

.PHONY: setup smoke-tiny smoke train test clean setup-cuda smoke-cuda train-cuda

setup:
	python3 -m venv .venv
	$(PIP) install -U pip
	$(PIP) install -r requirements.txt

# Random-init tiny backbone; runs in seconds; NO model download. Run this FIRST.
smoke-tiny:
	$(MPS) $(PY) scripts/smoke.py --mode tiny

# Real Llemma-7B weights (~14GB download on first run). Overfit + batch-ceiling probe.
smoke:
	$(MPS) $(PY) scripts/smoke.py --mode real

train:
	$(MPS) $(PY) src/train.py --config configs/default.yaml

test:
	$(PY) -m pytest tests/ -v

clean:
	rm -rf __pycache__ src/__pycache__ tests/__pycache__ .pytest_cache

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
