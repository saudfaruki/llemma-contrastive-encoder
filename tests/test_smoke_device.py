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
