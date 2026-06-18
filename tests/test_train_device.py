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
