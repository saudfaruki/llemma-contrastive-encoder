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
