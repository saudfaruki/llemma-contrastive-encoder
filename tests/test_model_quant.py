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
