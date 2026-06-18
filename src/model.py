"""Backbone-agnostic contrastive encoder wrapper.

Architecture (decoder-only Llemma by default):
    backbone (bf16, LoRA) -> masked-mean pool -> embed_linear(hidden->256) -> L2norm  == kept (inference)
                                                              \\-> proj_head(256->256->128) -> L2norm  == proj (train only)

dtype policy: the heavy backbone + LoRA run in bf16 (per the MPS constraint: bf16
not fp16, since the MPS AMP grad scaler is disabled and fp16 underflows). The two
SMALL projection heads and the SupCon loss run in fp32 for numerical stability and
so AdamW updates on the few trainable head params don't vanish into bf16 epsilon.
This is the standard bf16-backbone / fp32-head LoRA recipe; it is what lets the
overfit smoke test drive the loss toward ~0.

quantize: a 'none|4bit' seam is wired through but 4bit is hard-guarded to CUDA
(bitsandbytes). On MPS it stays 'none'; the bitsandbytes path is intentionally
not implemented here.
"""

import dataclasses
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]


@dataclass
class ModelConfig:
    backbone_id: str = "EleutherAI/llemma_7b"
    backbone_type: str = "decoder"          # decoder | encoder
    pooling: str = "mean"                    # mean | cls
    hidden_dim: int = 4096
    embed_dim: int = 256
    proj_dim: int = 128
    lora_targets: list = field(default_factory=lambda: list(DEFAULT_LORA_TARGETS))
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    quantize: str = "none"                   # none | 4bit (4bit is CUDA-only)
    max_length: int = 1024
    grad_checkpointing: bool = True

    @classmethod
    def from_dict(cls, d: dict):
        names = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})

    def to_dict(self):
        return dataclasses.asdict(self)


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
    try:  # transformers >=5 uses `dtype`; older uses `torch_dtype`
        m = AutoModel.from_pretrained(cfg.backbone_id, dtype=torch.bfloat16)
    except TypeError:
        m = AutoModel.from_pretrained(cfg.backbone_id, torch_dtype=torch.bfloat16)
    m.config.use_cache = False
    return m.to(device)


def _load_tokenizer(cfg: ModelConfig):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg.backbone_id)
    if tok.pad_token is None:                 # Llama has no pad token
        tok.pad_token = tok.eos_token
    return tok


class Encoder(nn.Module):
    def __init__(self, cfg: ModelConfig, device="mps", backbone=None, tokenizer=None,
                 _skip_lora: bool = False):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.label_map = None

        if cfg.quantize == "4bit" and not torch.cuda.is_available():
            raise ValueError(
                "quantize=4bit is CUDA-only (bitsandbytes); not available on this device. "
                "Leave quantize='none' here; 4bit is the lever for the 4090.")

        if backbone is None:
            backbone = _load_backbone(cfg, device)
        if tokenizer is None:
            tokenizer = _load_tokenizer(cfg)
        self.tokenizer = tokenizer

        hidden = getattr(getattr(backbone, "config", None), "hidden_size", None) or cfg.hidden_dim
        self.hidden_dim = int(hidden)

        self.backbone = backbone if _skip_lora else self._apply_lora(backbone, cfg)

        # Heads in fp32 (small; stability — see module docstring).
        self.embed_linear = nn.Linear(self.hidden_dim, cfg.embed_dim)
        self.proj_head = nn.Sequential(
            nn.Linear(cfg.embed_dim, cfg.embed_dim),
            nn.ReLU(),
            nn.Linear(cfg.embed_dim, cfg.proj_dim),
        )
        self.embed_linear.to(device)
        self.proj_head.to(device)

    # ---------- setup ----------
    @staticmethod
    def _apply_lora(backbone, cfg: ModelConfig):
        from peft import LoraConfig, get_peft_model, TaskType
        if cfg.quantize == "4bit":
            from peft import prepare_model_for_kbit_training
            backbone = prepare_model_for_kbit_training(
                backbone, use_gradient_checkpointing=cfg.grad_checkpointing)
        lconf = LoraConfig(
            r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
            target_modules=list(cfg.lora_targets), bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
        )
        model = get_peft_model(backbone, lconf)
        model.enable_input_require_grads()          # needed for grad checkpointing + frozen base
        if cfg.grad_checkpointing:
            try:
                model.gradient_checkpointing_enable()
            except Exception:
                pass
        if hasattr(model, "config"):
            model.config.use_cache = False
        return model

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    # ---------- forward paths ----------
    def _features(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        h = out.last_hidden_state                     # [N, L, H] (bf16)
        if self.cfg.pooling == "cls":
            pooled = h[:, 0]
        else:
            mask = attention_mask.unsqueeze(-1).to(h.dtype)
            pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return self.embed_linear(pooled.float())      # -> fp32 256-d (pre-norm)

    def embed(self, input_ids, attention_mask):
        return F.normalize(self._features(input_ids, attention_mask), dim=-1)  # kept

    def project(self, input_ids, attention_mask):
        e = self._features(input_ids, attention_mask)
        return F.normalize(self.proj_head(e), dim=-1)  # proj (train only)

    def forward(self, input_ids, attention_mask, labels=None):
        proj = self.project(input_ids, attention_mask)
        return (proj, labels) if labels is not None else proj

    @torch.no_grad()
    def encode(self, texts, batch_size: int = 16):
        """Inference: list[str] -> [N, embed_dim] L2-normalized kept embeddings."""
        was_training = self.training
        self.eval()
        outs = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i + batch_size]
            enc = self.tokenizer(chunk, padding=True, truncation=True,
                                 max_length=self.cfg.max_length, return_tensors="pt")
            ids = enc["input_ids"].to(self.device)
            mask = enc["attention_mask"].to(self.device)
            outs.append(self.embed(ids, mask))
        self.train(was_training)
        return torch.cat(outs, dim=0)

    # ---------- checkpoint ----------
    def save_checkpoint(self, ckpt_dir: str):
        import os
        os.makedirs(ckpt_dir, exist_ok=True)
        self.backbone.save_pretrained(os.path.join(ckpt_dir, "adapter"))
        torch.save({
            "embed_linear": self.embed_linear.state_dict(),
            "proj_head": self.proj_head.state_dict(),
            "cfg": self.cfg.to_dict(),
            "label_map": self.label_map,
        }, os.path.join(ckpt_dir, "heads.pt"))

    @classmethod
    def load(cls, ckpt_dir: str, device="mps", base=None, base_factory=None,
             tokenizer=None):
        """Reload: reconstruct the base (from id for real models, or via
        base_factory for a random-init tiny backbone), re-attach the saved LoRA
        adapter, and load the head weights."""
        import os
        from peft import PeftModel
        blob = torch.load(os.path.join(ckpt_dir, "heads.pt"), map_location="cpu",
                          weights_only=False)
        cfg = ModelConfig.from_dict(blob["cfg"])
        if base is None:
            base = base_factory() if base_factory is not None else _load_backbone(cfg, device)
        base = base.to(device)
        peft_backbone = PeftModel.from_pretrained(base, os.path.join(ckpt_dir, "adapter"))
        if tokenizer is None:
            tokenizer = _load_tokenizer(cfg)
        enc = cls(cfg, device=device, backbone=peft_backbone, tokenizer=tokenizer,
                  _skip_lora=True)
        enc.embed_linear.load_state_dict(blob["embed_linear"])
        enc.proj_head.load_state_dict(blob["proj_head"])
        enc.embed_linear.to(device)
        enc.proj_head.to(device)
        enc.label_map = blob["label_map"]
        return enc
