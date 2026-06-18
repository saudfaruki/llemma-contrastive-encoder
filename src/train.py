"""Contrastive-encoder training loop + CLI.

bf16 throughout the backbone; fp32 heads/loss (see model.py). MPS-only knobs:
PYTORCH_ENABLE_MPS_FALLBACK=1 set in-process, num_workers=0, pin_memory=False.
This module also exposes reusable helpers (set_seed, compute_loss,
build_scheduler, save_loss_curve) so the smoke tests drive the same code path.
"""

import os

# Must be set before any MPS op so unsupported ops fall back to CPU (and warn).
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import csv
import datetime as _dt
import random
import sys
import warnings

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data import build_label_map, ProofPairDataset, build_dataloader  # noqa: E402
from src.loss import supcon_loss  # noqa: E402
from src.model import Encoder, ModelConfig  # noqa: E402
from src.queue import EmbeddingQueue  # noqa: E402

DEFAULTS = dict(
    backbone_id="EleutherAI/llemma_7b", backbone_type="decoder", pooling="mean",
    hidden_dim=4096, embed_dim=256, proj_dim=128,
    lora_targets=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_r=16, lora_alpha=32, lora_dropout=0.05, quantize="none",
    v3_path="data/elementary_nt_proofs_v3.jsonl",
    pairs_path="data/same_theorem_pairs.jsonl", max_length=1024,
    batch_size=8, lr=1e-4, weight_decay=0.01, epochs=1, temperature=0.1,
    use_queue=False, queue_size=4096, grad_checkpointing=True, warmup_ratio=0.1,
    seed=0, log_every=10, ckpt_every=200, out_dir="runs", device="mps",
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def require_device(device: str):
    if device == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("FATAL: device=mps but MPS is unavailable. "
                         "Use a Mac with MPS, or pass --device cpu/cuda.")
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("FATAL: device=cuda but torch.cuda.is_available() is False. "
                         "Check the NVIDIA driver and that torch is the CUDA build.")


def build_scheduler(optimizer, total_steps: int, warmup_ratio: float):
    warmup = max(1, int(total_steps * warmup_ratio))
    try:
        from transformers import get_cosine_schedule_with_warmup
        return get_cosine_schedule_with_warmup(optimizer, warmup, max(total_steps, warmup + 1))
    except Exception:
        def lr_lambda(step):
            if step < warmup:
                return step / max(1, warmup)
            prog = (step - warmup) / max(1, total_steps - warmup)
            return 0.5 * (1.0 + np.cos(np.pi * min(1.0, prog)))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def compute_loss(model: Encoder, batch: dict, temperature: float,
                 queue: EmbeddingQueue = None):
    """Forward 2*B views -> proj -> SupCon (+queue). Enqueues detached proj if a
    queue is given. Returns the scalar loss."""
    device = model.device
    ids = batch["input_ids"].to(device)
    mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    proj = model(ids, mask)
    if queue is not None:
        qz, qlab, n = queue.contents()
        loss = supcon_loss(proj, labels, temperature,
                           queue_z=(qz if n > 0 else None),
                           queue_labels=(qlab if n > 0 else None))
        queue.enqueue(proj.detach(), labels)
    else:
        loss = supcon_loss(proj, labels, temperature)
    return loss


def save_loss_curve(steps, losses, png_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(7, 4))
    plt.plot(steps, losses, lw=1.2)
    plt.xlabel("step")
    plt.ylabel("SupCon loss")
    plt.title("Training loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(png_path, dpi=120)
    plt.close()


def load_config(args):
    cfg = dict(DEFAULTS)
    if args.config and os.path.exists(args.config):
        with open(args.config, encoding="utf-8") as f:
            cfg.update({k: v for k, v in (yaml.safe_load(f) or {}).items()})
    for k, v in vars(args).items():
        if k != "config" and v is not None:
            cfg[k] = v
    return cfg


def main():
    p = argparse.ArgumentParser(description="Contrastive-encoder training")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--backbone_id", type=str)
    p.add_argument("--batch_size", type=int)
    p.add_argument("--max_length", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--weight_decay", type=float)
    p.add_argument("--epochs", type=int)
    p.add_argument("--temperature", type=float)
    p.add_argument("--use_queue", action="store_true", default=None)
    p.add_argument("--queue_size", type=int)
    p.add_argument("--quantize", type=str, choices=["none", "4bit"])
    p.add_argument("--seed", type=int)
    p.add_argument("--log_every", type=int)
    p.add_argument("--ckpt_every", type=int)
    p.add_argument("--out_dir", type=str)
    p.add_argument("--device", type=str, choices=["mps", "cpu", "cuda"])
    args = p.parse_args()

    warnings.filterwarnings("default")  # surface MPS CPU-fallback warnings
    cfg = load_config(args)
    set_seed(cfg["seed"])
    require_device(cfg["device"])

    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = os.path.join(cfg["out_dir"], stamp)
    os.makedirs(out_dir, exist_ok=True)
    print(f"[train] run dir: {out_dir}")
    print(f"[train] device={cfg['device']} bf16 backbone | use_queue={cfg['use_queue']} "
          f"quantize={cfg['quantize']}")

    label_map = build_label_map(cfg["v3_path"], cfg["pairs_path"])
    dataset = ProofPairDataset(cfg["v3_path"], cfg["pairs_path"], label_map)

    mcfg = ModelConfig.from_dict(cfg)
    model = Encoder(mcfg, device=cfg["device"])
    model.label_map = label_map
    model.train()

    dl = build_dataloader(dataset, model.tokenizer, batch_size=cfg["batch_size"],
                          max_length=cfg["max_length"], shuffle=True)
    total_steps = max(1, cfg["epochs"] * len(dl))
    opt = torch.optim.AdamW(model.trainable_parameters(), lr=cfg["lr"],
                            weight_decay=cfg["weight_decay"])
    sched = build_scheduler(opt, total_steps, cfg["warmup_ratio"])

    queue = None
    if cfg["use_queue"]:
        queue = EmbeddingQueue(capacity=cfg["queue_size"], dim=cfg["proj_dim"],
                               device=cfg["device"], dtype=torch.float32)

    csv_path = os.path.join(out_dir, "loss.csv")
    csv_f = open(csv_path, "w", newline="")
    writer = csv.writer(csv_f)
    writer.writerow(["step", "loss", "lr"])

    steps_log, losses_log = [], []
    step = 0
    n_trainable = sum(p.numel() for p in model.trainable_parameters())
    print(f"[train] trainable params: {n_trainable:,} | total steps: {total_steps}")

    for epoch in range(cfg["epochs"]):
        for batch in dl:
            loss = compute_loss(model, batch, cfg["temperature"], queue)
            if not torch.isfinite(loss):
                dump = os.path.join(out_dir, f"naninf_step{step}.pt")
                torch.save({"step": step, "labels": batch["labels"],
                            "input_ids": batch["input_ids"]}, dump)
                csv_f.close()
                raise SystemExit(f"FATAL: non-finite loss at step {step}; dumped {dump}")

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()

            lr_now = sched.get_last_lr()[0]
            writer.writerow([step, float(loss.item()), lr_now])
            if step % cfg["log_every"] == 0:
                steps_log.append(step)
                losses_log.append(float(loss.item()))
                print(f"[train] step {step:>5} | loss {loss.item():.4f} | lr {lr_now:.2e}")
                csv_f.flush()
            if cfg["ckpt_every"] and step > 0 and step % cfg["ckpt_every"] == 0:
                model.save_checkpoint(os.path.join(out_dir, f"ckpt_step{step}"))
            step += 1

    csv_f.close()
    model.save_checkpoint(os.path.join(out_dir, "final"))
    if steps_log:
        save_loss_curve(steps_log, losses_log, os.path.join(out_dir, "loss_curve.png"))
    print(f"[train] done. checkpoint: {os.path.join(out_dir, 'final')}")


if __name__ == "__main__":
    main()
