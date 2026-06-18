"""End-to-end smoke harness for the contrastive-encoder pipeline.

Two modes:

  --mode tiny  (default; NO network)  Build a random-init 2-layer Llama, wrap it
      in the real Encoder (LoRA + fp32 heads), and drive the WHOLE pipeline
      (data -> collate -> forward -> SupCon -> queue -> backward -> AdamW ->
      checkpoint round-trip) in a few seconds. Validates mechanics, not learning.

  --mode real  (GATED; downloads Llemma-7B ~14GB)  Confirms bf16/MPS load, overfits
      a handful of theorems toward loss~0, probes the batch-size ceiling at seq
      1024, exercises the queue, and reports captured MPS CPU-fallback warnings.

Each check prints [PASS]/[FAIL]; the process exits non-zero if any check fails.
"""

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import sys
import warnings
import zlib

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data import ProofPairDataset, build_dataloader  # noqa: E402
from src.model import Encoder, ModelConfig  # noqa: E402
from src.queue import EmbeddingQueue  # noqa: E402
from src.train import compute_loss, set_seed  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V3 = os.path.join(_ROOT, "data", "elementary_nt_proofs_v3.jsonl")
PAIRS = os.path.join(_ROOT, "data", "same_theorem_pairs.jsonl")

TINY_SEED = 1234
TINY_VOCAB = 512
TINY_HIDDEN = 128
TINY_MAXLEN = 64


# --------------------------------------------------------------------------- #
# PASS/FAIL accounting
# --------------------------------------------------------------------------- #
class Checks:
    def __init__(self):
        self.results = []

    def __call__(self, name, ok, detail=""):
        ok = bool(ok)
        self.results.append((name, ok))
        tag = "PASS" if ok else "FAIL"
        line = f"  [{tag}] {name}"
        if detail:
            line += f"  ::  {detail}"
        print(line, flush=True)
        return ok

    def summary(self) -> bool:
        n_pass = sum(1 for _n, ok in self.results if ok)
        n_total = len(self.results)
        print(f"\n[smoke] {n_pass}/{n_total} checks passed.")
        return n_pass == n_total


# --------------------------------------------------------------------------- #
# Offline tiny tokenizer + backbone (no download)
# --------------------------------------------------------------------------- #
class TinyTokenizer:
    """Deterministic whitespace/crc32 tokenizer matching the HF __call__ surface.

    crc32 (not Python hash()) so two instances in the same process tokenize a
    string identically -> the checkpoint round-trip is exact.
    """
    pad_token = "<pad>"
    eos_token = "</s>"
    vocab_size = TINY_VOCAB

    def __call__(self, texts, padding=True, truncation=True,
                 max_length=TINY_MAXLEN, return_tensors="pt"):
        if isinstance(texts, str):
            texts = [texts]
        seqs = []
        for t in texts:
            toks = t.split()
            if truncation:
                toks = toks[:max_length]
            ids = [(zlib.crc32(w.encode("utf-8")) % (TINY_VOCAB - 1)) + 1 for w in toks]
            seqs.append(ids or [1])
        length = max(len(s) for s in seqs)
        input_ids = torch.zeros(len(seqs), length, dtype=torch.long)
        attn = torch.zeros(len(seqs), length, dtype=torch.long)
        for i, s in enumerate(seqs):
            input_ids[i, :len(s)] = torch.tensor(s, dtype=torch.long)
            attn[i, :len(s)] = 1
        return {"input_ids": input_ids, "attention_mask": attn}


def build_tiny_backbone():
    """Re-seeded so every call yields byte-identical frozen weights; that's what
    makes the LoRA-only checkpoint reloadable (base reconstructed, not saved)."""
    from transformers import LlamaConfig, LlamaModel
    torch.manual_seed(TINY_SEED)
    lcfg = LlamaConfig(
        vocab_size=TINY_VOCAB, hidden_size=TINY_HIDDEN, intermediate_size=256,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=512,
    )
    m = LlamaModel(lcfg)
    m.config.use_cache = False
    return m.to(torch.bfloat16)


def _tiny_cfg():
    return ModelConfig(
        backbone_id="tiny-random", backbone_type="decoder", pooling="mean",
        hidden_dim=TINY_HIDDEN, embed_dim=256, proj_dim=128,
        lora_targets=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_r=8, lora_alpha=16, lora_dropout=0.0,
        quantize="none", max_length=TINY_MAXLEN, grad_checkpointing=False,
    )


def pick_device(preferred: str = None) -> str:
    if preferred:
        return preferred
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def quant_for_device(device: str) -> str:
    # 4-bit QLoRA is the CUDA lever; MPS/CPU stay full bf16.
    return "4bit" if device == "cuda" else "none"


# --------------------------------------------------------------------------- #
# TINY MODE
# --------------------------------------------------------------------------- #
def run_tiny(device: str) -> bool:
    ck = Checks()
    print(f"[smoke:tiny] device={device} | random-init 2-layer Llama (no download)")
    set_seed(TINY_SEED)
    tok = TinyTokenizer()
    cfg = _tiny_cfg()
    backbone = build_tiny_backbone().to(device)
    model = Encoder(cfg, device=device, backbone=backbone, tokenizer=tok)
    model.train()

    ds = ProofPairDataset(V3, PAIRS)
    dl = build_dataloader(ds, tok, batch_size=4, max_length=TINY_MAXLEN, shuffle=True)
    batch = next(iter(dl))
    ids = batch["input_ids"].to(device)
    mask = batch["attention_mask"].to(device)
    n_rows = ids.shape[0]

    # ---- Check 1: forward shapes + L2 normalization + finite ----
    proj = model.project(ids, mask)
    emb = model.embed(ids, mask)
    proj_ok = tuple(proj.shape) == (n_rows, 128)
    emb_ok = tuple(emb.shape) == (n_rows, 256)
    norms = torch.cat([proj.norm(dim=-1), emb.norm(dim=-1)])
    norm_ok = torch.allclose(norms, torch.ones_like(norms), atol=1e-3)
    finite_ok = torch.isfinite(proj).all().item() and torch.isfinite(emb).all().item()
    ck("forward shapes (proj=[N,128], embed=[N,256])", proj_ok and emb_ok,
       f"proj={tuple(proj.shape)} embed={tuple(emb.shape)}")
    ck("outputs L2-normalized (||v||=1)", norm_ok,
       f"norm range [{norms.min():.4f}, {norms.max():.4f}]")
    ck("outputs finite", finite_ok)

    # ---- Check 2: SupCon finite + queue fill / FIFO / same-label masking ----
    q = EmbeddingQueue(capacity=16, dim=128, device=device, dtype=torch.float32)
    losses = []
    for b in dl:
        loss = compute_loss(model, b, temperature=0.1, queue=q)
        losses.append(float(loss.item()))
        if len(losses) >= 4:
            break
    loss_finite = all(torch.isfinite(torch.tensor(x)) for x in losses)
    ck("SupCon loss finite with queue ON (several batches)", loss_finite,
       f"losses={[round(x, 3) for x in losses]}")

    # Queue fill + FIFO wrap on REAL proj embeddings (capacity 16, 4 rows/batch).
    filled = len(q) == 16
    ck("queue fills + FIFO-caps at capacity", filled, f"len(queue)={len(q)}")

    # Same-label masking path: enqueue a block whose labels duplicate the batch's,
    # then confirm the loss stays finite (queued same-label entries are masked out).
    blab = batch["labels"].to(device)
    dup_z = torch.nn.functional.normalize(torch.randn(blab.shape[0], 128, device=device), dim=-1)
    q.enqueue(dup_z, blab)
    qz, qlab, qn = q.contents()
    overlap = bool((qlab.view(-1, 1).eq(blab.view(1, -1))).any().item())
    from src.loss import supcon_loss
    loss_masked = supcon_loss(model.project(ids, mask), blab, 0.1, queue_z=qz, queue_labels=qlab)
    ck("SupCon finite when queue holds same-label entries (masking path)",
       overlap and torch.isfinite(loss_masked).item(),
       f"queue_n={qn} overlap={overlap} loss={float(loss_masked.detach()):.4f}")

    # ---- Check 3: one backward + AdamW step changes params, stays finite ----
    head_param = model.embed_linear.weight
    before = head_param.detach().clone()
    opt = torch.optim.AdamW(model.trainable_parameters(), lr=1e-2)
    loss = compute_loss(model, batch, temperature=0.1, queue=None)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    grads_finite = all(
        torch.isfinite(p.grad).all().item()
        for p in model.trainable_parameters() if p.grad is not None
    )
    opt.step()
    changed = not torch.allclose(before, head_param.detach())
    params_finite = all(torch.isfinite(p).all().item() for p in model.parameters())
    ck("backward produces finite grads", grads_finite)
    ck("AdamW step updates a trainable head param", changed,
       f"max|delta|={ (head_param.detach() - before).abs().max():.3e}")
    ck("all params finite after step", params_finite)

    # ---- Check 4: checkpoint round-trip (save -> reload -> encode allclose) ----
    ckpt_dir = os.path.join(_ROOT, "runs", "_smoke_tiny_ckpt")
    model.save_checkpoint(ckpt_dir)
    texts = [
        "theorem foo : 1 + 1 = 2 := by norm_num",
        "lemma bar (n : Nat) : n + 0 = n := by simp",
    ]
    e1 = model.encode(texts)
    reloaded = Encoder.load(ckpt_dir, device=device,
                            base_factory=build_tiny_backbone, tokenizer=TinyTokenizer())
    e2 = reloaded.encode(texts)
    roundtrip_ok = e1.shape == e2.shape and torch.allclose(e1, e2, atol=1e-4, rtol=1e-3)
    ck("checkpoint round-trip: reloaded encode() matches", roundtrip_ok,
       f"max|delta|={ (e1 - e2).abs().max():.3e}")

    return ck.summary()


# --------------------------------------------------------------------------- #
# REAL MODE (gated; downloads Llemma-7B)
# --------------------------------------------------------------------------- #
def _overfit(model, device, n_theorems=4, n_views=4, steps=80, lr=1e-3):
    """Freeze a small fixed set of augmented views and drive SupCon to its floor.

    Multi-positive SupCon does NOT bottom out at 0: with k positives per anchor the
    minimum loss is log(k). Here each of n_theorems classes has n_views members, so
    every anchor sees (n_views-1) positives and the loss floor is log(n_views-1).
    Convergence to that floor (not 0) is the overfit success signal.

    Also tracks per-step wall-clock (MPS-synced) and within/across-theorem cosine
    similarity on the KEPT 256-d embedding (the actual inference concept space)."""
    import math
    import time
    from src.augment import augment
    from src.data import _load_jsonl
    from src.loss import supcon_loss

    recs = _load_jsonl(V3)[:n_theorems]
    texts, labels = [], []
    for lbl, rec in enumerate(recs):
        for _ in range(n_views):
            texts.append(augment(rec["full_source"]))
            labels.append(lbl)
    enc = model.tokenizer(texts, padding=True, truncation=True,
                          max_length=256, return_tensors="pt")
    ids = enc["input_ids"].to(device)
    mask = enc["attention_mask"].to(device)
    y = torch.tensor(labels, dtype=torch.long, device=device)

    n = ids.shape[0]
    same = y.view(-1, 1).eq(y.view(1, -1))
    eye = torch.eye(n, dtype=torch.bool, device=device)

    def separation():
        with torch.no_grad():
            e = model.embed(ids, mask).float()      # [N, 256] L2-normalized
        sim = e @ e.t()
        return sim[same & ~eye].mean().item(), sim[~same].mean().item()

    w0, a0 = separation()
    print(f"    [pre-train] cos within={w0:+.4f} across={a0:+.4f} gap={w0 - a0:+.4f}",
          flush=True)

    opt = torch.optim.AdamW(model.trainable_parameters(), lr=lr)
    curve, step_times = [], []
    for s in range(steps):
        if device == "mps":
            torch.mps.synchronize()
        t0 = time.perf_counter()
        proj = model.project(ids, mask)
        loss = supcon_loss(proj, y, temperature=0.1)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if device == "mps":
            torch.mps.synchronize()
        step_times.append(time.perf_counter() - t0)
        curve.append(float(loss.item()))
        if s % 10 == 0 or s == steps - 1:
            w, a = separation()
            print(f"    step {s:>3} | loss {curve[-1]:.4f} | cos within={w:+.4f} "
                  f"across={a:+.4f} gap={w - a:+.4f} | {step_times[-1] * 1000:.0f} ms/step",
                  flush=True)

    warm = step_times[1:] if len(step_times) > 1 else step_times  # drop step-0 warmup
    mean_ms = 1000.0 * sum(warm) / len(warm)
    wN, aN = separation()
    return {"curve": curve, "mean_ms": mean_ms, "within": wN, "across": aN,
            "floor": math.log(n_views - 1)}


def _probe_batch_ceiling(model, device, seq_len=1024, sizes=None):
    # macOS/MPS and CPU keep the original sweep. CUDA on Windows (WDDM) silently
    # spills VRAM into system RAM instead of raising OOM at large batches, which
    # turns this probe into a multi-hour hang; cap CUDA to sizes that comfortably
    # fit 24 GB so every attempt either succeeds or OOMs cleanly.
    if sizes is None:
        sizes = (8, 16) if device == "cuda" else (8, 16, 24, 32, 48, 64)
    vocab = int(getattr(getattr(model.backbone, "config", None), "vocab_size", 32000))
    max_ok = 0
    for bs in sizes:
        try:
            ids = torch.randint(1, vocab, (bs, seq_len), device=device)
            mask = torch.ones(bs, seq_len, dtype=torch.long, device=device)
            proj = model.project(ids, mask)
            proj.sum().backward()
            model.zero_grad(set_to_none=True)
            del ids, mask, proj
            if device == "mps":
                torch.mps.empty_cache()
            elif device == "cuda":
                torch.cuda.empty_cache()
            max_ok = bs
            print(f"    bs={bs:>3} @ seq{seq_len}: OK", flush=True)
        except RuntimeError as e:
            if device == "cuda":
                torch.cuda.empty_cache()
            print(f"    bs={bs:>3} @ seq{seq_len}: FAILED ({str(e)[:80]})", flush=True)
            break
    return max_ok


def run_real(device: str) -> bool:
    ck = Checks()
    if device != "mps":
        print("[smoke:real] WARNING: device != mps; the bf16/MPS recipe is "
              "what we actually want to confirm.")
    print(f"[smoke:real] device={device} | loading EleutherAI/llemma_7b bf16 (~14GB)...")
    set_seed(0)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cfg = ModelConfig(max_length=1024, quantize=quant_for_device(device))
        model = Encoder(cfg, device=device)
        model.train()
        ck("Llemma-7B loaded + wrapped (bf16 backbone, fp32 heads)", True,
           f"hidden_dim={model.hidden_dim}")

        if cfg.quantize == "4bit":
            is_4bit = any(p.dtype == torch.uint8 for p in model.backbone.parameters())
            ck("backbone is 4-bit quantized", is_4bit, "bitsandbytes nf4")
        else:
            bb_dtype = next(model.backbone.parameters()).dtype
            ck("backbone is bf16", bb_dtype == torch.bfloat16, f"dtype={bb_dtype}")
        head_dtype = model.embed_linear.weight.dtype
        ck("heads are fp32", head_dtype == torch.float32, f"dtype={head_dtype}")

        ov = _overfit(model, device)
        curve = ov["curve"]
        floor = ov["floor"]
        reached = (curve[-1] < curve[0] - 0.5) and (curve[-1] <= floor + 0.15)
        ck("overfit converges to SupCon multi-positive floor log(k)", reached,
           f"start={curve[0]:.4f} -> end={curve[-1]:.4f} (floor={floor:.4f})")
        ck("within > across cosine separation after overfit", ov["within"] > ov["across"],
           f"within={ov['within']:+.4f} across={ov['across']:+.4f} "
           f"gap={ov['within'] - ov['across']:+.4f}")
        print(f"    per-step wall-clock (MPS, warm mean): {ov['mean_ms']:.0f} ms/step")
        try:
            from src.train import save_loss_curve
            png = os.path.join(_ROOT, "runs", "_smoke_real_overfit.png")
            save_loss_curve(list(range(len(curve))), curve, png)
            print(f"    loss curve -> {png}")
        except Exception as e:  # plotting is non-essential to the check
            print(f"    (loss curve skipped: {e})")

        print("[smoke:real] batch-size ceiling probe @ seq 1024 (bf16 fwd+bwd):")
        max_bs = _probe_batch_ceiling(model, device)
        ck("batch-size ceiling found (>=1)", max_bs >= 1, f"max_ok_bs={max_bs}")

        model.zero_grad(set_to_none=True)
        if device == "mps":
            torch.mps.empty_cache()
        q = EmbeddingQueue(capacity=512, dim=cfg.proj_dim, device=device, dtype=torch.float32)
        ds = ProofPairDataset(V3, PAIRS)
        dl = build_dataloader(ds, model.tokenizer, batch_size=4, max_length=256, shuffle=True)
        qloss = []
        for i, b in enumerate(dl):
            qloss.append(float(compute_loss(model, b, 0.1, queue=q).item()))
            if i >= 2:
                break
        ck("queue ON path finite on real model", all(x == x for x in qloss),
           f"losses={[round(x, 3) for x in qloss]} queue_len={len(q)}")

    fallbacks = [str(w.message) for w in caught
                 if "fallback" in str(w.message).lower() or "MPS" in str(w.message)]
    uniq = sorted(set(fallbacks))
    print(f"[smoke:real] captured {len(uniq)} unique MPS/fallback warning(s):")
    for w in uniq[:20]:
        print(f"    - {w[:160]}")

    return ck.summary()


# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description="Contrastive-encoder smoke harness")
    p.add_argument("--mode", choices=["tiny", "real"], default="tiny")
    p.add_argument("--device", choices=["mps", "cpu", "cuda"], default=None)
    args = p.parse_args()

    warnings.filterwarnings("default")
    device = pick_device(args.device)

    if args.mode == "tiny":
        ok = run_tiny(device)
    else:
        if device == "cuda" and not torch.cuda.is_available():
            raise SystemExit("FATAL: --device cuda but CUDA unavailable.")
        if device == "mps" and not torch.backends.mps.is_available():
            raise SystemExit("FATAL: --device mps but MPS unavailable.")
        ok = run_real(device)

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
