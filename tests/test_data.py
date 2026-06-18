import json
import os

import torch

from src.data import build_label_map, ProofPairDataset, make_collate, build_dataloader

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V3 = os.path.join(_ROOT, "data", "elementary_nt_proofs_v3.jsonl")
PAIRS = os.path.join(_ROOT, "data", "same_theorem_pairs.jsonl")


def _names(path):
    with open(path) as f:
        return [json.loads(line)["full_name"] for line in f if line.strip()]


class FakeTok:
    """Minimal stand-in for an HF tokenizer __call__ surface."""
    pad_token = "<pad>"
    eos_token = "</s>"

    def __call__(self, texts, padding=True, truncation=True, max_length=8,
                 return_tensors="pt"):
        seqs = []
        for t in texts:
            toks = t.split()
            if truncation:
                toks = toks[:max_length]
            seqs.append([(abs(hash(w)) % 1000) + 1 for w in toks] or [1])
        length = max(len(s) for s in seqs)
        input_ids = torch.zeros(len(seqs), length, dtype=torch.long)
        attn = torch.zeros(len(seqs), length, dtype=torch.long)
        for i, s in enumerate(seqs):
            input_ids[i, :len(s)] = torch.tensor(s)
            attn[i, :len(s)] = 1
        return {"input_ids": input_ids, "attention_mask": attn}


def test_label_map_covers_all_names():
    lm = build_label_map(V3, PAIRS)
    v3_names = set(_names(V3))
    pair_names = set(_names(PAIRS))
    assert v3_names.issubset(lm.keys())
    assert pair_names.issubset(lm.keys())
    assert len(lm) == len(v3_names | pair_names) == 589


def test_label_map_contiguous_ids():
    lm = build_label_map(V3, PAIRS)
    assert sorted(lm.values()) == list(range(len(lm)))


def test_dataset_length_is_v3_plus_pairs():
    ds = ProofPairDataset(V3, PAIRS)
    assert len(ds) == 589 + 267


def test_getitem_structure_and_label_range():
    ds = ProofPairDataset(V3, PAIRS)
    a, b, y = ds[0]
    assert isinstance(a, str) and isinstance(b, str)
    assert a.strip() and b.strip()
    assert isinstance(y, int) and 0 <= y < len(ds.label_map)


def test_collate_shapes_and_positive_guarantee():
    ds = ProofPairDataset(V3, PAIRS)
    collate = make_collate(FakeTok(), max_length=8)
    B = 5
    out = collate([ds[i] for i in range(B)])
    assert out["input_ids"].shape[0] == 2 * B
    assert out["attention_mask"].shape == out["input_ids"].shape
    assert out["labels"].shape == (2 * B,)
    labels = out["labels"]
    for lbl in labels.unique():
        assert (labels == lbl).sum().item() >= 2   # every label has an in-batch positive


def test_collate_truncates_to_max_length():
    collate = make_collate(FakeTok(), max_length=4)
    long_text = " ".join(["tok"] * 50)
    out = collate([(long_text, long_text, 0)])
    assert out["input_ids"].shape[1] <= 4


def test_build_dataloader_yields_batch():
    ds = ProofPairDataset(V3, PAIRS)
    dl = build_dataloader(ds, FakeTok(), batch_size=4, max_length=8, shuffle=False)
    batch = next(iter(dl))
    assert batch["input_ids"].shape[0] == 8        # 2 views * 4 items
    assert batch["labels"].shape == (8,)
