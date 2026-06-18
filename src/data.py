"""Dataset + collate for the contrastive encoder.

Positive pairs come from two sources, unified by the full_name label:
  (1) augmentation pairs: two augmented views of a v3 proof's full_source
  (2) git-history pairs:  (old_source, new_source) for the same theorem

Augmentation is applied FRESH at __getitem__ time (never pre-baked). The collate
stacks 2*B views and returns a [2*B] label tensor; because the two views of each
item share a label, every label appears >= 2x per batch (in-batch positives).
"""

import json

import torch
from torch.utils.data import Dataset, DataLoader

from src.augment import augment


def _load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_label_map(v3_path: str, pairs_path: str) -> dict:
    """full_name -> contiguous int id over all names appearing in either file."""
    names = []
    seen = set()
    for rec in _load_jsonl(v3_path) + _load_jsonl(pairs_path):
        name = rec["full_name"]
        if name not in seen:
            seen.add(name)
            names.append(name)
    return {name: i for i, name in enumerate(names)}


class ProofPairDataset(Dataset):
    def __init__(self, v3_path: str, pairs_path: str, label_map: dict = None):
        self.label_map = label_map or build_label_map(v3_path, pairs_path)
        self._items = []  # (kind, payload, label_id)
        for rec in _load_jsonl(v3_path):
            self._items.append(("aug", rec["full_source"],
                                self.label_map[rec["full_name"]]))
        for rec in _load_jsonl(pairs_path):
            self._items.append(("pair", (rec["old_source"], rec["new_source"]),
                                self.label_map[rec["full_name"]]))

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        kind, payload, label = self._items[idx]
        if kind == "aug":
            return augment(payload), augment(payload), label
        old, new = payload
        return augment(old), augment(new), label


def make_collate(tokenizer, max_length: int = 1024):
    """Return a collate_fn that tokenizes 2*B views (A-block then B-block)."""
    def collate(batch):
        views_a = [a for a, _b, _y in batch]
        views_b = [b for _a, b, _y in batch]
        labels = [y for _a, _b, y in batch]
        texts = views_a + views_b
        enc = tokenizer(texts, padding=True, truncation=True,
                        max_length=max_length, return_tensors="pt")
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": torch.tensor(labels + labels, dtype=torch.long),
        }
    return collate


def build_dataloader(dataset, tokenizer, batch_size: int, max_length: int = 1024,
                     shuffle: bool = True):
    # MPS: num_workers=0, pin_memory=False (multiprocessing/pinning issues).
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=False,
        collate_fn=make_collate(tokenizer, max_length=max_length),
    )
