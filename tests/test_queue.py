import torch

from src.queue import EmbeddingQueue


def _rows(labels, dim):
    z = torch.stack([torch.full((dim,), float(lbl)) for lbl in labels])
    lab = torch.tensor(labels, dtype=torch.long)
    return z, lab


def test_queue_fills_below_capacity():
    q = EmbeddingQueue(capacity=10, dim=4, device="cpu", dtype=torch.float32)
    z, lab = _rows([1, 2, 3], 4)
    q.enqueue(z, lab)
    cz, cl, cnt = q.contents()
    assert cnt == 3
    assert cz.shape == (3, 4)
    assert set(cl.tolist()) == {1, 2, 3}


def test_queue_fifo_overwrite_oldest_first():
    q = EmbeddingQueue(capacity=4, dim=2, device="cpu", dtype=torch.float32)
    for lbl in [0, 1, 2, 3, 4, 5]:
        z, lab = _rows([lbl], 2)
        q.enqueue(z, lab)
    cz, cl, cnt = q.contents()
    assert cnt == 4
    assert set(cl.tolist()) == {2, 3, 4, 5}  # oldest (0, 1) overwritten


def test_queue_count_caps_at_capacity():
    q = EmbeddingQueue(capacity=4, dim=2, device="cpu", dtype=torch.float32)
    q.enqueue(*_rows([0, 1, 2], 2))
    q.enqueue(*_rows([3, 4, 5], 2))
    _, cl, cnt = q.contents()
    assert cnt == 4
    assert set(cl.tolist()) == {2, 3, 4, 5}


def test_queue_stores_detached():
    q = EmbeddingQueue(capacity=4, dim=2, device="cpu", dtype=torch.float32)
    z = torch.randn(2, 2, requires_grad=True)
    q.enqueue(z, torch.tensor([0, 1]))
    cz, _, _ = q.contents()
    assert cz.requires_grad is False


def test_queue_casts_to_buffer_dtype():
    q = EmbeddingQueue(capacity=4, dim=2, device="cpu", dtype=torch.float32)
    q.enqueue(*_rows([1, 2], 2))
    cz, _, _ = q.contents()
    assert cz.dtype == torch.float32


def test_queue_empty_contents():
    q = EmbeddingQueue(capacity=4, dim=2, device="cpu", dtype=torch.float32)
    cz, cl, cnt = q.contents()
    assert cnt == 0
    assert cz.shape == (0, 2)
