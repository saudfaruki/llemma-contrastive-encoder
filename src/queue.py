"""FIFO memory queue of detached projection embeddings (extra SupCon negatives).

Gated behind --use-queue (default OFF on this 64GB Mac; it's the lever for the
later 4090 run). This is NOT MoCo: there is no momentum encoder, just a ring
buffer of detached embeddings + their labels, with same-label masking handled
in the loss.
"""

import torch


class EmbeddingQueue:
    def __init__(self, capacity: int = 4096, dim: int = 128,
                 device="cpu", dtype=torch.float32):
        self.capacity = int(capacity)
        self.dim = int(dim)
        self.device = device
        self.dtype = dtype
        self._z = torch.zeros(self.capacity, self.dim, device=device, dtype=dtype)
        self._labels = torch.full((self.capacity,), -1, device=device, dtype=torch.long)
        self._ptr = 0
        self._count = 0

    @torch.no_grad()
    def enqueue(self, z: torch.Tensor, labels: torch.Tensor) -> None:
        z = z.detach().to(device=self.device, dtype=self.dtype)
        labels = labels.detach().to(device=self.device, dtype=torch.long).view(-1)
        b = z.shape[0]
        assert b <= self.capacity, "batch larger than queue capacity is unsupported"
        idx = (self._ptr + torch.arange(b, device=self.device)) % self.capacity
        self._z[idx] = z
        self._labels[idx] = labels
        self._ptr = (self._ptr + b) % self.capacity
        self._count = min(self._count + b, self.capacity)

    def contents(self):
        """Return (z[:count], labels[:count], count). Before the first wrap the
        valid rows occupy [0:count]; after wrapping count==capacity covers all."""
        n = self._count
        # Return a cloned snapshot, not a view: the loss saves these tensors for
        # backward, and the next enqueue() writes self._z/_labels in place. Without
        # the clone that in-place write mutates the saved tensor mid-step ->
        # autograd "variable needed for gradient ... modified by an inplace
        # operation" (only bites with use_queue once the queue is non-empty).
        return self._z[:n].clone(), self._labels[:n].clone(), n

    def __len__(self):
        return self._count
