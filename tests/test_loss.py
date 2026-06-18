import torch
import torch.nn.functional as F

from src.loss import supcon_loss


def test_supcon_finite_and_has_grad():
    torch.manual_seed(0)
    raw = torch.randn(8, 16, requires_grad=True)
    z = F.normalize(raw, dim=1)
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    loss = supcon_loss(z, labels, temperature=0.1)
    assert torch.isfinite(loss)
    assert loss.item() > 0
    loss.backward()
    assert raw.grad is not None and torch.isfinite(raw.grad).all()


def test_supcon_low_when_same_label_collapsed():
    u = torch.tensor([1.0, 0.0])
    v = torch.tensor([0.0, 1.0])
    z = torch.stack([u, u, v, v])           # labels 0,0,1,1 collapse per class
    labels = torch.tensor([0, 0, 1, 1])
    loss = supcon_loss(z, labels, temperature=0.1)
    assert loss.item() < 0.01

    z_bad = torch.stack([u, u, u, u])        # same vector but labels differ -> high loss
    loss_bad = supcon_loss(z_bad, labels, temperature=0.1)
    assert loss_bad.item() > loss.item()


def test_supcon_zero_positive_guard():
    z = F.normalize(torch.randn(2, 8), dim=1)
    labels = torch.tensor([0, 1])            # no duplicates -> no positives
    loss = supcon_loss(z, labels, temperature=0.1)
    assert torch.isfinite(loss)
    assert loss.item() == 0.0


def test_queue_same_label_masked_diff_label_negative():
    u = torch.tensor([1.0, 0.0])
    v = torch.tensor([0.0, 1.0])
    z = torch.stack([u, u])                  # 2 anchors, label 0, in-batch positive
    labels = torch.tensor([0, 0])
    base = supcon_loss(z, labels, temperature=0.1)

    # Queue entries with SAME label as anchors must be masked out entirely -> no effect.
    qz_same = torch.stack([u, u, u])
    qlab_same = torch.tensor([0, 0, 0])
    same = supcon_loss(z, labels, temperature=0.1,
                       queue_z=qz_same, queue_labels=qlab_same)
    assert torch.allclose(base, same, atol=1e-6)

    # Queue entries with DIFFERENT label act as extra negatives -> loss increases.
    qz_diff = torch.stack([v, v, v])
    qlab_diff = torch.tensor([7, 7, 7])
    diff = supcon_loss(z, labels, temperature=0.1,
                       queue_z=qz_diff, queue_labels=qlab_diff)
    assert diff.item() > base.item() + 1e-6


def test_supcon_multiple_positives():
    # 3 views of same label should all attract; finite loss, grad flows.
    torch.manual_seed(1)
    raw = torch.randn(6, 8, requires_grad=True)
    z = F.normalize(raw, dim=1)
    labels = torch.tensor([0, 0, 0, 1, 1, 1])
    loss = supcon_loss(z, labels, temperature=0.1)
    assert torch.isfinite(loss) and loss.item() > 0
    loss.backward()
    assert torch.isfinite(raw.grad).all()
