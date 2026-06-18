"""Supervised Contrastive loss (SupCon, Khosla et al. 2020).

Label = theorem full_name (int id). Multiple positives per anchor. Optional
memory-queue negatives passed in as extra columns; queued entries that share the
anchor's label are masked OUT entirely (neither positive nor negative) to avoid
false negatives. Guards anchors with zero in-batch positives.
"""

import torch


def supcon_loss(z: torch.Tensor, labels: torch.Tensor, temperature: float = 0.1,
                queue_z: torch.Tensor = None, queue_labels: torch.Tensor = None):
    """z: [M, d] L2-normalized embeddings. labels: [M] int ids.

    queue_z/queue_labels: optional [K, d]/[K] detached extra negatives.
    Returns a scalar loss tensor (0.0, grad-connected, if no anchor has a positive).
    """
    device = z.device
    m = z.shape[0]
    labels = labels.to(device).view(-1)

    logits = (z @ z.t()) / temperature                  # [M, M]
    col_labels = labels

    has_queue = queue_z is not None and queue_z.shape[0] > 0
    if has_queue:
        qz = queue_z.to(device=device, dtype=z.dtype)
        qlab = queue_labels.to(device).view(-1)
        logits = torch.cat([logits, (z @ qz.t()) / temperature], dim=1)  # [M, M+K]
        col_labels = torch.cat([labels, qlab], dim=0)                    # [M+K]

    # Numerical stability: subtract per-row max (detached). The self-similarity
    # (1/temperature) is the row max, so shifted logits are <= 0 -> exp in (0, 1].
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()

    n_cols = logits.shape[1]
    same = labels.view(m, 1).eq(col_labels.view(1, n_cols))     # [M, n_cols]
    self_mask = torch.zeros(m, n_cols, dtype=torch.bool, device=device)
    diag = torch.arange(m, device=device)
    self_mask[diag, diag] = True

    # Denominator columns.
    valid = torch.ones(m, n_cols, dtype=torch.bool, device=device)
    valid[:, :m] = ~self_mask[:, :m]            # batch: exclude self
    if has_queue:
        valid[:, m:] = ~same[:, m:]             # queue: drop same-label entirely

    # Positive columns (numerator): same label, exclude self; queue never positive.
    pos = same & ~self_mask
    if has_queue:
        pos[:, m:] = False

    exp_logits = torch.exp(logits) * valid
    denom = exp_logits.sum(dim=1, keepdim=True)
    log_prob = logits - torch.log(denom + 1e-12)

    pos_count = pos.sum(dim=1)
    mean_log_prob_pos = (pos * log_prob).sum(dim=1) / pos_count.clamp(min=1)
    loss_per = -mean_log_prob_pos

    valid_anchor = pos_count > 0
    if valid_anchor.any():
        return loss_per[valid_anchor].mean()
    return z.sum() * 0.0
