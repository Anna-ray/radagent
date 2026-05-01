"""
radagent.losses.asl
-------------------
Asymmetric Loss (ASL) — Ben-Baruch et al., ICCV 2021.

Why this and not BCE+pos_weight:
  - BCE+pos_weight uniformly inflates positive gradients, which on noisy
    multi-label datasets like NIH-14 amplifies label noise.
  - ASL down-weights *easy negatives* (which dominate the gradient when
    99% of (sample, class) pairs are negative) AND clips the very-low
    probability negatives that are likely mislabeled positives.
  - Empirically beats BCE+pos_weight by 1-2% mAUC on NIH-14 / CheXpert.

Reference implementation, simplified for clarity and correctness.
Original: https://github.com/Alibaba-MIIL/ASL
"""
from __future__ import annotations

import torch
import torch.nn as nn


class AsymmetricLoss(nn.Module):
    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 1.0,
        clip: float = 0.05,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits: [B, C] raw model outputs
        targets: [B, C] {0, 1}
        """
        x_sigmoid = torch.sigmoid(logits)
        xs_pos = x_sigmoid
        xs_neg = 1.0 - x_sigmoid

        # Probability margin shifting on negatives — clips the most ambiguous
        # ones (likely mislabeled positives) so they don't dominate gradient.
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1.0)

        # Standard log terms
        log_pos = torch.log(xs_pos.clamp(min=self.eps))
        log_neg = torch.log(xs_neg.clamp(min=self.eps))
        loss_pos = targets * log_pos
        loss_neg = (1 - targets) * log_neg
        loss = loss_pos + loss_neg

        # Asymmetric focusing
        if self.gamma_pos > 0 or self.gamma_neg > 0:
            with torch.amp.autocast(device_type="cuda", enabled=False):
                pt0 = xs_pos.float() * targets
                pt1 = xs_neg.float() * (1 - targets)
                pt = pt0 + pt1
                gamma = self.gamma_pos * targets + self.gamma_neg * (1 - targets)
                one_sided_w = torch.pow(1 - pt, gamma)
            loss = loss * one_sided_w

        return -loss.sum(dim=-1).mean()
