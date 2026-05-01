"""
radagent.models.specialist
--------------------------
The "specialist CV head" from the RadAgent system architecture.

Job: catch findings the VLM misses (small nodules, subtle infiltrates).
Output: per-finding probabilities + a feature map for Grad-CAM++.

Backbone choice: ConvNeXt-V2 Base (FCMAE pretrained, then in22k+in1k tuned).
Why not ViT? On medical imaging at modest dataset sizes (~100K), CNNs
still beat plain ViTs in clean benchmarks. ConvNeXt-V2 also exposes a
clean spatial feature map for Grad-CAM, which ViTs make awkward.
"""
from __future__ import annotations

import timm
import torch
import torch.nn as nn


class SpecialistCXR(nn.Module):
    def __init__(
        self,
        timm_name: str,
        num_classes: int,
        pretrained: bool = True,
        drop_path_rate: float = 0.2,
        grad_checkpointing: bool = True,
    ):
        super().__init__()
        # num_classes=0 → timm returns a feature pooler we can replace
        self.backbone = timm.create_model(
            timm_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="",
            drop_path_rate=drop_path_rate,
        )
        if grad_checkpointing and hasattr(self.backbone, "set_grad_checkpointing"):
            self.backbone.set_grad_checkpointing(True)

        feat_dim = self.backbone.num_features
        self.pool = nn.AdaptiveAvgPool2d(1)
        # Two-layer head — slightly more capacity than a single linear,
        # which helps when the head has a much higher LR than the backbone.
        self.head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, feat_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(feat_dim, num_classes),
        )

        # We keep a handle on the last spatial feature map for Grad-CAM++.
        self._last_feat: torch.Tensor | None = None

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        # ConvNeXt-V2 timm outputs [B, C, H, W] when global_pool=""
        feat = self.backbone(x)
        if feat.dim() == 3:  # some timm models return tokenized
            # rare path; reshape if needed
            B, N, C = feat.shape
            side = int(N ** 0.5)
            feat = feat.transpose(1, 2).reshape(B, C, side, side)
        return feat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.forward_features(x)
        self._last_feat = feat  # for Grad-CAM
        pooled = self.pool(feat).flatten(1)
        return self.head(pooled)

    # ---------- discriminative LR helper ----------
    def parameter_groups(
        self,
        lr_backbone: float,
        lr_head: float,
        weight_decay: float,
    ) -> list[dict]:
        """Return param groups so backbone and head can use different LRs.

        Also splits out no-decay params (norms, biases) — standard practice.
        """
        decay, no_decay = [], []
        for n, p in self.backbone.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim == 1 or n.endswith(".bias"):
                no_decay.append(p)
            else:
                decay.append(p)

        head_decay, head_no_decay = [], []
        for n, p in self.head.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim == 1 or n.endswith(".bias"):
                head_no_decay.append(p)
            else:
                head_decay.append(p)

        return [
            {"params": decay, "lr": lr_backbone, "weight_decay": weight_decay},
            {"params": no_decay, "lr": lr_backbone, "weight_decay": 0.0},
            {"params": head_decay, "lr": lr_head, "weight_decay": weight_decay},
            {"params": head_no_decay, "lr": lr_head, "weight_decay": 0.0},
        ]
