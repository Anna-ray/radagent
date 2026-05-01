"""
radagent.utils.training_utils
-----------------------------
Reusable training utilities: EMA, cosine warmup, temperature calibration,
seed setting, gradient-norm logging.
"""
from __future__ import annotations

import math
import random
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ModelEMA:
    """Exponential moving average of model parameters.

    EMA weights are noticeably more stable than the raw training weights
    for noisy multi-label tasks, often giving +0.3-0.7% AUC at zero cost.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.module = deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        msd = model.state_dict()
        for k, v in self.module.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(self.decay).add_(msd[k].detach(), alpha=1 - self.decay)
            else:
                v.copy_(msd[k])

    def state_dict(self):
        return self.module.state_dict()


class CosineWarmupScheduler:
    """LR scheduler: linear warmup → cosine decay to min_lr_ratio*base_lr.

    Per-param-group aware (so head and backbone scale together).
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr_ratio: float = 0.01,
    ):
        self.optimizer = optimizer
        self.warmup_steps = max(1, warmup_steps)
        self.total_steps = max(1, total_steps)
        self.min_lr_ratio = min_lr_ratio
        self.base_lrs = [g["lr"] for g in optimizer.param_groups]
        self._step = 0

    def step(self) -> None:
        self._step += 1
        scale = self._scale(self._step)
        for g, base in zip(self.optimizer.param_groups, self.base_lrs):
            g["lr"] = base * scale

    def _scale(self, step: int) -> float:
        if step < self.warmup_steps:
            return step / self.warmup_steps
        progress = (step - self.warmup_steps) / max(
            1, self.total_steps - self.warmup_steps
        )
        progress = min(1.0, progress)
        cos = 0.5 * (1 + math.cos(math.pi * progress))
        return self.min_lr_ratio + (1 - self.min_lr_ratio) * cos


class TemperatureScaler(nn.Module):
    """Single-temperature calibration on stacked logits.

    Tuned on the validation set after training to flatten over-confident
    sigmoid outputs — this matters because the downstream LLM agent
    consumes these as 'confidence' for the structured findings layer.
    """

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature.clamp(min=1e-3)

    def fit(self, logits: torch.Tensor, targets: torch.Tensor, n_iter: int = 200):
        self.train()
        opt = torch.optim.LBFGS([self.temperature], lr=0.1, max_iter=n_iter)
        bce = nn.BCEWithLogitsLoss()

        def closure():
            opt.zero_grad()
            loss = bce(self.forward(logits), targets)
            loss.backward()
            return loss

        opt.step(closure)
        self.eval()
        return float(self.temperature.detach().cpu().item())
