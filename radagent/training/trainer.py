"""
radagent.training.trainer
-------------------------
Production training loop.

Features:
  - Mixed precision (bf16 on Ada Lovelace / MI300X, fp16 fallback)
  - Gradient accumulation
  - EMA weights (used for validation + final checkpoint)
  - Discriminative LRs (backbone vs head)
  - Cosine warmup
  - TTA-enabled validation (hflip)
  - Per-class AUC + F1 logging
  - Early stopping
  - Best/Top-K checkpointing
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from ..utils.metrics import (
    find_optimal_thresholds,
    mean_auc,
    per_class_auc,
    per_class_f1,
)
from ..utils.training_utils import (
    CosineWarmupScheduler,
    ModelEMA,
    TemperatureScaler,
)


def _amp_dtype(name: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        classes: list[str],
        cfg: dict,
        device: torch.device,
        output_dir: str,
    ):
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.classes = classes
        self.cfg = cfg
        self.device = device
        self.out_dir = Path(output_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(self.out_dir / "tb")

        steps_per_epoch = max(
            1, len(train_loader) // cfg["train"]["grad_accum_steps"]
        )
        total_steps = steps_per_epoch * cfg["train"]["epochs"]
        warmup_steps = steps_per_epoch * cfg["scheduler"]["warmup_epochs"]
        self.scheduler = CosineWarmupScheduler(
            optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            min_lr_ratio=cfg["scheduler"]["min_lr_ratio"],
        )

        self.amp_dtype = _amp_dtype(cfg["train"]["amp_dtype"])
        # GradScaler only for fp16; bf16 doesn't need scaling.
        self.scaler = torch.amp.GradScaler("cuda", enabled=(self.amp_dtype == torch.float16))
        self.ema = ModelEMA(model, decay=cfg["train"]["ema_decay"])
        self.global_step = 0
        self.best_metric = -float("inf")
        self.epochs_no_improve = 0

    # ----------------- resume support -----------------
    def resume_from(self, path: str) -> int:
        """Load weights, EMA, optimizer, and scheduler step from a checkpoint.
        Returns the next epoch index to start training from (1-based).
        """
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model"])
        if "ema" in ckpt:
            self.ema.module.load_state_dict(ckpt["ema"])
        if "optimizer" in ckpt:
            try:
                self.optimizer.load_state_dict(ckpt["optimizer"])
            except (ValueError, RuntimeError) as e:
                print(f"[resume] WARN optimizer state mismatch ({e}); "
                      f"continuing with fresh optimizer state.")
        if "scheduler_step" in ckpt:
            self.scheduler._step = int(ckpt["scheduler_step"])
            # Re-apply scheduler to current step
            scale = self.scheduler._scale(self.scheduler._step)
            for g, base in zip(self.optimizer.param_groups, self.scheduler.base_lrs):
                g["lr"] = base * scale
        if "metrics" in ckpt and "mean_auc" in ckpt["metrics"]:
            self.best_metric = float(ckpt["metrics"]["mean_auc"])
        next_epoch = int(ckpt.get("epoch", 0)) + 1
        print(f"[resume] loaded {path}: next epoch = {next_epoch}, "
              f"best_metric = {self.best_metric:.4f}, "
              f"scheduler_step = {self.scheduler._step}")
        return next_epoch

    # ----------------- training -----------------
    def fit(self, start_epoch: int = 1):
        epochs = self.cfg["train"]["epochs"]
        for ep in range(start_epoch, epochs + 1):
            t0 = time.time()
            train_loss = self._train_one_epoch(ep)
            val_metrics = self._validate(use_tta=self.cfg["eval"]["tta"])
            elapsed = time.time() - t0

            self._log_epoch(ep, train_loss, val_metrics, elapsed)

            metric = val_metrics["mean_auc"]
            improved = metric > self.best_metric
            if improved:
                self.best_metric = metric
                self.epochs_no_improve = 0
                self._save_checkpoint("best.pt", ep, val_metrics)
            else:
                self.epochs_no_improve += 1

            self._save_checkpoint("last.pt", ep, val_metrics)

            if self.epochs_no_improve >= self.cfg["train"]["early_stopping_patience"]:
                print(
                    f"[early stop] no improvement for "
                    f"{self.epochs_no_improve} epochs."
                )
                break

        # Final calibration on val set with EMA weights
        cal = self._calibrate()
        with open(self.out_dir / "calibration.json", "w") as f:
            json.dump(cal, f, indent=2)
        print(f"[done] best mean AUC = {self.best_metric:.4f}")

    def _train_one_epoch(self, epoch: int) -> float:
        self.model.train()
        accum = self.cfg["train"]["grad_accum_steps"]
        clip_norm = self.cfg["train"]["gradient_clip_norm"]
        running = 0.0
        n_batches = 0

        self.optimizer.zero_grad(set_to_none=True)
        for step, (imgs, labels, _meta) in enumerate(self.train_loader):
            imgs = imgs.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            with autocast(device_type="cuda", dtype=self.amp_dtype):
                logits = self.model(imgs)
                loss = self.loss_fn(logits, labels) / accum

            if self.amp_dtype == torch.float16:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step + 1) % accum == 0:
                if self.amp_dtype == torch.float16:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), clip_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    nn.utils.clip_grad_norm_(self.model.parameters(), clip_norm)
                    self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.scheduler.step()
                self.ema.update(self.model)
                self.global_step += 1

                if self.global_step % self.cfg["logging"]["log_every_n_steps"] == 0:
                    cur_lr = self.optimizer.param_groups[0]["lr"]
                    self.writer.add_scalar("train/loss_step", loss.item() * accum, self.global_step)
                    self.writer.add_scalar("train/lr_backbone", cur_lr, self.global_step)
                    print(
                        f"  ep{epoch} step{self.global_step}  "
                        f"loss={loss.item()*accum:.4f}  lr_bb={cur_lr:.2e}"
                    )

            running += loss.item() * accum
            n_batches += 1

        return running / max(1, n_batches)

    # ----------------- validation -----------------
    @torch.no_grad()
    def _validate(self, use_tta: bool) -> dict:
        # Always evaluate the EMA weights — they are more stable
        ema_module = self.ema.module
        ema_module.eval()
        ema_module.to(self.device)

        all_logits, all_labels = [], []
        for imgs, labels, _meta in self.val_loader:
            imgs = imgs.to(self.device, non_blocking=True)
            with autocast(device_type="cuda", dtype=self.amp_dtype):
                logits = ema_module(imgs)
                if use_tta:
                    logits_flip = ema_module(torch.flip(imgs, dims=[-1]))
                    logits = (logits + logits_flip) / 2.0
            all_logits.append(logits.float().cpu())
            all_labels.append(labels.cpu())

        logits = torch.cat(all_logits).numpy()
        labels = torch.cat(all_labels).numpy()
        probs = 1.0 / (1.0 + np.exp(-logits))

        aucs = per_class_auc(labels, probs)
        m_auc = float(np.nanmean(aucs))
        thr = find_optimal_thresholds(labels, probs)
        f1s = per_class_f1(labels, probs, thr)
        m_f1 = float(np.nanmean(f1s))

        return {
            "mean_auc": m_auc,
            "mean_f1": m_f1,
            "per_class_auc": aucs.tolist(),
            "per_class_f1": f1s.tolist(),
            "thresholds": thr.tolist(),
            "_val_logits": logits,
            "_val_labels": labels,
        }

    # ----------------- calibration -----------------
    def _calibrate(self) -> dict:
        """Fit a single-temperature scaler on the val set (EMA weights)."""
        metrics = self._validate(use_tta=False)
        logits = torch.from_numpy(metrics["_val_logits"]).float()
        labels = torch.from_numpy(metrics["_val_labels"]).float()
        ts = TemperatureScaler()
        T = ts.fit(logits, labels, n_iter=200)
        return {"temperature": T, "thresholds": metrics["thresholds"]}

    # ----------------- I/O -----------------
    def _save_checkpoint(self, name: str, epoch: int, metrics: dict):
        path = self.out_dir / name
        # Strip the heavy numpy arrays before saving as JSON-side metadata
        meta = {k: v for k, v in metrics.items() if not k.startswith("_")}
        torch.save(
            {
                "epoch": epoch,
                "model": self.model.state_dict(),
                "ema": self.ema.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler_step": self.scheduler._step,
                "metrics": meta,
                "classes": self.classes,
                "cfg": self.cfg,
            },
            path,
        )

    def _log_epoch(self, ep: int, train_loss: float, val: dict, elapsed: float):
        print(
            f"[ep {ep:02d}] train_loss={train_loss:.4f}  "
            f"val_mean_auc={val['mean_auc']:.4f}  "
            f"val_mean_f1={val['mean_f1']:.4f}  "
            f"({elapsed:.0f}s)"
        )
        self.writer.add_scalar("val/mean_auc", val["mean_auc"], ep)
        self.writer.add_scalar("val/mean_f1", val["mean_f1"], ep)
        for c, name in enumerate(self.classes):
            self.writer.add_scalar(f"val_auc/{name}", val["per_class_auc"][c], ep)
            self.writer.add_scalar(f"val_f1/{name}", val["per_class_f1"][c], ep)
