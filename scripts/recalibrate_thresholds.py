"""
scripts/recalibrate_thresholds.py
---------------------------------
The training-time calibration step computes per-class F1-optimal
thresholds on UNCALIBRATED logits, then separately fits a temperature
scaler. If the fitted temperature significantly deviates from 1.0,
those thresholds become inconsistent with the eval-time probabilities.

This script fixes that by:
  1. Loading EMA weights from the checkpoint
  2. Running val-set inference (TTA matching eval)
  3. Applying the fitted temperature to the logits
  4. Re-searching per-class F1-optimal thresholds on the CALIBRATED probabilities
  5. Writing a corrected calibration.json

Usage (after running scripts.calibrate first):
    python -m scripts.recalibrate_thresholds \
        --config configs/nih14_convnextv2_base.yaml \
        --checkpoint runs/nih14_convnextv2_base_384/best.pt \
        --calibration runs/nih14_convnextv2_base_384/calibration.json \
        --output runs/nih14_convnextv2_base_384/calibration_fixed.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.amp import autocast
from torch.utils.data import DataLoader

from radagent.data.dataset import (
    NIHChestXray14,
    build_eval_transforms,
    build_train_transforms,
    load_nih14_dataframe,
    patient_disjoint_split,
)
from radagent.models.specialist import SpecialistCXR
from radagent.utils.metrics import (
    find_optimal_thresholds,
    per_class_auc,
    per_class_f1,
)
from radagent.utils.training_utils import set_seed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--calibration", type=str, required=True)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--no-tta", action="store_true")
    return p.parse_args()


def _amp_dtype(name: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    with open(args.calibration) as f:
        old_cal = json.load(f)
    classes = list(cfg["data"]["classes"])
    set_seed(cfg["experiment"]["seed"])

    device = torch.device("cuda")
    print(f"[device] {torch.cuda.get_device_name(0)}")
    print(f"[old cal] T={old_cal['temperature']:.4f}  "
          f"old val_mean_auc={old_cal.get('val_mean_auc', '?')}")

    # ----- recreate the same val split used in training -----
    train_val_df, _ = load_nih14_dataframe(
        labels_csv=cfg["data"]["labels_csv"],
        train_split_txt=cfg["data"]["train_split_txt"],
        test_split_txt=cfg["data"]["test_split_txt"],
    )
    _, val_df = patient_disjoint_split(
        train_val_df,
        val_fraction=cfg["data"]["val_fraction"],
        seed=cfg["experiment"]["seed"],
    )

    eval_tfms = build_eval_transforms(image_size=cfg["data"]["image_size"])
    train_tfms = build_train_transforms(
        image_size=cfg["data"]["image_size"],
        affine_deg=cfg["augment"]["random_affine_degrees"],
        affine_trans=cfg["augment"]["random_affine_translate"],
        elastic_alpha=cfg["augment"]["elastic_alpha"],
        elastic_sigma=cfg["augment"]["elastic_sigma"],
        rrc_scale=tuple(cfg["augment"]["random_resized_crop_scale"]),
        hflip_prob=cfg["augment"]["hflip_prob"],
    )
    val_ds = NIHChestXray14(
        labels_df=val_df,
        images_dir=cfg["data"]["images_dir"],
        classes=classes,
        image_size=cfg["data"]["image_size"],
        is_train=False,
        train_transforms=train_tfms,
        eval_transforms=eval_tfms,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=cfg["data"]["pin_memory"],
        persistent_workers=False,
    )
    print(f"[data] val={len(val_ds)}")

    # ----- model -----
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = SpecialistCXR(
        timm_name=cfg["model"]["name"],
        num_classes=len(classes),
        pretrained=False,
        drop_path_rate=cfg["model"]["drop_path_rate"],
        grad_checkpointing=False,
    )
    state_key = "ema" if "ema" in ckpt else "model"
    model.load_state_dict(ckpt[state_key])
    model = model.to(device).eval()

    # ----- inference -----
    amp_dt = _amp_dtype(cfg["train"]["amp_dtype"])
    use_tta = not args.no_tta
    all_logits, all_labels = [], []
    t0 = time.time()
    with torch.no_grad():
        for i, (imgs, labels, _meta) in enumerate(val_loader):
            imgs = imgs.to(device, non_blocking=True)
            with autocast(device_type="cuda", dtype=amp_dt):
                logits = model(imgs)
                if use_tta:
                    logits_flip = model(torch.flip(imgs, dims=[-1]))
                    logits = (logits + logits_flip) / 2.0
            all_logits.append(logits.float().cpu())
            all_labels.append(labels.cpu())
            if (i + 1) % 50 == 0:
                done = (i + 1) * args.batch_size
                rate = done / (time.time() - t0)
                print(f"  [val] {done:,}/{len(val_ds):,}  ({rate:.1f} img/s)")
    print(f"[val] inference done in {time.time()-t0:.1f}s")

    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()

    # ----- apply temperature, get calibrated probs -----
    T = float(old_cal["temperature"])
    probs_cal = 1.0 / (1.0 + np.exp(-logits / max(T, 1e-3)))

    # ----- AUC sanity check (should match old report exactly — temperature is monotonic) -----
    aucs = per_class_auc(labels, probs_cal)
    m_auc = float(np.nanmean(aucs))
    print(f"[val] mean AUC (calibrated probs) = {m_auc:.4f}")

    # ----- KEY STEP: find F1-optimal thresholds on the CALIBRATED probabilities -----
    new_thr = find_optimal_thresholds(labels, probs_cal)

    # Show before/after F1
    old_thr = np.array(old_cal["thresholds"])
    f1_old = per_class_f1(labels, probs_cal, old_thr)
    f1_new = per_class_f1(labels, probs_cal, new_thr)
    print(f"\n[fix] per-class F1 on val (using calibrated probs):")
    print(f"  {'Class':<22} {'old thr':>8} {'old F1':>8}  {'new thr':>8} {'new F1':>8}")
    for c, name in enumerate(classes):
        print(f"  {name:<22} {old_thr[c]:>8.4f} {f1_old[c]:>8.3f}  "
              f"{new_thr[c]:>8.4f} {f1_new[c]:>8.3f}")
    print(f"\n[fix] mean F1 old={np.nanmean(f1_old):.4f}  "
          f"new={np.nanmean(f1_new):.4f}")

    # ----- write fixed calibration -----
    fixed = {
        "temperature": T,
        "thresholds": new_thr.tolist(),
        "val_mean_auc": m_auc,
        "val_per_class_auc": aucs.tolist(),
        "val_mean_f1_calibrated": float(np.nanmean(f1_new)),
        "val_n": int(len(val_ds)),
        "checkpoint": str(args.checkpoint),
        "note": "thresholds re-derived on calibrated probabilities",
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(fixed, f, indent=2)
    print(f"\n[done] wrote fixed calibration → {out_path}")


if __name__ == "__main__":
    main()
