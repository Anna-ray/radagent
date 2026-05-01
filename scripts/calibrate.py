"""
scripts/calibrate.py
--------------------
Standalone calibration for a trained checkpoint.

Use this when training was interrupted before the auto-calibration step
ran at end of training. It:
  1. Loads EMA weights from the checkpoint
  2. Runs val-set inference (with TTA matching training-time eval)
  3. Searches per-class F1-optimal thresholds
  4. Fits a single-temperature scaler via LBFGS
  5. Writes calibration.json (temperature + thresholds)

Usage:
    python -m scripts.calibrate \
        --config configs/nih14_convnextv2_base.yaml \
        --checkpoint runs/nih14_convnextv2_base_384/best.pt \
        --output runs/nih14_convnextv2_base_384/calibration.json
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
    mean_auc,
    per_class_auc,
)
from radagent.utils.training_utils import TemperatureScaler, set_seed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
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
    classes = list(cfg["data"]["classes"])
    set_seed(cfg["experiment"]["seed"])

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available.")
    device = torch.device("cuda")
    print(f"[device] {torch.cuda.get_device_name(0)}")

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
    print(f"[data] val={len(val_df)}")

    eval_tfms = build_eval_transforms(image_size=cfg["data"]["image_size"])
    # Reuse NIHChestXray14 in eval mode — it just needs the same train_tfms
    # placeholder, never actually called at eval time.
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
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=cfg["data"]["pin_memory"],
        persistent_workers=False,
    )

    # ----- load model (EMA weights) -----
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
    print(f"[ckpt] loaded '{state_key}' from {args.checkpoint}")
    print(f"[ckpt] training epoch={ckpt.get('epoch', '?')} "
          f"val_mean_auc(at_save)={ckpt.get('metrics', {}).get('mean_auc', '?')}")

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

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    probs = torch.sigmoid(logits).numpy()
    print(f"[val] inference done in {time.time()-t0:.1f}s, TTA={use_tta}")

    # ----- per-class AUC (sanity check) -----
    aucs = per_class_auc(labels.numpy(), probs)
    m_auc = float(np.nanmean(aucs))
    print(f"[val] mean AUC = {m_auc:.4f}")
    for c, name in enumerate(classes):
        print(f"   {name:<22} AUC={aucs[c]:.4f}")

    # ----- thresholds + temperature -----
    thresholds = find_optimal_thresholds(labels.numpy(), probs)
    print(f"[cal] thresholds:")
    for c, name in enumerate(classes):
        print(f"   {name:<22} thr={thresholds[c]:.4f}")

    ts = TemperatureScaler()
    T = ts.fit(logits.float(), labels.float(), n_iter=200)
    print(f"[cal] fitted temperature: {T:.4f}")

    # ----- write -----
    cal = {
        "temperature": T,
        "thresholds": thresholds.tolist(),
        "val_mean_auc": m_auc,
        "val_per_class_auc": aucs.tolist(),
        "val_n": int(len(val_ds)),
        "checkpoint": str(args.checkpoint),
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(cal, f, indent=2)
    print(f"[done] wrote {out_path}")
    print(f"[done] val_mean_auc with EMA + TTA = {m_auc:.4f}")


if __name__ == "__main__":
    main()
