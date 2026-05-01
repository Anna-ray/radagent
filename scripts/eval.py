"""
scripts/eval.py
---------------
Submission-grade evaluation on the official NIH ChestX-ray14 test set.

Usage:
    python -m scripts.eval --config configs/nih14_convnextv2_base.yaml \
        --checkpoint runs/nih14_convnextv2_base_384/best.pt \
        --calibration runs/nih14_convnextv2_base_384/calibration.json \
        --output-dir runs/nih14_convnextv2_base_384/eval

Outputs (in --output-dir):
    test_metrics.json     -- full metrics dict (per-class AUC/AP/F1/sens/spec, macro/micro, CIs)
    test_report.md        -- submission-ready markdown table
    test_predictions.parquet -- raw probabilities per image (for downstream agent use)

Notes:
- Loads EMA weights from the checkpoint (key "ema") if present, else "model".
- Applies temperature calibration AND per-class F1-optimal thresholds from
  the calibration.json produced at end of training.
- Uses TTA (horizontal flip) by default — same as during validation.
- Bootstraps 1000 image-level resamples for 95% CIs.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.amp import autocast
from torch.utils.data import DataLoader

from radagent.data.dataset import build_eval_transforms
from radagent.data.test_dataset import NIHTestSet
from radagent.models.specialist import SpecialistCXR
from radagent.utils.bootstrap import per_class_metrics_with_ci
from radagent.utils.report import render_markdown_report


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to best.pt produced by training")
    p.add_argument("--calibration", type=str, required=True,
                   help="Path to calibration.json produced by training")
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--no-tta", action="store_true",
                   help="Disable horizontal-flip TTA (default: TTA on)")
    p.add_argument("--n-bootstrap", type=int, default=1000)
    return p.parse_args()


def _amp_dtype(name: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ----- config + classes -----
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    classes = list(cfg["data"]["classes"])

    # ----- device -----
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available.")
    device = torch.device("cuda")
    print(f"[device] {torch.cuda.get_device_name(0)}")

    # ----- load checkpoint -----
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = SpecialistCXR(
        timm_name=cfg["model"]["name"],
        num_classes=len(classes),
        pretrained=False,                      # weights from ckpt
        drop_path_rate=cfg["model"]["drop_path_rate"],
        grad_checkpointing=False,              # not needed for eval
    )
    state_key = "ema" if "ema" in ckpt else "model"
    model.load_state_dict(ckpt[state_key])
    model = model.to(device).eval()
    print(f"[ckpt] loaded '{state_key}' weights from {args.checkpoint}")
    print(f"[ckpt] training epoch={ckpt.get('epoch', '?')} "
          f"val_mean_auc={ckpt.get('metrics', {}).get('mean_auc', '?')}")

    # ----- load calibration (temperature + per-class thresholds) -----
    with open(args.calibration) as f:
        cal = json.load(f)
    temperature = float(cal["temperature"])
    thresholds = np.array(cal["thresholds"], dtype=np.float64)
    assert len(thresholds) == len(classes), "threshold count mismatch"
    print(f"[cal] temperature={temperature:.4f}  "
          f"per-class thresholds loaded ({len(thresholds)})")

    # ----- test set -----
    eval_tfms = build_eval_transforms(image_size=cfg["data"]["image_size"])
    test_ds = NIHTestSet(
        labels_csv=cfg["data"]["labels_csv"],
        test_split_txt=cfg["data"]["test_split_txt"],
        images_dir=cfg["data"]["images_dir"],
        classes=classes,
        eval_transforms=eval_tfms,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=cfg["data"]["pin_memory"],
        persistent_workers=False,
    )
    print(f"[data] test set N={len(test_ds):,}")

    # ----- inference -----
    amp_dt = _amp_dtype(cfg["train"]["amp_dtype"])
    use_tta = not args.no_tta
    all_logits, all_labels, all_names = [], [], []

    t0 = time.time()
    with torch.no_grad():
        for i, (imgs, labels, meta) in enumerate(test_loader):
            imgs = imgs.to(device, non_blocking=True)
            with autocast(device_type="cuda", dtype=amp_dt):
                logits = model(imgs)
                if use_tta:
                    logits_flip = model(torch.flip(imgs, dims=[-1]))
                    logits = (logits + logits_flip) / 2.0
            all_logits.append(logits.float().cpu())
            all_labels.append(labels.cpu())
            all_names.extend(meta["image_index"])
            if (i + 1) % 50 == 0:
                done = (i + 1) * args.batch_size
                rate = done / (time.time() - t0)
                eta = (len(test_ds) - done) / max(1.0, rate)
                print(f"  [infer] {done:,}/{len(test_ds):,}  "
                      f"({rate:.1f} img/s, ETA {eta:.0f}s)")

    elapsed = time.time() - t0
    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()
    print(f"[infer] done in {elapsed:.1f}s "
          f"({len(test_ds)/elapsed:.1f} img/s, TTA={use_tta})")

    # ----- apply temperature scaling -----
    logits_cal = logits / max(temperature, 1e-3)
    probs = 1.0 / (1.0 + np.exp(-logits_cal))

    # ----- bootstrap metrics -----
    print(f"[boot] computing metrics with {args.n_bootstrap} bootstrap samples...")
    t0 = time.time()
    metrics = per_class_metrics_with_ci(
        y_true=labels,
        y_prob=probs,
        thresholds=thresholds,
        n_bootstrap=args.n_bootstrap,
        seed=42,
    )
    print(f"[boot] done in {time.time()-t0:.1f}s")

    # ----- attach metadata for the report -----
    extra_meta = {
        "Backbone": cfg["model"]["name"],
        "Image size": cfg["data"]["image_size"],
        "Checkpoint": str(args.checkpoint),
        "TTA": "horizontal flip" if use_tta else "none",
        "Temperature": f"{temperature:.4f}",
        "Inference time": f"{elapsed:.1f}s ({len(test_ds)/elapsed:.1f} img/s)",
    }

    # ----- write outputs -----
    metrics_path = out_dir / "test_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[out] {metrics_path}")

    md = render_markdown_report(metrics, classes, extra_meta=extra_meta)
    report_path = out_dir / "test_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"[out] {report_path}")

    # Predictions parquet — one row per image
    pred_df = pd.DataFrame({"image_index": all_names})
    for c, name in enumerate(classes):
        pred_df[f"prob_{name}"] = probs[:, c]
        pred_df[f"label_{name}"] = labels[:, c].astype(np.int8)
    parquet_path = out_dir / "test_predictions.parquet"
    pred_df.to_parquet(parquet_path, index=False)
    print(f"[out] {parquet_path}")

    # ----- console summary -----
    print()
    print("=" * 64)
    print("RadAgent Specialist | NIH-14 Official Test Set")
    print("=" * 64)
    print(f"N cases:        {metrics['n_samples']:,}")
    print(f"Macro AUC:      {metrics['macro_auc']:.4f}  "
          f"[{metrics['macro_auc_ci'][0]:.4f}, {metrics['macro_auc_ci'][1]:.4f}]")
    print(f"Micro AUC:      {metrics['micro_auc']:.4f}  "
          f"[{metrics['micro_auc_ci'][0]:.4f}, {metrics['micro_auc_ci'][1]:.4f}]")
    print(f"Mean F1:        {metrics['mean_f1']:.4f}")
    print(f"Mean AP:        {metrics['mean_ap']:.4f}")
    print()
    print(f"{'Class':<22} {'AUC':>7}  {'95% CI':>20}  {'F1':>6}  {'AP':>6}")
    print("-" * 64)
    for c, name in enumerate(classes):
        auc = metrics["per_class"]["auc"][c]
        lo, hi = metrics["per_class"]["auc_ci"][c]
        f1 = metrics["per_class"]["f1"][c]
        ap = metrics["per_class"]["ap"][c]
        print(f"{name:<22} {auc:>7.4f}  [{lo:.4f}, {hi:.4f}]  {f1:>6.3f}  {ap:>6.3f}")
    print("=" * 64)


if __name__ == "__main__":
    main()
