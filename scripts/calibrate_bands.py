"""
scripts/calibrate_bands.py
--------------------------
Build per-class confidence bands on the calibrated probability scale.

Uses a thin val-only dataset modeled on NIHTestSet (which is known to work
in eval.py). Avoids the heavier NIHChestXray14 training class.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Sequence

import albumentations as A
import numpy as np
import pandas as pd
import torch
import yaml
from torch.amp import autocast
from torch.utils.data import DataLoader, Dataset

from radagent.data.dataset import build_eval_transforms, load_nih14_dataframe, patient_disjoint_split
from radagent.data.preprocessing import apply_clahe, load_cxr_grayscale, to_three_channel
from radagent.models.specialist import SpecialistCXR


MIN_POSITIVES_FOR_RELIABILITY = 30
MIN_BAND_WIDTH = 0.05
PCT_LOW_FALLBACK = 70.0
PCT_HIGH_FALLBACK = 90.0
PRECISION_LOW_TARGET = 0.50
PRECISION_HIGH_TARGET = 0.85


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--calibration", type=str, required=True)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=0,
                   help="Default 0 to avoid Windows worker crashes")
    p.add_argument("--cache", type=str, default=None)
    return p.parse_args()


def _amp_dtype(name: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


class NIHValSet(Dataset):
    """Thin val dataset: deterministic CLAHE, eval transforms only.

    Modeled on NIHTestSet. Takes a pre-filtered val dataframe rather than
    splitting internally, so we can use the exact patient_disjoint_split
    the trainer used.
    """

    def __init__(
        self,
        labels_df: pd.DataFrame,
        images_dir: str,
        classes: Sequence[str],
        eval_transforms: A.Compose,
        clahe_clip: float = 2.5,
    ):
        self.df = labels_df.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.classes = list(classes)
        self.tfms = eval_transforms
        self.clahe_clip = clahe_clip
        self._path_cache = self._build_path_index()
        self._label_matrix = self._build_label_matrix()

    def _build_path_index(self) -> dict[str, str]:
        idx: dict[str, str] = {}
        print(f"  [val] indexing images under {self.images_dir} ...", flush=True)
        for root, _, files in os.walk(self.images_dir):
            for f in files:
                if f.lower().endswith((".png", ".jpg", ".jpeg")):
                    idx[f] = os.path.join(root, f)
        print(f"  [val] indexed {len(idx):,} files on disk", flush=True)
        missing = [n for n in self.df["Image Index"] if n not in idx]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} val images missing on disk. "
                f"First missing: {missing[:3]}."
            )
        return idx

    def _build_label_matrix(self) -> np.ndarray:
        labels = np.zeros((len(self.df), len(self.classes)), dtype=np.float32)
        for i, finding_str in enumerate(self.df["Finding Labels"]):
            if finding_str == "No Finding":
                continue
            for f in finding_str.split("|"):
                if f in self.classes:
                    labels[i, self.classes.index(f)] = 1.0
        return labels

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        max_retries = 5
        attempt_idx = idx
        last_err = None
        for retry in range(max_retries):
            try:
                return self._load_item(attempt_idx)
            except (FileNotFoundError, OSError, ValueError) as e:
                last_err = e
                bad = self.df.iloc[attempt_idx]["Image Index"]
                print(f"[val] WARN skip corrupt '{bad}' "
                      f"(retry {retry+1}/{max_retries}): {e}", flush=True)
                attempt_idx = (attempt_idx + 1) % len(self.df)
        raise RuntimeError(f"val: 5 retries failed from idx={idx}: {last_err}")

    def _load_item(self, idx: int):
        row = self.df.iloc[idx]
        fname = row["Image Index"]
        path = self._path_cache[fname]
        gray = load_cxr_grayscale(path)
        gray = apply_clahe(gray, clip_limit=self.clahe_clip)
        rgb = to_three_channel(gray)
        out = self.tfms(image=rgb)
        image = out["image"].float()
        labels = torch.from_numpy(self._label_matrix[idx])
        meta = {"image_index": fname,
                "patient_id": int(row["Patient ID"]) if "Patient ID" in row else -1}
        return image, labels, meta


def _build_val_loader(cfg: dict, batch_size: int, num_workers: int) -> DataLoader:
    d = cfg["data"]
    print("[data] loading dataframe...", flush=True)
    train_val_df, _ = load_nih14_dataframe(
        labels_csv=d["labels_csv"],
        train_split_txt=d["train_split_txt"],
        test_split_txt=d["test_split_txt"],
    )
    print(f"[data] train_val rows={len(train_val_df):,}", flush=True)
    _, val_df = patient_disjoint_split(
        train_val_df,
        val_fraction=d["val_fraction"],
        seed=cfg["experiment"]["seed"],
    )
    print(f"[data] val rows after patient split={len(val_df):,}", flush=True)
    eval_tfms = build_eval_transforms(image_size=d["image_size"])
    val_ds = NIHValSet(
        labels_df=val_df,
        images_dir=d["images_dir"],
        classes=d["classes"],
        eval_transforms=eval_tfms,
    )
    print(f"[data] val dataset N={len(val_ds):,}", flush=True)
    return DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=False,
    )


def _run_val_inference(cfg, args) -> tuple[np.ndarray, np.ndarray]:
    if args.cache and Path(args.cache).exists():
        z = np.load(args.cache)
        print(f"[cache] reusing {args.cache}: "
              f"logits={z['logits'].shape} labels={z['labels'].shape}", flush=True)
        return z["logits"], z["labels"]

    classes = cfg["data"]["classes"]
    device = torch.device("cuda")
    print(f"[device] {torch.cuda.get_device_name(0)}", flush=True)

    print("[ckpt] loading checkpoint...", flush=True)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    print("[ckpt] building model...", flush=True)
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
    print(f"[ckpt] loaded '{state_key}' from {args.checkpoint}", flush=True)

    val_loader = _build_val_loader(cfg, args.batch_size, args.num_workers)
    amp_dt = _amp_dtype(cfg["train"]["amp_dtype"])

    all_logits, all_labels = [], []
    t0 = time.time()
    n_done = 0
    with torch.no_grad():
        for i, (imgs, labels, _meta) in enumerate(val_loader):
            imgs = imgs.to(device, non_blocking=True)
            with autocast(device_type="cuda", dtype=amp_dt):
                logits = model(imgs)
                logits_flip = model(torch.flip(imgs, dims=[-1]))
                logits = (logits + logits_flip) / 2.0
            all_logits.append(logits.float().cpu())
            all_labels.append(labels.cpu())
            n_done += imgs.shape[0]
            if (i + 1) % 25 == 0:
                rate = n_done / (time.time() - t0)
                print(f"  [infer] {n_done:,} ({rate:.1f} img/s)", flush=True)

    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()
    print(f"[infer] done in {time.time()-t0:.1f}s, logits={logits.shape}", flush=True)

    if args.cache:
        Path(args.cache).parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.cache, logits=logits, labels=labels)
        print(f"[cache] wrote {args.cache}", flush=True)

    return logits, labels


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _reliability_cuts(p: np.ndarray, y: np.ndarray, thr: float) -> tuple[float, float]:
    order = np.argsort(-p)
    p_sorted = p[order]
    y_sorted = y[order]
    cum_pos = np.cumsum(y_sorted)
    counts = np.arange(1, len(p_sorted) + 1)
    precision = cum_pos / counts

    def find_cut(target: float, default: float) -> float:
        ok = precision >= target
        if not ok.any():
            return default
        idx = np.where(ok)[0].max()
        return float(p_sorted[idx])

    high_cut = find_cut(PRECISION_HIGH_TARGET, default=max(thr, 0.85))
    low_cut = find_cut(PRECISION_LOW_TARGET, default=thr)
    return low_cut, high_cut


def _percentile_cuts(p: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    pos_p = p[y > 0.5]
    if len(pos_p) == 0:
        return 0.50, 0.85
    return (
        float(np.percentile(pos_p, PCT_LOW_FALLBACK)),
        float(np.percentile(pos_p, PCT_HIGH_FALLBACK)),
    )


def _clamp_band(low_cut: float, high_cut: float, threshold: float) -> tuple[float, float]:
    low_cut = float(np.clip(low_cut, 0.05, 0.99))
    high_cut = float(np.clip(high_cut, 0.05, 0.99))
    if low_cut >= threshold:
        low_cut = max(0.05, threshold - 0.01)
    if high_cut < threshold:
        high_cut = min(0.99, threshold + 0.05)
    if high_cut - low_cut < MIN_BAND_WIDTH:
        mid = (low_cut + high_cut) / 2.0
        low_cut = max(0.05, mid - MIN_BAND_WIDTH / 2)
        high_cut = min(0.99, mid + MIN_BAND_WIDTH / 2)
    return low_cut, high_cut


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    classes: list[str] = cfg["data"]["classes"]

    with open(args.calibration) as f:
        cal = json.load(f)
    temperature = float(cal["temperature"])
    thresholds = np.asarray(cal["thresholds"], dtype=np.float64)

    logits, labels = _run_val_inference(cfg, args)
    cal_probs = _sigmoid(logits / max(temperature, 1e-3))

    bands: list[list[float]] = []
    methods: list[str] = []
    report_rows: list[str] = []
    for c, name in enumerate(classes):
        p = cal_probs[:, c]
        y = labels[:, c]
        n_pos = int(y.sum())
        thr = float(thresholds[c])
        if n_pos >= MIN_POSITIVES_FOR_RELIABILITY:
            low, high = _reliability_cuts(p, y, thr)
            method = "reliability"
        else:
            low, high = _percentile_cuts(p, y)
            method = "percentile"
        low, high = _clamp_band(low, high, thr)
        bands.append([low, high])
        methods.append(method)
        report_rows.append(
            f"{name:<22} n_pos={n_pos:5d}  thr={thr:.3f}  "
            f"low={low:.3f}  high={high:.3f}  [{method}]"
        )

    out = {
        "bands": bands,
        "method": methods,
        "meta": {
            "temperature": temperature,
            "n_val_samples": int(labels.shape[0]),
            "min_positives_for_reliability": MIN_POSITIVES_FOR_RELIABILITY,
            "precision_low_target": PRECISION_LOW_TARGET,
            "precision_high_target": PRECISION_HIGH_TARGET,
            "pct_low_fallback": PCT_LOW_FALLBACK,
            "pct_high_fallback": PCT_HIGH_FALLBACK,
            "min_band_width": MIN_BAND_WIDTH,
        },
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[out] {out_path}", flush=True)

    print("\n" + "=" * 72)
    print("Per-class confidence bands")
    print("=" * 72)
    for row in report_rows:
        print(row)
    print("=" * 72)


if __name__ == "__main__":
    main()
