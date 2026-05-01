"""
radagent.data.dataset
---------------------
NIH ChestX-ray14 multi-label dataset with:
  - Official train_val / test split (NO leakage)
  - Patient-disjoint train/val sub-split (CRITICAL: same patient must not
    appear in both train and val, otherwise val AUC is inflated by 3-5%)
  - On-the-fly CLAHE with jittered clip limit (training only)
  - Albumentations pipeline (medical-safe)
  - Weighted sampling to combat class imbalance (Hernia ~0.2%)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, WeightedRandomSampler

from .preprocessing import apply_clahe, load_cxr_grayscale, to_three_channel


# ImageNet stats — appropriate because we use ImageNet-pretrained ConvNeXt-V2
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_train_transforms(
    image_size: int,
    affine_deg: float,
    affine_trans: float,
    elastic_alpha: float,
    elastic_sigma: float,
    rrc_scale: tuple[float, float],
    hflip_prob: float,
) -> A.Compose:
    """Albumentations pipeline for training.

    Notes:
      - HFlip is safe on PA/AP frontal CXR; situs inversus is rare enough
        (~0.01%) that it's not a concern for training stats. We do NOT
        flip vertically — that swaps cardiac apex for hepatic dome.
      - No color jitter — these are grayscale tripled to 3ch, color jitter
        would create unrealistic chromatic artifacts.
      - Elastic is mild; aggressive elastic on CXR creates anatomically
        impossible distortions.
    """
    return A.Compose([
        A.LongestMaxSize(max_size=int(image_size * 1.15)),
        A.PadIfNeeded(
            min_height=int(image_size * 1.15),
            min_width=int(image_size * 1.15),
            border_mode=cv2.BORDER_CONSTANT,
            fill=0,
        ),
        A.RandomResizedCrop(
            size=(image_size, image_size),
            scale=rrc_scale,
            ratio=(0.95, 1.05),
        ),
        A.HorizontalFlip(p=hflip_prob),
        A.Affine(
            rotate=(-affine_deg, affine_deg),
            translate_percent=(-affine_trans, affine_trans),
            scale=(0.95, 1.05),
            p=0.5,
            border_mode=cv2.BORDER_CONSTANT,
        ),
        A.ElasticTransform(
            alpha=elastic_alpha,
            sigma=elastic_sigma,
            p=0.2,
        ),
        A.CoarseDropout(
            num_holes_range=(1, 4),
            hole_height_range=(8, 24),
            hole_width_range=(8, 24),
            p=0.25,
        ),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def build_eval_transforms(image_size: int) -> A.Compose:
    return A.Compose([
        A.LongestMaxSize(max_size=image_size),
        A.PadIfNeeded(
            min_height=image_size,
            min_width=image_size,
            border_mode=cv2.BORDER_CONSTANT,
            fill=0,
        ),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


class NIHChestXray14(Dataset):
    """Multi-label CXR dataset.

    Each sample returns:
      image: float tensor [3, H, W] (normalized)
      labels: float tensor [num_classes] in {0, 1}
      meta: dict with image_index, patient_id (for downstream auditing)
    """

    def __init__(
        self,
        labels_df: pd.DataFrame,
        images_dir: str,
        classes: Sequence[str],
        image_size: int,
        is_train: bool,
        clahe_clip_jitter: tuple[float, float] = (1.5, 3.5),
        clahe_clip_eval: float = 2.5,
        train_transforms: A.Compose | None = None,
        eval_transforms: A.Compose | None = None,
    ):
        self.df = labels_df.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.classes = list(classes)
        self.image_size = image_size
        self.is_train = is_train
        self.clahe_clip_jitter = clahe_clip_jitter
        self.clahe_clip_eval = clahe_clip_eval
        self.train_tfms = train_transforms
        self.eval_tfms = eval_transforms

        # Build a fast index → file path map.
        # NIH images can live in images_001/images, images_002/images, ...
        # We pre-resolve once at construction time to avoid per-getitem stat() calls.
        self._path_cache = self._build_path_index()

        # Pre-compute label matrix once
        self._label_matrix = self._build_label_matrix()

    def _build_path_index(self) -> dict[str, str]:
        idx: dict[str, str] = {}
        # Walk once; expensive but only at startup
        for root, _, files in os.walk(self.images_dir):
            for f in files:
                if f.lower().endswith((".png", ".jpg", ".jpeg")):
                    idx[f] = os.path.join(root, f)
        missing = [n for n in self.df["Image Index"] if n not in idx]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} images listed in CSV but not found on disk. "
                f"First missing: {missing[:3]}. Check images_dir."
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
        # Resilience layer: NIH-14 contains rare corrupt PNGs. If one slips
        # past the PIL fallback in load_cxr_grayscale, skip it by retrying on
        # a deterministically chosen neighbor. We bound the retry depth to
        # avoid infinite loops on a hypothetical fully-broken dataset.
        max_retries = 5
        attempt_idx = idx
        last_err: Exception | None = None
        for retry in range(max_retries):
            try:
                return self._load_item(attempt_idx)
            except (FileNotFoundError, OSError, ValueError) as e:
                last_err = e
                bad = self.df.iloc[attempt_idx]["Image Index"]
                print(f"[dataset] WARN skipping corrupt image '{bad}' "
                      f"(attempt {retry+1}/{max_retries}): {e}")
                # Deterministic neighbor: walk forward, wrap around.
                attempt_idx = (attempt_idx + 1) % len(self.df)
        # If we burn 5 retries, something is structurally wrong — surface it.
        raise RuntimeError(
            f"Failed to load any image after {max_retries} retries starting "
            f"from idx={idx}. Last error: {last_err}"
        )

    def _load_item(self, idx: int):
        row = self.df.iloc[idx]
        fname = row["Image Index"]
        path = self._path_cache[fname]

        # Load + CLAHE (jittered for train, fixed for eval)
        gray = load_cxr_grayscale(path)
        if self.is_train:
            clip = float(np.random.uniform(*self.clahe_clip_jitter))
        else:
            clip = self.clahe_clip_eval
        gray = apply_clahe(gray, clip_limit=clip)
        rgb = to_three_channel(gray)  # HxWx3 uint8

        tfms = self.train_tfms if self.is_train else self.eval_tfms
        out = tfms(image=rgb)
        image = out["image"].float()  # [3,H,W]

        labels = torch.from_numpy(self._label_matrix[idx])

        meta = {
            "image_index": fname,
            "patient_id": int(row["Patient ID"]) if "Patient ID" in row else -1,
        }
        return image, labels, meta

    # ---------- helpers ----------
    @property
    def label_matrix(self) -> np.ndarray:
        return self._label_matrix

    def class_pos_counts(self) -> np.ndarray:
        return self._label_matrix.sum(axis=0)


def make_weighted_sampler(label_matrix: np.ndarray) -> WeightedRandomSampler:
    """Per-sample weights inversely proportional to the rarest positive class.

    Rationale: a sample positive for Hernia (rare) gets a high weight; a
    sample positive only for Infiltration (common) gets a lower weight;
    a "No Finding" sample gets the median weight. This is more stable than
    pure inverse-frequency weighting which oversamples noise.
    """
    pos_counts = label_matrix.sum(axis=0)            # [C]
    pos_counts = np.clip(pos_counts, 1, None)
    class_w = 1.0 / np.sqrt(pos_counts)              # sqrt damps extremes
    # Per-sample weight = max class weight among positives, or median for negatives
    sample_w = np.zeros(len(label_matrix), dtype=np.float32)
    median_w = float(np.median(class_w))
    for i in range(len(label_matrix)):
        positives = np.where(label_matrix[i] > 0.5)[0]
        if len(positives) == 0:
            sample_w[i] = median_w
        else:
            sample_w[i] = float(class_w[positives].max())
    return WeightedRandomSampler(
        weights=torch.from_numpy(sample_w).double(),
        num_samples=len(sample_w),
        replacement=True,
    )


def patient_disjoint_split(
    df: pd.DataFrame,
    val_fraction: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split rows into train/val so that no patient appears in both.

    NIH-14 has multiple images per patient (follow-ups). A naive random
    split leaks patient identity into val and inflates AUC by 3-5%.
    """
    rng = np.random.default_rng(seed)
    patients = df["Patient ID"].unique()
    rng.shuffle(patients)
    n_val = int(len(patients) * val_fraction)
    val_pids = set(patients[:n_val].tolist())
    is_val = df["Patient ID"].isin(val_pids)
    return df.loc[~is_val].copy(), df.loc[is_val].copy()


def load_nih14_dataframe(
    labels_csv: str,
    train_split_txt: str,
    test_split_txt: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the master CSV and partition by the official split files."""
    df = pd.read_csv(labels_csv)
    with open(train_split_txt) as f:
        train_val_set = {ln.strip() for ln in f if ln.strip()}
    with open(test_split_txt) as f:
        test_set = {ln.strip() for ln in f if ln.strip()}
    train_val_df = df[df["Image Index"].isin(train_val_set)].copy()
    test_df = df[df["Image Index"].isin(test_set)].copy()
    return train_val_df, test_df
