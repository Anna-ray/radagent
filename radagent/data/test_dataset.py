"""
radagent.data.test_dataset
--------------------------
Loads the official NIH ChestX-ray14 test set (test_list.txt) for evaluation.

This is intentionally separate from the training NIHChestXray14 class because:
- We never want test-time CLAHE jitter (deterministic eval only)
- We don't need the WeightedSampler / patient-disjoint logic
- We want explicit failure when test images are missing (no silent skips)
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import albumentations as A
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .preprocessing import apply_clahe, load_cxr_grayscale, to_three_channel


class NIHTestSet(Dataset):
    """Multi-label CXR test set, deterministic CLAHE only."""

    def __init__(
        self,
        labels_csv: str,
        test_split_txt: str,
        images_dir: str,
        classes: Sequence[str],
        eval_transforms: A.Compose,
        clahe_clip: float = 2.5,
    ):
        df = pd.read_csv(labels_csv)
        with open(test_split_txt) as f:
            test_set = {ln.strip() for ln in f if ln.strip()}
        self.df = df[df["Image Index"].isin(test_set)].reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.classes = list(classes)
        self.tfms = eval_transforms
        self.clahe_clip = clahe_clip

        self._path_cache = self._build_path_index()
        self._label_matrix = self._build_label_matrix()

    def _build_path_index(self) -> dict[str, str]:
        idx: dict[str, str] = {}
        for root, _, files in os.walk(self.images_dir):
            for f in files:
                if f.lower().endswith((".png", ".jpg", ".jpeg")):
                    idx[f] = os.path.join(root, f)
        missing = [n for n in self.df["Image Index"] if n not in idx]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} test images listed in CSV but not on disk. "
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
        # Resilience: NIH-14 contains a few corrupt PNGs. Skip them by
        # deterministically advancing to the next index; bound retries.
        max_retries = 5
        attempt_idx = idx
        last_err: Exception | None = None
        for retry in range(max_retries):
            try:
                return self._load_item(attempt_idx)
            except (FileNotFoundError, OSError, ValueError) as e:
                last_err = e
                bad = self.df.iloc[attempt_idx]["Image Index"]
                print(f"[test-dataset] WARN skipping corrupt image '{bad}' "
                      f"(attempt {retry+1}/{max_retries}): {e}")
                attempt_idx = (attempt_idx + 1) % len(self.df)
        raise RuntimeError(
            f"Failed to load any test image after {max_retries} retries "
            f"starting from idx={idx}. Last error: {last_err}"
        )

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

        meta = {
            "image_index": fname,
            "patient_id": int(row["Patient ID"]) if "Patient ID" in row else -1,
        }
        return image, labels, meta

    @property
    def label_matrix(self) -> np.ndarray:
        return self._label_matrix
