"""
radagent.data.cxr_datasets
--------------------------
Dataset loaders for federated learning with NIH ChestX-ray14 and CheXpert.

Implements 14-class harmonization with label masking for missing/uncertain labels.

Author: Rayane Aggoune
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import albumentations as A
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from radagent.data.dataset import build_eval_transforms, build_train_transforms
from radagent.data.preprocessing import apply_clahe, load_cxr_grayscale, to_three_channel


# Harmonized 14-class order (IMMUTABLE - matches v1 specialist)
CLASS_NAMES = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Effusion",
    "Emphysema",
    "Fibrosis",
    "Hernia",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pleural_Thickening",
    "Pneumonia",
    "Pneumothorax",
]


class NIH14FederatedDataset(Dataset):
    """NIH ChestX-ray14 dataset for federated learning.
    
    Args:
        root: Path to NIH ChestX-ray14 root directory
        split: 'train' or 'val'
        max_samples: Maximum number of samples to use (for demo)
        image_size: Image size for transforms
        augment: Whether to apply training augmentations
    """
    
    def __init__(
        self,
        root: str | Path,
        split: Literal["train", "val"],
        max_samples: int | None = None,
        image_size: int = 384,
        augment: bool = True,
    ):
        self.root = Path(root)
        self.split = split
        self.image_size = image_size
        
        # Load metadata
        df = pd.read_csv(self.root / "Data_Entry_2017_v2020.csv")
        
        # Use official train_val_list.txt and test_list.txt
        if split == "train":
            train_val_list = (self.root / "train_val_list.txt").read_text().strip().split("\n")
            df = df[df["Image Index"].isin(train_val_list)]
            # Further split train/val by patient ID (80/20)
            patient_ids = df["Patient ID"].unique()
            np.random.seed(42)
            np.random.shuffle(patient_ids)
            train_patients = patient_ids[:int(0.8 * len(patient_ids))]
            df = df[df["Patient ID"].isin(train_patients)]
        elif split == "val":
            train_val_list = (self.root / "train_val_list.txt").read_text().strip().split("\n")
            df = df[df["Image Index"].isin(train_val_list)]
            # Val patients
            patient_ids = df["Patient ID"].unique()
            np.random.seed(42)
            np.random.shuffle(patient_ids)
            val_patients = patient_ids[int(0.8 * len(patient_ids)):]
            df = df[df["Patient ID"].isin(val_patients)]
        else:
            raise ValueError(f"Invalid split: {split}")
        
        # Limit samples if requested
        if max_samples is not None:
            df = df.sample(n=min(max_samples, len(df)), random_state=42)
        
        self.df = df.reset_index(drop=True)
        
        # Transforms
        if augment and split == "train":
            self.transform = build_train_transforms(
                image_size=image_size,
                affine_deg=5.0,
                affine_trans=0.05,
                elastic_alpha=20.0,
                elastic_sigma=5.0,
                rrc_scale=(0.9, 1.0),
                hflip_prob=0.5,
            )
        else:
            self.transform = build_eval_transforms(image_size)
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        
        # Load image
        image_path = self.root / "images" / row["Image Index"]
        image = load_cxr_grayscale(str(image_path))
        image = apply_clahe(image, clip_limit=2.0)
        image = to_three_channel(image)
        
        # Parse labels
        finding_labels = row["Finding Labels"]
        labels = np.zeros(14, dtype=np.float32)
        label_mask = np.ones(14, dtype=np.float32)  # All classes present in NIH-14
        
        if finding_labels != "No Finding":
            findings = finding_labels.split("|")
            for finding in findings:
                if finding in CLASS_NAMES:
                    idx_class = CLASS_NAMES.index(finding)
                    labels[idx_class] = 1.0
        
        # Apply transforms
        transformed = self.transform(image=image)
        image_tensor = transformed["image"]
        
        return {
            "image": image_tensor,
            "labels": torch.from_numpy(labels),
            "label_mask": torch.from_numpy(label_mask),
        }


class CheXpertFederatedDataset(Dataset):
    """CheXpert dataset for federated learning with label harmonization.
    
    CheXpert has 14 classes but different names. We map to NIH-14 schema:
    - "Lung Opacity" -> Infiltration
    - "Pleural Effusion" -> Effusion
    - Direct: Atelectasis, Cardiomegaly, Consolidation, Edema, Pneumonia, Pneumothorax
    - Missing: Hernia, Fibrosis, Emphysema, Mass, Nodule, Pleural_Thickening
    - Uncertain (-1): label_mask = 0
    
    Args:
        root: Path to CheXpert root directory
        split: 'train' or 'val'
        max_samples: Maximum number of samples to use
        image_size: Image size for transforms
        augment: Whether to apply training augmentations
    """
    
    def __init__(
        self,
        root: str | Path,
        split: Literal["train", "val"],
        max_samples: int | None = None,
        image_size: int = 384,
        augment: bool = True,
    ):
        self.root = Path(root)
        self.split = split
        self.image_size = image_size
        
        # Load metadata
        if split == "train":
            csv_path = self.root / "train.csv"
        else:
            csv_path = self.root / "valid.csv"
        
        df = pd.read_csv(csv_path)
        
        # Filter to frontal views only
        df = df[df["Frontal/Lateral"] == "Frontal"]
        
        # Limit samples if requested
        if max_samples is not None:
            df = df.sample(n=min(max_samples, len(df)), random_state=42)
        
        self.df = df.reset_index(drop=True)
        
        # CheXpert column mapping to NIH-14 schema
        self.chexpert_to_nih = {
            "Atelectasis": "Atelectasis",
            "Cardiomegaly": "Cardiomegaly",
            "Consolidation": "Consolidation",
            "Edema": "Edema",
            "Pleural Effusion": "Effusion",
            "Pneumonia": "Pneumonia",
            "Pneumothorax": "Pneumothorax",
            "Lung Opacity": "Infiltration",
        }
        
        # Transforms
        if augment and split == "train":
            self.transform = build_train_transforms(
                image_size=image_size,
                affine_deg=5.0,
                affine_trans=0.05,
                elastic_alpha=20.0,
                elastic_sigma=5.0,
                rrc_scale=(0.9, 1.0),
                hflip_prob=0.5,
            )
        else:
            self.transform = build_eval_transforms(image_size)
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        
        # Load image
        image_path = self.root / row["Path"]
        image = load_cxr_grayscale(str(image_path))
        image = apply_clahe(image, clip_limit=2.0)
        image = to_three_channel(image)
        
        # Parse labels with harmonization
        labels = np.zeros(14, dtype=np.float32)
        label_mask = np.zeros(14, dtype=np.float32)  # Start with all masked
        
        for chexpert_col, nih_class in self.chexpert_to_nih.items():
            if chexpert_col in row:
                value = row[chexpert_col]
                nih_idx = CLASS_NAMES.index(nih_class)
                
                if pd.notna(value):
                    if value == 1.0:
                        labels[nih_idx] = 1.0
                        label_mask[nih_idx] = 1.0
                    elif value == 0.0:
                        labels[nih_idx] = 0.0
                        label_mask[nih_idx] = 1.0
                    # value == -1.0 (uncertain): keep mask = 0
        
        # Apply transforms
        transformed = self.transform(image=image)
        image_tensor = transformed["image"]
        
        return {
            "image": image_tensor,
            "labels": torch.from_numpy(labels),
            "label_mask": torch.from_numpy(label_mask),
        }


def build_nih_loader(
    root: str | Path,
    n: int,
    split: Literal["train", "val"] = "train",
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 384,
) -> DataLoader:
    """Build NIH-14 data loader for federated learning.
    
    Args:
        root: Path to NIH ChestX-ray14 root
        n: Number of samples to use
        split: 'train' or 'val'
        batch_size: Batch size
        num_workers: Number of data loading workers
        image_size: Image size
        
    Returns:
        DataLoader
    """
    dataset = NIH14FederatedDataset(
        root=root,
        split=split,
        max_samples=n,
        image_size=image_size,
        augment=(split == "train"),
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=True,
    )


def build_chexpert_loader(
    root: str | Path,
    n: int,
    split: Literal["train", "val"] = "train",
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 384,
) -> DataLoader:
    """Build CheXpert data loader for federated learning.
    
    Args:
        root: Path to CheXpert root
        n: Number of samples to use
        split: 'train' or 'val'
        batch_size: Batch size
        num_workers: Number of data loading workers
        image_size: Image size
        
    Returns:
        DataLoader
    """
    dataset = CheXpertFederatedDataset(
        root=root,
        split=split,
        max_samples=n,
        image_size=image_size,
        augment=(split == "train"),
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=True,
    )


def build_global_test_loader(
    root: str | Path,
    n: int | None = None,
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 384,
) -> DataLoader:
    """Build global test loader (CheXpert validation set).
    
    Args:
        root: Path to CheXpert root
        n: Number of samples (None = all)
        batch_size: Batch size
        num_workers: Number of workers
        image_size: Image size
        
    Returns:
        DataLoader
    """
    dataset = CheXpertFederatedDataset(
        root=root,
        split="val",
        max_samples=n,
        image_size=image_size,
        augment=False,
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

# Made with Bob
