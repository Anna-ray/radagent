"""
radagent.data.preprocessing
---------------------------
CXR-specific image preprocessing.

Design decisions (justified):

1. We DROP Sobel/Frangi from the input channel stack. Modern ConvNeXt-V2
   learns these in layer 1, and replacing RGB channels with hand-crafted
   edges destroys ImageNet/FCMAE pretraining alignment. Empirically this
   *hurts* AUC on CheXpert/NIH (see Pham et al. 2021; Cohen et al. 2022).
   If we ever want edge supervision, it should be an auxiliary head, not
   an input modification.

2. We DO use CLAHE — it's the one preprocessing trick that consistently
   helps on CXR because the dynamic range of raw DICOMs (and even PNGs
   exported from them) is highly variable across institutions.

3. The CLAHE-enhanced grayscale is replicated to 3 channels so we can
   leverage ImageNet-pretrained backbones without modifying the stem.

4. CLAHE clip-limit is jittered during training as a domain-randomization
   augmentation — different hospitals export with different windowing.
"""

from __future__ import annotations

import cv2
import numpy as np


def load_cxr_grayscale(path: str) -> np.ndarray:
    """Load a chest X-ray as a uint8 grayscale array.

    Handles:
    - 8-bit PNG (NIH ChestX-ray14 default)
    - 16-bit PNG (some MIMIC-CXR exports) → rescaled to 8-bit
    - 3-channel PNG (some sources) → converted to grayscale

    Robustness: NIH ChestX-ray14 contains a small number of corrupt PNGs
    (typically with bad libpng adaptive filter values). cv2.imread returns
    None for these. We fall back to PIL (which is more permissive) and
    only fail if both decoders refuse the file. This prevents a single
    bad file from killing a multi-hour training run.
    """
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        # Fallback: PIL is more lenient about malformed PNG filter bytes.
        try:
            from PIL import Image, ImageFile
            ImageFile.LOAD_TRUNCATED_IMAGES = True  # accept partial files
            with Image.open(path) as pil_img:
                pil_img.load()
                if pil_img.mode != "L":
                    pil_img = pil_img.convert("L")
                img = np.array(pil_img, dtype=np.uint8)
        except Exception:
            img = None
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")

    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if img.dtype == np.uint16:
        # Robust min-max to 8-bit; ignore extreme outliers (DICOM padding)
        lo, hi = np.percentile(img, [0.5, 99.5])
        img = np.clip((img.astype(np.float32) - lo) / max(hi - lo, 1.0), 0, 1)
        img = (img * 255).astype(np.uint8)

    return img


def apply_clahe(
    img_u8: np.ndarray,
    clip_limit: float = 2.5,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Apply CLAHE to a uint8 grayscale image. Returns uint8."""
    if img_u8.dtype != np.uint8:
        raise ValueError(f"CLAHE expects uint8, got {img_u8.dtype}")
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(img_u8)


def to_three_channel(img_u8: np.ndarray) -> np.ndarray:
    """Stack a single-channel image into 3 channels (HxWx3, uint8).

    We replicate the same CLAHE-enhanced channel three times. This preserves
    ImageNet RGB statistics (since R≈G≈B for grayscale inputs the model has
    seen during pretraining on grayscale-ish images like documents) and
    avoids modifying the backbone stem. Cheaper than building 3 different
    preprocessings (e.g. raw + CLAHE + edges) — and ablations show that
    extra hand-crafted channels do not help with strong backbones.
    """
    if img_u8.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {img_u8.shape}")
    return np.stack([img_u8, img_u8, img_u8], axis=-1)


def preprocess_cxr(
    path: str,
    clahe_clip: float = 2.5,
) -> np.ndarray:
    """Full deterministic preprocessing: load → CLAHE → 3-channel uint8.

    Used for validation, test, and inference. Training uses a jittered
    CLAHE clip drawn from a uniform distribution (see dataset.py).
    """
    g = load_cxr_grayscale(path)
    g = apply_clahe(g, clip_limit=clahe_clip)
    return to_three_channel(g)
