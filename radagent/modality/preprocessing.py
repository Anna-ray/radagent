"""
radagent.modality.preprocessing
-------------------------------
Preprocessing registry for multiple modalities.

Author: Rayane Aggoune
"""
from __future__ import annotations

from typing import Any, Callable

import cv2
import numpy as np
import torch

from radagent.data.preprocessing import apply_clahe, to_three_channel


# Preprocessing function registry
PREPROCESSING_REGISTRY: dict[str, Callable] = {}


def register_preprocessing(name: str):
    """Decorator to register a preprocessing function."""
    def decorator(func: Callable) -> Callable:
        PREPROCESSING_REGISTRY[name] = func
        return func
    return decorator


@register_preprocessing("cxr_clahe_normalize")
def cxr_clahe_normalize(image: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """CXR preprocessing: CLAHE + normalize (v1 pipeline - DO NOT MODIFY).
    
    Args:
        image: Grayscale image (H, W) or RGB (H, W, 3)
        
    Returns:
        (processed_image, processing_record)
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Apply CLAHE
    image = apply_clahe(image, clip_limit=2.0)
    
    # Convert to 3-channel
    image = to_three_channel(image)
    
    record = {
        "method": "cxr_clahe_normalize",
        "clahe_clip_limit": 2.0,
        "output_channels": 3,
    }
    
    return image, record


@register_preprocessing("bone_xray_normalize")
def bone_xray_normalize(image: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Bone X-ray preprocessing: same as CXR (reuse v1 augmentation).
    
    Args:
        image: Grayscale or RGB image
        
    Returns:
        (processed_image, processing_record)
    """
    # Same as CXR
    return cxr_clahe_normalize(image)


@register_preprocessing("ct_lung_window_hu")
def ct_lung_window_hu(image: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """CT lung windowing: HU clipping + CLAHE.
    
    Args:
        image: CT image in Hounsfield Units (or raw pixel values)
        
    Returns:
        (processed_image, processing_record)
    """
    # Lung window: [-1000, 200] HU
    image = np.clip(image, -1000, 200)
    
    # Normalize to [0, 255]
    image = ((image + 1000) / 1200 * 255).astype(np.uint8)
    
    # Apply CLAHE
    image = apply_clahe(image, clip_limit=2.0)
    
    # Convert to 3-channel
    image = to_three_channel(image)
    
    record = {
        "method": "ct_lung_window_hu",
        "window_min": -1000,
        "window_max": 200,
        "clahe_clip_limit": 2.0,
    }
    
    return image, record


@register_preprocessing("mammo_normalize")
def mammo_normalize(image: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Mammography preprocessing: z-score + resize.
    
    Args:
        image: Mammography image
        
    Returns:
        (processed_image, processing_record)
    """
    # Z-score normalization
    mean = image.mean()
    std = image.std()
    image = (image - mean) / (std + 1e-8)
    
    # Clip to [-3, 3] sigma
    image = np.clip(image, -3, 3)
    
    # Scale to [0, 255]
    image = ((image + 3) / 6 * 255).astype(np.uint8)
    
    # Convert to 3-channel
    image = to_three_channel(image)
    
    record = {
        "method": "mammo_normalize",
        "normalization": "z-score",
    }
    
    return image, record


@register_preprocessing("mri_normalize")
def mri_normalize(image: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """MRI preprocessing: z-score + resize.
    
    Args:
        image: MRI image
        
    Returns:
        (processed_image, processing_record)
    """
    # Same as mammography for now
    return mammo_normalize(image)


@register_preprocessing("us_normalize")
def us_normalize(image: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Ultrasound preprocessing: simple normalization.
    
    Args:
        image: Ultrasound image
        
    Returns:
        (processed_image, processing_record)
    """
    # Simple min-max normalization
    image = image.astype(np.float32)
    image = (image - image.min()) / (image.max() - image.min() + 1e-8)
    image = (image * 255).astype(np.uint8)
    
    # Convert to 3-channel
    image = to_three_channel(image)
    
    record = {
        "method": "us_normalize",
        "normalization": "min-max",
    }
    
    return image, record


@register_preprocessing("generic_normalize")
def generic_normalize(image: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Generic preprocessing: min-max + resize.
    
    Args:
        image: Any image
        
    Returns:
        (processed_image, processing_record)
    """
    # Same as ultrasound
    return us_normalize(image)


def get_preprocessing_function(name: str) -> Callable:
    """Get preprocessing function by name.
    
    Args:
        name: Preprocessing function name
        
    Returns:
        Preprocessing function
    """
    if name not in PREPROCESSING_REGISTRY:
        raise ValueError(f"Unknown preprocessing function: {name}")
    
    return PREPROCESSING_REGISTRY[name]

# Made with Bob
