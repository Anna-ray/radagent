"""
MURA Bone X-ray Specialist Module

Stanford MURA-v1.1 musculoskeletal radiograph abnormality detection.
Status: REGISTERED (placeholder for v2.1 production deployment)

Author: Rayane Aggoune
"""

from .infer import predict, MURAInferenceResult

__all__ = ["predict", "MURAInferenceResult"]

# Made with Bob
