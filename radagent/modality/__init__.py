"""
radagent.modality
-----------------
Universal DICOM modality routing for RadAgent v2.

Author: Rayane Aggoune
"""
from radagent.modality.router import ModalityRouter, ModalityInfo, RoutingDecision
from radagent.modality.dicom_io import load_dicom

__all__ = [
    "ModalityRouter",
    "ModalityInfo",
    "RoutingDecision",
    "load_dicom",
]

# Made with Bob
