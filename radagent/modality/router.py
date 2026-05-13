"""
radagent.modality.router
------------------------
Universal modality router with graceful fallback.

Author: Rayane Aggoune
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from radagent.modality.dicom_io import load_dicom, is_dicom_file


@dataclass
class ModalityInfo:
    """Identified modality information."""
    modality: str
    body_part: str | None
    confidence: float
    method: str  # "dicom_tag" or "filename_heuristic"


@dataclass
class RoutingDecision:
    """Routing decision for a study."""
    matched_entry: str  # Key from registry
    specialist_path: str | None
    rag_corpus: str | None
    preprocessing: str
    autonomy_tools: list[str]
    status: str  # "production", "registered", "vlm_only_fallback"
    fallback_reason: str | None
    modality_info: ModalityInfo


class ModalityRouter:
    """Universal modality router.
    
    Identifies modality from DICOM tags or filename, routes to appropriate
    specialist + RAG + tools, or gracefully falls back to VLM-only.
    
    Args:
        registry_path: Path to registry.yaml
    """
    
    def __init__(self, registry_path: str | Path = None):
        if registry_path is None:
            # Default to registry in same directory
            registry_path = Path(__file__).parent / "registry.yaml"
        
        self.registry_path = Path(registry_path)
        
        # Load registry
        with open(self.registry_path) as f:
            self.registry = yaml.safe_load(f)
    
    def identify(self, path: str | Path) -> ModalityInfo:
        """Identify modality from DICOM or image file.
        
        Args:
            path: Path to DICOM or image file
            
        Returns:
            ModalityInfo with modality, body part, confidence
        """
        path = Path(path)
        
        # Try DICOM first
        if is_dicom_file(path):
            try:
                dicom_data = load_dicom(path)
                metadata = dicom_data["metadata"]
                
                return ModalityInfo(
                    modality=metadata.modality,
                    body_part=metadata.body_part,
                    confidence=1.0,
                    method="dicom_tag",
                )
            except Exception as e:
                print(f"[router] DICOM read failed: {e}")
        
        # Fallback: filename heuristic
        filename = path.stem.lower()
        
        if "chest" in filename or "cxr" in filename:
            return ModalityInfo(
                modality="CR",
                body_part="CHEST",
                confidence=0.7,
                method="filename_heuristic",
            )
        elif "bone" in filename or "mura" in filename:
            return ModalityInfo(
                modality="CR",
                body_part="WRIST",  # Default
                confidence=0.6,
                method="filename_heuristic",
            )
        else:
            return ModalityInfo(
                modality="OT",
                body_part=None,
                confidence=0.3,
                method="filename_heuristic",
            )
    
    def route(self, path: str | Path) -> RoutingDecision:
        """Route a study to appropriate pipeline.
        
        Args:
            path: Path to DICOM or image file
            
        Returns:
            RoutingDecision with matched entry and pipeline config
        """
        # Identify modality
        modality_info = self.identify(path)
        
        # Match against registry
        matched_entry = None
        
        for entry_name, entry_config in self.registry.items():
            # Check modality match
            if modality_info.modality in entry_config["dicom_modalities"]:
                # Check body part filter (if specified)
                body_part_filters = entry_config.get("body_part_filters", [])
                
                if not body_part_filters:
                    # No filter = match
                    matched_entry = entry_name
                    break
                elif modality_info.body_part and modality_info.body_part in body_part_filters:
                    # Body part matches
                    matched_entry = entry_name
                    break
        
        # If no match, use "other" fallback
        if matched_entry is None:
            matched_entry = "other"
            fallback_reason = f"No registry match for modality={modality_info.modality}, body_part={modality_info.body_part}"
        else:
            fallback_reason = None
        
        # Get entry config
        entry_config = self.registry[matched_entry]
        
        # Build routing decision
        return RoutingDecision(
            matched_entry=matched_entry,
            specialist_path=entry_config.get("specialist_weights"),
            rag_corpus=entry_config.get("rag_corpus"),
            preprocessing=entry_config["preprocessing"],
            autonomy_tools=entry_config.get("autonomy_tools", []),
            status=entry_config["status"],
            fallback_reason=fallback_reason,
            modality_info=modality_info,
        )
    
    def graceful_fallback(self, modality: str) -> RoutingDecision:
        """Graceful fallback for unknown modality.
        
        Args:
            modality: Modality code
            
        Returns:
            RoutingDecision with VLM-only fallback
        """
        modality_info = ModalityInfo(
            modality=modality,
            body_part=None,
            confidence=0.0,
            method="fallback",
        )
        
        entry_config = self.registry["other"]
        
        return RoutingDecision(
            matched_entry="other",
            specialist_path=None,
            rag_corpus=entry_config.get("rag_corpus"),
            preprocessing=entry_config["preprocessing"],
            autonomy_tools=[],
            status="vlm_only_fallback",
            fallback_reason=f"Unknown modality: {modality}",
            modality_info=modality_info,
        )

# Made with Bob
