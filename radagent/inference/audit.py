"""
radagent.inference.audit
------------------------
Self-audit layer for RadAgent reports.

Runs AFTER the VLM produces its report and flags claims that disagree with:
1. Specialist probabilities (numerical audit)
2. Grad-CAM spatial attention (spatial audit)
3. Retrieved passage content (citation audit)

This provides a second layer of grounding verification to catch VLM hallucinations
that slip through the initial grounding prompt.
"""
from __future__ import annotations

import re
from typing import Any

import cv2
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Anatomical region parsing
# ---------------------------------------------------------------------------
ANATOMICAL_REGIONS = {
    # Vertical zones
    "upper": ["upper", "apex", "apical", "superior"],
    "mid": ["mid", "middle", "hilar", "central"],
    "lower": ["lower", "base", "basal", "inferior", "costophrenic"],
    # Lateral zones
    "left": ["left"],
    "right": ["right"],
    "center": ["central", "mediastinal", "cardiac", "heart"],
}


def _parse_anatomical_region(claim: str) -> tuple[str | None, str | None]:
    """
    Extract vertical (upper/mid/lower) and lateral (left/center/right) zones from claim.
    
    Returns:
        (vertical, lateral) or (None, None) if no region mentioned
    """
    claim_lower = claim.lower()
    
    vertical = None
    for zone, keywords in [("upper", ANATOMICAL_REGIONS["upper"]),
                           ("mid", ANATOMICAL_REGIONS["mid"]),
                           ("lower", ANATOMICAL_REGIONS["lower"])]:
        if any(kw in claim_lower for kw in keywords):
            vertical = zone
            break
    
    lateral = None
    for zone, keywords in [("left", ANATOMICAL_REGIONS["left"]),
                           ("right", ANATOMICAL_REGIONS["right"]),
                           ("center", ANATOMICAL_REGIONS["center"])]:
        if any(kw in claim_lower for kw in keywords):
            lateral = zone
            break
    
    return vertical, lateral


def _gradcam_peak_region(gradcam_array: np.ndarray, image_shape: tuple[int, int]) -> tuple[str, str]:
    """
    Find the 3x3 grid cell containing the Grad-CAM peak.
    
    Args:
        gradcam_array: 2D heatmap (H, W)
        image_shape: Original image (H, W)
        
    Returns:
        (vertical_zone, lateral_zone) e.g. ("upper", "left")
    """
    # Resize gradcam to match image if needed
    if gradcam_array.shape != image_shape:
        gradcam_array = cv2.resize(gradcam_array, (image_shape[1], image_shape[0]))
    
    # Find peak location
    peak_idx = np.argmax(gradcam_array)
    peak_y, peak_x = np.unravel_index(peak_idx, gradcam_array.shape)
    
    h, w = gradcam_array.shape
    
    # Vertical zone (3 rows)
    if peak_y < h / 3:
        vertical = "upper"
    elif peak_y < 2 * h / 3:
        vertical = "mid"
    else:
        vertical = "lower"
    
    # Lateral zone (3 columns)
    if peak_x < w / 3:
        lateral = "left"
    elif peak_x < 2 * w / 3:
        lateral = "center"
    else:
        lateral = "right"
    
    return vertical, lateral


# ---------------------------------------------------------------------------
# Audit functions
# ---------------------------------------------------------------------------
def audit_numerical(
    claim: str,
    finding_label: str,
    specialist_probs: dict[str, float],
    threshold: float = 0.5,
) -> dict[str, Any]:
    """
    Audit: Does the VLM claim match the specialist's probability?
    
    If the VLM asserts a finding but the specialist probability is below threshold,
    flag as a potential hallucination.
    
    Args:
        claim: Text claim from VLM report
        finding_label: Finding name (e.g., "Cardiomegaly")
        specialist_probs: Dict mapping finding names to calibrated probabilities
        threshold: Minimum probability to consider finding present
        
    Returns:
        {
            "passed": bool,
            "reason": str,
            "severity": float (0.0-1.0, higher = more severe disagreement)
        }
    """
    prob = specialist_probs.get(finding_label, 0.0)
    
    # Check if claim asserts the finding (positive assertion)
    claim_lower = claim.lower()
    finding_lower = finding_label.lower().replace("_", " ")
    
    # Simple heuristic: if finding name appears in claim, assume assertion
    # (More sophisticated: use NLP to detect negation, but keep it simple)
    asserts_finding = finding_lower in claim_lower
    
    if asserts_finding and prob < threshold:
        # VLM says yes, specialist says no
        severity = 1.0 - prob  # Higher severity when prob is very low
        return {
            "passed": False,
            "reason": f"VLM asserts '{finding_label}' but specialist probability is {prob:.3f} (< {threshold})",
            "severity": severity,
        }
    elif not asserts_finding and prob >= threshold:
        # VLM says no, specialist says yes (missed finding)
        severity = prob  # Higher severity when prob is very high
        return {
            "passed": False,
            "reason": f"VLM does not mention '{finding_label}' but specialist probability is {prob:.3f} (>= {threshold})",
            "severity": severity,
        }
    else:
        return {
            "passed": True,
            "reason": f"Agreement: specialist p={prob:.3f}, threshold={threshold}",
            "severity": 0.0,
        }


def audit_spatial(
    claim: str,
    finding_label: str,
    gradcam_array: np.ndarray | None,
    image_shape: tuple[int, int],
) -> dict[str, Any]:
    """
    Audit: Does the VLM's anatomical localization match Grad-CAM attention?
    
    Parse anatomical region from claim (e.g., "left lower lobe") and compare
    to the 3x3 grid cell containing the Grad-CAM peak.
    
    Args:
        claim: Text claim from VLM report
        finding_label: Finding name
        gradcam_array: 2D heatmap (H, W) or None if not available
        image_shape: Original image (H, W)
        
    Returns:
        {
            "passed": bool,
            "reason": str,
            "severity": float
        }
    """
    if gradcam_array is None:
        return {
            "passed": True,
            "reason": "No Grad-CAM available for spatial audit",
            "severity": 0.0,
        }
    
    # Parse claim for anatomical region
    claim_vert, claim_lat = _parse_anatomical_region(claim)
    
    if claim_vert is None and claim_lat is None:
        # Claim doesn't specify location
        return {
            "passed": True,
            "reason": "Claim does not specify anatomical location",
            "severity": 0.0,
        }
    
    # Find Grad-CAM peak region
    peak_vert, peak_lat = _gradcam_peak_region(gradcam_array, image_shape)
    
    # Check agreement
    vert_match = (claim_vert is None) or (claim_vert == peak_vert)
    lat_match = (claim_lat is None) or (claim_lat == peak_lat)
    
    if vert_match and lat_match:
        return {
            "passed": True,
            "reason": f"Spatial agreement: claim=({claim_vert},{claim_lat}), peak=({peak_vert},{peak_lat})",
            "severity": 0.0,
        }
    else:
        # Severity: 0.5 if one dimension mismatches, 1.0 if both
        severity = 0.5 if (vert_match or lat_match) else 1.0
        return {
            "passed": False,
            "reason": f"Spatial mismatch: claim=({claim_vert},{claim_lat}), Grad-CAM peak=({peak_vert},{peak_lat})",
            "severity": severity,
        }


def audit_citation(
    claim: str,
    citation_id: int,
    retrieved_passages: list[dict[str, Any]],
    embedder: Any,
    threshold: float = 0.4,
) -> dict[str, Any]:
    """
    Audit: Does the cited passage actually support the claim?
    
    Embed both claim and passage with BGE-M3, compute cosine similarity.
    Flag if similarity is below threshold (weak citation).
    
    Args:
        claim: Text claim from VLM report
        citation_id: Citation number (1-indexed)
        retrieved_passages: List of passage dicts with "text" key
        embedder: BGE-M3 model or similar with .encode() method
        threshold: Minimum cosine similarity to consider citation valid
        
    Returns:
        {
            "passed": bool,
            "reason": str,
            "severity": float
        }
    """
    # Get passage by citation ID (1-indexed)
    if citation_id < 1 or citation_id > len(retrieved_passages):
        return {
            "passed": False,
            "reason": f"Citation [{citation_id}] out of range (only {len(retrieved_passages)} passages)",
            "severity": 1.0,
        }
    
    passage = retrieved_passages[citation_id - 1]
    passage_text = passage.get("text", "")
    
    if not passage_text:
        return {
            "passed": False,
            "reason": f"Citation [{citation_id}] points to empty passage",
            "severity": 1.0,
        }
    
    # Embed claim and passage
    try:
        claim_emb = embedder.encode(
            claim,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        passage_emb = embedder.encode(
            passage_text,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        
        # Cosine similarity (embeddings are normalized, both are numpy arrays)
        similarity = float(np.dot(claim_emb, passage_emb))
        
        if similarity < threshold:
            severity = 1.0 - similarity  # Higher severity for lower similarity
            return {
                "passed": False,
                "reason": f"Weak citation: similarity={similarity:.3f} < {threshold}",
                "severity": severity,
            }
        else:
            return {
                "passed": True,
                "reason": f"Strong citation: similarity={similarity:.3f}",
                "severity": 0.0,
            }
    except Exception as e:
        return {
            "passed": False,
            "reason": f"Citation audit failed: {str(e)}",
            "severity": 0.5,
        }


# ---------------------------------------------------------------------------
# Full report audit
# ---------------------------------------------------------------------------
def _parse_claims_with_citations(report_text: str) -> list[dict[str, Any]]:
    """
    Parse VLM report into individual claims with their citations.
    
    Simple heuristic:
    - Split by sentence (. or newline)
    - Extract citation numbers [1], [2], etc.
    
    Returns:
        List of {"claim": str, "citations": list[int]}
    """
    # Split into sentences
    sentences = re.split(r'[.\n]+', report_text)
    
    claims = []
    for sent in sentences:
        sent = sent.strip()
        if not sent or len(sent) < 10:
            continue
        
        # Extract citation numbers [1], [2], [3]
        citations = [int(m) for m in re.findall(r'\[(\d+)\]', sent)]
        
        claims.append({
            "claim": sent,
            "citations": citations,
        })
    
    return claims


def audit_report(
    report_text: str,
    specialist_probs: dict[str, float],
    gradcam_dict: dict[str, np.ndarray],
    retrieved_passages: list[dict[str, Any]],
    embedder: Any,
    image_shape: tuple[int, int] = (384, 384),
    numerical_threshold: float = 0.5,
    citation_threshold: float = 0.4,
) -> dict[str, Any]:
    """
    Audit the entire VLM report.
    
    For each claim:
    1. Run numerical audit (check against specialist probs)
    2. Run spatial audit (check against Grad-CAM)
    3. Run citation audit (check against retrieved passages)
    
    Args:
        report_text: Full VLM-generated report
        specialist_probs: Dict of finding -> calibrated probability
        gradcam_dict: Dict of finding -> 2D heatmap array
        retrieved_passages: List of passage dicts
        embedder: BGE-M3 or similar
        image_shape: Original image dimensions (H, W)
        numerical_threshold: Threshold for numerical audit
        citation_threshold: Threshold for citation audit
        
    Returns:
        {
            "claims": [
                {
                    "claim": str,
                    "citations": list[int],
                    "audits": {
                        "numerical": {...},
                        "spatial": {...},
                        "citation": {...}
                    },
                    "any_flagged": bool,
                    "max_severity": float
                }
            ],
            "summary": {
                "total_claims": int,
                "flagged_claims": int,
                "flag_rate": float
            }
        }
    """
    claims = _parse_claims_with_citations(report_text)
    
    results = []
    for claim_obj in claims:
        claim = claim_obj["claim"]
        citations = claim_obj["citations"]
        
        audits = {}
        
        # 1. Numerical audit: check each finding mentioned in claim
        numerical_results = []
        for finding_label in specialist_probs.keys():
            if finding_label.lower().replace("_", " ") in claim.lower():
                audit_result = audit_numerical(
                    claim, finding_label, specialist_probs, numerical_threshold
                )
                numerical_results.append({
                    "finding": finding_label,
                    **audit_result
                })
        
        audits["numerical"] = numerical_results
        
        # 2. Spatial audit: check each finding with Grad-CAM
        spatial_results = []
        for finding_label, gradcam_array in gradcam_dict.items():
            if finding_label.lower().replace("_", " ") in claim.lower():
                audit_result = audit_spatial(
                    claim, finding_label, gradcam_array, image_shape
                )
                spatial_results.append({
                    "finding": finding_label,
                    **audit_result
                })
        
        audits["spatial"] = spatial_results
        
        # 3. Citation audit: check each citation
        citation_results = []
        for cit_id in citations:
            audit_result = audit_citation(
                claim, cit_id, retrieved_passages, embedder, citation_threshold
            )
            citation_results.append({
                "citation_id": cit_id,
                **audit_result
            })
        
        audits["citation"] = citation_results
        
        # Aggregate: any audit failed?
        all_audit_results = (
            numerical_results + spatial_results + citation_results
        )
        any_flagged = any(not a["passed"] for a in all_audit_results)
        max_severity = max(
            (a["severity"] for a in all_audit_results),
            default=0.0
        )
        
        results.append({
            "claim": claim,
            "citations": citations,
            "audits": audits,
            "any_flagged": any_flagged,
            "max_severity": max_severity,
        })
    
    # Summary
    flagged_count = sum(1 for r in results if r["any_flagged"])
    
    return {
        "claims": results,
        "summary": {
            "total_claims": len(results),
            "flagged_claims": flagged_count,
            "flag_rate": flagged_count / len(results) if results else 0.0,
        }
    }

# Made with Bob
