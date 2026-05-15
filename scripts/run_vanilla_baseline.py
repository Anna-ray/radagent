#!/usr/bin/env python3
"""
RadAgent v2 — Vanilla Baseline Script (Priority 3)
Author: Rayane Aggoune

Generates UNGROUNDED vanilla VLM output for side-by-side comparison.
Uses Featherless API (Qwen2.5-VL) with NO specialist, NO RAG, NO citations.

This is the "before" in the demo's "before/after" comparison.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any
import hashlib
import base64

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from radagent.models.specialist import SpecialistCXR
from radagent.data.preprocessing import load_cxr_grayscale, apply_clahe, to_three_channel
from radagent.data.dataset import build_eval_transforms
import torch
import yaml

# Featherless API (OpenAI-compatible)
try:
    from openai import OpenAI
except ImportError:
    print("ERROR: openai package not installed. Run: pip install openai")
    sys.exit(1)


# 14-class order (immutable, matches v1)
CLASS_NAMES = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Effusion", "Emphysema", "Fibrosis", "Hernia",
    "Infiltration", "Mass", "Nodule", "Pleural_Thickening",
    "Pneumonia", "Pneumothorax"
]


def load_specialist_for_comparison(checkpoint_path: str, config_path: str, device: str) -> SpecialistCXR:
    """Load v1 specialist to compare vanilla claims against ground truth."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = SpecialistCXR(
        timm_name=cfg["model"]["name"],
        num_classes=14,
        pretrained=False,
        drop_path_rate=cfg["model"]["drop_path_rate"],
        grad_checkpointing=False,
    )
    state_key = "ema" if "ema" in ckpt else "model"
    model.load_state_dict(ckpt[state_key])
    model = model.to(device).eval()
    return model


def load_calibration_thresholds(calibration_path: str) -> List[float]:
    """Load per-class thresholds from v1 calibration."""
    with open(calibration_path, "r") as f:
        calib = json.load(f)
    return calib["thresholds"]


def encode_image_base64(image_path: str) -> str:
    """Encode image to base64 for API."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def call_vanilla_vlm(image_path: str, api_key: str) -> Dict[str, Any]:
    """
    Call Featherless Qwen2.5-VL with NO grounding scaffolding.
    Pure image-to-text, no specialist, no RAG, no citations.
    """
    client = OpenAI(
        base_url="https://api.featherless.ai/v1",
        api_key=api_key
    )
    
    # Simple radiologist prompt (no grounding instructions)
    prompt = (
        "You are a radiologist. Describe what you see in this chest X-ray. "
        "Provide findings and impression."
    )
    
    # Encode image
    image_b64 = encode_image_base64(image_path)
    
    start_time = time.time()
    
    try:
        response = client.chat.completions.create(
            model="Qwen/Qwen2.5-VL-7B-Instruct",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}
                        }
                    ]
                }
            ],
            temperature=0.7,
            max_tokens=500
        )
        
        processing_time_ms = int((time.time() - start_time) * 1000)
        
        full_text = response.choices[0].message.content.strip()
        
        return {
            "full_text": full_text,
            "processing_time_ms": processing_time_ms,
            "model": "Qwen/Qwen2.5-VL-7B-Instruct",
            "prompt": prompt
        }
        
    except Exception as e:
        print(f"ERROR calling Featherless API: {e}")
        sys.exit(1)


def parse_report_sections(full_text: str) -> Dict[str, str]:
    """
    Parse VLM output into findings and impression.
    Simple heuristic: look for "FINDINGS:" and "IMPRESSION:" headers.
    """
    findings = ""
    impression = ""
    
    lines = full_text.split("\n")
    current_section = None
    
    for line in lines:
        line_upper = line.strip().upper()
        
        if line_upper.startswith("FINDINGS"):
            current_section = "findings"
            continue
        elif line_upper.startswith("IMPRESSION"):
            current_section = "impression"
            continue
        
        if current_section == "findings":
            findings += line.strip() + " "
        elif current_section == "impression":
            impression += line.strip() + " "
    
    # Fallback: if no sections found, treat entire text as findings
    if not findings and not impression:
        findings = full_text
    
    return {
        "findings": findings.strip(),
        "impression": impression.strip(),
        "full_text": full_text
    }


def detect_fabricated_claims(
    report: Dict[str, str],
    specialist_probs: torch.Tensor,
    thresholds: List[float]
) -> List[Dict[str, Any]]:
    """
    Detect claims in vanilla report that contradict specialist predictions.
    
    Heuristic:
    - Extract sentences from findings/impression
    - Match sentences to class names (case-insensitive substring)
    - If VLM states finding as fact BUT specialist confidence < threshold → fabricated
    - If VLM hedges ("possible", "may") → medium risk
    """
    fabricated = []
    
    # Combine findings and impression
    text = report["findings"] + " " + report["impression"]
    
    # Split into sentences (simple heuristic)
    sentences = [s.strip() for s in text.replace(".", ".\n").split("\n") if s.strip()]
    
    # Hedging words (reduce hallucination risk)
    hedging_words = ["possible", "may", "suggest", "could", "likely", "probable", "consider"]
    
    for sentence in sentences:
        sentence_lower = sentence.lower()
        
        # Check each class
        for idx, class_name in enumerate(CLASS_NAMES):
            class_lower = class_name.lower()
            
            # Match class name in sentence
            if class_lower in sentence_lower:
                specialist_conf = specialist_probs[idx].item()
                threshold = thresholds[idx]
                
                # Only flag if specialist says NO (below threshold) but VLM says YES
                if specialist_conf < threshold:
                    # Check if hedged
                    is_hedged = any(hedge in sentence_lower for hedge in hedging_words)
                    
                    # Determine risk
                    if is_hedged:
                        risk = "medium"
                        confidence = "hedged"
                    else:
                        risk = "high"
                        confidence = "stated_as_fact"
                    
                    # Build reason
                    reason = (
                        f"Specialist confidence on {class_name} is {specialist_conf:.2f}, "
                        f"below {threshold:.2f} threshold — "
                    )
                    
                    if is_hedged:
                        reason += "hedged language reduces risk but still unsupported"
                    else:
                        reason += "claim stated as fact without evidence"
                    
                    # Check for laterality (adds specificity without evidence)
                    if any(word in sentence_lower for word in ["right", "left", "bilateral"]):
                        reason += ". Laterality claim adds specificity without evidence"
                    
                    fabricated.append({
                        "claim": sentence,
                        "confidence": confidence,
                        "evidence_provided": False,
                        "hallucination_risk": risk,
                        "reason": reason
                    })
    
    return fabricated


def generate_request_id(image_path: str) -> str:
    """Generate deterministic request ID from image path."""
    return "vanilla_baseline_" + hashlib.sha256(image_path.encode()).hexdigest()[:16]


def main():
    parser = argparse.ArgumentParser(
        description="RadAgent v2 — Vanilla Baseline (ungrounded VLM)"
    )
    parser.add_argument("--image", required=True, help="Path to chest X-ray image")
    parser.add_argument(
        "--output",
        default="runs/vanilla_baseline",
        help="Output directory (default: runs/vanilla_baseline)"
    )
    parser.add_argument(
        "--specialist-checkpoint",
        default="runs/nih14_convnextv2_base_384/best_model.pt",
        help="Path to v1 specialist checkpoint (for comparison)"
    )
    parser.add_argument(
        "--config",
        default="configs/nih14_convnextv2_base.yaml",
        help="Path to model config"
    )
    parser.add_argument(
        "--calibration",
        default="runs/nih14_convnextv2_base_384/calibration.json",
        help="Path to v1 calibration thresholds"
    )
    parser.add_argument(
        "--featherless-api-key",
        default=None,
        help="Featherless API key (or set FEATHERLESS_API_KEY env var)"
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for specialist (cuda/cpu)"
    )
    
    args = parser.parse_args()
    
    # Get API key
    api_key = args.featherless_api_key or os.getenv("FEATHERLESS_API_KEY")
    if not api_key:
        print("ERROR: Featherless API key required. Set --featherless-api-key or FEATHERLESS_API_KEY env var")
        sys.exit(1)
    
    # Validate image
    if not os.path.exists(args.image):
        print(f"ERROR: Image not found: {args.image}")
        sys.exit(1)
    
    print(f"RadAgent v2 — Vanilla Baseline")
    print(f"Image: {args.image}")
    print(f"Output: {args.output}")
    print(f"Device: {args.device}")
    print()
    
    # Load specialist for comparison
    print("Loading v1 specialist for comparison...")
    specialist = load_specialist_for_comparison(
        args.specialist_checkpoint,
        args.config,
        args.device
    )
    thresholds = load_calibration_thresholds(args.calibration)
    
    # Run specialist on image (for comparison only, not shown to VLM)
    print("Running specialist (for fabrication detection)...")
    eval_tfms = build_eval_transforms(image_size=384)
    gray = load_cxr_grayscale(args.image)
    gray = apply_clahe(gray, clip_limit=2.5)
    rgb = to_three_channel(gray)
    image_tensor = eval_tfms(image=rgb)["image"].float().unsqueeze(0).to(args.device)
    
    with torch.no_grad():
        specialist_probs = torch.sigmoid(specialist(image_tensor)).squeeze(0).cpu()
    
    # Call vanilla VLM
    print("Calling Featherless Qwen2.5-VL (ungrounded)...")
    vlm_result = call_vanilla_vlm(args.image, api_key)
    
    # Parse report
    report = parse_report_sections(vlm_result["full_text"])
    
    # Detect fabricated claims
    print("Detecting fabricated claims...")
    fabricated_claims = detect_fabricated_claims(report, specialist_probs, thresholds)
    
    print(f"Found {len(fabricated_claims)} fabricated claims")
    
    # Build output
    request_id = generate_request_id(args.image)
    image_id = Path(args.image).stem
    
    output_json = {
        "request_id": request_id,
        "grounding_status": "UNGROUNDED",
        "report": report,
        "fabricated_claims": fabricated_claims,
        "metadata": {
            "prompt": vlm_result["prompt"],
            "temperature": 0.7,
            "max_tokens": 500,
            "processing_time_ms": vlm_result["processing_time_ms"],
            "warning": "UNGROUNDED VANILLA — for comparison only. No specialist findings, no retrieved evidence, no citation requirement."
        }
    }
    
    metadata_json = {
        "request_id": request_id,
        "image_filename": os.path.basename(args.image),
        "image_size_bytes": os.path.getsize(args.image),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "featherless_api_key_used": True,
        "model": vlm_result["model"]
    }
    
    # Write output
    output_dir = Path(args.output) / image_id
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / "output.json"
    metadata_path = output_dir / "metadata.json"
    
    with open(output_path, "w") as f:
        json.dump(output_json, f, indent=2)
    
    with open(metadata_path, "w") as f:
        json.dump(metadata_json, f, indent=2)
    
    print()
    print(f"✅ Vanilla baseline complete")
    print(f"   Output: {output_path}")
    print(f"   Metadata: {metadata_path}")
    print()
    print("REPORT:")
    print(report["full_text"])
    print()
    print(f"FABRICATED CLAIMS: {len(fabricated_claims)}")
    for claim in fabricated_claims:
        print(f"  - [{claim['hallucination_risk'].upper()}] {claim['claim']}")
        print(f"    Reason: {claim['reason']}")


if __name__ == "__main__":
    main()

# Made with Bob
