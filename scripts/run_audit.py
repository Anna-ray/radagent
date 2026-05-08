"""
scripts/run_audit.py
--------------------
Run self-audit on a RadAgent pipeline output.

Takes a directory containing pipeline results (from predict_one.py or server cache)
and runs the audit layer to flag potential VLM hallucinations.

Usage:
    python scripts/run_audit.py \
        --run-dir runs/nih14_convnextv2_base_384/predict_one_test \
        --output audit_results.json

Input files expected in run-dir:
    - structured_findings.json  (specialist probabilities)
    - gradcam_*.png             (Grad-CAM heatmaps)
    - retrieved_passages.json   (RAG results)
    - vlm_report.txt            (VLM-generated report)

Output:
    - audit_results.json with flagged claims
"""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from radagent.inference.audit import audit_report


def load_gradcam_from_png(png_path: Path) -> np.ndarray:
    """Load Grad-CAM heatmap from PNG file (grayscale)."""
    img = cv2.imread(str(png_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to load {png_path}")
    # Normalize to [0, 1]
    return img.astype(np.float32) / 255.0


def main():
    parser = argparse.ArgumentParser(description="Run self-audit on RadAgent output")
    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Directory containing pipeline output files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="audit_results.json",
        help="Output JSON file for audit results",
    )
    parser.add_argument(
        "--embedder-model",
        type=str,
        default="BAAI/bge-m3",
        help="BGE-M3 model name for citation audit",
    )
    parser.add_argument(
        "--numerical-threshold",
        type=float,
        default=0.5,
        help="Threshold for numerical audit (specialist probability)",
    )
    parser.add_argument(
        "--citation-threshold",
        type=float,
        default=0.4,
        help="Threshold for citation audit (cosine similarity)",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=384,
        help="Image size used during inference",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    print(f"[audit] Loading files from {run_dir}")

    # 1. Load structured findings (specialist probabilities)
    findings_path = run_dir / "structured_findings.json"
    if not findings_path.exists():
        raise FileNotFoundError(f"Missing {findings_path}")
    
    with open(findings_path) as f:
        structured = json.load(f)
    
    # Extract specialist probabilities
    specialist_probs = {
        f["name"]: f["calibrated_probability"]
        for f in structured["findings"]
    }
    print(f"[audit] Loaded {len(specialist_probs)} specialist probabilities")

    # 2. Load Grad-CAM heatmaps
    gradcam_dict = {}
    for png_path in run_dir.glob("gradcam_*.png"):
        # Extract finding name from filename: gradcam_Cardiomegaly.png
        finding_name = png_path.stem.replace("gradcam_", "")
        try:
            gradcam_array = load_gradcam_from_png(png_path)
            gradcam_dict[finding_name] = gradcam_array
        except Exception as e:
            print(f"[audit] WARN: Failed to load {png_path}: {e}")
    
    print(f"[audit] Loaded {len(gradcam_dict)} Grad-CAM heatmaps")

    # 3. Load retrieved passages
    passages_path = run_dir / "retrieved_passages.json"
    if not passages_path.exists():
        print(f"[audit] WARN: Missing {passages_path}, citation audit will be limited")
        retrieved_passages = []
    else:
        with open(passages_path) as f:
            retrieved_data = json.load(f)
        
        # Flatten passages from all findings into a single list
        retrieved_passages = []
        for finding_name, passages in retrieved_data.items():
            retrieved_passages.extend(passages)
        
        print(f"[audit] Loaded {len(retrieved_passages)} retrieved passages")

    # 4. Load VLM report
    report_path = run_dir / "vlm_report.txt"
    if not report_path.exists():
        raise FileNotFoundError(f"Missing {report_path}")
    
    report_text = report_path.read_text(encoding="utf-8")
    print(f"[audit] Loaded VLM report ({len(report_text)} chars)")

    # 5. Load embedder for citation audit
    print(f"[audit] Loading embedder: {args.embedder_model}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedder = SentenceTransformer(args.embedder_model, device=device)
    print(f"[audit] Embedder loaded on {device}")

    # 6. Run audit
    print("[audit] Running audit...")
    audit_results = audit_report(
        report_text=report_text,
        specialist_probs=specialist_probs,
        gradcam_dict=gradcam_dict,
        retrieved_passages=retrieved_passages,
        embedder=embedder,
        image_shape=(args.image_size, args.image_size),
        numerical_threshold=args.numerical_threshold,
        citation_threshold=args.citation_threshold,
    )

    # 7. Save results
    output_path = run_dir / args.output
    with open(output_path, "w") as f:
        json.dump(audit_results, f, indent=2)
    
    print(f"[audit] Results saved to {output_path}")

    # 8. Print summary
    summary = audit_results["summary"]
    print("\n" + "=" * 60)
    print("AUDIT SUMMARY")
    print("=" * 60)
    print(f"Total claims:   {summary['total_claims']}")
    print(f"Flagged claims: {summary['flagged_claims']}")
    print(f"Flag rate:      {summary['flag_rate']:.1%}")
    print("=" * 60)

    # Print flagged claims
    if summary["flagged_claims"] > 0:
        print("\nFLAGGED CLAIMS:")
        for i, claim_result in enumerate(audit_results["claims"], 1):
            if claim_result["any_flagged"]:
                print(f"\n[{i}] {claim_result['claim'][:100]}...")
                print(f"    Max severity: {claim_result['max_severity']:.2f}")
                
                # Show failed audits
                for audit_type in ["numerical", "spatial", "citation"]:
                    for audit in claim_result["audits"][audit_type]:
                        if not audit["passed"]:
                            print(f"    ❌ {audit_type}: {audit['reason']}")

    print("\n[audit] Done.")


if __name__ == "__main__":
    main()

# Made with Bob
