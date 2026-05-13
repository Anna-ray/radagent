"""
RadAgent v2 - Universal Modality Router Demo Script

Demonstrates Scene 3: Universal DICOM modality routing with graceful fallback.
Identifies modality from DICOM or image file, routes to appropriate specialist
pipeline, and handles unknown modalities gracefully.

Author: Rayane Aggoune
"""

import argparse
import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from radagent.modality.router import ModalityRouter
from radagent.modality.dicom_io import load_dicom


def print_routing_decision(decision: dict):
    """Pretty-print routing decision."""
    print("\n" + "=" * 80)
    print("ROUTING DECISION")
    print("=" * 80)
    
    print(f"\nModality: {decision['modality']}")
    print(f"Body Part: {decision['body_part']}")
    print(f"Confidence: {decision['confidence']:.2%}")
    print(f"Detection Method: {decision['method']}")
    
    print(f"\nMatched Entry: {decision['matched_entry']}")
    print(f"Status: {decision['status'].upper()}")
    
    if decision['status'] == 'production':
        print("  ✓ Production pipeline available")
        print(f"  Specialist: {decision['specialist_path']}")
        print(f"  RAG Corpus: {decision['rag_corpus']}")
        print(f"  Preprocessing: {decision['preprocessing']}")
        print(f"  Autonomy Tools: {', '.join(decision['autonomy_tools'])}")
    
    elif decision['status'] == 'registered':
        print("  ⚠️  Pipeline registered but specialist not trained")
        print(f"  Fallback: VLM-only with elevated uncertainty")
        print(f"  Reason: {decision.get('fallback_reason', 'Specialist coming in v2.1')}")
    
    elif decision['status'] == 'fallback':
        print("  ⚠️  Unknown modality - graceful fallback")
        print(f"  Fallback: VLM-only with elevated uncertainty")
        print(f"  Reason: {decision.get('fallback_reason', 'Modality not in registry')}")
    
    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="RadAgent v2 - Universal Modality Router Demo"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to DICOM file or image (PNG/JPG)",
    )
    parser.add_argument(
        "--audit-dir",
        type=str,
        default="runs/modality_demo",
        help="Directory to write audit results",
    )
    
    args = parser.parse_args()
    
    # Create audit directory
    audit_dir = Path(args.audit_dir)
    audit_dir.mkdir(parents=True, exist_ok=True)
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)
    
    print("=" * 80)
    print("RadAgent v2 - Universal Modality Router Demo (Scene 3)")
    print("=" * 80)
    print()
    
    # Step 1: Load input
    print("Step 1: Loading input file...")
    print(f"  File: {input_path}")
    print(f"  Extension: {input_path.suffix}")
    
    is_dicom = input_path.suffix.lower() in ['.dcm', '.dicom']
    
    if is_dicom:
        print("  Type: DICOM")
        try:
            dicom_data = load_dicom(str(input_path))
            print(f"  ✓ DICOM loaded successfully")
            print(f"    Modality: {dicom_data.get('modality', 'Unknown')}")
            print(f"    Body Part: {dicom_data.get('body_part', 'Unknown')}")
            print(f"    Study UID: {dicom_data.get('study_uid', 'N/A')}")
        except Exception as e:
            print(f"  ✗ Failed to load DICOM: {e}")
            sys.exit(1)
    else:
        print("  Type: Image (PNG/JPG)")
        print("  Note: Will attempt to infer modality from filename/content")
        dicom_data = None
    
    print()
    
    # Step 2: Initialize router
    print("Step 2: Initializing modality router...")
    router = ModalityRouter()
    print("  ✓ Router initialized")
    print(f"  Registry entries: {len(router.registry)}")
    print()
    
    # Step 3: Identify modality
    print("Step 3: Identifying modality...")
    
    if is_dicom:
        modality_info = router.identify(dicom_data)
    else:
        # For non-DICOM, router will attempt inference
        modality_info = router.identify(str(input_path))
    
    print(f"  Detected: {modality_info['modality']}")
    print(f"  Body Part: {modality_info['body_part']}")
    print(f"  Confidence: {modality_info['confidence']:.2%}")
    print(f"  Method: {modality_info['method']}")
    print()
    
    # Step 4: Route to pipeline
    print("Step 4: Routing to appropriate pipeline...")
    
    routing_decision = router.route(str(input_path))
    
    print_routing_decision(routing_decision)
    
    # Step 5: Save audit trace
    study_uid = dicom_data.get('study_uid', 'unknown') if dicom_data else 'non_dicom'
    trace_path = audit_dir / f"{study_uid}_trace.json"
    
    trace_data = {
        "input_path": str(input_path),
        "is_dicom": is_dicom,
        "modality_info": modality_info,
        "routing_decision": routing_decision,
    }
    
    if dicom_data:
        trace_data["dicom_metadata"] = {
            "modality": dicom_data.get('modality'),
            "body_part": dicom_data.get('body_part'),
            "study_uid": dicom_data.get('study_uid'),
            "series_uid": dicom_data.get('series_uid'),
        }
    
    with open(trace_path, 'w') as f:
        json.dump(trace_data, f, indent=2)
    
    print(f"\nAudit trace saved to: {trace_path}")
    
    # Step 6: Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    if routing_decision['status'] == 'production':
        print("\n✓ PRODUCTION PIPELINE AVAILABLE")
        print(f"  Modality: {routing_decision['modality']}")
        print(f"  Body Part: {routing_decision['body_part']}")
        print(f"  Specialist: {routing_decision['specialist_path']}")
        print(f"  RAG Corpus: {routing_decision['rag_corpus']}")
        print("\n  Next steps:")
        print("    1. Run specialist model")
        print("    2. Query RAG for evidence")
        print("    3. Generate grounded report")
        print("    4. Execute autonomy workflow")
    
    elif routing_decision['status'] == 'registered':
        print("\n⚠️  REGISTERED PIPELINE (SPECIALIST COMING)")
        print(f"  Modality: {routing_decision['modality']}")
        print(f"  Body Part: {routing_decision['body_part']}")
        print(f"  Reason: {routing_decision.get('fallback_reason', 'Specialist in development')}")
        print("\n  Fallback strategy:")
        print("    1. VLM-only analysis (no specialist)")
        print("    2. Explicit 'elevated uncertainty' prefix")
        print("    3. RAG grounding still available")
        print("    4. Limited autonomy (no specialist-dependent tools)")
    
    elif routing_decision['status'] == 'fallback':
        print("\n⚠️  GRACEFUL FALLBACK (UNKNOWN MODALITY)")
        print(f"  Detected: {routing_decision['modality']}")
        print(f"  Reason: {routing_decision.get('fallback_reason', 'Not in registry')}")
        print("\n  Fallback strategy:")
        print("    1. VLM-only analysis")
        print("    2. Explicit 'elevated uncertainty' prefix")
        print("    3. Generic RAG corpus")
        print("    4. No autonomy tools")
    
    print("\n" + "=" * 80)
    print("Demo complete. Universal routing with graceful fallback demonstrated.")
    print("=" * 80)


if __name__ == "__main__":
    main()

# Made with Bob
