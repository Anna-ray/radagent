"""
RadAgent v2 - Workflow Autonomy Demo Script

Demonstrates Scene 4: Autonomous workflow with replan on roadblocks.
Runs the v1 agentic-rag pipeline, then executes autonomous workflow tools
with confidence scoring, halt logic, and replan triggers.

Author: Rayane Aggoune
"""

import argparse
import json
import sys
import asyncio
from pathlib import Path
from typing import Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from radagent.autonomy.planner import WorkflowPlanner
from radagent.autonomy.halt import should_halt


async def load_study_context(image_path: Optional[str], findings_path: Optional[str]) -> dict:
    """Load study context from v1 pipeline output or mock data."""
    if findings_path:
        with open(findings_path, 'r') as f:
            findings_data = json.load(f)
        
        return {
            "study_id": findings_data.get("study_id", "unknown"),
            "image_path": image_path or findings_data.get("image_path", "unknown"),
            "findings": findings_data.get("findings", []),
            "specialist_confidence": findings_data.get("specialist_confidence", 0.85),
        }
    else:
        # Mock study context for demo
        return {
            "study_id": "demo_study_001",
            "image_path": image_path or "demo.jpg",
            "findings": [
                {
                    "finding": "Effusion",
                    "probability": 0.93,
                    "threshold": 0.45,
                    "above_threshold": True,
                    "citations": ["StatPearls: Pleural Effusion"],
                },
                {
                    "finding": "Infiltration",
                    "probability": 0.82,
                    "threshold": 0.38,
                    "above_threshold": True,
                    "citations": ["Wikipedia: Pulmonary Infiltrate"],
                },
            ],
            "specialist_confidence": 0.87,
        }


def print_tool_result(result: dict, step_num: int):
    """Pretty-print a tool execution result."""
    print(f"\n{'='*80}")
    print(f"Step {step_num}: {result['action'].upper()}")
    print(f"{'='*80}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Evidence: {len(result.get('evidence_refs', []))} citations")
    
    if 'result' in result:
        print(f"\nResult:")
        if isinstance(result['result'], dict):
            for key, value in result['result'].items():
                print(f"  {key}: {value}")
        else:
            print(f"  {result['result']}")
    
    # Check halt condition
    halt_decision = should_halt(result['action'], result['confidence'])
    if halt_decision.halt:
        print(f"\n⚠️  HALT TRIGGERED")
        print(f"   Reason: {halt_decision.reason}")
        print(f"   Action: Replan or escalate to human")
    else:
        print(f"\n✓ Confidence above floor, proceeding")
    
    print(f"\nAudit ID: {result.get('audit_id', 'N/A')}")


async def main():
    parser = argparse.ArgumentParser(
        description="RadAgent v2 - Workflow Autonomy Demo"
    )
    parser.add_argument(
        "--image",
        type=str,
        help="Path to chest X-ray image",
    )
    parser.add_argument(
        "--findings",
        type=str,
        help="Path to v1 pipeline findings JSON",
    )
    parser.add_argument(
        "--audit-dir",
        type=str,
        default="runs/autonomy_demo",
        help="Directory to write audit results",
    )
    parser.add_argument(
        "--inject-roadblock",
        type=str,
        choices=["confidence", "missing_passages", "missing_prior", "none"],
        default="none",
        help="Inject a roadblock to demonstrate replan",
    )
    parser.add_argument(
        "--retriever-url",
        type=str,
        default="http://localhost:8001",
        help="RAG retriever URL",
    )
    parser.add_argument(
        "--vllm-url",
        type=str,
        default="http://localhost:8000/v1",
        help="vLLM API URL",
    )
    
    args = parser.parse_args()
    
    # Create audit directory
    audit_dir = Path(args.audit_dir)
    audit_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("RadAgent v2 - Workflow Autonomy Demo (Scene 4)")
    print("=" * 80)
    print()
    
    # Step 1: Load study context
    print("Step 1: Loading study context...")
    study_context = await load_study_context(args.image, args.findings)
    print(f"  Study ID: {study_context['study_id']}")
    print(f"  Findings: {len(study_context['findings'])}")
    print()
    
    # Step 2: Initialize planner
    print("Step 2: Initializing workflow planner...")
    planner = WorkflowPlanner(
        retriever_url=args.retriever_url,
        vllm_url=args.vllm_url,
    )
    print("  ✓ Planner initialized")
    print()
    
    # Step 3: Generate plan
    print("Step 3: Generating workflow plan...")
    plan = await planner.plan(study_context)
    print(f"  Generated {len(plan)} workflow steps:")
    for i, step in enumerate(plan, 1):
        print(f"    {i}. {step['tool']} (priority: {step.get('priority', 'normal')})")
    print()
    
    # Step 4: Execute plan
    print("Step 4: Executing workflow...")
    print("=" * 80)
    
    # Inject roadblock if requested
    if args.inject_roadblock != "none":
        print(f"\n⚠️  INJECTING ROADBLOCK: {args.inject_roadblock}")
        print("=" * 80)
    
    results = await planner.execute(
        plan,
        study_context,
        inject_roadblock=args.inject_roadblock if args.inject_roadblock != "none" else None,
    )
    
    # Print results
    for i, result in enumerate(results, 1):
        print_tool_result(result, i)
        
        # Check if replan was triggered
        if result.get("replanned", False):
            print(f"\n🔄 REPLAN TRIGGERED")
            print(f"   Original confidence: {result.get('original_confidence', 0):.2%}")
            print(f"   Replan reason: {result.get('replan_reason', 'Unknown')}")
            print(f"   New confidence: {result['confidence']:.2%}")
    
    print("\n" + "=" * 80)
    print("Workflow Execution Complete")
    print("=" * 80)
    
    # Step 5: Save audit trail
    trace_path = audit_dir / f"{study_context['study_id']}_trace.json"
    trace_data = {
        "study_id": study_context["study_id"],
        "plan": plan,
        "results": results,
        "injected_roadblock": args.inject_roadblock,
    }
    
    with open(trace_path, 'w') as f:
        json.dump(trace_data, f, indent=2)
    
    print(f"\nAudit trail saved to: {trace_path}")
    
    # Step 6: Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    total_steps = len(results)
    halted_steps = sum(1 for r in results if should_halt(r['action'], r['confidence']).halt)
    replanned_steps = sum(1 for r in results if r.get('replanned', False))
    
    print(f"Total steps: {total_steps}")
    print(f"Halted steps: {halted_steps}")
    print(f"Replanned steps: {replanned_steps}")
    print(f"Success rate: {(total_steps - halted_steps) / total_steps * 100:.1f}%")
    
    # Check audit chain
    print("\nAudit Chain Verification:")
    for i, result in enumerate(results):
        if i > 0:
            prev_audit_id = results[i-1].get('audit_id')
            curr_prev_id = result.get('previous_audit_id')
            if prev_audit_id == curr_prev_id:
                print(f"  Step {i+1}: ✓ Chain valid")
            else:
                print(f"  Step {i+1}: ✗ Chain broken")
        else:
            print(f"  Step 1: ✓ Root step")
    
    print("\n" + "=" * 80)
    print("Demo complete. Autonomous workflow with replan demonstrated.")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())

# Made with Bob
