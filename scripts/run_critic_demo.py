"""
RadAgent v2 - CriticAgent Demo Script

Demonstrates the CriticAgent reviewing decisions and emitting verdicts.
The CriticAgent is "the AI that disagrees with itself" - it challenges
decisions when evidence is weak or confidence is borderline.

Author: Rayane Aggoune
"""

import argparse
import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from radagent.agents.critic import CriticAgent, MockCriticAgent


def load_decision_record(path: Path) -> dict:
    """Load a decision record from JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def load_evidence(path: Path) -> dict:
    """Load evidence from JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def print_verdict(challenge_record):
    """Pretty-print a CriticAgent verdict."""
    print("\n" + "=" * 80)
    print("CRITIC AGENT VERDICT")
    print("=" * 80)
    
    # Verdict with color coding
    verdict_colors = {
        "APPROVE": "✓",
        "CHALLENGE": "⚠️",
        "ESCALATE": "🔴",
    }
    icon = verdict_colors.get(challenge_record.verdict, "?")
    
    print(f"\n{icon} VERDICT: {challenge_record.verdict}")
    print(f"\nReasoning: {challenge_record.reasoning}")
    print(f"Confidence: {challenge_record.confidence:.2%}")
    
    if challenge_record.cited_concerns:
        print(f"\nCited Concerns:")
        for concern in challenge_record.cited_concerns:
            print(f"  - {concern}")
    
    if challenge_record.requested_replan:
        print(f"\n🔄 REPLAN REQUESTED")
        print(f"   Action: {challenge_record.replan_action}")
    
    print(f"\nAudit Trail:")
    print(f"  Audit ID: {challenge_record.audit_id}")
    print(f"  Previous (Decision) ID: {challenge_record.previous_audit_id}")
    print(f"  Timestamp: {challenge_record.timestamp}")
    
    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="RadAgent v2 - CriticAgent Demo"
    )
    parser.add_argument(
        "--decision-record",
        type=str,
        required=True,
        help="Path to decision record JSON",
    )
    parser.add_argument(
        "--evidence",
        type=str,
        required=True,
        help="Path to evidence JSON",
    )
    parser.add_argument(
        "--action-floor",
        type=float,
        default=0.65,
        help="Confidence floor for this action type (default: 0.65)",
    )
    parser.add_argument(
        "--audit-dir",
        type=str,
        default="runs/critic_demo",
        help="Directory to write audit results",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock CriticAgent (no API key required)",
    )
    
    args = parser.parse_args()
    
    # Create audit directory
    audit_dir = Path(args.audit_dir)
    audit_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("RadAgent v2 - CriticAgent Demo")
    print("The AI that disagrees with itself")
    print("=" * 80)
    print()
    
    # Step 1: Load decision and evidence
    print("Step 1: Loading decision record and evidence...")
    decision_record = load_decision_record(Path(args.decision_record))
    evidence = load_evidence(Path(args.evidence))
    
    print(f"  Decision: {decision_record.get('action', 'unknown')}")
    print(f"  Confidence: {decision_record.get('confidence', 0.0):.2%}")
    print(f"  Evidence passages: {len(evidence.get('passages', []))}")
    print()
    
    # Step 2: Initialize CriticAgent
    print("Step 2: Initializing CriticAgent...")
    
    if args.mock:
        print("  Using mock CriticAgent (no API key required)")
        critic = MockCriticAgent()
    else:
        try:
            critic = CriticAgent()
            print("  ✓ CriticAgent initialized with Featherless API")
        except (ImportError, ValueError) as e:
            print(f"  Error: {e}")
            print("  Falling back to mock CriticAgent")
            critic = MockCriticAgent()
    
    print()
    
    # Step 3: Review decision
    print("Step 3: CriticAgent reviewing decision...")
    print(f"  Action floor: {args.action_floor:.2%}")
    
    challenge_record = critic.review(
        decision_record=decision_record,
        evidence=evidence,
        action_floor=args.action_floor,
    )
    
    # Step 4: Display verdict
    print_verdict(challenge_record)
    
    # Step 5: Save audit record
    audit_path = audit_dir / f"critic_{challenge_record.audit_id}.json"
    with open(audit_path, 'w') as f:
        json.dump(challenge_record.to_dict(), f, indent=2)
    
    print(f"\nAudit record saved to: {audit_path}")
    
    # Step 6: Interpretation
    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    
    if challenge_record.verdict == "APPROVE":
        print("\n✓ Decision APPROVED by CriticAgent")
        print("  The decision has sufficient evidence and confidence.")
        print("  Proceed with the action.")
    
    elif challenge_record.verdict == "CHALLENGE":
        print("\n⚠️  Decision CHALLENGED by CriticAgent")
        print("  The decision has weaknesses that should be addressed.")
        
        if challenge_record.requested_replan:
            print(f"\n  Replan requested: {challenge_record.replan_action}")
            print("  The system should attempt to strengthen the evidence")
            print("  or adjust the decision before proceeding.")
        else:
            print("\n  No automatic replan available.")
            print("  Consider manual review or escalation.")
    
    elif challenge_record.verdict == "ESCALATE":
        print("\n🔴 Decision ESCALATED by CriticAgent")
        print("  The decision requires human review.")
        print("  Do NOT proceed automatically.")
        print("\n  Reasons for escalation:")
        for concern in challenge_record.cited_concerns:
            print(f"    - {concern}")
    
    print("\n" + "=" * 80)
    print("Demo complete. CriticAgent verdict emitted.")
    print("=" * 80)


if __name__ == "__main__":
    main()

# Made with Bob
