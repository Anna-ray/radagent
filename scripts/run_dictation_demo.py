"""
RadAgent v2 - Voice Dictation Demo Script

Demonstrates Scene 2.5: Voice-driven grounded dictation auditing.
Transcribes radiologist audio, compares against specialist findings,
and surfaces discrepancies.

Author: Rayane Aggoune
"""

import argparse
import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from radagent.voice.transcriber import SpeechmaticsTranscriber, MockTranscriber
from radagent.voice.dictation_auditor import DictationAuditor, MockDictationAuditor


def load_specialist_findings(findings_path: Path) -> list:
    """Load specialist findings from JSON file."""
    with open(findings_path, 'r') as f:
        data = json.load(f)
    
    # Extract findings in expected format
    findings = []
    for finding in data.get("findings", []):
        findings.append({
            "finding": finding["finding"],
            "probability": finding["probability"],
            "threshold": finding.get("threshold", 0.5),
            "above_threshold": finding.get("above_threshold", False),
        })
    
    return findings


def main():
    parser = argparse.ArgumentParser(
        description="RadAgent v2 - Voice Dictation Demo"
    )
    parser.add_argument(
        "--audio",
        type=str,
        required=True,
        help="Path to audio file (WAV, MP3, etc.)",
    )
    parser.add_argument(
        "--findings",
        type=str,
        help="Path to specialist findings JSON (from v1 pipeline)",
    )
    parser.add_argument(
        "--audit-dir",
        type=str,
        default="runs/dictation_demo",
        help="Directory to write audit results",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock transcriber/auditor (no API keys required)",
    )
    
    args = parser.parse_args()
    
    # Create audit directory
    audit_dir = Path(args.audit_dir)
    audit_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("RadAgent v2 - Voice Dictation Demo (Scene 2.5)")
    print("=" * 80)
    print()
    
    # Step 1: Transcribe audio
    print("Step 1: Transcribing audio...")
    print(f"  Audio file: {args.audio}")
    
    if args.mock:
        print("  Using mock transcriber (no Speechmatics API)")
        transcriber = MockTranscriber()
    else:
        try:
            transcriber = SpeechmaticsTranscriber()
        except (ImportError, ValueError) as e:
            print(f"  Error: {e}")
            print("  Falling back to mock transcriber")
            transcriber = MockTranscriber()
    
    transcript_result = transcriber.transcribe_file(args.audio)
    
    print(f"  Duration: {transcript_result.duration_seconds:.1f}s")
    print(f"  Word count: {transcript_result.word_count}")
    print(f"  Processing time: {transcript_result.processing_time_seconds:.2f}s")
    print()
    print("Transcript:")
    print("-" * 80)
    print(transcript_result.full_text)
    print("-" * 80)
    print()
    
    # Save transcript
    transcript_path = audit_dir / "transcript.json"
    with open(transcript_path, 'w') as f:
        json.dump(transcript_result.to_dict(), f, indent=2)
    print(f"Saved transcript to: {transcript_path}")
    print()
    
    # Step 2: Load specialist findings
    if args.findings:
        print("Step 2: Loading specialist findings...")
        print(f"  Findings file: {args.findings}")
        specialist_findings = load_specialist_findings(Path(args.findings))
        print(f"  Loaded {len(specialist_findings)} findings")
    else:
        print("Step 2: Using mock specialist findings...")
        # Mock findings for demo
        specialist_findings = [
            {
                "finding": "Effusion",
                "probability": 0.93,
                "threshold": 0.45,
                "above_threshold": True,
            },
            {
                "finding": "Infiltration",
                "probability": 0.82,
                "threshold": 0.38,
                "above_threshold": True,
            },
            {
                "finding": "Cardiomegaly",
                "probability": 0.28,
                "threshold": 0.42,
                "above_threshold": False,
            },
        ]
        print(f"  Using {len(specialist_findings)} mock findings")
    
    print()
    print("Specialist Findings:")
    print("-" * 80)
    for finding in specialist_findings:
        status = "✓ ABOVE" if finding["above_threshold"] else "✗ BELOW"
        print(f"  {finding['finding']:20s} p={finding['probability']:.2%}  "
              f"threshold={finding['threshold']:.2%}  {status}")
    print("-" * 80)
    print()
    
    # Step 3: Audit dictation
    print("Step 3: Auditing dictation against specialist...")
    
    if args.mock:
        print("  Using mock auditor (no Gemini API)")
        auditor = MockDictationAuditor()
    else:
        try:
            auditor = DictationAuditor()
        except (ImportError, ValueError) as e:
            print(f"  Error: {e}")
            print("  Falling back to mock auditor")
            auditor = MockDictationAuditor()
    
    audit_report = auditor.audit(
        transcript=transcript_result.full_text,
        specialist_findings=specialist_findings,
    )
    
    print(f"  Audit ID: {audit_report.audit_id}")
    print(f"  Dictated findings: {len(audit_report.dictated_findings)}")
    print(f"  Discrepancies: {len(audit_report.discrepancies)}")
    print()
    
    # Step 4: Display discrepancies
    print("Step 4: Discrepancy Analysis")
    print("=" * 80)
    
    if not audit_report.discrepancies:
        print("✓ No discrepancies found. Dictation and specialist agree.")
    else:
        # Group by severity
        high_severity = [d for d in audit_report.discrepancies if d.severity == "high"]
        medium_severity = [d for d in audit_report.discrepancies if d.severity == "medium"]
        low_severity = [d for d in audit_report.discrepancies if d.severity == "low"]
        
        if high_severity:
            print()
            print("🔴 HIGH SEVERITY DISCREPANCIES (RECONSIDER):")
            print("-" * 80)
            for disc in high_severity:
                print(f"  Finding: {disc.finding_name}")
                print(f"  Type: {disc.discrepancy_type.value}")
                print(f"  Dictated: {disc.dictated_state}")
                print(f"  Specialist: p={disc.specialist_probability:.2%} "
                      f"(threshold={disc.specialist_threshold:.2%})")
                print(f"  Explanation: {disc.explanation}")
                print()
        
        if medium_severity:
            print()
            print("🟡 MEDIUM SEVERITY DISCREPANCIES:")
            print("-" * 80)
            for disc in medium_severity:
                print(f"  Finding: {disc.finding_name}")
                print(f"  Type: {disc.discrepancy_type.value}")
                print(f"  Explanation: {disc.explanation}")
                print()
        
        if low_severity:
            print()
            print("🟢 LOW SEVERITY (CONSISTENT):")
            print("-" * 80)
            for disc in low_severity:
                print(f"  {disc.finding_name}: {disc.explanation}")
            print()
    
    # Step 5: Save audit report
    audit_path = audit_dir / "audit_report.json"
    with open(audit_path, 'w') as f:
        json.dump(audit_report.to_dict(), f, indent=2)
    
    print()
    print("=" * 80)
    print(f"Audit report saved to: {audit_path}")
    
    if audit_report.has_critical_discrepancies():
        print()
        print("⚠️  CRITICAL DISCREPANCIES DETECTED")
        print("    RadAgent audits the radiologist, not the other way around.")
        print("    Review recommended.")
    
    print("=" * 80)


if __name__ == "__main__":
    main()

# Made with Bob
