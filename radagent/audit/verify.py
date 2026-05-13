"""
radagent.audit.verify
---------------------
SHA-256 audit chain verifier.

Author: Rayane Aggoune
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def verify_audit_chain(audit_dir: str | Path) -> tuple[bool, list[str]]:
    """Verify SHA-256 audit chain integrity.
    
    Args:
        audit_dir: Directory containing audit receipts
        
    Returns:
        (is_valid, error_messages)
    """
    audit_dir = Path(audit_dir)
    
    if not audit_dir.exists():
        return False, [f"Audit directory not found: {audit_dir}"]
    
    # Find all audit files
    audit_files = sorted(audit_dir.glob("round_*.json"))
    
    if not audit_files:
        return False, ["No audit files found"]
    
    errors = []
    previous_hash = None
    
    for audit_file in audit_files:
        try:
            with open(audit_file) as f:
                audit = json.load(f)
            
            # Check required fields
            required_fields = [
                "round_number",
                "num_clients",
                "total_samples",
                "global_auc",
                "client_aucs",
                "parameter_divergence",
                "timestamp",
                "previous_audit_hash",
            ]
            
            for field in required_fields:
                if field not in audit:
                    errors.append(f"{audit_file.name}: Missing field '{field}'")
            
            # Verify hash chain
            if previous_hash is not None:
                if audit["previous_audit_hash"] != previous_hash:
                    errors.append(
                        f"{audit_file.name}: Hash chain broken. "
                        f"Expected previous_hash={previous_hash}, "
                        f"got {audit['previous_audit_hash']}"
                    )
            else:
                # First round should have None
                if audit["previous_audit_hash"] is not None:
                    errors.append(
                        f"{audit_file.name}: First round should have "
                        f"previous_audit_hash=null, got {audit['previous_audit_hash']}"
                    )
            
            # Compute hash of this round
            audit_content = json.dumps(
                {
                    "round_number": audit["round_number"],
                    "num_clients": audit["num_clients"],
                    "total_samples": audit["total_samples"],
                    "global_auc": audit["global_auc"],
                    "client_aucs": audit["client_aucs"],
                    "parameter_divergence": audit["parameter_divergence"],
                    "timestamp": audit["timestamp"],
                    "previous_audit_hash": audit["previous_audit_hash"],
                },
                sort_keys=True,
            ).encode("utf-8")
            
            computed_hash = hashlib.sha256(audit_content).hexdigest()
            previous_hash = computed_hash
            
        except json.JSONDecodeError as e:
            errors.append(f"{audit_file.name}: Invalid JSON - {e}")
        except Exception as e:
            errors.append(f"{audit_file.name}: Error - {e}")
    
    is_valid = len(errors) == 0
    return is_valid, errors


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Verify RadAgent audit chain")
    parser.add_argument("audit_dir", help="Directory containing audit receipts")
    
    args = parser.parse_args()
    
    print(f"Verifying audit chain in: {args.audit_dir}")
    print("=" * 60)
    
    is_valid, errors = verify_audit_chain(args.audit_dir)
    
    if is_valid:
        print("✅ PASS: Audit chain is valid")
        print(f"Verified {len(list(Path(args.audit_dir).glob('round_*.json')))} rounds")
    else:
        print("❌ FAIL: Audit chain verification failed")
        print(f"\nErrors ({len(errors)}):")
        for error in errors:
            print(f"  - {error}")
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

# Made with Bob
