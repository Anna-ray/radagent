"""
radagent.autonomy.halt
----------------------
Halt logic with per-action confidence floors.

Author: Rayane Aggoune
"""
from __future__ import annotations

from dataclasses import dataclass


# Per-action confidence floors (NON-NEGOTIABLE)
CONFIDENCE_FLOORS = {
    "triage_study": 0.70,
    "route_to_subspecialist": 0.65,
    "flag_critical_finding": 0.85,
    "schedule_follow_up": 0.75,
}


@dataclass
class HaltDecision:
    """Decision on whether to halt execution."""
    halt: bool
    reason: str
    recommended_action: str | None = None


def should_halt(action: str, confidence: float) -> HaltDecision:
    """Determine if execution should halt based on confidence.
    
    Args:
        action: Action name
        confidence: Confidence score (0.0 to 1.0)
        
    Returns:
        HaltDecision with halt flag and reason
    """
    floor = CONFIDENCE_FLOORS.get(action, 0.70)
    
    if confidence < floor:
        return HaltDecision(
            halt=True,
            reason=f"Confidence {confidence:.3f} below floor {floor:.3f} for action '{action}'",
            recommended_action="escalate_to_human" if action == "flag_critical_finding" else "refine_and_retry",
        )
    
    return HaltDecision(
        halt=False,
        reason=f"Confidence {confidence:.3f} meets floor {floor:.3f}",
        recommended_action=None,
    )


def get_confidence_floor(action: str) -> float:
    """Get confidence floor for an action.
    
    Args:
        action: Action name
        
    Returns:
        Confidence floor (0.0 to 1.0)
    """
    return CONFIDENCE_FLOORS.get(action, 0.70)

# Made with Bob
