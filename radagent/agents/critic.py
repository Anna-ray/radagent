"""
CriticAgent - The Skeptic

The ONE real agent in RadAgent v2. Challenges decisions made by other
components when evidence is weak, confidence is borderline, or reasoning
has gaps. Can force replans.

Author: Rayane Aggoune
"""

import os
import json
import hashlib
import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from enum import Enum

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False


class Verdict(Enum):
    """CriticAgent verdict types."""
    APPROVE = "APPROVE"
    CHALLENGE = "CHALLENGE"
    ESCALATE = "ESCALATE"


@dataclass
class ChallengeRecord:
    """
    A challenge record from the CriticAgent.
    
    This is the output of every CriticAgent review. It either approves
    the decision, challenges it with a replan request, or escalates to
    human handoff.
    """
    verdict: str  # "APPROVE" | "CHALLENGE" | "ESCALATE"
    reasoning: str
    cited_concerns: List[str]  # Specific weaknesses identified
    requested_replan: bool
    replan_action: Optional[str]  # e.g. "refine_rag_query", "lower_confidence_floor"
    confidence: float
    audit_id: str  # SHA-256 hex
    previous_audit_id: str  # SHA-256 hex of the decision being reviewed
    timestamp: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class CriticAgent:
    """
    The Skeptic - Challenges decisions when evidence is weak.
    
    This is the ONE real agent in RadAgent v2. It reviews decisions from:
    - Autonomy planner tools (triage, route, schedule, flag)
    - Dictation auditor (discrepancy reports)
    - Federation server (per-round weight updates)
    
    And emits a verdict: APPROVE, CHALLENGE, or ESCALATE.
    """
    
    # System prompt defining the critic's persona
    SYSTEM_PROMPT = """You are the skeptic. Your job is to challenge decisions made by other components of the RadAgent system. Every decision has evidence behind it. Your task: examine the evidence, identify weaknesses, and decide whether to approve, challenge, or escalate.

You CHALLENGE when:
- Confidence is between 0.55 and the action's floor
- Cited evidence has < 2 supporting passages
- Retrieved passages have low similarity scores (< 0.7)
- The decision contradicts findings from a higher-confidence component
- A required prior study is missing

You ESCALATE (human handoff) when:
- Confidence is below 0.5 AND no replan is likely to recover
- Decision would lead to an irreversible action (auto-send alert, auto-schedule procedure)
- Two components disagree at high confidence

You APPROVE when none of the above triggers fire.

You never agree silently. Every approval comes with a one-line reasoning. Every challenge cites specifics.

Respond in JSON format:
{
  "verdict": "APPROVE" | "CHALLENGE" | "ESCALATE",
  "reasoning": "one-line explanation",
  "cited_concerns": ["specific weakness 1", "specific weakness 2"],
  "requested_replan": true/false,
  "replan_action": "refine_rag_query" | "lower_confidence_floor" | "request_prior_study" | null,
  "confidence": 0.0-1.0
}"""
    
    def __init__(
        self,
        featherless_api_key: Optional[str] = None,
        model: str = "Qwen/Qwen2.5-7B-Instruct",
        api_base: str = "https://api.featherless.ai/v1",
    ):
        """
        Initialize CriticAgent.
        
        Args:
            featherless_api_key: Featherless API key (or set FEATHERLESS_API_KEY env var)
            model: Model to use (default: Qwen2.5-7B-Instruct)
            api_base: API base URL
        """
        if not HTTPX_AVAILABLE:
            raise ImportError(
                "httpx not installed. Install with: pip install httpx"
            )
        
        self.api_key = featherless_api_key or os.getenv("FEATHERLESS_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Featherless API key required. Set FEATHERLESS_API_KEY "
                "environment variable or pass featherless_api_key parameter."
            )
        
        self.model = model
        self.api_base = api_base
        self.client = httpx.Client(timeout=30.0)
    
    def review(
        self,
        decision_record: Dict[str, Any],
        evidence: Dict[str, Any],
        action_floor: float = 0.65,
    ) -> ChallengeRecord:
        """
        Review a decision and emit a verdict.
        
        Args:
            decision_record: The decision to review (from autonomy tool, dictation auditor, etc.)
            evidence: The evidence that supported the decision (RAG passages, specialist outputs, etc.)
            action_floor: The confidence floor for this action type
        
        Returns:
            ChallengeRecord with verdict and reasoning
        """
        # Extract key fields
        action = decision_record.get("action", "unknown")
        confidence = decision_record.get("confidence", 0.0)
        evidence_refs = evidence.get("passages", [])
        
        # Build prompt
        prompt = self._build_review_prompt(
            decision_record,
            evidence,
            action_floor,
        )
        
        # Call Featherless API
        try:
            response = self._call_featherless(prompt)
            verdict_data = json.loads(response)
        except Exception as e:
            # Fallback: if API fails, default to APPROVE with low confidence
            verdict_data = {
                "verdict": "APPROVE",
                "reasoning": f"API call failed ({str(e)}), defaulting to approval",
                "cited_concerns": [],
                "requested_replan": False,
                "replan_action": None,
                "confidence": 0.5,
            }
        
        # Generate audit IDs
        decision_audit_id = decision_record.get("audit_id", "unknown")
        
        challenge_content = json.dumps({
            "verdict": verdict_data["verdict"],
            "reasoning": verdict_data["reasoning"],
            "decision_audit_id": decision_audit_id,
        })
        audit_id = hashlib.sha256(challenge_content.encode()).hexdigest()[:16]
        
        return ChallengeRecord(
            verdict=verdict_data["verdict"],
            reasoning=verdict_data["reasoning"],
            cited_concerns=verdict_data.get("cited_concerns", []),
            requested_replan=verdict_data.get("requested_replan", False),
            replan_action=verdict_data.get("replan_action"),
            confidence=verdict_data.get("confidence", 0.5),
            audit_id=audit_id,
            previous_audit_id=decision_audit_id,
            timestamp=datetime.datetime.utcnow().isoformat() + "Z",
        )
    
    def _build_review_prompt(
        self,
        decision_record: Dict[str, Any],
        evidence: Dict[str, Any],
        action_floor: float,
    ) -> str:
        """Build the review prompt for the LLM."""
        prompt = f"""Review this decision:

DECISION:
Action: {decision_record.get('action', 'unknown')}
Confidence: {decision_record.get('confidence', 0.0):.2%}
Action Floor: {action_floor:.2%}
Result: {json.dumps(decision_record.get('result', {}), indent=2)}

EVIDENCE:
Number of passages: {len(evidence.get('passages', []))}
"""
        
        # Add passage details
        for i, passage in enumerate(evidence.get("passages", [])[:3], 1):
            prompt += f"\nPassage {i}:\n"
            prompt += f"  Similarity: {passage.get('similarity', 0.0):.2%}\n"
            prompt += f"  Source: {passage.get('source', 'unknown')}\n"
            prompt += f"  Text: {passage.get('text', '')[:200]}...\n"
        
        # Add specialist outputs if present
        if "specialist_outputs" in evidence:
            prompt += f"\nSpecialist Outputs:\n"
            for finding, prob in evidence["specialist_outputs"].items():
                prompt += f"  {finding}: {prob:.2%}\n"
        
        prompt += "\nProvide your verdict in JSON format as specified in the system prompt."
        
        return prompt
    
    def _call_featherless(self, prompt: str) -> str:
        """Call Featherless API."""
        response = self.client.post(
            f"{self.api_base}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.3,  # Lower temperature for more consistent verdicts
                "max_tokens": 500,
            },
        )
        response.raise_for_status()
        
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        
        # Extract JSON from markdown code blocks if present
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        
        return content
    
    def __del__(self):
        """Cleanup."""
        if hasattr(self, 'client'):
            self.client.close()


class MockCriticAgent:
    """
    Mock CriticAgent for testing without API key.
    
    Always approves decisions unless confidence is below 0.6,
    in which case it challenges.
    """
    
    def __init__(self, **kwargs):
        pass
    
    def review(
        self,
        decision_record: Dict[str, Any],
        evidence: Dict[str, Any],
        action_floor: float = 0.65,
    ) -> ChallengeRecord:
        """Mock review - simple rule-based logic."""
        confidence = decision_record.get("confidence", 0.0)
        action = decision_record.get("action", "unknown")
        evidence_count = len(evidence.get("passages", []))
        
        # Simple rules
        if confidence < 0.5:
            verdict = "ESCALATE"
            reasoning = f"Confidence {confidence:.2%} is critically low"
            concerns = ["Confidence below 0.5"]
            replan = False
            replan_action = None
        elif confidence < action_floor:
            verdict = "CHALLENGE"
            reasoning = f"Confidence {confidence:.2%} is below floor {action_floor:.2%}"
            concerns = [f"Confidence below action floor"]
            replan = True
            replan_action = "refine_rag_query" if evidence_count < 2 else "lower_confidence_floor"
        elif evidence_count < 2:
            verdict = "CHALLENGE"
            reasoning = f"Only {evidence_count} evidence passage(s), need 2+"
            concerns = ["Insufficient evidence passages"]
            replan = True
            replan_action = "refine_rag_query"
        else:
            verdict = "APPROVE"
            reasoning = f"Confidence {confidence:.2%} above floor, {evidence_count} passages"
            concerns = []
            replan = False
            replan_action = None
        
        # Generate audit IDs
        decision_audit_id = decision_record.get("audit_id", "mock_decision")
        challenge_content = json.dumps({
            "verdict": verdict,
            "reasoning": reasoning,
            "decision_audit_id": decision_audit_id,
        })
        audit_id = hashlib.sha256(challenge_content.encode()).hexdigest()[:16]
        
        return ChallengeRecord(
            verdict=verdict,
            reasoning=reasoning,
            cited_concerns=concerns,
            requested_replan=replan,
            replan_action=replan_action,
            confidence=0.85 if verdict == "APPROVE" else 0.65,
            audit_id=audit_id,
            previous_audit_id=decision_audit_id,
            timestamp=datetime.datetime.utcnow().isoformat() + "Z",
        )

# Made with Bob
