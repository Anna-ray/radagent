"""
Dictation Auditor - Compare Radiologist Dictation vs Specialist Findings

Audits radiologist voice dictations against specialist model predictions to
surface discrepancies. Uses Gemini Flash for structured parsing of dictated
findings and comparison logic.

Author: Rayane Aggoune
"""

import os
import json
import hashlib
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


class DiscrepancyType(Enum):
    """Types of discrepancies between dictation and specialist."""
    RECONSIDER = "RECONSIDER"  # Dictated absent but specialist found with high confidence
    CONFIRM = "CONFIRM"  # Dictated present but specialist says absent with high confidence
    CONSISTENT = "CONSISTENT"  # Both agree
    WEAK_EVIDENCE = "WEAK_EVIDENCE"  # Specialist confidence too low to flag


@dataclass
class DictatedFinding:
    """A finding extracted from radiologist dictation."""
    finding_name: str
    asserted_state: str  # "present" or "absent"
    confidence: float
    text_span: str  # The actual dictated phrase


@dataclass
class Discrepancy:
    """A discrepancy between dictation and specialist."""
    finding_name: str
    discrepancy_type: DiscrepancyType
    dictated_state: str
    specialist_probability: float
    specialist_threshold: float
    specialist_above_threshold: bool
    explanation: str
    severity: str  # "high", "medium", "low"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "finding_name": self.finding_name,
            "discrepancy_type": self.discrepancy_type.value,
            "dictated_state": self.dictated_state,
            "specialist_probability": self.specialist_probability,
            "specialist_threshold": self.specialist_threshold,
            "specialist_above_threshold": self.specialist_above_threshold,
            "explanation": self.explanation,
            "severity": self.severity,
        }


@dataclass
class AuditReport:
    """Complete audit report comparing dictation vs specialist."""
    transcript: str
    dictated_findings: List[DictatedFinding]
    discrepancies: List[Discrepancy]
    audit_id: str
    timestamp: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "transcript": self.transcript,
            "dictated_findings": [asdict(f) for f in self.dictated_findings],
            "discrepancies": [d.to_dict() for d in self.discrepancies],
            "audit_id": self.audit_id,
            "timestamp": self.timestamp,
        }
    
    def has_critical_discrepancies(self) -> bool:
        """Check if any high-severity discrepancies exist."""
        return any(
            d.severity == "high" and d.discrepancy_type == DiscrepancyType.RECONSIDER
            for d in self.discrepancies
        )


class DictationAuditor:
    """
    Audits radiologist dictations against specialist model predictions.
    
    Uses Gemini Flash for structured parsing of dictated findings and
    comparison logic to surface discrepancies.
    """
    
    # NIH ChestX-ray14 class names (must match specialist output)
    CHEST_XRAY_CLASSES = [
        "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
        "Effusion", "Emphysema", "Fibrosis", "Hernia",
        "Infiltration", "Mass", "Nodule", "Pleural_Thickening",
        "Pneumonia", "Pneumothorax"
    ]
    
    # Confidence threshold for flagging discrepancies
    HIGH_CONFIDENCE_THRESHOLD = 0.75
    
    def __init__(
        self,
        gemini_api_key: Optional[str] = None,
        model_name: str = "gemini-1.5-flash",
    ):
        """
        Initialize dictation auditor.
        
        Args:
            gemini_api_key: Google AI Studio API key (or set GEMINI_API_KEY env var)
            model_name: Gemini model to use for parsing
        """
        if not GEMINI_AVAILABLE:
            raise ImportError(
                "google-generativeai not installed. "
                "Install with: pip install google-generativeai"
            )
        
        self.api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Gemini API key required. Set GEMINI_API_KEY environment "
                "variable or pass gemini_api_key parameter."
            )
        
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model_name)
    
    def audit(
        self,
        transcript: str,
        specialist_findings: List[Dict[str, Any]],
    ) -> AuditReport:
        """
        Audit dictation against specialist findings.
        
        Args:
            transcript: Full transcribed text from radiologist
            specialist_findings: List of findings from specialist model, each with:
                - finding: str (class name)
                - probability: float
                - threshold: float
                - above_threshold: bool
        
        Returns:
            AuditReport with discrepancies
        """
        import datetime
        
        # Step 1: Parse dictation into structured findings
        dictated_findings = self._parse_dictation(transcript)
        
        # Step 2: Compare against specialist findings
        discrepancies = self._compare_findings(
            dictated_findings,
            specialist_findings,
        )
        
        # Step 3: Generate audit ID
        audit_id = hashlib.sha256(
            f"{transcript}{json.dumps([asdict(f) for f in dictated_findings])}".encode()
        ).hexdigest()[:16]
        
        return AuditReport(
            transcript=transcript,
            dictated_findings=dictated_findings,
            discrepancies=discrepancies,
            audit_id=audit_id,
            timestamp=datetime.datetime.utcnow().isoformat() + "Z",
        )
    
    def _parse_dictation(self, transcript: str) -> List[DictatedFinding]:
        """
        Parse radiologist dictation into structured findings.
        
        Uses Gemini Flash to extract findings and their states (present/absent).
        
        Args:
            transcript: Raw transcribed text
        
        Returns:
            List of DictatedFinding objects
        """
        prompt = f"""You are a medical NLP system parsing radiologist dictations.

Extract all chest X-ray findings mentioned in the following dictation.
For each finding, determine if it is PRESENT or ABSENT.

Valid finding names (use these exact names):
{', '.join(self.CHEST_XRAY_CLASSES)}

Dictation:
"{transcript}"

Return a JSON array of findings in this format:
[
  {{
    "finding_name": "Effusion",
    "asserted_state": "absent",
    "confidence": 0.95,
    "text_span": "no pleural effusion"
  }}
]

Rules:
- Use exact finding names from the list above
- asserted_state must be "present" or "absent"
- confidence is your certainty (0.0-1.0)
- text_span is the actual phrase from the dictation
- If a finding is not mentioned, do not include it
- Negations like "no", "clear", "unremarkable" mean "absent"
- Positive mentions like "present", "seen", "noted" mean "present"

Return ONLY the JSON array, no other text.
"""
        
        try:
            response = self.model.generate_content(prompt)
            findings_json = response.text.strip()
            
            # Remove markdown code blocks if present
            if findings_json.startswith("```"):
                findings_json = findings_json.split("```")[1]
                if findings_json.startswith("json"):
                    findings_json = findings_json[4:]
            
            findings_data = json.loads(findings_json)
            
            return [
                DictatedFinding(
                    finding_name=f["finding_name"],
                    asserted_state=f["asserted_state"],
                    confidence=f["confidence"],
                    text_span=f["text_span"],
                )
                for f in findings_data
            ]
        
        except Exception as e:
            # Fallback: return empty list if parsing fails
            print(f"Warning: Failed to parse dictation: {e}")
            return []
    
    def _compare_findings(
        self,
        dictated_findings: List[DictatedFinding],
        specialist_findings: List[Dict[str, Any]],
    ) -> List[Discrepancy]:
        """
        Compare dictated findings against specialist predictions.
        
        Args:
            dictated_findings: Findings extracted from dictation
            specialist_findings: Findings from specialist model
        
        Returns:
            List of Discrepancy objects
        """
        discrepancies = []
        
        # Build specialist lookup
        specialist_map = {
            f["finding"]: f for f in specialist_findings
        }
        
        # Check each dictated finding
        for dictated in dictated_findings:
            finding_name = dictated.finding_name
            
            if finding_name not in specialist_map:
                continue  # Skip if specialist doesn't have this finding
            
            specialist = specialist_map[finding_name]
            specialist_prob = specialist["probability"]
            specialist_threshold = specialist["threshold"]
            specialist_above = specialist["above_threshold"]
            
            # Determine discrepancy type
            discrepancy_type = None
            explanation = ""
            severity = "low"
            
            if dictated.asserted_state == "absent":
                # Radiologist says absent
                if specialist_above and specialist_prob >= self.HIGH_CONFIDENCE_THRESHOLD:
                    # But specialist found it with high confidence
                    discrepancy_type = DiscrepancyType.RECONSIDER
                    explanation = (
                        f"Radiologist dictated '{finding_name}' as absent "
                        f"('{dictated.text_span}'), but specialist detected it "
                        f"with {specialist_prob:.2%} confidence (threshold: "
                        f"{specialist_threshold:.2%}). Consider reviewing the image."
                    )
                    severity = "high"
                elif specialist_above:
                    # Specialist found it but with moderate confidence
                    discrepancy_type = DiscrepancyType.RECONSIDER
                    explanation = (
                        f"Radiologist dictated '{finding_name}' as absent, "
                        f"but specialist detected it with {specialist_prob:.2%} "
                        f"confidence. Moderate concern."
                    )
                    severity = "medium"
                else:
                    # Both agree it's absent (specialist below threshold)
                    discrepancy_type = DiscrepancyType.CONSISTENT
                    explanation = f"Both agree '{finding_name}' is absent."
                    severity = "low"
            
            elif dictated.asserted_state == "present":
                # Radiologist says present
                if specialist_above:
                    # Specialist also found it
                    discrepancy_type = DiscrepancyType.CONSISTENT
                    explanation = (
                        f"Both agree '{finding_name}' is present "
                        f"(specialist: {specialist_prob:.2%})."
                    )
                    severity = "low"
                elif specialist_prob < specialist_threshold * 0.5:
                    # Specialist strongly disagrees
                    discrepancy_type = DiscrepancyType.CONFIRM
                    explanation = (
                        f"Radiologist dictated '{finding_name}' as present "
                        f"('{dictated.text_span}'), but specialist probability "
                        f"is only {specialist_prob:.2%}. Consider confirming."
                    )
                    severity = "medium"
                else:
                    # Specialist is uncertain
                    discrepancy_type = DiscrepancyType.WEAK_EVIDENCE
                    explanation = (
                        f"Radiologist says '{finding_name}' is present, "
                        f"specialist is uncertain ({specialist_prob:.2%})."
                    )
                    severity = "low"
            
            if discrepancy_type:
                discrepancies.append(Discrepancy(
                    finding_name=finding_name,
                    discrepancy_type=discrepancy_type,
                    dictated_state=dictated.asserted_state,
                    specialist_probability=specialist_prob,
                    specialist_threshold=specialist_threshold,
                    specialist_above_threshold=specialist_above,
                    explanation=explanation,
                    severity=severity,
                ))
        
        return discrepancies


# Mock auditor for testing without API key
class MockDictationAuditor:
    """Mock auditor for testing without Gemini API."""
    
    def __init__(self, **kwargs):
        pass
    
    def audit(
        self,
        transcript: str,
        specialist_findings: List[Dict[str, Any]],
    ) -> AuditReport:
        """Return mock audit report."""
        import datetime
        
        # Mock: assume transcript says "no effusion" but specialist found it
        dictated_findings = [
            DictatedFinding(
                finding_name="Effusion",
                asserted_state="absent",
                confidence=0.95,
                text_span="no pleural effusion",
            )
        ]
        
        discrepancies = [
            Discrepancy(
                finding_name="Effusion",
                discrepancy_type=DiscrepancyType.RECONSIDER,
                dictated_state="absent",
                specialist_probability=0.93,
                specialist_threshold=0.45,
                specialist_above_threshold=True,
                explanation="Radiologist dictated 'Effusion' as absent, but specialist detected it with 93% confidence.",
                severity="high",
            )
        ]
        
        return AuditReport(
            transcript=transcript,
            dictated_findings=dictated_findings,
            discrepancies=discrepancies,
            audit_id="mock_audit_123",
            timestamp=datetime.datetime.utcnow().isoformat() + "Z",
        )

# Made with Bob
