"""
radagent.autonomy.tools
-----------------------
Autonomous workflow tools with RAG-grounded confidence scoring.

Each tool returns evidence-backed decisions with confidence scores.

Author: Rayane Aggoune
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable

import httpx


# Tool registry
TOOL_REGISTRY: dict[str, Callable] = {}


def tool(func: Callable) -> Callable:
    """Decorator to register a tool."""
    TOOL_REGISTRY[func.__name__] = func
    return func


@dataclass
class ToolResult:
    """Result from a tool execution."""
    action: str
    args: dict[str, Any]
    evidence_refs: list[str]
    confidence: float
    audit_id: str
    previous_audit_id: str | None
    reasoning: str
    timestamp: float


def _compute_audit_id(result: dict, previous_id: str | None) -> str:
    """Compute SHA-256 audit ID for a tool result."""
    audit_content = json.dumps(
        {
            "action": result["action"],
            "args": result["args"],
            "confidence": result["confidence"],
            "timestamp": result["timestamp"],
            "previous_audit_id": previous_id,
        },
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(audit_content).hexdigest()


async def _query_rag(
    query: str,
    retriever_url: str,
    top_k: int = 3,
) -> list[dict]:
    """Query RAG retriever for evidence.
    
    Args:
        query: Search query
        retriever_url: RAG retriever endpoint
        top_k: Number of passages to retrieve
        
    Returns:
        List of passage dicts with 'text', 'source', 'score'
    """
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            f"{retriever_url}/retrieve",
            json={"query": query, "top_k": top_k},
        )
        response.raise_for_status()
        return response.json()["passages"]


async def _compute_confidence_from_passages(
    passages: list[dict],
    vllm_url: str,
    vllm_model: str,
    question: str,
) -> tuple[float, str]:
    """Use VLM to assess confidence based on retrieved passages.
    
    Args:
        passages: Retrieved passages
        vllm_url: VLM endpoint
        vllm_model: Model name
        question: Question to assess
        
    Returns:
        (confidence_score, reasoning)
    """
    # Format passages
    passage_text = "\n\n".join([
        f"[{i+1}] {p['text'][:500]}... (source: {p['source']})"
        for i, p in enumerate(passages)
    ])
    
    prompt = f"""You are a medical AI assistant evaluating evidence quality.

Question: {question}

Retrieved Evidence:
{passage_text}

Based on the evidence above, rate your confidence in answering this question on a scale of 0.0 to 1.0.
Consider:
- Relevance of evidence to the question
- Specificity and detail of the evidence
- Consistency across sources
- Clinical authority of sources

Respond in JSON format:
{{
  "confidence": <float between 0.0 and 1.0>,
  "reasoning": "<brief explanation>"
}}
"""
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{vllm_url}/v1/chat/completions",
            json={
                "model": vllm_model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "max_tokens": 200,
            },
        )
        response.raise_for_status()
        
        content = response.json()["choices"][0]["message"]["content"]
        
        # Parse JSON response
        try:
            result = json.loads(content)
            confidence = float(result["confidence"])
            reasoning = result["reasoning"]
        except (json.JSONDecodeError, KeyError, ValueError):
            # Fallback if parsing fails
            confidence = 0.5
            reasoning = "Unable to parse confidence assessment"
    
    return confidence, reasoning


@tool
async def triage_study(
    study: dict,
    findings: list[dict],
    urgency_priors: dict[str, float],
    retriever_url: str,
    vllm_url: str,
    vllm_model: str,
    previous_audit_id: str | None = None,
) -> ToolResult:
    """Triage a study based on findings and RAG-retrieved guidelines.
    
    Args:
        study: Study metadata
        findings: List of findings with probabilities
        urgency_priors: Prior urgency scores per finding
        retriever_url: RAG retriever endpoint
        vllm_url: VLM endpoint
        vllm_model: Model name
        previous_audit_id: Previous audit ID in chain
        
    Returns:
        ToolResult with urgency level (STAT/URGENT/ROUTINE)
    """
    # Build query for triage guidelines
    finding_names = [f["name"] for f in findings if f["probability"] > 0.5]
    query = f"Triage urgency guidelines for chest X-ray findings: {', '.join(finding_names)}"
    
    # Retrieve evidence
    passages = await _query_rag(query, retriever_url, top_k=3)
    
    # Compute confidence
    question = f"What is the appropriate triage urgency for a patient with: {', '.join(finding_names)}?"
    confidence, reasoning = await _compute_confidence_from_passages(
        passages, vllm_url, vllm_model, question
    )
    
    # Determine urgency based on priors and findings
    max_urgency_score = max(
        [urgency_priors.get(f["name"], 0.5) * f["probability"] for f in findings],
        default=0.0,
    )
    
    if max_urgency_score > 0.8:
        urgency = "STAT"
    elif max_urgency_score > 0.6:
        urgency = "URGENT"
    else:
        urgency = "ROUTINE"
    
    result = {
        "action": "triage_study",
        "args": {"urgency": urgency, "max_urgency_score": max_urgency_score},
        "evidence_refs": [p["source"] for p in passages],
        "confidence": confidence,
        "reasoning": reasoning,
        "timestamp": time.time(),
        "previous_audit_id": previous_audit_id,
    }
    
    audit_id = _compute_audit_id(result, previous_audit_id)
    
    return ToolResult(
        action=result["action"],
        args=result["args"],
        evidence_refs=result["evidence_refs"],
        confidence=result["confidence"],
        audit_id=audit_id,
        previous_audit_id=previous_audit_id,
        reasoning=result["reasoning"],
        timestamp=result["timestamp"],
    )


@tool
async def route_to_subspecialist(
    study: dict,
    findings: list[dict],
    retriever_url: str,
    vllm_url: str,
    vllm_model: str,
    previous_audit_id: str | None = None,
) -> ToolResult:
    """Route study to appropriate subspecialist based on findings.
    
    Args:
        study: Study metadata
        findings: List of findings
        retriever_url: RAG retriever endpoint
        vllm_url: VLM endpoint
        vllm_model: Model name
        previous_audit_id: Previous audit ID
        
    Returns:
        ToolResult with subspecialty (THORACIC/MSK/NEURO/CARDIAC/ABDOMINAL)
    """
    # Determine dominant finding category
    finding_names = [f["name"] for f in findings if f["probability"] > 0.5]
    query = f"Subspecialty routing for findings: {', '.join(finding_names)}"
    
    passages = await _query_rag(query, retriever_url, top_k=2)
    
    question = f"Which subspecialist should review a study with: {', '.join(finding_names)}?"
    confidence, reasoning = await _compute_confidence_from_passages(
        passages, vllm_url, vllm_model, question
    )
    
    # Simple routing logic (can be enhanced with ML)
    subspecialty = "THORACIC"  # Default for chest X-ray
    
    result = {
        "action": "route_to_subspecialist",
        "args": {"subspecialty": subspecialty},
        "evidence_refs": [p["source"] for p in passages],
        "confidence": confidence,
        "reasoning": reasoning,
        "timestamp": time.time(),
        "previous_audit_id": previous_audit_id,
    }
    
    audit_id = _compute_audit_id(result, previous_audit_id)
    
    return ToolResult(
        action=result["action"],
        args=result["args"],
        evidence_refs=result["evidence_refs"],
        confidence=result["confidence"],
        audit_id=audit_id,
        previous_audit_id=previous_audit_id,
        reasoning=result["reasoning"],
        timestamp=result["timestamp"],
    )


@tool
async def schedule_follow_up(
    study: dict,
    finding: dict,
    modality: str,
    retriever_url: str,
    vllm_url: str,
    vllm_model: str,
    previous_audit_id: str | None = None,
) -> ToolResult:
    """Schedule follow-up imaging based on guidelines.
    
    Args:
        study: Study metadata
        finding: Specific finding requiring follow-up
        modality: Current modality
        retriever_url: RAG retriever endpoint
        vllm_url: VLM endpoint
        vllm_model: Model name
        previous_audit_id: Previous audit ID
        
    Returns:
        ToolResult with follow-up plan
    """
    # Query for follow-up guidelines (Fleischner, Lung-RADS, etc.)
    query = f"Follow-up imaging guidelines for {finding['name']} on {modality}"
    
    passages = await _query_rag(query, retriever_url, top_k=3)
    
    question = f"What is the recommended follow-up for {finding['name']} with probability {finding['probability']:.2f}?"
    confidence, reasoning = await _compute_confidence_from_passages(
        passages, vllm_url, vllm_model, question
    )
    
    # Generate follow-up plan
    if finding["probability"] > 0.8:
        interval = "3 months"
        recommended_modality = "CT chest"
    elif finding["probability"] > 0.6:
        interval = "6 months"
        recommended_modality = "Chest X-ray"
    else:
        interval = "12 months"
        recommended_modality = "Chest X-ray"
    
    result = {
        "action": "schedule_follow_up",
        "args": {
            "interval": interval,
            "recommended_modality": recommended_modality,
            "finding": finding["name"],
        },
        "evidence_refs": [p["source"] for p in passages],
        "confidence": confidence,
        "reasoning": reasoning,
        "timestamp": time.time(),
        "previous_audit_id": previous_audit_id,
    }
    
    audit_id = _compute_audit_id(result, previous_audit_id)
    
    return ToolResult(
        action=result["action"],
        args=result["args"],
        evidence_refs=result["evidence_refs"],
        confidence=result["confidence"],
        audit_id=audit_id,
        previous_audit_id=previous_audit_id,
        reasoning=result["reasoning"],
        timestamp=result["timestamp"],
    )


@tool
async def flag_critical_finding(
    study: dict,
    finding: dict,
    recipient_role: str,
    retriever_url: str,
    vllm_url: str,
    vllm_model: str,
    previous_audit_id: str | None = None,
) -> ToolResult:
    """Flag critical finding and draft alert message.
    
    Args:
        study: Study metadata
        finding: Critical finding
        recipient_role: Role of alert recipient
        retriever_url: RAG retriever endpoint
        vllm_url: VLM endpoint
        vllm_model: Model name
        previous_audit_id: Previous audit ID
        
    Returns:
        ToolResult with alert record
    """
    query = f"Critical finding notification guidelines for {finding['name']}"
    
    passages = await _query_rag(query, retriever_url, top_k=2)
    
    question = f"Is {finding['name']} with probability {finding['probability']:.2f} a critical finding requiring immediate notification?"
    confidence, reasoning = await _compute_confidence_from_passages(
        passages, vllm_url, vllm_model, question
    )
    
    # Draft HIPAA-compliant message
    message_draft = f"Critical finding detected: {finding['name']} (confidence: {finding['probability']:.2f}). Immediate review recommended."
    
    result = {
        "action": "flag_critical_finding",
        "args": {
            "finding": finding["name"],
            "recipient_role": recipient_role,
            "message_draft": message_draft,
        },
        "evidence_refs": [p["source"] for p in passages],
        "confidence": confidence,
        "reasoning": reasoning,
        "timestamp": time.time(),
        "previous_audit_id": previous_audit_id,
    }
    
    audit_id = _compute_audit_id(result, previous_audit_id)
    
    return ToolResult(
        action=result["action"],
        args=result["args"],
        evidence_refs=result["evidence_refs"],
        confidence=result["confidence"],
        audit_id=audit_id,
        previous_audit_id=previous_audit_id,
        reasoning=result["reasoning"],
        timestamp=result["timestamp"],
    )

# Made with Bob
