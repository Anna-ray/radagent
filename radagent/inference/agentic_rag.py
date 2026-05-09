"""
radagent.inference.agentic_rag
------------------------------
Agentic RAG: VLM-controlled evidence retrieval and self-audit.

The VLM evaluates passage sufficiency, refines queries when weak, and audits
its own report before finalizing. This is additive to the existing audit.py
trace builder — we're adding agentic decision-making, not replacing the audit.
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any

import httpx


@dataclass
class RetrievalDecision:
    """VLM decision on whether retrieved passages are sufficient."""
    sufficient: bool
    refined_query: str | None
    reasoning: str


async def evaluate_retrieval_sufficiency(
    finding_name: str,
    calibrated_prob: float,
    passages: list[dict],
    vllm_url: str,
    vllm_model: str,
) -> RetrievalDecision:
    """Ask the VLM whether retrieved passages are sufficient to write a
    clinically grounded report section.

    Args:
        finding_name: The finding being evaluated (e.g., "Cardiomegaly")
        calibrated_prob: Calibrated probability from specialist
        passages: List of retrieved passage dicts
        vllm_url: Base URL for vLLM server
        vllm_model: Model name

    Returns:
        RetrievalDecision with sufficiency assessment and optional refined query
    """
    print(f"[agentic-rag] evaluating sufficiency for {finding_name}", flush=True)

    # Format passages for prompt
    passages_formatted = []
    for i, p in enumerate(passages, 1):
        preview = (p.get("text", "")[:200] + "...") if len(p.get("text", "")) > 200 else p.get("text", "")
        passages_formatted.append(
            f"[{i}] {p.get('title', '?')} > {p.get('section', '?')} "
            f"({p.get('source', '?')}): {preview}"
        )
    passages_text = "\n".join(passages_formatted) if passages_formatted else "  (no passages retrieved)"

    prompt = f"""You are evaluating retrieval quality for a clinical radiology pipeline.

Finding: {finding_name}
Calibrated probability: {calibrated_prob:.3f}

Retrieved passages:
{passages_text}

Question: Are these passages sufficient to write a clinically grounded report section on this finding, with proper citations?

Reply with JSON ONLY in this exact schema:
{{
  "sufficient": true | false,
  "refined_query": "<a more specific query if not sufficient, else null>",
  "reasoning": "<one sentence>"
}}"""

    payload = {
        "model": vllm_model,
        "messages": [
            {"role": "system", "content": "You are a careful radiology assistant."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 200,
        "temperature": 0.1,
    }

    url = f"{vllm_url.rstrip('/')}/chat/completions"
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.post(url, json=payload)
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"]
            
            # Parse JSON response
            # Try to extract JSON from markdown code blocks if present
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            result = json.loads(content)
            return RetrievalDecision(
                sufficient=bool(result.get("sufficient", True)),
                refined_query=result.get("refined_query"),
                reasoning=result.get("reasoning", "no reasoning provided"),
            )
    except asyncio.TimeoutError:
        print(f"[agentic-rag] timeout evaluating {finding_name}, defaulting to sufficient=True", flush=True)
        return RetrievalDecision(sufficient=True, refined_query=None, reasoning="timeout")
    except json.JSONDecodeError as e:
        print(f"[agentic-rag] JSON parse error for {finding_name}: {e}, defaulting to sufficient=True", flush=True)
        return RetrievalDecision(sufficient=True, refined_query=None, reasoning="parse error")
    except Exception as e:
        print(f"[agentic-rag] error evaluating {finding_name}: {e}, defaulting to sufficient=True", flush=True)
        return RetrievalDecision(sufficient=True, refined_query=None, reasoning=f"error: {e}")


async def agentic_retrieve(
    finding: dict,
    retriever: Any,
    initial_query: str,
    vllm_url: str,
    vllm_model: str,
    k: int = 3,
    max_iterations: int = 2,
) -> tuple[list[dict], dict]:
    """Perform agentic retrieval: retrieve, evaluate sufficiency, refine if needed.

    Args:
        finding: Finding dict with name, calibrated_probability, etc.
        retriever: RadRetriever instance
        initial_query: Initial query string
        vllm_url: Base URL for vLLM server
        vllm_model: Model name
        k: Number of passages to retrieve per query
        max_iterations: Maximum refinement iterations

    Returns:
        Tuple of (passages, trace_dict) where trace contains queries_used,
        decisions, and total_iterations
    """
    finding_name = finding["name"]
    calibrated_prob = finding["calibrated_probability"]
    
    print(f"[agentic-rag] starting agentic retrieval for {finding_name}", flush=True)
    
    queries_used = [initial_query]
    decisions = []
    all_passages = []
    seen_chunk_ids = set()
    
    # First retrieval pass
    passages = retriever.query(initial_query, k=k, finding_filter=[finding_name])
    if not passages:
        passages = retriever.query(initial_query, k=k)
    
    # Convert to dicts and track chunk IDs
    for p in passages:
        p_dict = p.to_dict()
        chunk_id = p_dict.get("chunk_id")
        if chunk_id not in seen_chunk_ids:
            all_passages.append(p_dict)
            seen_chunk_ids.add(chunk_id)
    
    print(f"[agentic-rag] initial retrieval: {len(all_passages)} passages", flush=True)
    
    # Evaluate sufficiency
    decision = await evaluate_retrieval_sufficiency(
        finding_name=finding_name,
        calibrated_prob=calibrated_prob,
        passages=all_passages,
        vllm_url=vllm_url,
        vllm_model=vllm_model,
    )
    decisions.append({
        "iteration": 1,
        "sufficient": decision.sufficient,
        "refined_query": decision.refined_query,
        "reasoning": decision.reasoning,
    })
    
    print(f"[agentic-rag] decision: sufficient={decision.sufficient}, reasoning={decision.reasoning}", flush=True)
    
    # Refinement loop
    iteration = 1
    while not decision.sufficient and decision.refined_query and iteration < max_iterations:
        iteration += 1
        refined_query = decision.refined_query
        queries_used.append(refined_query)
        
        print(f"[agentic-rag] iteration {iteration}: refining with query: {refined_query[:80]}...", flush=True)
        
        # Re-query with refined query
        passages = retriever.query(refined_query, k=k, finding_filter=[finding_name])
        if not passages:
            passages = retriever.query(refined_query, k=k)
        
        # Merge results, dedup by chunk_id
        new_count = 0
        for p in passages:
            p_dict = p.to_dict()
            chunk_id = p_dict.get("chunk_id")
            if chunk_id not in seen_chunk_ids:
                all_passages.append(p_dict)
                seen_chunk_ids.add(chunk_id)
                new_count += 1
        
        print(f"[agentic-rag] added {new_count} new passages (total: {len(all_passages)})", flush=True)
        
        # Re-evaluate
        decision = await evaluate_retrieval_sufficiency(
            finding_name=finding_name,
            calibrated_prob=calibrated_prob,
            passages=all_passages,
            vllm_url=vllm_url,
            vllm_model=vllm_model,
        )
        decisions.append({
            "iteration": iteration,
            "sufficient": decision.sufficient,
            "refined_query": decision.refined_query,
            "reasoning": decision.reasoning,
        })
        
        print(f"[agentic-rag] decision: sufficient={decision.sufficient}, reasoning={decision.reasoning}", flush=True)
    
    trace = {
        "finding_name": finding_name,
        "queries_used": queries_used,
        "decisions": decisions,
        "total_iterations": iteration,
        "final_passage_count": len(all_passages),
    }
    
    print(f"[agentic-rag] completed for {finding_name}: {iteration} iterations, {len(all_passages)} passages", flush=True)
    
    return all_passages, trace


async def self_audit_report(
    report_text: str,
    structured_findings: dict,
    retrieved_passages_by_finding: dict,
    vllm_url: str,
    vllm_model: str,
) -> dict:
    """Ask the VLM to audit its own report against the evidence.

    Args:
        report_text: The generated report text
        structured_findings: The structured findings dict from the specialist
        retrieved_passages_by_finding: Dict mapping finding_name -> list[passage_dict]
        vllm_url: Base URL for vLLM server
        vllm_model: Model name

    Returns:
        Dict with flags (list of issues) and audit_summary
    """
    print("[agentic-rag] starting self-audit of report", flush=True)
    
    # Format findings
    findings_formatted = []
    for f in structured_findings.get("findings", []):
        if f.get("above_threshold"):
            findings_formatted.append(
                f"- {f['name']}: calibrated_probability={f['calibrated_probability']:.3f}, "
                f"threshold={f['threshold']:.3f}, confidence={f['confidence_level']}"
            )
    findings_text = "\n".join(findings_formatted) if findings_formatted else "  (no findings above threshold)"
    
    # Format passages
    passages_formatted = []
    for finding_name, passages in retrieved_passages_by_finding.items():
        for i, p in enumerate(passages, 1):
            preview = (p.get("text", "")[:150] + "...") if len(p.get("text", "")) > 150 else p.get("text", "")
            passages_formatted.append(
                f"[{finding_name} #{i}] {p.get('title', '?')} > {p.get('section', '?')}: {preview}"
            )
    passages_text = "\n".join(passages_formatted) if passages_formatted else "  (no passages retrieved)"
    
    prompt = f"""You are auditing a radiology report against the underlying evidence.

Specialist findings (calibrated probabilities):
{findings_text}

Retrieved passages (per finding):
{passages_text}

Generated report:
{report_text}

For each claim in the report, check:
1. Does it match the specialist probabilities? (Don't claim what wasn't above threshold.)
2. Do the cited passages actually support it?

Reply with JSON ONLY:
{{
  "flags": [
    {{"claim": "<text>", "reason": "<why flagged>", "severity": "low|med|high"}}
  ],
  "audit_summary": "<one sentence>"
}}

If everything is consistent, return {{"flags": [], "audit_summary": "all claims consistent"}}."""

    payload = {
        "model": vllm_model,
        "messages": [
            {"role": "system", "content": "You are a careful radiology assistant."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 400,
        "temperature": 0.1,
    }

    url = f"{vllm_url.rstrip('/')}/chat/completions"
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.post(url, json=payload)
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"]
            
            # Parse JSON response
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            result = json.loads(content)
            
            flags = result.get("flags", [])
            audit_summary = result.get("audit_summary", "audit completed")
            
            print(f"[agentic-rag] self-audit complete: {len(flags)} flags, summary: {audit_summary}", flush=True)
            
            return {
                "flags": flags,
                "audit_summary": audit_summary,
            }
    except asyncio.TimeoutError:
        print("[agentic-rag] self-audit timeout", flush=True)
        return {
            "flags": [],
            "audit_summary": "audit timed out",
        }
    except json.JSONDecodeError as e:
        print(f"[agentic-rag] self-audit JSON parse error: {e}", flush=True)
        return {
            "flags": [],
            "audit_summary": f"audit parse error: {e}",
        }
    except Exception as e:
        print(f"[agentic-rag] self-audit error: {e}", flush=True)
        return {
            "flags": [],
            "audit_summary": f"audit error: {e}",
        }

# Made with Bob
