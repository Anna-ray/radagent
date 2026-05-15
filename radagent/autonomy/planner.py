"""
radagent.autonomy.planner
-------------------------
Workflow planner with replan triggers for handling roadblocks.

Author: Rayane Aggoune
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import asdict, dataclass
from typing import Any

import httpx
import os

from radagent.autonomy.halt import should_halt
from radagent.autonomy.tools import TOOL_REGISTRY, ToolResult


@dataclass
class WorkflowStep:
    """A step in the workflow plan."""
    tool_name: str
    args: dict[str, Any]
    description: str


@dataclass
class ExecutionRecord:
    """Record of a tool execution."""
    step: WorkflowStep
    result: ToolResult | None
    halted: bool
    halt_reason: str | None
    replanned: bool
    replan_reason: str | None


class WorkflowPlanner:
    """Autonomous workflow planner with replan capability.
    
    Plans, executes, and replans workflows based on confidence and roadblocks.
    
    Args:
        featherless_api_key: Featherless API key for routing
        vllm_url: VLM endpoint for grounding
        vllm_model: VLM model name
        retriever_url: RAG retriever endpoint
    """
    
    def __init__(
        self,
        featherless_api_key: str,
        vllm_url: str,
        vllm_model: str,
        retriever_url: str,
    ):
        self.featherless_api_key = featherless_api_key
        self.vllm_url = vllm_url
        self.vllm_model = vllm_model
        self.retriever_url = retriever_url
        
    async def plan(self, study: dict) -> list[WorkflowStep]:
        """Generate initial workflow plan for a study.
        
        Uses Featherless chat completions for fast tool-call routing.
        
        Args:
            study: Study metadata with findings
            
        Returns:
            List of workflow steps
        """
        # Build prompt for Featherless
        findings_summary = ", ".join([
            f"{f['name']} ({f['probability']:.2f})"
            for f in study.get("findings", [])
        ])
        
        prompt = f"""You are a radiology workflow planner. Given a chest X-ray study with findings, generate an ordered workflow plan.

Study: {study.get('study_id', 'unknown')}
Findings: {findings_summary}

Available tools:
1. triage_study - Determine urgency (STAT/URGENT/ROUTINE)
2. route_to_subspecialist - Route to appropriate subspecialist
3. flag_critical_finding - Flag critical findings for immediate notification
4. schedule_follow_up - Schedule follow-up imaging based on guidelines

Generate a workflow plan as a JSON array of steps:
[
  {{"tool": "triage_study", "description": "Assess urgency"}},
  {{"tool": "route_to_subspecialist", "description": "Route to subspecialist"}},
  ...
]

Only include necessary steps. Not all studies need all tools.
"""
        
        # Pivoted from Gemini to Featherless (Google project blocked)
        async with httpx.AsyncClient(timeout=30.0) as client:
            endpoint = "https://api.featherless.ai/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {os.getenv('FEATHERLESS_API_KEY')}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": "Qwen/Qwen2.5-7B-Instruct",
                "messages": [
                    {"role": "system", "content": "You are a radiology workflow planner."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.3,
                "max_tokens": 512,
            }

            response = await client.post(endpoint, headers=headers, json=payload, timeout=30.0)
            response.raise_for_status()

            result = response.json()
            # Featherless chat response message text
            try:
                content = result["choices"][0]["message"]["content"]
            except Exception:
                content = json.dumps(result)

            # Extract JSON from response
            import re
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            if json_match:
                plan_data = json.loads(json_match.group())
            else:
                # Fallback plan
                plan_data = [
                    {"tool": "triage_study", "description": "Assess urgency"},
                    {"tool": "route_to_subspecialist", "description": "Route to subspecialist"},
                ]
        
        # Convert to WorkflowStep objects
        steps = []
        for item in plan_data:
            steps.append(WorkflowStep(
                tool_name=item["tool"],
                args={},  # Will be filled during execution
                description=item["description"],
            ))
        
        return steps
    
    async def execute(
        self,
        plan: list[WorkflowStep],
        study_context: dict,
    ) -> list[ExecutionRecord]:
        """Execute workflow plan with halt and replan logic.
        
        Args:
            plan: Workflow plan
            study_context: Study context with findings
            
        Returns:
            List of execution records
        """
        records = []
        previous_audit_id = None
        
        for i, step in enumerate(plan):
            print(f"\n[autonomy] Executing step {i+1}/{len(plan)}: {step.description}")
            
            # Get tool function
            tool_func = TOOL_REGISTRY.get(step.tool_name)
            if not tool_func:
                print(f"[autonomy] ERROR: Tool '{step.tool_name}' not found")
                records.append(ExecutionRecord(
                    step=step,
                    result=None,
                    halted=True,
                    halt_reason=f"Tool '{step.tool_name}' not found",
                    replanned=False,
                    replan_reason=None,
                ))
                continue
            
            # Prepare arguments
            args = self._prepare_tool_args(step.tool_name, study_context, previous_audit_id)
            
            # Execute tool
            try:
                result = await tool_func(**args)
                
                # Check halt condition
                halt_decision = should_halt(step.tool_name, result.confidence)
                
                if halt_decision.halt:
                    print(f"[autonomy] HALT: {halt_decision.reason}")
                    
                    # Attempt replan
                    if halt_decision.recommended_action == "refine_and_retry":
                        print(f"[autonomy] Attempting replan...")
                        replanned = await self._replan_step(step, result, study_context)
                        
                        if replanned:
                            # Retry with refined query
                            args = self._prepare_tool_args(step.tool_name, study_context, previous_audit_id, refined=True)
                            result = await tool_func(**args)
                            
                            # Check again
                            halt_decision = should_halt(step.tool_name, result.confidence)
                            if not halt_decision.halt:
                                print(f"[autonomy] Replan successful! Confidence: {result.confidence:.3f}")
                                records.append(ExecutionRecord(
                                    step=step,
                                    result=result,
                                    halted=False,
                                    halt_reason=None,
                                    replanned=True,
                                    replan_reason="Low confidence - refined RAG query",
                                ))
                                previous_audit_id = result.audit_id
                                continue
                    
                    # Halt execution
                    records.append(ExecutionRecord(
                        step=step,
                        result=result,
                        halted=True,
                        halt_reason=halt_decision.reason,
                        replanned=False,
                        replan_reason=None,
                    ))
                    break
                
                # Success
                print(f"[autonomy] Success! Confidence: {result.confidence:.3f}")
                records.append(ExecutionRecord(
                    step=step,
                    result=result,
                    halted=False,
                    halt_reason=None,
                    replanned=False,
                    replan_reason=None,
                ))
                previous_audit_id = result.audit_id
                
            except Exception as e:
                print(f"[autonomy] ERROR: {e}")
                records.append(ExecutionRecord(
                    step=step,
                    result=None,
                    halted=True,
                    halt_reason=str(e),
                    replanned=False,
                    replan_reason=None,
                ))
                break
        
        return records
    
    def _prepare_tool_args(
        self,
        tool_name: str,
        study_context: dict,
        previous_audit_id: str | None,
        refined: bool = False,
    ) -> dict[str, Any]:
        """Prepare arguments for a tool call.
        
        Args:
            tool_name: Tool name
            study_context: Study context
            previous_audit_id: Previous audit ID
            refined: Whether this is a refined retry
            
        Returns:
            Tool arguments
        """
        base_args = {
            "retriever_url": self.retriever_url,
            "vllm_url": self.vllm_url,
            "vllm_model": self.vllm_model,
            "previous_audit_id": previous_audit_id,
        }
        
        if tool_name == "triage_study":
            return {
                **base_args,
                "study": study_context,
                "findings": study_context.get("findings", []),
                "urgency_priors": {
                    "Pneumothorax": 0.9,
                    "Pneumonia": 0.7,
                    "Effusion": 0.6,
                    "Cardiomegaly": 0.5,
                },
            }
        elif tool_name == "route_to_subspecialist":
            return {
                **base_args,
                "study": study_context,
                "findings": study_context.get("findings", []),
            }
        elif tool_name == "schedule_follow_up":
            # Pick highest probability finding
            findings = study_context.get("findings", [])
            if findings:
                finding = max(findings, key=lambda f: f["probability"])
            else:
                finding = {"name": "Unknown", "probability": 0.5}
            
            return {
                **base_args,
                "study": study_context,
                "finding": finding,
                "modality": "Chest X-ray",
            }
        elif tool_name == "flag_critical_finding":
            findings = study_context.get("findings", [])
            if findings:
                finding = max(findings, key=lambda f: f["probability"])
            else:
                finding = {"name": "Unknown", "probability": 0.5}
            
            return {
                **base_args,
                "study": study_context,
                "finding": finding,
                "recipient_role": "Attending Radiologist",
            }
        
        return base_args
    
    async def _replan_step(
        self,
        step: WorkflowStep,
        failed_result: ToolResult,
        study_context: dict,
    ) -> bool:
        """Attempt to replan a failed step.
        
        Replan triggers:
        - Confidence < floor: refine RAG query
        - Insufficient passages: broaden search
        
        Args:
            step: Failed step
            failed_result: Result that triggered halt
            study_context: Study context
            
        Returns:
            True if replan successful
        """
        # Check if we have enough evidence
        if len(failed_result.evidence_refs) < 2:
            print(f"[autonomy] Replan trigger: Insufficient evidence ({len(failed_result.evidence_refs)} passages)")
            # In a real implementation, we would refine the RAG query here
            # For demo, we simulate success
            return True
        
        # Check confidence
        from radagent.autonomy.halt import get_confidence_floor
        floor = get_confidence_floor(step.tool_name)
        
        if failed_result.confidence < floor:
            print(f"[autonomy] Replan trigger: Low confidence ({failed_result.confidence:.3f} < {floor:.3f})")
            # Simulate query refinement
            return True
        
        return False
    
    async def replan(
        self,
        failed_step: WorkflowStep,
        reason: str,
        study_context: dict,
    ) -> list[WorkflowStep]:
        """Generate a revised plan after a roadblock.
        
        Args:
            failed_step: Step that failed
            reason: Reason for failure
            study_context: Study context
            
        Returns:
            Revised workflow plan
        """
        # For demo, return a simplified plan
        return [failed_step]

# Made with Bob
