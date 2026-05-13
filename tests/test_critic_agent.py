"""
Tests for RadAgent v2 CriticAgent

Author: Rayane Aggoune
"""

import pytest
from radagent.agents.critic import MockCriticAgent, ChallengeRecord, Verdict


class TestMockCriticAgent:
    """Test mock CriticAgent (no API required)."""
    
    def test_critic_approves_high_confidence_with_strong_evidence(self):
        """Test that critic approves decisions with high confidence and strong evidence."""
        critic = MockCriticAgent()
        
        decision_record = {
            "action": "triage_study",
            "confidence": 0.85,
            "result": {"urgency": "URGENT"},
            "audit_id": "test_decision_001",
        }
        
        evidence = {
            "passages": [
                {"text": "Evidence 1", "similarity": 0.9, "source": "StatPearls"},
                {"text": "Evidence 2", "similarity": 0.85, "source": "Wikipedia"},
                {"text": "Evidence 3", "similarity": 0.8, "source": "UpToDate"},
            ]
        }
        
        challenge = critic.review(decision_record, evidence, action_floor=0.70)
        
        assert challenge.verdict == "APPROVE"
        assert challenge.confidence > 0.8
        assert not challenge.requested_replan
        assert challenge.replan_action is None
        assert len(challenge.cited_concerns) == 0
    
    def test_critic_challenges_borderline_confidence(self):
        """Test that critic challenges decisions with borderline confidence."""
        critic = MockCriticAgent()
        
        decision_record = {
            "action": "schedule_follow_up",
            "confidence": 0.60,  # Below floor of 0.75
            "result": {"follow_up": "3 months"},
            "audit_id": "test_decision_002",
        }
        
        evidence = {
            "passages": [
                {"text": "Evidence 1", "similarity": 0.75, "source": "StatPearls"},
                {"text": "Evidence 2", "similarity": 0.70, "source": "Wikipedia"},
            ]
        }
        
        challenge = critic.review(decision_record, evidence, action_floor=0.75)
        
        assert challenge.verdict == "CHALLENGE"
        assert challenge.requested_replan
        assert challenge.replan_action is not None
        assert len(challenge.cited_concerns) > 0
        assert "below" in challenge.reasoning.lower() or "floor" in challenge.reasoning.lower()
    
    def test_critic_escalates_critically_low_confidence(self):
        """Test that critic escalates when confidence is critically low."""
        critic = MockCriticAgent()
        
        decision_record = {
            "action": "flag_critical_finding",
            "confidence": 0.45,  # Below 0.5
            "result": {"alert": "STAT"},
            "audit_id": "test_decision_003",
        }
        
        evidence = {
            "passages": [
                {"text": "Weak evidence", "similarity": 0.55, "source": "Unknown"},
            ]
        }
        
        challenge = critic.review(decision_record, evidence, action_floor=0.85)
        
        assert challenge.verdict == "ESCALATE"
        assert not challenge.requested_replan  # Escalation, not replan
        assert len(challenge.cited_concerns) > 0
        assert "0.5" in challenge.reasoning or "critically" in challenge.reasoning.lower()
    
    def test_critic_challenges_insufficient_evidence(self):
        """Test that critic challenges when evidence passages are insufficient."""
        critic = MockCriticAgent()
        
        decision_record = {
            "action": "route_to_subspecialist",
            "confidence": 0.75,  # Good confidence
            "result": {"subspecialist": "THORACIC"},
            "audit_id": "test_decision_004",
        }
        
        evidence = {
            "passages": [
                {"text": "Only one passage", "similarity": 0.85, "source": "StatPearls"},
            ]  # Only 1 passage, need 2+
        }
        
        challenge = critic.review(decision_record, evidence, action_floor=0.65)
        
        assert challenge.verdict == "CHALLENGE"
        assert challenge.requested_replan
        assert challenge.replan_action == "refine_rag_query"
        assert any("evidence" in concern.lower() or "passage" in concern.lower() 
                   for concern in challenge.cited_concerns)
    
    def test_critic_requests_replan_with_specific_action(self):
        """Test that critic requests specific replan actions."""
        critic = MockCriticAgent()
        
        # Case 1: Insufficient evidence → refine_rag_query
        decision_record = {
            "action": "triage_study",
            "confidence": 0.68,
            "result": {"urgency": "URGENT"},
            "audit_id": "test_decision_005",
        }
        
        evidence = {
            "passages": [
                {"text": "Single passage", "similarity": 0.8, "source": "StatPearls"},
            ]
        }
        
        challenge = critic.review(decision_record, evidence, action_floor=0.70)
        
        assert challenge.verdict == "CHALLENGE"
        assert challenge.requested_replan
        assert challenge.replan_action == "refine_rag_query"
        
        # Case 2: Low confidence with sufficient evidence → lower_confidence_floor
        decision_record["confidence"] = 0.62
        evidence["passages"].append(
            {"text": "Second passage", "similarity": 0.75, "source": "Wikipedia"}
        )
        
        challenge = critic.review(decision_record, evidence, action_floor=0.70)
        
        assert challenge.verdict == "CHALLENGE"
        assert challenge.requested_replan
        assert challenge.replan_action == "lower_confidence_floor"
    
    def test_critic_audit_chain_links_to_previous_record(self):
        """Test that critic audit chain links to the decision being reviewed."""
        critic = MockCriticAgent()
        
        decision_audit_id = "original_decision_abc123"
        
        decision_record = {
            "action": "triage_study",
            "confidence": 0.80,
            "result": {"urgency": "ROUTINE"},
            "audit_id": decision_audit_id,
        }
        
        evidence = {
            "passages": [
                {"text": "Evidence 1", "similarity": 0.85, "source": "StatPearls"},
                {"text": "Evidence 2", "similarity": 0.80, "source": "Wikipedia"},
            ]
        }
        
        challenge = critic.review(decision_record, evidence, action_floor=0.70)
        
        assert challenge.previous_audit_id == decision_audit_id
        assert challenge.audit_id != decision_audit_id
        assert len(challenge.audit_id) == 16  # SHA-256 truncated to 16 chars
        assert challenge.timestamp is not None
    
    def test_critic_works_in_mock_mode_without_api_key(self):
        """Test that mock critic works without any API configuration."""
        # This should not raise any errors
        critic = MockCriticAgent()
        
        decision_record = {
            "action": "test_action",
            "confidence": 0.75,
            "result": {},
            "audit_id": "test",
        }
        
        evidence = {
            "passages": [
                {"text": "Test", "similarity": 0.8, "source": "Test"},
                {"text": "Test2", "similarity": 0.75, "source": "Test"},
            ]
        }
        
        challenge = critic.review(decision_record, evidence)
        
        assert isinstance(challenge, ChallengeRecord)
        assert challenge.verdict in ["APPROVE", "CHALLENGE", "ESCALATE"]
        assert isinstance(challenge.confidence, float)
        assert 0.0 <= challenge.confidence <= 1.0


class TestChallengeRecord:
    """Test ChallengeRecord dataclass."""
    
    def test_challenge_record_creation(self):
        """Test creating a challenge record."""
        record = ChallengeRecord(
            verdict="CHALLENGE",
            reasoning="Test reasoning",
            cited_concerns=["Concern 1", "Concern 2"],
            requested_replan=True,
            replan_action="refine_rag_query",
            confidence=0.75,
            audit_id="abc123",
            previous_audit_id="def456",
            timestamp="2026-05-13T12:00:00Z",
        )
        
        assert record.verdict == "CHALLENGE"
        assert record.reasoning == "Test reasoning"
        assert len(record.cited_concerns) == 2
        assert record.requested_replan
        assert record.replan_action == "refine_rag_query"
        assert record.confidence == 0.75
    
    def test_challenge_record_to_dict(self):
        """Test converting challenge record to dictionary."""
        record = ChallengeRecord(
            verdict="APPROVE",
            reasoning="All good",
            cited_concerns=[],
            requested_replan=False,
            replan_action=None,
            confidence=0.90,
            audit_id="abc123",
            previous_audit_id="def456",
            timestamp="2026-05-13T12:00:00Z",
        )
        
        record_dict = record.to_dict()
        
        assert isinstance(record_dict, dict)
        assert record_dict["verdict"] == "APPROVE"
        assert record_dict["reasoning"] == "All good"
        assert record_dict["confidence"] == 0.90
        assert record_dict["audit_id"] == "abc123"


class TestIntegration:
    """Integration tests for CriticAgent workflow."""
    
    def test_full_review_workflow(self):
        """Test complete review workflow."""
        critic = MockCriticAgent()
        
        # Simulate a decision from autonomy planner
        decision_record = {
            "action": "triage_study",
            "confidence": 0.78,
            "result": {
                "urgency": "URGENT",
                "reasoning": "Multiple critical findings",
            },
            "audit_id": "autonomy_decision_001",
        }
        
        # Simulate evidence from RAG
        evidence = {
            "passages": [
                {
                    "text": "Pleural effusion requires urgent evaluation...",
                    "similarity": 0.92,
                    "source": "StatPearls: Pleural Effusion",
                },
                {
                    "text": "Infiltration patterns suggest acute process...",
                    "similarity": 0.88,
                    "source": "Wikipedia: Pulmonary Infiltrate",
                },
            ],
            "specialist_outputs": {
                "Effusion": 0.93,
                "Infiltration": 0.82,
            }
        }
        
        # Critic reviews
        challenge = critic.review(decision_record, evidence, action_floor=0.70)
        
        # Should approve (high confidence + strong evidence)
        assert challenge.verdict == "APPROVE"
        assert challenge.previous_audit_id == "autonomy_decision_001"
        
        # Verify audit chain
        assert challenge.audit_id != challenge.previous_audit_id
        assert len(challenge.audit_id) == 16
    
    def test_replan_trigger_workflow(self):
        """Test workflow when replan is triggered."""
        critic = MockCriticAgent()
        
        # Decision with borderline confidence
        decision_record = {
            "action": "schedule_follow_up",
            "confidence": 0.55,  # Below floor
            "result": {"follow_up": "6 months"},
            "audit_id": "autonomy_decision_002",
        }
        
        # Weak evidence
        evidence = {
            "passages": [
                {
                    "text": "Follow-up may be needed...",
                    "similarity": 0.65,
                    "source": "Generic guideline",
                },
            ]
        }
        
        # Critic reviews
        challenge = critic.review(decision_record, evidence, action_floor=0.75)
        
        # Should challenge and request replan
        assert challenge.verdict == "CHALLENGE"
        assert challenge.requested_replan
        assert challenge.replan_action in ["refine_rag_query", "lower_confidence_floor"]
        assert len(challenge.cited_concerns) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

# Made with Bob
