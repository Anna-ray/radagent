"""
RadAgent v2 - Multi-Agent System

The CriticAgent is the ONE real agent in RadAgent v2. It challenges decisions
made by other components when evidence is weak, confidence is borderline, or
reasoning has gaps.

Author: Rayane Aggoune
"""

from radagent.agents.critic import CriticAgent, ChallengeRecord, MockCriticAgent

__all__ = [
    "CriticAgent",
    "ChallengeRecord",
    "MockCriticAgent",
]

# Made with Bob
