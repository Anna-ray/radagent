"""
RadAgent v2 - Voice Dictation Module

Real-time speech-to-text transcription and dictation auditing for radiologist
voice reports. Integrates Speechmatics for STT and compares dictated findings
against specialist model predictions to surface discrepancies.

Author: Rayane Aggoune
"""

from radagent.voice.transcriber import SpeechmaticsTranscriber
from radagent.voice.dictation_auditor import DictationAuditor, AuditReport, Discrepancy

__all__ = [
    "SpeechmaticsTranscriber",
    "DictationAuditor",
    "AuditReport",
    "Discrepancy",
]

# Made with Bob
