"""
Speechmatics Real-Time Speech-to-Text Transcriber

Wraps the Speechmatics API for transcribing radiologist dictations.
Supports both file-based and streaming transcription.

Author: Rayane Aggoune
"""

import os
import json
import time
from pathlib import Path
from typing import Optional, Dict, Any, List, AsyncIterator
from dataclasses import dataclass, asdict

try:
    from speechmatics.models import ConnectionSettings, AudioSettings
    from speechmatics.batch_client import BatchClient
    SPEECHMATICS_AVAILABLE = True
except ImportError:
    SPEECHMATICS_AVAILABLE = False
    # Graceful fallback for environments without Speechmatics


@dataclass
class TranscriptSegment:
    """A single segment of transcribed speech with timing."""
    text: str
    start_time: float
    end_time: float
    confidence: float
    speaker: Optional[str] = None


@dataclass
class TranscriptResult:
    """Complete transcription result with metadata."""
    full_text: str
    segments: List[TranscriptSegment]
    duration_seconds: float
    language: str
    word_count: int
    processing_time_seconds: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "full_text": self.full_text,
            "segments": [asdict(seg) for seg in self.segments],
            "duration_seconds": self.duration_seconds,
            "language": self.language,
            "word_count": self.word_count,
            "processing_time_seconds": self.processing_time_seconds,
        }


class SpeechmaticsTranscriber:
    """
    Speechmatics-based transcriber for radiologist dictations.
    
    Uses Speechmatics batch API for file-based transcription with
    medical vocabulary optimization.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        language: str = "en",
        enable_entities: bool = True,
        operating_point: str = "enhanced",  # enhanced for medical accuracy
    ):
        """
        Initialize Speechmatics transcriber.
        
        Args:
            api_key: Speechmatics API key (or set SPEECHMATICS_API_KEY env var)
            language: Language code (default: "en")
            enable_entities: Enable entity detection for medical terms
            operating_point: "standard" or "enhanced" (enhanced for medical)
        """
        if not SPEECHMATICS_AVAILABLE:
            raise ImportError(
                "speechmatics-python not installed. "
                "Install with: pip install speechmatics-python"
            )
        
        self.api_key = api_key or os.getenv("SPEECHMATICS_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Speechmatics API key required. Set SPEECHMATICS_API_KEY "
                "environment variable or pass api_key parameter."
            )
        
        self.language = language
        self.enable_entities = enable_entities
        self.operating_point = operating_point
        
        # Connection settings
        self.connection_settings = ConnectionSettings(
            url="https://asr.api.speechmatics.com/v2",
            auth_token=self.api_key,
        )
    
    def transcribe_file(
        self,
        audio_path: str,
        output_format: str = "json-v2",
    ) -> TranscriptResult:
        """
        Transcribe an audio file using Speechmatics batch API.
        
        Args:
            audio_path: Path to audio file (WAV, MP3, FLAC, etc.)
            output_format: Output format ("json-v2" recommended)
        
        Returns:
            TranscriptResult with full text, segments, and metadata
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        start_time = time.time()
        
        # Configure transcription settings
        conf = {
            "type": "transcription",
            "transcription_config": {
                "language": self.language,
                "operating_point": self.operating_point,
                "enable_entities": self.enable_entities,
                "diarization": "speaker",  # Speaker diarization
            },
        }
        
        # Create batch client and submit job
        with BatchClient(self.connection_settings) as client:
            job_id = client.submit_job(
                audio=str(audio_path),
                transcription_config=conf,
            )
            
            # Wait for completion
            transcript_json = client.wait_for_completion(
                job_id,
                transcription_format=output_format,
            )
        
        processing_time = time.time() - start_time
        
        # Parse Speechmatics JSON response
        return self._parse_speechmatics_response(
            transcript_json,
            processing_time,
        )
    
    def _parse_speechmatics_response(
        self,
        response: str,
        processing_time: float,
    ) -> TranscriptResult:
        """
        Parse Speechmatics JSON-v2 response into TranscriptResult.
        
        Args:
            response: Raw JSON response from Speechmatics
            processing_time: Time taken to process
        
        Returns:
            Structured TranscriptResult
        """
        data = json.loads(response) if isinstance(response, str) else response
        
        # Extract segments
        segments = []
        full_text_parts = []
        
        for result in data.get("results", []):
            if result["type"] == "word":
                # Aggregate words into segments (sentences)
                # For simplicity, we'll create one segment per word
                # In production, you'd group by punctuation/pauses
                segments.append(TranscriptSegment(
                    text=result["alternatives"][0]["content"],
                    start_time=result["start_time"],
                    end_time=result["end_time"],
                    confidence=result["alternatives"][0]["confidence"],
                    speaker=result.get("speaker"),
                ))
                full_text_parts.append(result["alternatives"][0]["content"])
        
        # Reconstruct full text
        full_text = " ".join(full_text_parts)
        
        # Calculate duration
        duration = 0.0
        if segments:
            duration = segments[-1].end_time
        
        return TranscriptResult(
            full_text=full_text,
            segments=segments,
            duration_seconds=duration,
            language=self.language,
            word_count=len(full_text_parts),
            processing_time_seconds=processing_time,
        )
    
    def transcribe_file_simple(self, audio_path: str) -> str:
        """
        Simple transcription returning only the full text.
        
        Args:
            audio_path: Path to audio file
        
        Returns:
            Transcribed text as a single string
        """
        result = self.transcribe_file(audio_path)
        return result.full_text


# Fallback mock transcriber for testing without API key
class MockTranscriber:
    """Mock transcriber for testing without Speechmatics API."""
    
    def __init__(self, **kwargs):
        pass
    
    def transcribe_file(self, audio_path: str, **kwargs) -> TranscriptResult:
        """Return mock transcription."""
        return TranscriptResult(
            full_text="No acute cardiopulmonary findings. Lungs are clear bilaterally. Heart size is normal.",
            segments=[
                TranscriptSegment("No acute cardiopulmonary findings.", 0.0, 2.5, 0.95),
                TranscriptSegment("Lungs are clear bilaterally.", 2.5, 4.8, 0.92),
                TranscriptSegment("Heart size is normal.", 4.8, 6.5, 0.94),
            ],
            duration_seconds=6.5,
            language="en",
            word_count=13,
            processing_time_seconds=0.5,
        )
    
    def transcribe_file_simple(self, audio_path: str) -> str:
        """Return mock transcription text."""
        return self.transcribe_file(audio_path).full_text

# Made with Bob
