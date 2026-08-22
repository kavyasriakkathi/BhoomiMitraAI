"""
Voice / Speech-to-Text Data Models for BhoomiMitra AI.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class VoiceTranscriptionResult:
    """Outcome of audio speech-to-text processing."""
    text: str
    detected_language: Optional[str] = None
    confidence: Optional[float] = None
    is_success: bool = True
    error_message: Optional[str] = None
    provider_used: str = "google"
