from typing import Optional
from pydantic import BaseModel, Field, field_validator

class TranscriptionResponse(BaseModel):
    """
    Structured output returned by the internal Language Service after 
    processing a Speech-to-Text conversion.
    """
    transcription_text: str = Field(..., min_length=1, description="The resulting transcribed text")
    detected_language: Optional[str] = Field(None, max_length=10, description="The detected language code")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Confidence score of the transcription")
    provider_used: str = Field(..., max_length=50, description="The STT provider used (e.g., 'google', 'whisper')")

    @field_validator("transcription_text")
    @classmethod
    def validate_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Transcription text cannot be empty.")
        return v.strip()
