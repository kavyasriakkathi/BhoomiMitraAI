"""
Voice / Speech-to-Text Service for BhoomiMitra AI.
Provides an extensible abstraction for transcribing farmer voice messages
in Telugu, English, and other regional languages.
"""

from abc import ABC, abstractmethod
from typing import Optional
import json

from src.config import get_settings
from src.core.logging import logger
from src.voice.models import VoiceTranscriptionResult


class BaseSTTProvider(ABC):
    """Abstract interface for Speech-to-Text providers."""

    @abstractmethod
    async def transcribe(
        self,
        audio_bytes: bytes,
        mime_type: str,
        language_code: Optional[str] = None,
    ) -> VoiceTranscriptionResult:
        """Converts raw audio bytes into text."""
        pass


class GoogleSTTProvider(BaseSTTProvider):
    """Google Cloud Speech-to-Text Provider."""

    def __init__(self):
        self.settings = get_settings()
        self._client = None

    def _get_client(self):
        if not self._client:
            from google.cloud import speech
            from src.language.credentials import resolve_google_credentials

            creds = resolve_google_credentials(self.settings)
            if creds:
                self._client = speech.SpeechAsyncClient(credentials=creds)
            else:
                self._client = speech.SpeechAsyncClient()
        return self._client

    async def transcribe(
        self,
        audio_bytes: bytes,
        mime_type: str,
        language_code: Optional[str] = None,
    ) -> VoiceTranscriptionResult:
        from google.cloud import speech

        lang = language_code or self.settings.stt_default_language or "te-IN"
        logger.info(f"[VOICE Google STT] Starting transcription (Lang: {lang}, Size: {len(audio_bytes)}B)")

        try:
            client = self._get_client()
            audio = speech.RecognitionAudio(content=audio_bytes)

            encoding = speech.RecognitionConfig.AudioEncoding.OGG_OPUS
            if "amr" in mime_type.lower():
                encoding = speech.RecognitionConfig.AudioEncoding.AMR
            elif "mp3" in mime_type.lower() or "mpeg" in mime_type.lower():
                encoding = speech.RecognitionConfig.AudioEncoding.MP3

            config = speech.RecognitionConfig(
                encoding=encoding,
                sample_rate_hertz=16000,
                language_code=lang,
                alternative_language_codes=["te-IN", "en-IN", "hi-IN"],
            )

            response = await client.recognize(config=config, audio=audio)

            if not response.results:
                return VoiceTranscriptionResult(
                    text="",
                    detected_language=lang,
                    confidence=0.0,
                    is_success=False,
                    error_message="Empty transcription returned from Google STT",
                    provider_used="google",
                )

            best_result = response.results[0]
            best_alt = best_result.alternatives[0]
            transcript = (best_alt.transcript or "").strip()
            confidence = getattr(best_alt, "confidence", 0.9)
            detected_lang = getattr(best_result, "language_code", lang) or lang

            if not transcript:
                return VoiceTranscriptionResult(
                    text="",
                    detected_language=detected_lang,
                    confidence=0.0,
                    is_success=False,
                    error_message="Empty text transcript",
                    provider_used="google",
                )

            return VoiceTranscriptionResult(
                text=transcript,
                detected_language=detected_lang,
                confidence=confidence,
                is_success=True,
                provider_used="google",
            )

        except Exception as e:
            logger.exception(f"[VOICE Google STT ERROR] Failed to transcribe audio: {e}")
            return VoiceTranscriptionResult(
                text="",
                detected_language=lang,
                confidence=0.0,
                is_success=False,
                error_message=str(e),
                provider_used="google",
            )


class WhisperSTTProvider(BaseSTTProvider):
    """OpenAI Whisper Speech-to-Text Provider."""

    def __init__(self):
        self.settings = get_settings()

    async def transcribe(
        self,
        audio_bytes: bytes,
        mime_type: str,
        language_code: Optional[str] = None,
    ) -> VoiceTranscriptionResult:
        logger.info(f"[VOICE Whisper STT] Starting transcription (Size: {len(audio_bytes)}B)")
        try:
            from openai import AsyncOpenAI
            import io

            if not self.settings.openai_api_key:
                raise ValueError("OPENAI_API_KEY is not configured.")

            client = AsyncOpenAI(api_key=self.settings.openai_api_key)
            audio_file = io.BytesIO(audio_bytes)
            audio_file.name = "audio.ogg" if "ogg" in mime_type else "audio.mp3"

            transcript_resp = await client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language=language_code.split("-")[0] if language_code else None,
            )

            text = (transcript_resp.text or "").strip()
            if not text:
                return VoiceTranscriptionResult(
                    text="",
                    is_success=False,
                    error_message="Empty transcript from Whisper",
                    provider_used="whisper",
                )

            return VoiceTranscriptionResult(
                text=text,
                detected_language=language_code,
                confidence=0.95,
                is_success=True,
                provider_used="whisper",
            )
        except Exception as e:
            logger.exception(f"[VOICE Whisper STT ERROR] Whisper transcription failed: {e}")
            return VoiceTranscriptionResult(
                text="",
                is_success=False,
                error_message=str(e),
                provider_used="whisper",
            )


class VoiceService:
    """
    High-level Voice Service for BhoomiMitra AI.
    Converts audio into text and integrates seamlessly with the existing AI pipeline.
    """

    def __init__(self, provider: Optional[BaseSTTProvider] = None):
        settings = get_settings()
        if provider:
            self.provider = provider
        elif settings.stt_provider.lower() == "whisper":
            self.provider = WhisperSTTProvider()
        else:
            self.provider = GoogleSTTProvider()

    async def transcribe_audio(
        self,
        audio_bytes: bytes,
        mime_type: str,
        language_code: Optional[str] = None,
    ) -> VoiceTranscriptionResult:
        """
        Transcribes raw audio bytes into text.
        Handles empty inputs gracefully without crashing.
        """
        if not audio_bytes or len(audio_bytes) == 0:
            logger.warning("[VOICE SERVICE] Empty audio payload received.")
            return VoiceTranscriptionResult(
                text="",
                is_success=False,
                error_message="Audio payload is empty",
            )

        return await self.provider.transcribe(
            audio_bytes=audio_bytes,
            mime_type=mime_type,
            language_code=language_code,
        )

    def get_stt_failure_message(self, language: Optional[str] = "te") -> str:
        """
        Returns a friendly, localized retry message when STT fails or produces no text.
        """
        lang = (language or "te").lower()
        if "te" in lang or "telugu" in lang:
            return "క్షమించండి, మీ వాయిస్ మెసేజ్ స్పష్టంగా వినబడలేదు. దయచేసి మళ్లీ రికార్డ్ చేసి పంపండి లేదా మెసేజ్ టైప్ చేయండి."
        elif "hi" in lang or "hindi" in lang:
            return "क्षमा करें, आपका वॉयस संदेश स्पष्ट रूप से सुनाई नहीं दिया। कृपया पुनः रिकॉर्ड करके भेजें या संदेश टाइप करें।"
        else:
            return "Sorry, we could not clearly understand your voice message. Please try recording again or type your question."


_voice_service_instance: Optional[VoiceService] = None


def get_voice_service() -> VoiceService:
    """Returns the singleton instance of VoiceService."""
    global _voice_service_instance
    if _voice_service_instance is None:
        _voice_service_instance = VoiceService()
    return _voice_service_instance
