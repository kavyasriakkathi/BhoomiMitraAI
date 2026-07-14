import os
from typing import Optional
from google.cloud import speech

from src.config import get_settings
from src.core.logging import logger
from src.core.exceptions import BhoomiMitraException
from src.language.schemas import TranscriptionResponse


class LanguageService:
    def __init__(self):
        self.settings = get_settings()
        
        if self.settings.google_application_credentials:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = self.settings.google_application_credentials
            
        self._google_client: Optional[speech.SpeechAsyncClient] = None

    @property
    def google_client(self) -> speech.SpeechAsyncClient:
        if not self._google_client:
            self._google_client = speech.SpeechAsyncClient()
        return self._google_client

    async def transcribe_audio(self, audio_bytes: bytes, mime_type: str) -> TranscriptionResponse:
        """
        Converts raw speech audio into text using the configured STT provider.
        """
        logger.info(f"Transcribing audio (size: {len(audio_bytes)} bytes, type: {mime_type})")

        if not audio_bytes:
            logger.error("Received empty audio payload.")
            raise BhoomiMitraException("Audio payload is empty.", status_code=400)

        provider = self.settings.stt_provider.lower()

        try:
            if provider == "google":
                return await self._transcribe_with_google(audio_bytes, mime_type)
            elif provider == "whisper":
                return await self._transcribe_with_whisper(audio_bytes, mime_type)
            else:
                logger.error(f"Unsupported STT provider configured: {provider}")
                raise BhoomiMitraException(f"Unsupported STT provider: {provider}", status_code=501)
                
        except BhoomiMitraException:
            raise
        except Exception as e:
            logger.exception("Unexpected error occurred during audio transcription.")
            raise BhoomiMitraException("An error occurred while transcribing audio.", status_code=500) from e

    async def _transcribe_with_google(self, audio_bytes: bytes, mime_type: str) -> TranscriptionResponse:
        """
        Google Cloud Speech-to-Text integration.
        """
        logger.debug(f"Calling Google STT API (Language: {self.settings.stt_default_language})")
        
        try:
            audio = speech.RecognitionAudio(content=audio_bytes)
            
            # WhatsApp predominantly uses OGG/OPUS for voice notes
            # Explicitly declaring the encoding ensures Google STT parses it accurately.
            encoding = speech.RecognitionConfig.AudioEncoding.OGG_OPUS
            
            config = speech.RecognitionConfig(
                encoding=encoding,
                sample_rate_hertz=16000,
                language_code=self.settings.stt_default_language,
                alternative_language_codes=["te-IN", "hi-IN", "en-IN"],
            )

            response = await self.google_client.recognize(config=config, audio=audio)
            
            if not response.results:
                logger.warning("Google STT returned an empty response.")
                raise BhoomiMitraException("No transcription results returned from Google STT.", status_code=422)
                
            # Google STT orders results chronologically. The first alternative in the first result is the highest confidence.
            best_result = response.results[0]
            best_alternative = best_result.alternatives[0]
            
            transcript = best_alternative.transcript
            confidence = best_alternative.confidence
            
            # Attempt to extract the language code that was actually detected, fallback to default.
            detected_language = best_result.language_code if hasattr(best_result, "language_code") and best_result.language_code else self.settings.stt_default_language
            
            if not transcript or not transcript.strip():
                raise BhoomiMitraException("Transcription resulted in empty text.", status_code=422)

            logger.info(f"Google STT Success: Detected language '{detected_language}' with confidence {confidence}")

            return TranscriptionResponse(
                transcription_text=transcript.strip(),
                detected_language=detected_language,
                confidence=confidence,
                provider_used="google"
            )
            
        except BhoomiMitraException:
            raise
        except Exception as e:
            logger.exception("Google STT API call failed.")
            raise BhoomiMitraException("Failed to transcribe audio with Google STT.", status_code=502) from e

    async def _transcribe_with_whisper(self, audio_bytes: bytes, mime_type: str) -> TranscriptionResponse:
        """
        OpenAI Whisper integration.
        """
        # TODO: Implement real OpenAI Whisper API calls.
        logger.debug(f"Calling OpenAI Whisper API (Language: {self.settings.stt_default_language})")
        
        return TranscriptionResponse(
            transcription_text="This is a mock transcription from OpenAI Whisper.",
            detected_language=self.settings.stt_default_language,
            confidence=0.95,
            provider_used="whisper"
        )
