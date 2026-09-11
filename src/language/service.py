import os
import asyncio
import inspect
from typing import Optional
from google.cloud import speech, texttospeech

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
        self._google_tts_client: Optional[texttospeech.TextToSpeechAsyncClient] = None

    @property
    def google_client(self) -> speech.SpeechAsyncClient:
        if not self._google_client:
            self._google_client = speech.SpeechAsyncClient()
        return self._google_client

    @property
    def google_tts_client(self) -> texttospeech.TextToSpeechAsyncClient:
        if not self._google_tts_client:
            self._google_tts_client = texttospeech.TextToSpeechAsyncClient()
        return self._google_tts_client

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
            # For OGG_OPUS, sample_rate_hertz is not specified so Google STT reads it natively from the container header.
            encoding = speech.RecognitionConfig.AudioEncoding.OGG_OPUS

            # Support multi-lingual alternatives for Indian languages
            alt_codes = [
                "te-IN", "hi-IN", "en-IN", "ta-IN", "kn-IN", "ml-IN",
                "mr-IN", "bn-IN", "gu-IN", "pa-IN", "ur-IN", "or-IN", "as-IN"
            ]
            default_lang = self.settings.stt_default_language
            filtered_alts = [c for c in alt_codes if c != default_lang]

            config = speech.RecognitionConfig(
                encoding=encoding,
                language_code=default_lang,
                alternative_language_codes=filtered_alts,
            )

            call_res = self.google_client.recognize(config=config, audio=audio)
            if inspect.isawaitable(call_res):
                stt_timeout = float(getattr(self.settings, "stt_api_timeout_seconds", 10.0))
                response = await asyncio.wait_for(call_res, timeout=stt_timeout)
            else:
                response = call_res
            
            if not response.results:
                logger.warning("Google STT returned an empty response.")
                raise BhoomiMitraException("No transcription results returned from Google STT.", status_code=422)
                
            # Google STT orders results chronologically. The first alternative in the first result is the highest confidence.
            best_result = response.results[0]
            best_alternative = best_result.alternatives[0]
            
            transcript = best_alternative.transcript
            confidence = best_alternative.confidence
            
            # Attempt to extract the language code that was actually detected, fallback to default.
            detected_language = best_result.language_code if hasattr(best_result, "language_code") and best_result.language_code else default_lang
            
            if not transcript or not transcript.strip():
                raise BhoomiMitraException("Transcription resulted in empty text.", status_code=422)

            # Refine detected language using our deterministic script & text analyzer
            from src.language.detector import detect_language
            refined_lang = detect_language(transcript.strip(), fallback=detected_language[:2] if detected_language else "te")

            logger.info(f"Google STT Success: Detected STT code '{detected_language}', Refined language '{refined_lang}' with confidence {confidence}")

            return TranscriptionResponse(
                transcription_text=transcript.strip(),
                detected_language=refined_lang,
                confidence=confidence,
                provider_used="google"
            )
            
        except BhoomiMitraException:
            raise
        except (asyncio.TimeoutError, TimeoutError) as te:
            logger.warning(f"Google STT API timed out after {getattr(self.settings, 'stt_api_timeout_seconds', 10.0)}s: {te}")
            raise BhoomiMitraException("Google STT API timed out.", status_code=504) from te
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

    async def synthesize_speech(
        self,
        text: str,
        language_code: str = "te-IN",
        voice_name: Optional[str] = None,
        ssml_gender: Optional[str] = None,
    ) -> Optional[bytes]:
        """
        Synthesizes text into speech audio bytes using Google Cloud Text-to-Speech.
        Returns OGG_OPUS encoded audio bytes (native format for WhatsApp voice notes).
        Fails soft by returning None on error so caller can proceed.
        """
        if not text or not str(text).strip():
            logger.warning("[TTS] Received empty text for speech synthesis.")
            return None

        clean_text = str(text).strip()
        logger.info(f"[TTS] Synthesizing speech (Length: {len(clean_text)} chars, Lang: {language_code})")

        try:
            synthesis_input = texttospeech.SynthesisInput(text=clean_text)

            voice_params = {"language_code": language_code}
            if voice_name:
                voice_params["name"] = voice_name
            if ssml_gender:
                gender_enum = getattr(texttospeech.SsmlVoiceGender, ssml_gender.upper(), None)
                if gender_enum:
                    voice_params["ssml_gender"] = gender_enum

            voice = texttospeech.VoiceSelectionParams(**voice_params)

            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.OGG_OPUS
            )

            tts_timeout = float(getattr(self.settings, "tts_api_timeout_seconds", 10.0))
            call_res = self.google_tts_client.synthesize_speech(
                input=synthesis_input,
                voice=voice,
                audio_config=audio_config,
            )
            if inspect.isawaitable(call_res):
                response = await asyncio.wait_for(call_res, timeout=tts_timeout)
            else:
                response = call_res

            if response and response.audio_content:
                logger.info(f"[TTS] Successfully synthesized speech ({len(response.audio_content)} bytes)")
                return response.audio_content

            logger.warning("[TTS] Empty audio content returned from Google TTS.")
            return None

        except (asyncio.TimeoutError, TimeoutError) as te:
            logger.warning(f"[TTS] Google TTS API timed out: {te}")
            return None
        except Exception as e:
            logger.exception(f"[TTS] Google TTS synthesis failed: {e}")
            return None


async def synthesize_speech(
    text: str,
    language_code: str = "te-IN",
    voice_name: Optional[str] = None,
    ssml_gender: Optional[str] = None,
) -> Optional[bytes]:
    """Convenience module function for speech synthesis."""
    service = LanguageService()
    return await service.synthesize_speech(
        text=text,
        language_code=language_code,
        voice_name=voice_name,
        ssml_gender=ssml_gender,
    )
