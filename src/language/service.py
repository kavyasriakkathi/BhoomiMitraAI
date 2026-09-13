import os
import re
import unicodedata
import asyncio
import inspect
import struct
from typing import Optional, List, Union
from google.cloud import speech, texttospeech

from src.config import get_settings
from src.core.logging import logger
from src.core.exceptions import BhoomiMitraException
from src.language.schemas import TranscriptionResponse
from src.language.languages import get_language, LanguageMetadata, DEFAULT_LANGUAGE


def clean_text_for_speech(text: str) -> str:
    """
    Cleans formatted AI agricultural response text into natural, spoken-word friendly text.

    Removes:
    - Markdown headers, bold/italic markers, links, code blocks, bullet points, horizontal lines
    - Web URLs and navigation paths
    - Decorative emojis and extraneous UI symbols

    Preserves:
    - All numbers (integers, decimals, ranges, phone numbers)
    - Currency and prices (₹, Rs., etc.)
    - Units (kg, quintal, litres, acres, °C, mm, etc. in English and Indic scripts)
    - Dates, times, schedules
    - Crop and product names (Urea, DAP, Cotton, Paddy, etc.)
    - Crucial chemical dosages and safety warnings
    """
    if not text:
        return ""

    s = text

    # Remove code blocks and inline code
    s = re.sub(r"```[\s\S]*?```", "", s)
    s = re.sub(r"`([^`]+)`", r"\1", s)

    # Remove URLs (http://, https://) and navigation paths (/shops, /schemes)
    s = re.sub(r"https?://\S+", "", s)
    s = re.sub(r"(?:^|\s)/[a-zA-Z0-9_\-]+(?:\s|$)", " ", s)

    # Convert markdown links [Label](url) -> Label
    s = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", s)

    # Remove markdown headers (#, ##, ###)
    s = re.sub(r"(?m)^#{1,6}\s*", "", s)

    # Remove bold / italic / strike markdown formatting
    s = re.sub(r"\*\*([^*]+)\*\*", r"\1", s)
    s = re.sub(r"\*([^*]+)\*", r"\1", s)
    s = re.sub(r"__([^_]+)__", r"\1", s)
    s = re.sub(r"_([^_]+)_", r"\1", s)
    s = re.sub(r"~~([^~]+)~~", r"\1", s)

    # Remove horizontal rules
    s = re.sub(r"(?m)^[-*_]{3,}\s*$", "", s)

    # Remove bullet markers / list prefixes at line start
    s = re.sub(r"(?m)^\s*[-*+•]\s+", "", s)
    s = re.sub(r"(?m)^\s*\d+[\.\)]\s+", "", s)

    # Remove decorative emojis and symbol characters while preserving Indic scripts, letters,
    # numbers, marks (vowel signs), punctuation, math, and currency.
    cleaned_chars = []
    for char in s:
        code = ord(char)
        # Skip emoji Unicode code blocks
        if (
            0x1F300 <= code <= 0x1F9FF or  # Misc symbols, pictographs, emoticons
            0x1FA00 <= code <= 0x1FAFF or  # Chess, symbols extended
            0x2600 <= code <= 0x27BF or    # Misc symbols, dingbats (🌾, ⚠️, 🌧️, etc.)
            0xFE00 <= code <= 0xFE0F or    # Variation selectors
            0x1F000 <= code <= 0x1F02F or
            0x1F0A0 <= code <= 0x1F0FF
        ):
            continue

        cat = unicodedata.category(char)
        # Letters (L*), Marks (M*), Numbers (N*), Punctuation (P*), Separator (Z*), Currency (Sc)
        if cat.startswith(('L', 'M', 'N', 'P', 'Z', 'Sc')) or char in {'\n', '\t', '°', '+', '=', '<', '>', '/', '-', '%'}:
            cleaned_chars.append(char)
        elif cat == 'Sm' and char in {'+', '=', '<', '>', '/'}:
            cleaned_chars.append(char)

    s = "".join(cleaned_chars)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n\s*\n+", "\n\n", s)
    return s.strip()


def chunk_text_for_speech(text: str, max_chars: int = 3500, max_bytes: int = 4800) -> List[str]:
    """
    Splits text into sentence-aware chunks ensuring no critical agricultural content is silently dropped.
    Every chunk strictly satisfies:
    1. len(chunk) <= max_chars
    2. len(chunk.encode('utf-8')) <= max_bytes
    """
    if not text or not text.strip():
        return []

    cleaned = text.strip()

    # Quick check if whole text fits
    if len(cleaned) <= max_chars and len(cleaned.encode("utf-8")) <= max_bytes:
        return [cleaned]

    # Split into sentences on sentence terminators followed by space/newline,
    # preserving decimals like 2.5 or Rs. 500.00
    raw_sentences = re.split(r'(?<=[!?।;\n])\s+|(?<=[.])(?!\d)\s+', cleaned)
    raw_sentences = [s.strip() for s in raw_sentences if s.strip()]

    if not raw_sentences:
        raw_sentences = [cleaned]

    chunks: List[str] = []
    current_sentences: List[str] = []
    current_char_len = 0
    current_byte_len = 0

    def get_joined(sentences_list: List[str]) -> str:
        return " ".join(sentences_list)

    for sentence in raw_sentences:
        sent_char_len = len(sentence)
        sent_bytes = sentence.encode("utf-8")
        sent_byte_len = len(sent_bytes)

        # Check if single sentence exceeds limits by itself
        if sent_char_len > max_chars or sent_byte_len > max_bytes:
            # Flush existing accumulated chunk first
            if current_sentences:
                chunks.append(get_joined(current_sentences))
                current_sentences = []
                current_char_len = 0
                current_byte_len = 0

            # Sub-chunk oversized sentence by clauses / whitespace
            words = sentence.split(" ")
            sub_words: List[str] = []
            sub_char_len = 0
            sub_byte_len = 0
            for w in words:
                w_char = len(w)
                w_bytes = len(w.encode("utf-8"))
                add_char = w_char + (1 if sub_words else 0)
                add_bytes = w_bytes + (1 if sub_words else 0)
                if sub_char_len + add_char <= max_chars and sub_byte_len + add_bytes <= max_bytes:
                    sub_words.append(w)
                    sub_char_len += add_char
                    sub_byte_len += add_bytes
                else:
                    if sub_words:
                        chunks.append(" ".join(sub_words))
                    sub_words = [w]
                    sub_char_len = w_char
                    sub_byte_len = w_bytes
            if sub_words:
                chunks.append(" ".join(sub_words))
            continue

        # Normal sentence accumulation
        add_char = sent_char_len + (1 if current_sentences else 0)
        add_bytes = sent_byte_len + (1 if current_sentences else 0)

        if current_char_len + add_char <= max_chars and current_byte_len + add_bytes <= max_bytes:
            current_sentences.append(sentence)
            current_char_len += add_char
            current_byte_len += add_bytes
        else:
            if current_sentences:
                chunks.append(get_joined(current_sentences))
            current_sentences = [sentence]
            current_char_len = sent_char_len
            current_byte_len = sent_byte_len

    if current_sentences:
        chunks.append(get_joined(current_sentences))

    return [c.strip() for c in chunks if c.strip()]


def extract_ogg_opus_sample_rate(audio_bytes: bytes, default_rate: int = 16000) -> int:
    """
    Extracts the input sample rate from the RFC 7845 Ogg Opus header (OpusHead).
    Falls back to default_rate (16000 Hz for WhatsApp voice notes) if header is missing
    or if the rate is not one of Google STT's supported rates: 8000, 12000, 16000, 24000, 48000.
    """
    supported_rates = {8000, 12000, 16000, 24000, 48000}
    try:
        idx = audio_bytes.find(b"OpusHead")
        if idx != -1 and len(audio_bytes) >= idx + 16:
            rate = struct.unpack("<I", audio_bytes[idx + 12 : idx + 16])[0]
            if rate in supported_rates:
                return rate
    except Exception:
        pass
    return default_rate


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
            clean_mime = (mime_type or "").lower().strip()
            sample_rate_hertz = None

            if "ogg" in clean_mime or "opus" in clean_mime or audio_bytes.startswith(b"OggS"):
                encoding = speech.RecognitionConfig.AudioEncoding.OGG_OPUS
                sample_rate_hertz = extract_ogg_opus_sample_rate(audio_bytes, default_rate=16000)
            elif "mp3" in clean_mime or "mpeg" in clean_mime:
                encoding = speech.RecognitionConfig.AudioEncoding.MP3
            elif "amr-wb" in clean_mime:
                encoding = speech.RecognitionConfig.AudioEncoding.AMR_WB
                sample_rate_hertz = 16000
            elif "amr" in clean_mime:
                encoding = speech.RecognitionConfig.AudioEncoding.AMR
                sample_rate_hertz = 8000
            elif "wav" in clean_mime or "wave" in clean_mime or "linear16" in clean_mime:
                encoding = speech.RecognitionConfig.AudioEncoding.LINEAR16
                sample_rate_hertz = 16000
            elif "flac" in clean_mime:
                encoding = speech.RecognitionConfig.AudioEncoding.FLAC
            else:
                encoding = speech.RecognitionConfig.AudioEncoding.ENCODING_UNSPECIFIED

            # Support multi-lingual alternatives for Indian languages.
            # CRITICAL: Google Cloud Speech-to-Text v1 strictly allows at most 3 alternative language codes.
            default_lang = self.settings.stt_default_language or "te-IN"
            candidate_alts = ["te-IN", "hi-IN", "en-IN", "ta-IN"]
            filtered_alts = [c for c in candidate_alts if c != default_lang][:3]

            config_kwargs = {
                "encoding": encoding,
                "language_code": default_lang,
                "alternative_language_codes": filtered_alts,
            }
            if sample_rate_hertz is not None:
                config_kwargs["sample_rate_hertz"] = sample_rate_hertz

            config = speech.RecognitionConfig(**config_kwargs)

            call_res = self.google_client.recognize(config=config, audio=audio)
            if inspect.isawaitable(call_res):
                stt_timeout = float(getattr(self.settings, "stt_api_timeout_seconds", 10.0))
                response = await asyncio.wait_for(call_res, timeout=stt_timeout)
            else:
                response = call_res

            if not response.results:
                logger.warning("Google STT returned an empty response.")
                raise BhoomiMitraException("No transcription results returned from Google STT.", status_code=422)

            best_result = response.results[0]
            best_alternative = best_result.alternatives[0]

            transcript = best_alternative.transcript
            confidence = best_alternative.confidence

            detected_language = best_result.language_code if hasattr(best_result, "language_code") and best_result.language_code else default_lang

            if not transcript or not transcript.strip():
                raise BhoomiMitraException("Transcription resulted in empty text.", status_code=422)

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
        language_code: Optional[str] = "te-IN",
        voice_name: Optional[str] = None,
        ssml_gender: Optional[str] = None,
        return_chunks: Optional[bool] = None,
    ) -> Union[List[bytes], Optional[bytes]]:
        """
        Synthesizes speech audio from text.

        If return_chunks is True (or omitted for ISO 2-letter language codes like 'te', 'hi'):
            Returns a list of complete, standalone OGG_OPUS audio chunk payloads (in sentence order).
            Returns [] if unsupported, empty, or on failure.

        If return_chunks is False (or omitted for locale codes like 'te-IN' or when voice_name/ssml_gender provided):
            Returns single OGG_OPUS encoded audio bytes or None on failure/empty text.
        """
        if return_chunks is None:
            if voice_name is not None or ssml_gender is not None or (isinstance(language_code, str) and "-" in language_code):
                return_chunks = False
            else:
                return_chunks = True

        if not return_chunks:
            if not text or not str(text).strip():
                logger.warning("[TTS] Received empty text for speech synthesis.")
                return None

            clean_text = str(text).strip()
            logger.info(f"[TTS] Synthesizing speech (Length: {len(clean_text)} chars, Lang: {language_code})")

            try:
                synthesis_input = texttospeech.SynthesisInput(text=clean_text)

                voice_params = {"language_code": language_code or "te-IN"}
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

        # Multilingual sentence-chunked voice flow (returns List[bytes])
        if not text or not str(text).strip():
            return []

        lang_meta = get_language(language_code)
        if not lang_meta or not lang_meta.supported_tts or not lang_meta.tts_code:
            logger.info(
                f"TTS skipped: Language '{language_code}' does not have native TTS voice support in catalog."
            )
            return []

        cleaned_text = clean_text_for_speech(text)
        if not cleaned_text:
            return []

        provider = self.settings.tts_provider.lower()
        if provider != "google":
            logger.warning(f"Unsupported TTS provider configured: {provider}")
            return []

        try:
            return await self._synthesize_with_google(cleaned_text, lang_meta)
        except Exception as e:
            logger.exception(f"TTS synthesis failed safely for language '{lang_meta.code}': {e}")
            return []

    async def _synthesize_with_google(self, text: str, lang_meta: LanguageMetadata) -> List[bytes]:
        """
        Calls Google Cloud Text-to-Speech API using sentence-aware chunking and OGG_OPUS encoding.
        Returns a list of complete, valid OGG_OPUS byte payloads for each chunk in sentence order.
        """
        max_chars = int(getattr(self.settings, "tts_max_text_chars", 3500))
        chunks = chunk_text_for_speech(text, max_chars=max_chars)
        if not chunks:
            return []

        tts_timeout = float(getattr(self.settings, "tts_api_timeout_seconds", 10.0))
        audio_chunks: List[bytes] = []

        voice_params = texttospeech.VoiceSelectionParams(
            language_code=lang_meta.tts_code,
            name=lang_meta.tts_voice_name if lang_meta.tts_voice_name else None,
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.OGG_OPUS
        )

        for chunk in chunks:
            chunk_bytes_len = len(chunk.encode("utf-8"))
            logger.debug(
                f"Synthesizing TTS chunk ({len(chunk)} chars, {chunk_bytes_len} UTF-8 bytes) "
                f"for language '{lang_meta.code}' voice '{lang_meta.tts_voice_name}'"
            )

            synthesis_input = texttospeech.SynthesisInput(text=chunk)

            try:
                call_res = self.google_tts_client.synthesize_speech(
                    request=texttospeech.SynthesizeSpeechRequest(
                        input=synthesis_input,
                        voice=voice_params,
                        audio_config=audio_config,
                    )
                )
                if inspect.isawaitable(call_res):
                    response = await asyncio.wait_for(call_res, timeout=tts_timeout)
                else:
                    response = call_res

                if response and response.audio_content:
                    audio_chunks.append(response.audio_content)
                else:
                    logger.warning(f"Google TTS returned empty audio content for chunk in {lang_meta.code}")
            except (asyncio.TimeoutError, TimeoutError) as te:
                logger.warning(f"Google TTS API timed out after {tts_timeout}s for {lang_meta.code}: {te}")
                return []
            except Exception as chunk_err:
                # If named voice failed, attempt fallback with generic locale
                if lang_meta.tts_voice_name and "voice" in str(chunk_err).lower():
                    logger.warning(f"Named voice {lang_meta.tts_voice_name} failed. Retrying with generic locale {lang_meta.tts_code}")
                    fallback_voice = texttospeech.VoiceSelectionParams(language_code=lang_meta.tts_code)
                    fallback_call = self.google_tts_client.synthesize_speech(
                        request=texttospeech.SynthesizeSpeechRequest(
                            input=synthesis_input,
                            voice=fallback_voice,
                            audio_config=audio_config,
                        )
                    )
                    if inspect.isawaitable(fallback_call):
                        response = await asyncio.wait_for(fallback_call, timeout=tts_timeout)
                    else:
                        response = fallback_call
                    if response and response.audio_content:
                        audio_chunks.append(response.audio_content)
                        continue
                raise chunk_err

        return audio_chunks


async def synthesize_speech(
    text: str,
    language_code: str = "te-IN",
    voice_name: Optional[str] = None,
    ssml_gender: Optional[str] = None,
) -> Optional[bytes]:
    """Convenience module function for speech synthesis."""
    if not text or not str(text).strip():
        return None
    service = LanguageService()
    return await service.synthesize_speech(
        text=text,
        language_code=language_code,
        voice_name=voice_name,
        ssml_gender=ssml_gender,
        return_chunks=False,
    )
