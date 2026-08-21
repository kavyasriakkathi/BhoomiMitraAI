import os
import json
import base64
from pathlib import Path
from typing import Optional
from google.cloud import speech
from google.oauth2 import service_account

from src.config import get_settings
from src.core.logging import logger
from src.core.exceptions import BhoomiMitraException
from src.language.schemas import TranscriptionResponse


def resolve_google_credentials(settings) -> Optional[service_account.Credentials]:
    """
    Safely resolve Google Cloud service account credentials from multiple sources:
    1. Render secret files directory (/etc/secrets/service-account.json or *.json)
    2. Existing file path in GOOGLE_APPLICATION_CREDENTIALS (local Windows / Linux file)
    3. Direct JSON string or base64 in GOOGLE_APPLICATION_CREDENTIALS_JSON
    4. Direct JSON string in GOOGLE_APPLICATION_CREDENTIALS
    5. Local secret files (./secrets/*.json)
    6. Returns None for Application Default Credentials (ADC) fallback.

    Guarantees that raw JSON strings or non-existent placeholder paths do not poison os.environ['GOOGLE_APPLICATION_CREDENTIALS'].
    """
    # 1. Prefer Render Secret Files directory (/etc/secrets)
    render_secrets_dir = Path("/etc/secrets")
    if render_secrets_dir.is_dir():
        known_candidates = [
            render_secrets_dir / "service-account.json",
            render_secrets_dir / "gen-lang-client-0304347321-3ef730155599.json",
            render_secrets_dir / "google-credentials.json",
        ]
        for candidate in known_candidates:
            if candidate.is_file():
                try:
                    creds = service_account.Credentials.from_service_account_file(str(candidate))
                    project_id = getattr(creds, "project_id", "unknown")
                    logger.info(f"[GOOGLE STT AUTH] Loaded service account credentials from Render secret file: {candidate.name} (project: {project_id})")
                    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(candidate)
                    return creds
                except Exception as err:
                    logger.warning(f"[GOOGLE STT AUTH] Failed to load credentials from {candidate}: {err}")

        for json_file in render_secrets_dir.glob("*.json"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    content = json.load(f)
                if isinstance(content, dict) and content.get("type") == "service_account":
                    creds = service_account.Credentials.from_service_account_info(content)
                    project_id = content.get("project_id", "unknown")
                    logger.info(f"[GOOGLE STT AUTH] Discovered valid service account JSON in /etc/secrets: {json_file.name} (project: {project_id})")
                    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(json_file)
                    return creds
            except Exception:
                continue

    # 2. File path or raw JSON in GOOGLE_APPLICATION_CREDENTIALS
    creds_target = getattr(settings, "google_application_credentials", None) or os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if creds_target and creds_target.strip():
        target_str = creds_target.strip()

        # Check if the string itself is raw JSON
        if target_str.startswith("{"):
            # Ensure raw JSON does not stay in os.environ where Google Auth treats it as a file path
            if os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") == target_str:
                del os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
            try:
                creds_dict = json.loads(target_str)
                if isinstance(creds_dict, dict) and "type" in creds_dict:
                    creds = service_account.Credentials.from_service_account_info(creds_dict)
                    project_id = creds_dict.get("project_id", "unknown")
                    logger.info(f"[GOOGLE STT AUTH] Loaded service account credentials from raw JSON in GOOGLE_APPLICATION_CREDENTIALS (project: {project_id}).")
                    return creds
            except Exception as err:
                logger.warning(f"[GOOGLE STT AUTH] Failed to parse JSON from GOOGLE_APPLICATION_CREDENTIALS: {err}")

        # Check if it is a valid existing file on disk (Windows or Linux)
        else:
            target_path = Path(target_str)
            if target_path.is_file():
                try:
                    creds = service_account.Credentials.from_service_account_file(str(target_path))
                    project_id = getattr(creds, "project_id", "unknown")
                    logger.info(f"[GOOGLE STT AUTH] Loaded service account credentials from file: {target_path} (project: {project_id})")
                    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(target_path)
                    return creds
                except Exception as err:
                    logger.warning(f"[GOOGLE STT AUTH] Failed to load credentials from file {target_path}: {err}")
            else:
                logger.warning(
                    f"[GOOGLE STT AUTH] Configured credentials path '{target_str}' does not exist on disk. "
                    "Checking fallback secret directories..."
                )
                # Remove non-existent path from os.environ so it does not poison Google ADC
                if os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") == target_str:
                    del os.environ["GOOGLE_APPLICATION_CREDENTIALS"]

    # 3. Direct JSON in GOOGLE_APPLICATION_CREDENTIALS_JSON
    json_val = getattr(settings, "google_application_credentials_json", None) or os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")
    if json_val and json_val.strip():
        val = json_val.strip()
        if not val.startswith("{") and len(val) > 20:
            try:
                val = base64.b64decode(val).decode("utf-8")
            except Exception:
                pass
        try:
            creds_dict = json.loads(val)
            if isinstance(creds_dict, dict) and "type" in creds_dict:
                creds = service_account.Credentials.from_service_account_info(creds_dict)
                project_id = creds_dict.get("project_id", "unknown")
                logger.info(f"[GOOGLE STT AUTH] Loaded service account credentials from GOOGLE_APPLICATION_CREDENTIALS_JSON (project: {project_id}).")
                return creds
        except Exception as err:
            logger.warning(f"[GOOGLE STT AUTH] Failed to parse GOOGLE_APPLICATION_CREDENTIALS_JSON: {err}")

    # 4. Local secrets directory (./secrets)
    local_secrets_dir = Path("secrets")
    if local_secrets_dir.is_dir():
        for json_file in local_secrets_dir.glob("*.json"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    content = json.load(f)
                if isinstance(content, dict) and content.get("type") == "service_account":
                    creds = service_account.Credentials.from_service_account_info(content)
                    project_id = content.get("project_id", "unknown")
                    logger.info(f"[GOOGLE STT AUTH] Discovered local service account JSON: {json_file} (project: {project_id})")
                    return creds
            except Exception:
                continue

    # Clean up os.environ if it still contains a raw JSON string
    current_env = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
    if current_env.strip().startswith("{") or (current_env and not Path(current_env).is_file()):
        del os.environ["GOOGLE_APPLICATION_CREDENTIALS"]

    logger.info("[GOOGLE STT AUTH] No explicit service account credentials resolved; using Application Default Credentials (ADC).")
    return None


class LanguageService:
    def __init__(self):
        self.settings = get_settings()
        self._google_client: Optional[speech.SpeechAsyncClient] = None

    @property
    def google_client(self) -> speech.SpeechAsyncClient:
        if not self._google_client:
            creds = resolve_google_credentials(self.settings)
            if creds:
                self._google_client = speech.SpeechAsyncClient(credentials=creds)
            else:
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
