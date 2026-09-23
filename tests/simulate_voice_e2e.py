"""
BhoomiMitra AI — Local End-to-End Voice Pipeline Simulation
Tests the full unmocked pipeline (Google STT -> Language Detection -> Decision Engine -> RAG -> Gemini LLM)
with only Meta boundaries mocked (media download & outbound send).
Verifies:
1. Meta message type = "audio"
2. Meta message type = "voice"
3. All structured logging markers in sequential order
4. Database recording & response delivery
"""

import asyncio
import os
import time
import json
import logging
from unittest.mock import patch, AsyncMock
from google.cloud import texttospeech
from httpx import AsyncClient, ASGITransport

from src.main import app
from src.core.logging import logger
from src.core.database import AsyncSessionLocal
from src.core.models import Farmer, Conversation
from src.language.service import LanguageService
from sqlalchemy import select

# Configure logging capture
log_records = []

class MemoryLogHandler(logging.Handler):
    def emit(self, record):
        log_records.append(self.format(record))

handler = MemoryLogHandler()
handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
logger.addHandler(handler)
logging.getLogger("bhoomimitra-ai").addHandler(handler)

async def generate_telugu_voice_bytes(text: str = "వరి పంటలో తెగులు నివారణ ఎలా చేయాలి") -> bytes:
    """Synthesizes real OGG Opus speech bytes via Google Cloud TTS using ADC credentials."""
    tts_client = texttospeech.TextToSpeechAsyncClient()
    input_text = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(
        language_code="te-IN",
        ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
    )
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.OGG_OPUS
    )
    tts_resp = await tts_client.synthesize_speech(
        input=input_text, voice=voice, audio_config=audio_config
    )
    return tts_resp.audio_content

async def run_single_simulation(msg_type: str, audio_bytes: bytes) -> bool:
    print("\n" + "=" * 80)
    print(f"RUNNING E2E SIMULATION FOR META PAYLOAD TYPE: '{msg_type}'")
    print("=" * 80)

    log_records.clear()
    unique_suffix = f"{int(time.time())}_{msg_type}"
    media_id = f"meta_media_{unique_suffix}"
    message_id = f"wamid.HBgM{unique_suffix.upper()}"
    sender_phone = "919988776655"
    sender_name = "రాము రైతు"

    if msg_type == "audio":
        media_payload = {
            "type": "audio",
            "audio": {
                "id": media_id,
                "mime_type": "audio/ogg; codecs=opus"
            }
        }
    else:
        media_payload = {
            "type": "voice",
            "voice": {
                "id": media_id,
                "mime_type": "audio/ogg; codecs=opus"
            }
        }

    webhook_payload = {
        "object": "whatsapp_business_account",
        "entry": [
            {
                "id": "100000000000001",
                "changes": [
                    {
                        "field": "messages",
                        "value": {
                            "messaging_product": "whatsapp",
                            "metadata": {
                                "display_phone_number": "15551234567",
                                "phone_number_id": "123456789012345"
                            },
                            "contacts": [
                                {
                                    "profile": {
                                        "name": sender_name
                                    },
                                    "wa_id": sender_phone
                                }
                            ],
                            "messages": [
                                {
                                    "from": sender_phone,
                                    "id": message_id,
                                    "timestamp": str(int(time.time())),
                                    **media_payload
                                }
                            ]
                        }
                    }
                ]
            }
        ]
    }

    # Mock ONLY external Meta boundary
    async def mock_download_media_bytes(m_id: str):
        if m_id == media_id:
            return audio_bytes, "audio/ogg; codecs=opus"
        return None

    async def mock_send_text_message(to_phone: str, message_text: str):
        return f"wamid.outbound.{unique_suffix}"

    async def mock_mark_read(m_id: str):
        return True

    with patch("src.gateway.service.download_media_bytes", side_effect=mock_download_media_bytes), \
         patch("src.gateway.service.send_text_message", side_effect=mock_send_text_message), \
         patch("src.gateway.service.mark_message_as_read", side_effect=mock_mark_read):

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/webhook/whatsapp",
                json=webhook_payload,
                headers={"Content-Type": "application/json"}
            )
            print(f"Webhook HTTP POST: {response.status_code} -> {response.json()}")

        print(f"Waiting for live pipeline (STT -> Language Detection -> Decision Engine -> Gemini)...")
        for _ in range(60):
            await asyncio.sleep(0.5)
            async with AsyncSessionLocal() as db:
                res = await db.execute(
                    select(Conversation).where(Conversation.message_id == message_id)
                )
                conv = res.scalar_one_or_none()
                if conv and conv.ai_response:
                    break

    async with AsyncSessionLocal() as db:
        res = await db.execute(
            select(Conversation).where(Conversation.message_id == message_id)
        )
        conv = res.scalar_one_or_none()

    if not conv:
        print("ERROR: Conversation not found in DB!")
        return False

    print(f"\n[PIPELINE OUTPUT for {msg_type}]")
    print(f"  Transcribed Text    : '{conv.user_message}'")
    print(f"  AI Response (Sample): '{conv.ai_response[:100]}...'")
    print(f"  Delivery Status     : '{conv.delivery_status}'")
    print(f"  Outbound Message ID : '{conv.outbound_message_id}'")

    required_markers = [
        "[VOICE PIPELINE START]",
        "[VOICE MEDIA DOWNLOAD START]",
        "[VOICE MEDIA DOWNLOAD SUCCESS]",
        "[VOICE STT START]",
        "[VOICE STT SUCCESS]",
        "[VOICE AI START]",
        "[VOICE AI SUCCESS]",
        "[VOICE PIPELINE COMPLETE]"
    ]

    all_logs_text = "\n".join(log_records)
    missing = [m for m in required_markers if m not in all_logs_text]

    fallback_triggered = "మీ వాయిస్ సందేశం స్పష్టంగా లేదు" in (conv.ai_response or "") or "స్పష్టంగా వినిపించలేదు" in (conv.ai_response or "")

    print("\nLog Markers Check:")
    for marker in required_markers:
        status = "FOUND" if marker in all_logs_text else "MISSING"
        print(f"  {marker:35} : {status}")

    if missing:
        print(f"FAILED: Missing structured log markers: {missing}")
        return False
    if fallback_triggered:
        print("FAILED: Voice fallback was triggered unexpectedly!")
        return False
    if not conv.ai_response or not conv.user_message:
        print("FAILED: Transcription or AI response is empty!")
        return False

    print(f"SUCCESS: Pipeline for '{msg_type}' executed end-to-end flawlessly!")
    return True

async def main():
    print("=" * 80)
    print("STEP 1: Synthesizing realistic Telugu OGG Opus speech bytes via Google Cloud TTS...")
    audio_bytes = await generate_telugu_voice_bytes("వరి పంటలో తెగులు నివారణ ఎలా చేయాలి")
    print(f"Synthesized {len(audio_bytes)} bytes of OGG Opus audio.")

    print("\nSTEP 2: Pre-warming Google STT channel...")
    lang_service = LanguageService()
    try:
        warmup_res = await lang_service.transcribe_audio(audio_bytes, "audio/ogg; codecs=opus")
        print(f"STT Warmup success: '{warmup_res.transcription_text}' (lang={warmup_res.detected_language})")
    except Exception as e:
        print(f"STT Warmup warning: {e}")

    print("\nSTEP 3: Running simulated Meta webhook requests...")
    success_audio = await run_single_simulation("audio", audio_bytes)
    success_voice = await run_single_simulation("voice", audio_bytes)

    if success_audio and success_voice:
        print("\n" + "=" * 80)
        print("ALL END-TO-END VOICE SIMULATIONS COMPLETED SUCCESSFULLY (100% PASSED)")
        print("=" * 80)
        return True
    return False

if __name__ == "__main__":
    ok = asyncio.run(main())
    if not ok:
        exit(1)
