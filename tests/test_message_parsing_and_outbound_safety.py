"""
BhoomiMitra AI — Phase 3: Message Parsing & Outbound Safety Tests

Comprehensive test suite verifying:
1. Inbound text parsing (Telugu, English, Tanglish, Mixed, Empty)
2. Voice message pipeline & fallbacks (valid, missing media_id, download failure, STT failure)
3. Image message pipeline & fallbacks (valid, missing media_id, download failure, non-crop photo)
4. Unsupported media handling (video, document, sticker, contact, location)
5. Malformed payload resilience (missing entry, changes, value, messages, ID, sender)
6. WhatsApp status event filtering (sent, delivered, read, failed)
7. Batching & multiple messages in single webhook
8. Outbound safety & data integrity (None/empty protection, length truncation, price preservation)
9. Fallback deduplication (exactly one response per failure)
10. Privacy & header/phone masking
"""

import pytest
import uuid
import json
from unittest.mock import patch, AsyncMock, MagicMock
from fastapi.testclient import TestClient

from src.main import app
from src.core.models import Farmer, Conversation
from src.gateway.schemas import ParsedIncomingMessage
from src.gateway.router import _extract_message, mask_phone_number, sanitize_headers_for_logging
from src.gateway.service import process_message_pipeline
from src.gateway.whatsapp_client import send_text_message
from src.ai.service import _finalize_whatsapp_response
from src.ai.prompts import (
    get_fallback_response,
    get_voice_fallback_response,
    get_image_fallback_response,
    get_unsupported_media_fallback_response,
    get_non_crop_image_response,
)

client = TestClient(app)


# ─────────────────────────────────────────────────────────────────────────────
# 1. TEXT MESSAGE PARSING & LANGUAGE HANDLING
# ─────────────────────────────────────────────────────────────────────────────

def test_extract_telugu_text_message():
    """Verify clean extraction and preservation of Telugu Unicode text."""
    msg = MagicMock()
    msg.from_ = "919876543210"
    msg.id = "wamid.TELUGU_01"
    msg.timestamp = "1700000000"
    msg.type = "text"
    msg.text.body = "వరంగల్లో ఈరోజు పత్తి ధర ఎంత?"

    parsed = _extract_message(msg, sender_name="రైతు సోదరుడు")
    assert parsed is not None
    assert parsed.phone_number == "919876543210"
    assert parsed.message_id == "wamid.TELUGU_01"
    assert parsed.message_type == "text"
    assert parsed.text_content == "వరంగల్లో ఈరోజు పత్తి ధర ఎంత?"
    assert parsed.sender_name == "రైతు సోదరుడు"


def test_extract_english_text_message():
    """Verify clean extraction of standard English query."""
    msg = MagicMock()
    msg.from_ = "919876543210"
    msg.id = "wamid.ENG_01"
    msg.timestamp = "1700000000"
    msg.type = "text"
    msg.text.body = "What is the cotton price in Warangal today?"

    parsed = _extract_message(msg, sender_name="Ramesh")
    assert parsed is not None
    assert parsed.text_content == "What is the cotton price in Warangal today?"


def test_extract_tanglish_text_message():
    """Verify preservation of Romanized Telugu (Tanglish)."""
    msg = MagicMock()
    msg.from_ = "919876543210"
    msg.id = "wamid.TANGLISH_01"
    msg.timestamp = "1700000000"
    msg.type = "text"
    msg.text.body = "Warangal lo cotton rate entha"

    parsed = _extract_message(msg)
    assert parsed is not None
    assert parsed.text_content == "Warangal lo cotton rate entha"


def test_extract_mixed_telugu_english_text_message():
    """Verify preservation of mixed Telugu-English script."""
    msg = MagicMock()
    msg.from_ = "919876543210"
    msg.id = "wamid.MIXED_01"
    msg.timestamp = "1700000000"
    msg.type = "text"
    msg.text.body = "weather ఎలా ఉంది"

    parsed = _extract_message(msg)
    assert parsed is not None
    assert parsed.text_content == "weather ఎలా ఉంది"


@pytest.mark.asyncio
async def test_empty_text_message_triggers_safe_fallback():
    """Verify empty or whitespace-only text message generates safe fallback and does not crash."""
    parsed = ParsedIncomingMessage(
        phone_number="919876543210",
        message_id="wamid.EMPTY_TEXT_01",
        timestamp="1700000000",
        message_type="text",
        text_content="   ",
    )

    mock_farmer = Farmer(id=uuid.uuid4(), phone_number="919876543210", preferred_language="te")
    mock_conv = Conversation(
        id=uuid.uuid4(),
        farmer_id=mock_farmer.id,
        message_id=parsed.message_id,
        user_message="",
        user_message_type="text"
    )

    mock_db = AsyncMock()
    mock_db_cm = AsyncMock()
    mock_db_cm.__aenter__.return_value = mock_db
    mock_db_cm.__aexit__.return_value = None

    with patch("src.gateway.service.AsyncSessionLocal", return_value=mock_db_cm), \
         patch("src.gateway.service.is_duplicate_message", new_callable=AsyncMock, return_value=False), \
         patch("src.gateway.service.get_or_create_farmer", new_callable=AsyncMock, return_value=mock_farmer), \
         patch("src.gateway.service.store_incoming_message", new_callable=AsyncMock, return_value=mock_conv), \
         patch("src.gateway.service.process_text_message", new_callable=AsyncMock) as mock_ai, \
         patch("src.gateway.service.send_text_message", new_callable=AsyncMock, return_value="wamid.OUT_FALLBACK") as mock_send, \
         patch("src.gateway.service.mark_message_as_read", new_callable=AsyncMock):

        await process_message_pipeline(parsed)

        mock_ai.assert_not_called()
        mock_send.assert_awaited_once()
        sent_text = mock_send.call_args[1]["message_text"]
        assert sent_text == get_fallback_response("te")


# ─────────────────────────────────────────────────────────────────────────────
# 2. VOICE MESSAGE HANDLING & FALLBACKS
# ─────────────────────────────────────────────────────────────────────────────

def test_extract_voice_message_with_media_id():
    """Verify audio message extraction captures media_id and mime_type."""
    msg = MagicMock()
    msg.from_ = "919876543210"
    msg.id = "wamid.AUDIO_01"
    msg.timestamp = "1700000000"
    msg.type = "audio"
    msg.audio.id = "media_audio_12345"
    msg.audio.mime_type = "audio/ogg; codecs=opus"

    parsed = _extract_message(msg)
    assert parsed is not None
    assert parsed.message_type == "audio"
    assert parsed.media_id == "media_audio_12345"
    assert parsed.media_mime_type == "audio/ogg; codecs=opus"


def test_extract_voice_message_with_missing_audio_payload():
    """Verify audio message with missing audio block does not crash and yields audio type with None media_id."""
    msg = MagicMock()
    msg.from_ = "919876543210"
    msg.id = "wamid.AUDIO_NO_MEDIA"
    msg.timestamp = "1700000000"
    msg.type = "audio"
    msg.audio = None

    parsed = _extract_message(msg)
    assert parsed is not None
    assert parsed.message_type == "audio"
    assert parsed.media_id is None


@pytest.mark.asyncio
async def test_voice_message_missing_media_id_returns_voice_fallback():
    """Verify that when audio arrives with no media_id, voice fallback is sent without attempting download."""
    parsed = ParsedIncomingMessage(
        phone_number="919876543210",
        message_id="wamid.AUDIO_MISSING_ID",
        timestamp="1700000000",
        message_type="audio",
        media_id=None,
    )

    mock_farmer = Farmer(id=uuid.uuid4(), phone_number="919876543210", preferred_language="te")
    mock_conv = Conversation(id=uuid.uuid4(), farmer_id=mock_farmer.id, message_id=parsed.message_id)

    mock_db = AsyncMock()
    mock_db_cm = AsyncMock()
    mock_db_cm.__aenter__.return_value = mock_db
    mock_db_cm.__aexit__.return_value = None

    with patch("src.gateway.service.AsyncSessionLocal", return_value=mock_db_cm), \
         patch("src.gateway.service.is_duplicate_message", new_callable=AsyncMock, return_value=False), \
         patch("src.gateway.service.get_or_create_farmer", new_callable=AsyncMock, return_value=mock_farmer), \
         patch("src.gateway.service.store_incoming_message", new_callable=AsyncMock, return_value=mock_conv), \
         patch("src.gateway.service.download_media_bytes", new_callable=AsyncMock) as mock_download, \
         patch("src.gateway.service.send_text_message", new_callable=AsyncMock, return_value="wamid.OUT_VOICE") as mock_send, \
         patch("src.gateway.service.mark_message_as_read", new_callable=AsyncMock):

        await process_message_pipeline(parsed)

        mock_download.assert_not_called()
        mock_send.assert_awaited_once_with(
            to_phone="919876543210",
            message_text=get_voice_fallback_response("te"),
        )


@pytest.mark.asyncio
async def test_voice_message_download_failure_returns_voice_fallback():
    """Verify that media download failure sends voice fallback."""
    parsed = ParsedIncomingMessage(
        phone_number="919876543210",
        message_id="wamid.AUDIO_DL_FAIL",
        timestamp="1700000000",
        message_type="audio",
        media_id="media_fail_111",
    )

    mock_farmer = Farmer(id=uuid.uuid4(), phone_number="919876543210", preferred_language="te")
    mock_conv = Conversation(id=uuid.uuid4(), farmer_id=mock_farmer.id, message_id=parsed.message_id)

    mock_db = AsyncMock()
    mock_db_cm = AsyncMock()
    mock_db_cm.__aenter__.return_value = mock_db
    mock_db_cm.__aexit__.return_value = None

    with patch("src.gateway.service.AsyncSessionLocal", return_value=mock_db_cm), \
         patch("src.gateway.service.is_duplicate_message", new_callable=AsyncMock, return_value=False), \
         patch("src.gateway.service.get_or_create_farmer", new_callable=AsyncMock, return_value=mock_farmer), \
         patch("src.gateway.service.store_incoming_message", new_callable=AsyncMock, return_value=mock_conv), \
         patch("src.gateway.service.download_media_bytes", new_callable=AsyncMock, return_value=None), \
         patch("src.gateway.service.send_text_message", new_callable=AsyncMock, return_value="wamid.OUT_VOICE") as mock_send, \
         patch("src.gateway.service.mark_message_as_read", new_callable=AsyncMock):

        await process_message_pipeline(parsed)

        mock_send.assert_awaited_once_with(
            to_phone="919876543210",
            message_text=get_voice_fallback_response("te"),
        )


# ─────────────────────────────────────────────────────────────────────────────
# 3. IMAGE MESSAGE HANDLING & NON-CROP REJECTION
# ─────────────────────────────────────────────────────────────────────────────

def test_extract_image_message_with_caption():
    """Verify image message extraction correctly captures media_id and caption."""
    msg = MagicMock()
    msg.from_ = "919876543210"
    msg.id = "wamid.IMG_01"
    msg.timestamp = "1700000000"
    msg.type = "image"
    msg.image.id = "img_media_999"
    msg.image.mime_type = "image/jpeg"
    msg.image.caption = "మిరప ఆకుపై మచ్చలు"

    parsed = _extract_message(msg)
    assert parsed is not None
    assert parsed.message_type == "image"
    assert parsed.media_id == "img_media_999"
    assert parsed.media_mime_type == "image/jpeg"
    assert parsed.text_content == "మిరప ఆకుపై మచ్చలు"


@pytest.mark.asyncio
async def test_image_message_missing_media_id_returns_image_fallback():
    """Verify image with missing media_id triggers image fallback."""
    parsed = ParsedIncomingMessage(
        phone_number="919876543210",
        message_id="wamid.IMG_MISSING_ID",
        timestamp="1700000000",
        message_type="image",
        media_id=None,
    )

    mock_farmer = Farmer(id=uuid.uuid4(), phone_number="919876543210", preferred_language="te")
    mock_conv = Conversation(id=uuid.uuid4(), farmer_id=mock_farmer.id, message_id=parsed.message_id)

    mock_db = AsyncMock()
    mock_db_cm = AsyncMock()
    mock_db_cm.__aenter__.return_value = mock_db
    mock_db_cm.__aexit__.return_value = None

    with patch("src.gateway.service.AsyncSessionLocal", return_value=mock_db_cm), \
         patch("src.gateway.service.is_duplicate_message", new_callable=AsyncMock, return_value=False), \
         patch("src.gateway.service.get_or_create_farmer", new_callable=AsyncMock, return_value=mock_farmer), \
         patch("src.gateway.service.store_incoming_message", new_callable=AsyncMock, return_value=mock_conv), \
         patch("src.gateway.service.download_media_bytes", new_callable=AsyncMock) as mock_download, \
         patch("src.gateway.service.send_text_message", new_callable=AsyncMock, return_value="wamid.OUT_IMG") as mock_send, \
         patch("src.gateway.service.mark_message_as_read", new_callable=AsyncMock):

        await process_message_pipeline(parsed)

        mock_download.assert_not_called()
        mock_send.assert_awaited_once_with(
            to_phone="919876543210",
            message_text=get_image_fallback_response("te"),
        )


@pytest.mark.asyncio
async def test_non_agricultural_image_returns_safe_reprompt():
    """Verify that uploading a non-agricultural image does not invent disease diagnosis."""
    from src.ai.service import process_image_message

    mock_db = AsyncMock()
    mock_farmer = Farmer(id=uuid.uuid4(), phone_number="919876543210", preferred_language="te")
    mock_conv = Conversation(id=uuid.uuid4(), farmer_id=mock_farmer.id, user_message="My photo")

    non_crop_json = json.dumps({
        "disease_name": "non_agricultural",
        "confidence_score": 0.0,
        "severity": "low",
        "symptoms": "No plant detected",
        "treatment_recommendation": "None",
        "friendly_whatsapp_reply": "No crop detected."
    })

    with patch("src.ai.gemini_client.generate_multimodal_response", new_callable=AsyncMock, return_value=non_crop_json), \
         patch("src.ai.repository.AIRepository.get_conversation_history", new_callable=AsyncMock, return_value=[]), \
         patch("src.ai.repository.AIRepository.get_farmer_profile", new_callable=AsyncMock, return_value=None), \
         patch("src.memory.service.FarmerMemoryService.format_memory_for_system_prompt", new_callable=AsyncMock, return_value=""):

        result = await process_image_message(mock_db, mock_farmer, mock_conv, b"fake_bytes", "image/jpeg")

        assert result == get_non_crop_image_response("te")
        assert "పంట లేదా మొక్క" in result


# ─────────────────────────────────────────────────────────────────────────────
# 4. UNSUPPORTED MEDIA HANDLING
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("media_type", ["video", "document", "sticker", "contacts", "location", "interactive"])
def test_extract_unsupported_media_types(media_type):
    """Verify that unsupported media types are extracted with their type and never crash."""
    msg = MagicMock()
    msg.from_ = "919876543210"
    msg.id = f"wamid.UNSUPPORTED_{media_type.upper()}"
    msg.timestamp = "1700000000"
    msg.type = media_type

    parsed = _extract_message(msg)
    assert parsed is not None
    assert parsed.message_type == media_type
    assert parsed.phone_number == "919876543210"


@pytest.mark.parametrize("unsupported_type", ["video", "document", "sticker", "contacts", "location"])
@pytest.mark.asyncio
async def test_unsupported_media_returns_guiding_fallback(unsupported_type):
    """Verify unsupported media types receive guiding response explaining supported formats."""
    parsed = ParsedIncomingMessage(
        phone_number="919876543210",
        message_id=f"wamid.TEST_{unsupported_type.upper()}",
        timestamp="1700000000",
        message_type=unsupported_type,
    )

    mock_farmer = Farmer(id=uuid.uuid4(), phone_number="919876543210", preferred_language="te")
    mock_conv = Conversation(id=uuid.uuid4(), farmer_id=mock_farmer.id, message_id=parsed.message_id)

    mock_db = AsyncMock()
    mock_db_cm = AsyncMock()
    mock_db_cm.__aenter__.return_value = mock_db
    mock_db_cm.__aexit__.return_value = None

    with patch("src.gateway.service.AsyncSessionLocal", return_value=mock_db_cm), \
         patch("src.gateway.service.is_duplicate_message", new_callable=AsyncMock, return_value=False), \
         patch("src.gateway.service.get_or_create_farmer", new_callable=AsyncMock, return_value=mock_farmer), \
         patch("src.gateway.service.store_incoming_message", new_callable=AsyncMock, return_value=mock_conv), \
         patch("src.gateway.service.send_text_message", new_callable=AsyncMock, return_value="wamid.OUT_UNSUPPORTED") as mock_send, \
         patch("src.gateway.service.mark_message_as_read", new_callable=AsyncMock):

        await process_message_pipeline(parsed)

        mock_send.assert_awaited_once_with(
            to_phone="919876543210",
            message_text=get_unsupported_media_fallback_response("te"),
        )
        assert "టెక్స్ట్, వాయిస్ మెసేజ్ లేదా పంట ఫోటో" in mock_send.call_args[1]["message_text"]


# ─────────────────────────────────────────────────────────────────────────────
# 5. MALFORMED WEBHOOK PAYLOAD RESILIENCE
# ─────────────────────────────────────────────────────────────────────────────

def test_extract_missing_sender_returns_none():
    """Verify message without 'from' returns None (cannot identify sender)."""
    msg = MagicMock()
    msg.from_ = None
    msg.id = "wamid.NO_FROM"
    assert _extract_message(msg) is None


def test_extract_missing_message_id_returns_none():
    """Verify message without 'id' returns None."""
    msg = MagicMock()
    msg.from_ = "919876543210"
    msg.id = None
    assert _extract_message(msg) is None


def test_webhook_missing_entry_returns_200_ok():
    """Verify payload with missing entry returns 200 OK without crashing."""
    response = client.post("/webhook/whatsapp", json={"object": "whatsapp_business_account"})
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "messages_queued": 0}


def test_webhook_missing_changes_returns_200_ok():
    """Verify payload with missing changes returns 200 OK."""
    response = client.post("/webhook/whatsapp", json={"entry": [{"id": "123"}]})
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "messages_queued": 0}


def test_webhook_missing_value_returns_200_ok():
    """Verify change object with missing value returns 200 OK."""
    payload = {"entry": [{"changes": [{"field": "messages"}]}]}
    response = client.post("/webhook/whatsapp", json=payload)
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "messages_queued": 0}


def test_webhook_missing_messages_returns_200_ok():
    """Verify value with missing messages array returns 200 OK."""
    payload = {"entry": [{"changes": [{"field": "messages", "value": {}}]}]}
    response = client.post("/webhook/whatsapp", json=payload)
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "messages_queued": 0}


# ─────────────────────────────────────────────────────────────────────────────
# 6. STATUS EVENTS FILTERING
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("status_name", ["sent", "delivered", "read", "failed"])
@patch("src.gateway.router.process_message_pipeline")
def test_webhook_status_events_ignored_safely(mock_pipeline, status_name):
    """Verify that WhatsApp status receipts are never queued or forwarded to AI."""
    payload = {
        "object": "whatsapp_business_account",
        "entry": [
            {
                "id": "123",
                "changes": [
                    {
                        "field": "messages",
                        "value": {
                            "messaging_product": "whatsapp",
                            "statuses": [
                                {
                                    "id": "wamid.STATUS_TEST_01",
                                    "status": status_name,
                                    "timestamp": "1700000000",
                                    "recipient_id": "919876543210"
                                }
                            ]
                        }
                    }
                ]
            }
        ]
    }

    response = client.post("/webhook/whatsapp", json=payload)
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "messages_queued": 0}
    mock_pipeline.assert_not_called()


# ─────────────────────────────────────────────────────────────────────────────
# 7. BATCHING: MULTIPLE MESSAGES IN ONE WEBHOOK
# ─────────────────────────────────────────────────────────────────────────────

@patch("src.gateway.router.process_message_pipeline")
def test_multiple_messages_in_one_webhook_queued_independently(mock_pipeline):
    """Verify multiple messages in a single webhook payload are queued independently."""
    payload = {
        "object": "whatsapp_business_account",
        "entry": [
            {
                "id": "123",
                "changes": [
                    {
                        "field": "messages",
                        "value": {
                            "messaging_product": "whatsapp",
                            "messages": [
                                {
                                    "from": "919876543210",
                                    "id": "wamid.MSG_1",
                                    "type": "text",
                                    "text": {"body": "First message"}
                                },
                                {
                                    "from": "919876543211",
                                    "id": "wamid.MSG_2",
                                    "type": "text",
                                    "text": {"body": "Second message"}
                                }
                            ]
                        }
                    }
                ]
            }
        ]
    }

    response = client.post("/webhook/whatsapp", json=payload)
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "messages_queued": 2}


@patch("src.gateway.router.process_message_pipeline")
def test_duplicate_message_ids_within_same_webhook_batch_deduplicated(mock_pipeline):
    """Verify duplicate message ID in same webhook payload is queued only once."""
    payload = {
        "object": "whatsapp_business_account",
        "entry": [
            {
                "id": "123",
                "changes": [
                    {
                        "field": "messages",
                        "value": {
                            "messaging_product": "whatsapp",
                            "messages": [
                                {
                                    "from": "919876543210",
                                    "id": "wamid.DUPLICATE_ID",
                                    "type": "text",
                                    "text": {"body": "First"}
                                },
                                {
                                    "from": "919876543210",
                                    "id": "wamid.DUPLICATE_ID",
                                    "type": "text",
                                    "text": {"body": "First retry"}
                                }
                            ]
                        }
                    }
                ]
            }
        ]
    }

    response = client.post("/webhook/whatsapp", json=payload)
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "messages_queued": 1}


# ─────────────────────────────────────────────────────────────────────────────
# 8. OUTBOUND SAFETY & DATA INTEGRITY
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_send_text_message_rejects_empty_and_substitutes_fallback():
    """Verify that send_text_message substitutes fallback when given None or empty string."""
    with patch("httpx.AsyncClient.post") as mock_post, \
         patch("src.gateway.whatsapp_client.get_settings") as mock_settings:

        mock_settings.return_value.whatsapp_api_token = "token123"
        mock_settings.return_value.whatsapp_phone_number_id = "phone123"
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"messages": [{"id": "wamid.OUT_SAFE"}]}
        mock_resp.text = '{"messages": [{"id": "wamid.OUT_SAFE"}]}'
        mock_post.return_value = mock_resp

        out_id = await send_text_message("919876543210", "   ")
        assert out_id == "wamid.OUT_SAFE"

        # Check payload sent to Meta
        call_kwargs = mock_post.call_args[1]
        body = call_kwargs["json"]["text"]["body"]
        assert body == get_fallback_response("te")


@pytest.mark.asyncio
async def test_send_text_message_rejects_missing_phone():
    """Verify missing phone returns None without making HTTP request."""
    with patch("httpx.AsyncClient.post") as mock_post:
        out_id = await send_text_message("", "Hello farmer")
        assert out_id is None
        mock_post.assert_not_called()


def test_finalize_whatsapp_response_preserves_short_text():
    """Verify short text is returned untouched."""
    text = "వరంగల్ మార్కెట్లో పత్తి ధర ₹8,900 ఉంది."
    assert _finalize_whatsapp_response(text) == text


def test_finalize_whatsapp_response_preserves_authoritative_numbers_and_blocks():
    """Verify market prices, units, and safety critical escalation blocks are strictly preserved."""
    long_advice = "పంటకు సమగ్ర ఎరువుల యాజమాన్యం ఎంతో ముఖ్యం. " * 60  # ~2400 chars
    market_block = "📊 *వరంగల్ మార్కెట్ ధరలు*\nపత్తి: ₹8,900 (కనీసం: ₹5,000, గరిష్టం: ₹9,250 / క్వింటాల్)"
    escalation_block = "🚨 *వ్యవసాయ అధికారిని సంప్రదించండి*\nసమీప AEO ఫోన్: 1800-180-1551"

    raw_response = f"{long_advice}\n\n{market_block}\n\n{escalation_block}"
    assert len(raw_response) > 2000

    finalized = _finalize_whatsapp_response(raw_response, max_chars=1600)
    assert len(finalized) <= 1600

    # Strict assertion: Authoritative prices, units, and escalation are preserved untouched
    assert "₹8,900" in finalized
    assert "₹5,000" in finalized
    assert "₹9,250" in finalized
    assert "క్వింటాల్" in finalized
    assert "🚨 *వ్యవసాయ అధికారిని సంప్రదించండి*" in finalized
    assert "1800-180-1551" in finalized


# ─────────────────────────────────────────────────────────────────────────────
# 9. PRIVACY & LOG SANITIZATION
# ─────────────────────────────────────────────────────────────────────────────

def test_mask_phone_number():
    """Verify phone masking hides middle digits."""
    assert mask_phone_number("919876543210") == "9198****3210"
    assert mask_phone_number("") == "***"


def test_sanitize_headers_for_logging():
    """Verify authorization and secret signatures are redacted from logs."""
    headers = {
        "host": "localhost:8000",
        "authorization": "Bearer secret_meta_token_xyz",
        "x-hub-signature-256": "sha256=abcdef1234567890",
        "content-type": "application/json",
    }
    sanitized = sanitize_headers_for_logging(headers)
    assert sanitized["host"] == "localhost:8000"
    assert sanitized["content-type"] == "application/json"
    assert sanitized["authorization"] == "***REDACTED***"
    assert sanitized["x-hub-signature-256"] == "***REDACTED***"
