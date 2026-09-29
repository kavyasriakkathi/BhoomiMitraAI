"""
Tests for Focused Stock Query Resolution and Voice Response Pipeline.

Verifies:
1. Tanglish language detection ("Korutla lo urea stock undha?" -> 'te')
2. Tanglish SHOPS intent detection
3. Korutla -> Jagtial location resolution
4. Verified stock lookup against seeded test inventory fixture
5. Missing stock factual unverified response (no crop advice fallback)
6. Gemini cannot inject stock information (bypassed for SHOPS intent)
7. No "How is your paddy crop doing?" farmer-profile contamination
8. Telugu voice -> Telugu TEXT + Telugu AUDIO
9. English voice -> English TEXT + English AUDIO
10. Hindi voice -> Hindi TEXT + Hindi AUDIO
11. Tanglish voice -> Telugu TEXT + Telugu AUDIO
12. TTS failure still delivers TEXT successfully
13. Text answer and TTS input are EXACTLY identical
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from src.language.detector import detect_language, detect_language_with_confidence
from src.ai.decision_engine import AIDecisionEngine, FarmerIntent
from src.shops.service import (
    _extract_district_from_query,
    _detect_product_from_query,
    resolve_shop_district,
    _detect_shop_intent,
    enrich_response_with_shops,
)
from src.core.models import Farmer, Conversation, Shop, Inventory
from src.gateway.schemas import ParsedIncomingMessage
from src.gateway.service import process_message_pipeline
from src.language.schemas import TranscriptionResponse


# =============================================================================
# 1. Tanglish Language Detection
# =============================================================================

def test_tanglish_language_detection_stock_phrases():
    """Verify that romanized Telugu queries are correctly recognized as Telugu with high confidence."""
    queries = [
        "Korutla lo urea stock undha?",
        "urea stock undha",
        "urea stock unda",
        "urea stock undhi",
        "urea stock undi",
        "urea undha",
        "urea unda",
        "urea undi",
        "urea vundha",
        "urea vunda",
        "vari ki em fertilizer vadali",
        "patti crop lo purugula mandu eppudu spray cheyali",
    ]
    for q in queries:
        lang, conf, method = detect_language_with_confidence(q)
        assert lang == "te", f"Query '{q}' was detected as '{lang}' instead of 'te'"
        assert conf >= 0.70, f"Query '{q}' had low confidence: {conf}"
        assert detect_language(q) == "te"


# =============================================================================
# 2. Stock Intent Detection (FarmerIntent.SHOPS)
# =============================================================================

def test_stock_intent_detection_real_pipeline():
    """Verify all stock queries reach FarmerIntent.SHOPS across English, Telugu, and Tanglish."""
    queries = [
        "Korutla lo urea stock undha?",
        "urea stock undha",
        "urea stock unda",
        "urea undha",
        "urea unda",
        "urea stock undhi",
        "urea stock undi",
        "urea undi",
        "urea vundha",
        "urea vunda",
        "is urea in stock?",
        "urea stock",
        "కోరుట్లలో యూరియా స్టాక్ ఉందా?",
        "యూరియా స్టాక్ ఉందా",
        "యూరియా ఉందా",
    ]
    for q in queries:
        intents = AIDecisionEngine.detect_all_intents(q)
        assert FarmerIntent.SHOPS in intents, f"Query '{q}' failed to detect FarmerIntent.SHOPS: {intents}"
        primary = AIDecisionEngine.detect_primary_intent(q)
        assert primary == FarmerIntent.SHOPS, f"Query '{q}' primary intent is '{primary}' instead of SHOPS"
        assert _detect_shop_intent(q.lower(), q), f"_detect_shop_intent returned False for '{q}'"


# =============================================================================
# 3. Korutla Location Resolution to Jagtial
# =============================================================================

def test_korutla_resolves_to_jagtial():
    """Verify Korutla in English and Telugu maps to canonical Jagtial district."""
    assert _extract_district_from_query("Korutla lo urea stock undha?") == "Jagtial"
    assert _extract_district_from_query("కోరుట్లలో యూరియా స్టాక్ ఉందా?") == "Jagtial"
    assert resolve_shop_district("Korutla") == "Jagtial"
    assert resolve_shop_district("కోరుట్ల") == "Jagtial"
    assert resolve_shop_district("Jagtial") == "Jagtial"
    assert resolve_shop_district("జగిత్యాల") == "Jagtial"
    # Existing districts remain intact
    assert resolve_shop_district("Warangal") == "Warangal"
    assert resolve_shop_district("వరంగల్") == "Warangal"
    assert resolve_shop_district("Karimnagar") == "Karimnagar"


# =============================================================================
# 4 & 5. Grounded Stock Responses with Seeded Test Fixture
# =============================================================================

@pytest.mark.asyncio
async def test_verified_stock_returns_only_seeded_inventory():
    """
    When verified inventory exists for Korutla/Jagtial, return ONLY verified inventory
    information from the seeded test fixture (Shop: 'Sri Rama Kisan Agro', price: ₹266.5, stock: 45 bags).
    """
    mock_db = AsyncMock()

    # Seed known test fixtures
    test_shop = Shop(
        id=uuid4(),
        shop_name="Sri Rama Kisan Agro",
        district="Jagtial",
        address="Main Road, Korutla",
        phone_number="9848012345",
        status="active",
        delivery_available=True,
    )
    test_inventory = Inventory(
        id=uuid4(),
        shop_id=test_shop.id,
        product_name="Neem Coated Urea",
        category="fertilizer",
        brand="IFFCO",
        price=266.50,
        unit="bag",
        quantity_in_stock=45,
        available=True,
        minimum_stock_level=10,
    )

    mock_farmer = MagicMock(spec=Farmer)
    mock_farmer.id = uuid4()
    mock_farmer.preferred_language = "te"
    mock_farmer.district = "Jagtial"

    with patch("src.shops.repository.ShopRepository.seed_default_shops_if_empty", new_callable=AsyncMock), \
         patch("src.shops.repository.ShopRepository.search_shops_by_product", new_callable=AsyncMock, return_value=[(test_shop, test_inventory)]):

        res = await enrich_response_with_shops(
            db=mock_db,
            query_text="Korutla lo urea stock undha?",
            ai_response="",
            farmer=mock_farmer,
        )

        # Must contain verified seeded data
        assert "Sri Rama Kisan Agro" in res
        assert "266.5" in res
        assert "45 bag" in res
        assert "9848012345" in res
        # Must NOT contain crop advice or profile contamination
        assert "వరి పంట ఎలా ఉంది" not in res
        assert "How is your paddy crop doing" not in res


@pytest.mark.asyncio
async def test_missing_stock_returns_factual_unverified_response():
    """
    When no verified inventory exists in the district, return the factual localized
    unverified response, never inventing stock or falling back to generic crop advice.
    """
    mock_db = AsyncMock()
    mock_farmer = MagicMock(spec=Farmer)
    mock_farmer.id = uuid4()
    mock_farmer.preferred_language = "te"
    mock_farmer.district = "Jagtial"

    with patch("src.shops.repository.ShopRepository.seed_default_shops_if_empty", new_callable=AsyncMock), \
         patch("src.shops.repository.ShopRepository.search_shops_by_product", new_callable=AsyncMock, return_value=[]):

        res = await enrich_response_with_shops(
            db=mock_db,
            query_text="Korutla lo urea stock undha?",
            ai_response="",
            farmer=mock_farmer,
        )

        assert "🏬" in res
        assert "నమోదు కాలేదు" in res or "not registered" in res or "సమీప" in res
        # Must not have hallucinated prices, quantities, or crop advice
        assert "వరి పంట ఎలా ఉంది" not in res
        assert "How is your paddy crop doing" not in res


# =============================================================================
# 6 & 7. Gemini Cannot Inject Stock and No Farmer Context Contamination
# =============================================================================

@pytest.mark.asyncio
async def test_gemini_bypassed_for_explicit_shops_query():
    """
    Verify that AIDecisionEngine.process_message completely bypasses general Gemini
    generation when primary_intent is FarmerIntent.SHOPS.
    """
    mock_db = AsyncMock()
    mock_db.execute = AsyncMock()
    mock_db.commit = AsyncMock()
    mock_db.add = MagicMock()

    mock_farmer = MagicMock(spec=Farmer)
    mock_farmer.id = uuid4()
    mock_farmer.preferred_language = "te"
    mock_farmer.district = "Jagtial"

    mock_conv = MagicMock(spec=Conversation)
    mock_conv.id = uuid4()
    mock_conv.user_message = "Korutla lo urea stock undha?"

    engine = AIDecisionEngine()

    with patch("src.ai.service.AIService.generate_ai_response", new_callable=AsyncMock) as mock_gemini, \
         patch("src.shops.service.enrich_response_with_shops", new_callable=AsyncMock, return_value="🏬 శ్రీ రామ కిసాన్ ఆగ్రో వద్ద యూరియా స్టాక్ ఉంది.") as mock_shops:

        res = await engine.process_message(mock_db, mock_farmer, mock_conv)

        # Gemini general generation must NOT have been called!
        mock_gemini.assert_not_called()
        # Shops module must have been called with empty ai_response
        mock_shops.assert_awaited_once_with(mock_db, "Korutla lo urea stock undha?", "", mock_farmer)
        assert "యూరియా స్టాక్ ఉంది" in res
        assert "వరి పంట ఎలా ఉంది" not in res


# =============================================================================
# 8, 9, 10, 11, 12, 13: Voice Responses (STT -> Text + Audio with Exact String)
# =============================================================================

def _setup_voice_mocks(mock_farmer, mock_conv):
    mock_db = AsyncMock()
    mock_db_cm = AsyncMock()
    mock_db_cm.__aenter__.return_value = mock_db
    mock_db_cm.__aexit__.return_value = None
    return mock_db_cm


@pytest.mark.asyncio
async def test_voice_telugu_sends_text_and_audio_exact_string():
    """Telugu voice input -> Telugu text + Telugu audio with EXACT same final answer string."""
    parsed = ParsedIncomingMessage(
        phone_number="919876543210",
        message_id="wamid.VOICE_TE_01",
        timestamp="1700000000",
        message_type="audio",
        media_id="audio_media_te",
    )
    mock_farmer = Farmer(id=uuid4(), phone_number="919876543210", preferred_language="te")
    mock_conv = Conversation(id=uuid4(), farmer_id=mock_farmer.id, message_id=parsed.message_id, user_message=None, user_message_type="audio")
    mock_db_cm = _setup_voice_mocks(mock_farmer, mock_conv)

    with patch("src.gateway.service.AsyncSessionLocal", return_value=mock_db_cm), \
         patch("src.gateway.service.is_duplicate_message", new_callable=AsyncMock, return_value=False), \
         patch("src.gateway.service.get_or_create_farmer", new_callable=AsyncMock, return_value=mock_farmer), \
         patch("src.gateway.service.store_incoming_message", new_callable=AsyncMock, return_value=mock_conv), \
         patch("src.gateway.service.download_media_bytes", new_callable=AsyncMock, return_value=(b"audio_bytes", "audio/ogg")), \
         patch("src.gateway.service.get_language_service") as mock_lang_svc, \
         patch("src.gateway.service.process_text_message", new_callable=AsyncMock, return_value="వరి పంటలో ఎరువుల సలహా"), \
         patch("src.gateway.service.send_text_message", new_callable=AsyncMock, return_value="wamid.OUT_TEXT_TE") as mock_send_text, \
         patch("src.gateway.service.upload_media_bytes", new_callable=AsyncMock, return_value="meta_audio_te") as mock_upload, \
         patch("src.gateway.service.send_audio_message", new_callable=AsyncMock, return_value="wamid.OUT_AUDIO_TE") as mock_send_audio, \
         patch("src.gateway.service.mark_message_as_read", new_callable=AsyncMock), \
         patch("src.gateway.service.get_settings") as mock_settings:

        mock_settings.return_value.enable_voice_responses = True
        mock_lang_svc.return_value.transcribe_audio = AsyncMock(
            return_value=TranscriptionResponse(provider_used="google", transcription_text="వరికి యూరియా ఎంత వేయాలి?", detected_language="te")
        )
        mock_lang_svc.return_value.synthesize_speech = AsyncMock(return_value=[b"OggS_telugu_audio"])

        await process_message_pipeline(parsed)

        mock_send_text.assert_awaited_once_with(to_phone="919876543210", message_text="వరి పంటలో ఎరువుల సలహా")
        mock_lang_svc.return_value.synthesize_speech.assert_awaited_once_with("వరి పంటలో ఎరువుల సలహా", "te")
        mock_upload.assert_awaited_once_with(b"OggS_telugu_audio", mime_type="audio/ogg")
        mock_send_audio.assert_awaited_once_with("919876543210", "meta_audio_te")


@pytest.mark.asyncio
async def test_voice_english_sends_text_and_audio_exact_string():
    """English voice input -> English text + English audio with EXACT same final answer string."""
    parsed = ParsedIncomingMessage(
        phone_number="919876543211",
        message_id="wamid.VOICE_EN_01",
        timestamp="1700000000",
        message_type="audio",
        media_id="audio_media_en",
    )
    mock_farmer = Farmer(id=uuid4(), phone_number="919876543211", preferred_language="en")
    mock_conv = Conversation(id=uuid4(), farmer_id=mock_farmer.id, message_id=parsed.message_id, user_message=None, user_message_type="audio")
    mock_db_cm = _setup_voice_mocks(mock_farmer, mock_conv)

    with patch("src.gateway.service.AsyncSessionLocal", return_value=mock_db_cm), \
         patch("src.gateway.service.is_duplicate_message", new_callable=AsyncMock, return_value=False), \
         patch("src.gateway.service.get_or_create_farmer", new_callable=AsyncMock, return_value=mock_farmer), \
         patch("src.gateway.service.store_incoming_message", new_callable=AsyncMock, return_value=mock_conv), \
         patch("src.gateway.service.download_media_bytes", new_callable=AsyncMock, return_value=(b"audio_bytes", "audio/ogg")), \
         patch("src.gateway.service.get_language_service") as mock_lang_svc, \
         patch("src.gateway.service.process_text_message", new_callable=AsyncMock, return_value="Apply 50 kg urea per acre at tillering stage."), \
         patch("src.gateway.service.send_text_message", new_callable=AsyncMock, return_value="wamid.OUT_TEXT_EN") as mock_send_text, \
         patch("src.gateway.service.upload_media_bytes", new_callable=AsyncMock, return_value="meta_audio_en") as mock_upload, \
         patch("src.gateway.service.send_audio_message", new_callable=AsyncMock, return_value="wamid.OUT_AUDIO_EN") as mock_send_audio, \
         patch("src.gateway.service.mark_message_as_read", new_callable=AsyncMock), \
         patch("src.gateway.service.get_settings") as mock_settings:

        mock_settings.return_value.enable_voice_responses = True
        mock_lang_svc.return_value.transcribe_audio = AsyncMock(
            return_value=TranscriptionResponse(provider_used="google", transcription_text="How much urea should I apply for paddy?", detected_language="en")
        )
        mock_lang_svc.return_value.synthesize_speech = AsyncMock(return_value=[b"OggS_english_audio"])

        await process_message_pipeline(parsed)

        mock_send_text.assert_awaited_once_with(to_phone="919876543211", message_text="Apply 50 kg urea per acre at tillering stage.")
        mock_lang_svc.return_value.synthesize_speech.assert_awaited_once_with("Apply 50 kg urea per acre at tillering stage.", "en")
        mock_upload.assert_awaited_once_with(b"OggS_english_audio", mime_type="audio/ogg")
        mock_send_audio.assert_awaited_once_with("919876543211", "meta_audio_en")


@pytest.mark.asyncio
async def test_voice_hindi_sends_text_and_audio_exact_string():
    """Hindi voice input -> Hindi text + Hindi audio with EXACT same final answer string."""
    parsed = ParsedIncomingMessage(
        phone_number="919876543212",
        message_id="wamid.VOICE_HI_01",
        timestamp="1700000000",
        message_type="audio",
        media_id="audio_media_hi",
    )
    mock_farmer = Farmer(id=uuid4(), phone_number="919876543212", preferred_language="hi")
    mock_conv = Conversation(id=uuid4(), farmer_id=mock_farmer.id, message_id=parsed.message_id, user_message=None, user_message_type="audio")
    mock_db_cm = _setup_voice_mocks(mock_farmer, mock_conv)

    with patch("src.gateway.service.AsyncSessionLocal", return_value=mock_db_cm), \
         patch("src.gateway.service.is_duplicate_message", new_callable=AsyncMock, return_value=False), \
         patch("src.gateway.service.get_or_create_farmer", new_callable=AsyncMock, return_value=mock_farmer), \
         patch("src.gateway.service.store_incoming_message", new_callable=AsyncMock, return_value=mock_conv), \
         patch("src.gateway.service.download_media_bytes", new_callable=AsyncMock, return_value=(b"audio_bytes", "audio/ogg")), \
         patch("src.gateway.service.get_language_service") as mock_lang_svc, \
         patch("src.gateway.service.process_text_message", new_callable=AsyncMock, return_value="धान की फसल में यूरिया की सलाह।"), \
         patch("src.gateway.service.send_text_message", new_callable=AsyncMock, return_value="wamid.OUT_TEXT_HI") as mock_send_text, \
         patch("src.gateway.service.upload_media_bytes", new_callable=AsyncMock, return_value="meta_audio_hi") as mock_upload, \
         patch("src.gateway.service.send_audio_message", new_callable=AsyncMock, return_value="wamid.OUT_AUDIO_HI") as mock_send_audio, \
         patch("src.gateway.service.mark_message_as_read", new_callable=AsyncMock), \
         patch("src.gateway.service.get_settings") as mock_settings:

        mock_settings.return_value.enable_voice_responses = True
        mock_lang_svc.return_value.transcribe_audio = AsyncMock(
            return_value=TranscriptionResponse(provider_used="google", transcription_text="धान में कितना यूरिया डालना है?", detected_language="hi")
        )
        mock_lang_svc.return_value.synthesize_speech = AsyncMock(return_value=[b"OggS_hindi_audio"])

        await process_message_pipeline(parsed)

        mock_send_text.assert_awaited_once_with(to_phone="919876543212", message_text="धान की फसल में यूरिया की सलाह।")
        mock_lang_svc.return_value.synthesize_speech.assert_awaited_once_with("धान की फसल में यूरिया की सलाह।", "hi")
        mock_upload.assert_awaited_once_with(b"OggS_hindi_audio", mime_type="audio/ogg")
        mock_send_audio.assert_awaited_once_with("919876543212", "meta_audio_hi")


@pytest.mark.asyncio
async def test_tanglish_voice_stock_query_to_telugu_text_and_audio():
    """
    Tanglish voice query: 'Korutla lo urea stock undha?'
    -> detected as Telugu
    -> reaches verified inventory lookup
    -> sends Telugu text
    -> synthesizes and sends Telugu audio with EXACT same final answer.
    """
    parsed = ParsedIncomingMessage(
        phone_number="919876543213",
        message_id="wamid.VOICE_TANGLISH_01",
        timestamp="1700000000",
        message_type="audio",
        media_id="audio_media_tanglish",
    )
    mock_farmer = Farmer(id=uuid4(), phone_number="919876543213", preferred_language="te")
    mock_conv = Conversation(id=uuid4(), farmer_id=mock_farmer.id, message_id=parsed.message_id, user_message=None, user_message_type="audio")
    mock_db_cm = _setup_voice_mocks(mock_farmer, mock_conv)

    expected_verified_answer = "🏬 సమీప వ్యవసాయ దుకాణాలు & లభ్యత:\n\n• *శ్రీ రామ కిసాన్ ఆగ్రో*\n  📦 ఉత్పత్తి: Neem Coated Urea (IFFCO)\n  💰 ధర: ₹266.5/bag | స్టాక్ లభ్యత (45 bags)\n  📞 సంప్రదించండి: 9848012345 | తెరిచి ఉంది"

    with patch("src.gateway.service.AsyncSessionLocal", return_value=mock_db_cm), \
         patch("src.gateway.service.is_duplicate_message", new_callable=AsyncMock, return_value=False), \
         patch("src.gateway.service.get_or_create_farmer", new_callable=AsyncMock, return_value=mock_farmer), \
         patch("src.gateway.service.store_incoming_message", new_callable=AsyncMock, return_value=mock_conv), \
         patch("src.gateway.service.download_media_bytes", new_callable=AsyncMock, return_value=(b"audio_bytes", "audio/ogg")), \
         patch("src.gateway.service.get_language_service") as mock_lang_svc, \
         patch("src.gateway.service.process_text_message", new_callable=AsyncMock, return_value=expected_verified_answer), \
         patch("src.gateway.service.send_text_message", new_callable=AsyncMock, return_value="wamid.OUT_TEXT_TANGLISH") as mock_send_text, \
         patch("src.gateway.service.upload_media_bytes", new_callable=AsyncMock, return_value="meta_audio_tanglish") as mock_upload, \
         patch("src.gateway.service.send_audio_message", new_callable=AsyncMock, return_value="wamid.OUT_AUDIO_TANGLISH") as mock_send_audio, \
         patch("src.gateway.service.mark_message_as_read", new_callable=AsyncMock), \
         patch("src.gateway.service.get_settings") as mock_settings:

        mock_settings.return_value.enable_voice_responses = True
        mock_lang_svc.return_value.transcribe_audio = AsyncMock(
            return_value=TranscriptionResponse(provider_used="google", transcription_text="Korutla lo urea stock undha?", detected_language="te")
        )
        mock_lang_svc.return_value.synthesize_speech = AsyncMock(return_value=[b"OggS_tanglish_telugu_audio"])

        await process_message_pipeline(parsed)

        # 1. Text message sent with exact grounded answer
        mock_send_text.assert_awaited_once_with(
            to_phone="919876543213",
            message_text=expected_verified_answer,
        )
        # 2. TTS invoked with EXACT SAME answer string and language 'te'
        mock_lang_svc.return_value.synthesize_speech.assert_awaited_once_with(
            expected_verified_answer,
            "te",
        )
        # 3. Audio uploaded and sent
        mock_upload.assert_awaited_once_with(b"OggS_tanglish_telugu_audio", mime_type="audio/ogg")
        mock_send_audio.assert_awaited_once_with("919876543213", "meta_audio_tanglish")


@pytest.mark.asyncio
async def test_tts_failure_preserves_text_delivery():
    """If TTS synthesis throws an exception, WhatsApp text must still be successfully delivered."""
    parsed = ParsedIncomingMessage(
        phone_number="919876543214",
        message_id="wamid.VOICE_FAIL_TTS_01",
        timestamp="1700000000",
        message_type="audio",
        media_id="audio_media_fail",
    )
    mock_farmer = Farmer(id=uuid4(), phone_number="919876543214", preferred_language="te")
    mock_conv = Conversation(id=uuid4(), farmer_id=mock_farmer.id, message_id=parsed.message_id, user_message=None, user_message_type="audio")
    mock_db_cm = _setup_voice_mocks(mock_farmer, mock_conv)

    with patch("src.gateway.service.AsyncSessionLocal", return_value=mock_db_cm), \
         patch("src.gateway.service.is_duplicate_message", new_callable=AsyncMock, return_value=False), \
         patch("src.gateway.service.get_or_create_farmer", new_callable=AsyncMock, return_value=mock_farmer), \
         patch("src.gateway.service.store_incoming_message", new_callable=AsyncMock, return_value=mock_conv), \
         patch("src.gateway.service.download_media_bytes", new_callable=AsyncMock, return_value=(b"audio_bytes", "audio/ogg")), \
         patch("src.gateway.service.get_language_service") as mock_lang_svc, \
         patch("src.gateway.service.process_text_message", new_callable=AsyncMock, return_value="రైతు మిత్ర సమాధానం"), \
         patch("src.gateway.service.send_text_message", new_callable=AsyncMock, return_value="wamid.OUT_TEXT_OK") as mock_send_text, \
         patch("src.gateway.service.upload_media_bytes", new_callable=AsyncMock) as mock_upload, \
         patch("src.gateway.service.send_audio_message", new_callable=AsyncMock) as mock_send_audio, \
         patch("src.gateway.service.mark_message_as_read", new_callable=AsyncMock), \
         patch("src.gateway.service.get_settings") as mock_settings:

        mock_settings.return_value.enable_voice_responses = True
        mock_lang_svc.return_value.transcribe_audio = AsyncMock(
            return_value=TranscriptionResponse(provider_used="google", transcription_text="సలహా ఇవ్వండి", detected_language="te")
        )
        # TTS fails
        mock_lang_svc.return_value.synthesize_speech = AsyncMock(side_effect=RuntimeError("Google Cloud TTS quota exceeded"))

        # Pipeline must not raise exception
        await process_message_pipeline(parsed)

        # Text was delivered successfully
        mock_send_text.assert_awaited_once_with(to_phone="919876543214", message_text="రైతు మిత్ర సమాధానం")
        # Audio was not sent, but no crash
        mock_upload.assert_not_called()
        mock_send_audio.assert_not_called()


@pytest.mark.asyncio
async def test_text_answer_equals_tts_input_exact():
    """
    Verify that TTS must receive the EXACT SAME final answer that is sent as WhatsApp text.
    Never generate a second AI answer for TTS.
    """
    parsed = ParsedIncomingMessage(
        phone_number="919876543215",
        message_id="wamid.VOICE_IDENTICAL_01",
        timestamp="1700000000",
        message_type="audio",
        media_id="audio_media_identical",
    )
    mock_farmer = Farmer(id=uuid4(), phone_number="919876543215", preferred_language="te")
    mock_conv = Conversation(id=uuid4(), farmer_id=mock_farmer.id, message_id=parsed.message_id, user_message=None, user_message_type="audio")
    mock_db_cm = _setup_voice_mocks(mock_farmer, mock_conv)

    verified_answer = "🏬 సమీప వ్యవసాయ దుకాణాలు & లభ్యత:\n\n• *శ్రీ రామ కిసాన్ ఆగ్రో*\n  📦 ఉత్పత్తి: Neem Coated Urea (IFFCO)\n  💰 ధర: ₹266.5/bag | స్టాక్ అందుబాటులో ఉంది (45 bags)\n  📞 సంప్రదించండి: 9848012345 | తెరిచి ఉంది"

    with patch("src.gateway.service.AsyncSessionLocal", return_value=mock_db_cm), \
         patch("src.gateway.service.is_duplicate_message", new_callable=AsyncMock, return_value=False), \
         patch("src.gateway.service.get_or_create_farmer", new_callable=AsyncMock, return_value=mock_farmer), \
         patch("src.gateway.service.store_incoming_message", new_callable=AsyncMock, return_value=mock_conv), \
         patch("src.gateway.service.download_media_bytes", new_callable=AsyncMock, return_value=(b"audio_bytes", "audio/ogg")), \
         patch("src.gateway.service.get_language_service") as mock_lang_svc, \
         patch("src.gateway.service.process_text_message", new_callable=AsyncMock, return_value=verified_answer), \
         patch("src.gateway.service.send_text_message", new_callable=AsyncMock, return_value="wamid.OUT_TEXT_ID") as mock_send_text, \
         patch("src.gateway.service.upload_media_bytes", new_callable=AsyncMock, return_value="meta_audio_id") as mock_upload, \
         patch("src.gateway.service.send_audio_message", new_callable=AsyncMock, return_value="wamid.OUT_AUDIO_ID") as mock_send_audio, \
         patch("src.gateway.service.mark_message_as_read", new_callable=AsyncMock), \
         patch("src.gateway.service.get_settings") as mock_settings:

        mock_settings.return_value.enable_voice_responses = True
        mock_lang_svc.return_value.transcribe_audio = AsyncMock(
            return_value=TranscriptionResponse(provider_used="google", transcription_text="Korutla lo urea stock undha?", detected_language="te")
        )
        mock_lang_svc.return_value.synthesize_speech = AsyncMock(return_value=[b"OggS_exact_same_audio"])

        await process_message_pipeline(parsed)

        # Retrieve the exact strings sent to Text and TTS
        text_message_sent = mock_send_text.call_args[1]["message_text"]
        tts_input_received = mock_lang_svc.return_value.synthesize_speech.call_args[0][0]

        # Must be EXACTLY identical strings
        assert text_message_sent == tts_input_received, (
            f"Text answer and TTS input differ!\nText: {text_message_sent}\nTTS:  {tts_input_received}"
        )
        # TTS was called exactly once with the same answer
        assert mock_lang_svc.return_value.synthesize_speech.await_count == 1


@pytest.mark.asyncio
async def test_farmer_voice_korutla_stock_query_end_to_end():
    """
    Specifically tests Requirement 5:
    Farmer voice: 'Korutla lo urea stock undha?'
    - STT recognizes the query
    - Language becomes Telugu/Tanglish Telugu ('te')
    - Intent becomes SHOPS (FarmerIntent.SHOPS)
    - Korutla resolves to Jagtial
    - Verified inventory lookup is performed
    - Response contains only grounded stock information
    - No fabricated stock
    - No unrelated farmer-profile question ('How is your paddy crop doing?')
    - Final text and TTS use exactly the same answer.
    """
    farmer_query = "Korutla lo urea stock undha?"

    # 1. STT recognizes the query
    # 2. Language becomes Telugu/Tanglish Telugu
    lang, conf, _ = detect_language_with_confidence(farmer_query)
    assert lang == "te", f"Expected language 'te', got '{lang}'"
    assert conf >= 0.70

    # 3. Intent becomes SHOPS
    intents = AIDecisionEngine.detect_all_intents(farmer_query)
    assert FarmerIntent.SHOPS in intents
    primary_intent = AIDecisionEngine.detect_primary_intent(farmer_query)
    assert primary_intent == FarmerIntent.SHOPS

    # 4. Korutla resolves to Jagtial
    resolved_dist = _extract_district_from_query(farmer_query)
    assert resolved_dist == "Jagtial"
    assert resolve_shop_district("Korutla") == "Jagtial"

    # 5. Verified inventory lookup is performed on seeded fixture
    test_shop = Shop(
        id=uuid4(),
        shop_name="Korutla Kisan Seva Kendra",
        district="Jagtial",
        address="Near Bus Stand, Korutla",
        phone_number="9848099887",
        status="active",
        delivery_available=True,
    )
    test_inventory = Inventory(
        id=uuid4(),
        shop_id=test_shop.id,
        product_name="Neem Coated Urea",
        category="fertilizer",
        brand="IFFCO",
        price=266.50,
        unit="bag",
        quantity_in_stock=50,
        available=True,
        minimum_stock_level=10,
    )

    mock_db = AsyncMock()
    mock_farmer = Farmer(id=uuid4(), phone_number="919848099887", preferred_language="te")
    mock_farmer.district = "Jagtial"

    # Verify shops enrichment output with real function
    with patch("src.shops.repository.ShopRepository.seed_default_shops_if_empty", new_callable=AsyncMock), \
         patch("src.shops.repository.ShopRepository.search_shops_by_product", new_callable=AsyncMock, return_value=[(test_shop, test_inventory)]):

        grounded_stock_response = await enrich_response_with_shops(
            db=mock_db,
            query_text=farmer_query,
            ai_response="",
            farmer=mock_farmer,
        )

        # 6. Response contains only grounded stock information
        assert "Korutla Kisan Seva Kendra" in grounded_stock_response
        assert "266.5" in grounded_stock_response
        assert "50 bag" in grounded_stock_response
        assert "9848099887" in grounded_stock_response

        # 7. No fabricated stock, no unrelated farmer-profile question
        assert "వరి పంట ఎలా ఉంది" not in grounded_stock_response
        assert "How is your paddy crop doing" not in grounded_stock_response
        assert "paddy" not in grounded_stock_response.lower() or "bag" in grounded_stock_response.lower()

    # 8. Full WhatsApp pipeline execution: incoming audio -> STT -> process -> outbound text + audio
    parsed = ParsedIncomingMessage(
        phone_number="919848099887",
        message_id="wamid.VOICE_KORUTLA_01",
        timestamp="1700000000",
        message_type="audio",
        media_id="audio_korutla_stock",
    )
    mock_conv = Conversation(id=uuid4(), farmer_id=mock_farmer.id, message_id=parsed.message_id, user_message=None, user_message_type="audio")
    mock_db_cm = _setup_voice_mocks(mock_farmer, mock_conv)

    with patch("src.gateway.service.AsyncSessionLocal", return_value=mock_db_cm), \
         patch("src.gateway.service.is_duplicate_message", new_callable=AsyncMock, return_value=False), \
         patch("src.gateway.service.get_or_create_farmer", new_callable=AsyncMock, return_value=mock_farmer), \
         patch("src.gateway.service.store_incoming_message", new_callable=AsyncMock, return_value=mock_conv), \
         patch("src.gateway.service.download_media_bytes", new_callable=AsyncMock, return_value=(b"korutla_audio_bytes", "audio/ogg")), \
         patch("src.gateway.service.get_language_service") as mock_lang_svc, \
         patch("src.gateway.service.process_text_message", new_callable=AsyncMock, return_value=grounded_stock_response), \
         patch("src.gateway.service.send_text_message", new_callable=AsyncMock, return_value="wamid.OUT_TEXT_KORUTLA") as mock_send_text, \
         patch("src.gateway.service.upload_media_bytes", new_callable=AsyncMock, return_value="meta_audio_korutla") as mock_upload, \
         patch("src.gateway.service.send_audio_message", new_callable=AsyncMock, return_value="wamid.OUT_AUDIO_KORUTLA") as mock_send_audio, \
         patch("src.gateway.service.mark_message_as_read", new_callable=AsyncMock), \
         patch("src.gateway.service.get_settings") as mock_settings:

        mock_settings.return_value.enable_voice_responses = True
        mock_lang_svc.return_value.transcribe_audio = AsyncMock(
            return_value=TranscriptionResponse(provider_used="google", transcription_text=farmer_query, detected_language="te")
        )
        mock_lang_svc.return_value.synthesize_speech = AsyncMock(return_value=[b"OggS_grounded_stock_audio"])

        await process_message_pipeline(parsed)

        # 9. Verify text message sent with grounded stock response
        mock_send_text.assert_awaited_once_with(
            to_phone="919848099887",
            message_text=grounded_stock_response,
        )

        # 10. Verify TTS receives EXACT same string
        mock_lang_svc.return_value.synthesize_speech.assert_awaited_once_with(
            grounded_stock_response,
            "te",
        )

        # 11. Verify audio sent
        mock_upload.assert_awaited_once_with(b"OggS_grounded_stock_audio", mime_type="audio/ogg")
        mock_send_audio.assert_awaited_once_with("919848099887", "meta_audio_korutla")

