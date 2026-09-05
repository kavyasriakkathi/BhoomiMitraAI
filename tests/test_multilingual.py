"""
BhoomiMitra AI — Multilingual Test Suite

Comprehensive tests validating:
1. Centralized language configuration & metadata for 13 supported languages.
2. Fast deterministic language detection across all 13 languages (native script).
3. Romanized / Tanglish / Hinglish / Kanglish / Tamlish input detection.
4. Mixed-language inputs.
5. Graceful fallback on uncertain / empty inputs.
6. Voice STT transcription integration with language detection.
7. Decision Engine greetings and intent routing across languages.
8. Backward compatibility and preservation of Telugu, English, and Tanglish behavior.
9. Authoritative module formatting in detected languages (Market, Weather, Schemes, Shops, Escalation).
10. Safety and zero leakage of internal codes, stack traces, or developer terms.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4
from datetime import datetime

from src.language.languages import (
    SUPPORTED_LANGUAGES,
    LanguageMetadata,
    DEFAULT_LANGUAGE,
    FALLBACK_LANGUAGE,
    get_language,
    is_supported_language,
    normalize_language_code,
    list_supported_languages,
)
from src.language.detector import (
    detect_language,
    detect_language_with_confidence,
)
from src.ai.prompts import (
    get_fallback_response,
    get_voice_fallback_response,
    get_image_fallback_response,
    get_market_fallback_response,
    get_weather_fallback_response,
    get_schemes_fallback_response,
    get_shops_fallback_response,
    get_unsupported_media_fallback_response,
    get_non_crop_image_response,
)
from src.ai.decision_engine import AIDecisionEngine, FarmerIntent
from src.market.schemas import MarketPriceQueryResponse, MarketPriceResponse
from src.weather.schemas import WeatherForecastResponse, WeatherCondition, WeatherForecastItem
from src.core.models import Farmer, Conversation


# ─────────────────────────────────────────────────────────────────────────────
# 1. Centralized Language Configuration Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_centralized_language_configuration():
    """Verify all 13 supported languages are defined with complete metadata."""
    expected_codes = [
        "te", "hi", "en", "ta", "kn", "ml", "mr", "bn", "gu", "or", "pa", "as", "ur"
    ]
    assert len(SUPPORTED_LANGUAGES) == 13

    for code in expected_codes:
        assert code in SUPPORTED_LANGUAGES, f"Missing code: {code}"
        meta = SUPPORTED_LANGUAGES[code]
        assert meta.code == code
        assert meta.display_name
        assert meta.native_name
        assert meta.prompt_name
        assert meta.script
        assert meta.stt_code.endswith("-IN")
        assert meta.supported is True

    # Check helpers
    assert is_supported_language("te") is True
    assert is_supported_language("hi") is True
    assert is_supported_language("fr") is False
    assert get_language("hi-IN").code == "hi"
    assert normalize_language_code("kn") == "kn"
    assert normalize_language_code("invalid_code", default="en") == "en"
    assert len(list_supported_languages()) == 13


# ─────────────────────────────────────────────────────────────────────────────
# 2. Language Detection across all 13 Native Scripts
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "text,expected_lang",
    [
        ("పత్తి పంటలో గులాబీ రంగు పురుగు నివారణకు ఏ మందు వాడాలి?", "te"),  # Telugu
        ("कपास की फसल में गुलाबी सुंडी के नियंत्रण के लिए कौन सी दवा का छिड़काव करें?", "hi"),  # Hindi
        ("What is the recommended fertilizer schedule for cotton in black soil?", "en"),  # English
        ("பருத்தி பயிரில் புழு தாக்குதலை கட்டுப்படுத்த என்ன மருந்து அடிக்க வேண்டும்?", "ta"),  # Tamil
        ("ಹತ್ತಿ ಬೆಳೆಯಲ್ಲಿ ಕೀಟ ಬಾಧೆ ನಿಯಂತ್ರಣಕ್ಕೆ ಯಾವ ಔಷಧ ಸಿಂಪಡಿಸಬೇಕು?", "kn"),  # Kannada
        ("പരുത്തി കൃഷിയിൽ പുഴുക്കളെ നിയന്ത്രിക്കാൻ ഏത് മരുന്നാണ് തളിക്കേണ്ടത്?", "ml"),  # Malayalam
        ("कापूस पिकामध्ये बोंडअळीच्या नियंत्रणासाठी कोणते औषध फवारावे?", "mr"),  # Marathi
        ("তুলা ফসলে পোকা নিয়ন্ত্রণের জন্য কোন ওষুধ স্প্রে করতে হবে?", "bn"),  # Bengali
        ("કપાસના પાકમાં ગુલાબી ઈયળના નિયંત્રણ માટે કઈ દવાનો છંટકાવ કરવો?", "gu"),  # Gujarati
        ("କପା ଫସଲରେ ପୋକ ନିୟନ୍ତ୍ରଣ ପାଇଁ କେଉଁ ଔଷଧ ସ୍ପ୍ରେ କରିବା ଉଚିତ୍?", "or"),  # Odia
        ("ਕਪਾਹ ਦੀ ਫਸਲ ਵਿੱਚ ਸੁੰਡੀ ਦੀ ਰੋਕਥਾਮ ਲਈ ਕਿਹੜੀ ਦਵਾਈ ਦਾ ਛਿੜਕਾਅ ਕਰਨਾ ਚਾਹੀਦਾ ਹੈ?", "pa"),  # Punjabi
        ("কপাহ খেতিত পোক-পৰুৱা নিয়ন্ত্ৰণৰ বাবে কি ঔষধ ব্যৱহাৰ কৰিব লাগে?", "as"),  # Assamese
        ("کپاس کی فصل میں کیڑوں کی روک تھام کے لیے کون سی دوا کا اسپرے کریں؟", "ur"),  # Urdu
    ]
)
def test_detect_language_native_scripts(text, expected_lang):
    detected = detect_language(text)
    assert detected == expected_lang


# ─────────────────────────────────────────────────────────────────────────────
# 3. Romanized / Transliterated Indian Language Detection
# ─────────────────────────────────────────────────────────────────────────────

def test_detect_language_romanized_tanglish():
    """Test Romanized Tanglish queries mapping to Telugu (te)."""
    assert detect_language("vari ki em fertilizer vadali") == "te"
    assert detect_language("patti crop lo purugula mandu eppudu spray cheyali") == "te"
    assert detect_language("eeroju warangal mandi lo cotton rate entha undi") == "te"
    assert detect_language("repu varsham paduthunda") == "te"


def test_detect_language_romanized_hinglish():
    """Test Romanized Hinglish queries mapping to Hindi (hi)."""
    assert detect_language("kapas ko kitna pani chahiye") == "hi"
    assert detect_language("khet me kitna khad daalna chahiye") == "hi"
    assert detect_language("aaj mandi bhav kitna hai") == "hi"
    assert detect_language("fasal me keeda laga hai kya kare") == "hi"


def test_detect_language_romanized_kanglish():
    """Test Romanized Kanglish queries mapping to Kannada (kn)."""
    assert detect_language("nellu crop ge neeru eshtu") == "kn"
    assert detect_language("bele ge gobbara yavaga hakabeku") == "kn"


def test_detect_language_romanized_tamlish():
    """Test Romanized Tamlish queries mapping to Tamil (ta)."""
    assert detect_language("nellu payir ku uram eppadi podanum") == "ta"
    assert detect_language("thanni eppadi paaikkanum") == "ta"


# ─────────────────────────────────────────────────────────────────────────────
# 4. Mixed-Language Input & Graceful Fallback
# ─────────────────────────────────────────────────────────────────────────────

def test_mixed_language_input():
    """Test mixed English and Indic script inputs preserve primary Indic language."""
    assert detect_language("Warangal lo cotton mandi rate ఎంత ఉంది?") == "te"
    assert detect_language("Urea fertilizer kitna daalna hai फसल में?") == "hi"
    assert detect_language("NPK dose for நெல் பயிர்?") == "ta"


def test_uncertain_language_fallback():
    """Test empty, numeric, or unclassifiable inputs gracefully fallback to English/default."""
    assert detect_language("") == "en"
    assert detect_language("   ") == "en"
    assert detect_language("12345 67890", fallback="en") == "en"
    assert detect_language("??? !!! ...", fallback="en") == "en"
    assert detect_language("xyz abc 999", fallback="te") == "te"

    code, conf, method = detect_language_with_confidence("12345")
    assert method == "fallback"
    assert conf == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 5. Multilingual Fallback Responses Integrity
# ─────────────────────────────────────────────────────────────────────────────

def test_multilingual_fallback_responses():
    """Verify fallback responses for all 13 languages are non-empty and localized."""
    for code in SUPPORTED_LANGUAGES:
        fb = get_fallback_response(code)
        assert fb and len(fb) > 10
        assert "🙏" in fb

        v_fb = get_voice_fallback_response(code)
        assert v_fb and len(v_fb) > 10

        img_fb = get_image_fallback_response(code)
        assert img_fb and len(img_fb) > 10

        mkt_fb = get_market_fallback_response(code)
        assert mkt_fb and len(mkt_fb) > 10

        wtr_fb = get_weather_fallback_response(code)
        assert wtr_fb and len(wtr_fb) > 10

        sch_fb = get_schemes_fallback_response(code)
        assert sch_fb and len(sch_fb) > 10

        shp_fb = get_shops_fallback_response(code)
        assert shp_fb and len(shp_fb) > 10

        unsupp_fb = get_unsupported_media_fallback_response(code)
        assert unsupp_fb and len(unsupp_fb) > 10

        non_crop_fb = get_non_crop_image_response(code)
        assert non_crop_fb and len(non_crop_fb) > 10


# ─────────────────────────────────────────────────────────────────────────────
# 6. Decision Engine Multilingual Greetings & Intent Classification
# ─────────────────────────────────────────────────────────────────────────────

def test_decision_engine_greetings_all_languages():
    """Verify greeting replies in all 13 languages."""
    engine = AIDecisionEngine()

    assert "భూమిమిత్ర" in engine.get_greeting_reply("te")
    assert "भूमिमित्र" in engine.get_greeting_reply("hi")
    assert "BhoomiMitra" in engine.get_greeting_reply("en")
    assert "பூமிமித்ரா" in engine.get_greeting_reply("ta")
    assert "ಭೂಮಿಮಿತ್ರ" in engine.get_greeting_reply("kn")
    assert "ഭൂമിമിത്ര" in engine.get_greeting_reply("ml")
    assert "भूमिमित्र" in engine.get_greeting_reply("mr")
    assert "ভূমিমিত্র" in engine.get_greeting_reply("bn")
    assert "ભૂમિમિત્ર" in engine.get_greeting_reply("gu")
    assert "ଭୂମିମିତ୍ର" in engine.get_greeting_reply("or")
    assert "ਭੂਮੀਮਿੱਤਰ" in engine.get_greeting_reply("pa")
    assert "ভূমিমিত্ৰ" in engine.get_greeting_reply("as")
    assert "بھومی مترا" in engine.get_greeting_reply("ur")


def test_is_greeting_only_multilingual():
    """Verify greeting-only detection works across multilingual greetings."""
    assert AIDecisionEngine.is_greeting_only("నమస్తే") is True
    assert AIDecisionEngine.is_greeting_only("नमस्ते") is True
    assert AIDecisionEngine.is_greeting_only("Hello") is True
    assert AIDecisionEngine.is_greeting_only("வணக்கம்") is True
    assert AIDecisionEngine.is_greeting_only("ನಮಸ್ಕಾರ") is True
    assert AIDecisionEngine.is_greeting_only("നമസ്കാരം") is True
    assert AIDecisionEngine.is_greeting_only("নমস্কার") is True
    assert AIDecisionEngine.is_greeting_only("سلام") is True
    assert AIDecisionEngine.is_greeting_only("Hi BhoomiMitra") is True

    # Greetings with questions must NOT be greeting only
    assert AIDecisionEngine.is_greeting_only("नमस्ते, कपास का भाव क्या है?") is False
    assert AIDecisionEngine.is_greeting_only("నమస్తే వాతావరణం ఎలా ఉంది?") is False


def test_multilingual_intent_classification():
    """Verify intent classification across various languages."""
    # Market Price
    assert AIDecisionEngine.detect_primary_intent("કપાસનો ભાવ શું છે?") == FarmerIntent.MARKET_PRICE
    assert AIDecisionEngine.detect_primary_intent("காய்கறி சந்தை விலை என்ன?") == FarmerIntent.MARKET_PRICE
    assert AIDecisionEngine.detect_primary_intent("कपास का मंडी भाव कितना है?") == FarmerIntent.MARKET_PRICE

    # Weather
    assert AIDecisionEngine.detect_primary_intent("आज मौसम कैसा रहेगा बारिश होगी क्या?") == FarmerIntent.WEATHER
    assert AIDecisionEngine.detect_primary_intent("ಇಂದು ಮಳೆ ಬರುತ್ತಾ?") == FarmerIntent.WEATHER
    assert AIDecisionEngine.detect_primary_intent("இன்று மழை பெய்யுமா?") == FarmerIntent.WEATHER

    # Schemes
    assert AIDecisionEngine.detect_primary_intent("पीएम किसान योजना की जानकारी दीजिए") == FarmerIntent.GOVERNMENT_SCHEMES
    assert AIDecisionEngine.detect_primary_intent("రైతు భరోసా పథకం అర్హత ఏమిటి?") == FarmerIntent.GOVERNMENT_SCHEMES

    # Shops
    assert AIDecisionEngine.detect_primary_intent("यूरिया खाद की दुकान कहाँ है?") == FarmerIntent.SHOPS
    assert AIDecisionEngine.detect_primary_intent("ఎరువుల దుకాణం ఎక్కడ ఉంది?") == FarmerIntent.SHOPS


# ─────────────────────────────────────────────────────────────────────────────
# 7. Authoritative Market & Weather Formatting in Detected Languages
# ─────────────────────────────────────────────────────────────────────────────

def test_market_formatting_in_detected_languages():
    """Verify market prices block adapts labels to detected language without inventing data."""
    from src.market.service import MarketService

    mock_repo = MagicMock()
    mock_client = MagicMock()
    service = MarketService(mock_repo, mock_client)

    item = MarketPriceResponse(
        id=uuid4(),
        commodity="Cotton",
        commodity_telugu="పత్తి",
        market_name="Warangal Mandi",
        district="Warangal",
        state="Telangana",
        min_price=7000.0,
        max_price=7500.0,
        modal_price=7250.0,
        unit="Quintal",
        price_date=datetime(2026, 9, 5),
        source="local_db",
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )

    query_res = MarketPriceQueryResponse(
        commodity="Cotton",
        district="Warangal",
        state="Telangana",
        results=[item],
        data_available=True,
        source_note="Test",
        is_live=True,
    )

    # Telugu formatting
    te_reply = service.format_whatsapp_reply(query_res, language="te")
    assert "📊" in te_reply
    assert "మార్కెట్ ధరలు" in te_reply or "పత్తి" in te_reply
    assert "₹7,250" in te_reply

    # Hindi formatting
    hi_reply = service.format_whatsapp_reply(query_res, language="hi")
    assert "📊" in hi_reply
    assert "मंडी भाव" in hi_reply or "Cotton" in hi_reply or "कपास" in hi_reply
    assert "₹7,250" in hi_reply
    assert "औसत भाव" in hi_reply

    # Tamil formatting
    ta_reply = service.format_whatsapp_reply(query_res, language="ta")
    assert "📊" in ta_reply
    assert "சந்தை விலைகள்" in ta_reply or "Cotton" in ta_reply
    assert "₹7,250" in ta_reply

    # Kannada formatting
    kn_reply = service.format_whatsapp_reply(query_res, language="kn")
    assert "📊" in kn_reply
    assert "ಮಾರುಕಟ್ಟೆ ದರಗಳು" in kn_reply or "Cotton" in kn_reply
    assert "₹7,250" in kn_reply


def test_weather_formatting_in_detected_languages():
    """Verify weather forecast block adapts labels to detected language without inventing data."""
    from src.weather.service import WeatherService

    mock_client = MagicMock()
    service = WeatherService(mock_client)

    forecast_item = WeatherForecastItem(
        dt_txt="2026-09-06 12:00:00",
        temp=30.0,
        humidity=65,
        description="Light Rain",
        condition_code=500,
    )

    weather_res = WeatherForecastResponse(
        location_name="Warangal",
        current=WeatherCondition(
            temp=32.5,
            feels_like=34.0,
            humidity=60,
            wind_speed=12.5,
            description="Clear Sky",
            condition_code=800,
        ),
        forecast=[forecast_item],
        data_available=True,
        source_note="Test",
        is_live=True,
    )

    # Telugu formatting
    te_reply = service.format_whatsapp_reply(weather_res, language="te")
    assert "🌡️" in te_reply
    assert "వాతావరణ సమాచారం" in te_reply
    assert "32.5°C" in te_reply

    # Hindi formatting
    hi_reply = service.format_whatsapp_reply(weather_res, language="hi")
    assert "🌡️" in hi_reply
    assert "मौसम जानकारी" in hi_reply
    assert "32.5°C" in hi_reply
    assert "तापमान" in hi_reply

    # Tamil formatting
    ta_reply = service.format_whatsapp_reply(weather_res, language="ta")
    assert "🌡️" in ta_reply
    assert "வானிலை தகவல்" in ta_reply
    assert "32.5°C" in ta_reply


# ─────────────────────────────────────────────────────────────────────────────
# 8. Voice STT followed by Language Detection Integration
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_voice_stt_and_language_detection():
    """Verify voice transcription result accurately refines detected language."""
    from src.language.service import LanguageService
    from src.language.schemas import TranscriptionResponse

    service = LanguageService()

    # Mock Google STT client
    mock_alt = MagicMock()
    mock_alt.transcript = "పత్తి పంటలో గులాబీ రంగు పురుగు నివారణకు ఏ మందు వాడాలి"
    mock_alt.confidence = 0.96

    mock_result = MagicMock()
    mock_result.alternatives = [mock_alt]
    mock_result.language_code = "te-IN"

    mock_response = MagicMock()
    mock_response.results = [mock_result]

    service._google_client = MagicMock()
    service._google_client.recognize.return_value = mock_response

    resp = await service._transcribe_with_google(b"fake_audio_bytes", "audio/ogg")
    assert resp.transcription_text == "పత్తి పంటలో గులాబీ రంగు పురుగు నివారణకు ఏ మందు వాడాలి"
    assert resp.detected_language == "te"
    assert resp.confidence == 0.96


# ─────────────────────────────────────────────────────────────────────────────
# 9. Safety & Zero Internal Detail Leakage
# ─────────────────────────────────────────────────────────────────────────────

def test_no_internal_code_leakage():
    """Ensure internal model names, API keys, or technical jargon are absent from farmer messages."""
    for code in SUPPORTED_LANGUAGES:
        msg = get_fallback_response(code)
        assert "gemini" not in msg.lower()
        assert "model" not in msg.lower()
        assert "http" not in msg.lower()
        assert "500" not in msg
        assert "error" not in msg.lower()
        assert "exception" not in msg.lower()
        assert "traceback" not in msg.lower()
