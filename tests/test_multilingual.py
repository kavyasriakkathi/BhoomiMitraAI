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


def test_marathi_paddy_pest_query_language_detection():
    """Verify Marathi paddy pest query in Devanagari is detected as Marathi ('mr') instead of Hindi."""
    text = "धान पिकावर कीड पडली आहे, कोणती औषध फवारावी?"
    assert detect_language(text) == "mr"



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


# ─────────────────────────────────────────────────────────────────────────────
# 10. Explicit Target Language Directive & Prompt Construction (Step 4)
# ─────────────────────────────────────────────────────────────────────────────

from src.ai.service import build_target_language_directive, AIService
from src.ai.schemas import AIGenerateRequest


@pytest.mark.parametrize(
    "lang_code,expected_name,expected_script",
    [
        ("te", "Telugu", "telu"),
        ("hi", "Hindi", "deva"),
        ("en", "English", "latn"),
        ("ta", "Tamil", "taml"),
        ("kn", "Kannada", "knda"),
        ("ml", "Malayalam", "mlym"),
        ("mr", "Marathi", "deva"),
        ("bn", "Bengali", "beng"),
        ("gu", "Gujarati", "gujr"),
        ("pa", "Punjabi", "guru"),
        ("or", "Odia", "orya"),
        ("as", "Assamese", "beng"),
        ("ur", "Urdu", "arab"),
    ]
)
def test_build_target_language_directive_all_13_languages(lang_code, expected_name, expected_script):
    """Verify target language directive builds cleanly with metadata for all 13 languages."""
    directive = build_target_language_directive(lang_code)
    assert f"[TARGET_RESPONSE_LANGUAGE: {lang_code}]" in directive
    assert f"[TARGET_RESPONSE_LANGUAGE_NAME: {expected_name}]" in directive
    assert f"[TARGET_RESPONSE_LANGUAGE_SCRIPT: {expected_script}]" in directive
    assert "MANDATORY INSTRUCTIONS:" in directive
    assert "Do NOT switch to English" in directive
    assert "Strictly preserve all numbers" in directive


def test_target_language_directive_romanized_inputs():
    """Verify Romanized inputs resolve to their target language directives."""
    # Romanized Hindi (Hinglish) -> Hindi directive
    hi_lang = detect_language("khet me kitna khad daalna chahiye")
    assert hi_lang == "hi"
    hi_directive = build_target_language_directive(hi_lang)
    assert "[TARGET_RESPONSE_LANGUAGE: hi]" in hi_directive
    assert "[TARGET_RESPONSE_LANGUAGE_NAME: Hindi]" in hi_directive

    # Romanized Telugu (Tanglish) -> Telugu directive
    te_lang = detect_language("vari ki em fertilizer vadali")
    assert te_lang == "te"
    te_directive = build_target_language_directive(te_lang)
    assert "[TARGET_RESPONSE_LANGUAGE: te]" in te_directive
    assert "[TARGET_RESPONSE_LANGUAGE_NAME: Telugu]" in te_directive


def test_target_language_directive_hindi_vs_marathi_distinction():
    """Verify Hindi and Marathi produce distinct directives."""
    hi_directive = build_target_language_directive("hi")
    mr_directive = build_target_language_directive("mr")

    assert "[TARGET_RESPONSE_LANGUAGE: hi]" in hi_directive
    assert "[TARGET_RESPONSE_LANGUAGE_NAME: Hindi]" in hi_directive

    assert "[TARGET_RESPONSE_LANGUAGE: mr]" in mr_directive
    assert "[TARGET_RESPONSE_LANGUAGE_NAME: Marathi]" in mr_directive


def test_target_language_directive_bengali_vs_assamese_distinction():
    """Verify Bengali and Assamese produce distinct directives."""
    bn_directive = build_target_language_directive("bn")
    as_directive = build_target_language_directive("as")

    assert "[TARGET_RESPONSE_LANGUAGE: bn]" in bn_directive
    assert "[TARGET_RESPONSE_LANGUAGE_NAME: Bengali]" in bn_directive

    assert "[TARGET_RESPONSE_LANGUAGE: as]" in as_directive
    assert "[TARGET_RESPONSE_LANGUAGE_NAME: Assamese]" in as_directive


def test_target_language_directive_invalid_language_fallback():
    """Verify invalid or unsupported language strings fall back safely to configured default (te)."""
    directive = build_target_language_directive("invalid_lang_xyz", default="te")
    assert "[TARGET_RESPONSE_LANGUAGE: te]" in directive
    assert "[TARGET_RESPONSE_LANGUAGE_NAME: Telugu]" in directive

    directive_none = build_target_language_directive(None, default="en")
    assert "[TARGET_RESPONSE_LANGUAGE: en]" in directive_none
    assert "[TARGET_RESPONSE_LANGUAGE_NAME: English]" in directive_none


def test_target_language_directive_multimodal_format():
    """Verify multimodal image directive enforces JSON friendly_whatsapp_reply language."""
    directive = build_target_language_directive("ta", is_multimodal=True)
    assert "[TARGET_RESPONSE_LANGUAGE: ta]" in directive
    assert "[TARGET_RESPONSE_LANGUAGE_NAME: Tamil]" in directive
    assert "friendly_whatsapp_reply" in directive
    assert "Do NOT switch to English" in directive


@pytest.mark.asyncio
async def test_generate_ai_response_injects_target_language_directive():
    """Verify generate_ai_response injects the target language directive into the Gemini system prompt."""
    mock_repo = MagicMock()
    mock_repo.session = AsyncMock()
    mock_repo.get_farmer_profile = AsyncMock(return_value=Farmer(id=uuid4(), preferred_language="kn"))
    mock_repo.get_conversation_history = AsyncMock(return_value=[])

    captured_prompt = {}

    async def fake_gemini_response(system_prompt, conversation_history, user_message, timeout_seconds):
        captured_prompt["system_prompt"] = system_prompt
        return "ಬೆಳೆಗೆ ಸೂಕ್ತ ಪ್ರಮಾಣದ ರಸಗೊಬ್ಬರವನ್ನು ಬಳಸಿ."

    service = AIService(mock_repo)

    with patch("src.ai.service.generate_response", side_effect=fake_gemini_response), \
         patch("src.memory.service.FarmerMemoryService.format_memory_for_system_prompt", new_callable=AsyncMock, return_value=""), \
         patch("src.memory.service.trigger_background_memory_extraction"):

        req = AIGenerateRequest(
            farmer_id=uuid4(),
            message="ಹತ್ತಿ ಬೆಳೆಯಲ್ಲಿ ಕೀಟ ನಿಯಂತ್ರಣ ಹೇಗೆ?"  # Kannada query
        )
        response = await service.generate_ai_response(req)

        assert response.response_text == "ಬೆಳೆಗೆ ಸೂಕ್ತ ಪ್ರಮಾಣದ ರಸಗೊಬ್ಬರವನ್ನು ಬಳಸಿ."
        sys_prompt = captured_prompt["system_prompt"]
        assert "[TARGET_RESPONSE_LANGUAGE: kn]" in sys_prompt
        assert "[TARGET_RESPONSE_LANGUAGE_NAME: Kannada]" in sys_prompt
        assert "[TARGET_RESPONSE_LANGUAGE_SCRIPT: knda]" in sys_prompt


@pytest.mark.asyncio
async def test_dosage_hard_grounding_gate_preserved():
    """Verify dosage hard-grounding gate blocks ungrounded chemical dosage requests across languages."""
    mock_repo = MagicMock()
    mock_repo.session = AsyncMock()
    mock_repo.get_farmer_profile = AsyncMock(return_value=Farmer(id=uuid4(), preferred_language="hi"))
    mock_repo.get_conversation_history = AsyncMock(return_value=[])

    service = AIService(mock_repo)

    with patch("src.rag.service.RAGService.search_knowledge", new_callable=AsyncMock, return_value=[]), \
         patch("src.memory.service.FarmerMemoryService.format_memory_for_system_prompt", new_callable=AsyncMock, return_value=""):

        req = AIGenerateRequest(
            farmer_id=uuid4(),
            message="धान की फसल में कितना यूरिया डालना चाहिए प्रति एकड़?"  # Hindi dosage query
        )
        response = await service.generate_ai_response(req)

        assert response.provider_used == "hard_grounding_gate"
        assert response.intent == "dosage_unverified_fallback"
        # Hindi fallback response should be returned
        assert "सत्यापित डेटा" in response.response_text or "कृषि विस्तार अधिकारी" in response.response_text


# ─────────────────────────────────────────────────────────────────────────────
# 11. Centralized Downstream Localization (Step 5)
# ─────────────────────────────────────────────────────────────────────────────

from src.ai.formatting import (
    get_section_header,
    get_unavailable_label,
    get_market_labels,
    get_weather_labels,
    get_weather_condition_desc,
    get_schemes_labels,
    get_shops_labels,
    get_escalation_labels,
    compact_section,
    format_multi_intent_response,
)
from src.core.models import GovernmentScheme


ALL_13_LANGUAGES = [
    "te", "hi", "en", "ta", "kn", "ml", "mr", "bn", "gu", "pa", "or", "as", "ur"
]


@pytest.mark.parametrize("lang", ALL_13_LANGUAGES)
def test_market_formatting_all_13_languages(lang):
    """Verify MarketService WhatsApp formatting produces localized labels for all 13 languages."""
    from src.market.service import MarketService

    service = MarketService(MagicMock(), MagicMock())
    item = MarketPriceResponse(
        id=uuid4(),
        commodity="Paddy",
        commodity_telugu="వరి",
        market_name="Enumamula Market",
        district="Warangal",
        state="Telangana",
        min_price=2183.0,
        max_price=2203.0,
        modal_price=2195.0,
        unit="Quintal",
        price_date=datetime(2026, 9, 10),
        source="live_api",
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )

    query_res = MarketPriceQueryResponse(
        commodity="Paddy",
        district="Warangal",
        state="Telangana",
        results=[item],
        data_available=True,
        source_note="Live",
        is_live=True,
    )

    formatted = service.format_whatsapp_reply(query_res, language=lang)
    labels = get_market_labels(lang)

    assert "📊" in formatted
    assert "₹2,195" in formatted
    assert "₹2,183" in formatted
    assert "₹2,203" in formatted
    assert "Enumamula Market" in formatted
    assert labels["market"] in formatted
    assert labels["modal"] in formatted
    assert labels["min"] in formatted
    assert labels["max"] in formatted
    assert labels["source_live"] in formatted


@pytest.mark.parametrize("lang", ALL_13_LANGUAGES)
def test_weather_formatting_all_13_languages(lang):
    """Verify WeatherService WhatsApp formatting produces localized labels for all 13 languages."""
    from src.weather.service import WeatherService

    service = WeatherService(MagicMock())
    forecast_item = WeatherForecastItem(
        dt_txt=(datetime.utcnow() + datetime.resolution).strftime("%Y-%m-%d %H:%M:%S"),
        temp=28.0,
        humidity=75,
        description="Light Rain",
        condition_code=500,
    )

    weather_res = WeatherForecastResponse(
        location_name="Warangal",
        current=WeatherCondition(
            temp=29.4,
            feels_like=31.2,
            humidity=70,
            wind_speed=14.0,
            description="Clear Sky",
            condition_code=800,
        ),
        forecast=[forecast_item],
        data_available=True,
        source_note="OpenWeather",
        is_live=True,
    )

    formatted = service.format_whatsapp_reply(weather_res, language=lang)
    labels = get_weather_labels(lang)

    assert "🌡️" in formatted
    assert "29.4°C" in formatted
    assert "31.2°C" in formatted
    assert "70%" in formatted
    assert "14.0 km/h" in formatted
    assert "Warangal" in formatted
    assert labels["temp"] in formatted
    assert labels["feels_like"] in formatted
    assert labels["humidity"] in formatted
    assert labels["wind"] in formatted


@pytest.mark.parametrize("lang", ALL_13_LANGUAGES)
def test_schemes_formatting_all_13_languages(lang):
    """Verify SchemeService WhatsApp block formatting produces localized labels for all 13 languages."""
    from src.schemes.service import _format_scheme_block

    scheme = GovernmentScheme(
        id=uuid4(),
        scheme_name="PM-KISAN",
        scheme_code="PM_KISAN",
        state="All India",
        category="Income Support",
        benefits_summary="₹6,000 per year in 3 installments",
        eligibility_criteria="Small and marginal landholder farmers",
        required_documents="Aadhaar Card, Land ownership papers, Bank passbook",
        official_portal_url="https://pmkisan.gov.in",
        is_active=True,
    )

    labels = get_schemes_labels(lang)
    block = _format_scheme_block(scheme, labels, language=lang)

    assert "*PM-KISAN*" in block
    assert "₹6,000" in block
    assert "https://pmkisan.gov.in" in block
    assert labels["benefits"] in block
    assert labels["eligibility"] in block
    assert labels["documents"] in block
    assert labels["portal"] in block


@pytest.mark.parametrize("lang", ALL_13_LANGUAGES)
def test_shops_labels_all_13_languages(lang):
    """Verify ShopService localized labels are populated and non-empty across all 13 languages."""
    labels = get_shops_labels(lang)
    assert "🏬" in labels["title"]
    assert "📦" in labels["product"]
    assert "💰" in labels["price"]
    assert "📞" in labels["contact"]
    assert "🚚" in labels["delivery"]
    assert labels["stock_in"]
    assert labels["stock_low"]
    assert labels["stock_out"]
    assert labels["status_open"]
    assert labels["status_closed"]
    assert labels["no_local_dealers"]


@pytest.mark.parametrize("lang", ALL_13_LANGUAGES)
def test_escalation_labels_all_13_languages(lang):
    """Verify EscalationService localized labels are populated and non-empty across all 13 languages."""
    labels = get_escalation_labels(lang)
    assert "👨‍🌾" in labels["header"]
    assert "1800-180-1551" in labels["helpline_number"]
    assert labels["status"]
    assert labels["specialist"]
    assert labels["contact"]
    assert labels["callback"]
    assert labels["callback_time"]
    assert labels["hazard_warning"]


@pytest.mark.parametrize("lang", ALL_13_LANGUAGES)
def test_multi_intent_formatting_all_13_languages(lang):
    """Verify multi-intent section headers adapt to detected language for all 13 languages."""
    raw_ai = "Here is some crop advice about pesticide spray for cotton."
    raw_weather = "🌡️ Weather Information (Warangal)\n🌡️ Temperature: 30°C\n☁️ Condition: Clear"
    raw_market = "📊 Market Prices\nMarket: Warangal Mandi\nModal Price: ₹7,000"

    assembled = f"{raw_ai}\n\n{raw_weather}\n\n{raw_market}"
    user_query = "What pesticide to spray for cotton, and what is the weather and cotton price in Warangal?"

    output = format_multi_intent_response(assembled, user_message=user_query, language=lang)
    assert "🌱" in output
    assert "🌡️" in output
    assert "📊" in output
    header_crop = get_section_header("crop_advice", language=lang)
    header_weather = get_section_header("weather", language=lang)
    assert header_crop in output
    assert header_weather.split()[0] in output


@pytest.mark.parametrize("lang", ALL_13_LANGUAGES)
def test_unavailable_labels_all_13_languages(lang):
    """Verify unavailable messages exist for all sections across all 13 languages."""
    for section in ["weather", "shop", "market", "schemes", "escalation"]:
        unavail = get_unavailable_label(section, language=lang)
        assert unavail and len(unavail) > 10
        assert "ℹ️" in unavail


def test_telugu_and_english_exact_preservation():
    """Verify Telugu and English headers and outputs match expected strings without regression."""
    te_mkt = get_market_labels("te")
    assert te_mkt["modal"] == "మోడల్ ధర"
    assert te_mkt["unit_suffix"] == "క్వింటాల్కు"

    en_mkt = get_market_labels("en")
    assert en_mkt["modal"] == "Modal Price"
    assert en_mkt["unit_suffix"] == "per Quintal"

    te_sec = get_section_header("crop_advice", "te")
    assert te_sec == "🌱 *పంట సలహా*"

    en_sec = get_section_header("crop_advice", "en")
    assert en_sec == "🌱 *Crop Advice*"


def test_invalid_language_fallback_to_english():
    """Verify invalid language code safely falls back to English."""
    mkt = get_market_labels("invalid_lang_code_999")
    assert mkt["modal"] == "Modal Price"

    wtr = get_weather_labels("unknown")
    assert wtr["temp"] == "Temperature"

    sec = get_section_header("crop_advice", "invalid_code")
    assert sec == "🌱 *Crop Advice*"


# ─────────────────────────────────────────────────────────────────────────────
# 12. Multilingual Multi-Intent Detection & Formatting (Step 5 Correction)
# ─────────────────────────────────────────────────────────────────────────────

from src.ai.formatting import detect_user_intents


def test_detect_user_intents_multilingual():
    """Verify detect_user_intents detects discrete intents across all 13 languages."""
    # Hindi: market + weather
    hi_intents = detect_user_intents("कपास का मंडी भाव कितना है और क्या आज बारिश होगी?")
    assert hi_intents["market"] is True
    assert hi_intents["weather"] is True
    assert hi_intents["crop_advice"] is False

    # Tamil: weather + schemes
    ta_intents = detect_user_intents("இன்று மழை பெய்யுமா மற்றும் அரசு திட்டங்கள் என்ன?")
    assert ta_intents["weather"] is True
    assert ta_intents["schemes"] is True
    assert ta_intents["crop_advice"] is False

    # Marathi: market + shops
    mr_intents = detect_user_intents("कापूस बाजार भाव काय आहे आणि खताचे दुकान कुठे आहे?")
    assert mr_intents["market"] is True
    assert mr_intents["shop"] is True
    assert mr_intents["crop_advice"] is False

    # Bengali: market + weather
    bn_intents = detect_user_intents("ধানের বাজার দর কত এবং আজ বৃষ্টি হবে কি?")
    assert bn_intents["market"] is True
    assert bn_intents["weather"] is True
    assert bn_intents["crop_advice"] is False

    # Kannada: market + schemes
    kn_intents = detect_user_intents("ಹತ್ತಿ ಮಾರುಕಟ್ಟೆ ದರ ಎಷ್ಟು ಮತ್ತು ಸರ್ಕಾರಿ ಯೋಜನೆಗಳು ಯಾವುವು?")
    assert kn_intents["market"] is True
    assert kn_intents["schemes"] is True
    assert kn_intents["crop_advice"] is False

    # Gujarati: weather + shops
    gu_intents = detect_user_intents("આજે વરસાદ પડશે અને ખાતરની દુકાન ક્યાં છે?")
    assert gu_intents["weather"] is True
    assert gu_intents["shop"] is True
    assert gu_intents["crop_advice"] is False

    # Punjabi: market + weather
    pa_intents = detect_user_intents("ਕਪਾਹ ਦਾ ਮੰਡੀ ਭਾਅ ਕੀ ਹੈ ਅਤੇ ਕੀ ਮੀਂਹ ਪਵੇਗਾ?")
    assert pa_intents["market"] is True
    assert pa_intents["weather"] is True
    assert pa_intents["crop_advice"] is False

    # Odia: market + schemes
    or_intents = detect_user_intents("କପା ମଣ୍ଡି ଦର କେତେ ଏବଂ ସରକାରୀ ଯୋଜନା କଣ?")
    assert or_intents["market"] is True
    assert or_intents["schemes"] is True
    assert or_intents["crop_advice"] is False

    # Assamese: weather + shops
    as_intents = detect_user_intents("আজি বৰষুণ হ'ব নেকি আৰু সাৰৰ দোকান ক'ত আছে?")
    assert as_intents["weather"] is True
    assert as_intents["shop"] is True
    assert as_intents["crop_advice"] is False

    # Urdu: market + escalation
    ur_intents = detect_user_intents("کپاس کا منڈی ریٹ کیا ہے اور مجھے زرعی افسر سے بات کرنی ہے")
    assert ur_intents["market"] is True
    assert ur_intents["escalation"] is True
    assert ur_intents["crop_advice"] is False

    # Telugu: crop + weather + market
    te_intents = detect_user_intents("పత్తిలో పురుగు నివారణ ఏమిటి మరియు మార్కెట్ ధర, వర్షం పడుతుందా?")
    assert te_intents["crop_advice"] is True
    assert te_intents["weather"] is True
    assert te_intents["market"] is True

    # English: crop + weather + market
    en_intents = detect_user_intents("What is the cotton pest management advice, and what is the mandi price and will it rain today?")
    assert en_intents["crop_advice"] is True
    assert en_intents["weather"] is True
    assert en_intents["market"] is True


def test_multi_intent_formatting_hindi_market_weather():
    """Verify Hindi multi-intent formatting for market + weather."""
    user_msg = "कपास का मंडी भाव कितना है और क्या आज बारिश होगी?"
    raw_weather = "🌡️ *मौसम जानकारी* (वाराणसी)\n🌡️ तापमान: 30°C\n☁️ स्थिति: साफ"
    raw_market = "📊 *कपास मंडी भाव*\nमंडी: वाराणसी मंडी\nऔसत भाव: ₹7,250 प्रति क्विंटल"
    assembled = f"{raw_market}\n\n{raw_weather}"

    output = format_multi_intent_response(assembled, user_message=user_msg, language="hi")

    # Order verification: weather must come before market
    weather_idx = output.find("🌡️")
    market_idx = output.find("📊")
    assert weather_idx != -1
    assert market_idx != -1
    assert weather_idx < market_idx

    # Data preservation
    assert "₹7,250" in output
    assert "30°C" in output
    assert "वाराणसी" in output


def test_multi_intent_formatting_tamil_weather_schemes():
    """Verify Tamil multi-intent formatting for weather + schemes."""
    user_msg = "இன்று மழை பெய்யுமா மற்றும் அரசு திட்டங்கள் என்ன?"
    raw_weather = "🌡️ *வானிலை தகவல்* (மதுரை)\n🌡️ வெப்பநிலை: 32°C\n💧 ஈரப்பதம்: 65%"
    raw_schemes = "🏛️ *அரசு திட்டங்கள்*\n*PM-KISAN*\nபயன்கள்: ₹6,000"
    assembled = f"{raw_schemes}\n\n{raw_weather}"

    output = format_multi_intent_response(assembled, user_message=user_msg, language="ta")

    # Order: weather before schemes
    weather_idx = output.find("🌡️")
    schemes_idx = output.find("🏛️")
    assert weather_idx != -1
    assert schemes_idx != -1
    assert weather_idx < schemes_idx

    # Data & header preservation
    assert "32°C" in output
    assert "₹6,000" in output
    assert "மதுரை" in output


def test_multi_intent_formatting_marathi_market_shops():
    """Verify Marathi multi-intent formatting for market + shops."""
    user_msg = "कापूस बाजार भाव काय आहे आणि खताचे दुकान कुठे आहे?"
    raw_market = "📊 *कापूस बाजार भाव*\nबाजार: नागपूर\nऔसत दर: ₹7,100 प्रति क्विंटल"
    raw_shop = "🏬 *जवळची कृषी दुकाने*\n*किसान अ‍ॅग्रो*\nफोन: 9876543210"
    assembled = f"{raw_market}\n\n{raw_shop}"

    output = format_multi_intent_response(assembled, user_message=user_msg, language="mr")

    # Order: shop before market
    shop_idx = output.find("🏬")
    market_idx = output.find("📊")
    assert shop_idx != -1
    assert market_idx != -1
    assert shop_idx < market_idx

    assert "₹7,100" in output
    assert "9876543210" in output
    assert "नागपूर" in output


def test_multi_intent_formatting_bengali_market_weather():
    """Verify Bengali multi-intent formatting for market + weather."""
    user_msg = "ধানের বাজার দর কত এবং আজ বৃষ্টি হবে কি?"
    raw_weather = "🌡️ *আবহাওয়ার তথ্য* (বর্ধমান)\n🌡️ তাপমাত্রা: 29°C"
    raw_market = "📊 *ধান বাজার দর*\nবাজার: বর্ধমান মান্ডি\nগড় দর: ₹2,150 প্রতি কুইন্টাল"
    assembled = f"{raw_market}\n\n{raw_weather}"

    output = format_multi_intent_response(assembled, user_message=user_msg, language="bn")

    # Order: weather before market
    weather_idx = output.find("🌡️")
    market_idx = output.find("📊")
    assert weather_idx != -1
    assert market_idx != -1
    assert weather_idx < market_idx

    assert "29°C" in output
    assert "₹2,150" in output
    assert "বর্ধমান" in output


def test_multi_intent_formatting_kannada_market_schemes():
    """Verify Kannada multi-intent formatting for market + schemes."""
    user_msg = "ಹತ್ತಿ ಮಾರುಕಟ್ಟೆ ದರ ಎಷ್ಟು ಮತ್ತು ಸರ್ಕಾರಿ ಯೋಜನೆಗಳು ಯಾವುವು?"
    raw_market = "📊 *ಹತ್ತಿ ಮಾರುಕಟ್ಟೆ ದರಗಳು*\nಮಾರುಕಟ್ಟೆ: ರಾಯಚೂರು\nಮಾದರಿ ಬೆಲೆ: ₹7,300"
    raw_schemes = "🏛️ *ಸರ್ಕಾರಿ ಯೋಜನೆಗಳು*\n*PM-KISAN*\nಲಾಭಗಳು: ₹6,000"
    assembled = f"{raw_schemes}\n\n{raw_market}"

    output = format_multi_intent_response(assembled, user_message=user_msg, language="kn")

    # Order: market before schemes
    market_idx = output.find("📊")
    schemes_idx = output.find("🏛️")
    assert market_idx != -1
    assert schemes_idx != -1
    assert market_idx < schemes_idx

    assert "₹7,300" in output
    assert "₹6,000" in output
    assert "ರಾಯಚೂರು" in output


def test_multi_intent_formatting_gujarati_weather_shops():
    """Verify Gujarati multi-intent formatting for weather + shops."""
    user_msg = "આજે વરસાદ પડશે અને ખાતરની દુકાન ક્યાં છે?"
    raw_weather = "🌡️ *હવામાન માહિતી* (રાજકોટ)\n🌡️ તાપમાન: 31°C"
    raw_shop = "🏬 *નજીકની કૃષિ દુકાનો*\n*ગુજરાત એગ્રો*\nફોન: 9876543211"
    assembled = f"{raw_shop}\n\n{raw_weather}"

    output = format_multi_intent_response(assembled, user_message=user_msg, language="gu")

    # Order: weather before shop
    weather_idx = output.find("🌡️")
    shop_idx = output.find("🏬")
    assert weather_idx != -1
    assert shop_idx != -1
    assert weather_idx < shop_idx

    assert "31°C" in output
    assert "9876543211" in output
    assert "રાજકોટ" in output


def test_multi_intent_formatting_punjabi_market_weather():
    """Verify Punjabi multi-intent formatting for market + weather."""
    user_msg = "ਕਪਾਹ ਦਾ ਮੰਡੀ ਭਾਅ ਕੀ ਹੈ ਅਤੇ ਕੀ ਮੀਂਹ ਪਵੇਗਾ?"
    raw_weather = "🌡️ *ਮੌਸਮ ਜਾਣਕਾਰੀ* (ਬਠਿੰਡਾ)\n🌡️ ਤਾਪਮਾਨ: 33°C"
    raw_market = "📊 *ਕਪਾਹ ਮੰਡੀ ਭਾਅ*\nਮੰਡੀ: ਬਠਿੰਡਾ ਮੰਡੀ\nਔਸਤ ਭਾਅ: ₹7,400 ਪ੍ਰਤੀ ਕੁਇੰਟਲ"
    assembled = f"{raw_market}\n\n{raw_weather}"

    output = format_multi_intent_response(assembled, user_message=user_msg, language="pa")

    # Order: weather before market
    weather_idx = output.find("🌡️")
    market_idx = output.find("📊")
    assert weather_idx != -1
    assert market_idx != -1
    assert weather_idx < market_idx

    assert "33°C" in output
    assert "₹7,400" in output
    assert "ਬਠਿੰਡਾ" in output


def test_multi_intent_formatting_odia_market_schemes():
    """Verify Odia multi-intent formatting for market + schemes."""
    user_msg = "କପା ମଣ୍ଡି ଦର କେତେ ଏବଂ ସରକାରୀ ଯୋଜନା କଣ?"
    raw_market = "📊 *କପା ମଣ୍ଡି ଦର*\nମଣ୍ଡି: ବଲାଙ୍ଗୀର\nହାରାହାରି ଦର: ₹7,200 ପ୍ରତି କ୍ୱିଣ୍ଟାଲ"
    raw_schemes = "🏛️ *ସରକାରୀ ଯୋଜନା*\n*କାଳିଆ ଯୋଜନା*\nସହାୟତା: ₹10,000"
    assembled = f"{raw_schemes}\n\n{raw_market}"

    output = format_multi_intent_response(assembled, user_message=user_msg, language="or")

    # Order: market before schemes
    market_idx = output.find("📊")
    schemes_idx = output.find("🏛️")
    assert market_idx != -1
    assert schemes_idx != -1
    assert market_idx < schemes_idx

    assert "₹7,200" in output
    assert "₹10,000" in output
    assert "ବଲାଙ୍ଗୀର" in output


def test_multi_intent_formatting_assamese_weather_shops():
    """Verify Assamese multi-intent formatting for weather + shops."""
    user_msg = "আজি বৰষুণ হ'ব নেকি আৰু সাৰৰ দোকান ক'ত আছে?"
    raw_weather = "🌡️ *বতৰৰ তথ্য* (যোৰহাট)\n🌡️ উষ্ণতা: 28°C"
    raw_shop = "🏬 *ওচৰৰ কৃষি দোকান*\n*অসম এগ্রো*\nফোন: 9876543212"
    assembled = f"{raw_shop}\n\n{raw_weather}"

    output = format_multi_intent_response(assembled, user_message=user_msg, language="as")

    # Order: weather before shop
    weather_idx = output.find("🌡️")
    shop_idx = output.find("🏬")
    assert weather_idx != -1
    assert shop_idx != -1
    assert weather_idx < shop_idx

    assert "28°C" in output
    assert "9876543212" in output
    assert "যোৰহাট" in output


def test_multi_intent_formatting_urdu_market_escalation():
    """Verify Urdu multi-intent formatting for market + escalation."""
    user_msg = "کپاس کا منڈی ریٹ کیا ہے اور مجھے زرعی افسر سے بات کرنی ہے"
    raw_market = "📊 *کپاس منڈی کے نرخ*\nمنڈی: ملتان\nاوسط قیمت: ₹7,500 فی کوئنٹل"
    raw_escalation = "👨‍🌾 *زرعی افسر سے رابطہ*\nایک ماہر زرعی افسر آپ سے رابطہ کرے گا۔\nہیلپ لائن: 1800-180-1551"
    assembled = f"{raw_escalation}\n\n{raw_market}"

    output = format_multi_intent_response(assembled, user_message=user_msg, language="ur")

    # Order: market before escalation
    market_idx = output.find("📊")
    escalation_idx = output.find("👨‍🌾")
    assert market_idx != -1
    assert escalation_idx != -1
    assert market_idx < escalation_idx

    assert "₹7,500" in output
    assert "1800-180-1551" in output
    assert "ملتان" in output


def test_multi_intent_formatting_telugu_regression():
    """Verify Telugu multi-intent formatting regression: crop advice + weather + market."""
    user_msg = "పత్తిలో పురుగు నివారణ ఏమిటి మరియు మార్కెట్ ధర, వర్షం పడుతుందా?"
    raw_crop = "పత్తి పంటలో గులాబీ రంగు పురుగు నివారణకు ఎమామెక్టిన్ బెంజోయేట్ 5% SG 80 గ్రాములు పిచికారీ చేయండి."
    raw_weather = "🌡️ *వాతావరణ సమాచారం* (వరంగల్)\n🌡️ ఉష్ణోగ్రత: 31°C\n☁️ వాతావరణం: స్పష్టమైన ఆకాశం"
    raw_market = "📊 *పత్తి మార్కెట్ ధరలు*\nమండి: వరంగల్ మార్కెట్\nమోడల్ ధర: ₹7,200 క్వింటాల్కు"
    assembled = f"{raw_crop}\n\n{raw_market}\n\n{raw_weather}"

    output = format_multi_intent_response(assembled, user_message=user_msg, language="te")

    # Order: crop_advice -> weather -> market
    crop_idx = output.find("🌱")
    weather_idx = output.find("🌡️")
    market_idx = output.find("📊")
    assert crop_idx != -1
    assert weather_idx != -1
    assert market_idx != -1
    assert crop_idx < weather_idx < market_idx

    assert "🌱 *పంట సలహా*" in output
    assert "ఎమామెక్టిన్ బెంజోయేట్" in output
    assert "31°C" in output
    assert "₹7,200" in output


def test_multi_intent_formatting_english_regression():
    """Verify English multi-intent formatting regression: crop advice + weather + market."""
    user_msg = "What is the cotton pest management advice, and what is the mandi price and will it rain today?"
    raw_crop = "For bollworm management in cotton, spray Emamectin Benzoate 5% SG at 80g per acre."
    raw_weather = "🌡️ *Weather Information* (Warangal)\n🌡️ Temperature: 31°C\n☁️ Condition: Clear"
    raw_market = "📊 *Cotton Mandi Prices*\nMarket: Warangal Mandi\nModal Price: ₹7,200 per Quintal"
    assembled = f"{raw_crop}\n\n{raw_market}\n\n{raw_weather}"

    output = format_multi_intent_response(assembled, user_message=user_msg, language="en")

    # Order: crop_advice -> weather -> market
    crop_idx = output.find("🌱")
    weather_idx = output.find("🌡️")
    market_idx = output.find("📊")
    assert crop_idx != -1
    assert weather_idx != -1
    assert market_idx != -1
    assert crop_idx < weather_idx < market_idx

    assert "🌱 *Crop Advice*" in output
    assert "Emamectin Benzoate" in output
    assert "31°C" in output
    assert "₹7,200" in output
