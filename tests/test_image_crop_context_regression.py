"""
BhoomiMitra AI — Regression Tests for Image Diagnosis Context Contamination Fix

Verifies:
1. TEST 1: Previous message = paddy yellow leaves, Current = image-only, Old history has cotton.
   Expected: Image prompt prioritizes paddy context and filters out cotton history.
2. TEST 2: Previous message = cotton disease, Current = image-only.
   Expected: Cotton context is used.
3. TEST 3: No current crop context, Old history contains cotton & paddy, Current = image-only.
   Expected: Does not select cotton from old history; asks for crop clarification.
4. TEST 4: Current crop = paddy, Generated response mentions cotton.
   Expected: Crop consistency guard triggers second validation/reconsideration step.
5. TEST 5: Normal text query remains unchanged.
6. TEST 6: Voice query remains unchanged.
7. TEST 7: Image response produces exactly ONE ai_response reused for text and TTS.
"""

import json
import uuid
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.core.models import Farmer, Conversation
from src.ai.service import process_image_message, process_text_message
from src.gateway.schemas import ParsedIncomingMessage
from src.gateway.service import process_message_pipeline
from src.ai.crop_context import (
    extract_crop_mention,
    extract_all_crop_mentions,
    resolve_image_crop_context,
    filter_history_for_vision,
    detect_crop_mismatch,
    build_crop_reconsideration_instruction,
)
from src.ai.prompts import get_crop_clarification_response


@pytest.fixture
def sample_farmer():
    return Farmer(
        id=uuid.uuid4(),
        phone_number="917989271932",
        preferred_language="te",
    )


@pytest.fixture
def mock_db_session():
    mock_db = AsyncMock()
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = None
    mock_result.scalars.return_value.all.return_value = []
    mock_db.execute.return_value = mock_result
    return mock_db


# ─────────────────────────────────────────────────────────────────────────────
# Unit Tests for Crop Context Helpers
# ─────────────────────────────────────────────────────────────────────────────

def test_extract_crop_mention_paddy():
    assert extract_crop_mention("నా వరి పంటలో ఆకులు పసుపుగా మారుతున్నాయి") == "paddy"
    assert extract_crop_mention("paddy crop has yellow leaves") == "paddy"
    assert extract_crop_mention("వరిలో ఎరువులు ఎంత?") == "paddy"


def test_extract_crop_mention_cotton():
    assert extract_crop_mention("నా పత్తి ఆకులపై మచ్చలు ఉన్నాయి") == "cotton"
    assert extract_crop_mention("cotton price in warangal") == "cotton"
    assert extract_crop_mention("పత్తికి ఏ ఎరువు మంచిది?") == "cotton"


def test_extract_crop_mention_color_not_turmeric():
    # 'పసుపుగా' is yellow color adjective, NOT turmeric crop
    assert extract_crop_mention("ఆకులు పసుపుగా మారుతున్నాయి") is None
    # Real turmeric crop mention
    assert extract_crop_mention("పసుపు పంట సాగు విధానం") == "turmeric"


def test_resolve_image_crop_context_from_previous_message():
    history = [
        Conversation(
            id=uuid.uuid4(),
            user_message="నా వరి పంటలో ఆకులు పసుపుగా మారుతున్నాయి",
            ai_response="వరి ఆకులు పసుపుగా మారడానికి...",
        ),
        Conversation(
            id=uuid.uuid4(),
            user_message="నా పత్తి ఆకులపై మచ్చలు ఉన్నాయి",
            ai_response="పత్తి పంటలో ఆల్టర్నేరియా...",
        ),
    ]
    ctx = resolve_image_crop_context(user_caption=None, history_records=history)
    assert ctx is not None
    assert ctx["crop"] == "paddy"
    assert ctx["crop_display_en"] == "paddy/rice"
    assert "వరి" in ctx["farmer_message"]


def test_filter_history_for_vision_removes_conflicting_crops():
    records = [
        Conversation(id=uuid.uuid4(), user_message="నా వరి పంటలో తెగులు", ai_response="వరి తెగులు..."),
        Conversation(id=uuid.uuid4(), user_message="కోరుట్లలో వర్షం పడుతుందా?", ai_response="వాతావరణం..."),
        Conversation(id=uuid.uuid4(), user_message="నా పత్తి ఆకులపై మచ్చలు ఉన్నాయి", ai_response="పత్తిలో ఆల్టర్నేరియా..."),
        Conversation(id=uuid.uuid4(), user_message="వరంగల్లో పత్తి ధర ఎంత?", ai_response="పత్తి ధర..."),
    ]
    filtered = filter_history_for_vision(records, active_crop="paddy", max_turns=4)
    # The two cotton records must be excluded
    assert len(filtered) == 2
    assert "వరి" in filtered[0].user_message
    assert "వర్షం" in filtered[1].user_message


def test_detect_crop_mismatch():
    # Active crop is paddy, but diagnosis discusses cotton
    mismatch = detect_crop_mismatch(
        active_crop="paddy",
        response_text="మీ పత్తి ఆకును పరిశీలిస్తే బాక్టీరియల్ ఆకుమచ్చ తెగులు...",
        disease_name="Cotton Bacterial Blight",
    )
    assert mismatch == "cotton"

    # Active crop is paddy, diagnosis discusses paddy
    no_mismatch = detect_crop_mismatch(
        active_crop="paddy",
        response_text="మీ వరి ఆకును పరిశీలిస్తే బాక్టీరియల్ ఆకు ఎండు తెగులు...",
        disease_name="Paddy Bacterial Leaf Blight",
    )
    assert no_mismatch is None


def test_extract_all_crop_mentions():
    # Multi-crop text
    text_multi = "మీ వరి ఆకుల్లో లక్షణాలు కనిపిస్తున్నాయి. పత్తి పంటలో కూడా ఇలాంటి లక్షణాలు కనిపించవచ్చు."
    crops = extract_all_crop_mentions(text_multi)
    assert crops == ["paddy", "cotton"]

    # Single crop mentions
    assert extract_all_crop_mentions("మీ పత్తి ఆకులను పరిశీలిస్తే...") == ["cotton"]
    assert extract_all_crop_mentions("మీ వరి ఆకులను పరిశీలిస్తే...") == ["paddy"]

    # Non-crop general text
    assert extract_all_crop_mentions("కోరుట్లలో రేపు వర్షం పడుతుందా?") == []

    # Non-aggressive substring matching
    assert extract_all_crop_mentions("దయచేసి వివరించండి") == []


def test_detect_crop_mismatch_multi_crop_all_mentions():
    # Active paddy + response mentions both paddy and cotton -> cotton mismatch detected
    mismatch_both = detect_crop_mismatch(
        active_crop="paddy",
        response_text="మీ వరి ఆకుల్లో లక్షణాలు కనిపిస్తున్నాయి. పత్తి పంటలో కూడా ఇలాంటి లక్షణాలు కనిపించవచ్చు.",
    )
    assert mismatch_both == "cotton"

    # Active paddy + response mentions cotton anywhere in response
    mismatch_cotton = detect_crop_mismatch(
        active_crop="paddy",
        response_text="మీ పత్తి ఆకులను పరిశీలిస్తే...",
    )
    assert mismatch_cotton == "cotton"

    # Active paddy + response mentions paddy only -> no mismatch
    no_mismatch = detect_crop_mismatch(
        active_crop="paddy",
        response_text="మీ వరి ఆకులను పరిశీలిస్తే...",
    )
    assert no_mismatch is None


def test_filter_history_for_vision_multi_crop_conservative():
    # Multi-crop record should be conservatively excluded when active_crop is paddy
    multi_crop_conv = Conversation(
        id=uuid.uuid4(),
        user_message="వరి పంట మరియు పత్తి పంటలో ఏది మంచిది?",
        ai_response="వరి మరియు పత్తి రెండు పంటలలో...",
    )
    paddy_only_conv = Conversation(
        id=uuid.uuid4(),
        user_message="నా వరి పంటలో ఆకులు పసుపుగా మారుతున్నాయి",
        ai_response="వరి పోషకాల లోపం...",
    )
    weather_conv = Conversation(
        id=uuid.uuid4(),
        user_message="కోరుట్లలో రేపు వర్షం పడుతుందా?",
        ai_response="వాతావరణ నివేదిక...",
    )
    cotton_only_conv = Conversation(
        id=uuid.uuid4(),
        user_message="వరంగల్లో పత్తి ధర ఎంత?",
        ai_response="పత్తి ధర...",
    )

    records = [multi_crop_conv, paddy_only_conv, weather_conv, cotton_only_conv]
    filtered = filter_history_for_vision(records, active_crop="paddy", max_turns=4)

    # multi_crop_conv and cotton_only_conv must be excluded
    # paddy_only_conv and weather_conv must be kept
    assert len(filtered) == 2
    assert filtered[0].id == paddy_only_conv.id
    assert filtered[1].id == weather_conv.id


def test_resolve_image_crop_context_current_image_first_previous_paddy_second():
    # Production shape: history_records[0] = current image message with user_message=None
    current_img = Conversation(
        id=uuid.uuid4(),
        user_message=None,
        user_message_type="image",
    )
    previous_paddy = Conversation(
        id=uuid.uuid4(),
        user_message="నా వరి పంటలో ఆకులు పసుపుగా మారుతున్నాయి",
        user_message_type="text",
    )
    older_cotton = Conversation(
        id=uuid.uuid4(),
        user_message="నా పత్తి ఆకులపై మచ్చలు ఉన్నాయి",
        user_message_type="text",
    )

    history = [current_img, previous_paddy, older_cotton]
    ctx = resolve_image_crop_context(
        user_caption=None,
        history_records=history,
        current_conversation_id=current_img.id,
    )
    assert ctx is not None
    assert ctx["crop"] == "paddy"
    assert ctx["crop_display_en"] == "paddy/rice"
    assert "వరి" in ctx["farmer_message"]


def test_resolve_image_crop_context_current_image_first_previous_cotton_second():
    current_img = Conversation(
        id=uuid.uuid4(),
        user_message=None,
        user_message_type="image",
    )
    previous_cotton = Conversation(
        id=uuid.uuid4(),
        user_message="నా పత్తి ఆకులపై మచ్చలు ఉన్నాయి",
        user_message_type="text",
    )
    older_paddy = Conversation(
        id=uuid.uuid4(),
        user_message="నా వరి పంట సమాచారం",
        user_message_type="text",
    )

    history = [current_img, previous_cotton, older_paddy]
    ctx = resolve_image_crop_context(
        user_caption=None,
        history_records=history,
        current_conversation_id=current_img.id,
    )
    assert ctx is not None
    assert ctx["crop"] == "cotton"
    assert ctx["crop_display_en"] == "cotton"
    assert "పత్తి" in ctx["farmer_message"]


def test_resolve_image_crop_context_voice_derived_previous_message():
    current_img = Conversation(
        id=uuid.uuid4(),
        user_message=None,
        user_message_type="image",
    )
    # Voice-derived text stored by STT
    previous_voice_paddy = Conversation(
        id=uuid.uuid4(),
        user_message="వరి పంటలో తెగులు సోకింది ఏం చేయాలి",
        user_message_type="audio",
    )
    older_cotton = Conversation(
        id=uuid.uuid4(),
        user_message="పత్తి ధర ఎంత",
        user_message_type="text",
    )

    history = [current_img, previous_voice_paddy, older_cotton]
    ctx = resolve_image_crop_context(
        user_caption=None,
        history_records=history,
        current_conversation_id=current_img.id,
    )
    assert ctx is not None
    assert ctx["crop"] == "paddy"
    assert ctx["crop_display_en"] == "paddy/rice"
    assert "వరి" in ctx["farmer_message"]


def test_resolve_image_crop_context_no_crop_context_does_not_select_old_crop():
    current_img = Conversation(
        id=uuid.uuid4(),
        user_message=None,
        user_message_type="image",
    )
    previous_weather = Conversation(
        id=uuid.uuid4(),
        user_message="కోరుట్లలో రేపు వర్షం పడుతుందా?",
        user_message_type="text",
    )
    old_cotton = Conversation(
        id=uuid.uuid4(),
        user_message="వరంగల్లో పత్తి ధర ఎంత?",
        user_message_type="text",
    )
    old_paddy = Conversation(
        id=uuid.uuid4(),
        user_message="వరి ఎరువుల మోతాదు ఎంత?",
        user_message_type="text",
    )

    history = [current_img, previous_weather, old_cotton, old_paddy]
    ctx = resolve_image_crop_context(
        user_caption=None,
        history_records=history,
        current_conversation_id=current_img.id,
    )
    # Must NOT grab old cotton or paddy when the preceding message was non-crop
    assert ctx is None


# ─────────────────────────────────────────────────────────────────────────────
# TEST 1: Previous = Paddy, Current = Image-only, Old history has cotton
# Expected: Vision prompt prioritizes paddy and excludes cotton history.
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_1_image_prompt_prioritizes_paddy_context(sample_farmer, mock_db_session):
    paddy_conv = Conversation(
        id=uuid.uuid4(),
        farmer_id=sample_farmer.id,
        user_message="నా వరి పంటలో ఆకులు పసుపుగా మారుతున్నాయి",
        ai_response="వరి ఆకులు పసుపుగా మారడానికి పోషకాల లోపం కారణం కావచ్చు.",
    )
    cotton_conv = Conversation(
        id=uuid.uuid4(),
        farmer_id=sample_farmer.id,
        user_message="నా పత్తి ఆకులపై మచ్చలు ఉన్నాయి",
        ai_response="పత్తి పంటలో ఇది ఆల్టర్నేరియా ఆకుమచ్చ తెగులు కావచ్చు.",
    )

    image_conv = Conversation(
        id=uuid.uuid4(),
        farmer_id=sample_farmer.id,
        user_message=None,  # Image-only
        user_message_type="image",
    )

    paddy_diagnosis_json = json.dumps({
        "disease_name": "బాక్టీరియల్ ఆకు ఎండు తెగులు (Bacterial Leaf Blight)",
        "confidence_score": 0.88,
        "severity": "medium",
        "symptoms": "వరి ఆకుల అంచులు ఎండిపోవడం",
        "treatment_recommendation": "కాపర్ ఆక్సిక్లోరైడ్ పిచికారీ చేయండి",
        "friendly_whatsapp_reply": "మీ వరి ఆకును పరిశీలిస్తే, ఇది బాక్టీరియల్ ఆకు ఎండు తెగులు లక్షణాల వలె ఉంది."
    })

    with patch("src.ai.repository.AIRepository.get_farmer_profile", new_callable=AsyncMock, return_value=None), \
         patch("src.memory.service.FarmerMemoryService.format_memory_for_system_prompt", new_callable=AsyncMock, return_value="Primary Crops: cotton, Paddy"), \
         patch("src.ai.repository.AIRepository.get_conversation_history", new_callable=AsyncMock, return_value=[paddy_conv, cotton_conv]), \
         patch("src.ai.gemini_client.generate_multimodal_response", new_callable=AsyncMock, return_value=paddy_diagnosis_json) as mock_gemini:

        reply = await process_image_message(
            db=mock_db_session,
            farmer=sample_farmer,
            conversation=image_conv,
            image_bytes=b"fake_leaf_image_bytes",
            mime_type="image/jpeg",
        )

        mock_gemini.assert_awaited_once()
        call_kwargs = mock_gemini.await_args.kwargs

        # 1. System prompt prioritizes paddy and warns against cotton switch
        system_prompt = call_kwargs["system_prompt"]
        assert "paddy/rice" in system_prompt
        assert "Do not switch to another crop based only on older conversation history" in system_prompt

        # 2. User caption passed to vision specifies paddy context
        user_msg = call_kwargs["user_message"]
        assert "paddy/rice" in user_msg
        assert "నా వరి పంటలో ఆకులు పసుపుగా మారుతున్నాయి" in user_msg

        # 3. Cotton conversation was filtered OUT of conversation_history
        passed_history = call_kwargs["conversation_history"]
        history_str = json.dumps(passed_history, ensure_ascii=False)
        assert "వరి" in history_str
        assert "పత్తి" not in history_str

        # 4. Final reply is paddy diagnosis
        assert "వరి" in reply


# ─────────────────────────────────────────────────────────────────────────────
# TEST 2: Previous = Cotton, Current = Image-only
# Expected: Cotton context is used.
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_2_image_prompt_uses_cotton_context(sample_farmer, mock_db_session):
    cotton_conv = Conversation(
        id=uuid.uuid4(),
        farmer_id=sample_farmer.id,
        user_message="నా పత్తి ఆకులపై మచ్చలు ఉన్నాయి",
        ai_response="పత్తి పంటలో ఇది ఆల్టర్నేరియా ఆకుమచ్చ తెగులు కావచ్చు.",
    )

    image_conv = Conversation(
        id=uuid.uuid4(),
        farmer_id=sample_farmer.id,
        user_message=None,
        user_message_type="image",
    )

    cotton_diagnosis_json = json.dumps({
        "disease_name": "ఆల్టర్నేరియా ఆకుమచ్చ తెగులు (Alternaria Leaf Spot)",
        "confidence_score": 0.89,
        "severity": "medium",
        "symptoms": "పత్తి ఆకులపై గుండ్రటి మచ్చలు",
        "treatment_recommendation": "మాంకోజెబ్ 75% WP పిచికారీ చేయండి",
        "friendly_whatsapp_reply": "మీ పత్తి ఆకును పరిశీలిస్తే ఇది ఆల్టర్నేరియా ఆకుమచ్చ తెగులు లక్షణాల వలె ఉంది."
    })

    with patch("src.ai.repository.AIRepository.get_farmer_profile", new_callable=AsyncMock, return_value=None), \
         patch("src.memory.service.FarmerMemoryService.format_memory_for_system_prompt", new_callable=AsyncMock, return_value=""), \
         patch("src.ai.repository.AIRepository.get_conversation_history", new_callable=AsyncMock, return_value=[cotton_conv]), \
         patch("src.ai.gemini_client.generate_multimodal_response", new_callable=AsyncMock, return_value=cotton_diagnosis_json) as mock_gemini:

        reply = await process_image_message(
            db=mock_db_session,
            farmer=sample_farmer,
            conversation=image_conv,
            image_bytes=b"fake_cotton_leaf_bytes",
            mime_type="image/jpeg",
        )

        mock_gemini.assert_awaited_once()
        call_kwargs = mock_gemini.await_args.kwargs
        assert "cotton" in call_kwargs["system_prompt"]
        assert "cotton" in call_kwargs["user_message"]
        assert "పత్తి" in reply


# ─────────────────────────────────────────────────────────────────────────────
# TEST 3: No current crop context, Old history contains cotton & paddy
# Expected: Does NOT select cotton from old history; asks for crop clarification.
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_3_image_only_without_context_asks_clarification(sample_farmer, mock_db_session):
    # Old history with mixed crops
    weather_conv = Conversation(
        id=uuid.uuid4(),
        farmer_id=sample_farmer.id,
        user_message="కోరుట్లలో రేపు వర్షం పడుతుందా?",
        ai_response="ప్రస్తుతం వాతావరణ సమాచారం అందుబాటులో లేదు.",
    )
    old_cotton_conv = Conversation(
        id=uuid.uuid4(),
        farmer_id=sample_farmer.id,
        user_message="వరంగల్లో పత్తి ధర ఎంత?",
        ai_response="పత్తి ధర సమాచారం...",
    )
    old_paddy_conv = Conversation(
        id=uuid.uuid4(),
        farmer_id=sample_farmer.id,
        user_message="వరి ఎరువుల మోతాదు ఎంత?",
        ai_response="వరి ఎరువుల మోతాదు...",
    )

    image_conv = Conversation(
        id=uuid.uuid4(),
        farmer_id=sample_farmer.id,
        user_message=None,  # Image-only
        user_message_type="image",
    )

    # Gemini returns low confidence because crop is ambiguous without context
    ambiguous_json = json.dumps({
        "disease_name": "ఆకుమచ్చ (Uncertain Leaf Spot)",
        "confidence_score": 0.50,  # Low confidence (< 0.85)
        "severity": "low",
        "symptoms": "ఆకులపై మచ్చలు",
        "treatment_recommendation": "సరైన నివారణ కొరకు పంట పేరు నిర్ధారించండి.",
        "friendly_whatsapp_reply": "ఆకుపై మచ్చలు కనిపిస్తున్నాయి కానీ పంట స్పష్టంగా తెలియడం లేదు."
    })

    with patch("src.ai.repository.AIRepository.get_farmer_profile", new_callable=AsyncMock, return_value=None), \
         patch("src.memory.service.FarmerMemoryService.format_memory_for_system_prompt", new_callable=AsyncMock, return_value="Primary Crops: cotton, Paddy"), \
         patch("src.ai.repository.AIRepository.get_conversation_history", new_callable=AsyncMock, return_value=[weather_conv, old_cotton_conv, old_paddy_conv]), \
         patch("src.ai.gemini_client.generate_multimodal_response", new_callable=AsyncMock, return_value=ambiguous_json) as mock_gemini:

        reply = await process_image_message(
            db=mock_db_session,
            farmer=sample_farmer,
            conversation=image_conv,
            image_bytes=b"fake_ambiguous_image_bytes",
            mime_type="image/jpeg",
        )

        mock_gemini.assert_awaited_once()
        call_kwargs = mock_gemini.await_args.kwargs
        # System prompt explicitly informed model: NO ACTIVE CROP CONTEXT
        assert "=== NO ACTIVE CROP CONTEXT ===" in call_kwargs["system_prompt"]
        # Model history was filtered of old crop discussions
        passed_history = call_kwargs["conversation_history"]
        history_str = json.dumps(passed_history, ensure_ascii=False)
        assert "పత్తి" not in history_str
        assert "వరి" not in history_str

        # Response requires crop clarification rather than guessing cotton from profile
        expected_clarification = get_crop_clarification_response("te")
        assert reply == expected_clarification
        assert "ఏ పంటకు సంబంధించినది" in reply


# ─────────────────────────────────────────────────────────────────────────────
# TEST 4: Current crop = Paddy, Generated response mentions Cotton
# Expected: Crop consistency guard triggers second reconsideration step.
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_4_crop_consistency_guard_triggers_reconsideration(sample_farmer, mock_db_session):
    paddy_conv = Conversation(
        id=uuid.uuid4(),
        farmer_id=sample_farmer.id,
        user_message="నా వరి పంటలో ఆకులు పసుపుగా మారుతున్నాయి",
        ai_response="వరి ఆకులు పసుపుగా మారడానికి పోషకాల లోపం...",
    )

    image_conv = Conversation(
        id=uuid.uuid4(),
        farmer_id=sample_farmer.id,
        user_message=None,
        user_message_type="image",
    )

    # 1st call erroneously outputs Cotton diagnosis (the production bug)
    cotton_hallucination_json = json.dumps({
        "disease_name": "Cotton Bacterial Blight",
        "confidence_score": 0.88,
        "severity": "medium",
        "symptoms": "Blight spots on cotton leaf",
        "treatment_recommendation": "Spray Copper Oxychloride on cotton",
        "friendly_whatsapp_reply": "మీ పత్తి ఆకును పరిశీలిస్తే, ఇది 'బాక్టీరియల్ ఆకుమచ్చ తెగులు' లక్షణాల వలె కనిపిస్తోంది."
    })

    # 2nd call (reconsideration) correctly re-evaluates in Paddy context
    paddy_corrected_json = json.dumps({
        "disease_name": "బాక్టీరియల్ ఆకు ఎండు తెగులు (Bacterial Leaf Blight)",
        "confidence_score": 0.90,
        "severity": "medium",
        "symptoms": "వరి ఆకులపై పసుపు మరియు ఎండు లక్షణాలు",
        "treatment_recommendation": "కాపర్ ఆక్సిక్లోరైడ్ మరియు స్ట్రెప్టోసైక్లిన్ పిచికారీ చేయండి",
        "friendly_whatsapp_reply": "మీ వరి ఆకును పరిశీలిస్తే, ఇది బాక్టీరియల్ ఆకు ఎండు తెగులు లక్షణాల వలె కనిపిస్తోంది."
    })

    with patch("src.ai.repository.AIRepository.get_farmer_profile", new_callable=AsyncMock, return_value=None), \
         patch("src.memory.service.FarmerMemoryService.format_memory_for_system_prompt", new_callable=AsyncMock, return_value="Primary Crops: cotton, Paddy"), \
         patch("src.ai.repository.AIRepository.get_conversation_history", new_callable=AsyncMock, return_value=[paddy_conv]), \
         patch("src.ai.gemini_client.generate_multimodal_response", new_callable=AsyncMock, side_effect=[cotton_hallucination_json, paddy_corrected_json]) as mock_gemini:

        reply = await process_image_message(
            db=mock_db_session,
            farmer=sample_farmer,
            conversation=image_conv,
            image_bytes=b"fake_paddy_leaf_image_bytes",
            mime_type="image/jpeg",
        )

        # Gemini was called TWICE: 1st initial vision diagnosis, 2nd consistency guard reconsideration
        assert mock_gemini.await_count == 2

        # 2nd call was invoked with the exact reconsideration instruction
        second_call_kwargs = mock_gemini.await_args_list[1].kwargs
        second_user_message = second_call_kwargs["user_message"]
        expected_recon_instruction = build_crop_reconsideration_instruction("paddy/rice", "cotton")
        assert second_user_message == expected_recon_instruction
        assert "The current active crop context is paddy/rice, but the generated response refers to cotton" in second_user_message

        # Final reply was corrected to paddy
        assert "వరి" in reply
        assert "పత్తి" not in reply


# ─────────────────────────────────────────────────────────────────────────────
# TEST 5: Normal text query remains unchanged
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_5_normal_text_query_remains_unchanged(sample_farmer, mock_db_session):
    conv = Conversation(
        id=uuid.uuid4(),
        farmer_id=sample_farmer.id,
        user_message="వరిలో ఎరువుల మోతాదు ఎంత?",
        user_message_type="text",
    )

    with patch("src.ai.decision_engine.get_decision_engine") as mock_engine:
        mock_engine.return_value.process_message = AsyncMock(return_value="వరి పంటకు ఎకరాకు 100 కేజీల యూరియా అవసరం.")
        reply = await process_text_message(mock_db_session, sample_farmer, conv)

        assert reply is not None
        assert "వరి" in reply
        mock_engine.return_value.process_message.assert_awaited_once_with(mock_db_session, sample_farmer, conv)


# ─────────────────────────────────────────────────────────────────────────────
# TEST 6: Voice query remains unchanged
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_6_voice_query_remains_unchanged(sample_farmer):
    parsed = ParsedIncomingMessage(
        phone_number="917989271932",
        message_id="wamid.VOICE_REGRESSION_TEST",
        timestamp="1700000000",
        message_type="audio",
        media_id="voice_media_123",
    )
    conv = Conversation(
        id=uuid.uuid4(),
        farmer_id=sample_farmer.id,
        message_id=parsed.message_id,
        user_message=None,
        user_message_type="audio",
    )

    mock_db = AsyncMock()
    mock_res = MagicMock()
    mock_res.scalar_one_or_none.return_value = None
    mock_res.scalars.return_value.all.return_value = []
    mock_db.execute.return_value = mock_res
    mock_db_cm = AsyncMock()
    mock_db_cm.__aenter__.return_value = mock_db
    mock_db_cm.__aexit__.return_value = None

    from src.language.schemas import TranscriptionResponse

    with patch("src.gateway.service.AsyncSessionLocal", return_value=mock_db_cm), \
         patch("src.gateway.service.is_duplicate_message", new_callable=AsyncMock, return_value=False), \
         patch("src.gateway.service.get_or_create_farmer", new_callable=AsyncMock, return_value=sample_farmer), \
         patch("src.gateway.service.store_incoming_message", new_callable=AsyncMock, return_value=conv), \
         patch("src.gateway.service.download_media_bytes", new_callable=AsyncMock, return_value=(b"audio_bytes", "audio/ogg")), \
         patch("src.gateway.service.get_language_service") as mock_lang_svc, \
         patch("src.gateway.service.process_text_message", new_callable=AsyncMock, return_value="వరిలో తెగులు నివారణ సమాచారం.") as mock_proc_text, \
         patch("src.gateway.service.send_text_message", new_callable=AsyncMock, return_value="wamid.OUT_TEXT_1") as mock_send_text, \
         patch("src.gateway.service.upload_media_bytes", new_callable=AsyncMock, return_value="audio_media_out") as mock_upload, \
         patch("src.gateway.service.send_audio_message", new_callable=AsyncMock, return_value="wamid.OUT_AUDIO_1") as mock_send_audio, \
         patch("src.gateway.service.get_settings") as mock_settings:

        mock_settings.return_value.enable_voice_responses = True
        mock_lang_svc.return_value.transcribe_audio = AsyncMock(
            return_value=TranscriptionResponse(provider_used="google", transcription_text="వరిలో తెగులు వచ్చింది", detected_language="te")
        )
        mock_lang_svc.return_value.synthesize_speech = AsyncMock(return_value=[b"OggS_voice_bytes"])

        await process_message_pipeline(parsed)

        mock_proc_text.assert_awaited_once()
        mock_send_text.assert_awaited_once_with(to_phone="917989271932", message_text="వరిలో తెగులు నివారణ సమాచారం.")
        mock_lang_svc.return_value.synthesize_speech.assert_awaited_once_with("వరిలో తెగులు నివారణ సమాచారం.", "te")
        mock_send_audio.assert_awaited_once_with("917989271932", "audio_media_out")


# ─────────────────────────────────────────────────────────────────────────────
# TEST 7: Image response produces exactly ONE ai_response reused for text & TTS
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_7_image_response_produces_one_ai_response_reused_for_text_and_tts(sample_farmer):
    parsed = ParsedIncomingMessage(
        phone_number="917989271932",
        message_id="wamid.IMAGE_OMNI_TEST",
        timestamp="1700000000",
        message_type="image",
        media_id="img_media_123",
    )
    conv = Conversation(
        id=uuid.uuid4(),
        farmer_id=sample_farmer.id,
        message_id=parsed.message_id,
        user_message=None,
        user_message_type="image",
    )

    mock_db = AsyncMock()
    mock_res = MagicMock()
    mock_res.scalar_one_or_none.return_value = None
    mock_res.scalars.return_value.all.return_value = []
    mock_db.execute.return_value = mock_res
    mock_db_cm = AsyncMock()
    mock_db_cm.__aenter__.return_value = mock_db
    mock_db_cm.__aexit__.return_value = None

    single_image_diagnosis = "మీ వరి ఆకులో బాక్టీరియల్ ఆకు ఎండు తెగులు లక్షణాలు కనిపిస్తున్నాయి."

    with patch("src.gateway.service.AsyncSessionLocal", return_value=mock_db_cm), \
         patch("src.gateway.service.is_duplicate_message", new_callable=AsyncMock, return_value=False), \
         patch("src.gateway.service.get_or_create_farmer", new_callable=AsyncMock, return_value=sample_farmer), \
         patch("src.gateway.service.store_incoming_message", new_callable=AsyncMock, return_value=conv), \
         patch("src.gateway.service.download_media_bytes", new_callable=AsyncMock, return_value=(b"img_bytes", "image/jpeg")), \
         patch("src.gateway.service.process_image_message", new_callable=AsyncMock, return_value=single_image_diagnosis) as mock_proc_image, \
         patch("src.gateway.service.get_language_service") as mock_lang_svc, \
         patch("src.gateway.service.send_text_message", new_callable=AsyncMock, return_value="wamid.OUT_TEXT_IMG") as mock_send_text, \
         patch("src.gateway.service.upload_media_bytes", new_callable=AsyncMock, return_value="audio_media_img") as mock_upload, \
         patch("src.gateway.service.send_audio_message", new_callable=AsyncMock, return_value="wamid.OUT_AUDIO_IMG") as mock_send_audio, \
         patch("src.gateway.service.get_settings") as mock_settings:

        mock_settings.return_value.enable_voice_responses = True
        mock_lang_svc.return_value.synthesize_speech = AsyncMock(return_value=[b"OggS_tts_audio_bytes"])

        await process_message_pipeline(parsed)

        # 1. Exactly ONE AI diagnosis call made for image
        mock_proc_image.assert_awaited_once()

        # 2. Text send uses the exact single_image_diagnosis
        mock_send_text.assert_awaited_once_with(to_phone="917989271932", message_text=single_image_diagnosis)

        # 3. Speech synthesis receives the exact same single_image_diagnosis string
        mock_lang_svc.return_value.synthesize_speech.assert_awaited_once_with(single_image_diagnosis, "te")

        # 4. Outbound audio message sent
        mock_send_audio.assert_awaited_once_with("917989271932", "audio_media_img")


@pytest.mark.asyncio
async def test_image_process_with_current_image_record_at_history_index_0(sample_farmer, mock_db_session):
    """
    Regression test for Issue 1:
    history_records[0] = current image message with user_message=None
    history_records[1] = previous paddy message
    history_records[2] = older cotton message
    Expected: active crop = paddy, and previous paddy message must be used as active context.
    """
    image_conv = Conversation(
        id=uuid.uuid4(),
        farmer_id=sample_farmer.id,
        user_message=None,
        user_message_type="image",
    )
    previous_paddy = Conversation(
        id=uuid.uuid4(),
        farmer_id=sample_farmer.id,
        user_message="నా వరి పంటలో ఆకులు పసుపుగా మారుతున్నాయి",
        ai_response="వరి ఆకులు పసుపుగా మారడానికి పోషకాల లోపం కారణం కావచ్చు.",
    )
    older_cotton = Conversation(
        id=uuid.uuid4(),
        farmer_id=sample_farmer.id,
        user_message="వరంగల్లో పత్తి ధర ఎంత?",
        ai_response="పత్తి ధర సమాచారం...",
    )

    diagnosis_json = json.dumps({
        "disease_name": "వరి ఆకు ఎండిపోవడం (Bacterial Blight)",
        "confidence_score": 0.90,
        "severity": "medium",
        "symptoms": "ఆకుల చివర్లు పసుపుగా మారి ఎండిపోవడం",
        "treatment_recommendation": "నత్రజని ఎరువుల వాడకాన్ని తగ్గించండి.",
        "friendly_whatsapp_reply": "మీ వరి పంటలో బాక్టీరియల్ ఆకు ఎండు తెగులు లక్షణాలు కనిపిస్తున్నాయి."
    })

    with patch("src.ai.repository.AIRepository.get_farmer_profile", new_callable=AsyncMock, return_value=None), \
         patch("src.memory.service.FarmerMemoryService.format_memory_for_system_prompt", new_callable=AsyncMock, return_value=""), \
         patch("src.ai.repository.AIRepository.get_conversation_history", new_callable=AsyncMock, return_value=[image_conv, previous_paddy, older_cotton]), \
         patch("src.ai.gemini_client.generate_multimodal_response", new_callable=AsyncMock, return_value=diagnosis_json) as mock_gemini:

        reply = await process_image_message(
            db=mock_db_session,
            farmer=sample_farmer,
            conversation=image_conv,
            image_bytes=b"fake_image_bytes",
            mime_type="image/jpeg",
        )

        mock_gemini.assert_awaited_once()
        call_kwargs = mock_gemini.await_args.kwargs
        system_prompt = call_kwargs["system_prompt"]

        # 1. Paddy context was successfully extracted from previous_paddy despite image_conv at index 0
        assert "=== CURRENT ACTIVE CONVERSATION CONTEXT ===" in system_prompt
        assert "Active Crop: paddy/rice" in system_prompt
        assert "నా వరి పంటలో ఆకులు పసుపుగా మారుతున్నాయి" in system_prompt

        # 2. Older cotton history was excluded from model history
        passed_history = call_kwargs["conversation_history"]
        history_str = json.dumps(passed_history, ensure_ascii=False)
        assert "పత్తి" not in history_str

        # 3. Current image record was NOT added to Gemini history
        assert image_conv.user_message not in [h.get("parts") for h in passed_history]

        assert "వరి పంటలో" in reply
