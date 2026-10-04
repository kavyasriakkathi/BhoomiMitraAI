"""
BhoomiMitra AI — Image Diagnosis Crop Context & Consistency Guard

Resolves active crop context for image messages, filters conversation history
to eliminate cross-crop contamination, and provides validation guards against
hallucinated crop switches in Gemini Multimodal Vision responses.
"""

import re
from typing import List, Optional, Dict, Any


CROP_DEFINITIONS: Dict[str, Dict[str, Any]] = {
    "paddy": {
        "display_en": "paddy/rice",
        "display_te": "వరి",
        "display_hi": "धान",
        "aliases": [
            "వరి", "వరికి", "వరిలో", "వరితో", "వరిపై", "వరిని", "వరిపంట", "వరి పంట", "వరి చేను",
            "paddy", "rice", "vari", "dhan", "धान", "நெல்", "ಭತ್ತ",
        ],
    },
    "cotton": {
        "display_en": "cotton",
        "display_te": "పత్తి",
        "display_hi": "कपास",
        "aliases": [
            "పత్తి", "పత్తికి", "పత్తిలో", "పత్తితో", "పత్తిపై", "పత్తిని", "పత్తిపంట", "పత్తి పంట", "పత్తి చేను",
            "cotton", "patti", "kapas", "कपास", "பருத்தி", "ಹತ್ತి",
        ],
    },
    "chilli": {
        "display_en": "chilli",
        "display_te": "మిర్చి",
        "display_hi": "मिर्च",
        "aliases": [
            "మిర్చి", "మిరప", "మిరపకి", "మిరపలో", "మిరపతో", "మిరపపై", "మిరపని", "మిరపపంట", "మిరప పంట", "మిర్చి పంట",
            "chilli", "chili", "chillies", "mirchi", "mirapa", "मिर्च", "மிளகாய்", "ಮೆಣసినకాయి",
        ],
    },
    "maize": {
        "display_en": "maize/corn",
        "display_te": "మొక్కజొన్న",
        "display_hi": "मक्का",
        "aliases": [
            "మొక్కజొన్న", "మొక్కజొన్నలో", "మొక్కజొన్నకి", "మొక్కజొన్నతో", "మొక్కజొన్నపై", "మొక్కజొన్న పంట", "మొక్కజొన్న చేను",
            "maize", "corn", "mokkajonna", "मक्का", "மக்காச்சோளம்", "ಮೆಕ್ಕೆజోಳ",
        ],
    },
    "tomato": {
        "display_en": "tomato",
        "display_te": "టమోటా",
        "display_hi": "टमाटर",
        "aliases": [
            "టమోటా", "టమాటా", "టమోట", "టమాట", "టమోటా పంట", "టమాట పంట",
            "tomato", "tomatoes", "tamata", "टमाटर", "தக்காளி", "ಟೊಮ್ಯಾಟೊ",
        ],
    },
    "groundnut": {
        "display_en": "groundnut/peanut",
        "display_te": "వేరుశనగ",
        "display_hi": "मूंगफली",
        "aliases": [
            "వేరుశనగ", "వేరుశెనగ", "వేరుశనగలో", "వేరుశెనగలో", "వేరుశనగకి", "వేరుశెనగకి", "వేరుశనగ పంట", "వేరుశెనగ పంట",
            "groundnut", "peanut", "peanuts", "मूंगफली", "നിലക്കടല", "ಕಡಲೆకాయి",
        ],
    },
    "soybean": {
        "display_en": "soybean",
        "display_te": "సోయాబీన్",
        "display_hi": "सोयाबीन",
        "aliases": [
            "సోయాబీన్", "సోయా", "సోయా పంట",
            "soybean", "soyabean", "सोयाबीन",
        ],
    },
    "sugarcane": {
        "display_en": "sugarcane",
        "display_te": "చెరకు",
        "display_hi": "गन्ना",
        "aliases": [
            "చెరకు", "చెరుకు", "చెరకు పంట", "చెరుకు పంట",
            "sugarcane", "गन्ना", "கரும்பு",
        ],
    },
    "wheat": {
        "display_en": "wheat",
        "display_te": "గోధుమ",
        "display_hi": "गेहूं",
        "aliases": [
            "గోధుమ", "గోధుమలు", "గోధుమ పంట",
            "wheat", "गेहूं",
        ],
    },
    "onion": {
        "display_en": "onion",
        "display_te": "ఉల్లిపాయ",
        "display_hi": "प्याज",
        "aliases": [
            "ఉల్లిపాయ", "ఉల్లి", "ఉల్లి పంట",
            "onion", "onions", "प्याज", "வெங்காயம்",
        ],
    },
    "turmeric": {
        "display_en": "turmeric",
        "display_te": "పసుపు పంట",
        "display_hi": "हल्दी",
        "aliases": [
            "పసుపు పంట", "పసుపు దుంప", "పసుపు సాగు", "పసుపు చేను",
            "turmeric crop", "turmeric farm", "हल्दी की फसल",
        ],
    },
    "red_gram": {
        "display_en": "red gram/pigeon pea",
        "display_te": "కందులు",
        "display_hi": "अरहर/तुअर",
        "aliases": [
            "కందులు", "కంది", "కంది పంట", "కంది చేను",
            "red gram", "pigeon pea", "toor", "tur dal", "अरहर", "तुअर",
        ],
    },
}


def extract_all_crop_mentions(text: Optional[str]) -> List[str]:
    """
    Extracts all unique canonical crop keys (e.g. 'paddy', 'cotton') mentioned in text,
    ordered by their first appearance in the text.
    Handles Indic scripts and Romanized words with proper boundary matching.
    Distinguishes color descriptions like 'పసుపుగా' (yellowish) from crops like turmeric.
    """
    if not text or not isinstance(text, str):
        return []

    cleaned_text = text.lower().strip()
    crop_positions: Dict[str, int] = {}

    for crop_key, crop_meta in CROP_DEFINITIONS.items():
        earliest_pos = None
        for alias in crop_meta["aliases"]:
            alias_lower = alias.lower()
            if any(ord(c) > 127 for c in alias):
                # Indic scripts: ensure alias is not embedded within a longer Indic word
                pattern = r"(?<![\u0900-\u0d7f])" + re.escape(alias) + r"(?![\u0900-\u0d7f])"
                m = re.search(pattern, text)
                if m:
                    pos = m.start()
                    if earliest_pos is None or pos < earliest_pos:
                        earliest_pos = pos
            else:
                # Latin characters: word boundary match
                pattern = r"\b" + re.escape(alias_lower) + r"\b"
                m = re.search(pattern, cleaned_text)
                if m:
                    pos = m.start()
                    if earliest_pos is None or pos < earliest_pos:
                        earliest_pos = pos

        if earliest_pos is not None:
            crop_positions[crop_key] = earliest_pos

    # Return unique canonical crop keys sorted by their appearance in the text
    return sorted(crop_positions.keys(), key=lambda k: crop_positions[k])


def extract_crop_mention(text: Optional[str]) -> Optional[str]:
    """
    Extracts the first matching canonical crop key (e.g. 'paddy', 'cotton') from text.
    """
    crops = extract_all_crop_mentions(text)
    return crops[0] if crops else None


def resolve_image_crop_context(
    user_caption: Optional[str],
    history_records: List[Any],
    current_conversation_id: Optional[Any] = None,
) -> Optional[Dict[str, str]]:
    """
    Determines the active crop context for an inbound image.
    1. Checks the user caption if present.
    2. If no crop in caption (or image-only), inspects preceding farmer conversation history
       to find the most recent previous farmer text/voice-derived message (excluding the
       current image message and any image-only turns without text/voice).
    3. If that immediately preceding farmer message establishes a crop, uses that crop.
       Otherwise returns None (does not grab old stale crops from older turns).
    Returns a dict with crop metadata or None if no crop is established.
    """
    # 1. Caption check
    if user_caption and user_caption.strip():
        caption_crops = extract_all_crop_mentions(user_caption)
        if caption_crops:
            caption_crop = caption_crops[0]
            meta = CROP_DEFINITIONS[caption_crop]
            return {
                "crop": caption_crop,
                "crop_display_en": meta["display_en"],
                "crop_display_te": meta["display_te"],
                "farmer_message": user_caption.strip(),
                "source": "caption",
            }

    # 2. History check: find the most recent previous farmer text/voice-derived message
    if not history_records:
        return None

    prev_text_record = None
    for rec in history_records:
        # Exclude current conversation by ID if provided
        if current_conversation_id and getattr(rec, "id", None) == current_conversation_id:
            continue

        user_msg = getattr(rec, "user_message", None)
        # Skip image messages with no text/voice transcription (or empty messages)
        if not user_msg or not str(user_msg).strip():
            continue

        # If current_conversation_id is None, check if this record is the current uncommitted record
        if current_conversation_id is None and user_caption and user_msg.strip() == user_caption.strip() and not getattr(rec, "ai_response", None):
            continue

        # Found the most recent preceding farmer message with text/voice
        prev_text_record = rec
        break

    if prev_text_record:
        prev_msg = str(getattr(prev_text_record, "user_message", "")).strip()
        mentioned_crops = extract_all_crop_mentions(prev_msg)
        if mentioned_crops:
            prev_crop = mentioned_crops[0]
            meta = CROP_DEFINITIONS[prev_crop]
            return {
                "crop": prev_crop,
                "crop_display_en": meta["display_en"],
                "crop_display_te": meta["display_te"],
                "farmer_message": prev_msg,
                "source": "previous_message",
            }

    return None


def filter_history_for_vision(
    history_records: List[Any],
    active_crop: Optional[str],
    max_turns: int = 4,
    current_conversation_id: Optional[Any] = None,
) -> List[Any]:
    """
    Filters conversation history records specifically for Gemini Vision input.
    Eliminates cross-crop contamination:
    - Excludes the current image conversation turn if present in history.
    - If active_crop is established (e.g. 'paddy'), removes turns that discuss
      conflicting crops (e.g. 'cotton'). If a turn mentions both paddy and cotton,
      it is treated conservatively and excluded unless it only discusses the active crop.
    - If active_crop is None, removes turns that discuss any specific crop to avoid
      biasing the vision model with stale historical topics.
    - Preserves useful non-crop/general context (e.g. weather, market general info).
    Returns a list of relevant records (still ordered newest-first).
    """
    if not history_records:
        return []

    filtered: List[Any] = []

    for rec in history_records:
        if current_conversation_id and getattr(rec, "id", None) == current_conversation_id:
            continue

        user_msg = getattr(rec, "user_message", None) or ""
        ai_resp = getattr(rec, "ai_response", None) or ""
        combined_text = f"{user_msg} {ai_resp}"

        mentioned_crops = extract_all_crop_mentions(combined_text)

        if active_crop:
            # If the turn mentions any conflicting crop (different from active_crop),
            # skip it to prevent contamination (even if active_crop is also mentioned)
            has_conflicting = any(c != active_crop for c in mentioned_crops)
            if has_conflicting:
                continue
            filtered.append(rec)
        else:
            # When there is no active crop, exclude records that discuss specific crops
            if mentioned_crops:
                continue
            filtered.append(rec)

        if len(filtered) >= max_turns:
            break

    return filtered


def detect_crop_mismatch(
    active_crop: str,
    response_text: str,
    disease_name: Optional[str] = None,
) -> Optional[str]:
    """
    Detects if a generated diagnosis response refers to any crop that conflicts with active_crop.
    Example: active_crop='paddy', but response mentions 'cotton' or 'పత్తి'.
    Checks ALL crop mentions in the response and disease name against active_crop.
    Returns the conflicting crop's English display name if a mismatch is found, otherwise None.
    """
    if not active_crop or not response_text:
        return None

    combined_text = f"{disease_name or ''} {response_text}"
    mentioned_crops = extract_all_crop_mentions(combined_text)

    for crop_key in mentioned_crops:
        if crop_key != active_crop:
            return CROP_DEFINITIONS[crop_key]["display_en"]

    return None


def build_crop_reconsideration_instruction(
    active_crop_display: str,
    conflicting_crop_display: str,
) -> str:
    """
    Builds the explicit reconsideration instruction prompt required when a crop
    consistency mismatch is detected.
    """
    return (
        f"The current active crop context is {active_crop_display}, but the generated response refers to {conflicting_crop_display}. "
        f"Re-evaluate the image using the current active crop context. "
        f"If crop identity cannot be determined confidently, ask the farmer to confirm the crop rather than guessing."
    )
