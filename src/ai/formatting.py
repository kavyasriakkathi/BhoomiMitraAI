"""
BhoomiMitra AI — Multi-Intent Response Formatter & Authoritative Deduplicator

Optimizes WhatsApp responses when a farmer asks multiple questions in a single query.
Organizes answers into clear, farmer-friendly sections (Crop Advice, Weather, Shops, Market, Schemes, Escalation),
removes repetitive greetings and introductions, selects exactly ONE authoritative response per detected intent,
and compacts information for mobile readability.
"""
import re
from typing import Dict, Optional, Tuple, List

# Section Header Emojis & Titles
SECTION_HEADERS_EN = {
    "crop_advice": "🌱 *Crop Advice*",
    "weather": "🌡️ *Weather Information*",
    "shop": "🏬 *Nearby Shops & Availability*",
    "market": "📊 *Market Prices*",
    "schemes": "🏛️ *Government Schemes*",
    "escalation": "👨‍🌾 *Krishi Officer Escalation*",
}

SECTION_HEADERS_TE = {
    "crop_advice": "🌱 *పంట సలహా*",
    "weather": "🌡️ *వాతావరణ సమాచారం*",
    "shop": "🏬 *సమీప వ్యవసాయ దుకాణాలు*",
    "market": "📊 *మార్కెట్ ధరలు*",
    "schemes": "🏛️ *ప్రభుత్వ పథకాలు*",
    "escalation": "👨‍🌾 *వ్యవసాయ అధికారి సంప్రదింపు*",
}

# Unavailable/Fallback labels per section when requested intent has no active data
_UNAVAILABLE_LABELS_EN = {
    "weather": "ℹ️ Weather information is currently unavailable for this location.",
    "shop": "ℹ️ No registered shops found for this product currently.",
    "market": "ℹ️ Market prices are currently unavailable for this crop.",
    "schemes": "ℹ️ No government schemes found currently.",
}

_UNAVAILABLE_LABELS_TE = {
    "weather": "ℹ️ ఈ ప్రాంతానికి ప్రస్తుతం వాతావరణ సమాచారం అందుబాటులో లేదు.",
    "shop": "ℹ️ ప్రస్తుతం ఈ ఉత్పత్తికి సమీప దుకాణాలు అందుబాటులో లేవు.",
    "market": "ℹ️ ప్రస్తుతం ఈ పంటకు మార్కెట్ ధరలు అందుబాటులో లేవు.",
    "schemes": "ℹ️ ప్రస్తుతం సంబంధిత ప్రభుత్వ పథకాలు అందుబాటులో లేవు.",
}

# Introductory filler patterns to strip from the beginning of text/sections
_INTRO_PATTERNS = [
    r"^(?:hello|hi|hey|namaste|greetings)(?:\s+(?:farmer|friend|brother|farmer brother))?[!,\.\s\-]+",
    r"^(?:నమస్తే|హలో|నమస్కారం|నమస్కారాలు)(?:\s+(?:రైతు సోదరా|రైతు మిత్రమా|రైతు అన్న|రైతు|మిత్రమా|అన్న|సోదరా))?[!,\.\s\-]+",
    r"^(?:i am|my name is)\s+bhoomimitra(?:\s+ai)?(?:[^\n\.\!\:]*[\:\.\!])?\s*",
    r"^నేను\s+భూమిమిత్ర(?:\s+ai)?(?:[^\n\.\!\:]*[\:\.\!])?\s*",
    r"^here (?:is|are) the (?:details|answers|information)(?:[^\n\.\!\:]*[\:\.\!])?\s*",
    r"^మీరు అడిగిన (?:సమాచారం|వివరాలు|సలహాలు)(?:[^\n\.\!\:]*[\:\.\!])?\s*",
    r"^ఇక్కడ సమాచారం ఉంది(?:[^\n\.\!\:]*[\:\.\!])?\s*",
    r"^sure[,\s]+i can help you with (?:that|your questions)(?:[^\n\.\!\:]*[\:\.\!])?\s*",
    r"^ఖచ్చితంగా[,\s]+నేను మీకు సహాయం చేస్తాను(?:[^\n\.\!\:]*[\:\.\!])?\s*",
    r"^మీ పంటకు సంబంధించి[,\s]+(?:సమాచారం|వివరాలు)?(?:[^\n\.\!\:]*[\:\.\!])?\s*",
    r"^regarding your crop[,\s]+(?:information)?(?:[^\n\.\!\:]*[\:\.\!])?\s*",
]

_COMPILED_INTROS = [re.compile(p, re.IGNORECASE) for p in _INTRO_PATTERNS]

# Outro/trailing questions filler patterns to strip from the end of crop advice
_OUTRO_PATTERNS = [
    r"(?:మీకు ఇంకా ఏమైనా సహాయం కావాలా\??\s*|మీకు ఏమైనా సందేహాలు ఉంటే అడగండి[\.\?]?\s*|ఇంకా ఏదైనా సమాచారం కావాలంటే అడగండి[\.\?]?\s*|ధన్యవాదాలు[\.\!]?\s*|రైతే రాజు[\.\!]?\s*|శుభం[\.\!]?\s*)$",
    r"(?:do you need (?:any )?(?:further|more|other) assistance\??\s*|feel free to ask if you have (?:any )?questions[\.\?]?\s*|please let me know if you need anything else[\.\?]?\s*|let me know if you need anything else[\.\?]?\s*|thank you[\.\!]?\s*|thanks[\.\!]?\s*)$",
]

_COMPILED_OUTROS = [re.compile(p, re.IGNORECASE) for p in _OUTRO_PATTERNS]

# Keywords indicating pure agronomic/crop advice inquiry
_PURE_CROP_KEYWORDS_EN = [
    "spray", "disease", "pest", "fungus", "leaf", "rot", "spots", "dosage", "chemical",
    "pesticide", "bollworm", "alternaria", "control", "cure", "treatment", "prevent",
    "sowing", "stage", "cultivation", "water", "irrigate", "crop advice", "management",
    "how to", "symptoms", "deficiency", "fertilizer schedule", "what fertilizer", "which fertilizer",
    "how much fertilizer", "apply fertilizer", "blight", "blast", "rust", "wilt", "attack", "insects", "worms",
]

_PURE_CROP_KEYWORDS_TE = [
    "నివారణ", "తెగులు", "తెగుళ్ళు", "పురుగు", "పురుగులు", "ఆకు", "మచ్చలు", "మోతాదు",
    "పిచికారీ", "చికిత్స", "యాజమాన్యం", "సాగు", "లక్షణాలు", "ఎలా", "రోగం",
    "ఎరువుల మోతాదు", "మందు", "మందులు", "ఏం చేయాలి", "ఏమి చేయాలి", "రాలిపోవడం",
    "పచ్చదోమ", "తామర పురుగులు", "ఆల్టర్నేరియా", "అగ్గితెగులు", "ఎండిపోవడం", "పల్లాకు",
    "బూడిద తెగులు", "ఎరువు వాడాలి", "ఎరువులు వాడాలి", "పంట సలహా", "మందు పిచికారీ",
]

_PURE_BUY_PHRASES = [
    "where to buy", "where can i buy", "where i can buy", "shops near", "stores near",
    "dealer", "dealers", "buy urea", "buy dap", "buy pesticide", "buy seeds", "buy fertilizer",
    "కొనాలి", "ఎక్కడ దొరుకుతుంది", "ఎక్కడ కొనాలి", "దుకాణం", "దుకాణాలు", "షాపు", "షాపులు",
]

_WEATHER_PHRASES_EN = [
    "weather", "forecast", "rain", "raining", "rainy", "temperature",
    "wind", "humidity", "climate", "degree", "hot", "cold", "will it rain",
]

_WEATHER_PHRASES_TE = [
    "వాతావరణం", "వాతావరణ", "వర్షం", "వర్షాలు", "వాన", "కురుస్తుందా", "పడుతుందా",
    "ఉష్ణోగ్రత", "గాలి", "తేమ", "ఎండ", "చలి", "వాతావరణ అంచనా", "మంచు",
]

_MARKET_PHRASES_EN = [
    "market price", "mandi price", "price of", "prices of", "quintal",
    "market rate", "mandi rate", "selling price", "rate per quintal",
]

_MARKET_PHRASES_TE = [
    "మార్కెట్ ధర", "మార్కెట్ ధరలు", "మండి ధర", "మండి ధరలు", "క్వింటాల్", "క్వింటాలు",
    "ధర ఎంత", "రేటు ఎంత", "మార్కెట్లో", "అమ్ముకోవాలి", "గిట్టుబాటు ధర",
]

_SCHEME_PHRASES_EN = [
    "scheme", "schemes", "subsidy", "subsidies", "yojana", "kisan",
    "fasal bima", "insurance", "credit", "kcc", "solar pump", "government",
    "pm kisan", "rythu bandhu", "rythu bharosa", "kusum", "pmfby",
]

_SCHEME_PHRASES_TE = [
    "పథకం", "పథకాలు", "సబ్సిడీ", "సబ్సిడీలు", "పంట బీమా", "యోజన",
    "కిసాన్", "ప్రభుత్వ", "రైతు బంధు", "రైతు భరోసా", "అర్హత", "ప్రయోజనాలు",
    "క్రెడిట్ కార్డ్", "సౌర పంప్", "ఆర్థిక సహాయం", "గ్రాంట్",
]

_ESCALATION_PHRASES_EN = [
    "officer", "human", "expert", "scientist", "agent", "call", "talk", "escalation",
]

_ESCALATION_PHRASES_TE = [
    "అధికారి", "వ్యవసాయ అధికారి", "శాస్త్రవేత్త", "సంప్రదించండి", "హెల్ప్‌లైన్", "మాట్లాడాలి",
]


def detect_user_intents(user_message: str) -> Dict[str, bool]:
    """
    Analyze farmer user message to detect all requested domains/intents.
    """
    msg_lower = user_message.lower()

    # 1. Weather Intent
    has_weather = any(p in msg_lower for p in _WEATHER_PHRASES_EN) or any(p in user_message for p in _WEATHER_PHRASES_TE)

    # 2. Shop Intent
    has_shop = any(p in msg_lower for p in _PURE_BUY_PHRASES) or any(p in user_message for p in ["ఎక్కడ దొరుకుతుంది", "ఎక్కడ కొనాలి", "కొనాలి", "దుకాణం", "షాపు"])

    # 3. Market Intent
    has_market = (
        any(p in msg_lower for p in _MARKET_PHRASES_EN)
        or any(p in user_message for p in _MARKET_PHRASES_TE)
    )
    if not has_market:
        has_price_word = any(w in msg_lower for w in ["price", "rate", "mandi"]) or any(w in user_message for w in ["ధర", "రేటు"])
        if has_price_word and not has_shop:
            has_market = True

    # 4. Schemes Intent
    has_schemes = any(p in msg_lower for p in _SCHEME_PHRASES_EN) or any(p in user_message for p in _SCHEME_PHRASES_TE)
    # 5. Escalation Intent
    has_escalation = any(p in msg_lower for p in _ESCALATION_PHRASES_EN) or any(p in user_message for p in _ESCALATION_PHRASES_TE)

    # 6. Crop Advice Intent
    has_agri_kw = any(kw in msg_lower for kw in _PURE_CROP_KEYWORDS_EN) or any(kw in user_message for kw in _PURE_CROP_KEYWORDS_TE)
    is_pure_non_crop = (has_shop or has_weather or has_market or has_schemes) and not has_agri_kw
    has_crop_advice = has_agri_kw and not (is_pure_non_crop and not has_agri_kw)

    if not any([has_weather, has_shop, has_market, has_schemes, has_escalation]):
        has_crop_advice = True

    return {
        "crop_advice": has_crop_advice,
        "weather": has_weather,
        "shop": has_shop,
        "market": has_market,
        "schemes": has_schemes,
        "escalation": has_escalation,
    }


def clean_introductions(text: str) -> str:
    """Remove repetitive introductory greetings or preamble from text."""
    if not text:
        return ""
    cleaned = text.strip()
    changed = True
    while changed:
        changed = False
        for pattern in _COMPILED_INTROS:
            new_text = pattern.sub("", cleaned).strip()
            if new_text != cleaned:
                cleaned = new_text
                changed = True
    return cleaned.strip()


def clean_outros(text: str) -> str:
    """Remove repetitive trailing questions or closings from text."""
    if not text:
        return ""
    cleaned = text.strip()
    changed = True
    while changed:
        changed = False
        for pattern in _COMPILED_OUTROS:
            new_text = pattern.sub("", cleaned).strip()
            if new_text != cleaned:
                cleaned = new_text
                changed = True
    return cleaned.strip()


def clean_crop_advice_for_multi_intent(ai_text: str) -> str:
    """
    Clean speculative weather, market price, shop ads, scheme summaries, or refusal statements from primary AI text
    so it cleanly contains pure agronomic advisory when specialized structured enrichments exist.
    """
    if not ai_text:
        return ""

    refusal_markers = [
        "e-nam", "ఈ-నామ్", "ఈ - నామ్", "మార్కెట్ యార్డ్", "మార్కెట్ యార్డు",
        "market yard", "కేవలం వ్యవసాయం", "విషయాలపై మాత్రమే", "i can only help with farming",
        "only help with farming", "how can i help with your crops",
        "క్షమించండి, ప్రస్తుతం కనెక్ట్ అవడంలో", "i'm sorry, i'm having trouble connecting",
    ]
    ai_lower = ai_text.lower()
    for marker in refusal_markers:
        if marker in ai_lower and len(ai_text.strip().split("\n")) <= 2:
            return ""

    speculative_sentence_markers = [
        "ధర", "ధరలు", "క్వింటాల్", "క్వింటాలు", "రేటు", "రేట్లు",
        "price", "prices", "mandi", "rate", "rates", "quintal",
        "వాతావరణం విషయానికి వస్తే", "వాతావరణం గురించి", "regarding weather",
        "as for the weather", "weather forecast shows", "వర్షం పడే అవకాశం",
        "will rain", "rain expected", "ఉష్ణోగ్రత", "weather is expected",
        "forecast", "degree", "weather",
        "సమీప డీలర్ల", "స్థానిక డీలర్ల", "దుకాణాల్లో దొరుకుతుంది", "దొరుకుతుంది",
        "you can buy from local", "available at nearby shops", "buy urea at", "where to buy",
        "లభిస్తుంది", "కొనుగోలు చేయవచ్చు", "లభ్యత", "దుకాణాల్లో", "డీలర్ల వద్ద",
        "ప్రభుత్వ పథకాలు", "పిఎం కిసాన్", "రైతు బంధు", "పథకం ద్వారా", "పథకం కింద",
        "government schemes", "pm kisan", "rythu bandhu", "subsidy is available",
        "scheme", "schemes", "yojana", "kisan samman",
    ]

    cleaned_paragraphs = []
    for para in ai_text.split("\n"):
        para = para.strip()
        if not para:
            continue
        sentences = [s.strip() for s in re.split(r'(?<=[।\.\?\!])\s+', para) if s.strip()]
        valid_sentences = [
            s for s in sentences
            if not any(marker in s.lower() for marker in speculative_sentence_markers)
        ]
        if valid_sentences:
            cleaned_paragraphs.append(" ".join(valid_sentences))

    result = "\n".join(cleaned_paragraphs).strip()
    result = clean_introductions(result)
    result = clean_outros(result)
    return result


def decompose_assembled_response(assembled_text: str) -> Dict[str, str]:
    """
    Parse an assembled response string into discrete functional sections:
    - crop_advice (base AI text)
    - shop (🏬 ...)
    - market (📊 ...)
    - weather (🌡️ ...)
    - schemes (🏛️ ...)
    - escalation (👨‍🌾 ...)
    """
    sections: Dict[str, str] = {
        "crop_advice": "",
        "weather": "",
        "shop": "",
        "market": "",
        "schemes": "",
        "escalation": "",
    }

    if not assembled_text:
        return sections

    pattern = r"(?=(?:^|\n\n)(?:🏬|📊|🌡️|🌦️|🏛️|👨‍🌾))"
    chunks = re.split(pattern, assembled_text.strip())

    base_ai_parts = []
    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue

        if chunk.startswith("🏬") or "Available Nearby Shops" in chunk or "సమీప వ్యవసాయ దుకాణాలు" in chunk or "Nearby Agricultural Shops" in chunk:
            sections["shop"] = chunk
        elif chunk.startswith("📊") or "Mandi Prices" in chunk or "మార్కెట్ ధరలు" in chunk:
            sections["market"] = chunk
        elif chunk.startswith("🌡️") or chunk.startswith("🌦️") or "Weather Information" in chunk or "వాతావరణ సమాచారం" in chunk:
            sections["weather"] = chunk
        elif chunk.startswith("🏛️") or "Government Schemes" in chunk or "ప్రభుత్వ పథకాలు" in chunk:
            sections["schemes"] = chunk
        elif chunk.startswith("👨‍🌾") or "Escalation Ticket" in chunk or "సంప్రదింపు టికెట్" in chunk:
            sections["escalation"] = chunk
        else:
            base_ai_parts.append(chunk)

    if base_ai_parts:
        sections["crop_advice"] = "\n\n".join(base_ai_parts).strip()

    return sections


def compact_section(section_key: str, section_text: str, language: str = "en") -> str:
    """
    Format and compact an individual section for multi-intent display on WhatsApp.
    Ensures high density, removes duplicate header lines, and keeps top items.
    """
    headers = SECTION_HEADERS_TE if language == "te" else SECTION_HEADERS_EN
    section_header = headers.get(section_key, "")

    if not section_text or not section_text.strip():
        unavail = _UNAVAILABLE_LABELS_TE if language == "te" else _UNAVAILABLE_LABELS_EN
        msg = unavail.get(section_key, "")
        if msg:
            return f"{section_header}\n{msg}"
        return ""

    lines = [line.strip() for line in section_text.split("\n") if line.strip()]
    if not lines:
        return ""

    if section_key == "crop_advice":
        cleaned = clean_crop_advice_for_multi_intent(section_text)
        if not cleaned:
            return ""
        return f"{section_header}\n{cleaned}"

    elif section_key == "weather":
        body_lines = []
        for line in lines:
            if (line.startswith("🌡️ ") or line.startswith("🌦️ ")) and ("Weather Information" in line or "వాతావరణ సమాచారం" in line):
                loc_match = re.search(r"\((.*?)\)", line)
                loc_str = f" ({loc_match.group(1)})" if loc_match else ""
                section_header = f"🌡️ *{'వాతావరణ సమాచారం' if language == 'te' else 'Weather Information'}*{loc_str}"
                continue
            if line.startswith("📡"):
                continue
            body_lines.append(line)
        return f"{section_header}\n" + "\n".join(body_lines)

    elif section_key == "market":
        body_lines = []
        for line in lines:
            if line.startswith("📊"):
                section_header = line
                continue
            if line.startswith("📡"):
                continue
            body_lines.append(line)
        return f"{section_header}\n" + "\n".join(body_lines)

    elif section_key == "shop":
        body_lines = []
        for line in lines:
            if line.startswith("🏬"):
                continue
            if line.startswith("ℹ️") or "Find all shops at:" in line or "మరిన్ని దుకాణాల కోసం:" in line:
                continue
            body_lines.append(line)
        return f"{section_header}:\n" + "\n".join(body_lines)

    elif section_key == "schemes":
        body_lines = []
        for line in lines:
            if line.startswith("🏛️"):
                continue
            if line.startswith("⚠️") or "See all schemes at:" in line or "మరిన్ని పథకాల కోసం:" in line:
                continue
            body_lines.append(line)
        return f"{section_header}:\n" + "\n".join(body_lines)

    elif section_key == "escalation":
        body_lines = []
        for line in lines:
            if line.startswith("━━━━━━━━━━━━━━━━━━━━━━"):
                continue
            body_lines.append(line)
        return "\n".join(body_lines)

    return section_text


def format_multi_intent_response(
    assembled_text: str,
    user_message: str = "",
    language: str = "en",
) -> str:
    """
    Main entry point for WhatsApp response optimization and authoritative deduplication.

    - If single-intent: returns the response untouched/preserved for backwards compatibility.
    - If multi-intent (2+ domains detected):
      1. Detects which intents the farmer actually asked about.
      2. Strictly retains only ONE authoritative response per requested intent.
      3. Discards unrequested sections and generic AI summaries for domains covered by specialized modules.
      4. Cleans repetitive intros and outros.
      5. Organizes in logical order: Crop Advice -> Weather -> Shops -> Market -> Schemes -> Escalation.
    """
    if not assembled_text:
        return ""

    user_intents = detect_user_intents(user_message) if user_message else {}
    requested_intent_count = sum(1 for k, v in user_intents.items() if v)

    sections = decompose_assembled_response(assembled_text)

    active_enrichments = [
        k for k in ["weather", "shop", "market", "schemes", "escalation"]
        if sections[k] and sections[k].strip()
    ]

    has_crop = bool(sections["crop_advice"] and sections["crop_advice"].strip())
    has_requested_crop = user_intents.get("crop_advice", False)

    is_multi_intent = False
    if requested_intent_count >= 2:
        is_multi_intent = True
    elif len(active_enrichments) >= 2:
        is_multi_intent = True
    elif len(active_enrichments) == 1 and has_crop and has_requested_crop:
        is_multi_intent = True

    if not is_multi_intent:
        return assembled_text.strip()

    ordered_keys = ["crop_advice", "weather", "shop", "market", "schemes", "escalation"]
    formatted_blocks: List[str] = []

    for key in ordered_keys:
        if user_intents and not user_intents.get(key, False):
            continue

        section_raw = sections.get(key, "")
        compacted = compact_section(key, section_raw, language=language)
        if compacted and compacted.strip():
            formatted_blocks.append(compacted.strip())

    if not formatted_blocks:
        return assembled_text.strip()

    final_output = "\n\n".join(formatted_blocks).strip()
    return final_output
