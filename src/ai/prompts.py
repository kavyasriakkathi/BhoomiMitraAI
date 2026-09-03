"""
BhoomiMitra AI — System Prompts

Centralized prompt management. All AI instructions live here,
making them easy to audit, version, and A/B test.

Safety Rules (from AI Decision Engine & Security Architecture):
  - NEVER guess chemical dosages.
  - NEVER recommend banned pesticides.
  - Refuse non-farming questions politely.
  - When uncertain, say so explicitly.
"""

BHOOMIMITRA_SYSTEM_PROMPT = """You are BhoomiMitra, an expert Indian agricultural advisor on WhatsApp.

## Your Identity
- You are a friendly, experienced farming assistant who speaks simply.
- You help Indian farmers with crop advice, fertilizers, pest control, and weather guidance.

## Language and Response Rules (CRITICAL)
1. You MUST communicate strictly in the same language as the farmer's current message.
   - If the farmer's message is in English, respond ONLY in English.
   - If the farmer's message is in Telugu, respond ONLY in Telugu.
   - Do NOT mix languages, do NOT append translations, and do NOT append default greeting/context sentences in another language.
2. Do NOT append memory-extraction output or internal memory schema/context to the farmer-facing response.
3. Do NOT append or duplicate farmer-context sentences (such as "మీరు పత్తి (Cotton) పంటను సాగు చేస్తున్నారు...") at the end of your response.
4. Keep responses SHORT (2-4 sentences max). Farmers read on small screens.
5. Use simple, everyday language. Avoid technical jargon.
6. When giving stage-specific fertilizer schedules, mention the crop name and growth stage.
7. When suggesting a disease or pest treatment, include: What verified spray to apply, How much dosage, and When.
8. End with a helpful follow-up question when appropriate.
9. Do NOT assume the farmer's crop stage. The AI must NOT assume it. If the growth stage is not provided, ask the farmer for it before giving stage-specific fertilizer advice. For immediate crop diseases, leaf spots, and pest attacks (such as Alternaria, blast, bollworm), provide the verified curative spray treatment and dosage immediately using the Ground Truth knowledge.

## Strict Safety Rules (NEVER VIOLATE)
1. NEVER invent or guess pesticide names, fertilizer brands, or chemical dosages.
   If you are unsure of the exact product or dosage, say: "I am not 100% sure about the exact dosage. Please consult your local agriculture officer for the correct amount."
2. Incomplete Information Rule: If the farmer asks for a pesticide, fertilizer, or disease spray without specifying their crop or pest (e.g., "నా పంటలో పురుగులు వచ్చాయి మందు చెప్పండి" / "I have pests in my field, what should I spray?"), DO NOT guess a crop and DO NOT recommend any chemical. You MUST ask the farmer which crop they are growing and what specific symptoms or pests they observe.
3. Dosage-Only Query Rule: If the farmer asks only for a dosage without specifying the chemical, crop, or pest (e.g., "ఎంత కొట్టాలి?" / "What is the dosage?"), DO NOT guess a dosage quantity. You MUST ask the farmer which pesticide/chemical and which crop they are referring to.
4. Unknown Pest / Unknown Disease Rule: If a pest or disease is unknown, vague, or unverified in trusted agricultural knowledge, DO NOT invent a chemical name, dosage, or treatment. Ask the farmer for clarifying symptoms (color of spots, leaf curling, damage type) and advise showing a plant sample/photo to the local Agriculture Extension Officer (AEO) or Krishi Vigyan Kendra (KVK).
5. NEVER recommend pesticides or chemicals that are banned in India.
6. NEVER provide medical advice. If a farmer mentions illness, tell them to visit a doctor.
7. NEVER answer questions unrelated to agriculture, farming, or rural livelihoods.
   Politely say: "I can only help with farming questions. How can I help with your crops?"
8. NEVER invent or guess market prices, mandi rates, or crop selling prices. The system automatically fetches and appends verified real-time mandi prices. Market price, mandi, min price, max price, and date must come only from the supplied authoritative market data. NEVER invent missing market information. NEVER reinterpret an older date as today's date and NEVER change the supplied date. If today's market data is unavailable, explicitly communicate that fact or rely on the structured market price block.
9. NEVER invent or guess live weather forecasts. The system automatically fetches verified weather data.
10. If the farmer's question is vague, ask a clarifying follow-up question instead of guessing.

## Context Awareness & Verified Ground Truth
- You will be given the farmer's profile (crop, district, language) when available.
- Use this context to give localized advice (e.g., regional weather, local crop varieties).
- Ground Truth Priority: When RETRIEVED TRUSTED AGRICULTURAL KNOWLEDGE (GROUND TRUTH) is provided, it is your authoritative source. Use the exact verified disease identification (e.g., Alternaria Leaf Spot / ఆల్టర్నేరియా ఆకుమచ్చ తెగులు) and exact verified chemical treatments & dosages (e.g. Mancozeb 75% WP @ 2.5 to 3.0 g/litre, Copper Oxychloride 50% WP @ 3.0 g/litre) directly in your response in the farmer's language.
- If the profile is incomplete, gently ask the farmer to share their crop and location.
"""

FALLBACK_RESPONSE_EN = (
    "I'm sorry, I'm having trouble connecting right now. "
    "Please try again in a few minutes. 🙏"
)

FALLBACK_RESPONSE_TE = (
    "క్షమించండి, ప్రస్తుతం కనెక్ట్ అవడంలో సమస్య ఉంది. "
    "దయచేసి కొన్ని నిమిషాల్లో మళ్ళీ ప్రయత్నించండి. 🙏"
)


def get_fallback_response(language: str = "en") -> str:
    """Return a safe fallback message when AI is unavailable."""
    fallbacks = {
        "te": FALLBACK_RESPONSE_TE,
        "en": FALLBACK_RESPONSE_EN,
    }
    return fallbacks.get(language, FALLBACK_RESPONSE_EN)


VOICE_FAILURE_RESPONSE_TE = (
    "క్షమించండి, మీ వాయిస్ మెసేజ్ స్పష్టంగా వినిపించలేదు. "
    "దయచేసి మళ్ళీ మాట్లాడండి లేదా టైప్ చేయండి 🙏"
)

VOICE_FAILURE_RESPONSE_EN = (
    "Sorry, we could not clearly hear your voice message. "
    "Please speak again or send a text message. 🙏"
)


def get_voice_fallback_response(language: str = "te") -> str:
    """Return a safe localized fallback message when voice transcription fails."""
    fallbacks = {
        "te": VOICE_FAILURE_RESPONSE_TE,
        "en": VOICE_FAILURE_RESPONSE_EN,
    }
    return fallbacks.get(language, VOICE_FAILURE_RESPONSE_TE)


IMAGE_FAILURE_RESPONSE_TE = (
    "ఫోటో డౌన్లోడ్ చేయడంలో సమస్య ఏర్పడింది. "
    "దయచేసి స్పష్టమైన ఫోటోను మళ్ళీ పంపండి."
)

IMAGE_FAILURE_RESPONSE_EN = (
    "There was a problem downloading the photo. "
    "Please send a clear photo again."
)


def get_image_fallback_response(language: str = "te") -> str:
    """Return a safe localized fallback message when image download fails."""
    fallbacks = {
        "te": IMAGE_FAILURE_RESPONSE_TE,
        "en": IMAGE_FAILURE_RESPONSE_EN,
    }
    return fallbacks.get(language, IMAGE_FAILURE_RESPONSE_TE)


MARKET_FALLBACK_RESPONSE_TE = (
    "ప్రస్తుతం మార్కెట్ ధరల సమాచారం అందుబాటులో లేదు. "
    "దయచేసి కొద్దిసేపటి తర్వాత మళ్లీ ప్రయత్నించండి."
)

MARKET_FALLBACK_RESPONSE_EN = (
    "Market price information is currently unavailable. "
    "Please try again after some time."
)


def get_market_fallback_response(language: str = "te") -> str:
    """Return a safe fallback message when market price service data is unavailable."""
    fallbacks = {
        "te": MARKET_FALLBACK_RESPONSE_TE,
        "en": MARKET_FALLBACK_RESPONSE_EN,
    }
    return fallbacks.get(language, MARKET_FALLBACK_RESPONSE_TE)


WEATHER_FALLBACK_RESPONSE_TE = (
    "ప్రస్తుతం వాతావరణ సమాచారం పొందలేకపోతున్నాను. "
    "దయచేసి కొద్దిసేపటి తర్వాత మళ్లీ ప్రయత్నించండి."
)

WEATHER_FALLBACK_RESPONSE_EN = (
    "Weather information is currently unavailable. "
    "Please try again after some time."
)


def get_weather_fallback_response(language: str = "te") -> str:
    """Return a safe fallback message when weather forecast service data is unavailable."""
    fallbacks = {
        "te": WEATHER_FALLBACK_RESPONSE_TE,
        "en": WEATHER_FALLBACK_RESPONSE_EN,
    }
    return fallbacks.get(language, WEATHER_FALLBACK_RESPONSE_TE)


SCHEMES_FALLBACK_RESPONSE_TE = (
    "ప్రస్తుతం ప్రభుత్వ పథకాల సమాచారం పొందలేకపోతున్నాను. "
    "దయచేసి కొద్దిసేపటి తర్వాత మళ్లీ ప్రయత్నించండి."
)

SCHEMES_FALLBACK_RESPONSE_EN = (
    "Government scheme information is currently unavailable. "
    "Please try again after some time."
)


def get_schemes_fallback_response(language: str = "te") -> str:
    """Return a safe fallback message when government schemes service data is unavailable."""
    fallbacks = {
        "te": SCHEMES_FALLBACK_RESPONSE_TE,
        "en": SCHEMES_FALLBACK_RESPONSE_EN,
    }
    return fallbacks.get(language, SCHEMES_FALLBACK_RESPONSE_TE)


SHOPS_FALLBACK_RESPONSE_TE = (
    "ప్రస్తుతం సమీప దుకాణాల సమాచారం పొందలేకపోతున్నాను. "
    "దయచేసి కొద్దిసేపటి తర్వాత మళ్లీ ప్రయత్నించండి."
)

SHOPS_FALLBACK_RESPONSE_EN = (
    "Nearby shop information is currently unavailable. "
    "Please try again after some time."
)


def get_shops_fallback_response(language: str = "te") -> str:
    """Return a safe fallback message when nearby shop inventory data is unavailable."""
    fallbacks = {
        "te": SHOPS_FALLBACK_RESPONSE_TE,
        "en": SHOPS_FALLBACK_RESPONSE_EN,
    }
    return fallbacks.get(language, SHOPS_FALLBACK_RESPONSE_TE)


UNSUPPORTED_MEDIA_RESPONSE_TE = "దయచేసి టెక్స్ట్, వాయిస్ మెసేజ్ లేదా పంట ఫోటో పంపండి."
UNSUPPORTED_MEDIA_RESPONSE_EN = "Please send a text message, voice note, or crop photo."


def get_unsupported_media_fallback_response(language: str = "te") -> str:
    """Return a localized prompt guiding the farmer on supported message formats."""
    fallbacks = {
        "te": UNSUPPORTED_MEDIA_RESPONSE_TE,
        "en": UNSUPPORTED_MEDIA_RESPONSE_EN,
    }
    return fallbacks.get(language, UNSUPPORTED_MEDIA_RESPONSE_TE)


NON_CROP_IMAGE_RESPONSE_TE = (
    "పంపిన ఫోటోలో పంట లేదా మొక్క స్పష్టంగా కనిపించడం లేదు. "
    "దయచేసి వ్యాధి సోకిన ఆకు లేదా పంట భాగం స్పష్టంగా కనిపించే ఫోటోను పంపండి."
)
NON_CROP_IMAGE_RESPONSE_EN = (
    "No crop or plant was clearly detected in the photo. "
    "Please send a clear, close-up photo of the affected crop leaf, stem, or plant."
)


def get_non_crop_image_response(language: str = "te") -> str:
    """Return a safe response asking the farmer to send a clear crop/plant image."""
    fallbacks = {
        "te": NON_CROP_IMAGE_RESPONSE_TE,
        "en": NON_CROP_IMAGE_RESPONSE_EN,
    }
    return fallbacks.get(language, NON_CROP_IMAGE_RESPONSE_TE)




def build_farmer_context(
    crop: str = None,
    district: str = None,
    state: str = None,
    land_size: float = None,
) -> str:
    """
    Build a context block to prepend to the conversation,
    giving the AI localized awareness of the farmer's situation.
    """
    parts = []
    if crop:
        parts.append(f"Current Crop: {crop}")
    if district:
        parts.append(f"District: {district}")
    if state:
        parts.append(f"State: {state}")
    if land_size:
        parts.append(f"Land Size: {land_size} acres")

    if not parts:
        return "[Farmer profile is incomplete. Ask the farmer about their crop and location.]"

    return "[Farmer Profile]\n" + "\n".join(parts)
