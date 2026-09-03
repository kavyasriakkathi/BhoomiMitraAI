"""
BhoomiMitra AI — AI Decision Engine & Orchestration Layer

The central decision-making brain of BhoomiMitra AI.
Responsible for:
1. Multi-lingual intent classification (English, Telugu, Tanglish, Mixed).
2. Authoritative module routing (Market, Weather, Schemes, Shops, Advisory, Vision).
3. Anti-hallucination and agricultural safety enforcement.
4. Factual data guarantees (real data is preserved exactly; LLM never guesses prices/weather/dosages).
5. Single-intent and multi-intent response assembly with localized fallbacks.
6. Guarding greetings against unnecessary expensive enrichments.
"""
import re
from enum import Enum
from typing import List, Dict, Optional, Tuple
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.models import Farmer, Conversation
from src.core.logging import logger
from src.ai.prompts import (
    get_fallback_response,
    get_market_fallback_response,
    get_weather_fallback_response,
    get_schemes_fallback_response,
    get_shops_fallback_response,
)


class FarmerIntent(str, Enum):
    GREETING = "greeting"
    CROP_ADVICE = "crop_advice"
    CROP_HEALTH = "crop_health"  # disease / pest
    FERTILIZER = "fertilizer_nutrient"  # fertilizer / nutrient advice
    IRRIGATION = "irrigation"
    WEATHER = "weather"
    MARKET_PRICE = "market_price"
    GOVERNMENT_SCHEMES = "government_schemes"
    SHOPS = "shops_input_availability"
    SOWING = "sowing"
    HARVESTING = "harvesting"
    REMINDERS = "reminders"
    IMAGE_DIAGNOSIS = "image_diagnosis"
    GENERAL_FARMING = "general_farming"
    UNKNOWN = "unsupported_unknown"


# ─────────────────────────────────────────────────────────────────────────────
# Intent Keyword & Pattern Mappings (English, Telugu Unicode, Romanized Tanglish)
# ─────────────────────────────────────────────────────────────────────────────

GREETING_PATTERNS = [
    r"\b(?:hello|hi|hey|hai|helo|namaste|namaskar|namaskaram|namaskaralu)\b",
    r"\b(?:good\s+morning|good\s+afternoon|good\s+evening|greetings)\b",
    r"^(?:నమస్తే|నమస్కారం|నమస్కారాలు|హలో|హాయ్)[\s\!,\.]*$",
    r"^(?:హాయ్|నమస్తే|హలో|నమస్కారం)\s+(?:భూమిమిత్ర|bhoomimitra|రైతు\s*మిత్ర|రైతు\s*అన్న|రైతు)?[\s\!,\.]*$",
]

MARKET_PRICE_KEYWORDS_EN = [
    "market price", "mandi price", "mandi rate", "market rate", "selling price",
    "rate per quintal", "cotton price", "cotton rate", "paddy price", "chilli price",
    "tomato price", "price of", "prices of", "how much price", "how much rate",
    "market value", "mandi rates", "market prices",
]

MARKET_PRICE_KEYWORDS_TE = [
    "మార్కెట్ ధర", "మార్కెట్ ధరలు", "మండి ధర", "మండి ధరలు", "ధర ఎంత", "రేటు ఎంత",
    "క్వింటాల్", "క్వింటాలు", "అమ్ముకోవాలి", "గిట్టుబాటు ధర", "మార్కెట్లో", "మండిలో",
    "పత్తి ధర", "మిర్చి ధర", "వరి ధర", "టమాటా ధర", "రేట్లు", "ధరలు",
]

MARKET_PRICE_KEYWORDS_TANGLISH = [
    "rate entha", "dhara entha", "rate entho", "dhara entho", "cotton rate", "patti rate",
    "patti dhara", "mirchi rate", "mirapa rate", "tomato rate", "tamata rate", "mandi rate",
    "market lo", "mandi lo", "market price", "mandi price", "bajar rate", "ammukovali",
    "eeroju rate", "today rate", "rate ela undi", "dhara ela undi", "per quintal rate",
    "lo cotton rate", "lo patti rate", "patti rate entha", "cotton rate entha",
]

WEATHER_KEYWORDS_EN = [
    "weather", "forecast", "rain", "raining", "rainy", "temperature", "humidity",
    "wind", "storm", "thunderstorm", "will it rain", "precipitation", "cloudy", "sunny",
    "climate", "degrees",
]

WEATHER_KEYWORDS_TE = [
    "వాతావరణం", "వాతావరణ", "వర్షం", "వర్షాలు", "వాన", "వానలు", "కురుస్తుందా",
    "పడుతుందా", "కురుస్తుంది", "పడుతుంది", "ఉష్ణోగ్రత", "తేమ", "ఎండ", "చలి",
    "వాతావరణ అంచనా", "మంచు", "తుఫాను", "జల్లులు", "మేఘాలు",
]

WEATHER_KEYWORDS_TANGLISH = [
    "varsham", "varsham paduthunda", "varsham vasthunda", "vana vasthunda",
    "vana paduthunda", "weather ela undi", "eeroju varsham", "repu varsham",
    "eppudu paduthundi", "rain paduthunda", "rain vasthada", "temperature entha",
    "cloudy ga undi", "varsham padtada", "varsham padtundha", "rain padtundha",
    "varsham eppudu", "varsham ela",
]

SCHEMES_KEYWORDS_EN = [
    "scheme", "schemes", "subsidy", "subsidies", "yojana", "pm kisan", "rythu bandhu",
    "rythu bharosa", "crop insurance", "fasal bima", "kcc", "kisan credit",
    "solar pump subsidy", "government assistance", "grant", "subsidized",
]

SCHEMES_KEYWORDS_TE = [
    "పథకం", "పథకాలు", "సబ్సిడీ", "సబ్సిడీలు", "రైతు బంధు", "రైతు భరోసా", "పీఎం కిసాన్",
    "పంట బీమా", "రుణమాఫీ", "ప్రభుత్వ సహాయం", "ప్రభుత్వ", "అర్హత", "ప్రయోజనాలు",
    "ప్రభుత్వ పథకాలు", "రైతు పథకాలు",
]

SCHEMES_KEYWORDS_TANGLISH = [
    "pathakam", "pathakalu", "prabhutva pathakalu", "rythu pathakalu", "subsidilu",
    "subsidy", "pm kisan", "rythu bharosa", "rythu bandhu", "panta bheema",
    "kisan subsidy", "gov schemes", "sarakaru sahaym", "pathakam apply",
    "prabhutva", "pathakalu unnaya",
]

SHOPS_KEYWORDS_EN = [
    "where to buy", "where can i buy", "where i can buy", "shops near", "stores near",
    "buy urea", "buy dap", "dealer", "dealers", "input availability", "available in shop",
    "fertilizer store", "pesticide store", "buy fertilizer", "buy pesticide", "buy seeds",
    "shops nearby", "store nearby", "agro agency",
]

SHOPS_KEYWORDS_TE = [
    "ఎక్కడ దొరుకుతుంది", "ఎక్కడ కొనాలి", "సమీప దుకాణాలు", "ఎరువుల దుకాణం", "మందుల షాపు",
    "కొనుగోలు", "దుకాణం", "షాపు", "లభిస్తుంది", "యూరియా దొరుకుతుందా", "డీలర్", "డీలర్లు",
    "దుకాణాలు", "షాపులు", "దొరికే చోటు",
]

SHOPS_KEYWORDS_TANGLISH = [
    "ekkada dorukuthundi", "ekkada konali", "shops ekkada", "shop ekkada", "urea ekkada",
    "dap ekkada", "seeds ekkada", "fertilizer shop", "pesticide shop", "near shops",
    "daggara shop", "konadaniki", "dorukuthunda", "shops daggara", "dealer daggara",
    "dorukutundi", "konachu",
]

CROP_HEALTH_KEYWORDS_EN = [
    "disease", "pest", "pests", "leaf spot", "yellowing", "wilting", "fungus",
    "insects", "bollworm", "aphids", "whitefly", "blight", "rot", "pesticide for",
    "cure", "symptoms", "worms", "bugs", "fungicide", "infestation", "stem borer",
    "leaf curl", "caterpillar", "blast", "rust", "alternaria", "powdery mildew",
]

CROP_HEALTH_KEYWORDS_TE = [
    "తెగులు", "తెగుళ్ళు", "పురుగు", "పురుగులు", "ఆకుమచ్చ", "పసుపుగా", "రాలిపోవడం",
    "ముడత", "ఎండిపోవడం", "పచ్చదోమ", "తామర పురుగులు", "బూడిద తెగులు", "అగ్గితెగులు",
    "ఆల్టర్నేరియా", "నివారణ", "మందు పిచికారీ", "రోగం", "లక్షణాలు", "మచ్చలు", "పురుగుల",
]

CROP_HEALTH_KEYWORDS_TANGLISH = [
    "tegulu", "purugu", "purugulu", "aakulu pasupuga", "aaku machalu", "marutunnayi",
    "endipotundi", "mudatha", "dosa penu", "pacha doma", "kurchuku potundi",
    "panta rogamu", "pesticide spray", "cheda", "pulla rali", "pula rali",
    "purugula mandu", "tegulu mandu", "ralipothundi", "pasupuga marutunnayi",
]

FERTILIZER_KEYWORDS_EN = [
    "fertilizer", "fertilizers", "nutrient", "npk", "urea application", "which fertilizer",
    "what fertilizer", "micronutrient", "zinc deficiency", "potash dose", "fertilizer schedule",
    "how much fertilizer", "apply fertilizer", "manure", "compost", "dosage of urea", "dap dose",
]

FERTILIZER_KEYWORDS_TE = [
    "ఎరువు", "ఎరువులు", "ఏ ఎరువు వేయాలి", "ఎరువుల యాజమాన్యం", "పోషకాలు", "నత్రజని",
    "భాస్వరం", "పొటాష్", "సూక్ష్మ పోషకాలు", "ఎరువుల మోతాదు", "జింక్ లోపం", "ఎరువు వాడాలి",
    "బాస్వరం", "యూరియా మోతాదు",
]

FERTILIZER_KEYWORDS_TANGLISH = [
    "e eruvu veyali", "eruvulu eppudu", "fertilizer eppudu", "npk ela veyali", "poshakalu",
    "eruvu dose", "micronutrients", "urea dose", "e eruvu vadali", "fertilizer schedule",
    "eruvu entha", "eruvulu ela", "fertilizer dose",
]

IRRIGATION_KEYWORDS_EN = [
    "irrigation", "watering", "water schedule", "drip irrigation", "how much water",
    "when to water", "moisture", "flood irrigation", "sprinkler",
]

IRRIGATION_KEYWORDS_TE = [
    "నీరు", "నీటి యాజమాన్యం", "నీరు పెట్టాలి", "తడి ఇవ్వాలి", "డ్రిప్", "బిందు సేద్యం",
    "ఎప్పుడు నీరు", "నీరు కట్టాలి", "నీటి తడి", "నీటి పారుదల",
]

IRRIGATION_KEYWORDS_TANGLISH = [
    "neeru eppudu pettali", "thadi eppudu ivvali", "neeti yajamanyam", "water eppudu pettali",
    "drip irrigation", "water kattachu", "thadulu eppudu", "neeru eppudu",
]

SOWING_KEYWORDS_EN = [
    "sowing", "seed rate", "seed treatment", "planting time", "how to sow", "seed spacing",
    "nursery", "germination", "transplanting", "sowing depth",
]

SOWING_KEYWORDS_TE = [
    "విత్తనాలు", "విత్తన శుద్ధి", "విత్తే సమయం", "నాట్లు", "నాటడం", "ఎప్పుడు విత్తాలి",
    "విత్తన మోతాదు", "మొలక", "నారుమడి",
]

SOWING_KEYWORDS_TANGLISH = [
    "vithanalu eppudu veyali", "vithana shuddi", "sowing time", "seed rate entha",
    "natlu eppudu", "vithadam ela", "seeds eppudu",
]

HARVESTING_KEYWORDS_EN = [
    "harvesting", "harvest time", "when to harvest", "picking cotton", "harvest maturity",
    "threshing", "storage", "post harvest",
]

HARVESTING_KEYWORDS_TE = [
    "కోత", "కోత సమయం", "ఎప్పుడు కోయాలి", "పత్తి ఏరడం", "దిగుబడి", "నిల్వ", "కోయడం", "కోతలు",
]

HARVESTING_KEYWORDS_TANGLISH = [
    "kotha eppudu koyali", "harvesting time", "patti eppudu thiyyali", "kotha kosaru",
    "nilva cheyadam", "digubadi", "kotha samayam",
]

REMINDERS_KEYWORDS_EN = [
    "remind me", "set reminder", "schedule reminder", "alert me", "reminder",
]

REMINDERS_KEYWORDS_TE = [
    "గుర్తు చేయండి", "రిమైండర్", "షెడ్యూల్", "గుర్తుపెట్టుకో",
]

REMINDERS_KEYWORDS_TANGLISH = [
    "remind cheyandi", "gurthu cheyandi", "reminder pettandi", "schedule cheyandi",
]


class AIDecisionEngine:
    """
    Orchestration layer determining intents, data dependencies, and routing.
    """

    @staticmethod
    def is_greeting_only(user_message: str) -> bool:
        """
        Check whether the message is strictly a greeting/courtesy without any factual question.
        Guarantees greetings never trigger expensive external queries.
        """
        if not user_message or not user_message.strip():
            return False

        msg = user_message.strip().lower()

        # Check against pure greeting patterns
        is_greeting_match = any(re.search(pat, msg, re.IGNORECASE) for pat in GREETING_PATTERNS)

        if not is_greeting_match:
            tokens = [t.strip(",.!?") for t in msg.split() if t.strip(",.!?")]
            if len(tokens) <= 3 and any(t in ["hi", "hello", "hey", "namaste", "namaskaram", "హాయ్", "నమస్తే", "హలో"] for t in tokens):
                is_greeting_match = True

        if not is_greeting_match:
            return False

        # If any domain keywords are present, it's not a pure greeting
        domain_keywords = (
            MARKET_PRICE_KEYWORDS_EN + MARKET_PRICE_KEYWORDS_TE + MARKET_PRICE_KEYWORDS_TANGLISH +
            WEATHER_KEYWORDS_EN + WEATHER_KEYWORDS_TE + WEATHER_KEYWORDS_TANGLISH +
            SCHEMES_KEYWORDS_EN + SCHEMES_KEYWORDS_TE + SCHEMES_KEYWORDS_TANGLISH +
            SHOPS_KEYWORDS_EN + SHOPS_KEYWORDS_TE + SHOPS_KEYWORDS_TANGLISH +
            CROP_HEALTH_KEYWORDS_EN + CROP_HEALTH_KEYWORDS_TE + CROP_HEALTH_KEYWORDS_TANGLISH +
            FERTILIZER_KEYWORDS_EN + FERTILIZER_KEYWORDS_TE + FERTILIZER_KEYWORDS_TANGLISH +
            IRRIGATION_KEYWORDS_EN + IRRIGATION_KEYWORDS_TE + IRRIGATION_KEYWORDS_TANGLISH
        )

        for kw in domain_keywords:
            if kw in msg:
                return False

        return True

    @staticmethod
    def get_greeting_reply(language: str = "te") -> str:
        """Return a helpful, welcoming WhatsApp greeting response."""
        if language == "te":
            return (
                "నమస్తే! నేను మీ భూమిమిత్ర AI వ్యవసాయ సహాయకుడిని. 🙏\n\n"
                "పంట సలహాలు, తెగుళ్ల నివారణ, ఎరువుల సమాచారం, మార్కెట్ ధరలు "
                "మరియు వాతావరణ అంచనా కోసం నన్ను అడగవచ్చు. మీకు ఏ విధంగా సహాయపడగలను?"
            )
        return (
            "Hello! I am your BhoomiMitra AI farming assistant. 🙏\n\n"
            "You can ask me about crop advisory, pest/disease management, fertilizer recommendations, "
            "mandi prices, and weather forecasts. How can I assist you today?"
        )

    @classmethod
    def detect_all_intents(cls, user_message: str) -> List[FarmerIntent]:
        """
        Identify all intents present in the farmer's message.
        Supports multi-intent queries (e.g. market prices + weather).
        """
        if not user_message or not user_message.strip():
            return [FarmerIntent.UNKNOWN]

        msg = user_message.strip().lower()
        msg_original = user_message.strip()

        # Pure greeting check
        if cls.is_greeting_only(msg_original):
            return [FarmerIntent.GREETING]

        detected: List[FarmerIntent] = []

        # 0. Reminders (prioritized if user asks to be reminded)
        if (any(kw in msg for kw in REMINDERS_KEYWORDS_EN) or
            any(kw in msg_original for kw in REMINDERS_KEYWORDS_TE) or
            any(kw in msg for kw in REMINDERS_KEYWORDS_TANGLISH)):
            detected.append(FarmerIntent.REMINDERS)

        # 1. Market Price
        if (any(kw in msg for kw in MARKET_PRICE_KEYWORDS_EN) or
            any(kw in msg_original for kw in MARKET_PRICE_KEYWORDS_TE) or
            any(kw in msg for kw in MARKET_PRICE_KEYWORDS_TANGLISH)):
            detected.append(FarmerIntent.MARKET_PRICE)

        # 2. Weather
        if (any(kw in msg for kw in WEATHER_KEYWORDS_EN) or
            any(kw in msg_original for kw in WEATHER_KEYWORDS_TE) or
            any(kw in msg for kw in WEATHER_KEYWORDS_TANGLISH)):
            detected.append(FarmerIntent.WEATHER)

        # 3. Government Schemes
        if (any(kw in msg for kw in SCHEMES_KEYWORDS_EN) or
            any(kw in msg_original for kw in SCHEMES_KEYWORDS_TE) or
            any(kw in msg for kw in SCHEMES_KEYWORDS_TANGLISH)):
            detected.append(FarmerIntent.GOVERNMENT_SCHEMES)

        # 4. Shops / Input Availability
        if (any(kw in msg for kw in SHOPS_KEYWORDS_EN) or
            any(kw in msg_original for kw in SHOPS_KEYWORDS_TE) or
            any(kw in msg for kw in SHOPS_KEYWORDS_TANGLISH)):
            detected.append(FarmerIntent.SHOPS)

        # 5. Crop Health / Disease / Pest
        if (any(kw in msg for kw in CROP_HEALTH_KEYWORDS_EN) or
            any(kw in msg_original for kw in CROP_HEALTH_KEYWORDS_TE) or
            any(kw in msg for kw in CROP_HEALTH_KEYWORDS_TANGLISH)):
            detected.append(FarmerIntent.CROP_HEALTH)

        # 6. Fertilizer / Nutrients
        if (any(kw in msg for kw in FERTILIZER_KEYWORDS_EN) or
            any(kw in msg_original for kw in FERTILIZER_KEYWORDS_TE) or
            any(kw in msg for kw in FERTILIZER_KEYWORDS_TANGLISH)):
            detected.append(FarmerIntent.FERTILIZER)

        # 7. Irrigation
        if (any(kw in msg for kw in IRRIGATION_KEYWORDS_EN) or
            any(kw in msg_original for kw in IRRIGATION_KEYWORDS_TE) or
            any(kw in msg for kw in IRRIGATION_KEYWORDS_TANGLISH)):
            detected.append(FarmerIntent.IRRIGATION)

        # 8. Sowing
        if (any(kw in msg for kw in SOWING_KEYWORDS_EN) or
            any(kw in msg_original for kw in SOWING_KEYWORDS_TE) or
            any(kw in msg for kw in SOWING_KEYWORDS_TANGLISH)):
            detected.append(FarmerIntent.SOWING)

        # 9. Harvesting
        if (any(kw in msg for kw in HARVESTING_KEYWORDS_EN) or
            any(kw in msg_original for kw in HARVESTING_KEYWORDS_TE) or
            any(kw in msg for kw in HARVESTING_KEYWORDS_TANGLISH)):
            detected.append(FarmerIntent.HARVESTING)


        # Fallback: if no specific domain matched, identify as general farming or unknown
        if not detected:
            from src.rag.service import extract_crop_from_text
            mentioned_crop = extract_crop_from_text(msg_original)
            agri_words = [
                "crop", "crops", "farm", "farming", "land", "field", "acre", "acres", "yield",
                "soil", "cotton", "paddy", "chilli", "tomato", "maize", "agriculture",
                "పంట", "పొలం", "సాగు", "భూమి", "ఎకరం", "దిగుబడి", "నేల", "పత్తి", "వరి", "మిర్చి"
            ]
            if mentioned_crop or any(w in msg for w in agri_words):
                detected.append(FarmerIntent.GENERAL_FARMING)
            else:
                detected.append(FarmerIntent.UNKNOWN)

        return detected

    @classmethod
    def detect_primary_intent(cls, user_message: str) -> FarmerIntent:
        """Identify the single primary intent for a user message."""
        intents = cls.detect_all_intents(user_message)
        return intents[0] if intents else FarmerIntent.GENERAL_FARMING

    async def process_message(
        self,
        db: AsyncSession,
        farmer: Farmer,
        conversation: Conversation,
    ) -> str:
        """
        Main decision and orchestration pipeline.
        - Identifies intents
        - Bypasses LLM for pure greetings
        - Coordinates specialized services only when relevant
        - Enforces factual authoritativeness (never hallucinating numbers or prices)
        - Applies localized fallbacks when services produce no data
        - Formats and finalizes the outgoing response
        """
        user_message = conversation.user_message or ""
        logger.info(f"[DECISION ENGINE START] Farmer: {farmer.id} | Query: '{user_message}'")

        # Resolve preferred language
        language = getattr(farmer, "preferred_language", "en") or "en"
        if any(ord(c) > 127 for c in user_message):
            language = "te"
        elif any(kw in user_message.lower() for kw in (MARKET_PRICE_KEYWORDS_TANGLISH + WEATHER_KEYWORDS_TANGLISH + CROP_HEALTH_KEYWORDS_TANGLISH)):
            # Tanglish defaults to Telugu responses per user requirement
            language = "te"

        # 1. Pure Greeting Shortcut
        if self.is_greeting_only(user_message):
            logger.info(f"[DECISION ENGINE] Pure greeting detected. Returning instant greeting for farmer {farmer.id}.")
            reply = self.get_greeting_reply(language=language)
            conversation.ai_response = reply
            conversation.intent = FarmerIntent.GREETING.value
            db.add(conversation)
            await db.commit()
            return reply

        # 2. Detect all intents
        intents = self.detect_all_intents(user_message)
        primary_intent = intents[0]
        conversation.intent = primary_intent.value
        logger.info(f"[DECISION ENGINE] Detected intents: {[i.value for i in intents]} | Primary: {primary_intent.value}")

        has_market = FarmerIntent.MARKET_PRICE in intents
        has_weather = FarmerIntent.WEATHER in intents
        has_schemes = FarmerIntent.GOVERNMENT_SCHEMES in intents
        has_shops = FarmerIntent.SHOPS in intents
        has_crop_advice = any(i in intents for i in [
            FarmerIntent.CROP_ADVICE,
            FarmerIntent.CROP_HEALTH,
            FarmerIntent.FERTILIZER,
            FarmerIntent.IRRIGATION,
            FarmerIntent.SOWING,
            FarmerIntent.HARVESTING,
            FarmerIntent.REMINDERS,
            FarmerIntent.GENERAL_FARMING,
            FarmerIntent.UNKNOWN,
        ])

        # 3. AI Advisory Generation
        ai_response_text = ""
        from src.ai.repository import AIRepository
        from src.ai.service import AIService, _finalize_whatsapp_response
        from src.ai.schemas import AIGenerateRequest

        repo = AIRepository(db)
        ai_service = AIService(repo)
        request = AIGenerateRequest(
            farmer_id=farmer.id,
            conversation_id=conversation.id,
            message=user_message,
        )

        try:
            response = await ai_service.generate_ai_response(request)
            ai_response_text = response.response_text or ""
            logger.info(f"[DECISION ENGINE] Raw Gemini response ({len(ai_response_text)} chars)")
        except Exception as exc:
            logger.warning(f"[DECISION ENGINE] AI generation unavailable: {exc}. Deferring to specialized modules.")
            ai_response_text = ""

        # 4. Authoritative Module Routing (Only call enrichments when intent is relevant)
        # A. Shops / Input Availability
        if has_shops:
            try:
                from src.shops.service import enrich_response_with_shops
                logger.info("[DECISION ENGINE] Routing to shops service")
                ai_response_text = await enrich_response_with_shops(
                    db, user_message, ai_response_text, farmer
                )
            except Exception as err:
                logger.warning(f"Shops enrichment warning: {err}")

        # B. Market Price
        if has_market:
            try:
                from src.market.service import enrich_response_with_market_prices
                logger.info("[DECISION ENGINE] Routing to market price service")
                ai_response_text = await enrich_response_with_market_prices(
                    db, user_message, ai_response_text, farmer
                )
            except Exception as mkt_err:
                logger.warning(f"Market enrichment warning: {mkt_err}")

        # C. Weather
        if has_weather:
            try:
                from src.weather.service import enrich_response_with_weather
                logger.info("[DECISION ENGINE] Routing to weather service")
                ai_response_text = await enrich_response_with_weather(
                    db, user_message, ai_response_text, farmer
                )
            except Exception as weather_err:
                logger.warning(f"Weather enrichment warning: {weather_err}")

        # D. Government Schemes
        if has_schemes:
            try:
                from src.schemes.service import enrich_response_with_schemes
                logger.info("[DECISION ENGINE] Routing to schemes service")
                ai_response_text = await enrich_response_with_schemes(
                    db, user_message, ai_response_text, farmer
                )
            except Exception as scheme_err:
                logger.warning(f"Schemes enrichment warning: {scheme_err}")

        # E. Expert Escalation
        try:
            from src.escalation.service import enrich_response_with_escalation
            ai_response_text = await enrich_response_with_escalation(
                db, user_message, ai_response_text, farmer
            )
        except Exception as esc_err:
            logger.warning(f"Escalation enrichment warning: {esc_err}")

        # 5. Multi-Intent Response Formatting
        try:
            from src.ai.formatting import format_multi_intent_response
            ai_response_text = format_multi_intent_response(
                assembled_text=ai_response_text,
                user_message=user_message,
                language=language,
            )
        except Exception as fmt_err:
            logger.warning(f"Multi-intent formatting warning: {fmt_err}")

        # 6. Fallback Protection for Single Intent / Missing Responses
        ai_response_text = ai_response_text.strip() if ai_response_text else ""
        is_single_intent = len(intents) == 1

        if is_single_intent:
            if primary_intent == FarmerIntent.MARKET_PRICE and not any(k in ai_response_text for k in ["📊", "⚠️"]):
                ai_response_text = get_market_fallback_response(language)
            elif primary_intent == FarmerIntent.GOVERNMENT_SCHEMES and "🏛️" not in ai_response_text:
                ai_response_text = get_schemes_fallback_response(language)
            elif primary_intent == FarmerIntent.WEATHER and not any(w in ai_response_text for w in ["🌡️", "🌤️", "🌦️"]):
                ai_response_text = get_weather_fallback_response(language)
            elif primary_intent == FarmerIntent.SHOPS and "🏬" not in ai_response_text:
                ai_response_text = get_shops_fallback_response(language)
            elif not ai_response_text:
                ai_response_text = get_fallback_response(language)
        elif not ai_response_text:
            ai_response_text = get_fallback_response(language)

        # 7. Finalize WhatsApp Response (Length & block management)
        ai_response_text = _finalize_whatsapp_response(ai_response_text)

        logger.info(f"[DECISION ENGINE FINAL] Output chars: {len(ai_response_text)} | Preview: {ai_response_text[:120]}...")

        conversation.ai_response = ai_response_text
        db.add(conversation)
        await db.commit()

        return ai_response_text


_decision_engine_instance: Optional[AIDecisionEngine] = None


def get_decision_engine() -> AIDecisionEngine:
    """Return singleton instance of AIDecisionEngine."""
    global _decision_engine_instance
    if _decision_engine_instance is None:
        _decision_engine_instance = AIDecisionEngine()
    return _decision_engine_instance
