"""
BhoomiMitra AI — Market Price Service

Business logic for mandi price queries.
Also contains enrich_response_with_market_prices() which slots into
the existing AI pipeline (ai/service.py) the same way enrich_response_with_shops() does.

SAFETY RULES:
- Prices are NEVER invented or hardcoded.
- If no data is available, the farmer is told clearly.
- All exceptions are caught; the WhatsApp pipeline is never interrupted.
"""
import re
from typing import Optional, List
from datetime import datetime, timezone, timedelta
from uuid import uuid4
from zoneinfo import ZoneInfo

from src.core.logging import logger
from src.market.repository import MarketPriceRepository
from src.market.agmarknet_client import AgmarknetClient
from src.market.schemas import (
    MarketPriceCreate,
    MarketPriceResponse,
    MarketPriceQueryResponse,
)

try:
    IST_TZ = ZoneInfo("Asia/Kolkata")
except Exception:
    IST_TZ = timezone(timedelta(hours=5, minutes=30))


def get_current_ist_date():
    """Return the current calendar date in India Standard Time (Asia/Kolkata)."""
    return datetime.now(IST_TZ).date()


def is_record_from_today(price_dt) -> bool:
    """Check whether a price datetime matches today's date in IST."""
    if not price_dt:
        return False
    if isinstance(price_dt, str):
        try:
            price_dt = datetime.fromisoformat(price_dt.strip())
        except ValueError:
            for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y"):
                try:
                    price_dt = datetime.strptime(price_dt.strip(), fmt)
                    break
                except ValueError:
                    continue
            else:
                return False
    today_ist = get_current_ist_date()
    if isinstance(price_dt, datetime):
        return price_dt.date() == today_ist
    elif hasattr(price_dt, "year"):
        return price_dt == today_ist
    return False


# ------------------------------------------------------------------
# Commodity keyword map — Telugu and English → canonical English name
# Used to extract the commodity from a farmer's freeform query.
# ------------------------------------------------------------------
COMMODITY_MAP = {
    # Tomato
    "tomato": "Tomato", "tomatoes": "Tomato",
    "టమాటా": "Tomato", "టమాట": "Tomato",
    # Paddy / Rice
    "paddy": "Paddy", "rice": "Paddy",
    "వరి": "Paddy", "ధాన్యం": "Paddy",
    # Onion
    "onion": "Onion", "onions": "Onion",
    "ఉల్లిపాయ": "Onion", "ఉల్లి": "Onion",
    # Cotton
    "cotton": "Cotton", "kapas": "Cotton",
    "పత్తి": "Cotton", "కాపాస్": "Cotton", "ఖరీఫ్ పత్తి": "Cotton",
    # Maize / Corn
    "maize": "Maize", "corn": "Maize",
    "మొక్కజొన్న": "Maize",
    # Chilli
    "chilli": "Chilli", "chili": "Chilli", "chilies": "Chilli", "chillies": "Chilli",
    "మిర్చి": "Chilli",
    # Groundnut
    "groundnut": "Groundnut", "peanut": "Groundnut",
    "వేరుశనగ": "Groundnut",
    # Soybean
    "soybean": "Soybean", "soya": "Soybean",
    "సోయాబీన్": "Soybean",
    # Turmeric
    "turmeric": "Turmeric",
    "పసుపు": "Turmeric",
    # Sugarcane
    "sugarcane": "Sugarcane",
    "చెరుకు": "Sugarcane",
    # Banana
    "banana": "Banana", "bananas": "Banana",
    "అరటి": "Banana",
    # Wheat
    "wheat": "Wheat",
    "గోధుమ": "Wheat",
    # Jowar / Sorghum
    "jowar": "Jowar", "sorghum": "Jowar",
    "జొన్న": "Jowar",
    # Bengalgram / Chickpea
    "bengalgram": "Bengalgram", "chickpea": "Bengalgram", "gram": "Bengalgram",
    "శనగ": "Bengalgram",
}

# Intent keywords — triggers market price enrichment
PRICE_INTENT_KEYWORDS_EN = {
    "price", "mandi", "market price", "rate", "today price",
    "how much", "selling price", "market rate", "crop price",
}
PRICE_INTENT_KEYWORDS_TE = {
    "ధర", "ధరలు", "మండి", "రేటు", "రేట్లు", "ఎంత", "నేటి ధర", "మార్కెట్ ధర",
    "మార్కెట్ ధరలు", "మండి ధర", "మండి ధరలు", "ఎంత ధర", "అమ్మకం ధర",
    "క్వింటాల్", "క్వింటాలు", "ఖరీదు", "మార్కెట్", "మార్కెట్లో", "ధర ఎంత",
}

# Today-specific intent keywords — used to distinguish queries explicitly asking for today's price
TODAY_KEYWORDS = {
    # English
    "today", "today's", "todays", "current price", "today price", "today's price",
    "current rate", "today rate", "present price", "present rate", "current market",
    # Telugu
    "ఈరోజు", "ఈ రోజు", "నేడు", "నేటి", "ఈనాటి", "ఈ నాటి", "ఈరోజున", "ఈ రోజున",
    "ఈరోజుటి", "ఈ రోజుటి", "నేటిధర", "ఈరోజుధర", "ప్రస్తుతం", "ప్రస్తుత",
    "ఇప్పటి", "ఇప్పటికి", "ఇవాల్టి", "ఇవాళ", "ఇవాళ్టి", "ఈవేళ",
}

_TELUGU_DISTRICT_MAP = {
    "Warangal": "వరంగల్",
    "Enumamula": "వరంగల్ (ఎనుమాముల)",
    "Adilabad": "ఆదిలాబాద్",
    "Khammam": "ఖమ్మం",
    "Nizamabad": "నిజామాబాద్",
    "Karimnagar": "కరీంనగర్",
    "Nalgonda": "నల్గొండ",
    "Mahabubnagar": "మహబూబ్‌నగర్",
    "Hyderabad": "హైదరాబాద్",
    "Guntur": "గుంటూరు",
    "Krishna": "కృష్ణా",
    "Kurnool": "కర్నూలు",
    "Anantapur": "అనంతపురం",
    "Chittoor": "చిత్తూరు",
    "Visakhapatnam": "విశాఖపట్నం",
}


def is_today_price_query(query_text: str) -> bool:
    """Check if the user is explicitly asking for today's market rate."""
    if not query_text:
        return False
    q_lower = query_text.lower()
    return any(kw in q_lower for kw in TODAY_KEYWORDS)


# Non-price intent keywords to detect whether a query is purely about market prices or multi-intent
OTHER_INTENT_KEYWORDS = [
    # Disease / pest / spray / dosage
    "spray", "disease", "pest", "fungus", "leaf", "rot", "spots", "dosage", "chemical", "pesticide",
    "మందు", "పిచికారీ", "తెగులు", "పురుగు", "ఆకు", "మచ్చలు", "మోతాదు", "నివారణ",
    # Fertilizer / nutrient / sowing
    "fertilizer", "fertilizers", "urea", "dap", "sowing", "seed", "variety", "stage",
    "ఎరువు", "ఎరువులు", "యూరియా", "విత్తనం", "విత్తనాలు", "సాగు",
    # Weather
    "weather", "forecast", "rain", "temperature", "వాతావరణం", "వర్షం", "ఎండ",
    # Scheme
    "scheme", "subsidy", "yojana", "pm kisan", "rythu", "పథకం", "సబ్సిడీ",
    # Shops / purchase
    "where to buy", "store", "కొనాలి", "దుకాణం", "షాప్",
    # Escalation / contact
    "officer", "call", "agent", "expert", "మాట్లాడాలి", "అధికారి",
]


def _is_pure_price_query(query_text: str) -> bool:
    """Return True if the query is asking about market prices without asking other agronomic questions."""
    query_lower = query_text.lower()
    return not any(kw in query_lower for kw in OTHER_INTENT_KEYWORDS)


def _clean_ai_response_for_market_enrichment(ai_response: str) -> str:
    """
    Remove speculative price statements, refusal markers, or redundant mandi intros from AI text
    so that only genuine agronomic advisory (if any) is preserved alongside the structured price block.
    """
    if not ai_response:
        return ""

    refusal_markers = [
        "e-nam", "ఈ-నామ్", "ఈ - నామ్", "మార్కెట్ యార్డ్", "మార్కెట్ యార్డు",
        "market yard", "కేవలం వ్యవసాయం", "విషయాలపై మాత్రమే", "i can only help with farming",
        "only help with farming", "how can i help with your crops", "స్థానిక మార్కెట్",
        "క్షమించండి, ప్రస్తుతం కనెక్ట్ అవడంలో", "i'm sorry, i'm having trouble connecting",
    ]
    ai_lower = ai_response.lower()
    if any(marker in ai_lower for marker in refusal_markers):
        return ""

    # Sentence markers that indicate speculative or duplicate price discussion
    price_sentence_markers = [
        "ధర", "ధరలు", "మండి", "క్వింటాల్", "క్వింటాలు", "రేటు", "రేట్లు",
        "price", "prices", "mandi", "rate", "rates", "quintal", "₹", "rs."
    ]

    cleaned_paragraphs = []
    for para in ai_response.split("\n"):
        para = para.strip()
        if not para:
            continue
        sentences = [s.strip() for s in para.replace("।", ".").split(".") if s.strip()]
        non_price_sentences = [
            s for s in sentences
            if not any(pm in s.lower() for pm in price_sentence_markers)
        ]
        if non_price_sentences:
            cleaned_paragraphs.append(". ".join(non_price_sentences) + ".")

    return "\n\n".join(cleaned_paragraphs).strip()

# Telugu labels for formatted reply
_TE_LABELS = {
    "title": "📊 {commodity} మార్కెట్ ధరలు",
    "market": "మండి",
    "modal": "మోడల్ ధర",
    "min": "కనిష్ట",
    "max": "గరిష్ట",
    "date": "తేదీ",
    "source_live": "అగ్‌మార్క్‌నెట్ (లైవ్)",
    "source_local": "స్థానిక డేటాబేస్",
    "unit_suffix": "క్వింటాల్కు",
}

_EN_LABELS = {
    "title": "📊 {commodity} Mandi Prices",
    "market": "Market",
    "modal": "Modal Price",
    "min": "Min",
    "max": "Max",
    "date": "Date",
    "source_live": "Agmarknet (Live)",
    "source_local": "Local Database",
    "unit_suffix": "per Quintal",
}


class MarketService:
    def __init__(self, repository: MarketPriceRepository, client: AgmarknetClient):
        self.repository = repository
        self.client = client

    # ------------------------------------------------------------------
    # Public: price query entry point
    # ------------------------------------------------------------------

    async def get_prices_for_query(
        self,
        commodity: str,
        district: Optional[str] = None,
        state: Optional[str] = None,
        is_today_requested: bool = False,
    ) -> MarketPriceQueryResponse:
        """
        1. Try the Agmarknet API first (passing is_today_requested).
        2. If API returns data → upsert to local DB.
        3. Check if today's records exist (matching IST calendar date).
        4. If is_today_requested and live API has today records → return them as live data.
        5. If live API does not have today records or fails → query local DB.
        6. If local DB has today records → return them.
        7. If is_today_requested and NO today data exists anywhere → return latest DB records with is_today_requested=True.
        8. If not is_today_requested → return latest DB records.
        """
        is_live = False
        source_note = ""
        today_ist = get_current_ist_date()

        logger.info(
            f"[MARKET SERVICE] Query start -> commodity='{commodity}', district='{district}', "
            f"state='{state}', is_today_requested={is_today_requested}, today_ist={today_ist}"
        )

        # Step 1: Try live API
        api_records = await self.client.fetch_prices(
            commodity=commodity,
            state=state,
            district=district,
            is_today_requested=is_today_requested,
        )
        if api_records is None:
            api_records = []

        has_today_live = any(is_record_from_today(r.get("arrival_date")) for r in api_records if isinstance(r, dict)) if api_records else False
        logger.info(
            f"[MARKET SERVICE] External API result -> count={len(api_records)}, "
            f"has_today_live={has_today_live}"
        )

        if api_records:
            await self._upsert_api_records(api_records, commodity)

        # If today's price was specifically requested and live API provided today's records
        if is_today_requested and has_today_live:
            is_live = True
            source_note = "Live data from Agmarknet / data.gov.in"
            today_api_records = [r for r in api_records if is_record_from_today(r.get("arrival_date"))]
            results = []
            for r in today_api_records:
                telugu_name = None
                for kw, canon in COMMODITY_MAP.items():
                    if canon.lower() == commodity.lower() and any(ord(c) > 127 for c in kw):
                        telugu_name = kw
                        break
                arr_dt = r.get("arrival_date")
                if not isinstance(arr_dt, datetime):
                    try:
                        arr_dt = datetime.fromisoformat(str(arr_dt))
                    except Exception:
                        arr_dt = datetime.utcnow()
                results.append(
                    MarketPriceResponse(
                        id=uuid4(),
                        commodity=commodity,
                        commodity_telugu=telugu_name,
                        market_name=r.get("market", "Unknown Market"),
                        district=r.get("district", district or ""),
                        state=r.get("state", state or "Telangana"),
                        min_price=float(r.get("min_price", 0)),
                        max_price=float(r.get("max_price", 0)),
                        modal_price=float(r.get("modal_price", 0)),
                        unit=r.get("unit", "Quintal"),
                        price_date=arr_dt,
                        source="agmarknet_api",
                        created_at=datetime.utcnow(),
                        updated_at=datetime.utcnow(),
                    )
                )
            logger.info(
                f"[MARKET SERVICE] Selected LIVE TODAY record -> "
                f"market='{results[0].market_name}', modal={results[0].modal_price}, "
                f"price_date={results[0].price_date.strftime('%Y-%m-%d')}"
            )
            return MarketPriceQueryResponse(
                commodity=commodity,
                district=district,
                state=state,
                results=results,
                data_available=True,
                data_freshness_hours=0.0,
                source_note=source_note,
                is_live=True,
                is_today_requested=True,
            )

        # Step 2: Query local DB (for latest records or when API didn't return today's data)
        db_records = await self.repository.get_prices_by_commodity(
            commodity=commodity,
            district=district,
            state=state,
        )

        if not db_records:
            logger.info(
                f"[MARKET SERVICE] No price data found in DB or API for '{commodity}' "
                f"(district={district}, state={state})"
            )
            return MarketPriceQueryResponse(
                commodity=commodity,
                district=district,
                state=state,
                results=[],
                data_available=False,
                data_freshness_hours=None,
                source_note="No price data available in local database or API.",
                is_live=False,
                is_today_requested=is_today_requested,
            )

        has_today_db = any(is_record_from_today(getattr(r, "price_date", None)) for r in db_records)
        dates = [
            r.price_date
            for r in db_records
            if getattr(r, "price_date", None) and isinstance(r.price_date, datetime)
        ]
        newest = max(dates) if dates else None
        freshness_hours = (
            round((datetime.utcnow() - newest).total_seconds() / 3600, 1)
            if newest
            else 0.0
        )

        if api_records and not is_today_requested:
            is_live = True
            source_note = "Live data from Agmarknet / data.gov.in"
        else:
            is_live = False
            source_note = f"Local database (data is ~{freshness_hours}h old)"

        results = [MarketPriceResponse.model_validate(r) for r in db_records]

        logger.info(
            f"[MARKET SERVICE] Selected DB record -> "
            f"market='{results[0].market_name}', modal={results[0].modal_price}, "
            f"price_date={results[0].price_date.strftime('%Y-%m-%d') if results[0].price_date else 'N/A'}, "
            f"has_today_match={has_today_db}, is_live={is_live}"
        )

        return MarketPriceQueryResponse(
            commodity=commodity,
            district=district,
            state=state,
            results=results,
            data_available=True,
            data_freshness_hours=freshness_hours,
            source_note=source_note,
            is_live=is_live,
            is_today_requested=is_today_requested,
        )

    async def list_commodities(self) -> List[str]:
        """Return distinct commodity names stored in the local database."""
        return await self.repository.list_commodities()

    async def create_price(self, data: MarketPriceCreate):
        """Admin: manually insert a market price record."""
        return await self.repository.create_price(data)

    async def _upsert_api_records(self, records: List[dict], commodity: str):
        """Convert Agmarknet client dicts into MarketPriceCreate schemas and save to DB."""
        creates = []
        for r in records:
            try:
                # Determine Telugu name if commodity matches COMMODITY_MAP
                telugu_name = None
                for kw, canon in COMMODITY_MAP.items():
                    if canon.lower() == commodity.lower() and any(
                        ord(c) > 127 for c in kw
                    ):
                        telugu_name = kw
                        break

                arr_date = r.get("arrival_date")
                if isinstance(arr_date, datetime):
                    price_dt = arr_date
                elif isinstance(arr_date, str):
                    try:
                        price_dt = datetime.strptime(arr_date, "%Y-%m-%d")
                    except ValueError:
                        price_dt = datetime.fromisoformat(arr_date)
                else:
                    price_dt = datetime.utcnow()

                creates.append(
                    MarketPriceCreate(
                        commodity=commodity,
                        commodity_telugu=telugu_name,
                        market_name=r.get("market", "Unknown Market"),
                        district=r.get("district", ""),
                        state=r.get("state", ""),
                        min_price=float(r.get("min_price", 0)),
                        max_price=float(r.get("max_price", 0)),
                        modal_price=float(r.get("modal_price", 0)),
                        unit=r.get("unit", "Quintal"),
                        price_date=price_dt,
                        source="agmarknet_api",
                    )
                )
            except (ValueError, TypeError) as exc:
                logger.warning(
                    f"[MARKET SERVICE] Skipping invalid record {r}: {exc}"
                )
                continue

        if creates:
            await self.repository.upsert_prices(creates)

    # ------------------------------------------------------------------
    # WhatsApp response formatting
    # ------------------------------------------------------------------

    def format_whatsapp_reply(
        self,
        query_response: MarketPriceQueryResponse,
        language: str = "te",
        is_today_query: bool = False,
    ) -> str:
        """
        Render a MarketPriceQueryResponse into a localized WhatsApp text message.

        Handles:
        - Live vs local DB source attribution
        - Telugu and English
        - Price freshness notice (especially when today's price was requested but only older data exists)
        - Empty data fallback
        """
        labels = _TE_LABELS if language == "te" else _EN_LABELS
        commodity = query_response.commodity

        commodity_display = commodity
        if language == "te":
            for kw, canon in COMMODITY_MAP.items():
                if canon.lower() == commodity.lower() and any(ord(c) > 127 for c in kw):
                    commodity_display = kw
                    break

        if not query_response.data_available or not query_response.results:
            if language == "te":
                return (
                    f"క్షమించండి, ప్రస్తుతం {commodity_display} మండి ధర సమాచారం అందుబాటులో లేదు.\n"
                    "దయచేసి మీ స్థానిక మండిని సంప్రదించండి లేదా "
                    "రైతు సేవ కేంద్రాన్ని (1800-425-1422) సంప్రదించండి."
                )
            return (
                f"Sorry, I could not find current mandi prices for {commodity}.\n"
                "Please check your local mandi or call the Rythu Seva Kendra (1800-425-1422)."
            )

        # Use the most recent record per market
        seen_markets = set()
        deduplicated = []
        for r in query_response.results:
            key = r.market_name.lower()
            if key not in seen_markets:
                seen_markets.add(key)
                deduplicated.append(r)
            if len(deduplicated) >= 3:
                break

        has_today_data = any(is_record_from_today(r.price_date) for r in deduplicated)
        source_str = labels["source_live"] if query_response.is_live else labels["source_local"]

        # If farmer specifically asked for today's price, but all available records are from older dates
        if is_today_query and not has_today_data:
            newest_date = max(r.price_date for r in deduplicated)
            latest_date_str = newest_date.strftime("%d %b %Y")

            if language == "te":
                dist_str = ""
                if query_response.district:
                    tel_dist = _TELUGU_DISTRICT_MAP.get(query_response.district, query_response.district)
                    dist_str = f"{tel_dist} "
                lines = [
                    f"⚠️ ఈరోజు {dist_str}{commodity_display} మార్కెట్ ధర డేటా అందుబాటులో లేదు.",
                    f"📅 చివరిగా లభించిన ధర: {latest_date_str}",
                ]
            else:
                dist_str = f"{query_response.district} " if query_response.district else ""
                lines = [
                    f"⚠️ Today's market price data for {dist_str}{commodity} is unavailable.",
                    f"📅 Last available price: {latest_date_str}",
                ]
        else:
            lines = [labels["title"].format(commodity=commodity_display)]

        for r in deduplicated:
            date_str = r.price_date.strftime("%d %b %Y")
            block = (
                f"\n{labels['market']}: {r.market_name}, {r.state}\n"
                f"{labels['modal']}: ₹{r.modal_price:,.0f}/{labels['unit_suffix']}\n"
                f"{labels['min']}: ₹{r.min_price:,.0f} | {labels['max']}: ₹{r.max_price:,.0f}\n"
                f"{labels['date']}: {date_str}"
            )
            lines.append(block)

        lines.append(f"\n📡 {source_str}")
        return "\n".join(lines)


# ------------------------------------------------------------------
# Deduplication helper: cleans redundant/speculative price sentences
# ------------------------------------------------------------------

def _clean_market_duplicate_text(ai_response: str) -> str:
    """
    Safely clean duplicate/speculative market-price statements and refusals from AI response
    before appending the authoritative structured Market Prices block.

    Preserves:
    - Agronomic and crop management advice
    - General context / greetings that do not quote speculative prices

    Removes:
    - Refusal statements telling farmers to consult e-NAM / local market yards
    - Sentences quoting speculative market price numbers/ranges (e.g. containing ₹, Rs, per quintal, etc.)
    """
    import re
    if not ai_response or not ai_response.strip():
        return ""

    refusal_markers = [
        "e-nam", "ఈ-నామ్", "ఈ - నామ్", "మార్కెట్ యార్డ్", "మార్కెట్ యార్డు",
        "market yard", "కేవలం వ్యవసాయం", "విషయాలపై మాత్రమే", "i can only help with farming",
        "only help with farming", "how can i help with your crops", "స్థానిక మార్కెట్",
    ]

    lines = [line.strip() for line in ai_response.split("\n") if line.strip()]
    cleaned_lines = []

    for line in lines:
        line_lower = line.lower()
        if any(marker in line_lower for marker in refusal_markers):
            continue

        sentences = re.split(r'(?<=[.!?।])\s+', line)
        kept_sentences = []
        for s in sentences:
            s_strip = s.strip()
            if not s_strip:
                continue
            s_lower = s_strip.lower()

            if any(marker in s_lower for marker in refusal_markers):
                continue

            # Check for speculative price quotes / numbers / units
            has_currency = bool(re.search(r'(₹|rs\.?|inr|రూ\.?|రూపాయలు)', s_lower))
            has_unit = bool(re.search(r'(quintal|క్వింటా|క్వింటాల్|క్వింటాలు|per\s+kg|కిలోకి|kg)', s_lower))
            has_price_word = bool(re.search(r'(ధర|రేటు|మార్కెట్|మండి|price|rate|mandi)', s_lower))
            has_estimation = bool(re.search(r'(సుమారు|దాదాపు|around|ranges|between|సుమారుగా)', s_lower))

            is_price_quote = (
                (has_currency and has_price_word)
                or (has_unit and has_price_word and bool(re.search(r'\d', s_strip)))
                or (has_estimation and has_price_word)
                or bool(re.search(r'^(the\s+)?(cotton|tomato|paddy|onion|chilli|crop)\s+market\s+price\s+is', s_lower))
                or bool(re.search(r'మార్కెట్\s*ధర\s*క్వింటాలు', s_lower))
            )

            if is_price_quote:
                continue

            kept_sentences.append(s_strip)

        if kept_sentences:
            cleaned_lines.append(" ".join(kept_sentences))

    return "\n".join(cleaned_lines).strip()


# ------------------------------------------------------------------
# Pipeline integration function — mirrors enrich_response_with_shops()
# Called from ai/service.py inside a try/except block.
# ------------------------------------------------------------------

async def enrich_response_with_market_prices(
    db,
    query_text: str,
    ai_response: str,
    farmer,
) -> str:
    """
    Detect market-price intent in the farmer's query.
    If detected, append a formatted mandi price block to the AI response.

    Always returns the original ai_response unchanged if:
    - No price intent is detected
    - No commodity is identified
    - Any error occurs

    Never raises. Never invents prices.
    """
    query_lower = query_text.lower()

    # Step 1: Detect price intent
    has_price_intent = any(kw in query_lower for kw in PRICE_INTENT_KEYWORDS_EN)
    if not has_price_intent:
        has_price_intent = any(kw in query_lower for kw in PRICE_INTENT_KEYWORDS_TE)

    logger.info(
        f"[MARKET ENRICH] Diagnostic check -> query='{query_text}' | "
        f"has_price_intent={has_price_intent}"
    )

    if not has_price_intent:
        return ai_response

    # Step 2: Identify commodity
    matched_commodity = None
    import re
    sorted_keywords = sorted(COMMODITY_MAP.items(), key=lambda x: len(x[0]), reverse=True)
    for kw, canonical in sorted_keywords:
        kw_lower = kw.lower()
        if kw_lower.isascii() and kw_lower.isalnum():
            if re.search(rf"\b{re.escape(kw_lower)}\b", query_lower):
                matched_commodity = canonical
                break
        else:
            if kw in query_text or kw_lower in query_lower:
                matched_commodity = canonical
                break

    # Fallback to extract_crop_from_text if not matched directly in COMMODITY_MAP
    if not matched_commodity:
        try:
            from src.rag.service import extract_crop_from_text
            matched_commodity = extract_crop_from_text(query_text)
        except Exception:
            matched_commodity = None

    # Step 3: Get farmer location and profile
    district = None
    state = None
    language = getattr(farmer, "preferred_language", "te") or "te"

    # Infer language from query text if Telugu characters present
    if any(ord(c) > 127 for c in query_text):
        language = "te"

    # Extract district from query text if explicitly mentioned (e.g. "వరంగల్లో", "Warangal", "Enumamula")
    try:
        from src.weather.service import _extract_district_from_query
        q_dist = _extract_district_from_query(query_text)
        if q_dist:
            district = q_dist
    except Exception as dist_err:
        logger.debug(f"[MARKET ENRICH] Query district extraction skipped: {dist_err}")

    try:
        import inspect
        from sqlalchemy import select
        from src.core.models import FarmerProfile
        if farmer and hasattr(farmer, "id"):
            profile_result = await db.execute(
                select(FarmerProfile).where(FarmerProfile.farmer_id == farmer.id)
            )
            if inspect.iscoroutine(profile_result):
                profile_result = await profile_result
            profile = profile_result.scalar_one_or_none() if hasattr(profile_result, "scalar_one_or_none") else None
            if inspect.iscoroutine(profile):
                profile = await profile
            if profile:
                if not district and isinstance(getattr(profile, "district", None), str) and profile.district:
                    district = profile.district.strip()
                if isinstance(getattr(profile, "state", None), str) and profile.state:
                    state = profile.state.strip()
                if not matched_commodity and isinstance(getattr(profile, "current_crop", None), str) and profile.current_crop:
                    matched_commodity = profile.current_crop.strip()
    except Exception as exc:
        logger.warning(f"[MARKET ENRICH] Could not load farmer profile: {exc}")

    today_requested = is_today_price_query(query_text)

    logger.info(
        f"[MARKET ENRICH] Parameters -> detected_crop='{matched_commodity}', "
        f"detected_location_or_mandi='{district}', state='{state}', "
        f"is_today_requested={today_requested}, language='{language}'"
    )

    if not matched_commodity:
        logger.info("[MARKET ENRICH] Price intent detected but no commodity matched.")
        return ai_response

    # Step 4: Fetch prices
    try:
        from src.config import get_settings
        from src.market.repository import MarketPriceRepository
        settings = get_settings()
        repo = MarketPriceRepository(db)

        # Idempotently seed default market prices if table is empty
        await repo.seed_default_prices_if_empty()

        client = AgmarknetClient(
            api_key=settings.data_gov_api_key,
            api_url=settings.agmarknet_api_url,
            cache_ttl_seconds=settings.market_price_cache_ttl_seconds,
            timeout_seconds=settings.agmarknet_api_timeout_seconds,
        )
        svc = MarketService(repository=repo, client=client)
        query_response = await svc.get_prices_for_query(
            commodity=matched_commodity,
            district=district,
            state=state,
            is_today_requested=today_requested,
        )

        logger.info(
            f"[MARKET ENRICH] Query response -> data_available={query_response.data_available}, "
            f"record_count={len(query_response.results)}, is_live={query_response.is_live}, "
            f"is_today_requested={today_requested}"
        )

        price_block = svc.format_whatsapp_reply(
            query_response, language=language, is_today_query=today_requested
        )

        if query_response.data_available:
            logger.info(
                f"[MARKET ENRICH] Appending {len(query_response.results)} price records "
                f"for '{matched_commodity}' to AI response."
            )
            
            if _is_pure_price_query(query_text):
                # For pure market price inquiries, the structured block is the complete, authoritative answer.
                final_enriched = price_block
            else:
                # For multi-intent queries, preserve agronomic advice while stripping redundant price guesses.
                clean_ai = _clean_ai_response_for_market_enrichment(ai_response)
                if clean_ai:
                    final_enriched = clean_ai + "\n\n" + price_block
                else:
                    final_enriched = price_block

            logger.info(f"[MARKET ENRICH] Final enriched response length={len(final_enriched)}")
            return final_enriched
        else:
            # Data unavailable — do NOT append anything (Gemini's general answer still goes)
            logger.info(
                f"[MARKET ENRICH] No price data for '{matched_commodity}' — "
                "returning original AI response unchanged."
            )
            return ai_response

    except Exception as exc:
        logger.warning(f"[MARKET ENRICH] Price enrichment failed: {exc}")
        return ai_response
