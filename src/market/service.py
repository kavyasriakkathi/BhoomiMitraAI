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
from typing import Optional, List
from datetime import datetime, timezone

from src.core.logging import logger
from src.market.repository import MarketPriceRepository
from src.market.agmarknet_client import AgmarknetClient
from src.market.schemas import (
    MarketPriceCreate,
    MarketPriceResponse,
    MarketPriceQueryResponse,
)

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
    "cotton": "Cotton",
    "పత్తి": "Cotton",
    # Maize / Corn
    "maize": "Maize", "corn": "Maize",
    "మొక్కజొన్న": "Maize",
    # Chilli
    "chilli": "Chilli", "chili": "Chilli", "chilies": "Chilli", "chillies": "Chilli",
    "మిర్చి": "Chilli", "మిర్చి": "Chilli",  # noqa: F601
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
    ) -> MarketPriceQueryResponse:
        """
        1. Try the Agmarknet API first.
        2. If API returns data → upsert to local DB → return as live data.
        3. If API fails or key is absent → query local DB.
        4. If local DB is also empty → return data_available=False.
        """
        is_live = False
        source_note = ""

        # Step 1: Try live API
        api_records = await self.client.fetch_prices(
            commodity=commodity,
            state=state,
            district=district,
        )

        if api_records:
            is_live = True
            await self._upsert_api_records(api_records, commodity)
            source_note = "Live data from Agmarknet / data.gov.in"
            logger.info(
                f"[MARKET SERVICE] API returned {len(api_records)} records "
                f"for '{commodity}'"
            )

        # Step 2: Query local DB (regardless of API — DB may have fresher data from earlier call)
        db_records = await self.repository.get_prices_by_commodity(
            commodity=commodity,
            district=district,
            state=state,
        )

        if not db_records:
            logger.info(
                f"[MARKET SERVICE] No price data found for '{commodity}' "
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
            )

        # Compute freshness
        newest = max(r.price_date for r in db_records)
        freshness_hours = round(
            (datetime.utcnow() - newest).total_seconds() / 3600, 1
        )
        if not is_live:
            source_note = f"Local database (data is ~{freshness_hours}h old)"

        results = [MarketPriceResponse.model_validate(r) for r in db_records]

        return MarketPriceQueryResponse(
            commodity=commodity,
            district=district,
            state=state,
            results=results,
            data_available=True,
            data_freshness_hours=freshness_hours,
            source_note=source_note,
            is_live=is_live,
        )

    async def create_price(self, data: MarketPriceCreate):
        """Admin: manually insert a market price record."""
        from src.core.models import MarketPrice
        count = await self.repository.upsert_prices([data])
        # Re-fetch to return the stored object
        from sqlalchemy import select, and_, func
        result = await self.repository.db.execute(
            select(MarketPrice).where(
                and_(
                    MarketPrice.commodity.ilike(data.commodity),
                    MarketPrice.market_name.ilike(data.market_name),
                    func.date(MarketPrice.price_date) == data.price_date.date(),
                )
            ).order_by(MarketPrice.created_at.desc()).limit(1)
        )
        record = result.scalar_one_or_none()
        if record:
            return MarketPriceResponse.model_validate(record)
        return None

    async def list_commodities(self) -> List[str]:
        return await self.repository.list_commodities()

    # ------------------------------------------------------------------
    # Internal: upsert API records into the local DB
    # ------------------------------------------------------------------

    async def _upsert_api_records(self, api_records: List[dict], commodity: str) -> None:
        """Convert raw API dicts to MarketPriceCreate and upsert."""
        creates = []
        for rec in api_records:
            try:
                creates.append(MarketPriceCreate(
                    commodity=rec.get("commodity") or commodity,
                    market_name=rec.get("market") or "Unknown Market",
                    district=rec.get("district") or "",
                    state=rec.get("state") or "",
                    min_price=float(rec.get("min_price", 0)),
                    max_price=float(rec.get("max_price", 0)),
                    modal_price=float(rec.get("modal_price", 0)),
                    unit="Quintal",
                    price_date=rec.get("arrival_date") or datetime.utcnow(),
                    source="agmarknet_api",
                ))
            except Exception as exc:
                logger.warning(f"[MARKET SERVICE] Skipping malformed API record: {exc}")
                continue

        if creates:
            await self.repository.upsert_prices(creates)

    # ------------------------------------------------------------------
    # Public: WhatsApp reply formatter
    # ------------------------------------------------------------------

    def format_whatsapp_reply(
        self, query_response: MarketPriceQueryResponse, language: str = "en"
    ) -> str:
        """
        Format a MarketPriceQueryResponse into a WhatsApp-friendly text block.
        Telugu labels are used when language == "te".
        Returns an honest "unavailable" message when no data is found.
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

        source_str = labels["source_live"] if query_response.is_live else labels["source_local"]
        lines = [labels["title"].format(commodity=commodity_display)]

        for r in deduplicated:
            date_str = r.price_date.strftime("%d %b %Y")
            if language == "te":
                block = (
                    f"\n{labels['market']}: {r.market_name}, {r.state}\n"
                    f"{labels['modal']}: ₹{r.modal_price:,.0f}/{labels['unit_suffix']}\n"
                    f"{labels['min']}: ₹{r.min_price:,.0f} | {labels['max']}: ₹{r.max_price:,.0f}\n"
                    f"{labels['date']}: {date_str}"
                )
            else:
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
    for kw, canonical in COMMODITY_MAP.items():
        if kw.lower() in query_lower:
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

    try:
        from sqlalchemy import select
        from src.core.models import FarmerProfile
        if farmer and hasattr(farmer, "id"):
            profile_result = await db.execute(
                select(FarmerProfile).where(FarmerProfile.farmer_id == farmer.id)
            )
            profile = profile_result.scalar_one_or_none()
            if profile:
                if isinstance(getattr(profile, "district", None), str):
                    district = profile.district
                if isinstance(getattr(profile, "state", None), str):
                    state = profile.state
                if not matched_commodity and isinstance(getattr(profile, "current_crop", None), str) and profile.current_crop:
                    matched_commodity = profile.current_crop
    except Exception as exc:
        logger.warning(f"[MARKET ENRICH] Could not load farmer profile: {exc}")

    logger.info(
        f"[MARKET ENRICH] Parameters -> matched_commodity='{matched_commodity}', "
        f"district='{district}', state='{state}', language='{language}'"
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
        )
        svc = MarketService(repository=repo, client=client)
        query_response = await svc.get_prices_for_query(
            commodity=matched_commodity,
            district=district,
            state=state,
        )

        logger.info(
            f"[MARKET ENRICH] Query response -> data_available={query_response.data_available}, "
            f"record_count={len(query_response.results)}, is_live={query_response.is_live}"
        )

        price_block = svc.format_whatsapp_reply(query_response, language=language)

        if query_response.data_available:
            logger.info(
                f"[MARKET ENRICH] Appending {len(query_response.results)} price records "
                f"for '{matched_commodity}' to AI response."
            )
            
            # If the preceding AI response is a generic refusal, non-farming disclaimer,
            # or tells the farmer to contact e-NAM/local market yard, remove it to avoid contradiction.
            refusal_markers = [
                "e-nam", "ఈ-నామ్", "ఈ - నామ్", "మార్కెట్ యార్డ్", "మార్కెట్ యార్డు",
                "market yard", "కేవలం వ్యవసాయం", "విషయాలపై మాత్రమే", "i can only help with farming",
                "only help with farming", "how can i help with your crops", "స్థానిక మార్కెట్",
            ]
            ai_lower = ai_response.lower() if ai_response else ""
            is_refusal = any(marker in ai_lower for marker in refusal_markers)

            if is_refusal or not ai_response or len(ai_response.strip()) == 0:
                final_enriched = price_block
            else:
                final_enriched = ai_response.strip() + "\n\n" + price_block

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
