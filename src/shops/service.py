from typing import Optional, List, Tuple
from uuid import UUID
from fastapi import HTTPException, status
from src.core.logging import logger
from src.shops.repository import ShopRepository, haversine_distance
from src.shops.schemas import (
    ShopCreate,
    ShopUpdate,
    ShopResponse,
    PaginatedShopResponse,
    ShopSearchResponse,
    FarmerShopSearchResponse,
    FarmerShopSearchResult,
)


class ShopService:
    def __init__(self, repository: ShopRepository):
        self.repository = repository

    async def create_shop(self, data: ShopCreate) -> ShopResponse:
        shop = await self.repository.create(data)
        logger.info(f"Created new shop '{shop.shop_name}' ({shop.id})")
        return ShopResponse.model_validate(shop)

    async def get_shop_by_id(self, shop_id: UUID) -> ShopResponse:
        shop = await self.repository.get_by_id(shop_id)
        if not shop:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Shop with ID '{shop_id}' not found."
            )
        return ShopResponse.model_validate(shop)

    async def update_shop(self, shop_id: UUID, data: ShopUpdate) -> ShopResponse:
        shop = await self.repository.update(shop_id, data)
        if not shop:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Shop with ID '{shop_id}' not found."
            )
        logger.info(f"Updated shop '{shop.shop_name}' ({shop_id})")
        return ShopResponse.model_validate(shop)

    async def delete_shop(self, shop_id: UUID) -> None:
        deleted = await self.repository.delete(shop_id)
        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Shop with ID '{shop_id}' not found."
            )
        logger.info(f"Deleted shop '{shop_id}'")

    async def list_shops(
        self, page: int = 1, size: int = 20, status_filter: Optional[str] = None
    ) -> PaginatedShopResponse:
        shops, total = await self.repository.list_shops(page=page, size=size, status=status_filter)
        items = [ShopResponse.model_validate(s) for s in shops]
        return PaginatedShopResponse(items=items, total=total, page=page, size=size)

    async def search_by_location(
        self,
        district: Optional[str] = None,
        mandal: Optional[str] = None,
        village: Optional[str] = None,
        pin_code: Optional[str] = None,
    ) -> List[ShopResponse]:
        shops = await self.repository.search_by_location(
            district=district, mandal=mandal, village=village, pin_code=pin_code
        )
        return [ShopResponse.model_validate(s) for s in shops]

    async def get_nearby_shops(
        self, latitude: float, longitude: float, max_radius_km: float = 50.0
    ) -> List[ShopSearchResponse]:
        nearby = await self.repository.get_nearby_shops(
            latitude=latitude, longitude=longitude, max_radius_km=max_radius_km
        )
        res = []
        for shop, dist in nearby:
            s_dict = ShopResponse.model_validate(shop).model_dump()
            s_dict["distance_km"] = dist
            res.append(ShopSearchResponse(**s_dict))
        return res

    async def farmer_product_search(
        self,
        product_query: str,
        farmer_latitude: Optional[float] = None,
        farmer_longitude: Optional[float] = None,
        district: Optional[str] = None,
    ) -> FarmerShopSearchResponse:
        """
        Farmer search engine: Finds nearby or district shops selling the queried product.
        Formats the output according to the BhoomiMitra WhatsApp response contract.
        """
        matches = await self.repository.search_shops_by_product(product_query)

        # Filter by district if provided and shop coordinates not used
        if district:
            matches = [m for m in matches if m[0].district and district.lower() in m[0].district.lower()]

        results: List[FarmerShopSearchResult] = []
        for shop, item in matches:
            dist = None
            if (
                farmer_latitude is not None
                and farmer_longitude is not None
                and shop.latitude is not None
                and shop.longitude is not None
            ):
                dist = haversine_distance(
                    farmer_latitude, farmer_longitude, shop.latitude, shop.longitude
                )

            dist_str = f"{dist} km" if dist is not None else "Nearby"
            delivery_str = "Available" if shop.delivery_available else "Not Available"
            status_str = "Open" if shop.status == "active" else "Closed"

            formatted = (
                f"Shop Name: {shop.shop_name}\n"
                f"Distance: {dist_str}\n"
                f"Product: {item.product_name}\n"
                f"Brand: {item.brand}\n"
                f"Price: ₹{item.price:g}\n"
                f"Stock: {item.quantity_in_stock} {item.unit}s\n"
                f"Phone: {shop.phone_number}\n"
                f"Status: {status_str}\n"
                f"Delivery: {delivery_str}"
            )

            results.append(
                FarmerShopSearchResult(
                    shop_id=shop.id,
                    shop_name=shop.shop_name,
                    owner_name=shop.owner_name,
                    distance_km=dist,
                    product_name=item.product_name,
                    brand=item.brand,
                    price=item.price,
                    discount_price=item.discount_price,
                    unit=item.unit,
                    quantity_in_stock=item.quantity_in_stock,
                    phone_number=shop.phone_number,
                    opening_time=shop.opening_time,
                    closing_time=shop.closing_time,
                    status=status_str,
                    delivery_available=shop.delivery_available,
                    formatted_display=formatted,
                )
            )

        # Sort by distance if distance available
        results.sort(key=lambda r: (r.distance_km if r.distance_km is not None else 999999))

        return FarmerShopSearchResponse(
            query=product_query,
            total_results=len(results),
            results=results,
        )



# ---------------------------------------------------------------------------
# Intent Detection Keywords
# ---------------------------------------------------------------------------

_SHOP_INTENT_KEYWORDS_EN = {
    "buy", "purchase", "where", "shop", "shops", "store", "stores",
    "avail", "available", "availability", "price", "prices", "cost",
    "rate", "rates", "stock", "near", "nearby", "locate", "dealer",
    "dealers", "order", "get", "fertilizer shop", "pesticide shop",
}

_SHOP_INTENT_KEYWORDS_TE = {
    "కొనాలి", "ఎక్కడ", "ధర", "ధరలు", "స్టాక్", "షాప్", "షాపులు",
    "దొరుకుతుంది", "దొరుకుతాయి", "అందుబాటు", "రేటు", "డీలర్",
    "దుకాణం", "దుకాణాలు", "ఎరువుల షాప్", "పురుగుమందుల షాప్",
}

# Product keyword normalization mapping
_PRODUCT_MAPPING = {
    "urea": "urea",
    "యూరియా": "urea",
    "dap": "dap",
    "డిఎపి": "dap",
    "డి.ఎ.పి": "dap",
    "potash": "potash",
    "mop": "potash",
    "పోటాష్": "potash",
    "neem oil": "neem oil",
    "వేప నూనె": "neem oil",
    "వేపనూనె": "neem oil",
    "imidacloprid": "imidacloprid",
    "ఇమిడాక్లోప్రిడ్": "imidacloprid",
    "confidor": "imidacloprid",
    "chlorpyrifos": "chlorpyrifos",
    "క్లోరిపైరిఫాస్": "chlorpyrifos",
    "pesticide": "pesticide",
    "pesticides": "pesticide",
    "పురుగుమందు": "pesticide",
    "పురుగుల మందు": "pesticide",
    "fertilizer": "fertilizer",
    "fertilizers": "fertilizer",
    "ఎరువు": "fertilizer",
    "ఎరువులు": "fertilizer",
    "seeds": "seeds",
    "seed": "seeds",
    "విత్తనాలు": "seeds",
    "విత్తనం": "seeds",
    "fungicide": "fungicide",
    "fungicides": "fungicide",
    "శిలీంద్ర సంహారిణి": "fungicide",
    "herbicide": "herbicide",
    "herbicides": "herbicide",
    "కలుపు సంహారిణి": "herbicide",
    "micronutrient": "micronutrient",
    "micronutrients": "micronutrient",
    "సూక్ష్మపోషకాలు": "micronutrient",
    "zinc": "zinc",
    "జింక్": "zinc",
    "bayer": "bayer",
    "బేయర్": "bayer",
    "iffco": "iffco",
    "ఇఫ్కో": "iffco",
    "coromandel": "coromandel",
    "కోరమాండల్": "coromandel",
}

# ---------------------------------------------------------------------------
# Multilingual Formatting Labels
# ---------------------------------------------------------------------------

_EN_LABELS = {
    "title":             "🏬 Nearby Agricultural Shops & Availability:",
    "product":           "📦 Product",
    "price":             "💰 Price",
    "stock_in":          "In Stock",
    "stock_low":         "Low Stock",
    "stock_out":         "Out of Stock",
    "contact":           "📞 Contact",
    "status_open":       "Open",
    "status_closed":     "Closed",
    "delivery_avail":    "Available",
    "delivery_none":     "Not Available",
    "delivery":          "🚚 Delivery",
    "dist_fmt":          "{dist} km away",
    "dist_generic":      "Nearby",
    "footer_disclaimer": "ℹ️ Note: Prices and stock levels are subject to local dealer confirmation.",
    "more":              "Find all shops at: /shops",
}

_TE_LABELS = {
    "title":             "🏬 సమీప వ్యవసాయ దుకాణాలు & లభ్యత:",
    "product":           "📦 ఉత్పత్తి",
    "price":             "💰 ధర",
    "stock_in":          "స్టాక్ అందుబాటులో ఉంది",
    "stock_low":         "తక్కువ స్టాక్ ఉంది",
    "stock_out":         "స్టాక్ లేదు",
    "contact":           "📞 సంప్రదించండి",
    "status_open":       "తెరిచి ఉంది",
    "status_closed":     "మూసివేయబడింది",
    "delivery_avail":    "అందుబాటులో ఉంది",
    "delivery_none":     "అందుబాటులో లేదు",
    "delivery":          "🚚 డెలివరీ",
    "dist_fmt":          "{dist} కి.మీ దూరం",
    "dist_generic":      "సమీపంలో",
    "footer_disclaimer": "ℹ️ గమనిక: ధరలు మరియు స్టాక్ వివరాలు స్థానిక డీలర్ నిర్ధారణకు లోబడి ఉంటాయి.",
    "more":              "మరిన్ని దుకాణాల కోసం: /shops",
}


def _detect_shop_intent(query_lower: str, query_text: str) -> bool:
    """Detect if the query has shop or input purchase intent in English or Telugu."""
    if any(kw in query_lower for kw in _SHOP_INTENT_KEYWORDS_EN):
        return True
    if any(kw in query_text for kw in _SHOP_INTENT_KEYWORDS_TE):
        return True
    return False


def _detect_product_from_query(query_text: str, ai_response: str = "") -> Optional[str]:
    """Detect and normalize product keyword from user query or AI response."""
    query_lower = query_text.lower()
    response_lower = ai_response.lower()

    for kw, eng_product in _PRODUCT_MAPPING.items():
        if kw in query_lower or kw in query_text or kw in response_lower:
            return eng_product
    return None


def _format_stock_string(quantity: int, min_level: int, available: bool, unit: str, labels: dict) -> str:
    """Format stock status string accurately without inventing numbers."""
    if not available or quantity <= 0:
        return labels["stock_out"]
    if quantity <= min_level:
        return f"{labels['stock_low']} ({quantity} {unit}s)"
    return f"{labels['stock_in']} ({quantity} {unit}s)"


async def _resolve_farmer_location(db, farmer) -> Tuple[Optional[float], Optional[float], Optional[str], Optional[str]]:
    """
    Resolve farmer location using the 4-tier hierarchy:
    1. FarmerMemory.gps_coordinates
    2. FarmerProfile.district & state
    3. FarmerMemory.district & state
    4. None (all-active fallback)
    """
    if not farmer:
        return None, None, None, None

    latitude: Optional[float] = None
    longitude: Optional[float] = None
    district: Optional[str] = None
    state: Optional[str] = None

    try:
        from sqlalchemy import select
        from src.memory.models import FarmerMemory
        from src.core.models import FarmerProfile

        # 1. Check FarmerMemory for GPS coordinates
        mem_res = await db.execute(
            select(FarmerMemory).where(FarmerMemory.farmer_id == farmer.id)
        )
        memory = mem_res.scalar_one_or_none()
        if memory and memory.gps_coordinates:
            gps = memory.gps_coordinates
            try:
                lat = float(gps.get("latitude") or 0.0)
                lon = float(gps.get("longitude") or 0.0)
                if lat != 0.0 and lon != 0.0:
                    latitude = lat
                    longitude = lon
            except (ValueError, TypeError):
                pass

        # 2. Check FarmerProfile for district/state
        prof_res = await db.execute(
            select(FarmerProfile).where(FarmerProfile.farmer_id == farmer.id)
        )
        profile = prof_res.scalar_one_or_none()
        if profile and profile.district:
            district = profile.district.strip()
            state = profile.state.strip() if profile.state else None

        # 3. Check FarmerMemory for district if profile is blank
        if not district and memory and memory.district:
            district = memory.district.strip()
            state = memory.state.strip() if memory.state else None

    except Exception as loc_err:
        logger.warning(f"[SHOPS ENRICH] Failed to resolve farmer location: {loc_err}")

    return latitude, longitude, district, state


# ---------------------------------------------------------------------------
# Pipeline Integration Function — called from ai/service.py
# ---------------------------------------------------------------------------

async def enrich_response_with_shops(
    db,
    query_text: str,
    ai_response: str,
    farmer=None,
) -> str:
    """
    Auto-detect product recommendations or shop search intent in the conversation
    and append nearby shop availability & prices in Telugu or English.

    Always returns the original ai_response unchanged if:
    - No shop intent is detected
    - No product keyword is matched
    - No matching active shops/inventory are found
    - Any exception occurs
    """
    from src.shops.repository import ShopRepository, haversine_distance

    query_lower = query_text.lower()
    logger.info(f"[ENRICH SHOPS] Called with query_text: '{query_text}' | ai_response length: {len(ai_response)}")

    # Step 1: Detect intent (English or Telugu)
    has_intent = _detect_shop_intent(query_lower, query_text)
    if not has_intent:
        logger.info("[ENRICH SHOPS] Bypassing shop enrichment - No purchase/shop intent detected.")
        return ai_response

    # Step 2: Detect product keyword
    matched_product = _detect_product_from_query(query_text, ai_response)
    if not matched_product:
        logger.info("[ENRICH SHOPS] Bypassing shop enrichment - No product keyword matched.")
        return ai_response

    # Step 3: Resolve farmer location and language
    latitude, longitude, district, state = await _resolve_farmer_location(db, farmer)
    language = getattr(farmer, "preferred_language", "en") or "en"
    labels = _TE_LABELS if language == "te" else _EN_LABELS

    # Step 4: Fetch matching shops from DB (auto-seed defaults if empty)
    try:
        shop_repo = ShopRepository(db)
        await shop_repo.seed_default_shops_if_empty()
        matches = await shop_repo.search_shops_by_product(matched_product)
    except Exception as db_err:
        logger.warning(f"[ENRICH SHOPS] DB query failed: {db_err}")
        return ai_response

    if not matches:
        logger.info(f"[ENRICH SHOPS] No active shops found for product '{matched_product}'.")
        return ai_response

    # Step 5: Rank & Filter matches by location
    scored_matches = []
    for shop, item in matches:
        dist: Optional[float] = None
        if (
            latitude is not None
            and longitude is not None
            and shop.latitude is not None
            and shop.longitude is not None
        ):
            dist = haversine_distance(latitude, longitude, shop.latitude, shop.longitude)

        # Sort priority:
        # 1. Distant distance if GPS available (closest first)
        # 2. Exact district match if no GPS
        # 3. Other shops
        district_match = (
            district is not None
            and shop.district is not None
            and district.lower() in shop.district.lower()
        )
        sort_key = (
            0 if dist is not None else (1 if district_match else 2),
            dist if dist is not None else 99999.0,
        )
        scored_matches.append((sort_key, shop, item, dist))

    scored_matches.sort(key=lambda x: x[0])
    top = scored_matches[:3]

    # Step 6: Format WhatsApp reply block
    shop_entries = []
    for _, shop, item, dist in top:
        dist_str = labels["dist_fmt"].format(dist=dist) if dist is not None else labels["dist_generic"]
        status_str = labels["status_open"] if shop.status == "active" else labels["status_closed"]
        delivery_str = labels["delivery_avail"] if shop.delivery_available else labels["delivery_none"]
        stock_str = _format_stock_string(
            item.quantity_in_stock, item.minimum_stock_level, item.available, item.unit, labels
        )

        time_range = ""
        if shop.opening_time and shop.closing_time:
            time_range = f" ({shop.opening_time} - {shop.closing_time})"

        lines = [
            f"\n• *{shop.shop_name}* ({dist_str})",
            f"  {labels['product']}: {item.product_name} ({item.brand})",
            f"  {labels['price']}: ₹{item.price:g}/{item.unit} | {stock_str}",
            f"  {labels['contact']}: {shop.phone_number} | {status_str}{time_range}",
            f"  {labels['delivery']}: {delivery_str}",
        ]
        shop_entries.append("\n".join(lines))

    footer_parts = [labels["footer_disclaimer"]]
    if len(matches) > 3:
        footer_parts.append(labels["more"])

    full_block = "\n".join([
        labels["title"],
        *shop_entries,
        "",
        "\n".join(footer_parts),
    ])

    logger.info(
        f"[ENRICH SHOPS] Appending {len(top)} shops for '{matched_product}' "
        f"(district: {district}, coords: ({latitude}, {longitude}))."
    )
    return ai_response + "\n\n" + full_block

