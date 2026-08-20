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


async def enrich_response_with_shops(db, query_text: str, ai_response: str) -> str:
    """
    Auto-detect product recommendations or shop search intent in the conversation
    and append nearby shop availability & prices.
    """
    from src.shops.repository import ShopRepository
    shop_repo = ShopRepository(db)

    # Map terms (including Telugu equivalents) to English product terms for DB query matching
    product_mapping = {
        "urea": "urea",
        "యూరియా": "urea",
        "dap": "dap",
        "డిఎపి": "dap",
        "డి.ఎ.పి": "dap",
        "neem oil": "neem oil",
        "వేప నూనె": "neem oil",
        "imidacloprid": "imidacloprid",
        "ఇమిడాక్లోప్రిడ్": "imidacloprid",
        "pesticide": "pesticide",
        "పురుగుమందు": "pesticide",
        "fertilizer": "fertilizer",
        "ఎరువు": "fertilizer",
        "ఎరువులు": "fertilizer",
        "fungicide": "fungicide",
        "శిలీంద్ర సంహారిణి": "fungicide",
        "herbicide": "herbicide",
        "కలుపు సంహారిణి": "herbicide",
        "micronutrient": "micronutrient",
        "సూక్ష్మపోషకాలు": "micronutrient",
        "seeds": "seeds",
        "విత్తనాలు": "seeds",
        "bayer": "bayer",
        "iffco": "iffco",
        "coromandel": "coromandel"
    }

    query_lower = query_text.lower()
    response_lower = ai_response.lower()

    logger.info(f"[ENRICH SHOPS] Called with query_text: '{query_text}' | ai_response: '{ai_response[:100]}...'")

    # Check if query has explicit search/purchase intent
    intent_keywords = [
        "buy", "purchase", "where", "shop", "store", "avail", "price", "cost", "rate", "stock", "near", "locate", "sell", "order",
        "కొనాలి", "ఎక్కడ", "ధర", "స్టాక్", "షాప్", "దొరుకుతుంది", "అందుబాటు", "రేటు"
    ]
    
    matched_intent_kws = [kw for kw in intent_keywords if kw in query_lower]
    has_intent = len(matched_intent_kws) > 0
    
    logger.info(f"[ENRICH SHOPS] Intent check results: has_intent={has_intent} | matched_intent_kws={matched_intent_kws}")
            
    if not has_intent:
        logger.info("[ENRICH SHOPS] Bypassing shop enrichment - No purchase/shop intent detected.")
        return ai_response

    matched_kw = None
    for kw, eng_kw in product_mapping.items():
        if kw in query_lower or kw in response_lower:
            matched_kw = eng_kw
            break

    if not matched_kw:
        logger.info("[ENRICH SHOPS] Bypassing shop enrichment - No product keyword matched.")
        return ai_response

    matches = await shop_repo.search_shops_by_product(matched_kw)
    logger.info(f"[ENRICH SHOPS] DB Shop matches found for '{matched_kw}': {len(matches) if matches else 0}")
    if not matches:
        return ai_response

    import re
    qty_match = re.search(r'(\d+)\s*(bags?|bottles?|kg|litres?|packets?|pkts?)?', query_lower)
    qty = int(qty_match.group(1)) if qty_match else 1

    shop_section = "\n\n🏬 Available Nearby Shops:\n"
    for shop, item in matches[:3]:
        status_str = "Open" if shop.status == "active" else "Closed"
        delivery_str = "Available" if shop.delivery_available else "Not Available"
        total_est = item.price * qty
        shop_section += (
            f"• {shop.shop_name}\n"
            f"  Product: {item.product_name} ({item.brand})\n"
            f"  Price: ₹{item.price:g} | Stock: {item.quantity_in_stock} {item.unit}s\n"
            f"  Contact: {shop.phone_number} | Status: {status_str} | Delivery: {delivery_str}\n"
        )
        if qty_match:
            shop_section += f"  🛒 Order Request ({qty} {item.unit}s): ₹{total_est:g} - Order sent to Shop Owner!\n"

    final_result = ai_response + shop_section.rstrip()
    logger.info(f"[ENRICH SHOPS] Enrichment success. Final response: '{final_result[:150]}...' (Total length: {len(final_result)})")
    return final_result
