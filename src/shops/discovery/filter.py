"""
BhoomiMitra AI — Deterministic Agricultural Shop Filter

Validates whether an externally discovered business is genuinely an
agricultural input store (fertilizers, seeds, pesticides, farming supplies)
and decisively filters out non-agricultural entities (supermarkets, clinics,
apparel, jewelers, bakeries, mechanics, human pharmacies, etc.).
"""
from typing import Optional, Set
from src.core.logging import logger
from src.shops.discovery.schemas import DiscoveredShopItem

# Whitelist / positive agricultural keywords in English, Telugu, and Hindi
_AGRI_POSITIVE_KEYWORDS: Set[str] = {
    # English
    "fertilizer", "fertilizers", "fertiliser", "fertilisers",
    "seed", "seeds",
    "pesticide", "pesticides", "insecticide", "insecticides", "fungicide", "fungicides",
    "agro", "agri", "agricultural", "agriculture",
    "krishi", "kisan", "raithu", "rythu", "raitu",
    "farm supply", "farm supplies", "farming supply", "crop care", "crop protection", "crop science",
    "bio organic", "bio fertilizers", "bio-fertilizers", "micro nutrients",
    "agro chemicals", "agrochemicals", "agro agencies", "agro center", "agro centre",
    "pacs", "cooperative society", "rythu seva", "kisan seva",
    # Telugu
    "ఎరువు", "ఎరువులు", "విత్తనాలు", "పురుగుమందులు", "పురుగుల మందు",
    "రైతు", "వ్యవసాయ", "కృషి", "సొసైటీ", "యూరియా",
    # Hindi
    "खाद", "बीज", "कीटनाशक", "कृषि", "किसान", "यूरिया", "उर्वरक",
}

# Negative keywords for obviously non-agricultural businesses
_NON_AGRI_NEGATIVE_KEYWORDS: Set[str] = {
    "supermarket", "super market", "hypermarket", "grocery", "kirana", "provision",
    "departmental store", "general store",
    "apparel", "clothing", "textiles", "tailor", "footwear", "shoes", "garments",
    "jeweller", "jewellers", "jewelry", "gold", "silver",
    "medical hall", "pharmacy", "druggist", "chemist", "dental", "clinic", "hospital",
    "opticals", "eye care",
    "bakery", "sweets", "restaurant", "hotel", "cafe", "dhaba", "caterer", "bar",
    "wine", "liquor", "beverages",
    "mobile", "electronics", "computer", "gadgets", "appliances",
    "mechanic", "automobile", "motors", "tyres", "car wash", "bike service",
    "furniture", "timber", "hardware & sanitary", "paints",
    "salon", "beauty parlour", "spa", "tattoo",
    "pet shop", "aquarium", "pet clinic",
    "school", "college", "institute", "tuition", "academy",
    "bank", "atm", "finance", "chit fund", "real estate",
}


class AgriculturalShopFilter:
    """
    Deterministic rule-based filter for discovered shop candidates.
    Does NOT depend on LLM generation for shop inclusion.
    """

    @classmethod
    def is_agricultural_shop(cls, shop: DiscoveredShopItem) -> bool:
        """
        Evaluate candidate shop based on name, address, business_type, and raw payload.
        Returns True if candidate exhibits clear agricultural markers and no non-agri markers.
        """
        name = (shop.shop_name or "").lower().strip()
        address = (shop.address or "").lower().strip()
        b_type = (shop.business_type or "").lower().strip()
        combined_text = f"{name} {address} {b_type}"

        # 1. Negative check: if name contains strong non-agricultural indicators, reject
        for neg in _NON_AGRI_NEGATIVE_KEYWORDS:
            # Word boundary / substring check on name
            if f" {neg} " in f" {name} " or name.startswith(f"{neg} ") or name.endswith(f" {neg}") or neg == name:
                # Special exemption: if name ALSO explicitly says "agro" or "fertilizer" (e.g. "Kisan General & Agri Store")
                if not any(pos in name for pos in ["fertilizer", "agro", "seed", "pesticide", "krishi", "kisan"]):
                    logger.debug(f"[AGRI FILTER] Rejected non-agri shop '{shop.shop_name}' (matched negative: '{neg}')")
                    return False

        # 2. Positive check: does it match agricultural input indicators?
        has_positive = any(pos in combined_text for pos in _AGRI_POSITIVE_KEYWORDS)
        if has_positive:
            return True

        # 3. If business_type is explicitly an agricultural category
        if b_type in {"fertilizer_dealer", "seed_store", "agricultural_input_store", "farm_supply_store"}:
            return True

        logger.debug(f"[AGRI FILTER] Discarded candidate '{shop.shop_name}' — no agricultural markers matched.")
        return False
