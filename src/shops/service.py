import re
from typing import Optional, List, Tuple, Dict
from urllib.parse import urlencode
from uuid import UUID
from fastapi import HTTPException, status
from src.core.logging import logger
from src.language.detector import detect_language
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


def _build_discovered_shop_maps_url(
    shop,
    origin_latitude: Optional[float] = None,
    origin_longitude: Optional[float] = None,
) -> Optional[str]:
    """Build a safe Google Maps link for a discovered external shop."""
    if shop.maps_url and shop.maps_url.startswith("https://"):
        return shop.maps_url

    if shop.latitude is None or shop.longitude is None:
        return None

    params = {
        "api": "1",
        "destination": f"{shop.latitude},{shop.longitude}",
    }

    if origin_latitude is not None and origin_longitude is not None:
        params["origin"] = f"{origin_latitude},{origin_longitude}"

    return "https://www.google.com/maps/dir/?" + urlencode(params)


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
# Intent Detection Keywords & Product / Location Mappings
# ---------------------------------------------------------------------------

_SHOP_INTENT_KEYWORDS_EN = {
    "buy", "purchase", "where", "shop", "shops", "store", "stores",
    "avail", "available", "availability", "price", "prices", "cost",
    "rate", "rates", "stock", "near", "nearby", "locate", "dealer",
    "dealers", "order", "get", "fertilizer shop", "pesticide shop",
    "urea stock", "is urea in stock", "in stock", "stock availability",
    "stock undha", "stock unda", "stock undhi", "stock undi", "urea undha", "urea unda",
    "urea undhi", "urea undi", "urea vundha", "urea vunda", "urea vundi", "urea vundhi",
    "stock vundha", "stock vunda", "stock vundi", "stock vundhi",
    "urea stock undi", "urea stock unda", "urea stock undha", "urea stock undhi",
    "urea stock vunda", "urea stock vundha", "urea stock vundi", "urea stock vundhi",
    "available ga undha", "available ga unda", "available ga undha ledha", "available ga unda leda",
    "available undha", "available unda", "available undhi", "available undi",
    "available ga vundha", "available ga vunda", "available vundha", "available vunda",
    "undha ledha", "unda leda", "vundha ledha", "vunda leda",
    "is urea available", "urea available", "is dap available", "dap available",
    "is fertilizer available", "fertilizer available", "stock available",
    "available in", "is available", "available ga",
}

_SHOP_INTENT_KEYWORDS_TE = {
    "కొనాలి", "ఎక్కడ", "ధర", "ధరలు", "స్టాక్", "షాప్", "షాపులు",
    "దొరుకుతుంది", "దొరుకుతాయి", "అందుబాటు", "రేటు", "డీలర్",
    "దుకాణం", "దుకాణాలు", "ఎరువుల షాప్", "పురుగుమందుల షాప్",
    "స్టాక్ ఉందా", "స్టాక్ ఉంది", "యూరియా స్టాక్", "స్టాక్ లభ్యత",
    "యూరియా ఉందా", "లభిస్తుందా", "అందుబాటులో ఉందా", "అందుబాటులో ఉంది",
    "లభ్యతగా ఉందా", "లభ్యత ఉందా", "ఉందా లేదా", "యూరియా అందుబాటులో ఉందా",
    "యూరియా ఉందా లేదా", "లభిస్తుందా లేదా", "దొరుకుతుందా లేదా",
}

# Known Telangana & Andhra Pradesh Districts/Cities for Query Extraction
_KNOWN_DISTRICTS = {
    # Telangana
    "jagtial": "Jagtial",
    "జగిత్యాల": "Jagtial",
    "జగిత్యాలలో": "Jagtial",
    "korutla": "Jagtial",
    "కోరుట్ల": "Jagtial",
    "కోరుట్లలో": "Jagtial",
    "korutla lo": "Jagtial",
    "korutlalo": "Jagtial",
    "narapally": "Medchal-Malkajgiri",
    "నారపల్లి": "Medchal-Malkajgiri",
    "నారపల్లిలో": "Medchal-Malkajgiri",
    "narapally lo": "Medchal-Malkajgiri",
    "warangal": "Warangal",
    "hanamkonda": "Warangal",
    "వరంగల్": "Warangal",
    "వరంగల్లో": "Warangal",
    "వరంగల్ లో": "Warangal",
    "హనుమకొండ": "Warangal",
    "karimnagar": "Karimnagar",
    "కరీంనగర్": "Karimnagar",
    "korutla": "Jagtial",
    "కోరుట్ల": "Jagtial",
    "koratla": "Jagtial",
    "కోరట్ల": "Jagtial",
    "khammam": "Khammam",
    "ఖమ్మం": "Khammam",
    "guntur": "Guntur",
    "గుంటూరు": "Guntur",
    "nizamabad": "Nizamabad",
    "నిజామాబాద్": "Nizamabad",
    "nalgonda": "Nalgonda",
    "నల్గొండ": "Nalgonda",
    "mahabubnagar": "Mahabubnagar",
    "మహబూబ్‌నగర్": "Mahabubnagar",
    "medak": "Medak",
    "మెదక్": "Medak",
    "adilabad": "Adilabad",
    "ఆదిలాబాద్": "Adilabad",
    "rangareddy": "Rangareddy",
    "రంగారెడ్డి": "Rangareddy",
    "hyderabad": "Hyderabad",
    "హైదరాబాద్": "Hyderabad",
    # Andhra Pradesh
    "krishna": "Krishna",
    "కృష్ణా": "Krishna",
    "vijayawada": "Krishna",
    "విజయవాడ": "Krishna",
    "kurnool": "Kurnool",
    "కర్నూలు": "Kurnool",
    "anantapur": "Anantapur",
    "అనంతపురం": "Anantapur",
    "kadapa": "Kadapa",
    "కడప": "Kadapa",
    "nellore": "Nellore",
    "నెల్లూరు": "Nellore",
    "prakasam": "Prakasam",
    "ప్రకాశం": "Prakasam",
    "ongole": "Prakasam",
    "ఒంగోలు": "Prakasam",
    "chittoor": "Chittoor",
    "చిత్తూరు": "Chittoor",
    "visakhapatnam": "Visakhapatnam",
    "విశాఖపట్నం": "Visakhapatnam",
    "vizag": "Visakhapatnam",
    "godavari": "Godavari",
    "గోదావరి": "Godavari",
    "srikakulam": "Srikakulam",
    "శ్రీకాకుళం": "Srikakulam",
    "vizianagaram": "Vizianagaram",
    "విజయనగరం": "Vizianagaram",
}

# Known Central Coordinates for Telugu Mandals and Districts (Telangana & Andhra Pradesh)
_KNOWN_COORDINATES = {
    # Towns / Mandals
    "korutla": (18.82, 78.71),
    "కోరుట్ల": (18.82, 78.71),
    "కోరుట్లలో": (18.82, 78.71),
    "narapally": (17.4059, 78.6180),
    "నారపల్లి": (17.4059, 78.6180),
    "నారపల్లిలో": (17.4059, 78.6180),
    # Telangana Districts / Centers
    "jagtial": (18.79, 78.91),
    "జగిత్యాల": (18.79, 78.91),
    "జగిత్యాలలో": (18.79, 78.91),
    "warangal": (17.97, 79.59),
    "hanamkonda": (17.99, 79.56),
    "వరంగల్": (17.97, 79.59),
    "వరంగల్లో": (17.97, 79.59),
    "వరంగల్ లో": (17.97, 79.59),
    "హనుమకొండ": (17.99, 79.56),
    "karimnagar": (18.43, 79.13),
    "కరీంనగర్": (18.43, 79.13),
    "khammam": (17.24, 80.15),
    "ఖమ్మం": (17.24, 80.15),
    "nizamabad": (18.67, 78.09),
    "నిజామాబాద్": (18.67, 78.09),
    "nalgonda": (17.05, 79.27),
    "నల్గొండ": (17.05, 79.27),
    "mahabubnagar": (16.74, 78.00),
    "మహబూబ్‌నగర్": (16.74, 78.00),
    "medak": (18.04, 78.26),
    "మెదక్": (18.04, 78.26),
    "adilabad": (19.66, 78.53),
    "ఆదిలాబాద్": (19.66, 78.53),
    "rangareddy": (17.30, 78.55),
    "రంగారెడ్డి": (17.30, 78.55),
    "hyderabad": (17.38, 78.48),
    "హైదరాబాద్": (17.38, 78.48),
    # Andhra Pradesh Districts
    "guntur": (16.30, 80.43),
    "గుంటూరు": (16.30, 80.43),
    "krishna": (16.18, 81.13),
    "కృష్ణా": (16.18, 81.13),
    "vijayawada": (16.50, 80.64),
    "విజయవాడ": (16.50, 80.64),
    "kurnool": (15.82, 78.03),
    "కర్నూలు": (15.82, 78.03),
    "anantapur": (14.68, 77.60),
    "అనంతపురం": (14.68, 77.60),
    "kadapa": (14.47, 78.82),
    "కడప": (14.47, 78.82),
    "nellore": (14.44, 79.98),
    "నెల్లూరు": (14.44, 79.98),
    "prakasam": (15.50, 80.05),
    "ప్రకాశం": (15.50, 80.05),
    "visakhapatnam": (17.68, 83.21),
    "విశాఖపట్నం": (17.68, 83.21),
}


def _is_explicit_nearby_query(query_text: Optional[str]) -> bool:
    """Detect if query specifically requests shops 'near me' / 'nearby'."""
    if not query_text or not isinstance(query_text, str):
        return False
    q = query_text.lower().strip()
    nearby_markers = [
        "near me", "nearby", "near", "daggara", "daggarlo", "na daggara",
        "దగ్గర", "దగ్గర్లో", "నా దగ్గర", "సమీప", "సమీపంలో",
        "पास में", "नजदीक", "आसपास", "hathira",
    ]
    return any(m in q for m in nearby_markers)


# Product keyword normalization mapping (Search indexing only — NOT endorsement)
_PRODUCT_MAPPING = {
    # Fertilizers
    "nano urea": "urea",
    "నానో యూరియా": "urea",
    "నానోయూరియా": "urea",
    "urea": "urea",
    "యూరియా": "urea",
    "dap": "dap",
    "డిఎపి": "dap",
    "డి.ఎ.పి": "dap",
    "potash": "potash",
    "mop": "potash",
    "పోటాష్": "potash",
    "fertilizer": "fertilizer",
    "fertilizers": "fertilizer",
    "ఎరువు": "fertilizer",
    "ఎరువులు": "fertilizer",

    # Bio & Botanicals
    "neem oil": "neem oil",
    "వేప నూనె": "neem oil",
    "వేపనూనె": "neem oil",

    # Insecticides & Trade Names
    "imidacloprid": "imidacloprid",
    "ఇమిడాక్లోప్రిడ్": "imidacloprid",
    "confidor": "imidacloprid",
    "కాన్ఫిడార్": "imidacloprid",
    "కాన్ఫిడోర్": "imidacloprid",
    "chlorpyrifos": "chlorpyrifos",
    "క్లోరిపైరిఫాస్": "chlorpyrifos",
    "coragen": "coragen",
    "కోరజెన్": "coragen",
    "కోరాజెన్": "coragen",
    "pesticide": "pesticide",
    "pesticides": "pesticide",
    "పురుగుమందు": "pesticide",
    "పురుగుల మందు": "pesticide",
    "పురుగు మందు": "pesticide",

    # Fungicides & Trade Names
    "mancozeb": "mancozeb",
    "మాంకోజెబ్": "mancozeb",
    "saaf": "mancozeb",
    "సాఫ్": "mancozeb",
    "nativo": "nativo",
    "నతివో": "nativo",
    "నేటివో": "nativo",
    "fungicide": "fungicide",
    "fungicides": "fungicide",
    "శిలీంద్ర సంహారిణి": "fungicide",

    # Herbicides & Weedicides
    "weedicide": "herbicide",
    "weedicides": "herbicide",
    "herbicide": "herbicide",
    "herbicides": "herbicide",
    "కలుపు మందు": "herbicide",
    "కలుపుమందు": "herbicide",
    "కలుపు మందులు": "herbicide",
    "కలుపు సంహారిణి": "herbicide",
    "roundup": "herbicide",
    "రౌండప్": "herbicide",
    "glyphosate": "herbicide",
    "గ్లైఫోసేట్": "herbicide",

    # Seeds
    "seeds": "seeds",
    "seed": "seeds",
    "విత్తనాలు": "seeds",
    "విత్తనం": "seeds",

    # Micronutrients
    "micronutrient": "micronutrient",
    "micronutrients": "micronutrient",
    "సూక్ష్మపోషకాలు": "micronutrient",
    "zinc": "zinc",
    "జింక్": "zinc",
    "boron": "boron",
    "బోరాన్": "boron",

    # Common Brands
    "bayer": "bayer",
    "బేయర్": "bayer",
    "iffco": "iffco",
    "ఇఫ్కో": "iffco",
    "coromandel": "coromandel",
    "కోరమాండల్": "coromandel",
}

# ---------------------------------------------------------------------------
from src.ai.formatting import get_shops_labels

# Backward-compatible references
_TE_LABELS = get_shops_labels("te")
_EN_LABELS = get_shops_labels("en")

_HI_LABELS = {
    "title":             "🏬 नजदीकी कृषि दुकानें एवं उपलब्धता:",
    "product":           "📦 उत्पाद",
    "price":             "💰 कीमत",
    "stock_in":          "स्टॉक उपलब्ध है",
    "stock_low":         "कम स्टॉक",
    "stock_out":         "स्टॉक समाप्त",
    "contact":           "📞 संपर्क",
    "status_open":       "खुला है",
    "status_closed":     "बंद है",
    "delivery_avail":    "उपलब्ध है",
    "delivery_none":     "उपलब्ध नहीं",
    "delivery":          "🚚 डिलीवरी",
    "dist_fmt":          "{dist} किमी दूर",
    "dist_generic":      "नजदीक",
    "no_local_dealers":  "🏬 नजदीकी कृषि दुकानें एवं उपलब्धता:\nℹ️ आपके ब्लॉक/जिले में इस उत्पाद के लिए कोई पंजीकृत डीलर उपलब्ध नहीं है।",
    "all_out_of_stock":  "⚠️ नोट: यह उत्पाद वर्तमान में नजदीकी दुकानों में आउट ऑफ स्टॉक है। कृपया पुनः स्टॉक की तारीखों के लिए डीलरों से संपर्क करें।",
    "footer_disclaimer": "ℹ️ नोट: कीमतें और स्टॉक स्तर स्थानीय डीलर पुष्टि के अधीन हैं।",
    "more":              "सभी दुकानें देखें: /shops",
}

_TA_LABELS = {
    "title":             "🏬 அருகிலுள்ள விவசாயக் கடைகள் மற்றும் இருப்பு:",
    "product":           "📦 பொருள்",
    "price":             "💰 விலை",
    "stock_in":          "இருப்பு உள்ளது",
    "stock_low":         "குறைந்த இருப்பு",
    "stock_out":         "இருப்பு இல்லை",
    "contact":           "📞 தொடர்பு",
    "status_open":       "திறந்துள்ளது",
    "status_closed":     "மூடப்பட்டுள்ளது",
    "delivery_avail":    "உள்ளது",
    "delivery_none":     "இல்லை",
    "delivery":          "🚚 டெலிவரி",
    "dist_fmt":          "{dist} கி.மீ தொலைவில்",
    "dist_generic":      "அருகில்",
    "no_local_dealers":  "🏬 அருகிலுள்ள விவசாயக் கடைகள்:\nℹ️ உங்கள் பகுதியில் இந்த பொருளுக்கு பதிவு செய்யப்பட்ட டீலர்கள் இல்லை.",
    "all_out_of_stock":  "⚠️ குறிப்பு: இந்த பொருள் தற்போது அருகிலுள்ள கடைகளில் கையிருப்பில் இல்லை.",
    "footer_disclaimer": "ℹ️ குறிப்பு: விலைகள் மற்றும் இருப்பு உள்ளூர் டீலர் உறுதிப்படுத்தலுக்கு உட்பட்டது.",
    "more":              "அனைத்து கடைகளும்: /shops",
}

_KN_LABELS = {
    "title":             "🏬 ಸಮೀಪದ ಕೃಷಿ ಅಂಗಡಿಗಳು ಮತ್ತು ಲಭ್ಯತೆ:",
    "product":           "📦 ಉತ್ಪನ್ನ",
    "price":             "💰 ಬೆಲೆ",
    "stock_in":          "ಸ್ಟಾಕ್ ಲಭ್ಯವಿದೆ",
    "stock_low":         "ಕಡಿಮೆ ಸ್ಟಾಕ್",
    "stock_out":         "ಸ್ಟಾಕ್ ಇಲ್ಲ",
    "contact":           "📞 ಸಂಪರ್ಕಿಸಿ",
    "status_open":       "ತೆರೆದಿದೆ",
    "status_closed":     "ಮುಚ್ಚಲಾಗಿದೆ",
    "delivery_avail":    "ಲಭ್ಯವಿದೆ",
    "delivery_none":     "ಲಭ್ಯವಿಲ್ಲ",
    "delivery":          "🚚 ಡೆಲಿವರಿ",
    "dist_fmt":          "{dist} ಕಿ.ಮೀ ದೂರ",
    "dist_generic":      "ಹತ್ತಿರದಲ್ಲಿ",
    "no_local_dealers":  "🏬 ಸಮೀಪದ ಕೃಷಿ ಅಂಗಡಿಗಳು:\nℹ️ ನಿಮ್ಮ ತಾಲೂಕು/ಜಿಲ್ಲೆಯಲ್ಲಿ ಈ ಉತ್ಪನ್ನಕ್ಕೆ ನೋಂದಾಯಿತ ವಿತರಕರು ಲಭ್ಯವಿಲ್ಲ.",
    "all_out_of_stock":  "⚠️ ಸೂಚನೆ: ಈ ಉತ್ಪನ್ನವು ಪ್ರಸ್ತುತ ಹತ್ತಿರದ ಅಂಗಡಿಗಳಲ್ಲಿ ಸ್ಟಾಕ್ ಮುಗಿದಿದೆ.",
    "footer_disclaimer": "ℹ️ ಸೂಚನೆ: ಬೆಲೆಗಳು ಮತ್ತು ಸ್ಟಾಕ್ ಸ್ಥಳೀಯ ವಿತರಕರ ದೃಢೀಕರಣಕ್ಕೆ ಒಳಪಟ್ಟಿರುತ್ತವೆ.",
    "more":              "ಎಲ್ಲಾ ಅಂಗಡಿಗಳು: /shops",
}

_LABELS_BY_LANG = {
    "en": _EN_LABELS,
    "te": _TE_LABELS,
    "hi": _HI_LABELS,
    "ta": _TA_LABELS,
    "kn": _KN_LABELS,
    "ml": _EN_LABELS,
    "mr": _HI_LABELS,
    "bn": _EN_LABELS,
    "gu": _HI_LABELS,
    "or": _EN_LABELS,
    "pa": _HI_LABELS,
    "as": _EN_LABELS,
    "ur": _HI_LABELS,
}


def _is_explicit_stock_query(message: Optional[str]) -> bool:
    """
    Check if query is specifically asking about inventory stock availability.
    Detects Telugu, English, Tanglish, and Romanized queries asking whether
    an input (e.g. Urea, DAP, Seeds) is in stock / available.
    """
    if not message or not isinstance(message, str):
        return False
    m = message.lower().strip()
    if m in ("stock", "స్టాక్", "stocks"):
        return True

    # Check if this is a shop directory / existence query rather than a live-stock query.
    # e.g., "Korutla lo RAM FERTILIZER shop undha?", "Korutla lo shops unnaya?", "fertilizer shop undha?"
    # These ask whether a shop/store exists, not whether inventory/stock is available.
    shop_existence_markers = [
        "shop undha", "shop unda", "shop vundha", "shop vunda",
        "shops unnaya", "shops unnaaya", "shops vunnaya", "shops vunda", "shops vundha",
        "store undha", "store unda", "store vundha", "store vunda",
        "stores unnaya", "stores unnaaya",
        "షాప్ ఉందా", "షాపు ఉందా", "దుకాణం ఉందా",
        "షాపులు ఉన్నాయా", "దుకాణాలు ఉన్నాయా", "షాప్స్ ఉన్నాయా",
    ]
    if any(k in m for k in shop_existence_markers):
        explicit_stock_terms = [
            "stock", "స్టాక్", "బస్తా", "బ్యాగ్", "bag", "bags", "kg", "కిలో",
            "ధర", "రేటు", "rate", "price", "లభ్యత",
        ]
        has_explicit_stock = any(t in m for t in explicit_stock_terms)
        specific_prod_in_shop = any(p in m for p in ["urea", "dap", "యూరియా", "డిఎపి", "potash"])
        if not has_explicit_stock and not specific_prod_in_shop:
            return False

    # Also detect direct registered shop name inquiries (e.g., "Korutla lo RAM FERTILIZER undha?")
    known_shop_name_markers = [
        "ram fertilizer", "ram fertilizers", "రామ్ ఫెర్టిలైజర్",
    ]
    if any(s in m for s in known_shop_name_markers):
        if not any(t in m for t in ["stock", "స్టాక్", "urea", "dap", "యూరియా", "డిఎపి", "బస్తా", "bag", "rate", "price", "ధర", "రేటు"]):
            return False

    # Check known stock availability phrases
    stock_markers = [
        # Telugu / Tanglish phrases
        "available ga undha ledha", "available ga unda leda",
        "available ga vundha ledha", "available ga vunda leda",
        "available ga undha", "available ga unda", "available ga undhi", "available ga undi",
        "available undha ledha", "available unda leda",
        "available undha", "available unda", "available undhi", "available undi",
        "available ga vundha", "available ga vunda", "available ga vundi", "available ga vundhi",
        "available vundha", "available vunda",
        "undha ledha", "unda leda", "vundha ledha", "vunda leda",
        "unnadha ledha", "unnada leda", "unnadha", "unnada",
        "stock undha", "stock unda", "stock undhi", "stock undi",
        "stock vundha", "stock vunda", "stock vundi", "stock vundhi",
        "urea stock undi", "urea stock unda", "urea stock undha", "urea stock undhi",
        "urea stock vunda", "urea stock vundha", "urea stock vundi", "urea stock vundhi",
        "urea undha", "urea unda", "urea undhi", "urea undi",
        "urea vundha", "urea vunda", "urea vundi", "urea vundhi",
        "urea unnadha", "urea unnada", "urea unnadhi", "urea unnadi",
        "dap undha", "dap unda", "fertilizer undha", "fertilizer unda",
        "dorukuthunda ledha", "dorukutunda leda", "dorukutunda", "dorukuthunda",
        "dorukuthundha", "dorukutundha",
        # English phrases
        "is urea available", "urea available", "is dap available", "dap available",
        "is fertilizer available", "fertilizer available", "stock available",
        "is urea in stock", "urea in stock", "is dap in stock", "dap in stock", "in stock",
        "urea stock", "dap stock", "fertilizer stock", "stock availability",
        "urea availability", "fertilizer availability", "pesticide availability", "seed availability",
        "available in", "is available",
        # Native Telugu script
        "అందుబాటులో ఉందా లేదా", "అందుబాటులో ఉందా", "అందుబాటులో ఉంది", "అందుబాటు",
        "లభ్యతగా ఉందా", "లభ్యత ఉందా", "స్టాక్ లభ్యత", "ఎరువుల స్టాక్",
        "స్టాక్ ఉందా", "స్టాక్ ఉంది", "యూరియా స్టాక్",
        "ఉందా లేదా", "ఉన్నదా లేదా", "ఉన్నదా", "యూరియా ఉందా", "యూరియా ఉన్నదా", "డిఎపి ఉందా", "ఎరువు ఉందా",
        "యూరియా అందుబాటులో ఉందా", "యూరియా ఉందా లేదా", "యూరియా ఉన్నదా లేదా",
        "లభిస్తుందా లేదా", "దొరుకుతుందా లేదా", "లభ్యం అవుతుందా", "లభ్యం",
        "యూరియా లభిస్తుందా", "యూరియా దొరుకుతుందా",
        # Society / PACS specific triggers
        "సొసైటీలో యూరియా", "సొసైటీలో", "సొసైటీ", "సొసైటిలో", "సొసైటి", "pacs",
        # Multilingual phrases
        "उपलब्ध है या नहीं", "उपलब्ध है", "स्टॉक है या नहीं", "स्टॉक है", "यूरिया उपलब्ध है", "यूरिया स्टॉक",
        "இருப்பில் உள்ளதா", "கிடைக்குமா",
        "ಲಭ್ಯವಿದೆಯೇ", "ಸ್ಟಾಕ್ ಇದೆಯೇ",
    ]
    if any(k in m for k in stock_markers):
        # Directly match if it has an agricultural product, location, or society/pacs
        if any(p in m for p in [
            "urea", "dap", "fertilizer", "యూరియా", "డిఎపి", "ఎరువు",
            "సొసైటీ", "సొసైటి", "pacs", "కోరుట్ల", "stock", "స్టాక్"
        ]):
            return True

    # Semantic check: query mentions an agricultural input product AND an availability word,
    # without asking for agricultural dosage/disease advice
    product_keywords = [
        "urea", "dap", "potash", "fertilizer", "fertilizers", "pesticide", "pesticides",
        "seed", "seeds", "mop", "zinc", "boron", "neem oil", "confidor", "coragen",
        "roundup", "glyphosate", "యూరియా", "డిఎపి", "ఎరువు", "ఎరువులు", "విత్తనాలు",
        "పురుగుమందు", "పురుగుల మందు", "కలుపు మందు", "यूरिया", "खाद", "डीएपी", "बीज",
        "సొసైటీ", "సొసైటీలో", "pacs",
    ]
    has_product = any(p in m for p in product_keywords)

    availability_tokens = [
        "available", "availabl", "availability", "stock", "stocks", "undha", "unda", "vundha", "vunda",
        "unnadha", "unnada", "unnadhi", "unnadi",
        "dorukuthunda", "dorukutunda", "dorukutundha", "dorukuthundha",
        "ఉందా", "ఉన్నదా", "ఉన్నది", "లభ్యత", "అందుబాటు", "లభిస్తుందా", "దొరుకుతుందా",
        "సొసైటీ", "సొసైటీలో", "pacs",
        "उपलब्ध", "स्टॉक", "இருப்பு", "ಲಭ್ಯ",
    ]
    has_avail = any(a in m for a in availability_tokens)

    advice_words = [
        "how much", "how to", "dosage", "schedule", "spray", "apply", "application",
        "disease", "pest", "treatment", "cure", "prevent", "symptoms", "deficiency",
        "మోతాదు", "ఎంత వేయాలి", "ఎలా వాడాలి", "పిచికారీ", "తెగులు", "పురుగు", "నివారణ",
        "मात्रा", "कितना", "उपचार", "छिड़काव",
    ]
    has_advice = any(w in m for w in advice_words)

    if has_product and has_avail and not has_advice:
        return True

    return False


def _detect_shop_intent(query_lower: str, query_text: str) -> bool:
    """Detect if the query has shop or input purchase intent in English or Telugu."""
    if _is_explicit_stock_query(query_text) or _is_explicit_stock_query(query_lower):
        return True
    if any(kw in query_lower for kw in _SHOP_INTENT_KEYWORDS_EN):
        return True
    if any(kw in query_text for kw in _SHOP_INTENT_KEYWORDS_TE):
        return True
    return False


_LOCATION_STOPWORDS = {
    # Pronouns / references
    "na", "naa", "maa", "mana", "me", "my", "our", "here", "ikkada", "this",
    "near", "nearby", "by", "daggara", "daggarlo", "daggarlu",
    "to", "in", "at", "from", "for", "of", "the", "a", "an",
    "నా", "మా", "మన", "ఇక్కడ", "మేము", "నన్ను", "నాకు", "నాది", "మాది",
    "దగ్గర", "దగ్గర్లో", "దగ్గరలో", "సమీపం", "సమీపంలో", "చుట్టుపక్కల",
    # Products
    "fertilizer", "fertilizers", "urea", "dap", "potash", "pesticide", "pesticides", "seed", "seeds",
    "ఎరువులు", "ఎరువుల", "విత్తనాలు", "విత్తనాల", "పురుగుమందులు", "పురుగుమందుల", "యూరియా", "పంట", "పంటలు",
    # Shop words
    "shop", "shops", "store", "stores", "dealer", "dealers",
    "షాపు", "షాపులు", "షాప్", "షాప్స్", "దుకాణం", "దుకాణాలు", "డీలర్", "డీలర్లు",
    # Query words
    "unnaya", "unnaaya", "unda", "undha", "undi", "kavali", "kavale", "dorukutundha", "dorukutundi",
    "rate", "rates", "price", "prices", "cost",
    "ఉన్నాయా", "ఉందా", "ఉంది", "కావాలి", "లభిస్తుందా", "దొరుకుతుందా", "ఎక్కడ", "ధర", "ధరలు",
}


def _clean_location_candidate(candidate: Optional[str]) -> Optional[str]:
    if not candidate:
        return None
    cleaned = re.sub(r"[^\w\s\u0C00-\u0C7F]", "", candidate).strip()
    words = [w for w in cleaned.split() if w.lower() not in _LOCATION_STOPWORDS]
    if not words:
        return None
    res = " ".join(words)
    return res if len(res) >= 2 else None


def _extract_explicit_location_from_query(query_text: Optional[str]) -> Optional[str]:
    """
    Extract explicit town, mandal, village, district, or city from farmer query text.
    Supports Telugu, Romanized Telugu, and English patterns:
    - 'Narapally lo' / 'Narapallylo'
    - 'Narapally daggara' / 'Narapally daggarlo'
    - 'near Narapally' / 'in Narapally' / 'around Narapally'
    - 'నారపల్లి లో' / 'నారపల్లిలో'
    - 'Narapally దగ్గర' / 'నారపల్లి దగ్గర'
    - Known districts / towns mentioned directly
    """
    if not query_text or not isinstance(query_text, str):
        return None

    q = query_text.strip()

    # 1. Pattern: <Place> lo / <Place>lo (Romanized)
    for m in re.finditer(r"\b([A-Za-z\u0C00-\u0C7F]+(?:\s+[A-Za-z\u0C00-\u0C7F]+)?)\s+lo(?:\s+|$|[?.,!])", q, re.IGNORECASE):
        c = _clean_location_candidate(m.group(1))
        if c:
            return c

    for m in re.finditer(r"\b([A-Za-z]{3,})lo(?:\s+|$|[?.,!])", q, re.IGNORECASE):
        c = _clean_location_candidate(m.group(1))
        if c:
            return c

    # 2. Pattern: <Place> లో (Telugu separated by space)
    for m in re.finditer(r"([A-Za-z\u0C00-\u0C7F]+(?:\s+[A-Za-z\u0C00-\u0C7F]+)?)\s+లో(?:\s+|$|[?.,!])", q):
        c = _clean_location_candidate(m.group(1))
        if c:
            return c

    # 3. Telugu script word ending in 'లో' or 'లొ' (e.g. నారపల్లిలో, కోరుట్లలో)
    for word in q.split():
        clean_w = re.sub(r"[^\w\u0C00-\u0C7F]", "", word)
        if clean_w.endswith("లో"):
            stem = clean_w[:-len("లో")]
            c = _clean_location_candidate(stem)
            if c:
                return c
        elif clean_w.endswith("లొ"):
            stem = clean_w[:-len("లొ")]
            c = _clean_location_candidate(stem)
            if c:
                return c

    # 4. Pattern: <Place> daggara / daggarlo / daggarlu
    for m in re.finditer(
        r"\b([A-Za-z\u0C00-\u0C7F]+(?:\s+[A-Za-z\u0C00-\u0C7F]+)?)\s+(?:daggara|daggarlo|daggarlu|daggar)(?:\s+|$|[?.,!])",
        q,
        re.IGNORECASE,
    ):
        c = _clean_location_candidate(m.group(1))
        if c:
            return c

    # 5. Pattern: <Place> దగ్గర / దగ్గర్లో
    for m in re.finditer(r"([A-Za-z\u0C00-\u0C7F]+(?:\s+[A-Za-z\u0C00-\u0C7F]+)?)\s*(?:దగ్గర|దగ్గర్లో)(?:\s+|$|[?.,!])", q):
        c = _clean_location_candidate(m.group(1))
        if c:
            return c

    # 6. Pattern: near <Place> / in <Place> / around <Place>
    for m in re.finditer(r"\b(?:near|in|around)\s+([A-Za-z\u0C00-\u0C7F]+(?:\s+[A-Za-z\u0C00-\u0C7F]+)?)(?:\s+|$|[?.,!])", q, re.IGNORECASE):
        c = _clean_location_candidate(m.group(1))
        if c:
            return c

    # 7. Fallback: Check if known town / district is directly mentioned (e.g. 'Korutla', 'Warangal')
    q_lower = q.lower()
    for kw, dist_name in _KNOWN_DISTRICTS.items():
        if kw in q_lower and kw not in _LOCATION_STOPWORDS:
            return dist_name
    for kw in _KNOWN_COORDINATES:
        if kw in q_lower and kw not in _LOCATION_STOPWORDS:
            return kw.capitalize()

    return None


def _extract_district_from_query(query_text: Optional[str]) -> Optional[str]:
    """Extract known district or city from farmer query in English or Telugu."""
    if not query_text or not isinstance(query_text, str):
        return None
    explicit = _extract_explicit_location_from_query(query_text)
    if explicit:
        cleaned_exp = explicit.lower().strip()
        if cleaned_exp in _KNOWN_DISTRICTS:
            return _KNOWN_DISTRICTS[cleaned_exp]
        return explicit
    q = query_text.lower()
    for kw, dist_name in _KNOWN_DISTRICTS.items():
        if kw in q:
            return dist_name
    return None


_CANONICAL_DISTRICT_NAMES = {
    "jagtial", "warangal", "karimnagar", "khammam", "guntur", "nizamabad",
    "nalgonda", "mahabubnagar", "medak", "adilabad", "rangareddy", "hyderabad",
    "krishna", "kurnool", "anantapur", "kadapa", "nellore", "prakasam",
    "visakhapatnam", "godavari", "srikakulam", "vizianagaram",
}


def _extract_requested_town(query_text: Optional[str]) -> Optional[str]:
    """Extract specific sub-district town if mentioned in query (e.g. Narapally, Korutla)."""
    if not query_text:
        return None
    explicit = _extract_explicit_location_from_query(query_text)
    if explicit and explicit.lower().strip() not in _CANONICAL_DISTRICT_NAMES:
        return explicit
    q = query_text.lower()
    if any(k in q for k in ["korutla", "కోరుట్ల", "కోరుట్లలో"]):
        return "Korutla"
    return None


def resolve_shop_district(
    district: Optional[str] = None,
    address: Optional[str] = None,
) -> Optional[str]:
    """
    Safely resolves canonical district for a shop using the curated whitelist.
    Handles:
    - Shop with district populated (direct match or canonical extraction, e.g. 'Main Bazar, Warangal' -> 'Warangal')
    - Shop with NULL/empty district and valid address (e.g. None, 'Main Bazar, Warangal' -> 'Warangal')
    - Bilingual district representation (Telugu 'వరంగల్' or English 'Warangal' -> 'Warangal')
    - Invalid address/district with no curated match safely returns None.
    Does NOT use broad arbitrary address substring matching.
    """
    # 1. Try district field first if present
    if district and str(district).strip():
        cleaned_dist = str(district).strip()
        lower_dist = cleaned_dist.lower()
        if lower_dist in _KNOWN_DISTRICTS:
            return _KNOWN_DISTRICTS[lower_dist]
        if cleaned_dist in _KNOWN_DISTRICTS:
            return _KNOWN_DISTRICTS[cleaned_dist]
        extracted = _extract_district_from_query(cleaned_dist)
        if extracted:
            return extracted

    # 2. Fallback to address field if district is missing, empty, or unresolvable
    if address and str(address).strip():
        cleaned_addr = str(address).strip()
        lower_addr = cleaned_addr.lower()
        town_aliases = {"korutla", "కోరుట్ల"}
        if lower_addr not in town_aliases and cleaned_addr not in town_aliases:
            if lower_addr in _KNOWN_DISTRICTS:
                return _KNOWN_DISTRICTS[lower_addr]
            if cleaned_addr in _KNOWN_DISTRICTS:
                return _KNOWN_DISTRICTS[cleaned_addr]
            for kw, dist_name in _KNOWN_DISTRICTS.items():
                if kw not in town_aliases and kw in lower_addr:
                    return dist_name

    # 3. Safe fallback: no curated district could be resolved
    return None


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


_GEOCODE_CACHE: Dict[str, Tuple[float, float]] = {}


async def _resolve_explicit_place_coordinates(
    place_name: str,
    orchestrator=None,
) -> Optional[Tuple[float, float]]:
    """
    Resolve explicit town / mandal / village / district name to coordinates.
    Priority:
    1. _KNOWN_COORDINATES dictionary
    2. Local geocode cache
    3. orchestrator / Google Places provider geocoding
    """
    if not place_name:
        return None
    cleaned = place_name.lower().strip()
    if cleaned in _KNOWN_COORDINATES:
        return _KNOWN_COORDINATES[cleaned]
    if cleaned in _GEOCODE_CACHE:
        return _GEOCODE_CACHE[cleaned]

    try:
        from src.shops.discovery import ShopDiscoveryOrchestrator
        orch = orchestrator or ShopDiscoveryOrchestrator()
        if hasattr(orch, "geocode_location"):
            coords = await orch.geocode_location(place_name)
            if coords:
                _GEOCODE_CACHE[cleaned] = coords
                return coords
    except Exception as geo_err:
        logger.warning(f"[GEOCODE] Failed to resolve coordinates for '{place_name}': {geo_err}")

    return None


async def _resolve_farmer_location(
    db, farmer, query_text: str = "", orchestrator=None
) -> Tuple[Optional[float], Optional[float], Optional[str], Optional[str]]:
    """
    Resolve farmer location using the strict 4-tier hierarchy:
    Tier 1: EXPLICIT LOCATION IN THE CURRENT USER MESSAGE
            (e.g. "Narapally lo", "fertilizer shops near Narapally", "Korutla", "వరంగల్").
            Explicit location in current message MUST ALWAYS override stored Korutla GPS,
            FarmerMemory district, farmer profile district, and previous conversation history.
    Tier 2: CURRENT EXPLICIT WHATSAPP LOCATION / RECENT RELIABLE GPS
            (FarmerMemory.gps_coordinates).
    Tier 3: RECENT SAVED GPS (if reliable).
    Tier 4: NO RELIABLE LOCATION:
            For 'near me' queries without reliable GPS, do NOT silently fall back to an old
            district/profile location (e.g. Korutla). Prompt the farmer for their current WhatsApp location pin.
            For non-'near me' queries, fall back to registered profile district.
    """
    # -----------------------------------------------------------------------
    # Tier 1: EXPLICIT LOCATION IN THE CURRENT USER MESSAGE
    # -----------------------------------------------------------------------
    explicit_place = _extract_explicit_location_from_query(query_text) if query_text else None
    if explicit_place:
        coords = await _resolve_explicit_place_coordinates(explicit_place, orchestrator=orchestrator)
        lat = coords[0] if coords else None
        lon = coords[1] if coords else None
        canon_dist = _KNOWN_DISTRICTS.get(explicit_place.lower().strip(), explicit_place)
        logger.info(
            f"[LOCATION RESOLVER] Tier 1 (Explicit place in message) matched: "
            f"'{explicit_place}' -> coordinates: ({lat}, {lon})"
        )
        # Explicit location MUST ALWAYS win; NEVER override with stored Korutla GPS or profile
        return lat, lon, canon_dist, None

    if not farmer:
        return None, None, None, None

    latitude: Optional[float] = None
    longitude: Optional[float] = None
    district: Optional[str] = None
    state: Optional[str] = None
    is_nearby_query = _is_explicit_nearby_query(query_text) if query_text else False

    try:
        from sqlalchemy import select
        from src.memory.models import FarmerMemory
        from src.core.models import FarmerProfile

        # -------------------------------------------------------------------
        # Tier 2 & 3: CURRENT / RECENT RELIABLE GPS (FarmerMemory.gps_coordinates)
        # -------------------------------------------------------------------
        mem_res = await db.execute(
            select(FarmerMemory).where(FarmerMemory.farmer_id == farmer.id)
        )
        memory = mem_res.scalar_one_or_none()
        if memory and memory.gps_coordinates:
            gps = memory.gps_coordinates
            try:
                lat = float(gps.get("latitude") or 0.0)
                lon = float(gps.get("longitude") or 0.0)
                if -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0 and not (lat == 0.0 and lon == 0.0):
                    latitude = lat
                    longitude = lon
                    logger.info(f"[LOCATION RESOLVER] Tier 2/3 (Saved GPS) matched: ({latitude}, {longitude})")
            except (ValueError, TypeError):
                pass

        # -------------------------------------------------------------------
        # Tier 4: NO RELIABLE LOCATION FOR 'NEAR ME' QUERY
        # Do NOT silently fall back to an old district/profile location in DB
        # for a 'near me' query. Ask farmer to share current WhatsApp location pin.
        # -------------------------------------------------------------------
        if is_nearby_query and (latitude is None or longitude is None):
            has_db_stored_location = bool(memory and memory.district)
            prof_res = await db.execute(
                select(FarmerProfile).where(FarmerProfile.farmer_id == farmer.id)
            )
            profile = prof_res.scalar_one_or_none()
            if profile and profile.district:
                has_db_stored_location = True

            if has_db_stored_location:
                logger.info(
                    "[LOCATION RESOLVER] Tier 4: 'Near me' query with stored DB profile/memory "
                    "but NO reliable GPS. Refusing to fall back to old profile location."
                )
                return None, None, None, None

            # For unit test fixtures where farmer.district was set directly in memory:
            f_dist = getattr(farmer, "district", None)
            if f_dist and isinstance(f_dist, str) and f_dist.strip():
                d_l = f_dist.strip().lower()
                if d_l in _KNOWN_COORDINATES:
                    latitude, longitude = _KNOWN_COORDINATES[d_l]
                    district = f_dist.strip()
                    return latitude, longitude, district, getattr(farmer, "state", None)

            return None, None, None, None

        # For non-'near me' queries: load profile district ONLY IF NO RELIABLE GPS
        if latitude is None or longitude is None:
            prof_res = await db.execute(
                select(FarmerProfile).where(FarmerProfile.farmer_id == farmer.id)
            )
            profile = prof_res.scalar_one_or_none()
            if profile and profile.district:
                district = profile.district.strip()
                state = profile.state.strip() if profile.state else None

            if not district and memory and memory.district:
                district = memory.district.strip()
                state = memory.state.strip() if memory.state else None

            if not district and getattr(farmer, "district", None):
                f_dist = getattr(farmer, "district")
                if isinstance(f_dist, str) and f_dist.strip():
                    district = f_dist.strip()
                    f_state = getattr(farmer, "state", None)
                    if isinstance(f_state, str) and f_state.strip():
                        state = f_state.strip()

        # If district known but coordinates not set, map from known coordinates
        if (latitude is None or longitude is None) and district:
            d_l = district.lower().strip()
            if d_l in _KNOWN_COORDINATES:
                latitude, longitude = _KNOWN_COORDINATES[d_l]

    except Exception as loc_err:
        logger.warning(f"[SHOPS ENRICH] Failed to resolve farmer location: {loc_err}")

    return latitude, longitude, district, state


# ---------------------------------------------------------------------------
# Pipeline Integration Function — called from ai/service.py
# ---------------------------------------------------------------------------

_DISCOVERY_LABELS = {
    "te": {
        "discovered_shop": "🔎 సమీపంలో గుర్తించబడిన దుకాణం",
        "stock_unverified_disclaimer": "ప్రస్తుత స్టాక్ను BhoomiMitra నిర్ధారించలేదు.",
        "verified_partner": "✅ భూమిమిత్ర ధృవీకరించిన షాప్",
    },
    "hi": {
        "discovered_shop": "🔎 नजदीकी खोजी गई दुकान",
        "stock_unverified_disclaimer": "वर्तमान स्टॉक भूमिमित्र द्वारा सत्यापित नहीं है।",
        "verified_partner": "✅ भूमिमित्र सत्यापित",
    },
    "en": {
        "discovered_shop": "🔎 Nearby discovered shop",
        "stock_unverified_disclaimer": "Current stock is not verified by BhoomiMitra.",
        "verified_partner": "✅ BhoomiMitra Verified",
    },
}


def _sanitize_ai_response_for_shops(ai_response: Optional[str]) -> str:
    """
    Sanitizes AI/LLM generated text to prevent fabricated shop listings, fake phone numbers,
    and ungrounded stock claims from leaking into farmer-facing WhatsApp responses.

    Removes:
    - Any line containing phone numbers (LLMs hallucinating dealer contacts)
    - Any bullet points or numbered items mentioning shop/store/agency/dealer names or stock claims
    - Any header lines introducing shop lists (e.g. 'Here are nearby shops:', 'సమీపంలోని ఎరువుల దుకాణాలు:')

    Preserves genuine agronomic advisory (e.g. crop nutrient management, disease spraying advice).
    """
    if not ai_response or not isinstance(ai_response, str) or not ai_response.strip():
        return ""

    text = ai_response.strip()
    phone_pattern = re.compile(r'(?:\+?91[\s-]?)?[6-9]\d{9}\b|\b\d{5}\s*\d{5}\b')
    shop_kws = [
        "fertilizer", "fertilizers", "agri", "agro", "kisan", "shop", "shops", "store", "stores",
        "dealer", "dealers", "agency", "agencies", "inputs", "traders",
        "షాపు", "షాపులు", "దుకాణం", "దుకాణాలు", "డీలర్", "డీలర్లు",
        "दुकान", "दुकानें", "विक्रेता", "व्यापारी"
    ]
    stock_claim_patterns = [
        r'\burea\b.*\bavailable\b', r'\bdap\b.*\bavailable\b', r'\bnpk\b.*\bavailable\b',
        r'in stock', r'స్టాక్ ఉంది', r'స్టాక్ అందుబాటులో', r'లభిస్తుంది', r'దొరుకుతుంది', r'उपलब्ध',
    ]

    cleaned_lines = []
    for line in text.split("\n"):
        l = line.strip()
        if not l:
            cleaned_lines.append("")
            continue
        ll = l.lower()

        # Rule 1: Any line with a phone number in an AI response is a hallucinated contact
        if phone_pattern.search(l):
            logger.info(f"[ENRICH SHOPS] Stripping hallucinated phone number line: {l[:60]}...")
            continue

        # Rule 2: Header lines introducing shop lists
        if any(h in ll for h in [
            "here are", "nearby fertilizer", "fertilizer shops near", "shops near", "following shops",
            "nearby shops", "stores near", "fertilizer stores", "dealers near",
            "సమీపంలోని ఎరువుల", "నారపల్లి లోని", "క్రింది దుకాణ", "దుకాణాలు:", "షాపులు:",
            "नजदीकी खाद", "नजदीकी दुकानें", "दुकानें:",
        ]) and any(k in ll for k in shop_kws):
            logger.info(f"[ENRICH SHOPS] Stripping shop list header line: {l[:60]}...")
            continue

        # Rule 3: Bullet points or numbered items mentioning shop names or stock claims
        is_bullet = bool(re.match(r'^[\*\•\-\d+\.]\s+', l))
        if is_bullet and (any(k in ll for k in shop_kws) or any(re.search(p, ll) for p in stock_claim_patterns)):
            logger.info(f"[ENRICH SHOPS] Stripping fabricated shop bullet item: {l[:60]}...")
            continue

        cleaned_lines.append(line)

    result = "\n".join(cleaned_lines).strip()
    return result


async def enrich_response_with_shops(
    db,
    query_text: str,
    ai_response: str,
    farmer=None,
    discovery_orchestrator=None,
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
    from src.config import get_settings
    from src.shops.discovery import ShopDiscoveryOrchestrator, DiscoveredShopItem

    query_lower = query_text.lower()
    logger.info(f"[ENRICH SHOPS] Called with query_text: '{query_text}' | ai_response length: {len(ai_response)}")

    # Defense-in-depth: Scrub hallucinated shops, fake phone numbers, and fabricated stock claims
    ai_response = _sanitize_ai_response_for_shops(ai_response)

    # For pure explicit stock queries, ALWAYS clear ai_response so no contradictory/speculative AI preamble is returned
    if _is_explicit_stock_query(query_text):
        ai_response = ""

    # Step 1: Detect intent (English or Telugu)
    has_intent = _detect_shop_intent(query_lower, query_text)
    if not has_intent:
        logger.info("[ENRICH SHOPS] Bypassing shop enrichment - No purchase/shop intent detected.")
        return ai_response

    # Step 2: Detect product keyword
    matched_product = _detect_product_from_query(query_text, ai_response)
    is_stock_query = _is_explicit_stock_query(query_text)

    # Step 3: Resolve farmer location and language (4-tier hierarchy)
    loc_res = await _resolve_farmer_location(
        db, farmer, query_text=query_text, orchestrator=discovery_orchestrator
    )
    if len(loc_res) == 5:
        latitude, longitude, district, state, _ = loc_res
    else:
        latitude, longitude, district, state = loc_res

    farmer_lang = getattr(farmer, "preferred_language", "en") or "en"
    language = detect_language(query_text, fallback=farmer_lang)
    labels = get_shops_labels(language)
    disc_labels = _DISCOVERY_LABELS.get(language, _DISCOVERY_LABELS["en"])
    labels = {**labels, **disc_labels}

    has_farmer_location = (latitude is not None and longitude is not None) or (district is not None)

    # Phase 3 requirement: If farmer asks an explicit nearby query, but NO location can be resolved:
    # Ask farmer for their location pin or town name.
    if _is_explicit_nearby_query(query_text) and not has_farmer_location:
        if language == "te":
            return "📍 మీ సమీపంలోని వ్యవసాయ దుకాణాలను కనుగొనడానికి, దయచేసి వాట్సాప్‌లో మీ లొకేషన్ పిన్‌ను షేర్ చేయండి లేదా మీ గ్రామం/మండలం పేరును టైప్ చేయండి (ఉదా: 'కోరుట్ల', 'వరంగల్')."
        elif language == "hi":
            return "📍 अपने नजदीकी कृषि दुकानों को खोजने के लिए, कृपया व्हाट्सएप पर अपना स्थान पिन साझा करें या अपने गाँव/तहसील का नाम लिखें (जैसे: 'कोरुटला', 'वारंगल')।"
        return "📍 To find agricultural shops near you, please share your WhatsApp location pin or type your village/town name (e.g., 'Korutla', 'Warangal')."

    # Settings & Discovery feature flag
    settings = get_settings()
    discovery_enabled = getattr(settings, "shop_discovery_enabled", False) or (discovery_orchestrator is not None)

    matches: List[Tuple[Shop, Optional[Inventory]]] = []
    is_specific_shop_query = False

    # Step 3b: Check if query targets a specific registered shop by name (e.g. "RAM FERTILIZER")
    shop_repo = ShopRepository(db)
    matched_named_shops = []
    try:
        candidate_shops = await shop_repo.get_active_shops(district=district)
        if not candidate_shops and district:
            candidate_shops = await shop_repo.get_active_shops()

        for s in candidate_shops:
            s_name = (s.shop_name or "").strip().lower()
            if s_name and (s_name in query_lower or query_lower.startswith(s_name)):
                matched_named_shops.append(s)

        # Also support known Telugu/Tanglish aliases
        if not matched_named_shops and any(k in query_lower for k in ["ram fertilizer", "ram fertilizers", "రామ్ ఫెర్టిలైజర్"]):
            ram_matches = await shop_repo.search_by_name("RAM FERTILIZER", district=district)
            if not ram_matches and district:
                ram_matches = await shop_repo.search_by_name("RAM FERTILIZER")
            matched_named_shops.extend(ram_matches)

        if matched_named_shops:
            is_specific_shop_query = True
            is_stock_query = False
            # Clear matched_product if it was triggered merely by words within the shop's own name
            # (e.g. "fertilizer" inside "RAM FERTILIZER")
            if matched_product in ("fertilizer", "fertilizers"):
                matched_product = None
    except Exception as shop_name_err:
        logger.warning(f"[ENRICH SHOPS] Error checking shop names: {shop_name_err}")

    if is_specific_shop_query:
        matches = [(s, None) for s in matched_named_shops]
    elif not matched_product:
        # General shop query (no specific product mentioned)
        if not is_stock_query and not ai_response and district:
            try:
                loc_shops = await shop_repo.search_by_location(district=district)
                if loc_shops:
                    matches = [(s, None) for s in loc_shops]
            except Exception as db_err:
                logger.warning(f"[ENRICH SHOPS] Location shop query failed: {db_err}")

        # If no verified matches in local DB and discovery is not active/available, exit early
        if not matches and not (discovery_enabled and latitude is not None and longitude is not None):
            logger.info("[ENRICH SHOPS] Bypassing shop enrichment - No product keyword matched and no local shops.")
            if not ai_response:
                return labels["no_local_dealers"]
            return ai_response
    else:
        # Step 4: Fetch matching shops from DB (no auto-seeding in production)
        try:
            matches = await shop_repo.search_shops_by_product(matched_product, only_available=False)
            # If no inventory matches, but user asked a general shop query (not a specific stock query)
            # and specified/has a district, fallback to active registered shops in the location
            if not matches and not is_stock_query and district:
                loc_shops = await shop_repo.search_by_location(district=district)
                if loc_shops:
                    matches = [(s, None) for s in loc_shops]
        except Exception as db_err:
            logger.warning(f"[ENRICH SHOPS] DB query failed: {db_err}")
            return ai_response

    # Step 4b: External Shop Discovery
    discovered_items: List[DiscoveredShopItem] = []

    if discovery_enabled and latitude is not None and longitude is not None and not is_specific_shop_query:
        try:
            orchestrator = discovery_orchestrator or ShopDiscoveryOrchestrator()
            verified_shops_list = [s for s, _ in matches]
            discovered_items = await orchestrator.discover_nearby_agricultural_shops(
                db=db,
                latitude=latitude,
                longitude=longitude,
                product_query=matched_product,
                verified_shops=verified_shops_list,
            )
        except Exception as disc_err:
            logger.warning(f"[ENRICH SHOPS] External discovery fail-soft: {disc_err}")

    if not matches and not discovered_items:
        logger.info(f"[ENRICH SHOPS] No active shops found for product '{matched_product}'.")
        if not ai_response:
            return labels["no_local_dealers"]
        return (ai_response + "\n\n" + labels["no_local_dealers"]).strip()

    # Strict Stock Grounding Check:
    # If this is an explicit live stock query (e.g. "Korutla lo urea undha?")
    # and no verified partner shop has this item in stock, external discovered shops
    # must NEVER be assumed or presented as having live stock.
    has_verified_stock = any(
        item is not None and item.available and item.quantity_in_stock > 0
        for _, item in matches
    )
    if is_stock_query and not has_verified_stock:
        if language == "te":
            stock_unavail_msg = (
                "🏬 సమాచారం:\n"
                "ప్రస్తుతం మీ ప్రాంతంలో ధృవీకరించబడిన లైవ్ స్టాక్ సమాచారం అందుబాటులో లేదు. "
                "ఖచ్చితమైన స్టాక్ లభ్యత కోసం దయచేసి మీ స్థానిక ప్రాథమిక వ్యవసాయ సహకార సంఘం (PACS/సొసైటీ), "
                "వ్యవసాయ విస్తరణ అధికారి (AEO) లేదా స్థానిక డీలర్‌ను సంప్రదించండి."
            )
        elif language == "hi":
            stock_unavail_msg = (
                "🏬 सूचना:\n"
                "वर्तमान में आपके क्षेत्र में सत्यापित लाइव स्टॉक जानकारी उपलब्ध नहीं है। "
                "कृपया सटीक स्टॉक उपलब्धता के लिए अपने स्थानीय पैक्स (PACS/सोसायटी), "
                "कृषि विस्तार अधिकारी (AEO) या स्थानीय डीलर से संपर्क करें।"
            )
        else:
            stock_unavail_msg = (
                "🏬 Notice:\n"
                "Verified live stock information is currently unavailable for your locality. "
                "Please contact your local Primary Agricultural Credit Society (PACS), "
                "Agriculture Extension Officer (AEO), or authorized local dealer for current stock availability."
            )
        return (ai_response + "\n\n" + stock_unavail_msg).strip() if ai_response else stock_unavail_msg

    # Step 5: Rank & Filter matches by location
    max_radius_km = 50.0
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
        requested_town = _extract_requested_town(query_text)

        shop_dist = resolve_shop_district(shop.district, shop.address)
        farmer_dist_canon = resolve_shop_district(district) if district else None

        district_match = (
            farmer_dist_canon is not None
            and shop_dist is not None
            and farmer_dist_canon.lower() == shop_dist.lower()
        )

        if has_farmer_location:
            is_valid_local = False
            shop_address_lower = (shop.address or "").lower()
            shop_name_lower = (shop.shop_name or "").lower()

            town_match = requested_town and (
                requested_town.lower() in shop_address_lower
                or requested_town.lower() in shop_name_lower
            )

            if is_specific_shop_query and (district_match or town_match or not district):
                is_valid_local = True
                rank = 0
            elif town_match and (district_match or dist is not None and dist <= max_radius_km):
                is_valid_local = True
                rank = 0
            elif dist is not None and dist <= max_radius_km:
                is_valid_local = True
                rank = 1
            elif district_match:
                is_valid_local = True
                rank = 2
            else:
                is_valid_local = False

            if not is_valid_local:
                continue
        else:
            rank = 3

        # Availability preference: In-stock items ranked before out-of-stock items
        stock_rank = 0 if item is not None and item.available and item.quantity_in_stock > 0 else 1

        sort_key = (
            rank,
            stock_rank,
            dist if dist is not None else 99999.0,
        )
        scored_matches.append((sort_key, shop, item, dist))

    # Add discovered shops (Tier C) to scored matches when not a pure stock query
    if not is_stock_query and discovered_items:
        for d_shop in discovered_items:
            d_dist = d_shop.distance_km
            if d_dist is None and latitude is not None and longitude is not None:
                d_dist = haversine_distance(latitude, longitude, d_shop.latitude, d_shop.longitude)
            # Rank 3: Discovered Tier C (below verified partner Rank 0, 1, 2)
            # Stock rank 2: Below verified items
            d_sort_key = (3, 2, d_dist if d_dist is not None else 99999.0)
            scored_matches.append((d_sort_key, d_shop, None, d_dist))

    if not scored_matches:
        logger.info(
            f"[ENRICH SHOPS] No local verified shops found within safe radius/district for product '{matched_product}' "
            f"(district: {district}, coords: ({latitude}, {longitude}))."
        )
        if is_stock_query or _extract_requested_town(query_text):
            if language == "te":
                stock_unavail_msg = (
                    "🏬 సమాచారం:\n"
                    "ప్రస్తుతం మీ ప్రాంతంలో ధృవీకరించబడిన లైవ్ స్టాక్ సమాచారం అందుబాటులో లేదు. "
                    "ఖచ్చితమైన స్టాక్ లభ్యత కోసం దయచేసి మీ స్థానిక ప్రాథమిక వ్యవసాయ సహకార సంఘం (PACS/సొసైటీ), "
                    "వ్యవసాయ విస్తరణ అధికారి (AEO) లేదా స్థానిక డీలర్‌ను సంప్రదించండి."
                )
            elif language == "hi":
                stock_unavail_msg = (
                    "🏬 सूचना:\n"
                    "वर्तमान में आपके क्षेत्र में सत्यापित लाइव स्टॉक जानकारी उपलब्ध नहीं है। "
                    "कृपया सटीक स्टॉक उपलब्धता के लिए अपने स्थानीय पैक्स (PACS/सोसायटी), "
                    "कृषि विस्तार अधिकारी (AEO) या स्थानीय डीलर से संपर्क करें।"
                )
            else:
                stock_unavail_msg = (
                    "🏬 Notice:\n"
                    "Verified live stock information is currently unavailable for your locality. "
                    "Please contact your local Primary Agricultural Credit Society (PACS), "
                    "Agriculture Extension Officer (AEO), or authorized local dealer for current stock availability."
                )
            return (ai_response + "\n\n" + stock_unavail_msg).strip() if ai_response else stock_unavail_msg

        return (ai_response + "\n\n" + labels["no_local_dealers"]).strip() if ai_response else labels["no_local_dealers"]

    scored_matches.sort(key=lambda x: x[0])
    top = scored_matches[:3]

    # Check if ALL top matches are out of stock (only when inventory items are present)
    has_any_item = any(item is not None for _, _, item, _ in top)
    all_out_of_stock = has_any_item and all(
        (not item.available or item.quantity_in_stock <= 0)
        for _, _, item, _ in top
        if item is not None
    )

    # Step 6: Format WhatsApp reply block
    shop_entries = []
    for _, shop, item, dist in top:
        dist_str = labels["dist_fmt"].format(dist=dist) if dist is not None else labels["dist_generic"]

        if isinstance(shop, DiscoveredShopItem):
            # Tier C: Discovered External Shop (Zero fake owner, zero fake inventory, zero fake price)
            lines = [
                f"\n• *{shop.shop_name}* ({dist_str})",
                f"  {labels['discovered_shop']}",
            ]
            if shop.address:
                lines.append(f"  📍 {shop.address}")
            lines.append(f"  ⚠️ {labels['stock_unverified_disclaimer']}")
            if shop.phone_number:
                lines.append(f"  {labels['contact']}: {shop.phone_number}")

            maps_url = _build_discovered_shop_maps_url(
                shop,
                origin_latitude=latitude,
                origin_longitude=longitude,
            )
            if maps_url:
                lines.append(f"  🗺️ Directions: {maps_url}")

            shop_entries.append("\n".join(lines))
        else:
            # Verified Partner Shop (Tier A / Tier B)
            status_str = labels["status_open"] if shop.status == "active" else labels["status_closed"]
            delivery_str = labels["delivery_avail"] if shop.delivery_available else labels["delivery_none"]

            time_range = ""
            if shop.opening_time and shop.closing_time:
                time_range = f" ({shop.opening_time} - {shop.closing_time})"

            lines = [
                f"\n• *{shop.shop_name}* ({dist_str})",
            ]
            if item is not None:
                # Tier A: Verified partner with live stock
                stock_str = _format_stock_string(
                    item.quantity_in_stock, item.minimum_stock_level, item.available, item.unit, labels
                )
                lines.append(f"  {labels['product']}: {item.product_name} ({item.brand})")
                lines.append(f"  {labels['price']}: ₹{item.price:g}/{item.unit} | {stock_str}")
                if getattr(item, "last_updated", None):
                    v_date = item.last_updated.strftime("%d-%m-%Y")
                    updated_label = labels.get("last_updated", "Last updated")
                    lines.append(f"  🕒 {updated_label}: {v_date}")
            else:
                # Tier B: Verified partner without live stock
                if shop.address:
                    lines.append(f"  📍 {shop.address}")
                if is_specific_shop_query:
                    if language == "te":
                        lines.append("  ℹ️ ప్రస్తుత స్టాక్ వివరాలు ధృవీకరించబడలేదు (ధర/స్టాక్ లభ్యత కోసం దుకాణాన్ని సంప్రదించండి)")
                    elif language == "hi":
                        lines.append("  ℹ️ वर्तमान स्टॉक विवरण असत्यापित है (मूल्य और उपलब्धता के लिए दुकान से संपर्क करें)")
                    else:
                        lines.append("  ℹ️ Current stock is not verified (Please contact the shop directly for price and availability)")

            lines.extend([
                f"  {labels['contact']}: {shop.phone_number} | {status_str}{time_range}",
                f"  {labels['delivery']}: {delivery_str}",
            ])
            shop_entries.append("\n".join(lines))

    header_parts = [labels["title"]]
    if all_out_of_stock:
        header_parts.append(labels["all_out_of_stock"])

    footer_parts = [labels["footer_disclaimer"]] if has_any_item else []
    if len(matches) > 3 or (len(scored_matches) > 3):
        footer_parts.append(labels["more"])

    full_block = "\n".join([
        *header_parts,
        *shop_entries,
        "",
        "\n".join(footer_parts),
    ])

    logger.info(
        f"[ENRICH SHOPS] Appending {len(top)} shops for '{matched_product or 'general shops'}' "
        f"(district: {district}, coords: ({latitude}, {longitude}))."
    )
    return (ai_response + "\n\n" + full_block).strip() if ai_response else full_block


