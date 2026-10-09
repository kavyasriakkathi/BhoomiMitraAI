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
    "cotton": "Cotton", "kapas": "Cotton", "patti": "Cotton", "patthi": "Cotton",
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
    # Hindi / Multilingual common crop names
    "टमाटर": "Tomato", "धान": "Paddy", "चावल": "Paddy", "प्याज": "Onion",
    "कपास": "Cotton", "मक्का": "Maize", "मिर्च": "Chilli", "मूंगफली": "Groundnut",
    "सोयाबीन": "Soybean", "हल्दी": "Turmeric", "गन्ना": "Sugarcane", "केला": "Banana",
    "गेहूं": "Wheat", "ज्वार": "Jowar", "चना": "Bengalgram",
    # Tamil
    "தக்காளி": "Tomato", "நெல்": "Paddy", "வெங்காயம்": "Onion", "பருத்தி": "Cotton",
    "மக்காச்சோளம்": "Maize", "மிளகாய்": "Chilli", "வேர்க்கடலை": "Groundnut",
    # Kannada
    "ಟೊಮೆಟೊ": "Tomato", "ಭತ್ತ": "Paddy", "ಈರುಳ್ಳಿ": "Onion", "ಹತ್ತಿ": "Cotton",
    "ಮೆಕ್ಕೆಜೋಳ": "Maize", "ಮೆಣಸಿನಕಾಯಿ": "Chilli", "ಕಡಲೆಕಾಯಿ": "Groundnut",
}

# Intent keywords — triggers market price enrichment
PRICE_INTENT_KEYWORDS_EN = {
    "price", "mandi", "market price", "rate", "today price",
    "how much", "selling price", "market rate", "crop price",
}
PRICE_INTENT_KEYWORDS_TE = {
    "ధర", "ధరలు", "మండి", "రేటు", "రేట్లు", "నేటి ధర", "మార్కెట్ ధర",
    "మార్కెట్ ధరలు", "మండి ధర", "మండి ధరలు", "ఎంత ధర", "ధర ఎంత", "రేటు ఎంత", "అమ్మకం ధర",
    "క్వింటాల్", "క్వింటాలు", "ఖరీదు", "మార్కెట్", "మార్కెట్లో",
}
PRICE_INTENT_KEYWORDS_MULTILINGUAL = {
    "भाव", "मंडी", "बाजार भाव", "दर", "दाम", "रेट", "ਕੀਮਤ", "ਭਾਅ", "দাম", "দর", "ભાવ", "ଦର", "விலை", "மண்டி", "ಬೆಲೆ", "ದರ", "വില",
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
    "Jagtial": "జగిత్యాల",
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

TELANGANA_DISTRICTS = {
    "warangal", "enumamula", "hanamkonda", "karimnagar", "khammam", "nizamabad",
    "nalgonda", "mahabubnagar", "hyderabad", "medak", "adilabad", "rangareddy",
    "siddipet", "suryapet", "jagtial", "mancherial", "bhadradri kothagudem",
    "vikarabad", "sangareddy", "kamareddy", "rajanna sircilla", "peddapalli",
    "wanaparthy", "jogulamba gadwal", "nagarkurnool", "narayanpet", "mulugu",
    "jayashankar bhupalpally", "jangaon", "yadadri bhuvanagiri",
    "komaram bheem asifabad", "nirmal", "medchal-malkajgiri", "mahabubabad",
    "korutla", "metpally", "sircilla", "tandur", "miryalaguda", "badepally", "bowenpally",
}

TELUGU_TO_ENGLISH_PLACES = {
    "కోరుట్ల": "Korutla",
    "మెట్పల్లి": "Metpally",
    "జగిత్యాల": "Jagtial",
    "వరంగల్": "Warangal",
    "ఎనుమాముల": "Enumamula",
    "నిజామాబాద్": "Nizamabad",
    "సిరిసిల్ల": "Sircilla",
    "రాజన్న సిరిసిల్ల": "Rajanna Sircilla",
    "కరీంనగర్": "Karimnagar",
    "ఖమ్మం": "Khammam",
    "నల్గొండ": "Nalgonda",
    "సూర్యాపేట": "Suryapet",
    "మహబూబ్‌నగర్": "Mahabubnagar",
    "ఆదిలాబాద్": "Adilabad",
    "హైదరాబాద్": "Hyderabad",
    "సిద్దిపేట": "Siddipet",
    "మంచిర్యాల": "Mancherial",
    "భద్రాద్రి": "Bhadradri Kothagudem",
    "కొత్తగూడెం": "Bhadradri Kothagudem",
    "వికారాబాద్": "Vikarabad",
    "తాండూరు": "Tandur",
    "సంగారెడ్డి": "Sangareddy",
    "కామారెడ్డి": "Kamareddy",
    "పెద్దపల్లి": "Peddapalli",
    "వనపర్తి": "Wanaparthy",
    "గద్వాల": "Jogulamba Gadwal",
    "నాగర్‌కర్నూల్": "Nagarkurnool",
    "నారాయణపేట": "Narayanpet",
    "ములుగు": "Mulugu",
    "భూపాలపల్లి": "Jayashankar Bhupalpally",
    "జనగామ": "Jangaon",
    "యాదాద్రి": "Yadadri Bhuvanagiri",
    "ఆసిఫాబాద్": "Komaram Bheem Asifabad",
    "నిర్మల్": "Nirmal",
    "మేడ్చల్": "Medchal-Malkajgiri",
    "మిర్యాలగూడ": "Miryalaguda",
    "బాదేపల్లి": "Badepally",
    "బోయిన్‌పల్లి": "Bowenpally",
    "గుంటూరు": "Guntur",
    "విజయవాడ": "Vijayawada",
    "కర్నూలు": "Kurnool",
    "అనంతపురం": "Anantapur",
    "కడప": "Kadapa",
    "నెల్లూరు": "Nellore",
    "ఒంగోలు": "Ongole",
    "చిత్తూరు": "Chittoor",
    "తిరుపతి": "Tirupati",
    "విశాఖపట్నం": "Visakhapatnam",
    "మదనపల్లె": "Madanapalle",
}

_MARKET_LOCATION_STOPWORDS = {
    # Pronouns / references
    "na", "naa", "maa", "mana", "me", "my", "our", "here", "ikkada", "this",
    "near", "nearby", "by", "daggara", "daggarlo", "daggarlu",
    "to", "in", "at", "from", "for", "of", "the", "a", "an",
    "నా", "మా", "మన", "ఇక్కడ", "మేము", "నన్ను", "నాకు", "నాది", "మాది",
    "దగ్గర", "దగ్గర్లో", "దగ్గరలో", "సమీపం", "సమీపంలో", "చుట్టుపక్కల",
    # Generic place words (e.g. 'oorilo', 'in my village')
    "oori", "ooru", "oorilo", "oorlo", "village", "town", "city", "district", "state",
    "apmc", "yard", "area", "region", "place",
    "ఊరు", "ఊరి", "ఊరిలో", "ఊర్లో", "గ్రామం", "గ్రామంలో", "పట్టణం", "పట్టణంలో", "జిల్లా", "రాష్ట్రం",
    # Temporal / time words
    "today", "todays", "today's", "yesterday", "yesterdays", "yesterday's", "tomorrow", "now",
    "current", "latest", "daily", "present", "update", "updates", "info", "details",
    "eroju", "ee roju", "eeroju", "ninna", "repu", "ivala", "ipudu", "ippudu", "prastutam", "taza", "taaza",
    "eerojullo", "eerojuna",
    "ఈరోజు", "ఈ రోజు", "నేడు", "నేటి", "ఈనాటి", "నిన్న", "రేపు", "ఇవాళ", "ప్రస్తుతం", "తాజా",
    "ఈరోజుల్లో", "ఈరోజులొ", "ఈరోజులలో", "ఈరోజున",
    # Crops / Commodities
    "cotton", "kapas", "paddy", "rice", "chilli", "chili", "maize", "corn", "turmeric",
    "soybean", "groundnut", "wheat", "onion", "tomato", "sugarcane", "banana", "jowar",
    "పత్తి", "కాపాస్", "వరి", "ధాన్యం", "మిర్చి", "మొక్కజొన్న", "పసుపు", "సోయాబీన్",
    "వేరుశనగ", "గోధుమ", "ఉల్లి", "ఉల్లిపాయ", "టమాటా", "చెరుకు", "అరటి", "జొన్న", "శనగ",
    "patti", "patthi", "dhara", "dhare", "rate", "rates", "price", "prices", "cost",
    "రేటు", "రేట్లు", "ధర", "ధరలు", "ఖరీదు", "అమ్మకం",
    # Question / conversational / verbs / conjunctions
    "entha", "enti", "undha", "undi", "unnaya", "how", "much", "what", "is", "are",
    "cheppandi", "cheppu", "teliyacheyandi", "please", "tell", "show", "give", "kavali", "telusukovalani",
    "and", "also", "or", "but", "then", "today", "yesterday", "tomorrow", "now",
    "ఎంత", "ఏంటి", "ఉందా", "ఉంది", "ఉన్నాయా", "చెప్పండి", "చెప్పు", "తెలియజేయండి",
    "మరియు", "ఇంకా", "కానీ", "కూడా", "మరి", "లేదా", "ఈరోజు", "నిన్న", "రేపు", "ఇప్పుడు",
    "market", "mandi", "మార్కెట్", "మండి", "మార్కెట్లో", "మార్కెట్ లో",
}


def clean_location_candidate(candidate: Optional[str]) -> Optional[str]:
    """Clean candidate string and strip stopwords/punctuation."""
    if not candidate or not isinstance(candidate, str):
        return None
    c = candidate.strip()
    c_clean = re.sub(r"^[^\w\u0C00-\u0C7F]+|[^\w\u0C00-\u0C7F]+$", "", c)
    if not c_clean:
        return None
    c_lower = c_clean.lower()
    if c_lower in _MARKET_LOCATION_STOPWORDS:
        return None
    words = [w for w in c_clean.split() if w.lower() not in _MARKET_LOCATION_STOPWORDS]
    if not words:
        return None
    res = " ".join(words)
    return res if len(res) >= 2 else None


def normalize_extracted_location(loc: Optional[str]) -> Optional[str]:
    """Normalize extracted town/village/mandal/district to standard English name if known, else title-cased."""
    if not loc or not isinstance(loc, str):
        return None
    loc = loc.strip()
    if loc in TELUGU_TO_ENGLISH_PLACES:
        return TELUGU_TO_ENGLISH_PLACES[loc]
    loc_lower = loc.lower()
    for tel, eng in TELUGU_TO_ENGLISH_PLACES.items():
        if tel == loc or eng.lower() == loc_lower:
            return eng
    return loc[0].upper() + loc[1:] if len(loc) > 1 and loc.isascii() else loc


def extract_market_explicit_location(query_text: Optional[str]) -> Optional[str]:
    """
    Extract explicitly requested location from farmer's market-price query text.
    Handles ANY location mentioned (village, town, mandal, district, city, APMC, or mandi).
    Supports English, Tanglish, and Telugu patterns:
      - 'Korutla lo' / 'metpally lo' / 'Jagtial lo' / 'Warangal lo'
      - 'కోరుట్లలో' / 'మెట్పల్లిలో' / 'జగిత్యాలలో' / 'వరంగల్లో'
      - 'at Korutla' / 'in Korutla' / 'near Korutla'
      - 'Korutla market' / 'Korutla mandi' / 'కోరుట్ల మార్కెట్'
      - 'Korutla lo cotton rate' / 'Korutla cotton price'
      - 'Sircilla lo patti dhara entha?'
    Returns None if no explicit place was specified (e.g. 'Cotton price entha?', 'నా ఊరిలో', 'Na daggara').
    """
    if not query_text or not isinstance(query_text, str):
        return None
    q = query_text.strip()

    # 1. Pattern: <Place> lo / <Place>lo (Romanized or Telugu with space)
    for m in re.finditer(r"\b([A-Za-z\u0C00-\u0C7F]+(?:\s+[A-Za-z\u0C00-\u0C7F]+)?)\s+lo(?:\s+|$|[?.,!])", q, re.IGNORECASE):
        c = clean_location_candidate(m.group(1))
        if c:
            return normalize_extracted_location(c)

    for m in re.finditer(r"\b([A-Za-z]{3,})lo(?:\s+|$|[?.,!])", q, re.IGNORECASE):
        c = clean_location_candidate(m.group(1))
        if c:
            return normalize_extracted_location(c)

    # 2. Pattern: <Place> లో
    for m in re.finditer(r"([A-Za-z\u0C00-\u0C7F]+(?:\s+[A-Za-z\u0C00-\u0C7F]+)?)\s+లో(?:\s+|$|[?.,!])", q):
        c = clean_location_candidate(m.group(1))
        if c:
            return normalize_extracted_location(c)

    # 3. Telugu script words ending in 'లో' or 'ల్లో' or 'లొ' (e.g. కోరుట్లలో, మెట్పల్లిలో, జగిత్యాలలో, వరంగల్లో)
    for word in q.split():
        clean_w = re.sub(r"[^\w\u0C00-\u0C7F]", "", word)
        if not clean_w or clean_w.lower() in _MARKET_LOCATION_STOPWORDS:
            continue
        if clean_w.endswith("ల్లో"):
            # e.g. వరంగల్లో -> stem వరంగల్, or stem వరంగ
            candidate_stems = [clean_w[:-len("ల్లో")] + "ల్", clean_w[:-len("ల్లో")]]
            for st in candidate_stems:
                c = clean_location_candidate(st)
                if c:
                    return normalize_extracted_location(c)
        elif clean_w.endswith("లో") or clean_w.endswith("లొ"):
            suffix_len = len("లో") if clean_w.endswith("లో") else len("లొ")
            stem = clean_w[:-suffix_len]
            c = clean_location_candidate(stem)
            if c:
                return normalize_extracted_location(c)

    # 4. Pattern: at <Place> / in <Place> / near <Place> / around <Place>
    for m in re.finditer(r"\b(?:at|in|near|around)\s+([A-Za-z\u0C00-\u0C7F]+(?:\s+[A-Za-z\u0C00-\u0C7F]+)?)(?:\s+|$|[?.,!])", q, re.IGNORECASE):
        c = clean_location_candidate(m.group(1))
        if c:
            return normalize_extracted_location(c)

    # 5. Pattern: <Place> market / <Place> mandi / <Place> మార్కెట్ / <Place> మండి
    for m in re.finditer(r"\b([A-Za-z\u0C00-\u0C7F]+(?:\s+[A-Za-z\u0C00-\u0C7F]+)?)\s+(?:market|mandi|మార్కెట్|మండి)(?:\s+|$|[?.,!])", q, re.IGNORECASE):
        c = clean_location_candidate(m.group(1))
        if c:
            return normalize_extracted_location(c)

    # 6. Pattern: <Place> [lo] <crop> price / rate (e.g. 'Korutla cotton price', 'Metpally cotton rate')
    crop_keywords = "|".join([
        "cotton", "kapas", "paddy", "rice", "chilli", "chili", "maize", "corn", "turmeric",
        "soybean", "groundnut", "wheat", "onion", "tomato", "sugarcane", "banana", "jowar",
        "పత్తి", "కాపాస్", "వరి", "ధాన్యం", "మిర్చి", "మొక్కజొన్న", "పసుపు", "సోయాబీన్",
        "వేరుశనగ", "గోధుమ", "ఉల్లి", "ఉల్లిపాయ", "టమాటా", "చెరుకు", "అరటి", "జొన్న", "శనగ", "patti", "patthi"
    ])
    price_keywords = "price|prices|rate|rates|mandi|market|ధర|ధరలు|రేటు|రేట్లు|ఖరీదు|dhara|dhare"
    m_cp = re.search(rf"\b([A-Za-z\u0C00-\u0C7F]+)\s+(?:lo\s+)?(?:{crop_keywords})\s+(?:{price_keywords})\b", q, re.IGNORECASE)
    if m_cp:
        c = clean_location_candidate(m_cp.group(1))
        if c:
            return normalize_extracted_location(c)

    # 7. Fallback to known districts/towns if mentioned standalone
    q_words = re.findall(r"[A-Za-z\u0C00-\u0C7F]+", q)
    for w in q_words:
        w_clean = clean_location_candidate(w)
        if not w_clean:
            continue
        if w_clean in TELUGU_TO_ENGLISH_PLACES:
            return TELUGU_TO_ENGLISH_PLACES[w_clean]
        w_lower = w_clean.lower()
        for tel, eng in TELUGU_TO_ENGLISH_PLACES.items():
            if eng.lower() == w_lower:
                return eng

    return None


def resolve_gps_to_nearest_district(lat: float, lon: float) -> Optional[str]:
    """Find the nearest district or major mandi center from GPS coordinates using distance calculation."""
    if lat == 0.0 and lon == 0.0:
        return None
    from src.shops.service import _KNOWN_COORDINATES
    closest_dist = None
    min_dist_sq = float("inf")
    for place_name, coords in _KNOWN_COORDINATES.items():
        if not place_name.isascii() or coords[0] == 0:
            continue
        dist_sq = (coords[0] - lat) ** 2 + (coords[1] - lon) ** 2
        if dist_sq < min_dist_sq:
            min_dist_sq = dist_sq
            closest_dist = place_name.capitalize()
    return closest_dist


TOWN_TO_DISTRICT_MAP = {
    "korutla": "Jagtial",
    "కోరుట్ల": "Jagtial",
    "metpally": "Jagtial",
    "మెట్పల్లి": "Jagtial",
    "sircilla": "Rajanna Sircilla",
    "సిరిసిల్ల": "Rajanna Sircilla",
}


def normalize_district_name(raw_district: Optional[str]) -> Optional[str]:
    """
    Normalize Telugu and English district/location names to canonical English district names.
    Maps sub-district towns/mandals (e.g. Korutla, Metpally) to their administrative districts (e.g. Jagtial).
    """
    if not raw_district or not isinstance(raw_district, str):
        return None
    d = raw_district.strip()
    d_lower = d.lower()
    if d_lower in TOWN_TO_DISTRICT_MAP:
        return TOWN_TO_DISTRICT_MAP[d_lower]
    if d in TELUGU_TO_ENGLISH_PLACES:
        eng_p = TELUGU_TO_ENGLISH_PLACES[d]
        if eng_p.lower() in TOWN_TO_DISTRICT_MAP:
            return TOWN_TO_DISTRICT_MAP[eng_p.lower()]
        return eng_p
    if d_lower in TELUGU_TO_ENGLISH_PLACES:
        eng_p = TELUGU_TO_ENGLISH_PLACES[d_lower]
        if eng_p.lower() in TOWN_TO_DISTRICT_MAP:
            return TOWN_TO_DISTRICT_MAP[eng_p.lower()]
        return eng_p
    from src.weather.service import _KNOWN_DISTRICTS
    if d in _KNOWN_DISTRICTS:
        return _KNOWN_DISTRICTS[d]
    if d_lower in _KNOWN_DISTRICTS:
        return _KNOWN_DISTRICTS[d_lower]
    for kw, canon in _KNOWN_DISTRICTS.items():
        if kw in d_lower or kw in d:
            return canon
    return d[0].upper() + d[1:] if len(d) > 1 and d.isascii() else d


def infer_state_from_district(district: Optional[str], current_state: Optional[str] = None) -> Optional[str]:
    """
    If current_state is NULL/empty, safely infer Telangana only when the district
    is unambiguously a known Telangana district.
    """
    if current_state and current_state.strip():
        return current_state.strip()
    if not district:
        return None
    canon = normalize_district_name(district)
    if canon and canon.lower() in TELANGANA_DISTRICTS:
        return "Telangana"
    return None



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

from src.ai.formatting import get_market_labels

# Backward-compatible references
_TE_LABELS = get_market_labels("te")
_EN_LABELS = get_market_labels("en")

_LABELS_BY_LANG = {
    "te": _TE_LABELS,
    "en": _EN_LABELS,
    "hi": {
        "title": "📊 {commodity} मंडी भाव",
        "market": "मंडी",
        "modal": "औसत भाव",
        "min": "न्यूनतम",
        "max": "अधिकतम",
        "date": "दिनांक",
        "source_live": "एगमार्कनेट (लाइव)",
        "source_local": "स्थानीय डेटाबेस",
        "unit_suffix": "प्रति क्विंटल",
    },
    "ta": {
        "title": "📊 {commodity} சந்தை விலைகள்",
        "market": "சந்தை",
        "modal": "சராசரி விலை",
        "min": "குறைந்தபட்சம்",
        "max": "அதிகபட்சம்",
        "date": "தேதி",
        "source_live": "அக்மார்க்நெட் (நேரலை)",
        "source_local": "உள்ளூர் தரவுத்தளம்",
        "unit_suffix": "குவிண்டால்",
    },
    "kn": {
        "title": "📊 {commodity} ಮಾರುಕಟ್ಟೆ ದರಗಳು",
        "market": "ಮಂಡಿ",
        "modal": "ಮಾದರಿ ದರ",
        "min": "ಕನಿಷ್ಠ",
        "max": "ಗರಿಷ್ಠ",
        "date": "ದಿನಾಂಕ",
        "source_live": "ಅಗ್ಮಾರ್ಕ್‌ನೆಟ್ (ಲೈವ್)",
        "source_local": "ಸ್ಥಳೀಯ ಡೇಟಾಬೇಸ್",
        "unit_suffix": "ಪ್ರತಿ ಕ್ವಿಂಟಾಲ್",
    },
    "ml": {
        "title": "📊 {commodity} വിപണി വിലകൾ",
        "market": "വിപണി",
        "modal": "ശരാശരി വില",
        "min": "കുറഞ്ഞത്",
        "max": "കൂടിയത്",
        "date": "തീയതി",
        "source_live": "ആഗ്മാർക്ക്നെറ്റ് (തത്സമയം)",
        "source_local": "പ്രാദേശിക ഡാറ്റാബേസ്",
        "unit_suffix": "ക്വിന്റലിന്",
    },
    "mr": {
        "title": "📊 {commodity} बाजारभाव",
        "market": "बाजार समिती",
        "modal": "सरासरी भाव",
        "min": "किमान",
        "max": "कमाल",
        "date": "दिनांक",
        "source_live": "अॅगमार्कनेट (थेट)",
        "source_local": "स्थानिक डेटाबेस",
        "unit_suffix": "प्रति क्विंटल",
    },
    "bn": {
        "title": "📊 {commodity} বাজার দর",
        "market": "বাজার",
        "modal": "গড় দর",
        "min": "সর্বনিম্ন",
        "max": "সর্বোচ্চ",
        "date": "তারিখ",
        "source_live": "অ্যাগমার্কনেট (লাইভ)",
        "source_local": "স্থানীয় ডেটাবেস",
        "unit_suffix": "প্রতি কুইন্টাল",
    },
    "gu": {
        "title": "📊 {commodity} બજાર ભાવ",
        "market": "માર્કેટ યાર્ડ",
        "modal": "સરેરાશ ભાવ",
        "min": "નીચામાં નીચો",
        "max": "ઊંચામાં ઊંચો",
        "date": "તારીખ",
        "source_live": "એગમાર્કનેટ (લાઈવ)",
        "source_local": "સ્થાનિક ડેટાબેઝ",
        "unit_suffix": "પ્રતિ ક્વિન્ટલ",
    },
    "or": {
        "title": "📊 {commodity} ବଜାର ଦର",
        "market": "ମଣ୍ଡି",
        "modal": "ହାରାହାରି ଦର",
        "min": "ସର୍ବନିମ୍ନ",
        "max": "ସର୍ବାଧିକ",
        "date": "ତାରିଖ",
        "source_live": "ଆଗମାର୍କନେଟ୍ (ଲାଇଭ୍)",
        "source_local": "ସ୍ଥାନୀୟ ଡାଟାବେସ୍",
        "unit_suffix": "କ୍ୱିଣ୍ଟାଲ ପିଛା",
    },
    "pa": {
        "title": "📊 {commodity} ਮੰਡੀ ਭਾਅ",
        "market": "ਮੰਡੀ",
        "modal": "ਔਸਤ ਭਾਅ",
        "min": "ਘੱਟੋ-ਘੱਟ",
        "max": "ਵੱਧ ਤੋਂ ਵੱਧ",
        "date": "ਮਿਤੀ",
        "source_live": "ਐਗਮਾਰਕਨੈੱਟ (ਲਾਈਵ)",
        "source_local": "ਸਥਾਨਕ ਡਾਟਾਬੇਸ",
        "unit_suffix": "ਪ੍ਰਤੀ ਕੁਇੰਟਲ",
    },
    "as": {
        "title": "📊 {commodity} বজাৰ দৰ",
        "market": "বজাৰ",
        "modal": "গড় দৰ",
        "min": "সৰ্বনিম্ন",
        "max": "সৰ্বোচ্চ",
        "date": "তাৰিখ",
        "source_live": "এগমাৰ্কনেট (লাইভ)",
        "source_local": "স্থানীয় ডাটাবেছ",
        "unit_suffix": "প্ৰতি কুইন্টল",
    },
    "ur": {
        "title": "📊 {commodity} منڈی کے بھاؤ",
        "market": "منڈی",
        "modal": "اوسط بھاؤ",
        "min": "کم از کم",
        "max": "زیادہ سے زیادہ",
        "date": "تاریخ",
        "source_live": "ایگمارک نیٹ (لائیو)",
        "source_local": "مقامی ڈیٹا بیس",
        "unit_suffix": "فی کوئنٹل",
    },
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
        commodity: Optional[str] = None,
        district: Optional[str] = None,
        state: Optional[str] = None,
        is_today_requested: bool = False,
        explicit_location: Optional[str] = None,
        raw_commodity: Optional[str] = None,
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

        If explicit_location is provided, strict filtering is enforced:
        - Only prices for that requested location are returned.
        - Never silently falls back to state or national default markets.
        """
        is_live = False
        source_note = ""
        today_ist = get_current_ist_date()

        if district:
            district = normalize_district_name(district)
        state = infer_state_from_district(district, state)

        logger.info(
            f"[MARKET SERVICE] Query start -> commodity='{commodity}', district='{district}', "
            f"state='{state}', is_today_requested={is_today_requested}, explicit_location='{explicit_location}', today_ist={today_ist}"
        )

        # Handle queries where no commodity is specified (location-level multi-commodity query)
        if not commodity:
            target_loc = explicit_location or district
            if not target_loc:
                return MarketPriceQueryResponse(
                    commodity=None,
                    district=district,
                    state=state,
                    results=[],
                    data_available=False,
                    data_freshness_hours=None,
                    source_note="No commodity or location provided.",
                    is_live=False,
                    is_today_requested=is_today_requested,
                    explicit_location=explicit_location,
                    raw_commodity=raw_commodity,
                )

            # Query local DB for this location across all commodities
            db_records = await self.repository.get_prices_by_location(
                location=target_loc,
                state=state,
                strict_location=bool(explicit_location),
            )

            if explicit_location and db_records:
                loc_candidates = {explicit_location.lower()}
                if district:
                    loc_candidates.add(district.lower())
                matching_multi = [
                    r for r in db_records
                    if any(
                        cand in getattr(r, "district", "").lower()
                        or cand in getattr(r, "market_name", "").lower()
                        for cand in loc_candidates
                    )
                ]
                exact_multi = [
                    r for r in matching_multi
                    if explicit_location.lower() in getattr(r, "district", "").lower()
                    or explicit_location.lower() in getattr(r, "market_name", "").lower()
                ]
                db_records = exact_multi if exact_multi else matching_multi

            if not db_records:
                logger.info(
                    f"[MARKET SERVICE] No price data found in DB for location='{target_loc}'"
                )
                return MarketPriceQueryResponse(
                    commodity=None,
                    district=district,
                    state=state,
                    results=[],
                    data_available=False,
                    data_freshness_hours=None,
                    source_note=f"No price data available for {target_loc}.",
                    is_live=False,
                    is_today_requested=is_today_requested,
                    explicit_location=explicit_location,
                    raw_commodity=raw_commodity,
                )

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

            results = [MarketPriceResponse.model_validate(r) for r in db_records]

            return MarketPriceQueryResponse(
                commodity=None,
                district=district,
                state=state,
                results=results,
                data_available=True,
                data_freshness_hours=freshness_hours,
                source_note=f"Local database (data is ~{freshness_hours}h old)",
                is_live=False,
                is_today_requested=is_today_requested,
                explicit_location=explicit_location,
                raw_commodity=raw_commodity,
            )

        # Step 1: Try live API with canonical district (e.g. Jagtial for Korutla)
        api_district = normalize_district_name(district) if district else district
        api_records = await self.client.fetch_prices(
            commodity=commodity,
            state=state,
            district=api_district,
            is_today_requested=is_today_requested,
        )
        if api_records is None:
            api_records = []

        # If explicit_location is provided, filter live API records to ensure they match requested location
        if explicit_location and api_records:
            norm_explicit = normalize_district_name(explicit_location)
            loc_candidates = {explicit_location.lower()}
            if norm_explicit:
                loc_candidates.add(norm_explicit.lower())
            if district:
                loc_candidates.add(district.lower())
            matching_api = [
                r for r in api_records
                if any(
                    cand in r.get("district", "").lower() or cand in r.get("market", "").lower()
                    for cand in loc_candidates
                )
            ]
            exact_api = [
                r for r in matching_api
                if explicit_location.lower() in r.get("district", "").lower()
                or explicit_location.lower() in r.get("market", "").lower()
            ]
            api_records = exact_api if exact_api else matching_api

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
                explicit_location=explicit_location,
                raw_commodity=raw_commodity,
            )

        # Step 2: Query local DB (for latest records or when API didn't return today's data)
        db_records = await self.repository.get_prices_by_commodity(
            commodity=commodity,
            district=district,
            state=state,
        )

        # If explicit_location is provided, enforce strict location matching:
        # Never silently accept state-wide fallback records (e.g. Warangal) for a different requested village/town.
        if explicit_location and db_records:
            norm_explicit = normalize_district_name(explicit_location)
            loc_candidates = {explicit_location.lower()}
            if norm_explicit:
                loc_candidates.add(norm_explicit.lower())
            if district:
                loc_candidates.add(district.lower())
            matching_db = [
                r for r in db_records
                if any(
                    cand in getattr(r, "district", "").lower()
                    or cand in getattr(r, "market_name", "").lower()
                    for cand in loc_candidates
                )
            ]
            exact_db = [
                r for r in matching_db
                if explicit_location.lower() in getattr(r, "district", "").lower()
                or explicit_location.lower() in getattr(r, "market_name", "").lower()
            ]
            db_records = exact_db if exact_db else matching_db

        if not db_records:
            logger.info(
                f"[MARKET SERVICE] No price data found in DB or API for '{commodity}' "
                f"(district={district}, state={state}, explicit_location={explicit_location})"
            )
            return MarketPriceQueryResponse(
                commodity=commodity,
                district=district,
                state=state,
                results=[],
                data_available=False,
                data_freshness_hours=None,
                source_note=f"No price data available for {explicit_location or district or commodity}.",
                is_live=False,
                is_today_requested=is_today_requested,
                explicit_location=explicit_location,
                raw_commodity=raw_commodity,
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
            explicit_location=explicit_location,
            raw_commodity=raw_commodity,
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
        labels = get_market_labels(language)
        commodity = query_response.commodity

        commodity_display = commodity or ""
        if commodity:
            if language == "te":
                for kw, canon in COMMODITY_MAP.items():
                    if canon.lower() == commodity.lower() and any(ord(c) > 127 for c in kw):
                        commodity_display = kw
                        break
            elif language in ["hi", "ta", "kn", "mr", "bn", "gu", "or", "pa", "as", "ur"]:
                for kw, canon in COMMODITY_MAP.items():
                    if canon.lower() == commodity.lower() and any(ord(c) > 127 for c in kw):
                        commodity_display = kw
                        break

        if not query_response.data_available or not query_response.results:
            req_loc = getattr(query_response, "explicit_location", None)
            if not query_response.commodity:
                no_crop_loc = req_loc or query_response.district
                if no_crop_loc:
                    if language == "te":
                        tel_loc = _TELUGU_DISTRICT_MAP.get(no_crop_loc, no_crop_loc)
                        if tel_loc == no_crop_loc:
                            for tel, eng in TELUGU_TO_ENGLISH_PLACES.items():
                                if eng.lower() == no_crop_loc.lower():
                                    tel_loc = tel
                                    break
                        return f"📍 {tel_loc} మార్కెట్ ధరల సమాచారం ప్రస్తుతం అందుబాటులో లేదు. సమీప మార్కెట్లలో ధరలు చూడటానికి దయచేసి నిర్దిష్ట పంట పేరును (ఉదా: పత్తి, వరి, మిర్చి) తెలపండి."
                    elif language == "hi":
                        return f"📍 {no_crop_loc} के लिए वर्तमान में मंडी भाव उपलब्ध नहीं है। नजदीकी मंडियों के भाव देखने के लिए कृपया फसल का नाम (उदा: कपास, धान, मिर्च) बताएं।"
                    else:
                        return f"📍 Currently market-price data is not available for {no_crop_loc}. Please specify which crop you are looking for (e.g., Cotton, Paddy, Chilli) to check prices in nearby markets."
                else:
                    if language == "te":
                        return "📊 మార్కెట్ ధరలు తెలుసుకోవడానికి దయచేసి పంట మరియు మార్కెట్ పేరు తెలపండి (ఉదా: 'వరంగల్‌లో పత్తి ధర' లేదా 'సూర్యాపేటలో వరి ధర')."
                    elif language == "hi":
                        return "📊 मंडी भाव जानने के लिए कृपया फसल और मंडी का नाम बताएं (जैसे: 'वारंगल में कपास का भाव')।"
                    else:
                        return "📊 Please specify the crop and market location to check prices (e.g., 'Cotton price in Warangal' or 'Paddy price in Suryapet')."

            comm_name = getattr(query_response, "raw_commodity", None) or commodity_display
            if comm_name and isinstance(comm_name, str) and comm_name.isascii():
                comm_name = comm_name.capitalize()
            if req_loc:
                if language == "te":
                    return f"📍 {req_loc} కోసం ప్రస్తుతం {comm_name} market-price data అందుబాటులో లేదు."
                elif language == "hi":
                    return f"📍 {req_loc} के लिए वर्तमान में {comm_name} का मंडी भाव डेटा उपलब्ध नहीं है।"
                else:
                    return f"📍 Currently market-price data for {comm_name} is not available for {req_loc}."
            return labels["no_data"].format(commodity=commodity_display)

        if not query_response.commodity:
            loc_name = getattr(query_response, "explicit_location", None) or query_response.district or "Market"
            source_str = labels["source_live"] if query_response.is_live else labels["source_local"]

            seen_comms = set()
            deduplicated = []
            for r in query_response.results:
                ck = r.commodity.lower()
                if ck not in seen_comms:
                    seen_comms.add(ck)
                    deduplicated.append(r)
                if len(deduplicated) >= 6:
                    break

            if language == "te":
                tel_loc = _TELUGU_DISTRICT_MAP.get(loc_name, loc_name)
                if tel_loc == loc_name:
                    for tel, eng in TELUGU_TO_ENGLISH_PLACES.items():
                        if eng.lower() == loc_name.lower():
                            tel_loc = tel
                            break
                market_title = deduplicated[0].market_name
                lines = [
                    f"📊 {tel_loc} మార్కెట్ ధరలు\n"
                    f"మార్కెట్: {market_title}, {deduplicated[0].state}\n"
                ]
                for r in deduplicated:
                    c_name = r.commodity_telugu or r.commodity
                    lines.append(
                        f"• {c_name}: ₹{r.modal_price:,.0f}/{labels['unit_suffix']} "
                        f"(కనిష్టం ₹{r.min_price:,.0f} | గరిష్టం ₹{r.max_price:,.0f})"
                    )
                newest_date = max(r.price_date for r in deduplicated)
                lines.append(f"\n{labels['date']}: {newest_date.strftime('%d %b %Y')}")
                lines.append(f"📡 {source_str}")
                return "\n".join(lines)
            else:
                market_title = deduplicated[0].market_name
                lines = [
                    f"📊 {loc_name} Mandi Prices\n"
                    f"Market: {market_title}, {deduplicated[0].state}\n"
                ]
                for r in deduplicated:
                    lines.append(
                        f"• {r.commodity}: Modal ₹{r.modal_price:,.0f}/{labels['unit_suffix']} "
                        f"(Min ₹{r.min_price:,.0f} | Max ₹{r.max_price:,.0f})"
                    )
                newest_date = max(r.price_date for r in deduplicated)
                lines.append(f"\n{labels['date']}: {newest_date.strftime('%d %b %Y')}")
                lines.append(f"📡 {source_str}")
                return "\n".join(lines)

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

            dist_str = ""
            if query_response.district:
                if language == "te":
                    tel_dist = _TELUGU_DISTRICT_MAP.get(query_response.district, query_response.district)
                    dist_str = f"{tel_dist} "
                else:
                    dist_str = f"{query_response.district} "
            lines = [
                labels["today_unavailable"].format(district=dist_str, commodity=commodity_display),
                labels["last_available"].format(date=latest_date_str),
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

    # Step 1: Detect price intent (strictly checked via AIDecisionEngine)
    from src.ai.decision_engine import AIDecisionEngine, FarmerIntent
    intents = AIDecisionEngine.detect_all_intents(query_text)
    has_price_intent = FarmerIntent.MARKET_PRICE in intents

    if not has_price_intent:
        has_price_intent = any(kw in query_lower for kw in PRICE_INTENT_KEYWORDS_EN)
        if not has_price_intent:
            has_price_intent = any(kw in query_text for kw in PRICE_INTENT_KEYWORDS_TE)
        if not has_price_intent:
            has_price_intent = any(kw in query_text for kw in PRICE_INTENT_KEYWORDS_MULTILINGUAL)

    logger.info(
        f"[MARKET ENRICH] Diagnostic check -> query='{query_text}' | "
        f"has_price_intent={has_price_intent}"
    )

    if not has_price_intent:
        return ai_response

    # Step 2: Identify commodity
    matched_commodity = None
    raw_commodity_word = None
    import re
    sorted_keywords = sorted(COMMODITY_MAP.items(), key=lambda x: len(x[0]), reverse=True)
    for kw, canonical in sorted_keywords:
        kw_lower = kw.lower()
        if kw_lower.isascii() and kw_lower.isalnum():
            if re.search(rf"\b{re.escape(kw_lower)}\b", query_lower):
                matched_commodity = canonical
                raw_commodity_word = kw
                break
        else:
            if kw in query_text or kw_lower in query_lower:
                matched_commodity = canonical
                raw_commodity_word = kw
                break

    # Fallback to extract_crop_from_text if not matched directly in COMMODITY_MAP
    if not matched_commodity:
        try:
            from src.rag.service import extract_crop_from_text
            matched_commodity = extract_crop_from_text(query_text)
            if matched_commodity:
                raw_commodity_word = matched_commodity
        except Exception:
            matched_commodity = None

    # Step 3: Location resolution with strict priority:
    # 1. Explicit location mentioned in the CURRENT USER QUERY
    # 2. Current GPS location
    # 3. Saved farmer location (profile or memory)
    # 4. Existing generic/default behavior
    explicit_location = extract_market_explicit_location(query_text)
    district = None
    state = None
    is_explicit = bool(explicit_location)

    farmer_lang = getattr(farmer, "preferred_language", "te") or "te"
    from src.language.detector import detect_language
    language = detect_language(query_text, fallback=farmer_lang)

    if explicit_location:
        district = explicit_location
        logger.info(f"[MARKET ENRICH] Priority 1: Explicit location in query -> '{explicit_location}'")
    else:
        # Priority 2: GPS coordinates from FarmerMemory or farmer object
        lat = None
        lon = None
        if hasattr(farmer, "gps_coordinates") and isinstance(farmer.gps_coordinates, dict):
            lat = farmer.gps_coordinates.get("latitude")
            lon = farmer.gps_coordinates.get("longitude")
        else:
            farmer_memory = None
            try:
                from sqlalchemy.inspection import inspect as sa_inspect
                insp = sa_inspect(farmer)
                if insp is not None and hasattr(insp, "unloaded") and "memory" in insp.unloaded:
                    farmer_memory = None
                else:
                    try:
                        farmer_memory = getattr(farmer, "memory", None)
                    except Exception:
                        farmer_memory = None
            except Exception:
                try:
                    farmer_memory = getattr(farmer, "memory", None) if not hasattr(farmer, "__table__") else None
                except Exception:
                    farmer_memory = None

            if farmer_memory and hasattr(farmer_memory, "gps_coordinates"):
                gps = getattr(farmer_memory, "gps_coordinates", None)
                if isinstance(gps, dict):
                    lat = gps.get("latitude")
                    lon = gps.get("longitude")

        # If not present on farmer object in-memory, query FarmerMemory from DB
        if (lat is None or lon is None) and farmer and hasattr(farmer, "id"):
            try:
                import inspect
                from sqlalchemy import select
                from src.memory.models import FarmerMemory
                mem_res = await db.execute(
                    select(FarmerMemory).where(FarmerMemory.farmer_id == farmer.id)
                )
                if inspect.iscoroutine(mem_res):
                    mem_res = await mem_res
                memory = mem_res.scalar_one_or_none() if hasattr(mem_res, "scalar_one_or_none") else None
                if inspect.iscoroutine(memory):
                    memory = await memory
                gps_coords = getattr(memory, "gps_coordinates", None)
                if isinstance(gps_coords, dict):
                    lat = gps_coords.get("latitude")
                    lon = gps_coords.get("longitude")
            except Exception as mem_err:
                logger.debug(f"[MARKET ENRICH] FarmerMemory GPS check skipped: {mem_err}")

        if lat is not None and lon is not None:
            try:
                lat_f = float(lat or 0.0)
                lon_f = float(lon or 0.0)
                if not (lat_f == 0.0 and lon_f == 0.0):
                    gps_dist = resolve_gps_to_nearest_district(lat_f, lon_f)
                    if gps_dist:
                        district = gps_dist
                        logger.info(f"[MARKET ENRICH] Priority 2: GPS coordinates ({lat_f}, {lon_f}) -> nearest district '{district}'")
            except (ValueError, TypeError):
                pass

    # Priority 3: Saved farmer profile/memory (load profile for state/crop and district if no GPS)
    try:
        import inspect
        from sqlalchemy.inspection import inspect as sa_inspect
        from sqlalchemy import select
        from src.core.models import FarmerProfile

        profile = None
        try:
            insp = sa_inspect(farmer)
            if insp is not None and hasattr(insp, "unloaded") and "profile" in insp.unloaded:
                profile = None
            else:
                try:
                    profile = getattr(farmer, "profile", None)
                except Exception:
                    profile = None
        except Exception:
            try:
                profile = getattr(farmer, "profile", None) if not hasattr(farmer, "__table__") else None
            except Exception:
                profile = None

        if profile is None and farmer and hasattr(farmer, "id"):
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
                logger.info(f"[MARKET ENRICH] Priority 3: Saved farmer profile district -> '{district}'")
            if isinstance(getattr(profile, "state", None), str) and profile.state:
                state = profile.state.strip()
            if not matched_commodity and isinstance(getattr(profile, "current_crop", None), str) and profile.current_crop:
                matched_commodity = profile.current_crop.strip()
                if not raw_commodity_word:
                    raw_commodity_word = matched_commodity
    except Exception as exc:
        logger.warning(f"[MARKET ENRICH] Could not load farmer profile: {exc}")

    if district:
        district = normalize_district_name(district)
    state = infer_state_from_district(district, state)

    today_requested = is_today_price_query(query_text)

    logger.info(
        f"[MARKET ENRICH] Parameters -> detected_crop='{matched_commodity}', "
        f"detected_location_or_mandi='{district}', explicit_location='{explicit_location}', state='{state}', "
        f"is_today_requested={today_requested}, language='{language}'"
    )

    if not matched_commodity:
        logger.info(
            f"[MARKET ENRICH] Price intent detected without specific commodity. "
            f"Querying location-level prices for location='{explicit_location or district}'"
        )

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
            explicit_location=explicit_location,
            raw_commodity=raw_commodity_word,
        )

        logger.info(
            f"[MARKET ENRICH] Query response -> data_available={query_response.data_available}, "
            f"record_count={len(query_response.results)}, is_live={query_response.is_live}, "
            f"is_today_requested={today_requested}, explicit_location='{explicit_location}'"
        )

        price_block = svc.format_whatsapp_reply(
            query_response, language=language, is_today_query=today_requested
        )

        if query_response.data_available:
            logger.info(
                f"[MARKET ENRICH] Returning {len(query_response.results)} price records "
                f"for '{matched_commodity or explicit_location or district}'"
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
            # Data unavailable
            if is_explicit or not matched_commodity:
                # Explicit location requested or price query without crop:
                # Return authoritative guidance/unavailable notice
                logger.info(
                    f"[MARKET ENRICH] Location '{explicit_location or district}' has no data for '{matched_commodity}' — "
                    "returning clear location unavailable / crop specification guidance."
                )
                if _is_pure_price_query(query_text):
                    return price_block
                else:
                    clean_ai = _clean_ai_response_for_market_enrichment(ai_response)
                    return (clean_ai + "\n\n" + price_block) if clean_ai else price_block
            elif _is_pure_price_query(query_text):
                return price_block
            else:
                # Data unavailable on multi-intent query with no explicit location — Gemini advisory still goes
                logger.info(
                    f"[MARKET ENRICH] No price data for '{matched_commodity}' — "
                    "returning original AI response unchanged."
                )
                return ai_response

    except Exception as exc:
        logger.warning(f"[MARKET ENRICH] Price enrichment failed: {exc}")
        return ai_response
