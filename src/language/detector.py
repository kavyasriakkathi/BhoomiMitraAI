"""
BhoomiMitra AI — Ultra-Fast Multi-Tier Language Detector

Provides accurate, deterministic, zero-LLM-overhead language detection for 13 Indian languages:
1. Telugu (te)
2. Hindi (hi)
3. English (en)
4. Tamil (ta)
5. Kannada (kn)
6. Malayalam (ml)
7. Marathi (mr)
8. Bengali (bn)
9. Gujarati (gu)
10. Odia (or)
11. Punjabi (pa)
12. Assamese (as)
13. Urdu (ur)

Supports:
- Direct Unicode script recognition
- Script disambiguation (Devanagari: Hindi vs Marathi; Eastern Nagari: Bengali vs Assamese)
- Romanized/Transliterated Indian language classification (Tanglish, Hinglish, Kanglish, Tamlish, etc.)
- Mixed-language tolerance
- Graceful fallback for uncertain inputs
"""

import re
import unicodedata
from typing import Tuple, Optional, Dict, Set
from src.language.languages import (
    SUPPORTED_LANGUAGES,
    DEFAULT_LANGUAGE,
    FALLBACK_LANGUAGE,
    is_supported_language,
    normalize_language_code,
)


# -----------------------------------------------------------------------------
# Unicode Script Character Counters
# -----------------------------------------------------------------------------

def _count_script_chars(text: str) -> Dict[str, int]:
    """Count characters belonging to distinct Indian and Arabic/Latin scripts."""
    counts = {
        "telugu": 0,
        "tamil": 0,
        "kannada": 0,
        "malayalam": 0,
        "gujarati": 0,
        "odia": 0,
        "gurmukhi": 0,
        "arabic": 0,
        "devanagari": 0,
        "bengali_assamese": 0,
        "latin": 0,
    }
    for ch in text:
        cp = ord(ch)
        if 0x0C00 <= cp <= 0x0C7F:
            counts["telugu"] += 1
        elif 0x0B80 <= cp <= 0x0BFF:
            counts["tamil"] += 1
        elif 0x0C80 <= cp <= 0x0CFF:
            counts["kannada"] += 1
        elif 0x0D00 <= cp <= 0x0D7F:
            counts["malayalam"] += 1
        elif 0x0A80 <= cp <= 0x0AFF:
            counts["gujarati"] += 1
        elif 0x0B00 <= cp <= 0x0B7F:
            counts["odia"] += 1
        elif 0x0A00 <= cp <= 0x0A7F:
            counts["gurmukhi"] += 1
        elif (0x0600 <= cp <= 0x06FF) or (0x0750 <= cp <= 0x077F) or (0xFB50 <= cp <= 0xFDFF) or (0xFE70 <= cp <= 0xFEFF):
            counts["arabic"] += 1
        elif 0x0900 <= cp <= 0x097F:
            counts["devanagari"] += 1
        elif 0x0980 <= cp <= 0x09FF:
            counts["bengali_assamese"] += 1
        elif (0x0041 <= cp <= 0x005A) or (0x0061 <= cp <= 0x007A):
            counts["latin"] += 1
    return counts


# -----------------------------------------------------------------------------
# Script Disambiguation Helpers
# -----------------------------------------------------------------------------

_MARATHI_DISTINCT_WORDS = {
    "आहे", "नाही", "शेतकरी", "पाहिजे", "कसा", "कशी", "कसे", "खत", "पाऊस", "बाजारभाव",
    "पिकांवर", "औषध", "कापूस", "सोयाबीन", "तणनाशक", "झाले", "द्यावे", "करावे", "आहेत",
    "काय", "मिळेल", "शेती", "योजना", "फवारणी", "पाणी", "सांगा", "माहिती", "भाव",
    "कीड", "रोग", "किती", "लागवड", "पिकाचे", "खते", "अनुदान", "दर", "नुकसान",
}

_HINDI_DISTINCT_WORDS = {
    "है", "हैं", "नहीं", "किसान", "चाहिए", "कितना", "कितने", "खाद", "बारिश", "दवा",
    "कीटनाशक", "फसल", "करना", "करे", "होगा", "बताएं", "जानकारी", "भाव", "कीड़ा",
    "रोग", "पानी", "क्या", "कैसे", "योजना", "मंडी", "छिड़काव", "गेहूं", "धान", "कपास",
    "टमाटर", "मिर्च", "बताओ", "कीजिए", "सकते", "दाम", "लागत", "सब्सिडी",
}

_ASSAMESE_DISTINCT_CHARS = {"\u09F0", "\u09F1"}  # 'ৰ', 'ৱ'
_ASSAMESE_DISTINCT_WORDS = {
    "অসম", "কৃষক", "বজাৰ", "কীটনাশক", "সাৰ", "পানী", "ধান", "ৰোগ", "লাগে", "কিমান",
    "কেনেকৈ", "কৰিব", "দৰকাৰ", "খেতি", "বৰষুণ", "দৰ", "কপাহ", "ঔষধ", "উপায়",
}

_BENGALI_DISTINCT_WORDS = {
    "আমার", "কী", "কেমন", "দর", "পোকা", "সার", "পানি", "কীটনাশক", "কত", "ফসল",
    "চাষ", "বৃষ্টি", "ঔষধ", "কৃষক", "বাজার", "ধান", "টমেটো", "ইউরিয়া", "রোগ",
    "উপায়", "হবে", "করব", "দিতে", "চাই", "দাম", "ক্ষতি",
}


def _disambiguate_devanagari(text: str) -> str:
    """Disambiguate Devanagari script between Hindi ('hi') and Marathi ('mr')."""
    words = set(re.findall(r"[\u0900-\u097F]+", text))
    mr_score = len(words.intersection(_MARATHI_DISTINCT_WORDS))
    hi_score = len(words.intersection(_HINDI_DISTINCT_WORDS))
    # Check for Marathi suffixes/inflections (e.g. -चे, -च्या, -तील, -मध्ये, -साठी)
    for w in words:
        if any(w.endswith(sfx) for sfx in ["साठी", "मध्ये", "वरील", "नुसार", "च्या", "चे", "तील", "तात"]):
            mr_score += 1

    if mr_score > hi_score:
        return "mr"
    return "hi"


def _disambiguate_bengali_assamese(text: str) -> str:
    """Disambiguate Eastern Nagari script between Bengali ('bn') and Assamese ('as')."""
    # Check for unique Assamese Unicode characters 'ৰ' (\u09F0), 'ৱ' (\u09F1)
    if any(ch in text for ch in _ASSAMESE_DISTINCT_CHARS):
        return "as"

    words = set(re.findall(r"[\u0980-\u09FF]+", text))
    as_score = len(words.intersection(_ASSAMESE_DISTINCT_WORDS))
    bn_score = len(words.intersection(_BENGALI_DISTINCT_WORDS))

    if as_score > bn_score:
        return "as"
    return "bn"


# -----------------------------------------------------------------------------
# Romanized / Transliterated Indian Language Lexicons
# -----------------------------------------------------------------------------

_ROMANIZED_LEXICONS: Dict[str, Set[str]] = {
    "te": {
        # Tanglish (Telugu)
        "vari", "eruvu", "eruvulu", "patti", "neeru", "entha", "entho", "eppudu",
        "pettali", "veyali", "koyali", "purugu", "purugulu", "tegulu", "aakulu",
        "vadali", "ela", "undi", "cheyandi", "dhara", "mandi", "panta", "mirapa",
        "tamata", "annam", "rythu", "polam", "kotha", "vithanalu", "natlu",
        "pasupuga", "mudatha", "thadi", "cheda", "daggara", "ekkada", "dorukuthundi",
        "konali", "ammukovali", "pathakam", "varsham", "paduthunda", "padtundha",
        "vasthunda", "eeroju", "repu", "nenu", "maaku", "meeru", "sahayam",
    },
    "hi": {
        # Hinglish (Hindi)
        "kapas", "pani", "paani", "kisan", "kitna", "kitni", "kitne", "chahiye", "kya",
        "kare", "khat", "khad", "khaad", "keeda", "keede", "rog", "dawa", "dawai",
        "kheton", "khet", "fasal", "daalna", "lagana", "kaise", "bhaav", "bhav",
        "gehu", "dhan", "tamatar", "mirch", "bima", "yojana", "kheti", "barish",
        "kab", "kaha", "kahan", "milega", "sarkar", "sarkari",
        "upchar", "chhidkaw", "beej", "batao", "bataye",
    },
    "kn": {
        # Kanglish (Kannada)
        "nellu", "neeru", "eshtu", "bele", "gotta", "rogha", "aushadha", "haki",
        "bajaru", "beku", "yava", "kodi", "gobbara", "krishi", "adike", "tottu",
        "bettada", "hege", "yavaga", "madabeku", "marata", "beleya", "kaayi",
        "tumba", "illava", "yavudu", "kheduta", "raitha", "hola",
    },
    "ta": {
        # Tamlish (Tamil)
        "nellu", "thanni", "ennathu", "marunthu", "poochi", "vilai", "eppadi",
        "podanum", "payir", "vivasayi", "uram", "mazhai", "eppozhuthu", "enga",
        "kedaikkum", "vaanga", "vilpadi", "nalla", "illai", "vanakkam", "thittam",
        "vivasayam", "vidhai", "kattai", "aruvadai",
    },
    "ml": {
        # Malenglish (Malayalam)
        "vellam", "valam", "kedu", "marunnu", "vilavu", "enthannu", "vilayil",
        "mazha", "karshakan", "nattil", "eppol", "cheyyannam", "krishikaran",
        "vithu", "koythu", "choodu", "thottam", "puzhu", "keedangal",
    },
    "mr": {
        # Marathi in Latin
        "kapus", "kiti", "pahije", "khat", "sheti", "aushadh", "rogh", "tannashak",
        "paus", "kasa", "ahe", "shatkari", "kay", "sang", "lagwad", "bajarbhav",
        "favarni", "favarani", "kadhi", "kuthe", "bhetel", "pika", "shashan",
    },
    "bn": {
        # Bengali in Latin
        "dhan", "jol", "koto", "sar", "poka", "osudh", "lagbe", "kamon", "krishak",
        "bristi", "chas", "foshol", "ki", "kivabe", "dite", "hobe",
        "dam", "bij", "poriman",
    },
    "gu": {
        # Gujarati in Latin
        "ketlu", "khatar", "khedut", "varsad", "joiye", "kem", "kyare",
        "aavshe", "bhav", "malashe", "biyarano", "vavetar", "kapi",
    },
    "or": {
        # Odia in Latin
        "dhana", "kete", "sara", "oushadha", "dara", "chasa", "barsha", "krushaka",
        "fasala", "kemiti", "milba", "kariba", "biha",
    },
    "pa": {
        # Punjabi in Latin
        "kanak", "kinna", "khad", "dawai", "barsaat", "jhotta", "kiddan", "kadon",
        "chahida", "daso", "kareye",
    },
    "as": {
        # Assamese in Latin
        "kiman", "puk", "boroxun", "krixok", "kenekoi", "lagibo", "karibo",
        "khetir",
    },
    "ur": {
        # Urdu in Latin
        "keere", "zaroorat", "barish", "tariqa", "kijiye", "bataiye",
        "malumat", "nuskha", "ilaj",
    },
}

_ENGLISH_WORDS = {
    "what", "how", "when", "where", "which", "why", "who", "is", "are", "the",
    "for", "to", "in", "on", "with", "and", "can", "should", "i", "my", "crop",
    "fertilizer", "fertilizers", "pesticide", "pesticides", "disease", "pest", "pests",
    "water", "irrigation", "soil", "price", "prices", "mandi", "rate", "rates", "weather", "rain",
    "forecast", "scheme", "schemes", "subsidy", "shop", "shops", "store", "stores", "buy",
    "sell", "cotton", "paddy", "rice", "tomato", "chilli", "maize", "spray", "urea", "dap",
    "dosage", "acre", "yield", "hello", "hi", "help", "please", "today", "tomorrow",
    "give", "tell", "details", "information", "about", "near", "nearby",
}


def _classify_romanized_text(text: str) -> Tuple[Optional[str], float]:
    """
    Score romanized text against Indian languages & English.
    Returns (detected_code, confidence).
    """
    tokens = [re.sub(r"[^\w]", "", t.lower()) for t in text.split()]
    tokens = [t for t in tokens if t and not t.isdigit()]
    if not tokens:
        return None, 0.0

    scores: Dict[str, int] = {lang: 0 for lang in _ROMANIZED_LEXICONS}
    en_score = 0

    for token in tokens:
        if token in _ENGLISH_WORDS:
            en_score += 1
        for lang, lexicon in _ROMANIZED_LEXICONS.items():
            if token in lexicon:
                scores[lang] += 2

    # Check multi-word phrase patterns
    lowered = f" {text.lower()} "
    if " em fertilizer " in lowered or " em vadali " in lowered or " ela vadali " in lowered or " rate entha " in lowered or " ela undi " in lowered or " vari ki " in lowered:
        scores["te"] += 3
    if " kitna pani " in lowered or " kya kare " in lowered or " kitna khad " in lowered or " mandi bhav " in lowered or " ko kitna " in lowered:
        scores["hi"] += 3
    if " ge neeru " in lowered or " neeru eshtu " in lowered or " eshtu beku " in lowered or " crop ge " in lowered:
        scores["kn"] += 3
    if " ku thanni " in lowered or " thanni eppadi " in lowered or " uram podanum " in lowered:
        scores["ta"] += 3

    # Find highest Indian language score
    best_lang, best_score = max(scores.items(), key=lambda x: x[1])
    total_tokens = len(tokens)

    # If English matches dominate or no Indian words matched
    if en_score >= 1 and en_score >= best_score:
        conf = min(0.95, 0.6 + (en_score / total_tokens))
        return "en", conf

    # If strong Indian romanized signal exists
    if best_score >= 2:
        conf = min(0.95, 0.5 + (best_score / (total_tokens * 2)))
        return best_lang, conf

    if en_score >= 1:
        return "en", 0.6

    return None, 0.0


# -----------------------------------------------------------------------------
# Main Detection Entry Points
# -----------------------------------------------------------------------------

def detect_language_with_confidence(
    text: Optional[str],
    fallback: str = FALLBACK_LANGUAGE,
) -> Tuple[str, float, str]:
    """
    Detect language with confidence score and detection method.

    Returns:
        (language_code, confidence, method_used)
        where method_used is 'script', 'romanized', 'english', or 'fallback'.
    """
    if not text or not text.strip():
        return normalize_language_code(fallback), 0.0, "fallback"

    cleaned = text.strip()
    script_counts = _count_script_chars(cleaned)
    total_script_chars = sum(script_counts.values())

    if total_script_chars == 0:
        return normalize_language_code(fallback), 0.0, "fallback"

    # 1. Check Native Indic / Arabic Scripts
    if script_counts["telugu"] > 0:
        return "te", 0.99, "script"
    if script_counts["tamil"] > 0:
        return "ta", 0.99, "script"
    if script_counts["kannada"] > 0:
        return "kn", 0.99, "script"
    if script_counts["malayalam"] > 0:
        return "ml", 0.99, "script"
    if script_counts["gujarati"] > 0:
        return "gu", 0.99, "script"
    if script_counts["odia"] > 0:
        return "or", 0.99, "script"
    if script_counts["gurmukhi"] > 0:
        return "pa", 0.99, "script"
    if script_counts["arabic"] > 0:
        return "ur", 0.99, "script"
    if script_counts["devanagari"] > 0:
        lang = _disambiguate_devanagari(cleaned)
        return lang, 0.95, "script"
    if script_counts["bengali_assamese"] > 0:
        lang = _disambiguate_bengali_assamese(cleaned)
        return lang, 0.95, "script"

    # 2. Check Romanized / Latin script text
    if script_counts["latin"] > 0:
        detected_roman, conf = _classify_romanized_text(cleaned)
        if detected_roman:
            return detected_roman, conf, "romanized"

    # 3. Graceful Fallback
    safe_fallback = normalize_language_code(fallback)
    return safe_fallback, 0.3, "fallback"


def detect_language(
    text: Optional[str],
    fallback: str = FALLBACK_LANGUAGE,
) -> str:
    """
    Fast, deterministic language detector for BhoomiMitra AI.
    Returns 2-letter ISO language code (e.g. 'te', 'hi', 'en', 'ta', 'kn', 'ml', 'mr', 'bn', 'gu', 'or', 'pa', 'as', 'ur').
    """
    code, _, _ = detect_language_with_confidence(text, fallback=fallback)
    return code
