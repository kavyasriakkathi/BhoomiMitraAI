"""
Decision Engine Safety Validators for BhoomiMitra AI.
Provides deterministic safety checks to detect unknown pesticides, dosage inquiries,
and missing agricultural context without calling external services.
"""

from typing import List, Optional, Union
import re

from src.decision_engine.models import FarmerInput

# Known example / placeholder / synthetic pesticide tokens
KNOWN_EXAMPLE_PESTICIDES: List[str] = [
    "ABC-999",
    "XYZ-999",
    "TEST-999",
    "FAKE-999",
    "SAMPLE-123",
]

# Regex pattern for synthetic example codes like ABC-999, XYZ-999, etc.
EXAMPLE_PESTICIDE_PATTERN = re.compile(
    r"\b(?:[A-Za-z]{2,5}-\d{2,4}|[A-Za-z]+\s*999)\b",
    re.IGNORECASE,
)

# Dosage keywords and patterns (English)
DOSAGE_KEYWORDS_EN = [
    "dosage",
    "dose",
    "how much",
    "per acre",
    "per-acre",
    "how many ml",
    "how much ml",
    "how much dose",
    "application rate",
    "quantity to spray",
]

# Dosage keywords and phrases (Telugu)
DOSAGE_KEYWORDS_TE = [
    "మోతాదు",        # Dosage
    "ఎంత వేయాలి",    # How much to apply
    "ఎకరానికి ఎంత",   # How much per acre
    "ఎంత మందు",      # How much medicine/chemical
    "ఎంత మోతాదు",    # What dosage
    "ఎకరానికి",      # Per acre
]

# Regex for English dosage terms and unit mentions
DOSAGE_REGEX_EN = re.compile(
    r"\b(?:dosage|dose|how\s+much|per\s+acre|how\s+many\s+ml|application\s+rate)\b|"
    r"\b\d+\s*(?:ml|l|gm|grams?|kg|liters?|litres?)(?:\s*/\s*|\s+per\s+)(?:acre|tank|hectare|plant)\b|"
    r"\b(?:ml|grams?)\b",
    re.IGNORECASE,
)


def detect_unknown_or_example_pesticides(text: str) -> List[str]:
    """
    Detects unknown, placeholder, or synthetic example pesticide names
    (e.g., 'ABC-999', 'XYZ-999') in the provided text.

    Returns a list of detected suspicious pesticide names/tokens.
    """
    if not text or not isinstance(text, str):
        return []

    found: List[str] = []

    # Check known example list
    for item in KNOWN_EXAMPLE_PESTICIDES:
        if re.search(r"\b" + re.escape(item) + r"\b", text, re.IGNORECASE):
            if item not in found:
                found.append(item)

    # Check synthetic patterns like ABC-999, XYZ-999
    matches = EXAMPLE_PESTICIDE_PATTERN.findall(text)
    for m in matches:
        m_clean = m.strip()
        if m_clean and m_clean not in found:
            found.append(m_clean)

    return found


def has_unknown_or_example_pesticides(text: str) -> bool:
    """Returns True if any unknown or example pesticide name is detected."""
    return len(detect_unknown_or_example_pesticides(text)) > 0


def detect_dosage_requests(text: str) -> List[str]:
    """
    Detects dosage-related questions and keywords in both English and Telugu.

    Returns a list of detected dosage keywords or patterns.
    """
    if not text or not isinstance(text, str):
        return []

    detected: List[str] = []

    # Check English keywords
    for kw in DOSAGE_KEYWORDS_EN:
        if re.search(r"\b" + re.escape(kw) + r"\b", text, re.IGNORECASE):
            if kw not in detected:
                detected.append(kw)

    # Check standalone dosage units like 'ml', 'grams'
    for match in DOSAGE_REGEX_EN.finditer(text):
        matched_str = match.group(0).strip().lower()
        if matched_str and matched_str not in [d.lower() for d in detected]:
            detected.append(matched_str)

    # Check Telugu phrases
    for kw in DOSAGE_KEYWORDS_TE:
        if kw in text:
            if kw not in detected:
                detected.append(kw)

    return detected


def is_dosage_request(text: str) -> bool:
    """Returns True if the text asks for or mentions chemical/pesticide dosage."""
    return len(detect_dosage_requests(text)) > 0


def _extract_field(target: Union[FarmerInput, Optional[str]], field_name: str) -> Optional[str]:
    """Helper to extract a string field from FarmerInput or a raw string."""
    if isinstance(target, FarmerInput):
        return getattr(target, field_name, None)
    if isinstance(target, str):
        return target
    return None


def is_crop_missing(target: Union[FarmerInput, Optional[str]]) -> bool:
    """Checks whether crop information is missing, None, or empty."""
    val = _extract_field(target, "crop") if isinstance(target, FarmerInput) else target
    return val is None or not str(val).strip() or str(val).strip().lower() in {"unknown", "none", "null"}


def is_growth_stage_missing(target: Union[FarmerInput, Optional[str]]) -> bool:
    """Checks whether growth stage information is missing, None, or empty."""
    val = _extract_field(target, "growth_stage") if isinstance(target, FarmerInput) else target
    return val is None or not str(val).strip() or str(val).strip().lower() in {"unknown", "none", "null"}


def is_problem_missing(target: Union[FarmerInput, Optional[str]]) -> bool:
    """Checks whether problem description is missing, None, or empty."""
    val = _extract_field(target, "problem") if isinstance(target, FarmerInput) else target
    return val is None or not str(val).strip() or str(val).strip().lower() in {"unknown", "none", "null"}


def get_missing_context(farmer_input: FarmerInput) -> List[str]:
    """
    Identifies which essential agricultural context fields are missing from the FarmerInput.
    Returns a list of missing field names (e.g., ['crop', 'growth_stage', 'problem']).
    """
    missing: List[str] = []
    if is_crop_missing(farmer_input):
        missing.append("crop")
    if is_growth_stage_missing(farmer_input):
        missing.append("growth_stage")
    if is_problem_missing(farmer_input):
        missing.append("problem")
    return missing


# =====================================================================
# STEP 6: POST-GENERATION AI SAFETY VALIDATION
# =====================================================================

# Banned / Restricted pesticides in India
BANNED_PESTICIDES_EN: List[str] = [
    "endosulfan",
    "monocrotophos",
    "paraquat",
    "phorate",
    "methyl parathion",
    "dichlorvos",
    "ddvp",
    "lindane",
    "diazinon",
    "captafol",
    "carbofuran",
    "aldicarb",
    "alachlor",
    "chlordane",
    "ddt",
    "heptachlor",
    "methomyl",
    "phosphamidon",
    "triazophos",
    "sodium cyanide",
    "methyl bromide",
    "benomyl",
    "fenitrothion",
    "nicotine sulfate",
]

BANNED_PESTICIDES_TE: List[str] = [
    "ఎండోసల్ఫాన్",
    "మోనోక్రోటోఫాస్",
    "పారాక్వాట్",
    "ఫోరేట్",
    "మిథైల్ పారాథియాన్",
    "డైక్లోరోవాస్",
    "లిండేన్",
    "డయాజినాన్",
    "కార్బోఫ్యూరాన్",
    "ఆల్డికార్బ్",
    "డిడిటి",
]

# Medical advice keywords (English & Telugu)
MEDICAL_KEYWORDS_EN = [
    "paracetamol", "ibuprofen", "amoxicillin", "azithromycin", "ciprofloxacin",
    "cetirizine", "metformin", "insulin", "omeprazole", "aspirin", "cough syrup",
    "blood pressure", "hypertension", "human diabetes", "human fever", "headache remedy",
    "pregnancy advice", "medical doctor prescription", "mg tablet", "take this tablet"
]

MEDICAL_KEYWORDS_TE = [
    "జ్వరం వస్తే", "తలనొప్పి", "రక్తపోటు", "షుగర్ వ్యాధి", "మధుమేహం",
    "పారాసిటమాల్", "డాక్టర్ ప్రిస్క్రిప్షన్", "మానవ ఆరోగ్యం", "మాత్రలు వేసుకోండి"
]

# Non-agricultural / Out-of-domain patterns
NON_AGRICULTURAL_PATTERNS_EN = [
    r"\b(?:python\s+code|javascript\s+function|write\s+html|react\s+component|sql\s+query|debug\s+code)\b",
    r"\b(?:crypto\s+trading|bitcoin\s+investment|forex\s+trading|ethereum\s+wallet|binance)\b",
    r"\b(?:online\s+casino|sports\s+betting|ipl\s+betting|roulette|poker\s+table)\b",
    r"\b(?:movie\s+review|box\s+office\s+collection|celebrity\s+gossip)\b",
]

NON_AGRICULTURAL_KEYWORDS_TE = [
    "క్రిప్టో", "బెట్టింగ్", "సినిమా రివ్యూ", "రాజకీయ పార్టీలకు ఓటు"
]


def detect_banned_pesticides(text: str) -> List[str]:
    """Detects banned or highly hazardous pesticides mentioned in the text."""
    if not text or not isinstance(text, str):
        return []
    found: List[str] = []
    for chem in BANNED_PESTICIDES_EN:
        if re.search(r"\b" + re.escape(chem) + r"\b", text, re.IGNORECASE):
            if chem not in found:
                found.append(chem)
    for chem_te in BANNED_PESTICIDES_TE:
        if chem_te in text and chem_te not in found:
            found.append(chem_te)
    return found


def has_banned_pesticides(text: str) -> bool:
    """Returns True if any banned pesticide is detected."""
    return len(detect_banned_pesticides(text)) > 0


def detect_medical_advice(text: str) -> List[str]:
    """Detects human medical advice or pharmaceutical drug recommendations."""
    if not text or not isinstance(text, str):
        return []
    found: List[str] = []
    for kw in MEDICAL_KEYWORDS_EN:
        if re.search(r"\b" + re.escape(kw) + r"\b", text, re.IGNORECASE):
            if kw not in found:
                found.append(kw)
    for kw_te in MEDICAL_KEYWORDS_TE:
        if kw_te in text and kw_te not in found:
            found.append(kw_te)
    return found


def is_medical_advice(text: str) -> bool:
    """Returns True if human medical advice is detected."""
    return len(detect_medical_advice(text)) > 0


def detect_non_agricultural_content(text: str) -> List[str]:
    """Detects out-of-domain/non-agricultural content such as coding, betting, crypto."""
    if not text or not isinstance(text, str):
        return []
    found: List[str] = []
    for pattern in NON_AGRICULTURAL_PATTERNS_EN:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            found.append(match.group(0))
    for kw_te in NON_AGRICULTURAL_KEYWORDS_TE:
        if kw_te in text and kw_te not in found:
            found.append(kw_te)
    return found


def is_non_agricultural(text: str) -> bool:
    """Returns True if non-agricultural or out-of-domain content is detected."""
    return len(detect_non_agricultural_content(text)) > 0


# Unsafe chemical formulation patterns (e.g. 22.9 EC, 10% EC @ 2 ml/l, unverified chemical rates)
UNSAFE_FORMULATION_PATTERNS = [
    re.compile(r"\b\d+(\.\d+)?\s*%\s*(?:EC|SC|WP|SL|WDG|SP|GR|FS|ec|sc|wp|sl)\b"),
    re.compile(r"\b\d+(\.\d+)?\s*(?:EC|SC|WP|SL|WDG|SP|GR|FS)\b"),
    re.compile(r"@\s*\d+(\.\d+)?\s*(?:ml|g|gm|grams?|kg)"),
]


def detect_unsafe_chemical_formulations(text: str) -> List[str]:
    """Detects unverified chemical formulations or guessed dosage rates in text."""
    if not text or not isinstance(text, str):
        return []
    found: List[str] = []
    for pattern in UNSAFE_FORMULATION_PATTERNS:
        for match in pattern.finditer(text):
            m_str = match.group(0).strip()
            if m_str and m_str not in found:
                found.append(m_str)
    return found


def clean_telugu_text(text: str) -> str:
    """
    Cleans Telugu output text:
    - Removes parenthetical English clauses (e.g. '(If nymph population is high, spray Pyriproxyfen...)').
    - Cleans corrupted leading vowel diacritics / signs that lack a base consonant.
    - Strips leaked prompt artifacts.
    - Cleans excess whitespace.
    """
    if not text or not isinstance(text, str):
        return ""

    cleaned = text

    # Remove parenthetical English clauses
    cleaned = re.sub(r"\([A-Za-z0-9\s,%@./:;-]+\)", "", cleaned)

    # Remove leaked internal prompt tags
    cleaned = re.sub(r"\[Farmer Profile\].*?(?:\n|$)", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"## Farmer Long-Term Memory Profile.*?(?:\n|$)", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"=== RETRIEVED TRUSTED AGRICULTURAL KNOWLEDGE.*?(?:\n|$)", "", cleaned, flags=re.IGNORECASE)

    # Clean corrupted leading vowel diacritics (Telugu dependent vowels U+0C01-U+0C03, U+0C3E-U+0C4D, U+0C55, U+0C56)
    cleaned = re.sub(r"^[\u0C01-\u0C03\u0C3E-\u0C4D\u0C55\u0C56\s]+", "", cleaned)

    # Clean double spaces
    cleaned = re.sub(r"[ \t]+", " ", cleaned).strip()

    return cleaned


def is_mixed_language_output(text: str, is_user_telugu: bool) -> bool:
    """
    Checks if a response violates language purity:
    - For Telugu input: detects if there are leaked English sentences or significant English words (> 15%).
    - For English input: detects if Telugu script is present.
    """
    if not text:
        return False

    if is_user_telugu:
        # Check for English sentences (sequences of 3+ English words)
        english_sentence_match = re.search(r"\b[A-Za-z]{2,}\s+[A-Za-z]{2,}\s+[A-Za-z]{2,}\b", text)
        if english_sentence_match:
            return True
        # Calculate English word ratio
        words = text.split()
        if not words:
            return False
        eng_words = [w for w in words if re.match(r"^[A-Za-z]+[.,!?]?$", w)]
        if len(eng_words) / len(words) > 0.15:
            return True
        return False
    else:
        # English user: no Telugu script allowed
        return any("\u0C00" <= ch <= "\u0C7F" for ch in text)


def validate_generated_ai_response(
    response_text: str,
    user_message: str = "",
) -> "ResponseValidationResult":
    """
    Validates a generated AI response against safety and language standards:
    1. Language purity (Telugu input must receive pure Telugu; English input pure English)
    2. Invented/unknown pesticide names
    3. Banned pesticide recommendations
    4. Human medical advice
    5. Non-agricultural/out-of-domain content
    6. Unsafe or guessed chemical dosages / technical formulations

    Returns ResponseValidationResult with safety status and safe fallback if needed.
    """
    from src.decision_engine.models import ResponseValidationResult

    if not response_text or not isinstance(response_text, str):
        return ResponseValidationResult(is_safe=True, safe_response="")

    if user_message:
        is_user_telugu = any("\u0C00" <= ch <= "\u0C7F" for ch in user_message)
    else:
        is_user_telugu = any("\u0C00" <= ch <= "\u0C7F" for ch in response_text)

    # Pre-clean Telugu text (stripping English parentheticals and corrupted leading diacritics)
    processed_text = clean_telugu_text(response_text) if is_user_telugu else response_text.strip()

    violations: List[str] = []
    reasons: List[str] = []

    # 1. Check for invented/unknown pesticides
    unknown_chems = detect_unknown_or_example_pesticides(processed_text)
    if unknown_chems:
        violations.append("invented_pesticide")
        reasons.append(f"Response contains unverified/synthetic pesticide names: {', '.join(unknown_chems)}")

    # 2. Check for banned pesticides
    banned_chems = detect_banned_pesticides(processed_text)
    if banned_chems:
        violations.append("banned_pesticide")
        reasons.append(f"Response recommends banned or hazardous pesticide: {', '.join(banned_chems)}")

    # 3. Check for medical advice
    medical_terms = detect_medical_advice(processed_text)
    if medical_terms:
        violations.append("medical_advice")
        reasons.append(f"Response provides human medical advice: {', '.join(medical_terms)}")

    # 4. Check for non-agricultural content
    non_agri_terms = detect_non_agricultural_content(processed_text)
    if non_agri_terms:
        violations.append("non_agricultural")
        reasons.append(f"Response contains non-agricultural content: {', '.join(non_agri_terms)}")

    # 5. Check for unsafe/unverified chemical formulations or dosage claims
    unsafe_forms = detect_unsafe_chemical_formulations(processed_text)
    if unsafe_forms:
        violations.append("unsafe_formulation")
        reasons.append(f"Response contains unverified chemical formulations or dosage: {', '.join(unsafe_forms)}")

    # 6. Check for mixed language violations
    if is_mixed_language_output(processed_text, is_user_telugu):
        violations.append("mixed_language")
        reasons.append("Response violates language consistency requirement")

    if not violations:
        return ResponseValidationResult(is_safe=True, safe_response=processed_text)

    # Construct safe fallback response based on highest severity violation
    if "banned_pesticide" in violations:
        if is_user_telugu:
            safe_resp = (
                "హెచ్చరిక: ఈ ప్రతిస్పందనలో నిషేధించబడిన/హానికరమైన పురుగుమందు ప్రస్తావించబడింది. "
                "ప్రభుత్వ నిబంధనల ప్రకారం ఇది నిషేధం. దయచేసి సురక్షితమైన ప్రత్యామ్నాయాల కొరకు మీ స్థానిక వ్యవసాయ అధికారిని సంప్రదించండి."
            )
        else:
            safe_resp = (
                "Safety Warning: The generated recommendation references a banned or restricted agricultural chemical. "
                "In accordance with safety regulations, please avoid prohibited chemicals and consult your local Agricultural Extension Officer for approved alternatives."
            )
    elif "medical_advice" in violations:
        if is_user_telugu:
            safe_resp = (
                "భూమిమిత్ర కేవలం వ్యవసాయ సంబంధిత సలహాలను మాత్రమే అందిస్తుంది. "
                "మానవ ఆరోగ్య సమస్యలు లేదా మందుల కొరకు దయచేసి అర్హత కలిగిన వైద్యుడిని సంప్రదించండి."
            )
        else:
            safe_resp = (
                "BhoomiMitra is designed specifically for agricultural advisory. "
                "For human health guidance or medical prescriptions, please consult a qualified healthcare professional."
            )
    elif "non_agricultural" in violations:
        if is_user_telugu:
            safe_resp = (
                "భూమిమిత్ర కేవలం వ్యవసాయం, పంటల సంరక్షణ మరియు రైతు సంక్షేమం గురించిన ప్రశ్నలకు మాత్రమే సమాధానం ఇస్తుంది."
            )
        else:
            safe_resp = (
                "BhoomiMitra exclusively assists with agriculture, crop health, farming practices, and farmer welfare."
            )
    elif "mixed_language" in violations:
        if is_user_telugu:
            safe_resp = (
                "పంటలో తెగులు లేదా పురుగుల నివారణకు వేపనూనె (5 మి.లీ/లీటర్ నీటికి) పిచికారీ చేయడం మరియు పసుపు జిగురు అట్టలు అమర్చడం మంచిది. "
                "ఖచ్చితమైన రసాయన పిచికారీ కొరకు దయచేసి స్థానిక వ్యవసాయ అధికారిని సంప్రదించండి."
            )
        else:
            safe_resp = (
                "For pest management, we recommend starting with neem oil spray and yellow sticky traps as safe cultural practices. "
                "Please consult your local agriculture officer for exact chemical recommendations."
            )
    else:  # invented_pesticide / unsafe_formulation / unsafe dosage
        if is_user_telugu:
            safe_resp = (
                "రసాయన లేదా పురుగుమందుల ఖచ్చితమైన సమాచారాన్ని ఇక్కడ నేరుగా నిర్ధారించలేము. "
                "పంట రక్షణ కొరకు దయచేసి అధికారిక ఉత్పత్తి లేబుల్ చూడండి లేదా స్థానిక వ్యవసాయ అధికారిని సంప్రదించండి."
            )
        else:
            safe_resp = (
                "Exact chemical specifications cannot be safely verified. "
                "To protect your crop, please refer to the official product label or consult your local Agriculture Officer."
            )

    return ResponseValidationResult(
        is_safe=False,
        violations=violations,
        reasons=reasons,
        safe_response=safe_resp,
    )


