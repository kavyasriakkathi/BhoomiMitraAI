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


def validate_generated_ai_response(
    response_text: str,
    user_message: str = "",
) -> "ResponseValidationResult":
    """
    Validates a generated AI response against safety standards:
    1. Invented/unknown pesticide names
    2. Banned pesticide recommendations
    3. Human medical advice
    4. Non-agricultural/out-of-domain content
    5. Unsafe dosage claims

    Returns ResponseValidationResult with safety status and safe fallback if needed.
    """
    from src.decision_engine.models import ResponseValidationResult

    if not response_text or not isinstance(response_text, str):
        return ResponseValidationResult(is_safe=True, safe_response="")

    combined_text = f"{user_message} {response_text}"
    is_te = any("\u0C00" <= ch <= "\u0C7F" for ch in combined_text)

    violations: List[str] = []
    reasons: List[str] = []

    # 1. Check for invented/unknown pesticides
    unknown_chems = detect_unknown_or_example_pesticides(response_text)
    if unknown_chems:
        violations.append("invented_pesticide")
        reasons.append(f"Response contains unverified/synthetic pesticide names: {', '.join(unknown_chems)}")

    # 2. Check for banned pesticides
    banned_chems = detect_banned_pesticides(response_text)
    if banned_chems:
        violations.append("banned_pesticide")
        reasons.append(f"Response recommends banned or hazardous pesticide: {', '.join(banned_chems)}")

    # 3. Check for medical advice
    medical_terms = detect_medical_advice(response_text)
    if medical_terms:
        violations.append("medical_advice")
        reasons.append(f"Response provides human medical advice: {', '.join(medical_terms)}")

    # 4. Check for non-agricultural content
    non_agri_terms = detect_non_agricultural_content(response_text)
    if non_agri_terms:
        violations.append("non_agricultural")
        reasons.append(f"Response contains non-agricultural content: {', '.join(non_agri_terms)}")

    if not violations:
        return ResponseValidationResult(is_safe=True, safe_response=response_text)

    # Construct safe fallback response based on highest severity violation
    if "banned_pesticide" in violations:
        if is_te:
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
        if is_te:
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
        if is_te:
            safe_resp = (
                "భూమిమిత్ర కేవలం వ్యవసాయం, పంటల సంరక్షణ మరియు రైతు సంక్షేమం గురించిన ప్రశ్నలకు మాత్రమే సమాధానం ఇస్తుంది."
            )
        else:
            safe_resp = (
                "BhoomiMitra exclusively assists with agriculture, crop health, farming practices, and farmer welfare."
            )
    else:  # invented_pesticide / unsafe dosage
        if is_te:
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

