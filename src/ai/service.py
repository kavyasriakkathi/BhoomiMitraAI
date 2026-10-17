import time
from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from src.core.models import Farmer, Conversation, Crop, Farm
from src.core.logging import logger
from src.ai.repository import AIRepository
from src.ai.schemas import AIGenerateRequest, AIGenerateResponse, MultimodalDiagnosisResponse
from src.crop_health.service import CropHealthService
from src.crop_health.repository import CropHealthRepository
from src.crop_health.schemas import CropHealthCreate
from src.crops.repository import CropRepository
from src.farmers.repository import FarmerRepository
from src.ai.prompts import (
    BHOOMIMITRA_SYSTEM_PROMPT,
    build_farmer_context,
    get_fallback_response,
    get_unverified_dosage_fallback_response,
    get_crop_clarification_response,
)
from src.ai.crop_context import (
    resolve_image_crop_context,
    filter_history_for_vision,
    detect_crop_mismatch,
    build_crop_reconsideration_instruction,
)
from src.config import get_settings
from src.ai.gemini_client import generate_response
import re
from typing import Optional, Tuple, List, Dict, Any


def build_target_language_directive(
    language_code: Optional[str],
    default: str = "te",
    is_multimodal: bool = False
) -> str:
    """
    Constructs an explicit target-language directive for Gemini system prompts.
    Normalizes language against the centralized registry and safely falls back.
    """
    from src.language.languages import get_language, normalize_language_code, DEFAULT_LANGUAGE
    normalized = normalize_language_code(language_code, default=default or DEFAULT_LANGUAGE)
    meta = get_language(normalized)
    if not meta:
        normalized = DEFAULT_LANGUAGE
        meta = get_language(DEFAULT_LANGUAGE)

    code = meta.code
    name = meta.prompt_name or meta.display_name
    script = meta.script

    if is_multimodal:
        return (
            f"=== TARGET RESPONSE LANGUAGE DIRECTIVE ===\n"
            f"[TARGET_RESPONSE_LANGUAGE: {code}]\n"
            f"[TARGET_RESPONSE_LANGUAGE_NAME: {name}]\n"
            f"[TARGET_RESPONSE_LANGUAGE_SCRIPT: {script}]\n"
            f"MANDATORY INSTRUCTIONS:\n"
            f"1. In the JSON output, the 'friendly_whatsapp_reply' field MUST be written completely in {name} ({code}) using {script} script where applicable.\n"
            f"2. Do NOT switch to English merely because the user caption contains English words or Romanized terms.\n"
            f"3. Strictly preserve numbers, units, crop names, chemical names, and prices accurately.\n"
            f"4. This language directive strictly governs linguistic output format and does NOT override factual safety or diagnostic caution rules."
        )

    return (
        f"=== TARGET RESPONSE LANGUAGE DIRECTIVE ===\n"
        f"[TARGET_RESPONSE_LANGUAGE: {code}]\n"
        f"[TARGET_RESPONSE_LANGUAGE_NAME: {name}]\n"
        f"[TARGET_RESPONSE_LANGUAGE_SCRIPT: {script}]\n"
        f"MANDATORY INSTRUCTIONS:\n"
        f"1. You MUST respond to the farmer in {name} ({code}) using {script} script where applicable.\n"
        f"2. Do NOT switch to English merely because the farmer's input contains English words, Latin script, or Romanized terms.\n"
        f"3. If the input is Romanized / transliterated text, provide your full response in {name} native script.\n"
        f"4. Strictly preserve all numbers, units (e.g. kg/acre, ml/L), prices, dates, crop names, product names, and verified dosage values exactly without alteration.\n"
        f"5. This language directive strictly governs linguistic output format and does NOT override factual safety or Ground Truth agronomic constraints."
    )


def is_dosage_sensitive_query(text: str) -> bool:
    """
    Deterministically detects whether a farmer query is asking for exact
    fertilizer/pesticide/chemical quantities, application rates, or stage-specific
    chemical recommendations that require verified Ground Truth figures.

    Distinguishes these from pure educational/conceptual inquiries (e.g., 'why is nitrogen important?').
    """
    if not text:
        return False

    t = text.lower().strip()

    # 1. Pure educational/conceptual inquiries (no exact rate/unit asked)
    is_pure_educational_pattern = bool(re.search(
        r'^(?:why\s+(?:is|are|do|does)|what\s+is\s+(?:the\s+role\s+of|the\s+function\s+of|nitrogen|phosphorus|potash|urea|dap|biofertilizer|vermicompost)|'
        r'benefits\s+of|importance\s+of|types\s+of\s+fertilizers?|'
        r'నత్రజని\s+ప్రాముఖ్యత|భాస్వరం\s+ప్రాముఖ్యత|ప్రాముఖ్యత\s+ఏమిటి|'
        r'నైట్రోజన్\s+ఎందుకు|యూరియా\s+ఎందుకు|'
        r'नाइट्रोजन\s+का\s+महत्व|यूरिया\s+क्या\s+है|खाद\s+क्यों|'
        r'உரத்தின்\s+பயன்கள்|ಗೊಬ್ಬರದ\s+ಮಹತ್ವ)\b',
        t
    ))

    # 2. Explicit dosage units or quantity phrases (ALWAYS dosage-sensitive)
    has_explicit_dosage_unit = bool(re.search(
        r'(?:kg\s*(?:per|\/)\s*acre|kg\s*(?:per|\/)\s*ha|kg\s*(?:per|\/)\s*hectare|'
        r'grams?\s*(?:per|\/)\s*l(?:itre)?|gm\s*(?:per|\/)\s*l|g\s*\/\s*l|'
        r'ml\s*(?:per|\/)\s*l(?:itre)?|ml\s*\/\s*l|ml\s*(?:per|\/)\s*acre|'
        r'litres?\s*(?:per|\/)\s*acre|per\s+acre|per\s+hectare|per\s+litre|'
        r'how\s+many\s+kg|how\s+much\s+(?:urea|dap|potash|npk|fertilizer|pesticide|chemical|spray|zinc|sulphate|chlorantraniliprole)|'
        r'what\s+is\s+the\s+dose|what\s+is\s+the\s+dosage|what\s+dosage|spray\s+dosage|application\s+rate|dosage\s+rate|correct\s+dosage|correct\s+dose|'
        r'dose\s+of|dosage\s+of|quantity\s+of\s+fertilizer|what\s+quantities|in\s+what\s+quantities|what\s+quantity|'
        r'ఎంత\s*(?:మోతాదు|ఎరువు|యూరియా|మందు|స్ప్రే|డిఎపి|పొటాష్|జింక్|కేజీలు|గ్రాములు)|'
        r'(?:ఎకరాకి|ఎకరానికి|లీటరుకు|లీటరుకి|హెక్టారుకు)\s*ఎంత|మోతాదు\s*ఎంత|డోసేజ్|డోస్|'
        r'దశలో\s*(?:ఉన్న\s*)?(?:వరి|పత్తి|మిర్చి|మొక్కజొన్న|పంటకు)\s*ఎరువు\s*ఏది|'
        r'పిలకల\s*దశలో\s*ఎరువు|పూత\s*దశలో\s*ఎరువు|కాయ\s*దశలో\s*ఎరువు|'
        r'పిలకల\s*దశలో\s*వరి|'
        r'(?:ఏ|ఏం|ఏమి)\s*ఎరువులు?\s*(?:వేయాలి|వాడాలి|వేసుకోవాలి|అవసరం)|'
        r'(?:which|what)\s+fertilizers?\s+(?:should|to|can|need\s+to)?\s*(?:be\s+applied|apply|use|put)\b|'
        r'ఎరువు\s*ఏది|మందు\s*ఏది|స్ప్రే\s*ఏది|'
        r'వరికి\s*ఏం\s*ఎరువు|వరికి\s*ఏమి\s*ఎరువు|'
        r'పత్తికి\s*ఏం\s*ఎరువు|మిర్చికి\s*ఏం\s*ఎరువు|'
        r'వరి\s*పంటకు\s*ఎరువు|వరికి\s*ఎరువు|'
        r'వరిలో\s*ఎరువుల\s*యాజమాన్యం|వరి\s*ఎరువుల\s*మోతాదు|'
        r'(?:कितना|कितनी|कितने)\s*(?:मात्रा|खाद|यूरिया|दवा|दवाइयां|स्प्रे|डोज़)|प्रति\s*एकड़|प्रति\s*लीटर|डोज़|'
        r'एकड़\s*में\s*(?:कितना|कितनी|कितने)|लीटर\s*में\s*(?:कितना|कितनी|कितने)|'
        r'धान\s*में\s*(?:खाद|यूरिया|दवा)|गेहूं\s*में\s*(?:खाद|यूरिया|दवा)|'
        r'எவ்வளவு\s*(?:உரம்|மருந்து|அளவு|யூரியா|உரங்கள்|பூச்சிக்கொல்லி)|ஏக்கருக்கு\s*எவ்வளவு|'
        r'ಎಷ್ಟು\s*(?:ಗೊಬ್ಬರ|ಪ್ರಮಾಣ|ಔಷಧ|ಯೂರಿಯಾ|ಔಷಧಿ)|ಎಕರೆಗೆ\s*ಎಷ್ಟು|'
        r'എത്ര\s*(?:വളം|അളവ്|മരുന്ന്|യൂറിയ)|'
        r'किती\s*(?:खत|मात्रा|औषध|युरिया|डोस)|'
        r'কতটা\s*(?:সার|ওষুধ|ইউরিয়া|ডোজ)|কত\s*ডোজ|'
        r'કેટલું\s*(?:ખાતર|દવા|યુરિયા|ડોઝ)|'
        r'କେତେ\s*(?:ସାର|ଔଷଧ|ୟୁରିଆ|ଡୋଜ୍)|'
        r'ਕਿੰਨੀ\s*(?:ਖਾਦ|ਦਵਾਈ|ਯੂਰੀਆ|ਡੋਜ਼)|'
        r'কিমান\s*(?:সাৰ|ঔষধ|ইউৰিয়া|ড\'জ)|'
        r'کتنی\s*(?:کھاد|دوا|یوریا|خوراک)|'
        r'entha\s+(?:fertilizer|urea|eruvu|mandhu|dose|dosage|dap|potash|zinc|mothadulo|mothadu)|'
        r'(?:ekaraniki|ekaraki|per\s+acre)\s+entha|'
        r'kitna\s+(?:fertilizer|urea|khad|dawa|dose|dosage)|'
        r'kitni\s+(?:khad|dawa|matra)|'
        r'eshtu\s+(?:fertilizer|gobbara|aushadha|dose)|'
        r'evvalavu\s+(?:fertilizer|uram|marundhu)|'
        r'(?:vari|cotton|chilli|mirchi|paddy)\s*ki\s*(?:em|yemi|entha)\s*(?:fertilizer|eruvu|mandhu|urea)|'
        r'paddy\s*(?:fertilizer|urea)\s*dosage)',
        t
    ))

    if has_explicit_dosage_unit:
        return True

    if is_pure_educational_pattern:
        return False

    # 3. General fertilizer dosage / application queries
    has_general_dosage_intent = bool(re.search(
        r'(?:fertilizer\s+schedule|fertilizer\s+recommendation|nutrient\s+dosage|pesticide\s+dosage|chemical\s+dosage|'
        r'top\s+dressing|basal\s+dose|foliar\s+spray|'
        r'యూరియా\s+మోతాదు|ఎరువుల\s+మోతాదు|పురుగుమందు\s+మోతాదు|'
        r'యూరియా\s+ఎంత|ఎరువు\s+ఎంత|మందు\s+ఎంత)',
        t
    ))

    return has_general_dosage_intent


def extract_query_dosage_topics(query: str, crop: Optional[str] = None) -> dict:
    """
    Extracts the crop, specific fertilizer / chemical substances, and intent category
    from a dosage-sensitive query across multiple languages and Romanized inputs.
    """
    if not query:
        return {
            "crop": crop.lower() if crop else None,
            "is_fertilizer_query": False,
            "requested_fertilizers": [],
            "is_pest_disease_query": False,
            "requested_chemicals": [],
        }

    t = query.lower().strip()

    # 1. Detect crop
    # 1. Detect crop
    detected_crop = None
    if any(k in t for k in ["వరి", "వరికి", "వరిలో", "వరిపంట", "రి పంటకు", "రి పంట", "paddy", "rice", "vari", "dhan", "धान", "நெல்", "ಭತ್ತ"]) or ("నాట్లు" in t or "నాట్లలో" in t):
        detected_crop = "paddy"
    elif any(k in t for k in ["పత్తి", "పత్తికి", "పత్తిలో", "cotton", "patti", "kapas", "कपास", "பருத்தி", "ಹತ್ತಿ"]):
        detected_crop = "cotton"
    elif any(k in t for k in ["మిర్చి", "మిరప", "మిరపకి", "chilli", "chili", "mirchi", "mirapa", "मिर्च", "மிளகாய்", "ಮೆಣಸಿನಕಾಯಿ"]):
        detected_crop = "chilli"
    elif any(k in t for k in ["మొక్కజొన్న", "maize", "corn", "mokkajonna", "मक्का", "மக்காச்சோளம்", "ಮೆಕ್ಕೆಜೋಳ"]):
        detected_crop = "maize"
    elif any(k in t for k in ["టమాటా", "టమాట", "tomato", "tamata", "टमाटर", "தக்காளி", "ಟೊಮ್ಯಾಟೊ"]):
        detected_crop = "tomato"
    elif crop:
        c_l = crop.lower()
        if any(k in c_l for k in ["paddy", "rice", "వరి"]):
            detected_crop = "paddy"
        elif any(k in c_l for k in ["cotton", "పత్తి"]):
            detected_crop = "cotton"
        elif any(k in c_l for k in ["chilli", "మిర్చి"]):
            detected_crop = "chilli"
        elif any(k in c_l for k in ["maize", "మొక్కజొన్న"]):
            detected_crop = "maize"
        elif any(k in c_l for k in ["tomato", "టమాటా"]):
            detected_crop = "tomato"

    crop = detected_crop

    # 2. Specific fertilizer substances
    is_fertilizer_query = False
    requested_fertilizers = []

    if any(k in t for k in ["యూరియా", "urea", "यूरिया", "ইউরিয়া", "യൂറിയ", "యురియా"]):
        requested_fertilizers.append("urea")
        is_fertilizer_query = True
    if any(k in t for k in ["డిఎపి", "డి.ఎ.పి", "dap", "डीएपी"]):
        requested_fertilizers.append("dap")
        is_fertilizer_query = True
    if any(k in t for k in ["పొటాష్", "potash", "mop", "पोटाश"]):
        requested_fertilizers.append("potash")
        is_fertilizer_query = True
    if any(k in t for k in ["npk", "ఎన్పికె", "కాంప్లెక్స్", "19:19:19", "20:20:0:13", "10:26:26"]):
        requested_fertilizers.append("npk")
        is_fertilizer_query = True
    if any(k in t for k in ["జింక్", "జింకు", "zinc", "जिंक"]):
        requested_fertilizers.append("zinc")
        is_fertilizer_query = True
    if any(k in t for k in [
        "నత్రజని", "భాస్వరం", "nitrogen", "phosphorus", "fertilizer", "fertilizers",
        "ఎరువు", "ఎరువులు", "ఎరువుల", "eruvu", "eruvulu", "ఖాతరు", "खाद", "उर्वरक", "உரம்", "ಗೊಬ್ಬರ"
    ]):
        is_fertilizer_query = True

    # 3. Specific pests / diseases / chemicals
    requested_chemicals = []
    if any(k in t for k in ["mancozeb", "మాంకోజెబ్"]):
        requested_chemicals.append("mancozeb")
    if any(k in t for k in ["tricyclazole", "ట్రైసైక్లాజోల్", "ట్రైసైక్లజోల్"]):
        requested_chemicals.append("tricyclazole")
    if any(k in t for k in ["chlorantraniliprole", "క్లోరాంట్రానిలిప్రోల్", "కోరాజెన్", "coragen"]):
        requested_chemicals.append("chlorantraniliprole")
    if any(k in t for k in ["emamectin", "ఎమామెక్టిన్"]):
        requested_chemicals.append("emamectin")
    if any(k in t for k in ["fipronil", "ఫిప్రోనిల్"]):
        requested_chemicals.append("fipronil")
    if any(k in t for k in ["spinetoram", "స్పైనిటోరం"]):
        requested_chemicals.append("spinetoram")
    if any(k in t for k in ["profenofos", "ప్రొఫెనోఫాస్"]):
        requested_chemicals.append("profenofos")

    is_pest_disease_query = False
    if any(k in t for k in [
        "blast", "అగ్గి తెగులు", "అగ్గితెగులు",
        "stem borer", "కాండం తొలిచే పురుగు", "కాండం తొలిచే",
        "leaf spot", "ఆకుమచ్చ", "ఆకు మచ్చ", "alternaria", "ఆల్టర్నేరియా",
        "pink bollworm", "గులాబీ రంగు పురుగు",
        "thrips", "తామర పురుగులు", "తామరపురుగులు",
        "fall armyworm", "కత్తెర పురుగు", "కత్తెరపురుగు",
        "pesticide", "fungicide", "insecticide", "పురుగుమందు", "తెగులు", "పురుగు", "మందు", "పిచికారీ", "స్ప్రే"
    ]) or requested_chemicals:
        is_pest_disease_query = True

    return {
        "crop": crop,
        "is_fertilizer_query": is_fertilizer_query,
        "requested_fertilizers": requested_fertilizers,
        "is_pest_disease_query": is_pest_disease_query,
        "requested_chemicals": requested_chemicals,
    }


def extract_land_size_from_text(text: Optional[str]) -> Optional[float]:
    """
    Extract land area in acres from user queries across English, Telugu, Hindi, etc.
    e.g., '2 acres', '2.5 acre', '2 ఎకరాలు', '2 ఎకరాల్లో', '2 ఎకరాల', '1 ఎకరం', '2 एकड़'
    """
    if not text:
        return None

    pattern = r'(\d+(?:\.\d+)?)\s*(?:acres?|ac|ఎకరాల్లో|ఎకరాల|ఎకరాలు|ఎకరా|ఎకరం|एकड़|एकर)\b'
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        try:
            return float(match.group(1))
        except (ValueError, TypeError):
            return None
    return None


def has_relevant_dosage_ground_truth(query: str, rag_snippets: list) -> bool:
    """
    Deterministically evaluates whether the retrieved RAG snippets contain verified,
    topically relevant Ground Truth for the exact substance (fertilizer / pesticide / chemical)
    and crop requested in a dosage-sensitive query.

    Safety Guarantees:
    - An empty RAG list -> False.
    - An unrelated crop snippet (e.g. Cotton for a Paddy query) -> DISCARDED.
    - If user asks for fertilizer/urea dosage, a disease document with only pesticide dosages -> DISCARDED.
    - If user asks for a specific chemical/fertilizer (e.g. Urea, DAP, Tricyclazole), the snippet MUST
      explicitly mention that substance AND provide verified numeric dosage/application rates.
    - If user asks for general fertilizer dosage for a crop, the snippet MUST belong to that crop
      and provide verified fertilizer dosage recommendations.
    """
    if not rag_snippets:
        return False

    topics = extract_query_dosage_topics(query)
    q_crop = topics["crop"]
    is_fert = topics["is_fertilizer_query"]
    req_ferts = topics["requested_fertilizers"]
    is_pest = topics["is_pest_disease_query"]
    req_chems = topics["requested_chemicals"]

    for snippet in rag_snippets:
        s_lower = snippet.lower()

        # 1. Crop Match Verification
        # If query specifies a crop, ensure the snippet does not belong to a different crop
        if q_crop:
            crop_decl_match = re.search(r'crop:\s*([a-zA-Z]+)', s_lower)
            if crop_decl_match:
                s_crop = crop_decl_match.group(1).strip()
                if s_crop not in ["general", "all"] and s_crop != q_crop:
                    # Snippet belongs to a different crop
                    continue
            else:
                if q_crop == "paddy" and ("cotton" in s_lower or "పత్తి" in s_lower) and ("paddy" not in s_lower and "వరి" not in s_lower and "rice" not in s_lower):
                    continue
                if q_crop == "cotton" and ("paddy" in s_lower or "వరి" in s_lower) and ("cotton" not in s_lower and "పత్తి" not in s_lower):
                    continue
                if q_crop == "chilli" and ("cotton" in s_lower or "paddy" in s_lower) and ("chilli" not in s_lower and "మిర్చి" not in s_lower):
                    continue

        # 2. Check for numeric dosage patterns in snippet (e.g., 25 kg, 2.5 g per litre, 1.0 ml/l)
        has_dosage_in_snippet = bool(re.search(
            r'\b\d+(?:[.,]\d+)?\s*(?:-\s*\d+(?:[.,]\d+)?)?\s*'
            r'(?:kg|kgs|g|gm|gms|grams?|ml|litres?|ltr|l|కేజీలు|కేజీల|కేజీ|కిలోలు|కిలోల|కిలో|గ్రాములు|గ్రాముల|గ్రాము|మి\.లీ|లీటర్లు|లీటర్ల|లీటరు|किलो|ग्राम|मिली|लीटर)'
            r'(?:\s*(?:per|\/|ప్రతి|ఎకరాకి|ఎకరానికి|प्रति|లీటరుకు|लीटर|@)\s*(?:acre|hectare|ha|l(?:itre)?|లీటరు|एकड़|water|నీరు)?)?',
            s_lower
        ))
        if not has_dosage_in_snippet:
            continue

        # 3. Fertilizer Query Grounding Check
        if is_fert:
            # If user explicitly requested specific fertilizers (e.g. Urea, DAP, Potash, Zinc)
            if req_ferts:
                matched_specific_fert = any(
                    fert in s_lower or
                    (fert == "urea" and ("యూరియా" in s_lower or "यूरिया" in s_lower or "urea" in s_lower)) or
                    (fert == "dap" and ("డిఎపి" in s_lower or "डीएपी" in s_lower or "dap" in s_lower)) or
                    (fert == "potash" and ("పొటాష్" in s_lower or "पोटाश" in s_lower or "mop" in s_lower or "potash" in s_lower)) or
                    (fert == "zinc" and ("జింక్" in s_lower or "జింకు" in s_lower or "जिंक" in s_lower or "zinc" in s_lower))
                    for fert in req_ferts
                )
                if matched_specific_fert:
                    return True
            else:
                # General fertilizer query: snippet must discuss fertilizer / nutrient management
                is_fert_snippet = any(k in s_lower for k in [
                    "fertilizer", "fertilizers", "nutrient", "npk", "urea", "dap", "potash", "manure",
                    "ఎరువు", "ఎరువులు", "నత్రజని", "భాస్వరం", "యూరియా", "పోషకాలు", "खाद", "उर्वरक"
                ])
                # Ensure snippet is not merely a pest/disease guide with fungicide/insecticide only
                if is_fert_snippet and not (
                    ("fungicide" in s_lower or "blast" in s_lower or "stem borer" in s_lower)
                    and "fertilizer" not in s_lower and "urea" not in s_lower and "npk" not in s_lower and "ఎరువు" not in s_lower
                ):
                    return True

        # 4. Pest / Disease / Chemical Query Grounding Check
        if is_pest and not is_fert:
            if req_chems:
                matched_chem = any(chem in s_lower for chem in req_chems)
                if matched_chem:
                    return True
            else:
                is_pest_snippet = any(k in s_lower for k in [
                    "pest", "disease", "fungicide", "insecticide", "spray", "control", "management",
                    "తెగులు", "పురుగు", "నివారణ", "పిచికారీ", "రోగం", "कीट", "रोग"
                ])
                if any(w in s_lower for w in [
                    "blast", "stem borer", "leaf spot", "pink bollworm", "thrips", "fall armyworm",
                    "అగ్గి తెగులు", "అగ్గితెగులు", "కాండం తొలిచే పురుగు", "ఆకుమచ్చ", "గులాబీ రంగు పురుగు", "తామర పురుగులు", "కత్తెర పురుగు"
                ]):
                    return True
                if is_pest_snippet:
                    return True

        # 5. General dosage query
        if not is_fert and not is_pest and has_dosage_in_snippet:
            return True

    return False


def validate_output_chemical_dosage_safety(
    text: str,
    rag_snippets: list,
    effective_crop: Optional[str] = None,
) -> Tuple[bool, Optional[str]]:
    """
    Validates that every pesticide/chemical/fertilizer dosage recommendation strictly matches
    retrieved ground truth by:
      - crop
      - disease/pest
      - chemical/active ingredient
      - numeric dosage
      - unit
    Returns (is_valid: bool, reason: Optional[str]).
    """
    if not text:
        return True, None

    t_lower = text.lower()

    # Known chemicals (canonical name: [list of English and vernacular aliases])
    chemical_map = {
        "mancozeb": ["mancozeb", "మాంకోజెబ్"],
        "copper oxychloride": ["copper oxychloride", "కాపర్ ఆక్సీక్లోరైడ్", "కాపర్ ఆక్సిక్లోరైడ్"],
        "propiconazole": ["propiconazole", "ప్రొపికొనజోల్", "ప్రోపికొనజోల్"],
        "pyraclostrobin": ["pyraclostrobin", "పైరాక్లోస్ట్రోబిన్"],
        "azoxystrobin": ["azoxystrobin", "అజాక్సిస్ట్రోబిన్"],
        "difenoconazole": ["difenoconazole", "డైఫెనోకొనజోల్"],
        "profenofos": ["profenofos", "ప్రొఫెనోఫాస్"],
        "emamectin benzoate": ["emamectin benzoate", "emamectin", "ఎమామెక్టిన్ బెంజోయేట్", "ఎమామెక్టిన్"],
        "spinetoram": ["spinetoram", "స్పైనిటోరం"],
        "fipronil": ["fipronil", "ఫిప్రోనిల్"],
        "spinosad": ["spinosad", "స్పైనోసాడ్"],
        "acetamiprid": ["acetamiprid", "అసిటామప్రిడ్"],
        "diafenthiuron": ["diafenthiuron", "డయాఫెంథియురాన్"],
        "chlorantraniliprole": ["chlorantraniliprole", "క్లోరాంట్రానిలిప్రోల్", "coragen", "కోరాజెన్"],
        "cartap hydrochloride": ["cartap hydrochloride", "cartap", "కార్టాప్ హైడ్రోక్లోరైడ్", "కార్టాప్"],
        "tricyclazole": ["tricyclazole", "ట్రైసైక్లాజోల్", "ట్రైసైక్లజోల్"],
        "isoprothiolane": ["isoprothiolane", "ఐసోప్రోతియోలేన్"],
        "glyphosate": ["glyphosate", "గ్లైఫోసేట్"],
        "hexaconazole": ["hexaconazole", "హెక్సాకొనజోల్"],
    }

    detected_chems = []
    for canonical_name, aliases in chemical_map.items():
        for alias in aliases:
            if re.search(r'\b' + re.escape(alias) + r'\b', t_lower):
                detected_chems.append((canonical_name, alias))
                break

    # Fertilizers
    fertilizer_map = {
        "urea": ["urea", "యూరియా", "यूरिया"],
        "dap": ["dap", "డిఎపి", "डीएपी"],
        "potash": ["potash", "mop", "పొటాష్", "पोटाश"],
        "ssp": ["ssp"],
        "zinc": ["zinc", "జింక్", "जिंक"],
    }
    detected_ferts = []
    for canonical_name, aliases in fertilizer_map.items():
        for alias in aliases:
            if re.search(r'\b' + re.escape(alias) + r'\b', t_lower):
                detected_ferts.append((canonical_name, alias))
                break

    dosage_pattern = (
        r'\b\d+(?:[.,]\d+)?\s*(?:-\s*\d+(?:[.,]\d+)?)?\s*'
        r'(?:kg|kgs|g|gm|gms|grams?|ml|litres?|ltr|l|కేజీలు|కేజీల|కేజీ|కిలోలు|కిలోల|కిలో|గ్రాములు|గ్రాముల|గ్రాము|మి\.లీ|లీటర్లు|లీటర్ల|లీటరు|किलो|ग्राम|मिली|लीटर)\b'
    )
    has_dosage = bool(re.search(dosage_pattern, t_lower))

    if not detected_chems and not detected_ferts:
        return True, None

    if not has_dosage:
        return True, None

    if not rag_snippets:
        return False, "No verified RAG ground truth available for recommended chemical/dosage."

    # Validate chemicals
    for canonical_name, alias in detected_chems:
        matching_snippets = []
        for s in rag_snippets:
            s_lower = s.lower()
            if any(re.search(r'\b' + re.escape(a) + r'\b', s_lower) for a in chemical_map[canonical_name]):
                matching_snippets.append(s)

        if not matching_snippets:
            return False, f"Unverified chemical '{canonical_name}' not approved in RAG ground truth."

        # Check disease alignment
        disease_keywords = {
            "alternaria leaf spot": ["alternaria", "leaf spot", "ఆల్టర్నేరియా", "ఆకుమచ్చ"],
            "pink bollworm": ["pink bollworm", "bollworm", "గులాబీ రంగు పురుగు"],
            "blast": ["blast", "అగ్గి తెగులు", "అగ్గితెగులు"],
            "stem borer": ["stem borer", "కాండం తొలిచే పురుగు"],
            "thrips": ["thrips", "తామర పురుగులు", "తామరపురుగులు"],
            "fall armyworm": ["fall armyworm", "armyworm", "కత్తెర పురుగు"],
        }
        for d_name, d_kws in disease_keywords.items():
            if any(kw in t_lower for kw in d_kws):
                disease_matched_in_snippet = False
                for ms in matching_snippets:
                    ms_lower = ms.lower()
                    if any(kw in ms_lower for kw in d_kws):
                        disease_matched_in_snippet = True
                        break
                if not disease_matched_in_snippet:
                    return False, f"Chemical '{canonical_name}' is not approved for {d_name} in RAG ground truth."

        # Check dosage numbers near chemical
        chem_dosage_matches = re.findall(
            r'\b(\d+(?:[.,]\d+)?)\s*(?:-\s*(\d+(?:[.,]\d+)?))?\s*(?:g|gm|gms|grams?|ml|litres?|ltr|l|గ్రాములు|గ్రాముల|గ్రాము|మి\.లీ|లీటర్లు|లీటర్ల|లీటరు)\b',
            t_lower
        )
        if chem_dosage_matches:
            rag_combined = " ".join(matching_snippets).lower()
            rag_numbers = set(re.findall(r'\b\d+(?:[.,]\d+)?\b', rag_combined))
            for match in chem_dosage_matches:
                for num in match:
                    if num and num not in rag_numbers:
                        return False, f"Dosage figure '{num}' for chemical '{canonical_name}' not found in verified RAG ground truth."

    # Validate fertilizers
    for canonical_name, alias in detected_ferts:
        fert_dosage_matches = re.findall(
            r'\b(\d+(?:[.,]\d+)?)\s*(?:-\s*(\d+(?:[.,]\d+)?))?\s*(?:kg|kgs|కేజీలు|కేజీల|కిలోలు|కిలోల|కిలో|किलो)\b',
            t_lower
        )
        if fert_dosage_matches:
            rag_combined = " ".join(rag_snippets).lower()
            rag_numbers = set(re.findall(r'\b\d+(?:[.,]\d+)?\b', rag_combined))
            for match in fert_dosage_matches:
                for num in match:
                    if num and num not in rag_numbers:
                        return False, f"Dosage figure '{num}' for fertilizer {canonical_name} not found in verified RAG ground truth."

    return True, None


def is_unconfirmed_crop_diagnosis(
    message: str,
    history_records: Optional[list] = None,
) -> bool:
    """
    Detects whether the crop diagnosis is uncertain, conditional, or if the farmer
    explicitly stated they cannot provide a photo or diagnostic sample.
    """
    t = (message or "").lower()
    combined_history = ""
    if history_records:
        for r in history_records[-3:]:
            u_msg = getattr(r, "user_message", None) or (r.get("parts") if isinstance(r, dict) and r.get("role") == "user" else "")
            combined_history += " " + (u_msg or "").lower()

    # 1. Farmer explicitly states they cannot provide/take/send a photo
    photo_negated_patterns = [
        r"(?:ఫోటో|ఫోటోలు|ఫోటోని|చిత్రం|ఫొటో)\s*(?:ఇవ్వడం|ఇవ్వలేను|ఇవ్వడం\s*కుదరదు|కుదరదు|వీలు\s*కాదు|లేదు|పంపలేను|తీసుకోలేను|పంపడం\s*కుదరదు|సాధ్యం\s*కాదు)",
        r"(?:photo|picture|image)\s*(?:ivvadam\s*kudaradu|kudaradu|ledhu|ledu|ivvalenu|pampalenu|veelu\s*kaadhu)",
        r"(?:cannot|can'?t|unable\s+to|not\s+possible\s+to)\s+(?:send|provide|share|take|upload|give)\s+(?:a\s+)?(?:photo|picture|image)",
        r"(?:no\s+photo|don'?t\s+have\s+(?:a\s+)?photo|photo\s+(?:is\s+)?(?:not\s+available|not\s+possible))",
        r"(?:फोटो)\s*(?:नहीं\s*भेज\s*सकता|नहीं\s*है|देना\s*संभव\s*नहीं)",
    ]
    if any(re.search(p, t, re.IGNORECASE) for p in photo_negated_patterns) or any(re.search(p, combined_history, re.IGNORECASE) for p in photo_negated_patterns):
        return True

    # 2. Symptoms are partial or conditional (e.g. general spots/yellowing but cannot share photo or inconclusive)
    if re.search(r"(?:మచ్చలు|మచ్చ|spots|yellow|పసుపు).*?(?:కానీ|but|ఫోటో|photo)", t, re.IGNORECASE):
        return True

    # 3. Farmer is describing generic visual symptoms (spots, yellowing, drying, curling) without a confirmed disease name
    has_symptoms = any(k in t for k in [
        "మచ్చలు", "మచ్చ", "పసుపు", "ముడుచు", "ఎండిపో", "రాలిపో", "పురుగులు వస్తున్నాయి",
        "spots", "yellow", "yellowing", "curling", "drying", "falling", "dying"
    ])
    has_confirmed_disease = any(k in t for k in [
        "alternaria", "ఆల్టర్నేరియా", "blast", "అగ్గి తెగులు", "stem borer", "కాండం తొలిచే",
        "pink bollworm", "గులాబీ రంగు", "thrips", "తామర పురుగులు", "fall armyworm", "కత్తెర పురుగు"
    ])
    if has_symptoms and not has_confirmed_disease:
        return True

    return False


def check_fertilizer_history_details(
    message: str,
    history_records: Optional[list] = None,
) -> dict:
    """
    Identifies whether essential fertilizer history and variety information
    are available in the farmer's current query or recent conversation history.

    Distinguishes positive application history from explicit statements that
    fertilizer was NOT applied (e.g., "నాట్లలో DAP వేయలేదు", "యూరియా ఇంకా వేయలేదు",
    "ఇంతవరకు ఏ ఎరువూ వేయలేదు", "SSP లేదా కాంప్లెక్స్ ఎరువులు వేయలేదు").
    Also tracks whether variety is unknown to the farmer and whether soil test is unavailable.

    Returns dict with keys:
    - has_basal_history (bool): Positive application of DAP, SSP, or complex fertilizers.
    - has_prior_urea_or_potash (bool): Positive application of post-transplanting urea or potash.
    - has_variety_info (bool): Rice variety or duration group (short, medium, long duration).
    - variety_unknown (bool): Farmer explicitly stated variety or duration is unknown.
    - has_soil_test_info (bool): Soil test results or Soil Health Card available.
    - soil_test_negated (bool): Farmer explicitly stated no soil test report is available.
    - has_essential_history (bool): True if basal, prior urea, or soil test is provided.
    - asks_if_details_needed (bool): True if farmer is asking whether history/details are needed.
    - dap_negated (bool): DAP was not applied at transplanting.
    - ssp_or_complex_applied (bool): SSP or complex fertilizer was applied.
    - ssp_or_complex_negated (bool): SSP or complex was not applied.
    - urea_negated (bool): Urea or potash was not applied.
    - general_fert_negated (bool): General fertilizer was not applied.
    - all_clarifications_answered (bool): All clarification questions have been addressed.
    """
    text_corpus = message or ""
    if history_records:
        for r in history_records:
            if hasattr(r, "user_message") and r.user_message:
                text_corpus += " " + r.user_message
            elif isinstance(r, dict) and r.get("role") == "user":
                parts = r.get("parts")
                if isinstance(parts, str):
                    text_corpus += " " + parts
                elif isinstance(parts, list):
                    text_corpus += " " + " ".join(str(p) for p in parts)

    t = text_corpus.lower()
    msg_l = (message or "").lower()

    # Detect if farmer is asking whether details are needed rather than providing them
    asks_if_details_needed = bool(re.search(
        r"(?:వివరాలు|సమాచారం|details|information)\s*(?:అవసరమా|కావాలా|చెప్పాలా|needed|required|should\s+i\s+provide|necessary)",
        msg_l,
        re.IGNORECASE
    ))

    # General negation: e.g. "ఇంతవరకు ఏ ఎరువూ వేయలేదు", "no fertilizer applied"
    general_fert_negated = bool(re.search(
        r"(?:ఇంతవరకు|ఇప్పటివరకు|ఇప్పటిదాకా|గతంలో)\s*(?:ఎటువంటి|ఎలాంటి|ఏ)?\s*(?:ఎరువు|ఎరువులు|ఖాతరు)?\s*(?:కూడా|ఏమీ|ఏమి)?\s*(?:వేయలేదు|వేయ్యలేదు|చల్లలేదు|వాడలేదు|వేసుకోలేదు)"
        r"|(?:ఎటువంటి|ఎలాంటి|ఏ)\s*(?:ఎరువు|ఎరువులు|ఖాతరు)\s*(?:కూడా|ఏమీ|ఏమి)?\s*(?:వేయలేదు|వేయ్యలేదు|చల్లలేదు|వాడలేదు|వేసుకోలేదు)"
        r"|(?:haven'?t\s+applied|have\s+not\s+applied|didn'?t\s+apply|did\s*not\s+apply|never\s+applied|no)\s+(?:\w+\s*){0,2}(?:fertilizer|inputs?)",
        t,
        re.IGNORECASE
    ))

    # 1. Basal Fertilizer Application History (DAP, SSP, Complex)
    dap_negated = general_fert_negated or bool(re.search(
        r"(?:dap|డిఎపి|డి\.ఎ\.పి|డీఏపీ)\s*(?:ఏమీ|ఏమి|ఏదీ)?\s*(?:ఇంకా\s*)?(?:వేయలేదు|వేయ్యలేదు|చల్లలేదు|వాడలేదు|వేసుకోలేదు)"
        r"|(?:నాట్లలో|నాట్ల\s*సమయంలో|చివరి\s*దుక్కిలో|basal)\s*(?:\w+\s*){0,3}(?:dap|డిఎపి|డి\.ఎ\.పి|డీఏపీ)\s*(?:వేయలేదు|వేయ్యలేదు|చల్లలేదు|వాడలేదు|వేసుకోలేదు)"
        r"|(?:not\s+applied|didn'?t\s+apply|did\s*not\s+apply|haven'?t\s+applied|have\s*not\s+applied|never\s+applied|no)\s+(?:\w+\s*){0,3}(?:dap)"
        r"|(?:dap)\s+(?:\w+\s*){0,4}(?:not\s+applied|not\s+used|not\s+given|wasn'?t\s+applied|has\s+not\s+been\s+applied|didn'?t\s+apply|did\s*not\s+apply)",
        t,
        re.IGNORECASE
    ))

    ssp_or_complex_applied = bool(re.search(
        r"(?:ssp|ఎస్ఎస్పి|కాంప్లెక్స్|complex|20:20:0|28:28:0)\s*(?:\d+\s*(?:కేజీల|కిలోల|kg|బస్తాల|బ్యాగుల)?\s*)?(?:వేశాను|వేసాను|వేశాం|వేసాం|చల్లాను|చల్లాం|వాడాను|వాడాం|applied|added|put|used)",
        t,
        re.IGNORECASE
    ))

    ssp_or_complex_negated = bool(re.search(
        r"(?:ssp|ఎస్ఎస్పి|కాంప్లెక్స్|complex|20:20:0|28:28:0)\s*(?:లేదా|మరియు|గానీ|గాని|or|and)?\s*(?:ఇతర|other)?\s*(?:ssp|ఎస్ఎస్పి|కాంప్లెక్స్|complex)?\s*(?:ఎరువులు|ఎరువు|fertilizer|fertilizers|खाद)?\s*(?:కూడా|ఏమీ|ఏమి|ఏదీ)?\s*(?:ఇంకా\s*)?(?:వేయలేదు|వేయ్యలేదు|చల్లలేదు|వాడలేదు|వేసుకోలేదు)"
        r"|(?:నాట్లలో|నాట్ల\s*సమయంలో|చివరి\s*దుక్కిలో|basal)\s*(?:\w+\s*){0,3}(?:ssp|కాంప్లెక్స్|complex)\s*(?:కూడా|ఏమీ|ఏమి)?\s*(?:వేయలేదు|వేయ్యలేదు|చల్లలేదు|వాడలేదు|వేసుకోలేదు)"
        r"|(?:not\s+applied|didn'?t\s+apply|did\s*not\s+apply|haven'?t\s+applied|have\s*not\s+applied|never\s+applied|no|neither)\s+(?:\w+\s*){0,4}(?:ssp|complex)"
        r"|(?:ssp|complex)\s+(?:\w+\s*){0,4}(?:not\s+applied|not\s+used|not\s+given|wasn'?t\s+applied|has\s+not\s+been\s+applied|didn'?t\s+apply|did\s*not\s+apply|also\s+not\s+applied)"
        r"|(?:ssp|कॉम्प्लेक्स|खाद)\s*(?:भी\s*)?(?:नहीं\s*डाली|नहीं\s*दिया|नहीं\s*डाला)",
        t,
        re.IGNORECASE
    ))

    basal_negated = general_fert_negated or dap_negated or ssp_or_complex_negated or bool(re.search(
        r"(?:dap|డిఎపి|డి\.ఎ\.పి|డీఏపీ|ssp|ఎస్ఎస్పి|కాంప్లెక్స్|complex|20:20:0|28:28:0)\s*(?:ఏమీ|ఏమి|ఏదీ)?\s*(?:ఇంకా\s*)?(?:వేయలేదు|వేయ్యలేదు|చల్లలేదు|వాడలేదు|వేసుకోలేదు)"
        r"|(?:నాట్లలో|నాట్ల\s*సమయంలో|చివరి\s*దుక్కిలో|basal)\s*(?:\w+\s*){0,3}(?:dap|ssp|కాంప్లెక్స్|complex|ఎరువు|fertilizer)?\s*(?:ఏమీ|ఏమి|కూడా)?\s*(?:వేయలేదు|వేయ్యలేదు|చల్లలేదు|వాడలేదు|వేసుకోలేదు)"
        r"|(?:not\s+applied|didn'?t\s+apply|did\s*not\s+apply|haven'?t\s+applied|never\s+applied|no)\s+(?:\w+\s*){0,3}(?:dap|ssp|complex|basal|fertilizer)"
        r"|(?:dap|ssp|complex|basal)\s+(?:\w+\s*){0,2}(?:not\s+applied|not\s+used|not\s+given)",
        t,
        re.IGNORECASE
    ))

    has_basal_history = False
    if not (basal_negated and not ssp_or_complex_applied):
        basal_applied_patterns = [
            r"(?:నాట్లలో|నాట్ల\s*సమయంలో|చివరి\s*దుక్కిలో|basal|at\s*transplanting)\s*(?:ఎకరానికి\s*)?\d*\s*(?:కేజీల|కిలోల|kg|బస్తాల|బ్యాగుల)?\s*(?:dap|డిఎపి|డి\.ఎ\.పి|డీఏపీ|ssp|ఎస్ఎస్పి|కాంప్లెక్స్|complex|20:20:0|28:28:0)\s*(?:వేశాను|వేసాను|వేశాం|వేసాం|చల్లాను|చల్లాం|వాడాను|వాడాం|applied|put|used|added)",
            r"(?:dap|డిఎపి|డి\.ఎ\.పి|డీఏపీ|ssp|ఎస్ఎస్పి|కాంప్లెక్స్|complex|20:20:0|28:28:0)\s*(?:\d+\s*(?:కేజీల|కిలోల|kg|బస్తాల|బ్యాగుల)?\s*)?(?:వేశాను|వేసాను|వేశాం|వేసాం|చల్లాను|చల్లాం|వాడాను|వాడాం|applied|added|put|used)",
            r"(?:applied|put|used|added)\s*(?:\d+\s*(?:kg|bags?)?\s*(?:of\s*)?)?(?:dap|ssp|complex)",
            r"ఎకరానికి\s*\d+\s*(?:కిలోల|కేజీల|kg)?\s*(?:dap|ssp|కాంప్లెక్స్)\s*(?:వేశాను|వేసాను|వేశాం|వేసాం|చల్లాను|చల్లాం|వాడాను|వాడాం|applied|used)",
        ]
        if any(re.search(p, t, re.IGNORECASE) for p in basal_applied_patterns):
            has_basal_history = True

    # 2. Prior Post-Transplanting Urea or Potash Application
    urea_negated = general_fert_negated or bool(re.search(
        r"(?:యూరియా|urea|పొటాష్|potash|mop)\s*(?:ఏమీ|ఏమి|ఏదీ|కూడా)?\s*(?:ఇంకా\s*)?(?:వేయలేదు|వేయ్యలేదు|చల్లలేదు|వాడలేదు|వేసుకోలేదు)"
        r"|(?:ఇంకా|ఇంతవరకు|ఇప్పటివరకు|ఇప్పటిదాకా)\s*(?:యూరియా|urea|పొటాష్|potash)\s*(?:కూడా\s*)?(?:వేయలేదు|వేయ్యలేదు|చల్లలేదు|వాడలేదు)"
        r"|(?:not\s+applied|didn'?t\s+apply|did\s*not\s+apply|haven'?t\s+applied|have\s*not\s+applied|never\s+applied|no)\s+(?:\w+\s*){0,3}(?:urea|potash|mop|top-dressing|top\s*dressing)"
        r"|(?:urea|potash|mop)\s+(?:\w+\s*){0,4}(?:not\s+applied|not\s+used|not\s+given|not\s+yet\s+applied|has\s+not\s+been\s+applied|wasn'?t\s+applied|didn'?t\s+apply|did\s*not\s+apply)",
        t,
        re.IGNORECASE
    ))
    has_prior_urea_or_potash = False
    if not urea_negated:
        prior_urea_patterns = [
            r"(?:యూరియా|urea|పొటాష్|potash|mop)\s*(?:\d+\s*(?:కేజీల|కిలోల|kg|బస్తాల|బ్యాగుల)?\s*)?(?:వేశాను|వేసాను|వేశాం|వేసాం|చల్లాను|చల్లాం|వాడాను|వాడాం|applied|put|used|added)",
            r"(?:applied|put|used|added)\s*(?:\d+\s*(?:kg|bags?)?\s*(?:of\s*)?)?(?:urea|potash|mop)",
            r"(?:25|20|15|10)\s*రోజులప్పుడు\s*(?:\d+\s*(?:కేజీల|కిలోల|kg)?\s*)?(?:యూరియా|urea|పొటాష్|potash)\s*(?:వేశాను|వేసాను|వేశాం|వేసాం|చల్లాను|చల్లాం|applied|used)",
            r"మొన్ననే\s*(?:\d+\s*(?:కేజీల|కిలోల|kg)?\s*)?(?:యూరియా|urea)\s*(?:వేశాను|వేసాను|వేశాం|చల్లాను|applied|used)",
            r"(?:యూరియా|urea)\s*వేసి\s*\d+\s*రోజులు",
        ]
        if any(re.search(p, t, re.IGNORECASE) for p in prior_urea_patterns):
            has_prior_urea_or_potash = True

    # 3. Variety / Duration Group Info
    variety_unknown = bool(re.search(
        r"(?:వరి\s*)?(?:రకం|రకము|వంగడం|variety|duration)\s*(?:పేరు|పేరు\s*కూడా)?\s*(?:నాకు\s*)?(?:కూడా\s*)?(?:తెలియదు|తెలీదు|గుర్తులేదు|చెప్పలేను)"
        r"|(?:నాకు\s*)?(?:వరి\s*)?(?:రకం|వంగడం)\s*(?:గురించి\s*)?(?:తెలియదు|తెలీదు)"
        r"|(?:రకం|వంగడం)\s*పేరు\s*(?:కూడా\s*)?తెలియదు"
        r"|(?:i\s*)?(?:do\s*not|don'?t)\s*know\s+(?:\w+\s*){0,4}(?:variety|cultivar|duration|name)"
        r"|(?:variety|duration)\s+(?:is\s*)?(?:unknown|not\s*known)"
        r"|(?:unknown|not\s*known)\s+(?:variety|duration)"
        r"|(?:धान\s*की\s*)?(?:किस्म|वैरायटी)\s*(?:का\s*नाम\s*)?(?:मुझे\s*)?(?:नहीं\s*पता|मालूम\s*नहीं)",
        t,
        re.IGNORECASE
    ))

    has_variety_info = False
    if not variety_unknown:
        variety_patterns = [
            r"\b(?:short|medium|long)\s*duration\b",
            r"(?:స్వల్ప|మధ్య|దీర్ఘ)\s*కాలిక",
            r"(?:రకం|వంగడం|variety)\s*[:=]?\s*(?:bpt|mtu|rnr|telangana\s*sona|sambha|సాంబ|1010|5204|15048)",
            r"\b(?:bpt\s*5204|mtu\s*1010|rnr\s*15048|telangana\s*sona)\b",
        ]
        if any(re.search(p, t, re.IGNORECASE) for p in variety_patterns):
            has_variety_info = True

    # 4. Soil-test Info (Soil Health Card)
    soil_test_negated = bool(re.search(
        r"(?:సాయిల్\s*టెస్ట్|భూసార\s*పరీక్ష|నేల\s*పరీక్ష|మట్టి\s*పరీక్ష|soil\s*test|soil\s*health\s*card)\s*(?:రిపోర్ట్|రిపోర్టు|నివేదిక|card)?\s*(?:కూడా\s*)?(?:లేదు|చేయించలేదు|చేయలేదు|అందుబాటులో\s*లేదు)"
        r"|(?:దగ్గర\s*)?(?:నేల|భూసార|మట్టి|సాయిల్)\s*(?:పరీక్ష\s*)?(?:రిపోర్ట్|నివేదిక)\s*(?:కూడా\s*)?(?:లేదు|అందుబాటులో\s*లేదు)"
        r"|(?:no\s+|don'?t\s+have\s+(?:a\s+)?|haven'?t\s+done\s+(?:a\s+)?)(?:soil\s*test|soil\s*health\s*card|soil\s*testing)(?:\s+report)?"
        r"|(?:soil\s*test|soil\s*health\s*card)(?:\s+report)?\s+(?:is\s+)?(?:not\s+available|not\s+done|unavailable|absent|missing|either)"
        r"|(?:मृदा\s*परीक्षण|मिट्टी\s*की\s*जांच|सॉइल\s*टेस्ट)\s*(?:रिपोर्ट\s*)?(?:भी\s*)?(?:नहीं\s*है|उपलब्ध\s*नहीं\s*है|नहीं\s*करवाई)",
        t,
        re.IGNORECASE
    ))

    has_soil_test_info = False
    if not soil_test_negated:
        soil_test_patterns = [
            r"(?:సాయిల్\s*టెస్ట్|భూసార\s*పరీక్ష|నేల\s*పరీక్ష|soil\s*test|soil\s*health\s*card).*(?:రిపోర్ట్\s*ఉంది|రిపోర్టు\s*ఉంది|ఉంది|చేయించాను|shows|report\s*available|available)",
            r"(?:నత్రజని|నైట్రోజన్|భాస్వరం|nitrogen|phosphorus)\s*(?:ఎక్కువ|తక్కువ|హై|లో|high|low|medium|మధ్యస్థం)",
        ]
        if any(re.search(p, t, re.IGNORECASE) for p in soil_test_patterns):
            has_soil_test_info = True

    has_essential_history = has_basal_history or has_prior_urea_or_potash or has_soil_test_info

    basal_answered = (
        has_basal_history
        or general_fert_negated
        or (dap_negated and (ssp_or_complex_negated or ssp_or_complex_applied))
        or (ssp_or_complex_negated and (dap_negated or has_basal_history))
    )
    urea_answered = has_prior_urea_or_potash or urea_negated or general_fert_negated
    variety_answered = has_variety_info or variety_unknown
    soil_answered = has_soil_test_info or soil_test_negated
    all_clarifications_answered = bool(basal_answered and urea_answered and variety_answered and soil_answered)

    return {
        "has_basal_history": has_basal_history,
        "has_prior_urea_or_potash": has_prior_urea_or_potash,
        "has_variety_info": has_variety_info,
        "variety_unknown": variety_unknown,
        "has_soil_test_info": has_soil_test_info,
        "soil_test_negated": soil_test_negated,
        "has_essential_history": has_essential_history,
        "asks_if_details_needed": asks_if_details_needed,
        "dap_negated": dap_negated,
        "ssp_or_complex_applied": ssp_or_complex_applied,
        "ssp_or_complex_negated": ssp_or_complex_negated,
        "urea_negated": urea_negated,
        "general_fert_negated": general_fert_negated,
        "all_clarifications_answered": all_clarifications_answered,
    }
def is_paddy_tillering_fertilizer_query(
    message: str,
    crop: Optional[str] = None,
    history_records: Optional[list] = None,
) -> bool:
    """
    Detects if query is asking for paddy fertilizer advice at tillering stage / around 30 days DAT.
    Guards against classifying clearly identified non-paddy crops as paddy.
    Accepts history_records to preserve context in multi-turn clarification conversations.
    """
    if not message:
        return False
    t = message.lower()
    crop_lower = (crop or "").strip().lower()

    has_paddy_mention = (
        any(k in crop_lower for k in ["paddy", "rice", "వరి"])
        or any(k in t for k in ["వరి", "వరికి", "వరిలో", "వరిపంట", "paddy", "rice", "vari", "dhan", "धान", "நெల్"])
    )

    # Guard: do not classify clearly identified non-paddy crops as paddy
    from src.rag.service import extract_crop_from_text
    msg_crop = extract_crop_from_text(message)
    if msg_crop and msg_crop.lower() not in ["paddy", "rice", "వరి"]:
        return False
    if crop_lower and not any(k in crop_lower for k in ["paddy", "rice", "వరి"]) and not has_paddy_mention:
        return False

    is_paddy = has_paddy_mention or ("నాట్లు" in t or "నాట్లలో" in t or "transplanting" in t)
    is_fert = any(k in t for k in [
        "ఎరువు", "ఎరువులు", "యూరియా", "మోతాదు", "fertilizer", "fertilizers", "urea", "dose", "dosage", "eruvu", "eruvulu", "ఖాతరు", "खाद", "dap", "ssp", "కాంప్లెక్స్", "complex", "potash", "పొటాష్", "అంచనా", "guess", "estimate"
    ])
    is_stage = any(k in t for k in [
        "30", "35", "25", "20", "రోజులు", "రోజుల", "days", "dat", "పిలక", "పిలకల", "tillering", "నాట్లు వేసి", "transplanting"
    ])

    if is_paddy and is_fert and is_stage:
        return True

    # Multi-turn context: If recent turn in history was a paddy tillering fertilizer query,
    # and current message provides fertilizer/crop/soil clarifications without switching to non-paddy
    if history_records:
        recent_paddy_tillering = False
        for r in history_records[-2:]:
            prev_msg = getattr(r, "user_message", None) or (r.get("parts") if isinstance(r, dict) and r.get("role") == "user" else None)
            if prev_msg and is_paddy_tillering_fertilizer_query(prev_msg, crop=crop):
                recent_paddy_tillering = True
                break
        if recent_paddy_tillering:
            is_clarification_reply = any(k in t for k in [
                "ssp", "dap", "urea", "కాంప్లెక్స్", "complex", "ఎరువు", "యూరియా", "రకం", "వంగడం", "variety", "duration",
                "సాయిల్", "భూసార", "నేల", "పరీక్ష", "soil", "health card", "card", "నివేదిక", "వేయలేదు", "లేదు", "తెలియదు",
                "తెలీదు", "not applied", "don't know", "unknown", "not available", "no report", "खाद", "किस्म", "नहीं पता",
                "అంచనా", "guess", "estimate", "మోతాదు", "dose", "dosage", "ఎంత", "how much"
            ])
            if is_clarification_reply:
                return True

    return False
def build_unconfirmed_paddy_fertilizer_guidance(
    user_lang: str = "te",
    acres: float = 2.0,
    fert_hist_info: Optional[dict] = None,
) -> str:
    """
    Constructs an attributed, safe unconfirmed guidance response for paddy at 30 DAT tillering stage
    when essential fertilizer history is missing.
    NEVER includes numerical dosage benchmarks (e.g. 30-35 kg/acre or 60-70 kg/2 acres) before history is established.
    Personalizes clarifying inquiries based on what information has already been supplied or negated.
    """
    info = fert_hist_info or {}
    dap_neg = info.get("dap_negated", False)
    urea_neg = info.get("urea_negated", False)
    has_basal = info.get("has_basal_history", False)
    has_prior_urea = info.get("has_prior_urea_or_potash", False)
    has_variety = info.get("has_variety_info", False)
    var_unk = info.get("variety_unknown", False)
    has_soil = info.get("has_soil_test_info", False)
    soil_neg = info.get("soil_test_negated", False)
    general_fert_neg = info.get("general_fert_negated", False)
    ssp_neg = info.get("ssp_or_complex_negated", False)
    ssp_applied = info.get("ssp_or_complex_applied", False)

    if user_lang == "te":
        questions = []
        if dap_neg:
            if not ssp_neg and not ssp_applied and not general_fert_neg:
                questions.append("నాట్ల సమయంలో SSP లేదా ఇతర కాంప్లెక్స్ ఎరువులు ఏవైనా వేశారా, ఎంత మోతాదులో వేశారు?")
        elif not has_basal and not general_fert_neg and not ssp_neg:
            questions.append("నాట్ల సమయంలో DAP, SSP లేదా కాంప్లెక్స్ ఎరువులు ఏవైనా వేశారా, ఎంత మోతాదులో వేశారు?")

        if not urea_neg and not has_prior_urea and not general_fert_neg:
            questions.append("నాట్లు వేసిన తర్వాత ఇప్పటికే ఏదైనా యూరియా లేదా పొటాష్ వేశారా (ఎంత మోతాదు, ఏ తేదీ/దశలో)?")

        if not has_variety and not var_unk:
            questions.append("మీరు సాగు చేస్తున్న వరి రకం/వంగడం ఏది (స్వల్పకాలిక లేదా దీర్ఘకాలిక)?")

        if not has_soil and not soil_neg:
            questions.append("నేల పరీక్ష నివేదిక (Soil Health Card) అందుబాటులో ఉందా? 🙏")

        if not questions:
            return (
                "మీరు నాట్లలో DAP, SSP లేదా కాంప్లెక్స్ ఎరువులు ఏవీ వేయలేదని, యూరియా కూడా ఇంకా వేయలేదని, "
                "అలాగే వరి రకం పేరు తెలియదని మరియు నేల పరీక్ష నివేదిక అందుబాటులో లేదని అర్థమైంది.\n\n"
                "వరి రకం (స్వల్ప/దీర్ఘ కాలిక) మరియు నేల స్వభావం తెలియకుండా, ప్రాథమిక ఎరువులు వేయనప్పుడు నిర్దిష్ట రసాయన ఎరువుల మోతాదును "
                "ఖరారు చేయడం సురక్షితం కాదు; అధిక నత్రజని పంట నేలకొరిగే ప్రమాదాన్ని, అగ్గి తెగులు మరియు కాండం తొలిచే పురుగుల ఉధృతిని పెంచుతుంది.\n\n"
                "తదుపరి సురక్షిత చర్యలు:\n"
                "1. నిర్ధారించని అధిక మోతాదులో యూరియాను ఒకేసారి వేయకండి.\n"
                "2. మీ పొలం మట్టి స్వభావం మరియు ప్రస్తుత పిలకల దశను బట్టి సరైన దిద్దుబాటు పోషక యాజమాన్యం కొరకు స్థానిక వ్యవసాయ విస్తరణ అధికారి (AEO) "
                "లేదా సమీప కృషి విజ్ఞాన కేంద్రం (KVK) నిపుణులను సంప్రదించి క్షేత్ర స్థాయి సలహా తీసుకోండి. 🙏"
            )

        q_block = "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions))

        return (
            "వరి పంట 30 రోజుల పిలకల దశలో ప్రాంతీయ వ్యవసాయ విస్తరణ మార్గదర్శకాల ప్రకారం "
            "సాధారణ ప్రచురిత రిఫరెన్స్ మాత్రమే, ఇది మీ పొలానికి ఖరారైన నిర్ధారణ కాదు. నాట్ల మరియు మునుపటి ఎరువుల వివరాలు లేకుండా "
            "అదనపు మోతాదును నిర్ధారించలేము; అధిక నత్రజని పంట నేలకొరిగే ప్రమాదాన్ని, అగ్గి తెగులు మరియు కాండం తొలిచే పురుగుల ఉధృతిని పెంచుతుంది.\n\n"
            "సరైన మోతాదును నిర్ణయించడానికి దయచేసి ఈ వివరాలు తెలపండి:\n"
            f"{q_block}"
        )
    elif user_lang == "hi":
        questions = []
        if dap_neg:
            if not ssp_neg and not ssp_applied and not general_fert_neg:
                questions.append("रोपाई के समय क्या SSP या अन्य कॉम्प्लेक्स खाद डाली थी, और कितनी मात्रा में?")
        elif not has_basal and not general_fert_neg and not ssp_neg:
            questions.append("रोपाई के समय कौन सी बेसल खाद (DAP, SSP या कॉम्प्लेक्स) डाली थी, और कितनी मात्रा में?")

        if not urea_neg and not has_prior_urea and not general_fert_neg:
            questions.append("रोपाई के बाद क्या कोई यूरिया या पोटाश डाला गया है (मात्रा और तारीख/दशा)?")

        if not has_variety and not var_unk:
            questions.append("आप धान की कौन सी किस्म (कम अवधि या लंबी अवधि) उगा रहे हैं?")

        if not has_soil and not soil_neg:
            questions.append("क्या मृदा परीक्षण रिपोर्ट (Soil Health Card) उपलब्ध है? 🙏")

        if not questions:
            return (
                "यह समझ आ गया कि रोपाई के समय DAP, SSP या कोई कॉम्प्लेक्स खाद नहीं डाली गई थी, अभी तक यूरिया भी नहीं डाला गया है, "
                "धान की किस्म की जानकारी नहीं है और मृदा परीक्षण रिपोर्ट भी उपलब्ध नहीं है।\n\n"
                "किस्म की अवधि और मिट्टी की स्थिति जाने बिना तथा बिना बेसल खाद के विशिष्ट रासायनिक खाद की मात्रा "
                "तय करना सुरक्षित नहीं है; अत्यधिक नाइट्रोजन से फसल गिरने का खतरा तथा झुलसा (ब्लास्ट) व तना छेदक कीट का प्रकोप बढ़ जाता है।\n\n"
                "सुरक्षित कदम:\n"
                "1. एक साथ अधिक मात्रा में यूरिया न डालें।\n"
                "2. अपने खेत की मिट्टी और वर्तमान कल्ले फूटने की स्थिति के अनुसार सही सलाह के लिए स्थानीय कृषि विस्तार अधिकारी (AEO) "
                "या निकटतम कृषि विज्ञान केंद्र (KVK) से संपर्क करें। 🙏"
            )

        q_block = "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions))

        return (
            "धान की फसल में 30 दिन की कल्ले फूटने की अवस्था पर क्षेत्रीय कृषि विस्तार दिशा-निर्देशों के अनुसार "
            "केवल एक सामान्य प्रकाशित संदर्भ (रेफरेंस) है, यह आपके खेत के लिए अंतिम सिफारिश नहीं है। पिछली खाद की जानकारी के बिना "
            "अतिरिक्त मात्रा तय नहीं की जा सकती; अधिक नाइट्रोजन से फसल गिरने का खतरा तथा झुलसा (ब्लास्ट) व तना छेदक कीट का प्रकोप बढ़ जाता है।\n\n"
            "सटीक मात्रा तय करने के लिए कृपया बताएं:\n"
            f"{q_block}"
        )
    else:
        questions = []
        if dap_neg:
            if not ssp_neg and not ssp_applied and not general_fert_neg:
                questions.append("Whether SSP or another complex fertilizer was applied at transplanting, and in what quantity?")
        elif not has_basal and not general_fert_neg and not ssp_neg:
            questions.append("Which basal fertilizers (DAP, SSP, or complex) were applied at transplanting, and in what quantities?")

        if not urea_neg and not has_prior_urea and not general_fert_neg:
            questions.append("Has any urea or potash already been applied after transplanting (amounts and dates/crop stages)?")

        if not has_variety and not var_unk:
            questions.append("Which rice variety or duration group (short, medium, or long duration) are you cultivating?")

        if not has_soil and not soil_neg:
            questions.append("Do you have a soil test report or Soil Health Card? 🙏")

        if not questions:
            return (
                "Understood that no basal fertilizers (DAP, SSP, or complex) were applied at transplanting, "
                "urea has not been applied so far, the paddy variety is unknown, and no soil test report is available.\n\n"
                "Without knowing the variety duration and soil fertility status, and with no basal fertilizer history, "
                "confirming a field-specific chemical fertilizer dose is not safe. Excess or unbalanced nitrogen increases "
                "risks of crop lodging, blast disease, and stem borer attack.\n\n"
                "Safe next steps:\n"
                "1. Do not apply heavy or unverified doses of urea at once.\n"
                "2. Please consult your local Agricultural Extension Officer (AEO) or nearest Krishi Vigyan Kendra (KVK) "
                "for an on-field soil and crop assessment to receive tailored, safe nutrient recommendations for your field. 🙏"
            )

        q_block = "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions))

        return (
            "For paddy at the 30-day tillering stage, regional agricultural extension guidelines provide "
            "strictly as an attributed general extension reference benchmark, not a confirmed field-specific recommendation. "
            "An additional fertilizer dose cannot be confirmed without knowing prior fertilizer applications and variety duration, "
            "as excess nitrogen increases risks of crop lodging, blast disease, and stem borer attack.\n\n"
            "To determine the correct field recommendation, please share:\n"
            f"{q_block}"
        )


class AIService:
    def __init__(self, repository: AIRepository):
        self.repository = repository

    async def generate_ai_response(self, request: AIGenerateRequest) -> AIGenerateResponse:
        service_start_time = time.time()
        rag_snippets = []
        is_dosage_req = False
        user_lang = "te"
        try:
            # 1. Fetch farmer profile for context
            profile = await self.repository.get_farmer_profile(request.farmer_id)

            # 2. Fetch conversation history (oldest first for Gemini, max 3 DB turns / 6 messages)
            history_records = await self.repository.get_conversation_history(request.farmer_id, limit=3)
            history_records.reverse()
            history = []
            recent_context_crop = None

            from src.rag.service import extract_crop_from_text
            query_crop = extract_crop_from_text(request.message)

            for record in history_records:
                if record.user_message:
                    history.append({"role": "user", "parts": record.user_message})
                    c = extract_crop_from_text(record.user_message)
                    if c:
                        recent_context_crop = c
                if record.ai_response:
                    # Clean all structured enrichment sections from history to prevent LLM prompt contamination
                    clean_response = record.ai_response
                    for marker in ["Available Nearby Shops:", "🏬", "📊", "🌤️", "🌡️", "🏛️", "🎫", "📜", "👨‍🌾", "🚨", "🆘"]:
                        clean_response = clean_response.split(marker)[0]
                    clean_response = clean_response.strip()
                    if clean_response:
                        history.append({"role": "model", "parts": clean_response})

            # Priority for crop:
            # 1. Explicit crop mentioned in current message (e.g. "టమాటా" / Tomato overrides profile's "Cotton")
            # 2. If short follow-up and query_crop is None, crop from recent conversation history
            # 3. Farmer profile current_crop
            effective_crop = query_crop or (recent_context_crop if recent_context_crop else (getattr(profile, "current_crop", None) if profile else None))

            # 3. Fetch farmer long-term memory context
            from src.memory.service import FarmerMemoryService
            from src.memory.repository import FarmerMemoryRepository
            mem_repo = FarmerMemoryRepository(self.repository.session)
            mem_service = FarmerMemoryService(mem_repo)
            memory_obj = await mem_service.get_memory(request.farmer_id)
            memory_context = await mem_service.format_memory_for_system_prompt(request.farmer_id)

            # Location Priority for farmer context:
            # 1. Explicit location in current user message (highest priority)
            # 2. If memory has active GPS coordinates, treat GPS as authoritative current location; do not let older profile district override
            # 3. Farmer profile district (fallback only when no GPS and no explicit location in query)
            from src.weather.service import _extract_district_from_query
            query_district = _extract_district_from_query(request.message)

            has_gps = False
            if memory_obj and memory_obj.gps_coordinates and isinstance(memory_obj.gps_coordinates, dict):
                lat = memory_obj.gps_coordinates.get("latitude")
                lon = memory_obj.gps_coordinates.get("longitude")
                if lat is not None and lon is not None and not (lat == 0.0 and lon == 0.0):
                    has_gps = True

            effective_district = None
            if query_district:
                effective_district = query_district
            elif not has_gps:
                profile_dist = getattr(profile, "district", None) if profile else None
                if profile_dist:
                    effective_district = profile_dist
                elif memory_obj and getattr(memory_obj, "district", None):
                    effective_district = memory_obj.district

            # Land size detection from query or profile
            query_land_size = extract_land_size_from_text(request.message)
            profile_land_size = getattr(profile, "land_size_acres", None) if profile else None
            effective_land_size = query_land_size if query_land_size is not None else profile_land_size

            # Build farmer context string
            farmer_context = build_farmer_context(
                crop=effective_crop or (getattr(profile, "current_crop", None) if profile else None),
                district=effective_district,
                state=getattr(profile, "state", None) if profile else None,
                land_size=effective_land_size,
                gps_coordinates=memory_obj.gps_coordinates if (has_gps and memory_obj) else None,
            )

            # Build enriched RAG query for short follow-ups (e.g. "ఎకరానికి ఎంత కావాలి?" / "ఈ వ్యాధికి ఎంత మందు వేయాలి?")
            rag_query = request.message
            msg_tokens = request.message.strip().split()
            if not query_crop and len(msg_tokens) <= 7 and history_records:
                recent_user_msgs = [r.user_message for r in history_records[-2:] if r.user_message]
                if recent_user_msgs:
                    rag_query = f"{' '.join(recent_user_msgs)} {request.message}"

            # 3.5 Retrieve trusted agricultural RAG knowledge
            rag_snippets = []
            rag_context_text = ""
            from src.language.detector import detect_language
            user_lang = detect_language(request.message, fallback=getattr(profile, "preferred_language", "te") or "te")
            is_dosage_req = is_dosage_sensitive_query(request.message)

            try:
                from src.rag.service import RAGService
                from src.rag.repository import RAGRepository
                rag_repo = RAGRepository(self.repository.session)
                rag_service = RAGService(rag_repo)
                rag_top_k = 3 if is_dosage_req else 2
                rag_results = await rag_service.search_knowledge(
                    query=rag_query,
                    top_k=rag_top_k,
                    state=getattr(profile, "state", None) if profile else None,
                    crop=effective_crop,
                )
                if rag_results:
                    if not is_dosage_req:
                        # Conservative RAG compaction for general non-dosage farming questions (max 2 chunks, char-capped)
                        rag_snippets = []
                        total_rag_chars = 0
                        for r in rag_results[:2]:
                            text = r.chunk_text.strip()
                            if len(text) > 650:
                                text = text[:650] + "..."
                            snippet = f"• Document: {r.document_title} (Crop: {r.crop or 'General'}, Source: {r.source}): {text}"
                            if total_rag_chars + len(snippet) <= 1600:
                                rag_snippets.append(snippet)
                                total_rag_chars += len(snippet)
                            else:
                                break
                    else:
                        # Full verified Ground Truth for dosage-sensitive queries to preserve exact verified figures
                        rag_snippets = [f"• Document: {r.document_title} (Crop: {r.crop or 'General'}, Source: {r.source}): {r.chunk_text}" for r in rag_results]

                    if rag_snippets:
                        rag_context_text = (
                            "=== RETRIEVED TRUSTED AGRICULTURAL KNOWLEDGE (GROUND TRUTH) ===\n"
                            "CRITICAL INSTRUCTION: The following knowledge is verified agronomic ground truth. You MUST strictly use ONLY these verified treatments, products, and numeric dosages. "
                            "For fertilizer guidance: Identify whether fertilizer history (basal DAP/SSP/complex, prior urea/potash) and variety duration are already known. "
                            "If essential fertilizer history is missing, do NOT instruct the farmer to apply fertilizer or present any quantity as a confirmed field recommendation. "
                            "Explain published figures strictly as attributed general extension reference benchmarks, explain risks of excess nitrogen (lodging, blast, stem borer), and ask for the missing basal, prior urea, variety, and soil-test details. "
                            "Do NOT add, alter, extrapolate, or invent different dosages or unverified chemicals. Translate directly into the farmer's language:\n"
                            + "\n".join(rag_snippets)
                        )
            except Exception as rag_err:
                logger.warning(f"RAG knowledge retrieval warning: {rag_err}")

            # 3.8 HARD GROUNDING GATE: Block unverified numeric dosage generation
            has_grounding = has_relevant_dosage_ground_truth(request.message, rag_snippets)

            query_topics = extract_query_dosage_topics(request.message)
            is_fertilizer_only_query = query_topics["is_fertilizer_query"] and not query_topics["is_pest_disease_query"]

            if is_dosage_req and not has_grounding:
                logger.warning(
                    f"[HARD GROUNDING GATE TRIGGERED] Dosage-sensitive query lacking relevant verified Ground Truth. "
                    f"Farmer ID: {request.farmer_id}, Language: '{user_lang}', Is Fertilizer: {is_fertilizer_only_query}, Query: '{request.message[:80]}...'"
                )
                safe_fallback = get_unverified_dosage_fallback_response(user_lang, is_fertilizer=is_fertilizer_only_query)
                return AIGenerateResponse(
                    response_text=safe_fallback,
                    intent="dosage_unverified_fallback",
                    confidence=1.0,
                    provider_used="hard_grounding_gate",
                )

            # 4. Build target-language directive and system prompt combining profile, memory engine, and RAG ground truth
            lang_directive = build_target_language_directive(
                user_lang,
                default=getattr(profile, "preferred_language", "te") or "te"
            )
            full_system_prompt = f"{BHOOMIMITRA_SYSTEM_PROMPT}\n\n{lang_directive}\n\n{farmer_context}\n\n{memory_context}"
            if rag_context_text:
                full_system_prompt += f"\n\n{rag_context_text}"

            is_paddy_tillering_fert = is_paddy_tillering_fertilizer_query(request.message, crop=effective_crop, history_records=history_records)
            fert_hist_info = check_fertilizer_history_details(request.message, history_records)
            if is_paddy_tillering_fert and not fert_hist_info["has_essential_history"]:
                history_known_notes = []
                missing_inquiry_points = []
                if fert_hist_info.get("dap_negated"):
                    history_known_notes.append("- The farmer ALREADY stated that DAP was NOT applied at transplanting. Do NOT ask if DAP was applied.")
                    if not fert_hist_info.get("has_basal_history") and not fert_hist_info.get("ssp_or_complex_negated") and not fert_hist_info.get("ssp_or_complex_applied"):
                        missing_inquiry_points.append("- Whether SSP or another complex fertilizer was applied at transplanting, and in what quantity.")
                elif not fert_hist_info.get("has_basal_history") and not fert_hist_info.get("ssp_or_complex_negated") and not fert_hist_info.get("general_fert_negated"):
                    missing_inquiry_points.append("- Basal DAP, SSP, and complex fertilizers, including quantities applied.")

                if fert_hist_info.get("ssp_or_complex_negated"):
                    history_known_notes.append("- The farmer ALREADY stated that SSP or complex fertilizer was NOT applied at transplanting. Do NOT ask if SSP or complex fertilizer was applied.")

                if fert_hist_info.get("urea_negated"):
                    history_known_notes.append("- The farmer ALREADY stated that urea has NOT been applied so far. Do NOT ask if urea was applied.")
                elif not fert_hist_info.get("has_prior_urea_or_potash") and not fert_hist_info.get("general_fert_negated"):
                    missing_inquiry_points.append("- Urea or potash applied after transplanting, including amounts and dates or crop stages.")

                if fert_hist_info.get("variety_unknown"):
                    history_known_notes.append("- The farmer ALREADY stated that they do NOT know the variety or duration. Do NOT ask for variety or duration.")
                elif not fert_hist_info.get("has_variety_info"):
                    missing_inquiry_points.append("- Rice variety / duration group (short, medium, or long duration).")

                if fert_hist_info.get("soil_test_negated"):
                    history_known_notes.append("- The farmer ALREADY stated that a soil test report is NOT available. Do NOT ask for a soil test report or Soil Health Card.")
                elif not fert_hist_info.get("has_soil_test_info"):
                    missing_inquiry_points.append("- Soil-test information (Soil Health Card) when available.")

                notes_block = ("\nKNOWN CONTEXT (DO NOT RE-ASK):\n" + "\n".join(history_known_notes)) if history_known_notes else ""
                inquiry_block = ("\nMISSING DETAILS TO ASK ABOUT:\n" + "\n".join(missing_inquiry_points)) if missing_inquiry_points else ""

                benchmark_instruction = (
                    "1. Do NOT recommend, calculate, prescribe, or include ANY numerical fertilizer dose benchmarks (such as 30–35 kg/acre or 60–70 kg for 2 acres) now, "
                    "because essential fertilizer history and variety information are missing and the farmer can easily mistake benchmark numbers for an immediate instruction to apply.\n"
                    "2. If the farmer asks you to guess, estimate, or assume a dose without providing context, explain that guessing chemical fertilizer doses without knowing variety duration, basal fertilizer history, and soil conditions is unsafe, and direct them to local AEO or KVK.\n"
                )

                if fert_hist_info.get("all_clarifications_answered") or not missing_inquiry_points:
                    inquiry_instruction = (
                        "3. All clarification questions have been answered by the farmer (no DAP, no SSP/complex, no urea applied, variety unknown, no soil test report).\n"
                        "4. Do NOT ask ANY questions or repeat previous questions. Acknowledge all the answers provided.\n"
                        "5. Clearly explain that without variety duration and soil test data, and with no basal fertilizer applied, an additional field-specific chemical dose cannot be confirmed or prescribed.\n"
                        "6. Explain that excess nitrogen increases risks of lodging (నేలకొరిగే ప్రమాదం), blast disease (అగ్గి తెగులు), and stem borer (కాండం తొలిచే పురుగు).\n"
                        "7. Explain the next safe steps: do NOT apply heavy unverified urea doses at once; advise consulting the local Agricultural Extension Officer (AEO) or Krishi Vigyan Kendra (KVK) for an on-field soil and crop assessment.\n"
                    )
                else:
                    inquiry_instruction = (
                        "3. Clearly explain that without knowing prior fertilizer applications and variety, an additional dose cannot be confirmed or recommended yet, "
                        "because duplicate or excess nitrogen increases risks of crop lodging (నేలకొరిగే ప్రమాదం) and susceptibility to blast disease (అగ్గి తెగులు) and stem borer (కాండం తొలిచే పురుగు).\n"
                        f"4. Ask the farmer ONLY about genuinely missing details. Do NOT repeat questions for information already provided.{notes_block}{inquiry_block}\n"
                        "5. Do NOT automatically tell every farmer to apply more urea or to skip the next application."
                    )

                full_system_prompt += (
                    "\n\n=== MANDATORY PADDY TILLERING FERTILIZER RULE ===\n"
                    "The farmer is asking what additional fertilizer to apply for paddy at around 30 days after transplanting (tillering stage), "
                    "but essential fertilizer history (basal DAP/SSP/complex quantities, prior urea/potash applied, variety duration, soil test) is missing.\n"
                    "MANDATORY REQUIREMENTS:\n"
                    f"{benchmark_instruction}"
                    f"{inquiry_instruction}"
                )

            is_unconfirmed_diag = is_unconfirmed_crop_diagnosis(request.message, history_records=history_records)
            if is_unconfirmed_diag:
                full_system_prompt += (
                    "\n\n=== UNCONFIRMED DIAGNOSIS SAFETY DIRECTIVE ===\n"
                    "The diagnosis is unconfirmed or conditional, or the farmer cannot provide a crop photo.\n"
                    "MANDATORY REQUIREMENTS:\n"
                    "1. You MUST NOT prescribe, recommend, or mention ANY pesticide, fungicide, or chemical dosages (e.g. NEVER mention Mancozeb 2.5-3 g/L or any dosage).\n"
                    "2. You MUST state clearly in the farmer's language:\n"
                    "   In Telugu: 'ఖచ్చితమైన నిర్ధారణ లేకుండా మందు మోతాదును సూచించను. దయచేసి పంట ఫోటోను పంపండి లేదా వ్యవసాయ అధికారిని సంప్రదించండి.'\n"
                    "   In Hindi: 'सटीक पुष्टि के बिना दवा की खुराक नहीं बता सकते। कृपया फसल की फोटो भेजें या कृषि अधिकारी से संपर्क करें।'\n"
                    "   In English: 'Without a confirmed diagnosis, chemical dosages cannot be recommended. Please share a crop photo or consult your local agriculture officer.'\n"
                    "3. Explain the possible reasons conceptually (e.g., fungal leaf spots or nutrient deficiency) without prescribing specific chemical doses.\n"
                    "4. Advise consulting the local Agricultural Extension Officer (AEO) or visiting the nearest Rythu Vedika / Krishi Vigyan Kendra.\n"
                )

            # 6. Call Gemini API
            logger.info(f"Processing AI request for farmer {request.farmer_id}: '{request.message[:80]}...'")

            ai_text = await generate_response(
                system_prompt=full_system_prompt,
                conversation_history=history,
                user_message=request.message,
                timeout_seconds=getattr(get_settings(), "gemini_api_timeout_seconds", 15.0),
            )

            if ai_text is None or not ai_text.strip():
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="AI provider timed out or returned empty response."
                )

            # 6.5 Defense-in-depth: Ensure Gemini did not invent ungrounded numeric dosage patterns
            dosage_pattern = (
                r'\b\d+(?:[.,]\d+)?\s*(?:-\s*\d+(?:[.,]\d+)?)?\s*'
                r'(?:kg|kgs|g|gm|gms|grams?|ml|litres?|ltr|l|కేజీలు|కేజీల|కేజీ|కిలోలు|కిలోల|కిలో|గ్రాములు|గ్రాముల|గ్రాము|మి\.లీ|లీటర్లు|లీటర్ల|లీటరు|किलो|ग्राम|मिली|लीटर)'
                r'(?:\s*(?:per|\/|ప్రతి|ఎకరాకి|ఎకరానికి|प्रति|లీటరుకు|लीटर)\s*(?:acre|hectare|ha|l(?:itre)?|లీటరు|एकड़)?)?'
                r'(?:\s*(?:urea|dap|mop|zinc|sulphate|potash|fertilizer|యూరియా|ఎరువు|జింక్|यूरिया|खाद))?'
            )
            output_dosage_matches = re.findall(dosage_pattern, ai_text, re.IGNORECASE)

            if output_dosage_matches:
                if not has_grounding:
                    logger.warning(
                        f"[DEFENSE-IN-DEPTH GATE TRIGGERED] Unverified dosage generated in ungrounded output for farmer {request.farmer_id}. "
                        f"Replacing with safe fallback."
                    )
                    if is_paddy_tillering_fert and not fert_hist_info["has_essential_history"]:
                        ai_text = build_unconfirmed_paddy_fertilizer_guidance(user_lang, acres=effective_land_size or 2.0, fert_hist_info=fert_hist_info)
                    else:
                        ai_text = get_unverified_dosage_fallback_response(user_lang, is_fertilizer=is_fertilizer_only_query, fert_hist_info=fert_hist_info)
                else:
                    # Verify generated numeric figures exist in the provided RAG ground truth or are valid area-scaled amounts
                    rag_text_lower = "\n".join(rag_snippets).lower() if rag_snippets else rag_context_text.lower()
                    generated_numbers = re.findall(r'\b\d+(?:[.,]\d+)?\b', " ".join(output_dosage_matches))
                    rag_numbers = set(re.findall(r'\b\d+(?:[.,]\d+)?\b', rag_text_lower))

                    # Allowed numbers: RAG numbers + user query numbers (e.g. 2 acres, 30 days) + acreage-scaled totals
                    valid_numbers = set(rag_numbers)
                    query_numbers = set(re.findall(r'\b\d+(?:[.,]\d+)?\b', request.message))
                    valid_numbers.update(query_numbers)

                    if effective_land_size and effective_land_size > 0:
                        for n_str in rag_numbers:
                            try:
                                val = float(n_str)
                                scaled_val = val * effective_land_size
                                if scaled_val.is_integer():
                                    valid_numbers.add(str(int(scaled_val)))
                                valid_numbers.add(f"{scaled_val:.1f}")
                                valid_numbers.add(f"{scaled_val:.2f}")
                                valid_numbers.add(str(scaled_val))
                            except ValueError:
                                pass

                    unverified_numbers = [n for n in generated_numbers if n not in valid_numbers]
                    if unverified_numbers:
                        logger.warning(
                            f"[DEFENSE-IN-DEPTH GATE TRIGGERED] Hallucinated dosage numbers {unverified_numbers} not found in RAG ground truth for farmer {request.farmer_id}. "
                            f"Replacing with safe fallback."
                        )
                        ai_text = get_unverified_dosage_fallback_response(user_lang, is_fertilizer=is_fertilizer_only_query, fert_hist_info=fert_hist_info)
            elif is_dosage_req and not has_grounding:
                ai_text = get_unverified_dosage_fallback_response(user_lang, is_fertilizer=is_fertilizer_only_query, fert_hist_info=fert_hist_info)

            # 6.6 Safety Guardrail: Prevent unconfirmed additional fertilizer dosage or premature application instruction
            if is_paddy_tillering_fert and not fert_hist_info["has_essential_history"]:
                has_premature_dose_benchmark = bool(re.search(
                    r'\b(?:60\s*(?:-|to|నుండి|–)\s*70|30\s*(?:-|to|నుండి|–)\s*35)\s*(?:kg|kgs|కేజీల|కిలోల|కిలోలు)?\b',
                    ai_text,
                    re.IGNORECASE
                ))
                proposes_specific_fertilizer_dose = bool(re.search(
                    r'\b\d+(?:\s*(?:-|to|నుండి|–)\s*\d+)?\s*(?:kg|kgs|కేజీల|కిలోల|కిలోలు)\s*(?:యూరియా|urea|dap|డిఎపి|ssp|పొటాష్|potash|mop)\b'
                    r'|(?:యూరియా|urea|dap|ssp|పొటాష్|potash|mop)\s*(?:సుమారు\s*)?\b\d+(?:\s*(?:-|to|నుండి|–)\s*\d+)?\s*(?:kg|kgs|కేజీల|కిలోల|కిలోలు)\b',
                    ai_text,
                    re.IGNORECASE
                ))
                instructs_immediate_apply = bool(re.search(
                    r'(?:60\s*(?:-|to|నుండి|–)\s*70|30\s*(?:-|to|నుండి|–)\s*35)?\s*(?:kg|kgs|కేజీల|కిలోల|కిలోలు)?\s*(?:యూరియా|urea)?.*'
                    r'(?:సిఫార్సు\s*చేయబడుతుంది|సిఫారసు\s*చేయబడుతుంది|వేయండి|వేయాలి|వేసుకోవాలి|చల్లుకోండి|చల్లాలి|apply\s+now|is\s+recommended\s+to\s+apply)',
                    ai_text,
                    re.IGNORECASE
                ))
                confirms_without_qualification = (
                    ("సిఫార్సు చేయబడుతుంది" in ai_text or "వేయాలి" in ai_text or "వేయండి" in ai_text or "వేసుకోవాలి" in ai_text)
                    and not any(w in ai_text for w in ["రిఫరెన్స్", "నిర్ధారణ కాదు", "ఖరారైన నిర్ధారణ కాదు", "నిర్ధారించలేము", "ఖరారు చేయలేము", "reference only", "not confirmed", "cannot confirm"])
                )

                repeats_answered_questions = False
                if fert_hist_info.get("dap_negated") and re.search(r'\bdap\b.*(?:వేశారా|వేయండి|applied|did\s+you\s+apply)', ai_text, re.I):
                    repeats_answered_questions = True
                if fert_hist_info.get("ssp_or_complex_negated") and re.search(r'(?:ssp|కాంప్లెక్స్|complex).*(?:వేశారా|వేయండి|applied|did\s+you\s+apply)', ai_text, re.I):
                    repeats_answered_questions = True
                if fert_hist_info.get("urea_negated") and re.search(r'(?:యూరియా|urea).*(?:వేశారా|already\s+applied|already\s+been\s+applied)', ai_text, re.I):
                    repeats_answered_questions = True
                if fert_hist_info.get("variety_unknown") and re.search(r'(?:రకం|వంగడం|variety|duration).*(?:ఏది|తెలుపండి|which|what\s+variety)', ai_text, re.I):
                    repeats_answered_questions = True
                if fert_hist_info.get("soil_test_negated") and re.search(r'(?:సాయిల్\s*టెస్ట్|భూసార\s*పరీక్ష|నేల\s*పరీక్ష|soil\s*test|soil\s*health\s*card).*(?:ఉందా|available|report)', ai_text, re.I):
                    repeats_answered_questions = True
                if fert_hist_info.get("all_clarifications_answered") and ("?" in ai_text or "తెలపండి" in ai_text or "తెలుపండి" in ai_text or "please share" in ai_text.lower() or "please tell" in ai_text.lower()):
                    repeats_answered_questions = True

                if has_premature_dose_benchmark or proposes_specific_fertilizer_dose or instructs_immediate_apply or confirms_without_qualification or repeats_answered_questions:
                    logger.warning(
                        f"[FERTILIZER GUIDANCE GUARD TRIGGERED] Unconfirmed paddy dosage, premature instruction, or repeated question detected for farmer {request.farmer_id}. "
                        f"Replacing with compliant safe guidance."
                    )
                    ai_text = build_unconfirmed_paddy_fertilizer_guidance(user_lang, acres=effective_land_size or 2.0, fert_hist_info=fert_hist_info)

            # 6.7 Unconfirmed / Conditional Diagnosis Dosage Gate
            if is_unconfirmed_diag:
                has_any_dosage = bool(re.search(dosage_pattern, ai_text, re.IGNORECASE))
                chem_words = [
                    "mancozeb", "మాంకోజెబ్", "propiconazole", "ప్రొపికొనజోల్", "profenofos", "ప్రొఫెనోఫాస్",
                    "fipronil", "ఫిప్రోనిల్", "spinosad", "స్పైనోసాడ్", "chlorantraniliprole", "క్లోరాంట్రానిలిప్రోల్",
                    "acetamiprid", "అసిటామప్రిడ్", "spinetoram", "స్పైనిటోరం", "cartap", "కార్టాప్",
                    "tricyclazole", "ట్రైసైక్లాజోల్", "hexaconazole", "హెక్సాకొనజోల్", "diafenthiuron", "డయాఫెంథియురాన్",
                    "carbendazim", "కార్బెండజిమ్", "copper oxychloride", "కాపర్ ఆక్సిక్లోరైడ్",
                    "మందును పిచికారీ", "పిచికారీ చేయండి", "పిచికారీ చేసుకోండి", "లీటరు నీటికి", "పురుగుల మందు",
                    "కీటకనాశిని", "శిలీంద్రనాశిని", "రసాయన మందు", "pesticide", "fungicide", "insecticide"
                ]
                has_chem_keyword = any(k in ai_text.lower() for k in chem_words)
                if has_any_dosage or has_chem_keyword:
                    logger.warning(
                        f"[UNCONFIRMED DIAGNOSIS GATE] Blocked pesticide dosage for unconfirmed diagnosis. Farmer: {request.farmer_id}."
                    )
                    combined_u_hist = ""
                    if history_records:
                        for r in history_records[-3:]:
                            u_msg = getattr(r, "user_message", None) or (r.get("parts") if isinstance(r, dict) and r.get("role") == "user" else "")
                            combined_u_hist += " " + (u_msg or "").lower()
                    is_photo_neg = bool(re.search(r"(?:ఫోటో|చిత్రం|photo).*?(?:కుదరదు|లేదు|పంపలేను|cannot|not|వీలు|ఇవ్వలేను|సాధ్యం)", request.message + " " + combined_u_hist, re.I))
                    if is_photo_neg or "కుదరదు" in request.message or "లేదు" in request.message:
                        if user_lang == "te":
                            ai_text = "ఖచ్చితమైన నిర్ధారణ లేకుండా మందు మోతాదును సూచించను. దయచేసి పంట ఫోటోను పంపండి లేదా వ్యవసాయ అధికారిని సంప్రదించండి."
                        elif user_lang == "hi":
                            ai_text = "सटीक पुष्टि के बिना दवा की खुराक नहीं बता सकते। कृपया फसल की फोटो भेजें या कृषि अधिकारी से संपर्क करें।"
                        else:
                            ai_text = "Without a confirmed diagnosis, chemical dosages cannot be recommended. Please share a crop photo or consult your local agriculture officer."
                    else:
                        crop_name_te = "పంట"
                        if effective_crop:
                            if any(k in effective_crop.lower() for k in ["cotton", "పత్తి"]):
                                crop_name_te = "పత్తి"
                            elif any(k in effective_crop.lower() for k in ["paddy", "rice", "వరి"]):
                                crop_name_te = "వరి"
                            elif any(k in effective_crop.lower() for k in ["chilli", "mirchi", "మిరప"]):
                                crop_name_te = "మిరప"
                        if user_lang == "te":
                            ai_text = f"{crop_name_te} ఆకులపై పసుపు మరియు గోధుమ మచ్చలు ఆల్టర్నేరియా ఆకుమచ్చ తెగులు లేదా పోషకాల లోపం వల్ల రావచ్చు. ఖచ్చితమైన నిర్ధారణ లేకుండా మందు మోతాదును సూచించను. దయచేసి ఆకులపై ఏకకేంద్ర వలయాలు ఉన్నాయో గమనించండి లేదా స్పష్టమైన పంట ఆకుల ఫోటో పంపండి."
                        elif user_lang == "hi":
                            ai_text = "फसल की पत्तियों पर पीले व भूरे धब्बे कई कारणों से हो सकते हैं। सटीक पुष्टि के बिना दवा की खुराक नहीं बता सकते। कृपया पत्तियों की स्पष्ट फोटो भेजें या कृषि अधिकारी से संपर्क करें।"
                        else:
                            ai_text = "Without confirmed diagnosis, chemical dosage cannot be recommended. Please share a clear photo of the affected leaves or consult your local agriculture officer."

            # 6.8 Universal Chemical & Dosage Validation Gate
            is_chem_valid, chem_reason = validate_output_chemical_dosage_safety(
                ai_text,
                rag_snippets,
                effective_crop=effective_crop
            )
            if not is_chem_valid:
                logger.warning(f"[CHEMICAL DOSAGE SAFETY GATE TRIGGERED] {chem_reason}. Replacing with safe fallback.")
                ai_text = get_unverified_dosage_fallback_response(
                    user_lang,
                    is_fertilizer=is_fertilizer_only_query,
                    fert_hist_info=fert_hist_info
                )

            # 6.9 Safe Closing Sanitization: Ensure model never asks farmer to choose among pesticides
            if re.search(r'(?:మీరు\s*ఏ\s*మందు\s*పిచికారీ\s*చేయాలనుకుంటున్నారు|ఏ\s*మందు\s*పిచికారీ\s*చేయాలనుకుంటున్నారు|which\s+(?:medicine|pesticide|spray)\s+(?:do|would)\s+you\s+(?:want|like)\s+to\s+spray)', ai_text, re.I):
                if user_lang == "te":
                    safe_close = "మీ పంట పరిస్థితిని బట్టి సరైన మందును ఎంపిక చేసుకునే ముందు స్థానిక వ్యవసాయ అధికారిని సంప్రదించండి. అవసరమైతే మీ పంట ఆకుల ఫోటో పంపండి."
                elif user_lang == "hi":
                    safe_close = "अपनी फसल की स्थिति के अनुसार सही दवा चुनने से पहले स्थानीय कृषि अधिकारी से सलाह लें। यदि आवश्यक हो तो अपनी फसल की पत्तियों की फोटो भेजें।"
                else:
                    safe_close = "Before selecting the appropriate chemical depending on your crop condition, please consult your local Agriculture Extension Officer. If needed, send a photo of your crop leaves."
                ai_text = re.sub(
                    r'(?:మీరు\s*ఏ\s*మందు\s*పిచికారీ\s*చేయాలనుకుంటున్నారు|ఏ\s*మందు\s*పిచికారీ\s*చేయాలనుకుంటున్నారు|which\s+(?:medicine|pesticide|spray)\s+(?:do|would)\s+you\s+(?:want|like)\s+to\s+spray)[\s\?]*',
                    safe_close,
                    ai_text,
                    flags=re.I
                )

            total_ai_time = time.time() - service_start_time
            logger.info(
                f"[AI SERVICE GENERATION SUCCESS]\n"
                f"  Farmer ID    : {request.farmer_id}\n"
                f"  Total Time   : {total_ai_time:.2f}s\n"
                f"  Output Chars : {len(ai_text)}\n"
                f"  Preview      : '{ai_text[:120]}...'"
            )

            # 7. Non-blocking automatic memory extraction in background
            try:
                from src.memory.service import trigger_background_memory_extraction
                trigger_background_memory_extraction(
                    farmer_id=request.farmer_id,
                    user_message=request.message,
                    ai_response=ai_text
                )
            except Exception as mem_err:
                logger.warning(f"Automatic memory extraction scheduling warning for farmer {request.farmer_id}: {mem_err}")

            # 8. Return structured response
            return AIGenerateResponse(
                response_text=ai_text,
                intent=None,
                confidence=None,
                provider_used="gemini"
            )

        except Exception as e:
            elapsed = time.time() - service_start_time
            logger.warning(
                f"[AI SERVICE GENERATION FAILED] Failed communicating with AI provider after {elapsed:.2f}s: {e}. "
                "Falling back to safe agricultural guidance or localized AEO fallback."
            )
            # If trusted RAG context is already retrieved and not dosage-sensitive, provide safe excerpt
            if rag_snippets and not is_dosage_req:
                safe_rag_excerpt = "\n".join(rag_snippets[:2])
                disclaimer = (
                    "\n\n(గమనిక: AI సేవ ప్రస్తుతం పరిమితంగా ఉంది. అధికారిక సలహా కోసం మీ స్థానిక వ్యవసాయ విస్తరణ అధికారిని (AEO) సంప్రదించండి.)"
                    if user_lang == "te"
                    else "\n\n(Note: AI service is temporarily limited. Please consult your local Agriculture Extension Officer (AEO) for official advice.)"
                )
                return AIGenerateResponse(
                    response_text=safe_rag_excerpt + disclaimer,
                    intent="rag_safe_fallback",
                    confidence=1.0,
                    provider_used="rag_grounding",
                )

            safe_fallback = get_fallback_response(user_lang)
            return AIGenerateResponse(
                response_text=safe_fallback,
                intent="ai_unavailable_fallback",
                confidence=1.0,
                provider_used="safe_fallback",
            )

async def process_text_message(
    db: AsyncSession,
    farmer: Farmer,
    conversation: Conversation,
) -> str:
    """
    Main orchestration entry point for incoming WhatsApp text messages.
    Uses AIDecisionEngine to classify intents, route to authoritative
    services, prevent LLM hallucinations, and assemble a polished, verified response.
    """
    from src.ai.decision_engine import get_decision_engine
    engine = get_decision_engine()
    return await engine.process_message(db, farmer, conversation)



def _finalize_whatsapp_response(response_text: str, max_chars: int = 1600) -> str:
    """
    Cleans, deduplicates, and optimizes the final outgoing WhatsApp message.

    Guarantees:
    - Eliminates redundant blank lines
    - Prevents multi-block bloat while strictly preserving Expert Escalation,
      Market Prices, Weather, and core agronomic advice
    - Never partially truncates safety, escalation, or structured blocks mid-sentence
    - Preserves numbers, prices, dosages, and units untouched
    """
    import re
    if not response_text or not response_text.strip():
        return ""

    text = re.sub(r'\n{3,}', '\n\n', response_text).strip()

    # If text is within comfortable limit, return directly
    if len(text) <= max_chars:
        return text

    blocks = [b.strip() for b in text.split("\n\n") if b.strip()]

    core_blocks = []
    enrichment_blocks = []

    def is_structured_block(block: str) -> bool:
        return any(m in block for m in ["👨‍🌾", "🚨", "🆘", "📊", "⚠️", "🌡️", "🌤️", "🌦️", "🏬", "🏛️", "📜", "🎫"])

    for b in blocks:
        if is_structured_block(b):
            enrichment_blocks.append(b)
        else:
            core_blocks.append(b)

    def _block_priority(block: str) -> int:
        if any(m in block for m in ["👨‍🌾", "🚨", "🆘"]):
            return 0
        if any(m in block for m in ["📊", "⚠️"]):
            return 1
        if any(m in block for m in ["🌡️", "🌤️", "🌦️"]):
            return 2
        if "🏬" in block:
            return 2
        if any(m in block for m in ["🏛️", "📜"]):
            return 4
        return 5

    enrichment_blocks.sort(key=_block_priority)

    # First allocate budget for high-priority structured blocks (escalation, market, weather)
    reserved_enrichments = []
    reserved_len = 0
    for eb in enrichment_blocks:
        priority = _block_priority(eb)
        # Always reserve escalation and market/weather if present
        if priority <= 2 or (reserved_len + len(eb) + 2 <= max_chars):
            reserved_enrichments.append(eb)
            reserved_len += len(eb) + 2

    # Remaining budget for core advice
    advice_budget = max(200, max_chars - reserved_len)
    final_core = []
    core_len = 0
    for cb in core_blocks:
        if core_len + len(cb) + 2 <= advice_budget:
            final_core.append(cb)
            core_len += len(cb) + 2
        else:
            rem = advice_budget - core_len
            if rem >= 100:
                # Safe sentence-level truncation
                candidate = cb[:rem]
                m = re.search(r'([.!?।॥\n])\s*(?!.*[.!?।॥\n])', candidate)
                if m and m.end() >= rem * 0.4:
                    safe_chunk = candidate[:m.end()].strip()
                else:
                    last_space = candidate.rfind(' ')
                    safe_chunk = candidate[:last_space].strip() if last_space > 0 else candidate.strip()
                if safe_chunk:
                    final_core.append(safe_chunk)
            break

    combined = final_core + reserved_enrichments
    return "\n\n".join(b for b in combined if b).strip()


async def process_image_message(
    db: AsyncSession,
    farmer: Farmer,
    conversation: Conversation,
    image_bytes: bytes,
    mime_type: str,
) -> str:
    """
    Multimodal pipeline: Takes an image and optional caption, queries Gemini Vision,
    and returns agronomic advice.
    """
    repo = AIRepository(db)

    # 1. Fetch farmer profile
    profile = await repo.get_farmer_profile(farmer.id)
    user_caption = (conversation.user_message or "").strip()

    from src.memory.service import FarmerMemoryService
    from src.memory.repository import FarmerMemoryRepository
    mem_repo = FarmerMemoryRepository(db)
    mem_service = FarmerMemoryService(mem_repo)
    memory_obj = await mem_service.get_memory(farmer.id)
    memory_context = await mem_service.format_memory_for_system_prompt(farmer.id)

    from src.weather.service import _extract_district_from_query
    query_district = _extract_district_from_query(user_caption) if user_caption else None

    has_gps = False
    if memory_obj and memory_obj.gps_coordinates and isinstance(memory_obj.gps_coordinates, dict):
        lat = memory_obj.gps_coordinates.get("latitude")
        lon = memory_obj.gps_coordinates.get("longitude")
        if lat is not None and lon is not None and not (lat == 0.0 and lon == 0.0):
            has_gps = True

    effective_district = None
    if query_district:
        effective_district = query_district
    elif not has_gps:
        if profile and profile.district:
            effective_district = profile.district
        elif memory_obj and getattr(memory_obj, "district", None):
            effective_district = memory_obj.district

    farmer_context = build_farmer_context(
        crop=getattr(profile, "current_crop", None) if profile else None,
        district=effective_district,
        state=getattr(profile, "state", None) if profile else None,
        land_size=getattr(profile, "land_size_acres", None) if profile else None,
        gps_coordinates=memory_obj.gps_coordinates if (has_gps and memory_obj) else None,
    )
    from src.language.detector import detect_language
    from src.language.languages import normalize_language_code
    img_lang = detect_language(user_caption, fallback=getattr(farmer, "preferred_language", "te") or "te") if user_caption else (getattr(farmer, "preferred_language", "te") or "te")
    img_lang_code = normalize_language_code(img_lang, default=getattr(farmer, "preferred_language", "te") or "te")
    lang_directive = build_target_language_directive(img_lang_code, default=getattr(farmer, "preferred_language", "te") or "te", is_multimodal=True)

    # 2. History & Active Crop Context Determination
    history_records = await repo.get_conversation_history(farmer.id)
    # history_records are ordered newest-first (descending)
    current_conv_id = getattr(conversation, "id", None)
    active_context = resolve_image_crop_context(
        user_caption,
        history_records,
        current_conversation_id=current_conv_id,
    )
    active_crop = active_context.get("crop") if active_context else None

    # Filter history specifically for Vision (prioritizing current topic, eliminating conflicting crops)
    filtered_history_records = filter_history_for_vision(
        history_records,
        active_crop,
        max_turns=4,
        current_conversation_id=current_conv_id,
    )
    filtered_history_records.reverse()  # chronological order for Gemini contents
    history = []
    for record in filtered_history_records:
        if record.user_message:
            history.append({"role": "user", "parts": record.user_message})
        if record.ai_response:
            # Clean structured enrichment sections from history to prevent LLM contamination
            clean_response = (
                record.ai_response
                .split("Available Nearby Shops:")[0]
                .split("🏬")[0]
                .split("📊")[0]
                .split("🌡️")[0]
                .split("🏛️")[0]
                .split("🎫")[0]
                .strip()
            )
            if clean_response:
                history.append({"role": "model", "parts": clean_response})

    # 3. Vision Prompt Scoping & Caption
    if active_context:
        active_crop_display = active_context["crop_display_en"]
        prev_msg = active_context.get("farmer_message", "")
        crop_scope_instruction = (
            f"=== CURRENT ACTIVE CONVERSATION CONTEXT ===\n"
            f"- Active Crop: {active_crop_display}\n"
            f"- Previous Farmer Message: \"{prev_msg}\"\n"
            f"MANDATORY INSTRUCTION: Analyze the attached image specifically in the context of the farmer's current {active_crop_display} issue. "
            f"Do not switch to another crop based only on older conversation history or profile memory."
        )
        if not user_caption:
            caption_to_send = (
                f"Analyze the attached image specifically in the context of the farmer's current {active_crop_display} issue: '{prev_msg}'. "
                f"Do not switch to another crop based only on older conversation history."
            )
        else:
            caption_to_send = f"{user_caption}\n[Active Crop Context: {active_crop_display}]"
    else:
        active_crop_display = None
        crop_scope_instruction = (
            "=== NO ACTIVE CROP CONTEXT ===\n"
            "The farmer has uploaded an image without specifying the crop name, and there is no preceding conversation establishing the crop.\n"
            "DO NOT guess or infer the crop from unrelated historical memory or profile memory.\n"
            "If you can identify both the crop and the disease from visual evidence alone with high confidence (confidence_score >= 0.85), provide the diagnosis.\n"
            "Otherwise, if the crop identity cannot be determined with certainty, you MUST set confidence_score to low and in 'friendly_whatsapp_reply' ask the farmer to confirm the crop name (e.g., in Telugu: 'ఈ చిత్రం ఏ పంటకు సంబంధించినది? పంట పేరు చెప్పగలరా?')."
        )
        caption_to_send = user_caption if user_caption else "Please analyze this image. If crop is not clearly identifiable, ask which crop this is."

    # Add vision-specific system prompt instruction enforcing JSON, crop context, and diagnostic safety
    full_system_prompt = (
        f"{BHOOMIMITRA_SYSTEM_PROMPT}\n\n"
        f"{lang_directive}\n\n"
        f"{crop_scope_instruction}\n\n"
        "The user has uploaded an image of their crop. Diagnose any visible diseases, pests, or deficiencies.\n"
        "IMAGE DIAGNOSIS SAFETY RULES:\n"
        "- Never claim that an image proves a disease with absolute certainty. Use cautious wording like 'appears consistent with', 'may indicate', or 'possible symptoms of'.\n"
        "- State that visual symptoms alone cannot be 100% confirmed from a single photo and ask the farmer to check front/back of leaf, close-up, or whole plant if uncertain.\n"
        "- Do not recommend unverified chemical pesticides or dosages unless grounded in trusted knowledge. Mention standard cultural practices and advise checking with a local Agriculture Extension Officer (AEO).\n"
        "- If the image does not show a crop, plant, leaf, or agricultural subject, set disease_name to 'non_agricultural', confidence_score to 0.0, and politely ask the farmer to send a clear photo of the crop or affected plant part.\n"
        "You MUST return a strictly valid JSON object matching this exact schema:\n"
        '{"disease_name": "Name", "confidence_score": 0.85, "severity": "low/medium/high", "symptoms": "Visible symptoms", "treatment_recommendation": "Cautious agronomic steps", "friendly_whatsapp_reply": "Natural language reply for the farmer"}\n'
        "Provide actionable agronomic advice.\n\n"
        f"{farmer_context}\n\n{memory_context}"
    )

    # 4. Call Gemini Multimodal
    from src.ai.gemini_client import generate_multimodal_response
    from src.ai.prompts import get_non_crop_image_response
    import json

    try:
        ai_response_text = await generate_multimodal_response(
            system_prompt=full_system_prompt,
            conversation_history=history,
            image_bytes=image_bytes,
            mime_type=mime_type,
            user_message=caption_to_send
        )
        if not ai_response_text:
            raise Exception("Empty response from AI")

        # Parse the structured JSON output
        parsed_json = json.loads(ai_response_text)
        diagnosis_data = MultimodalDiagnosisResponse(**parsed_json)

        # Check if non-agricultural or no crop detected
        is_non_crop = (
            not diagnosis_data.disease_name
            or diagnosis_data.disease_name.lower() in ("non_agricultural", "non_crop", "not_plant", "not_a_plant", "none")
            or (diagnosis_data.confidence_score is not None and diagnosis_data.confidence_score <= 0.1)
        )

        if is_non_crop:
            reply_text = get_non_crop_image_response(getattr(farmer, "preferred_language", "te") or "te")
            conversation.ai_response = reply_text
            db.add(conversation)
            await db.commit()
            return reply_text

        # 5. Crop Consistency Guard & Validation / Reconsideration Step
        if active_crop:
            combined_resp = f"{diagnosis_data.disease_name} {diagnosis_data.friendly_whatsapp_reply} {diagnosis_data.treatment_recommendation} {diagnosis_data.symptoms}"
            conflicting_crop = detect_crop_mismatch(active_crop, combined_resp, diagnosis_data.disease_name)
            if conflicting_crop:
                logger.warning(
                    f"[CROP CONSISTENCY GUARD] Active crop is '{active_crop}', but generated response refers to '{conflicting_crop}'. "
                    f"Triggering second validation/reconsideration step..."
                )
                validation_prompt = build_crop_reconsideration_instruction(
                    active_context["crop_display_en"],
                    conflicting_crop
                )
                try:
                    reconsidered_text = await generate_multimodal_response(
                        system_prompt=full_system_prompt,
                        conversation_history=history,
                        image_bytes=image_bytes,
                        mime_type=mime_type,
                        user_message=validation_prompt
                    )
                    if reconsidered_text:
                        reconsidered_json = json.loads(reconsidered_text)
                        reconsidered_data = MultimodalDiagnosisResponse(**reconsidered_json)
                        reconsidered_combined = f"{reconsidered_data.disease_name} {reconsidered_data.friendly_whatsapp_reply} {reconsidered_data.treatment_recommendation}"
                        if not detect_crop_mismatch(active_crop, reconsidered_combined, reconsidered_data.disease_name):
                            diagnosis_data = reconsidered_data
                        else:
                            # Reconsideration still has crop mismatch or cannot confirm -> ask farmer to confirm
                            logger.info("[CROP CONSISTENCY GUARD] Reconsideration remained conflicting. Asking farmer for crop clarification.")
                            diagnosis_data.friendly_whatsapp_reply = get_crop_clarification_response(img_lang_code)
                except Exception as recon_err:
                    logger.warning(f"[CROP CONSISTENCY GUARD] Reconsideration call failed: {recon_err}")
                    diagnosis_data.friendly_whatsapp_reply = get_crop_clarification_response(img_lang_code)

        elif not user_caption:
            # Image-only without active crop context
            is_low_conf = (diagnosis_data.confidence_score is None or diagnosis_data.confidence_score < 0.85)
            is_vague_disease = (not diagnosis_data.disease_name or diagnosis_data.disease_name.lower() in ("unknown", "unidentified", "none", "uncertain"))
            if is_low_conf or is_vague_disease:
                logger.info("[IMAGE WITHOUT CONTEXT] No active crop and insufficient confidence. Requesting crop clarification.")
                diagnosis_data.friendly_whatsapp_reply = get_crop_clarification_response(img_lang_code)

        reply_text = diagnosis_data.friendly_whatsapp_reply

        # 4. Save to Crop Health Module
        # Attempt to find the farmer's most recent crop to link the diagnosis
        result = await db.execute(
            select(Crop.id)
            .join(Farm)
            .where(Farm.farmer_id == farmer.id)
            .order_by(Crop.created_at.desc())
            .limit(1)
        )
        crop_id = result.scalar_one_or_none()

        if crop_id:
            crop_health_service = CropHealthService(
                repository=CropHealthRepository(db),
                crop_repository=CropRepository(db),
                farmer_repository=FarmerRepository(db)
            )
            create_data = CropHealthCreate(
                crop_id=crop_id,
                farmer_id=farmer.id,
                image_url=None, # Media ID is handled via Conversation temporarily
                symptoms=diagnosis_data.symptoms,
                disease_name=diagnosis_data.disease_name,
                diagnosis_result=diagnosis_data.friendly_whatsapp_reply,
                treatment_recommendation=diagnosis_data.treatment_recommendation,
                confidence_score=diagnosis_data.confidence_score,
            )
            await crop_health_service.create_diagnosis(create_data)
            logger.info(f"Structured diagnosis saved to CropHealth for farmer {farmer.id}")

        # Update Farmer Memory with diagnosis in background
        try:
            from src.memory.service import trigger_background_memory_extraction
            trigger_background_memory_extraction(
                farmer_id=farmer.id,
                user_message=user_caption or "",
                ai_response=f"Disease Diagnosed: {diagnosis_data.disease_name}. Treatment: {diagnosis_data.treatment_recommendation}"
            )
        except Exception as mem_err:
            logger.warning(f"Memory update scheduling failed for image message: {mem_err}")

        # Check if escalation is required for unknown disease or explicit farmer request
        try:
            from src.escalation.service import enrich_response_with_escalation, _detect_escalation_intent
            caption_lower = user_caption.lower() if user_caption else ""
            has_caption_intent, _ = _detect_escalation_intent(caption_lower, user_caption or "")
            is_unidentified = not diagnosis_data.disease_name or diagnosis_data.disease_name.lower() in ("unknown", "unidentified", "none")

            if has_caption_intent or is_unidentified:
                reply_text = await enrich_response_with_escalation(
                    db,
                    user_caption or "Image Crop Diagnosis",
                    reply_text,
                    farmer,
                    force_escalation=is_unidentified,
                    force_reason="inspection" if is_unidentified else None,
                )
        except Exception as esc_err:
            logger.warning(f"Failed to enrich image response with escalation: {esc_err}")

    except Exception as e:
        logger.warning(f"AI Vision unavailable or failed to parse for farmer {farmer.id}: {e}")
        reply_text = get_fallback_response(farmer.preferred_language)

    conversation.ai_response = reply_text
    db.add(conversation)
    await db.commit()

    return reply_text

