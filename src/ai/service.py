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
)
from src.config import get_settings
from src.ai.gemini_client import generate_response
import re


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
        r'what\s+is\s+the\s+dose|what\s+is\s+the\s+dosage|spray\s+dosage|application\s+rate|dosage\s+rate|'
        r'dose\s+of|dosage\s+of|quantity\s+of\s+fertilizer|'
        r'ఎంత\s*(?:మోతాదు|ఎరువు|యూరియా|మందు|స్ప్రే|డిఎపి|పొటాష్|జింక్|కేజీలు|గ్రాములు)|'
        r'(?:ఎకరాకి|ఎకరానికి|లీటరుకు|లీటరుకి|హెక్టారుకు)\s*ఎంత|మోతాదు\s*ఎంత|డోసేజ్|డోస్|'
        r'దశలో\s*(?:ఉన్న\s*)?(?:వరి|పత్తి|మిర్చి|మొక్కజొన్న|పంటకు)\s*ఎరువు\s*ఏది|'
        r'పిలకల\s*దశలో\s*ఎరువు|పూత\s*దశలో\s*ఎరువు|కాయ\s*దశలో\s*ఎరువు|'
        r'పిలకల\s*దశలో\s*వరి|'
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
        r'entha\s+(?:fertilizer|urea|eruvu|mandhu|dose|dosage|dap|potash|zinc)|'
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


def extract_query_dosage_topics(query: str) -> dict:
    """
    Extracts the crop, specific fertilizer / chemical substances, and intent category
    from a dosage-sensitive query across multiple languages and Romanized inputs.
    """
    if not query:
        return {
            "crop": None,
            "is_fertilizer_query": False,
            "requested_fertilizers": [],
            "is_pest_disease_query": False,
            "requested_chemicals": [],
        }

    t = query.lower().strip()

    # 1. Detect crop
    crop = None
    if any(k in t for k in ["వరి", "వరికి", "వరిలో", "వరిపంట", "paddy", "rice", "vari", "dhan", "धान", "நெல்", "ಭತ್ತ"]):
        crop = "paddy"
    elif any(k in t for k in ["పత్తి", "పత్తికి", "పత్తిలో", "cotton", "patti", "kapas", "कपास", "பருத்தி", "ಹತ್ತಿ"]):
        crop = "cotton"
    elif any(k in t for k in ["మిర్చి", "మిరప", "మిరపకి", "chilli", "chili", "mirchi", "mirapa", "मिर्च", "மிளகாய்", "ಮೆಣಸಿನಕಾಯಿ"]):
        crop = "chilli"
    elif any(k in t for k in ["మొక్కజొన్న", "maize", "corn", "mokkajonna", "मक्का", "மக்காச்சோளம்", "ಮೆಕ್ಕೆಜೋಳ"]):
        crop = "maize"
    elif any(k in t for k in ["టమాటా", "టమాట", "tomato", "tamata", "टमाटर", "தக்காளி", "ಟೊಮ್ಯಾಟೊ"]):
        crop = "tomato"

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


class AIService:
    def __init__(self, repository: AIRepository):
        self.repository = repository

    async def generate_ai_response(self, request: AIGenerateRequest) -> AIGenerateResponse:
        service_start_time = time.time()
        try:
            # 1. Fetch farmer profile for context
            profile = await self.repository.get_farmer_profile(request.farmer_id)

            # 2. Fetch conversation history (oldest first for Gemini)
            history_records = await self.repository.get_conversation_history(request.farmer_id)
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
            effective_crop = query_crop or (recent_context_crop if recent_context_crop else (profile.current_crop if profile else None))

            # Build farmer context string
            farmer_context = build_farmer_context(
                crop=effective_crop or (profile.current_crop if profile else None),
                district=profile.district if profile else None,
                state=profile.state if profile else None,
                land_size=profile.land_size_acres if profile else None,
            )

            # 3. Fetch farmer long-term memory context
            from src.memory.service import FarmerMemoryService
            from src.memory.repository import FarmerMemoryRepository
            mem_repo = FarmerMemoryRepository(self.repository.session)
            mem_service = FarmerMemoryService(mem_repo)
            memory_context = await mem_service.format_memory_for_system_prompt(request.farmer_id)

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

            try:
                from src.rag.service import RAGService
                from src.rag.repository import RAGRepository
                rag_repo = RAGRepository(self.repository.session)
                rag_service = RAGService(rag_repo)
                rag_results = await rag_service.search_knowledge(
                    query=rag_query,
                    top_k=3,
                    state=profile.state if profile else None,
                    crop=effective_crop,
                )
                if rag_results:
                    rag_snippets = [f"• Document: {r.document_title} (Crop: {r.crop or 'General'}, Source: {r.source}): {r.chunk_text}" for r in rag_results]
                    rag_context_text = (
                        "=== RETRIEVED TRUSTED AGRICULTURAL KNOWLEDGE (GROUND TRUTH) ===\n"
                        "CRITICAL INSTRUCTION: The following knowledge is verified agronomic ground truth. You MUST strictly use ONLY these verified treatments, products, and numeric dosages. Do NOT add, alter, extrapolate, or invent different dosages or unverified chemicals. Translate directly into the farmer's language:\n"
                        + "\n".join(rag_snippets)
                    )
            except Exception as rag_err:
                logger.warning(f"RAG knowledge retrieval warning: {rag_err}")

            # 3.8 HARD GROUNDING GATE: Block unverified numeric dosage generation
            is_dosage_req = is_dosage_sensitive_query(request.message)
            has_grounding = has_relevant_dosage_ground_truth(request.message, rag_snippets)

            if is_dosage_req and not has_grounding:
                logger.warning(
                    f"[HARD GROUNDING GATE TRIGGERED] Dosage-sensitive query lacking relevant verified Ground Truth. "
                    f"Farmer ID: {request.farmer_id}, Language: '{user_lang}', Query: '{request.message[:80]}...'"
                )
                safe_fallback = get_unverified_dosage_fallback_response(user_lang)
                return AIGenerateResponse(
                    response_text=safe_fallback,
                    intent="dosage_unverified_fallback",
                    confidence=1.0,
                    provider_used="hard_grounding_gate",
                )

            # 4. Build system prompt combining profile, memory engine, and RAG ground truth
            full_system_prompt = f"{BHOOMIMITRA_SYSTEM_PROMPT}\n\n{farmer_context}\n\n{memory_context}"
            if rag_context_text:
                full_system_prompt += f"\n\n{rag_context_text}"


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
                    ai_text = get_unverified_dosage_fallback_response(user_lang)
                else:
                    # Verify generated numeric figures exist in the provided RAG ground truth
                    rag_text_lower = rag_context_text.lower()
                    generated_numbers = re.findall(r'\b\d+(?:[.,]\d+)?\b', " ".join(output_dosage_matches))
                    rag_numbers = set(re.findall(r'\b\d+(?:[.,]\d+)?\b', rag_text_lower))
                    unverified_numbers = [n for n in generated_numbers if n not in rag_numbers]
                    if unverified_numbers:
                        logger.warning(
                            f"[DEFENSE-IN-DEPTH GATE TRIGGERED] Hallucinated dosage numbers {unverified_numbers} not found in RAG ground truth for farmer {request.farmer_id}. "
                            f"Replacing with safe fallback."
                        )
                        ai_text = get_unverified_dosage_fallback_response(user_lang)
            elif is_dosage_req and not has_grounding:
                ai_text = get_unverified_dosage_fallback_response(user_lang)

            total_ai_time = time.time() - service_start_time
            logger.info(
                f"[AI SERVICE GENERATION SUCCESS]\n"
                f"  Farmer ID    : {request.farmer_id}\n"
                f"  Total Time   : {total_ai_time:.2f}s\n"
                f"  Output Chars : {len(ai_text)}\n"
                f"  Preview      : '{ai_text[:120]}...'"
            )

            # 7. Automatic memory extraction from exchange (runs safely & swiftly)
            try:
                await mem_service.extract_and_update_memory(
                    farmer_id=request.farmer_id,
                    user_message=request.message,
                    ai_response=ai_text
                )
            except Exception as mem_err:
                logger.warning(f"Automatic memory extraction warning for farmer {request.farmer_id}: {mem_err}")

            # 8. Return structured response
            return AIGenerateResponse(
                response_text=ai_text,
                intent=None,
                confidence=None,
                provider_used="gemini"
            )

        except HTTPException:
            raise
        except Exception as e:
            elapsed = time.time() - service_start_time
            logger.exception(f"[AI SERVICE ERROR] Failed generating AI response after {elapsed:.2f}s: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"An unexpected error occurred while communicating with the AI provider: {str(e)}"
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
            return 3
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
    farmer_context = build_farmer_context(
        crop=profile.current_crop if profile else None,
        district=profile.district if profile else None,
        state=profile.state if profile else None,
        land_size=profile.land_size_acres if profile else None,
    )
    
    from src.memory.service import FarmerMemoryService
    from src.memory.repository import FarmerMemoryRepository
    mem_repo = FarmerMemoryRepository(db)
    mem_service = FarmerMemoryService(mem_repo)
    memory_context = await mem_service.format_memory_for_system_prompt(farmer.id)

    # Add a vision-specific system prompt instruction enforcing JSON and diagnostic safety
    full_system_prompt = (
        f"{BHOOMIMITRA_SYSTEM_PROMPT}\n\n"
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

    
    # 2. History
    history_records = await repo.get_conversation_history(farmer.id)
    history_records.reverse()
    history = []
    for record in history_records:
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
            
    # 3. Call Gemini Multimodal
    from src.ai.gemini_client import generate_multimodal_response
    from src.ai.prompts import get_non_crop_image_response
    import json
    
    user_caption = conversation.user_message or "Please analyze this image."
    
    try:
        ai_response_text = await generate_multimodal_response(
            system_prompt=full_system_prompt,
            conversation_history=history,
            image_bytes=image_bytes,
            mime_type=mime_type,
            user_message=user_caption
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
            reply_text = get_non_crop_image_response(farmer.preferred_language or "te")
            conversation.ai_response = reply_text
            db.add(conversation)
            await db.commit()
            return reply_text

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

        # Update Farmer Memory with diagnosis
        try:
            await mem_service.extract_and_update_memory(
                farmer_id=farmer.id,
                user_message=user_caption,
                ai_response=f"Disease Diagnosed: {diagnosis_data.disease_name}. Treatment: {diagnosis_data.treatment_recommendation}"
            )
        except Exception as mem_err:
            logger.warning(f"Memory update failed for image message: {mem_err}")

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

