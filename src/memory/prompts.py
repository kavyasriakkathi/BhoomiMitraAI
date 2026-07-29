"""
BhoomiMitra AI — Farmer Memory Prompts & Context Builder

Centralized prompts for automatic memory extraction, summarization,
and injection into the Gemini LLM context window.
"""

MEMORY_EXTRACTION_SYSTEM_PROMPT = """You are an AI Memory Extraction Engine for an agricultural assistant.
Analyze the farmer's input message and the AI's response to extract structured long-term memory updates.

Return ONLY a valid JSON object matching this schema (do not include markdown codeblocks or extra text):
{
    "updates": {
        "farm_size": float or null,
        "village": string or null,
        "district": string or null,
        "state": string or null,
        "soil_type": string or null,
        "water_source": string or null,
        "irrigation_method": string or null,
        "primary_crops": [list of strings] or null,
        "secondary_crops": [list of strings] or null,
        "favorite_shops": [list of strings] or null,
        "preferred_brands": [list of strings] or null,
        "preferred_language": string or null (e.g. "te", "hi", "en"),
        "preferred_voice": string or null,
        "voice_speed": float or null,
        "voice_gender": string or null ("FEMALE" or "MALE"),
        "government_schemes_used": [list of strings] or null,
        "disease_mentioned": string or null,
        "pesticide_mentioned": string or null,
        "fertilizer_mentioned": string or null
    },
    "confidence_scores": {
        "farm_size": float (0.0 to 1.0),
        "village": float (0.0 to 1.0),
        "district": float (0.0 to 1.0),
        "primary_crops": float (0.0 to 1.0),
        "favorite_shops": float (0.0 to 1.0)
    }
}

Rules:
1. Only extract fields where the user explicitly stated or strongly implied information.
2. If no new information is present for a field, leave it null.
3. Assign confidence scores between 0.0 and 1.0. High confidence (>=0.8) means explicit statement like "My farm is 5 acres" or "My village is Karimnagar".
4. Do NOT guess or hallucinate parameters.
"""

MEMORY_SUMMARIZATION_SYSTEM_PROMPT = """You are an AI Summarizer for Indian Agriculture.
Summarize the following farmer interaction history into a concise, high-value long-term memory summary (3-5 bullet points).
Focus on: crop issues faced, remedies recommended, farm equipment/inputs used, and recurring farmer preferences.
Keep the summary under 150 words.
"""

def build_memory_context_prompt(memory) -> str:
    """
    Constructs a compact, token-efficient system prompt block from non-empty
    fields of the farmer's long-term memory profile.
    """
    if not memory:
        return ""

    parts = []

    # Geographic & Land Context
    geo_parts = []
    if memory.village:
        geo_parts.append(f"Village: {memory.village}")
    if memory.district:
        geo_parts.append(f"District: {memory.district}")
    if memory.state:
        geo_parts.append(f"State: {memory.state}")
    if memory.farm_size:
        geo_parts.append(f"Farm Size: {memory.farm_size} acres")
    if geo_parts:
        parts.append("Land & Location: " + ", ".join(geo_parts))

    # Soil & Irrigation
    soil_parts = []
    if memory.soil_type:
        soil_parts.append(f"Soil: {memory.soil_type}")
    if memory.water_source:
        soil_parts.append(f"Water Source: {memory.water_source}")
    if memory.irrigation_method:
        soil_parts.append(f"Irrigation: {memory.irrigation_method}")
    if soil_parts:
        parts.append("Soil & Water: " + ", ".join(soil_parts))

    # Crops & Agronomy
    crop_parts = []
    if memory.primary_crops:
        crop_parts.append(f"Primary Crops: {', '.join(memory.primary_crops)}")
    if memory.secondary_crops:
        crop_parts.append(f"Secondary Crops: {', '.join(memory.secondary_crops)}")
    if crop_parts:
        parts.append("Crops: " + "; ".join(crop_parts))

    # History highlights (Diseases, Fertilizers, Pesticides)
    if memory.disease_history:
        recent_diseases = [d.get("disease", str(d)) if isinstance(d, dict) else str(d) for d in memory.disease_history[-3:]]
        parts.append(f"Disease History: {', '.join(recent_diseases)}")
    if memory.fertilizer_history:
        recent_fert = [f.get("name", str(f)) if isinstance(f, dict) else str(f) for f in memory.fertilizer_history[-3:]]
        parts.append(f"Fertilizers Used: {', '.join(recent_fert)}")
    if memory.pesticide_history:
        recent_pest = [p.get("name", str(p)) if isinstance(p, dict) else str(p) for p in memory.pesticide_history[-3:]]
        parts.append(f"Pesticides Used: {', '.join(recent_pest)}")

    # Commerce & Brands
    comm_parts = []
    if memory.favorite_shops:
        comm_parts.append(f"Favorite Shops: {', '.join(memory.favorite_shops)}")
    if memory.preferred_brands:
        comm_parts.append(f"Preferred Brands: {', '.join(memory.preferred_brands)}")
    if comm_parts:
        parts.append("Commerce: " + "; ".join(comm_parts))

    # Schemes
    if memory.government_schemes_used:
        parts.append(f"Government Schemes: {', '.join(memory.government_schemes_used)}")

    # Risk Factors & AI Learned Preferences
    if memory.risk_factors:
        parts.append(f"Risk Factors: {', '.join(memory.risk_factors)}")
    if memory.conversation_summary:
        parts.append(f"Past Interaction Summary: {memory.conversation_summary}")

    if not parts:
        return "[Farmer Memory: Profile is newly initialized. Learn details through conversation.]"

    return "## Farmer Long-Term Memory Profile\n" + "\n".join(f"- {p}" for p in parts)
