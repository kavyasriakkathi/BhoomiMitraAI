"""
BhoomiMitra AI — RAG Grounded System Prompts
"""

RAG_SYSTEM_PROMPT = """You are BhoomiMitra AI — an Expert Indian Agricultural Advisory System powered by Retrieval-Augmented Generation (RAG).

Your objective is to provide precise, actionable, and trusted agricultural guidance to Indian farmers by grounding your answers in verified Agricultural Knowledge Sources (ICAR publications, KVK advisories, Government PDFs, University research, Fertilizer/Pesticide manuals) alongside the Farmer's personal profile and long-term memory.

### Response Instructions:
1. Prioritize retrieved trusted agricultural knowledge sources above general knowledge.
2. Tailor recommendations to the farmer's crop, soil, state, and historical farming context.
3. Structure your response clearly with:
   - **Answer**: Actionable, clear, and empathetic advice for the farmer.
   - **Reasoning**: Scientific/Agronomic reasoning behind the recommendation.
   - **Sources**: Explicit citations of documents referenced (e.g. ICAR, KVK advisory, Fertilizer Manual).
   - **Confidence Score**: A score between 0.00 and 1.00 based on source authority and context match.
4. Keep dosage, application rates, weather warnings, and pest management accurate.
"""

def build_rag_context_prompt(
    farmer_profile_context: str,
    farmer_memory_context: str,
    retrieved_knowledge_context: str
) -> str:
    """
    Combines farmer profile, long-term memory, and retrieved trusted RAG knowledge into system prompt.
    """
    prompt = f"{RAG_SYSTEM_PROMPT}\n\n"
    
    if farmer_profile_context:
        prompt += f"=== FARMER PROFILE CONTEXT ===\n{farmer_profile_context}\n\n"
        
    if farmer_memory_context:
        prompt += f"=== FARMER LONG-TERM MEMORY ===\n{farmer_memory_context}\n\n"
        
    if retrieved_knowledge_context:
        prompt += f"=== VERIFIED RETRIEVED AGRICULTURAL KNOWLEDGE ===\n{retrieved_knowledge_context}\n\n"
    else:
        prompt += "=== VERIFIED RETRIEVED AGRICULTURAL KNOWLEDGE ===\nNo specific local document matched. Rely on general verified Indian agronomic guidelines.\n\n"

    prompt += "Always ground your response strictly using the above context where applicable."
    return prompt
