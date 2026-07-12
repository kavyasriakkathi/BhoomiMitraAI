"""
BhoomiMitra AI — AI Service (Orchestrator)

High-level service that ties together:
  - Farmer profile context
  - Conversation history
  - Prompt construction
  - Gemini API call
  - Response storage

This is the ONLY module that the gateway should call for AI responses.
"""

from typing import List, Dict
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from src.core.models import Farmer, FarmerProfile, Conversation
from src.core.logging import logger
from src.ai.prompts import (
    BHOOMIMITRA_SYSTEM_PROMPT,
    build_farmer_context,
    get_fallback_response,
)
from src.ai.gemini_client import generate_response


# Maximum number of recent messages to include as context
MAX_CONTEXT_MESSAGES = 10


async def get_conversation_history(
    db: AsyncSession, farmer_id, limit: int = MAX_CONTEXT_MESSAGES
) -> List[Dict[str, str]]:
    """
    Fetch recent conversation history for context injection.
    Returns a list of {"role": "user"|"model", "parts": "..."} dicts
    compatible with Gemini's chat format.
    """
    result = await db.execute(
        select(Conversation)
        .where(Conversation.farmer_id == farmer_id)
        .where(Conversation.ai_response.isnot(None))  # Only completed exchanges
        .order_by(Conversation.created_at.desc())
        .limit(limit)
    )
    rows = result.scalars().all()

    # Reverse to chronological order (oldest first)
    rows.reverse()

    history = []
    for row in rows:
        if row.user_message:
            history.append({"role": "user", "parts": row.user_message})
        if row.ai_response:
            history.append({"role": "model", "parts": row.ai_response})

    return history


async def get_farmer_profile(db: AsyncSession, farmer_id) -> FarmerProfile | None:
    """Fetch the farmer's profile for context injection."""
    result = await db.execute(
        select(FarmerProfile).where(FarmerProfile.farmer_id == farmer_id)
    )
    return result.scalar_one_or_none()


async def process_text_message(
    db: AsyncSession,
    farmer: Farmer,
    conversation: Conversation,
) -> str:
    """
    End-to-end AI processing pipeline for a text message.

    Steps:
      1. Load farmer profile for context.
      2. Load recent conversation history.
      3. Build the system prompt with farmer context.
      4. Call Gemini API.
      5. If AI fails, return a safe fallback.
      6. Store the AI response in the conversation record.
      7. Return the response text.
    """
    # Step 1: Load farmer profile
    profile = await get_farmer_profile(db, farmer.id)

    # Step 2: Build farmer context string
    farmer_context = build_farmer_context(
        crop=profile.current_crop if profile else None,
        district=profile.district if profile else None,
        state=profile.state if profile else None,
        land_size=profile.land_size_acres if profile else None,
    )

    # Step 3: Build system prompt with context
    full_system_prompt = f"{BHOOMIMITRA_SYSTEM_PROMPT}\n\n{farmer_context}"

    # Step 4: Load conversation history
    history = await get_conversation_history(db, farmer.id)

    # Step 5: Call Gemini
    user_text = conversation.user_message or ""
    logger.info(f"Processing AI request for farmer {farmer.id}: '{user_text[:80]}...'")

    ai_response = await generate_response(
        system_prompt=full_system_prompt,
        conversation_history=history,
        user_message=user_text,
    )

    # Step 6: Handle fallback
    if ai_response is None:
        logger.warning(f"AI unavailable for farmer {farmer.id}. Using fallback.")
        ai_response = get_fallback_response(farmer.preferred_language)

    # Step 7: Store AI response in conversation record
    conversation.ai_response = ai_response
    db.add(conversation)
    await db.commit()

    logger.info(
        f"AI response stored for conversation {conversation.id} "
        f"({len(ai_response)} chars)"
    )

    return ai_response
