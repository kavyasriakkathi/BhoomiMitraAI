"""
BhoomiMitra AI — Gemini Client

Low-level wrapper around the Google Generative AI SDK.
Handles API calls, timeouts, error handling, and provider fallback.
"""

import asyncio
from typing import List, Dict, Optional
import google.generativeai as genai
from src.config import get_settings
from src.core.logging import logger

# Module-level flag to track initialization
_initialized = False


def _ensure_initialized():
    """Configure the Gemini SDK once on first use."""
    global _initialized
    if not _initialized:
        settings = get_settings()
        if not settings.google_gemini_api_key:
            logger.error("GOOGLE_GEMINI_API_KEY is not set.")
            raise RuntimeError("Gemini API key is not configured.")
        genai.configure(api_key=settings.google_gemini_api_key)
        _initialized = True
        logger.info("Gemini SDK initialized.")


async def generate_response(
    system_prompt: str,
    conversation_history: List[Dict[str, str]],
    user_message: str,
    timeout_seconds: int = 30,
) -> Optional[str]:
    """
    Send a message to the Gemini model and return the response text.

    Args:
        system_prompt: The system-level instruction for the AI persona.
        conversation_history: List of {"role": "user"|"model", "parts": "..."} dicts
                              representing the recent conversation context.
        user_message: The farmer's current message.
        timeout_seconds: Max time to wait for the API response.

    Returns:
        The AI response text, or None if the call fails.
    """
    _ensure_initialized()
    settings = get_settings()

    try:
        model = genai.GenerativeModel(
            model_name="gemini-flash-latest",
            system_instruction=system_prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.4,        # Low temperature for factual farming advice
                max_output_tokens=512,   # Keep responses short for WhatsApp
                top_p=0.9,
            ),
        )

        # Build the full message history for context
        history = []
        for msg in conversation_history:
            history.append({"role": msg["role"], "parts": [msg["parts"]]})

        chat = model.start_chat(history=history)

        logger.info(f"Sending message to Gemini (context_len={len(history)})")

        # Run the synchronous SDK call in a thread with a timeout
        response = await asyncio.wait_for(
            asyncio.to_thread(chat.send_message, user_message),
            timeout=timeout_seconds,
        )

        ai_text = response.text.strip()
        logger.info(f"Gemini response received ({len(ai_text)} chars)")
        return ai_text

    except asyncio.TimeoutError as e:
        logger.error(f"Gemini API timed out after {timeout_seconds}s.")
        raise TimeoutError("Gemini API timed out") from e

    except Exception as e:
        logger.exception(f"Gemini API call failed: {e}")
        raise RuntimeError(f"Gemini SDK Error: {str(e)}") from e
