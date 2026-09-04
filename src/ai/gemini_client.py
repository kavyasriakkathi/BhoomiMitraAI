"""
BhoomiMitra AI — Gemini Client

Low-level wrapper around the Google Generative AI SDK.
Handles API calls, timeouts, error handling, and provider fallback.
"""

import asyncio
import time
from typing import List, Dict, Optional
import google.generativeai as genai
from src.config import get_settings
from src.core.logging import logger

# Module-level flag to track initialization
_initialized = False

# Resilient fallback chain of supported models
FALLBACK_MODELS = [
    "gemini-3.5-flash",
    "gemini-flash-latest",
]


def _ensure_initialized():
    """Configure the Gemini SDK once on first use with REST transport."""
    global _initialized
    if not _initialized:
        settings = get_settings()
        if not settings.google_gemini_api_key:
            logger.error("[GEMINI CONFIG ERROR] GOOGLE_GEMINI_API_KEY is not configured in settings or environment.")
            raise RuntimeError("Gemini API key is not configured.")
        genai.configure(api_key=settings.google_gemini_api_key, transport="rest")
        _initialized = True
        logger.info("Gemini SDK initialized successfully with transport='rest'.")


async def generate_response(
    system_prompt: str,
    conversation_history: List[Dict[str, str]],
    user_message: str,
    timeout_seconds: Optional[float] = None,
    model_override: Optional[str] = None,
) -> Optional[str]:
    """
    Send a message to the Gemini model and return the response text.
    Implements automatic model fallback in case of 429 / 503 errors.

    Args:
        system_prompt: The system-level instruction for the AI persona.
        conversation_history: List of {"role": "user"|"model", "parts": "..."} dicts
                              representing the recent conversation context.
        user_message: The farmer's current message.
        timeout_seconds: Max time to wait for API response (default: from settings or 5.0s).
        model_override: Optional model name to use instead of default.

    Returns:
        The AI response text, or raises exception if all attempts fail.
    """
    _ensure_initialized()
    settings = get_settings()
    if timeout_seconds is None:
        timeout_seconds = float(getattr(settings, "gemini_api_timeout_seconds", 5.0))
    primary_model = model_override or getattr(settings, "gemini_model", None) or "gemini-3.6-flash"

    # Build candidates list starting with primary model
    candidate_models = [primary_model]
    for fallback in FALLBACK_MODELS:
        if fallback not in candidate_models:
            candidate_models.append(fallback)

    history = []
    for msg in conversation_history:
        history.append({"role": msg["role"], "parts": [msg["parts"]]})

    total_start_time = time.time()
    last_error = None
    timeout_count = 0

    for attempt_idx, model_name in enumerate(candidate_models):
        req_start_time = time.time()
        current_timeout = timeout_seconds if attempt_idx == 0 else min(timeout_seconds, 3.0)
        logger.info(
            f"[GEMINI API REQUEST START] (Attempt {attempt_idx + 1}/{len(candidate_models)})\n"
            f"  Model            : {model_name}\n"
            f"  Timeout          : {current_timeout}s\n"
            f"  Context History  : {len(history)} messages\n"
            f"  User Message     : '{user_message[:120]}' (len={len(user_message)})\n"
            f"  System Prompt Len: {len(system_prompt)} chars"
        )

        try:
            model = genai.GenerativeModel(
                model_name=model_name,
                system_instruction=system_prompt,
                generation_config=genai.GenerationConfig(
                    temperature=0.4,
                    max_output_tokens=1024,
                    top_p=0.9,
                ),
            )

            chat = model.start_chat(history=history)

            response = await asyncio.wait_for(
                asyncio.to_thread(chat.send_message, user_message),
                timeout=current_timeout,
            )

            elapsed = time.time() - req_start_time
            total_elapsed = time.time() - total_start_time
            logger.info(
                f"[GEMINI API RESPONSE RECEIVED]\n"
                f"  Model            : {model_name}\n"
                f"  Status           : 200 OK\n"
                f"  Call Duration    : {elapsed:.2f}s\n"
                f"  Total Duration   : {total_elapsed:.2f}s"
            )

            # Response parsing
            ai_text = response.text.strip() if response.text else ""
            logger.info(
                f"[GEMINI RESPONSE PARSED]\n"
                f"  Model Used       : {model_name}\n"
                f"  Output Length    : {len(ai_text)} chars\n"
                f"  Preview          : '{ai_text[:120]}...'"
            )
            return ai_text

        except asyncio.TimeoutError as e:
            elapsed = time.time() - req_start_time
            timeout_count += 1
            logger.warning(
                f"[GEMINI TIMEOUT] Model {model_name} timed out after {elapsed:.2f}s "
                f"(limit={current_timeout}s). Trying next model if available..."
            )
            last_error = e
            if timeout_count >= 2:
                logger.warning(f"[GEMINI TIMEOUT CEILING] {timeout_count} models timed out. Aborting model fallback to yield fast response.")
                break

        except Exception as e:
            elapsed = time.time() - req_start_time
            logger.warning(
                f"[GEMINI ERROR] Model {model_name} failed after {elapsed:.2f}s: {type(e).__name__} - {e}. "
                f"Trying next model if available..."
            )
            last_error = e

    total_elapsed = time.time() - total_start_time
    logger.exception(
        f"[GEMINI ALL MODELS EXHAUSTED] All {len(candidate_models)} models failed after {total_elapsed:.2f}s. "
        f"Last error: {last_error}"
    )
    if isinstance(last_error, asyncio.TimeoutError):
        raise TimeoutError(f"Gemini API timed out after {total_elapsed:.1f}s across attempts") from last_error
    raise RuntimeError(f"Gemini SDK Error: {str(last_error)}") from last_error


async def generate_multimodal_response(
    system_prompt: str,
    conversation_history: List[Dict[str, str]],
    image_bytes: bytes,
    mime_type: str,
    user_message: str = "",
    timeout_seconds: int = 15,
    model_override: Optional[str] = None,
) -> Optional[str]:
    """
    Send an image and an optional text prompt to the Gemini Vision model.
    """
    _ensure_initialized()
    settings = get_settings()
    primary_model = model_override or getattr(settings, "gemini_model", None) or "gemini-3.6-flash"

    candidate_models = [primary_model]
    for fallback in FALLBACK_MODELS:
        if fallback not in candidate_models:
            candidate_models.append(fallback)

    history = []
    for msg in conversation_history:
        history.append({"role": msg["role"], "parts": [msg["parts"]]})

    total_start_time = time.time()
    last_error = None

    for attempt_idx, model_name in enumerate(candidate_models):
        req_start_time = time.time()
        logger.info(
            f"[GEMINI MULTIMODAL REQUEST START] (Attempt {attempt_idx + 1}/{len(candidate_models)})\n"
            f"  Model            : {model_name}\n"
            f"  Timeout          : {timeout_seconds}s\n"
            f"  Image Size       : {len(image_bytes)} bytes ({mime_type})\n"
            f"  Caption          : '{user_message}'\n"
            f"  Context History  : {len(history)} messages"
        )

        try:
            model = genai.GenerativeModel(
                model_name=model_name,
                system_instruction=system_prompt,
                generation_config=genai.GenerationConfig(
                    temperature=0.4,
                    max_output_tokens=1024,
                    top_p=0.9,
                    response_mime_type="application/json",
                ),
            )

            chat = model.start_chat(history=history)

            message_parts = [{"mime_type": mime_type, "data": image_bytes}]
            if user_message:
                message_parts.append(user_message)

            response = await asyncio.wait_for(
                asyncio.to_thread(chat.send_message, message_parts),
                timeout=timeout_seconds,
            )

            elapsed = time.time() - req_start_time
            total_elapsed = time.time() - total_start_time
            logger.info(
                f"[GEMINI MULTIMODAL RESPONSE RECEIVED]\n"
                f"  Model            : {model_name}\n"
                f"  Status           : 200 OK\n"
                f"  Call Duration    : {elapsed:.2f}s\n"
                f"  Total Duration   : {total_elapsed:.2f}s"
            )

            ai_text = response.text.strip() if response.text else ""
            logger.info(
                f"[GEMINI MULTIMODAL RESPONSE PARSED]\n"
                f"  Model Used       : {model_name}\n"
                f"  Output Length    : {len(ai_text)} chars\n"
                f"  Preview          : '{ai_text[:120]}...'"
            )
            return ai_text

        except asyncio.TimeoutError as e:
            elapsed = time.time() - req_start_time
            logger.warning(
                f"[GEMINI MULTIMODAL TIMEOUT] Model {model_name} timed out after {elapsed:.2f}s. "
                f"Trying next model if available..."
            )
            last_error = e

        except Exception as e:
            elapsed = time.time() - req_start_time
            logger.warning(
                f"[GEMINI MULTIMODAL ERROR] Model {model_name} failed after {elapsed:.2f}s: {type(e).__name__} - {e}. "
                f"Trying next model if available..."
            )
            last_error = e

    total_elapsed = time.time() - total_start_time
    logger.exception(
        f"[GEMINI MULTIMODAL ALL MODELS EXHAUSTED] All {len(candidate_models)} models failed after {total_elapsed:.2f}s. "
        f"Last error: {last_error}"
    )
    if isinstance(last_error, asyncio.TimeoutError):
        raise TimeoutError(f"Gemini Multimodal API timed out after {timeout_seconds}s across all attempts") from last_error
    raise RuntimeError(f"Gemini SDK Error: {str(last_error)}") from last_error
