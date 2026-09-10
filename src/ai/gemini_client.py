"""
BhoomiMitra AI — Gemini Client

Low-level wrapper around the Google Generative AI SDK.
Handles API calls, timeouts, error handling, and provider fallback.
"""

import asyncio
import time
from typing import List, Dict, Optional
import google.generativeai as genai
import google.api_core.exceptions
import requests.exceptions
from src.config import get_settings
from src.core.logging import logger

# Module-level flag to track initialization
_initialized = False

# Resilient fallback chain of supported models
FALLBACK_MODELS = [
    "gemini-3.5-flash",
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


def _is_auth_error(e: Exception) -> bool:
    """
    Deterministically detects authentication, permission, or invalid API key errors.
    Prevents pointless fallback attempts that share the same invalid credentials.
    """
    if e is None:
        return False
    err_str = str(e).lower()
    err_type = type(e).__name__.lower()

    if "permissiondenied" in err_type or "unauthenticated" in err_type:
        return True
    if getattr(e, "code", None) in (401, 403) or getattr(e, "status_code", None) in (401, 403) or getattr(e, "http_status", None) in (401, 403):
        return True

    auth_signals = (
        "api_key_invalid",
        "api key not valid",
        "invalid api key",
        "permission_denied",
        "permissiondenied",
        "unauthenticated",
        "unauthorized",
        "forbidden",
    )
    return any(sig in err_str for sig in auth_signals)


def _is_quota_exhausted_error(e: Exception) -> bool:
    """
    Deterministically detects confirmed Gemini HTTP 429 / ResourceExhausted quota errors.
    Prevents pointless fallback attempts across models that share the same project API key quota.
    """
    if e is None:
        return False
    err_str = str(e).lower()
    err_type = type(e).__name__.lower()

    if "resourceexhausted" in err_type or "toomanyrequests" in err_type:
        return True
    if getattr(e, "code", None) == 429 or getattr(e, "status_code", None) == 429 or getattr(e, "http_status", None) == 429:
        return True

    quota_signals = (
        "429",
        "resourceexhausted",
        "resource_exhausted",
        "toomanyrequests",
        "too many requests",
        "quota exceeded",
        "quota_exceeded",
        "free_tier_requests",
        "rate limit exceeded",
        "rate_limit_exceeded",
    )
    return any(sig in err_str for sig in quota_signals)


def _is_timeout_error(e: Exception) -> bool:
    """
    Deterministically detects whether an exception is an HTTP, socket, requests,
    or asyncio timeout error (e.g. requests.exceptions.ReadTimeout, requests.exceptions.ConnectTimeout,
    asyncio.TimeoutError, TimeoutError, google.api_core.exceptions.DeadlineExceeded).
    """
    if e is None:
        return False
    if isinstance(e, (asyncio.TimeoutError, TimeoutError, requests.exceptions.Timeout, google.api_core.exceptions.DeadlineExceeded)):
        return True
    err_type = type(e).__name__.lower()
    if any(t in err_type for t in ("timeout", "deadlineexceeded")):
        return True
    err_str = str(e).lower()
    return "timed out" in err_str or "timeout" in err_str or "deadline exceeded" in err_str


async def generate_response(
    system_prompt: str,
    conversation_history: List[Dict[str, str]],
    user_message: str,
    timeout_seconds: Optional[float] = None,
    model_override: Optional[str] = None,
    allow_fallback: bool = True,
) -> Optional[str]:
    """
    Send a message to the Gemini model and return the response text.
    Implements automatic model fallback in case of 429 / 503 errors.

    Args:
        system_prompt: The system-level instruction for the AI persona.
        conversation_history: List of {"role": "user"|"model", "parts": "..."} dicts
                              representing the recent conversation context.
        user_message: The farmer's current message.
        timeout_seconds: Max time to wait for API response (default: from settings or 15.0s).
        model_override: Optional model name to use instead of default.
        allow_fallback: Whether to attempt fallback models on failure (default: True).

    Returns:
        The AI response text, or raises exception if all attempts fail.
    """
    _ensure_initialized()
    settings = get_settings()
    if timeout_seconds is None:
        timeout_seconds = float(getattr(settings, "gemini_api_timeout_seconds", 15.0))
    primary_model = model_override or getattr(settings, "gemini_model", None) or "gemini-3.6-flash"

    # Build candidates list starting with primary model
    candidate_models = [primary_model]
    if allow_fallback:
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
        current_timeout = timeout_seconds if attempt_idx == 0 else min(timeout_seconds, 10.0)
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

            # In google-generativeai with transport='rest', HTTP calls execute synchronously via `requests`.
            # We offload the blocking call to a thread pool to protect the asyncio event loop while enforcing
            # socket-level timeouts via `request_options={"timeout": ...}` and asyncio-level timeouts via `wait_for`.
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    chat.send_message,
                    user_message,
                    request_options={"timeout": float(current_timeout)},
                ),
                timeout=float(current_timeout),
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

        except Exception as e:
            elapsed = time.time() - req_start_time
            last_error = e
            if _is_quota_exhausted_error(e):
                logger.warning(
                    f"[GEMINI QUOTA EXHAUSTED] Model {model_name} failed with quota exhaustion after {elapsed:.2f}s: {type(e).__name__} - {e}. "
                    "Aborting model fallback chain to prevent quota burn."
                )
                break
            if _is_auth_error(e):
                logger.error(
                    f"[GEMINI AUTH ERROR] Model {model_name} failed with authentication error after {elapsed:.2f}s: {type(e).__name__} - {e}. "
                    "Aborting model fallback chain because credentials are invalid."
                )
                break
            if _is_timeout_error(e):
                timeout_count += 1
                logger.warning(
                    f"[GEMINI TIMEOUT] Model {model_name} timed out after {elapsed:.2f}s "
                    f"(limit={current_timeout}s): {type(e).__name__} - {e}. Trying next model if available..."
                )
                if timeout_count >= 2:
                    logger.warning(f"[GEMINI TIMEOUT CEILING] {timeout_count} models timed out. Aborting model fallback to yield fast response.")
                    break
            else:
                logger.warning(
                    f"[GEMINI ERROR] Model {model_name} failed after {elapsed:.2f}s: {type(e).__name__} - {e}. "
                    f"Trying next model if available..."
                )

    total_elapsed = time.time() - total_start_time
    logger.exception(
        f"[GEMINI ALL MODELS EXHAUSTED] All {len(candidate_models)} models failed after {total_elapsed:.2f}s. "
        f"Last error: {last_error}"
    )
    if _is_timeout_error(last_error):
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
    timeout_count = 0

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
                asyncio.to_thread(
                    chat.send_message,
                    message_parts,
                    request_options={"timeout": float(timeout_seconds)},
                ),
                timeout=float(timeout_seconds),
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

        except Exception as e:
            elapsed = time.time() - req_start_time
            last_error = e
            if _is_quota_exhausted_error(e):
                logger.warning(
                    f"[GEMINI MULTIMODAL QUOTA EXHAUSTED] Model {model_name} failed with quota exhaustion after {elapsed:.2f}s: {type(e).__name__} - {e}. "
                    "Aborting model fallback chain to prevent quota burn."
                )
                break
            if _is_auth_error(e):
                logger.error(
                    f"[GEMINI MULTIMODAL AUTH ERROR] Model {model_name} failed with authentication error after {elapsed:.2f}s: {type(e).__name__} - {e}. "
                    "Aborting model fallback chain because credentials are invalid."
                )
                break
            if _is_timeout_error(e):
                timeout_count += 1
                logger.warning(
                    f"[GEMINI MULTIMODAL TIMEOUT] Model {model_name} timed out after {elapsed:.2f}s (limit={timeout_seconds}s): {type(e).__name__} - {e}. "
                    f"Trying next model if available..."
                )
                if timeout_count >= 2:
                    logger.warning(f"[GEMINI MULTIMODAL TIMEOUT CEILING] {timeout_count} models timed out. Aborting fallback.")
                    break
            else:
                logger.warning(
                    f"[GEMINI MULTIMODAL ERROR] Model {model_name} failed after {elapsed:.2f}s: {type(e).__name__} - {e}. "
                    f"Trying next model if available..."
                )

    total_elapsed = time.time() - total_start_time
    logger.exception(
        f"[GEMINI MULTIMODAL ALL MODELS EXHAUSTED] All {len(candidate_models)} models failed after {total_elapsed:.2f}s. "
        f"Last error: {last_error}"
    )
    if _is_timeout_error(last_error):
        raise TimeoutError(f"Gemini Multimodal API timed out after {timeout_seconds}s across all attempts") from last_error
    raise RuntimeError(f"Gemini SDK Error: {str(last_error)}") from last_error
