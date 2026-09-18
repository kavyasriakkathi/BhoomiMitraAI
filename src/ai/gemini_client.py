"""
BhoomiMitra AI — Gemini Client

Low-level wrapper around Google Cloud Gemini via the google-genai SDK.
Handles API calls, timeouts, error handling, and model fallback.
"""

import asyncio
import time
from typing import List, Dict, Optional

from google import genai
from google.genai import types

from src.config import get_settings
from src.core.logging import logger


# Module-level client
_client = None


# Resilient fallback chain of supported models
FALLBACK_MODELS = [
    "gemini-3.5-flash",
]


def _ensure_initialized():
    """Initialize the Google Cloud Gemini client once."""

    global _client

    if _client is None:
        settings = get_settings()

        project_id = getattr(settings, "google_cloud_project_id", None)

        if not project_id:
            logger.error(
                "[GEMINI CONFIG ERROR] "
                "GOOGLE_CLOUD_PROJECT_ID is not configured."
            )
            raise RuntimeError(
                "Google Cloud project ID is not configured."
            )

        _client = genai.Client(
            vertexai=True,
            project=project_id,
            location="global",
        )

        logger.info(
            f"Google Cloud Gemini SDK initialized successfully "
            f"using project={project_id}, location=global."
        )

    return _client


def _is_auth_error(e: Exception) -> bool:
    """
    Detect authentication / permission errors.
    """

    if e is None:
        return False

    err_str = str(e).lower()
    err_type = type(e).__name__.lower()

    if "permissiondenied" in err_type:
        return True

    if "unauthenticated" in err_type:
        return True

    status = getattr(e, "code", None)
    if status in (401, 403):
        return True

    status = getattr(e, "status_code", None)
    if status in (401, 403):
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
        "authentication failed",
    )

    return any(signal in err_str for signal in auth_signals)


def _is_quota_exhausted_error(e: Exception) -> bool:
    """
    Detect confirmed 429 / quota / rate-limit errors.
    """

    if e is None:
        return False

    err_str = str(e).lower()
    err_type = type(e).__name__.lower()

    if "resourceexhausted" in err_type:
        return True

    if "toomanyrequests" in err_type:
        return True

    status = getattr(e, "code", None)
    if status == 429:
        return True

    status = getattr(e, "status_code", None)
    if status == 429:
        return True

    quota_signals = (
        "429",
        "resourceexhausted",
        "resource_exhausted",
        "toomanyrequests",
        "too many requests",
        "quota exceeded",
        "quota_exceeded",
        "rate limit exceeded",
        "rate_limit_exceeded",
    )

    return any(signal in err_str for signal in quota_signals)


def _is_timeout_error(e: Exception) -> bool:
    """
    Detect timeout-related errors.
    """

    if e is None:
        return False

    if isinstance(
        e,
        (
            asyncio.TimeoutError,
            TimeoutError,
        ),
    ):
        return True

    err_type = type(e).__name__.lower()

    if any(
        value in err_type
        for value in ("timeout", "deadlineexceeded")
    ):
        return True

    err_str = str(e).lower()

    return (
        "timed out" in err_str
        or "timeout" in err_str
        or "deadline exceeded" in err_str
    )


def _build_history(
    conversation_history: List[Dict[str, str]],
) -> List[types.Content]:
    """
    Convert the existing BhoomiMitra conversation format
    into google-genai Content objects.
    """

    history = []

    for msg in conversation_history:
        role = msg.get("role", "user")
        text = msg.get("parts", "")

        if not text:
            continue

        history.append(
            types.Content(
                role=role,
                parts=[
                    types.Part.from_text(text=text)
                ],
            )
        )

    return history


async def generate_response(
    system_prompt: str,
    conversation_history: List[Dict[str, str]],
    user_message: str,
    timeout_seconds: Optional[float] = None,
    model_override: Optional[str] = None,
    allow_fallback: bool = True,
) -> Optional[str]:
    """
    Send a text message to Google Cloud Gemini.

    Keeps the existing BhoomiMitra function interface unchanged.
    """

    client = _ensure_initialized()
    settings = get_settings()

    if timeout_seconds is None:
        timeout_seconds = float(
            getattr(
                settings,
                "gemini_api_timeout_seconds",
                15.0,
            )
        )

    primary_model = (
        model_override
        or getattr(settings, "gemini_model", None)
        or "gemini-3.6-flash"
    )

    # Build model candidates
    candidate_models = [primary_model]

    if allow_fallback:
        for fallback in FALLBACK_MODELS:
            if fallback not in candidate_models:
                candidate_models.append(fallback)

    history = _build_history(conversation_history)

    # Add the current user message
    contents = history + [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(
                    text=user_message
                )
            ],
        )
    ]

    total_start_time = time.time()
    last_error = None
    timeout_count = 0

    for attempt_idx, model_name in enumerate(candidate_models):

        req_start_time = time.time()

        current_timeout = (
            timeout_seconds
            if attempt_idx == 0
            else min(timeout_seconds, 10.0)
        )

        logger.info(
            f"[GEMINI API REQUEST START] "
            f"(Attempt {attempt_idx + 1}/{len(candidate_models)})\n"
            f"  Model            : {model_name}\n"
            f"  Timeout          : {current_timeout}s\n"
            f"  Context History  : {len(history)} messages\n"
            f"  User Message     : "
            f"'{user_message[:120]}' "
            f"(len={len(user_message)})\n"
            f"  System Prompt Len: "
            f"{len(system_prompt)} chars"
        )

        try:

            config = types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.4,
                max_output_tokens=1024,
                top_p=0.9,
            )

            response = await asyncio.wait_for(
                client.aio.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=config,
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

            ai_text = (
                response.text.strip()
                if response.text
                else ""
            )

            logger.info(
                f"[GEMINI RESPONSE PARSED]\n"
                f"  Model Used       : {model_name}\n"
                f"  Output Length    : {len(ai_text)} chars\n"
                f"  Preview          : "
                f"'{ai_text[:120]}...'"
            )

            return ai_text

        except Exception as e:

            elapsed = time.time() - req_start_time
            last_error = e

            if _is_quota_exhausted_error(e):

                logger.warning(
                    f"[GEMINI QUOTA EXHAUSTED] "
                    f"Model {model_name} failed with quota "
                    f"exhaustion after {elapsed:.2f}s: "
                    f"{type(e).__name__} - {e}. "
                    f"Aborting model fallback chain."
                )

                break

            if _is_auth_error(e):

                logger.error(
                    f"[GEMINI AUTH ERROR] "
                    f"Model {model_name} failed with "
                    f"authentication error after "
                    f"{elapsed:.2f}s: "
                    f"{type(e).__name__} - {e}. "
                    f"Aborting fallback chain."
                )

                break

            if _is_timeout_error(e):

                timeout_count += 1

                logger.warning(
                    f"[GEMINI TIMEOUT] "
                    f"Model {model_name} timed out after "
                    f"{elapsed:.2f}s "
                    f"(limit={current_timeout}s). "
                    f"Trying next model..."
                )

                if timeout_count >= 2:
                    logger.warning(
                        f"[GEMINI TIMEOUT CEILING] "
                        f"{timeout_count} models timed out. "
                        f"Aborting fallback."
                    )
                    break

            else:

                logger.warning(
                    f"[GEMINI ERROR] "
                    f"Model {model_name} failed after "
                    f"{elapsed:.2f}s: "
                    f"{type(e).__name__} - {e}. "
                    f"Trying next model..."
                )

    total_elapsed = time.time() - total_start_time

    logger.exception(
        f"[GEMINI ALL MODELS EXHAUSTED] "
        f"All {len(candidate_models)} models failed "
        f"after {total_elapsed:.2f}s. "
        f"Last error: {last_error}"
    )

    if _is_timeout_error(last_error):
        raise TimeoutError(
            f"Gemini API timed out after "
            f"{total_elapsed:.1f}s across attempts"
        ) from last_error

    raise RuntimeError(
        f"Gemini SDK Error: {str(last_error)}"
    ) from last_error


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
    Send an image and optional text prompt to Gemini.

    Keeps the existing BhoomiMitra function interface unchanged.
    """

    client = _ensure_initialized()
    settings = get_settings()

    primary_model = (
        model_override
        or getattr(settings, "gemini_model", None)
        or "gemini-3.6-flash"
    )

    candidate_models = [primary_model]

    for fallback in FALLBACK_MODELS:
        if fallback not in candidate_models:
            candidate_models.append(fallback)

    history = _build_history(conversation_history)

    total_start_time = time.time()
    last_error = None
    timeout_count = 0

    for attempt_idx, model_name in enumerate(candidate_models):

        req_start_time = time.time()

        logger.info(
            f"[GEMINI MULTIMODAL REQUEST START] "
            f"(Attempt {attempt_idx + 1}/{len(candidate_models)})\n"
            f"  Model            : {model_name}\n"
            f"  Timeout          : {timeout_seconds}s\n"
            f"  Image Size       : "
            f"{len(image_bytes)} bytes ({mime_type})\n"
            f"  Caption          : '{user_message}'\n"
            f"  Context History  : {len(history)} messages"
        )

        try:

            message_parts = [
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type=mime_type,
                )
            ]

            if user_message:
                message_parts.append(
                    types.Part.from_text(
                        text=user_message
                    )
                )

            contents = history + [
                types.Content(
                    role="user",
                    parts=message_parts,
                )
            ]

            config = types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.4,
                max_output_tokens=1024,
                top_p=0.9,
                response_mime_type="application/json",
            )

            response = await asyncio.wait_for(
                client.aio.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=config,
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

            ai_text = (
                response.text.strip()
                if response.text
                else ""
            )

            logger.info(
                f"[GEMINI MULTIMODAL RESPONSE PARSED]\n"
                f"  Model Used       : {model_name}\n"
                f"  Output Length    : {len(ai_text)} chars\n"
                f"  Preview          : "
                f"'{ai_text[:120]}...'"
            )

            return ai_text

        except Exception as e:

            elapsed = time.time() - req_start_time
            last_error = e

            if _is_quota_exhausted_error(e):

                logger.warning(
                    f"[GEMINI MULTIMODAL QUOTA EXHAUSTED] "
                    f"Model {model_name} failed after "
                    f"{elapsed:.2f}s: "
                    f"{type(e).__name__} - {e}. "
                    f"Aborting fallback."
                )

                break

            if _is_auth_error(e):

                logger.error(
                    f"[GEMINI MULTIMODAL AUTH ERROR] "
                    f"Model {model_name} failed after "
                    f"{elapsed:.2f}s: "
                    f"{type(e).__name__} - {e}. "
                    f"Aborting fallback."
                )

                break

            if _is_timeout_error(e):

                timeout_count += 1

                logger.warning(
                    f"[GEMINI MULTIMODAL TIMEOUT] "
                    f"Model {model_name} timed out after "
                    f"{elapsed:.2f}s "
                    f"(limit={timeout_seconds}s). "
                    f"Trying next model..."
                )

                if timeout_count >= 2:
                    logger.warning(
                        f"[GEMINI MULTIMODAL TIMEOUT CEILING] "
                        f"{timeout_count} models timed out. "
                        f"Aborting fallback."
                    )
                    break

            else:

                logger.warning(
                    f"[GEMINI MULTIMODAL ERROR] "
                    f"Model {model_name} failed after "
                    f"{elapsed:.2f}s: "
                    f"{type(e).__name__} - {e}. "
                    f"Trying next model..."
                )

    total_elapsed = time.time() - total_start_time

    logger.exception(
        f"[GEMINI MULTIMODAL ALL MODELS EXHAUSTED] "
        f"All {len(candidate_models)} models failed "
        f"after {total_elapsed:.2f}s. "
        f"Last error: {last_error}"
    )

    if _is_timeout_error(last_error):
        raise TimeoutError(
            f"Gemini Multimodal API timed out after "
            f"{timeout_seconds}s across all attempts"
        ) from last_error

    raise RuntimeError(
        f"Gemini SDK Error: {str(last_error)}"
    ) from last_error