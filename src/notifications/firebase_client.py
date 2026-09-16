"""
BhoomiMitra AI — Firebase Cloud Messaging Client

Provides safe, authenticated interaction with Firebase Cloud Messaging (FCM)
for high-priority Stock Siren push notifications.

Guarantees:
- Masked logging of FCM registration tokens (never exposes full tokens in logs).
- Lazy initialization of Firebase Admin SDK from environment variables or secret files.
- Fail-soft behavior if Firebase is unconfigured or credentials are missing.
- Identification and collection of stale/unregistered tokens for automatic deactivation.
"""

from typing import List, Dict, Tuple, Optional
import os
import json
import firebase_admin
from firebase_admin import credentials, messaging
from firebase_admin.exceptions import FirebaseError

from src.core.logging import logger
from src.config import get_settings
from src.notifications.schemas import mask_token

_firebase_app_initialized = False


def _get_firebase_app() -> Optional[firebase_admin.App]:
    """
    Lazily initialize and return the default Firebase Admin app.
    Returns None if credentials are not configured or initialization fails.
    """
    global _firebase_app_initialized
    if _firebase_app_initialized:
        try:
            return firebase_admin.get_app()
        except ValueError:
            pass

    settings = get_settings()
    cred = None

    # Option 1: Inline JSON string (e.g. from Render environment variable)
    json_str = getattr(settings, "firebase_credentials_json", "").strip()
    if json_str:
        try:
            cred_dict = json.loads(json_str)
            cred = credentials.Certificate(cred_dict)
            logger.info("[FCM CLIENT] Loaded Firebase credentials from FIREBASE_CREDENTIALS_JSON.")
        except Exception as json_err:
            logger.error(f"[FCM CLIENT] Failed to parse FIREBASE_CREDENTIALS_JSON: {json_err}")

    # Option 2: File path (e.g. /etc/secrets/firebase-sa.json mounted on Render)
    if not cred:
        file_path = getattr(settings, "firebase_credentials_path", "").strip()
        if file_path and os.path.exists(file_path):
            try:
                cred = credentials.Certificate(file_path)
                logger.info(f"[FCM CLIENT] Loaded Firebase credentials from file: {file_path}")
            except Exception as f_err:
                logger.error(f"[FCM CLIENT] Failed to load credentials from {file_path}: {f_err}")

    # Option 3: Fallback to GOOGLE_APPLICATION_CREDENTIALS if valid
    if not cred:
        gcp_cred = getattr(settings, "google_application_credentials", "").strip()
        if gcp_cred and os.path.exists(gcp_cred):
            try:
                cred = credentials.Certificate(gcp_cred)
                logger.info(f"[FCM CLIENT] Reusing GOOGLE_APPLICATION_CREDENTIALS for Firebase: {gcp_cred}")
            except Exception as g_err:
                logger.debug(f"[FCM CLIENT] Could not reuse GOOGLE_APPLICATION_CREDENTIALS: {g_err}")

    if not cred:
        logger.debug("[FCM CLIENT] Firebase credentials not configured. FCM push dispatch will be skipped.")
        return None

    try:
        app_options = {}
        project_id = getattr(settings, "firebase_project_id", "").strip()
        if project_id:
            app_options["projectId"] = project_id

        app = firebase_admin.initialize_app(cred, options=app_options)
        _firebase_app_initialized = True
        logger.info("[FCM CLIENT] Successfully initialized Firebase Admin App.")
        return app
    except ValueError:
        _firebase_app_initialized = True
        return firebase_admin.get_app()
    except Exception as init_err:
        logger.error(f"[FCM CLIENT] Firebase Admin initialization failed: {init_err}")
        return None


async def send_stock_siren_push(
    tokens: List[str],
    title: str,
    body: str,
    data_payload: Dict[str, str],
) -> Tuple[int, List[str]]:
    """
    Send high-priority Stock Siren notification to a list of FCM device tokens.

    Args:
        tokens: Target FCM registration tokens.
        title: Localized notification title.
        body: Factual notification body containing product and shop details.
        data_payload: String-keyed dictionary containing verified restock values
                      (alert_type='stock_siren', product_name, shop_name, etc.)

    Returns:
        Tuple of (success_count: int, invalid_tokens: List[str]).
    """
    if not tokens:
        return 0, []

    # Clean and filter tokens
    unique_tokens = list(dict.fromkeys(t.strip() for t in tokens if t and t.strip()))
    if not unique_tokens:
        return 0, []

    app = _get_firebase_app()
    if not app:
        logger.info("[FCM CLIENT] Skipping FCM push dispatch: Firebase App is not initialized.")
        return 0, []

    # Android High-Importance Config for custom Stock Siren notification channel
    android_config = messaging.AndroidConfig(
        priority="high",
        notification=messaging.AndroidNotification(
            channel_id="stock_siren",
            sound="stock_siren",
            click_action="OPEN_STOCK_SIREN",
            default_sound=False,
            default_vibrate_timings=False,
        ),
    )

    # Convert all payload values to string to comply with FCM data specifications
    safe_data = {str(k): str(v) for k, v in data_payload.items()}
    # Always ensure alert_type and localized strings are available in data payload
    safe_data.setdefault("alert_type", "stock_siren")
    safe_data["title"] = title
    safe_data["body"] = body

    message = messaging.MulticastMessage(
        tokens=unique_tokens,
        notification=messaging.Notification(
            title=title,
            body=body,
        ),
        data=safe_data,
        android=android_config,
    )

    success_count = 0
    invalid_tokens: List[str] = []

    try:
        response = messaging.send_each_for_multicast(message)
        logger.info(
            f"[FCM CLIENT] Sent Stock Siren push: {response.success_count} succeeded, "
            f"{response.failure_count} failed out of {len(unique_tokens)} recipient(s)."
        )

        for idx, resp in enumerate(response.responses):
            token = unique_tokens[idx]
            if resp.success:
                success_count += 1
            else:
                err = resp.exception
                logger.warning(
                    f"[FCM CLIENT] Push delivery failed for token {mask_token(token)}: {err}"
                )
                # Check for token expiration/invalidation errors
                if isinstance(err, (messaging.UnregisteredError, messaging.SenderIdMismatchError)):
                    invalid_tokens.append(token)
                elif hasattr(err, "code") and err.code in ("UNREGISTERED", "INVALID_ARGUMENT"):
                    invalid_tokens.append(token)

    except FirebaseError as fb_err:
        logger.error(f"[FCM CLIENT] Firebase API error sending multicast message: {fb_err}")
    except Exception as exc:
        logger.exception(f"[FCM CLIENT] Unexpected error during FCM push send: {exc}")

    return success_count, invalid_tokens
