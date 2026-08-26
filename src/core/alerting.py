"""
BhoomiMitra AI — Founder Critical Alerting Service

Lightweight, non-blocking alerting mechanism for founder/on-call notifications.
Triggers on:
1. Meta WhatsApp API 401 Unauthorized (token expiration)
2. PostgreSQL health failures (503)
3. Delivery failure rate > 10%
4. Unhandled server exceptions

Features:
- Fire-and-forget async execution (never blocks farmer responses)
- Automatic in-memory rate-limiting / cooldown (15 mins per alert category)
- Zero secret/credential exposure
- Webhook agnostic (Discord, Telegram, Slack, or generic HTTP endpoint)
"""

import time
import httpx
from datetime import datetime, timezone
from enum import Enum
from typing import Optional, Dict, Any
from src.core.logging import logger
from src.config import get_settings


class AlertCategory(str, Enum):
    AUTH_FAILURE = "AUTH_FAILURE"
    DATABASE_DOWN = "DATABASE_DOWN"
    HIGH_DELIVERY_FAILURE = "HIGH_DELIVERY_FAILURE"
    UNHANDLED_EXCEPTION = "UNHANDLED_EXCEPTION"


class AlertSeverity(str, Enum):
    CRITICAL = "CRITICAL"
    WARNING = "WARNING"
    INFO = "INFO"


# In-memory cooldown tracking: {alert_category: timestamp_last_sent}
_ALERT_COOLDOWN_SECONDS = 900.0  # 15 minutes
_last_alert_times: Dict[str, float] = {}


def reset_alert_cooldowns() -> None:
    """Helper for unit tests to clear in-memory alert throttling state."""
    global _last_alert_times
    _last_alert_times.clear()


async def dispatch_founder_alert(
    category: AlertCategory,
    severity: AlertSeverity,
    component: str,
    summary: str,
    recommended_action: str,
    details: Optional[Dict[str, Any]] = None,
    force: bool = False,
) -> bool:
    """
    Dispatches a structured alert to the configured founder webhook.
    Returns True if an alert was dispatched/logged, False if throttled or failed.
    """
    try:
        settings = get_settings()
        now = time.time()
        cat_key = category.value if isinstance(category, AlertCategory) else str(category)

        # 1. Rate-limiting check
        if not force and cat_key in _last_alert_times:
            elapsed = now - _last_alert_times[cat_key]
            if elapsed < _ALERT_COOLDOWN_SECONDS:
                logger.debug(
                    f"[FOUNDER ALERT THROTTLED] Suppressed {cat_key} alert "
                    f"({int(_ALERT_COOLDOWN_SECONDS - elapsed)}s remaining in cooldown)"
                )
                return False

        # 2. Build sanitized payload
        utc_timestamp = datetime.now(timezone.utc).isoformat()
        clean_details = {}
        if details:
            for k, v in details.items():
                k_lower = str(k).lower()
                if any(s in k_lower for s in ["token", "password", "secret", "key", "auth"]):
                    clean_details[k] = "[REDACTED]"
                else:
                    clean_details[k] = str(v)[:200]

        payload = {
            "timestamp": utc_timestamp,
            "environment": settings.app_env,
            "service": "bhoomimitra-ai",
            "category": cat_key,
            "severity": severity.value if isinstance(severity, AlertSeverity) else str(severity),
            "component": component,
            "summary": summary,
            "recommended_action": recommended_action,
            "details": clean_details,
        }

        webhook_url = getattr(settings, "founder_alert_webhook_url", "").strip()

        if not webhook_url:
            logger.warning(
                f"[FOUNDER ALERT (LOCAL LOG ONLY)] Category: {cat_key} | Severity: {payload['severity']} | "
                f"Component: {component} | Summary: {summary} | Action: {recommended_action}"
            )
            _last_alert_times[cat_key] = now
            return True

        # 3. Non-blocking HTTP POST
        discord_payload = {
            "content": (
                f"🚨 **[BhoomiMitra {payload['severity']}] {cat_key}**\n"
                f"• **Component:** `{component}`\n"
                f"• **Summary:** {summary}\n"
                f"• **Action Required:** {recommended_action}\n"
                f"• **Time:** `{utc_timestamp}` | **Env:** `{settings.app_env}`"
            ),
            **payload,
        }

        async with httpx.AsyncClient(timeout=4.0) as client:
            resp = await client.post(webhook_url, json=discord_payload)
            if resp.status_code in [200, 204]:
                logger.info(f"[FOUNDER ALERT SENT] Successfully notified founder of {cat_key}")
                _last_alert_times[cat_key] = now
                return True
            else:
                logger.warning(f"[FOUNDER ALERT DISPATCH WARNING] Webhook returned status {resp.status_code}")
                return False
    except Exception as exc:
        logger.warning(f"[FOUNDER ALERT DISPATCH FAILED] Isolated error notifying founder: {exc}")
        return False
