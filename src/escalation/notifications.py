"""
BhoomiMitra AI — Expert Escalation Outbound Notifications

Dispatches secure, asynchronous WhatsApp notifications to verified agricultural
officers when a new high-priority escalation ticket is generated.
Reuses WhatsApp Cloud API client from src/gateway/whatsapp_client.py.
"""

from typing import Optional
from src.config import get_settings
from src.gateway.whatsapp_client import send_text_message
from src.core.logging import logger


def build_expert_alert_message(
    ticket_id: str,
    topic: str,
    region: str,
    reason: str,
    farmer_phone: Optional[str] = None,
    is_group: bool = False,
) -> str:
    """
    Build structured, triage-focused WhatsApp alert for agricultural officers.

    PRIVACY SAFEGUARD:
    - When is_group is True: farmer_phone, raw history, GPS, and personal details
      are strictly omitted.
    - When is_group is False (direct message to verified officer): farmer_phone is
      provided solely for direct callback within the SLA.
    """
    reason_labels = {
        "hazard": "⚠️ Urgent Chemical Hazard / Toxicity Caution",
        "inspection": "🌾 Field Crop Damage / Inspection Request",
        "explicit": "👨‍🌾 Farmer Requested Officer Consultation",
    }
    reason_display = reason_labels.get(reason.lower(), f"🚨 {reason.capitalize()} Escalation")

    lines = [
        "🚨 *BhoomiMitra Expert Escalation Alert*",
        "━━━━━━━━━━━━━━━━━━━━━━",
        f"📋 *Ticket ID:* #{ticket_id}",
        f"⚠️ *Priority:* {reason_display}",
        f"📍 *Region:* {region or 'District Agriculture Office'}",
        f"🌾 *Inquiry Summary:* {topic[:120]}",
    ]

    # Privacy Rule: Include farmer callback number ONLY for direct 1:1 expert alerts
    if not is_group and farmer_phone:
        lines.append(f"📞 *Farmer Contact:* {farmer_phone}")

    lines.extend([
        "━━━━━━━━━━━━━━━━━━━━━━",
        "👉 *Action Required:* Please review this ticket in the dashboard and connect with the farmer within 30–60 minutes.",
    ])

    return "\n".join(lines)


async def notify_expert_escalation_ticket(
    ticket_id: str,
    topic: str,
    region: str,
    reason: str,
    expert_phone: Optional[str],
    farmer_phone: Optional[str] = None,
    group_id: Optional[str] = None,
) -> Optional[str]:
    """
    Asynchronously notify a verified on-duty expert via WhatsApp.
    Non-blocking: any delivery failure or network issue is caught and logged,
    ensuring ticket creation and farmer responses are never aborted.
    """
    if not expert_phone and not group_id:
        logger.info(f"[EXPERT ALERT] No expert phone or group configured for ticket {ticket_id}. Skipping dispatch.")
        return None

    settings = get_settings()
    last_msg_id = None

    # 1. Direct Alert to Verified Expert Phone (with farmer callback contact)
    if expert_phone:
        raw_expert_phone = expert_phone.replace("+", "").replace(" ", "").replace("-", "").strip()
        if raw_expert_phone.isdigit() and len(raw_expert_phone) >= 10:
            direct_msg = build_expert_alert_message(
                ticket_id=ticket_id,
                topic=topic,
                region=region,
                reason=reason,
                farmer_phone=farmer_phone,
                is_group=False,
            )
            try:
                logger.info(f"[EXPERT ALERT DISPATCH] Sending direct alert for ticket {ticket_id} to expert ({raw_expert_phone[:4]}***).")
                msg_id = await send_text_message(to_phone=raw_expert_phone, message_text=direct_msg)
                if msg_id:
                    last_msg_id = msg_id
                    logger.info(f"[EXPERT ALERT DELIVERED] Direct alert sent to expert {raw_expert_phone} (wa_id={msg_id}).")
            except Exception as err:
                logger.warning(f"[EXPERT ALERT NON-BLOCKING FAILURE] Direct alert delivery failed for ticket {ticket_id}: {err}")

    # 2. Group Alert (PRIVACY COMPLIANT: NEVER includes farmer phone or raw data)
    target_group = group_id or settings.expert_whatsapp_group_id
    if target_group and target_group.strip():
        group_msg = build_expert_alert_message(
            ticket_id=ticket_id,
            topic=topic,
            region=region,
            reason=reason,
            farmer_phone=None,
            is_group=True,
        )
        try:
            clean_group_id = target_group.replace("+", "").replace(" ", "").replace("-", "").strip()
            logger.info(f"[EXPERT GROUP ALERT DISPATCH] Sending privacy-safe broadcast alert for ticket {ticket_id} to group {clean_group_id}.")
            group_res = await send_text_message(to_phone=clean_group_id, message_text=group_msg)
            if group_res:
                last_msg_id = last_msg_id or group_res
        except Exception as group_err:
            logger.warning(f"[EXPERT GROUP ALERT NON-BLOCKING FAILURE] Group alert delivery failed for ticket {ticket_id}: {group_err}")

    return last_msg_id
