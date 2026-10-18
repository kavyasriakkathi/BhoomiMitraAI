from __future__ import annotations

import re
from typing import Any

from sqlalchemy import select

from src.core.models import Farmer, FarmerProfile


def normalize_whatsapp_phone(phone: Any) -> str:
    """Normalize an Indian WhatsApp number to +91XXXXXXXXXX."""
    if phone is None:
        raise ValueError("Phone number is required")

    digits = re.sub(r"\D", "", str(phone))

    if digits.startswith("91") and len(digits) == 12:
        return f"+{digits}"

    if len(digits) == 10:
        return f"+91{digits}"

    raise ValueError(f"Invalid Indian phone number: {phone}")


async def onboard_single_farmer(
    *,
    db,
    name: str,
    phone: str,
    village: str | None = None,
    language: str = "te",
    crop: str | None = None,
) -> dict[str, Any]:
    """Create or update a pilot farmer and their profile."""

    normalized_phone = normalize_whatsapp_phone(phone)

    result = await db.execute(
        select(Farmer).where(Farmer.phone_number == normalized_phone)
    )
    farmer = result.scalar_one_or_none()

    if farmer is None:
        farmer = Farmer(
            phone_number=normalized_phone,
            preferred_language=language,
            is_active=True,
        )
        db.add(farmer)
        await db.flush()
        status = "CREATED"
    else:
        farmer.preferred_language = language
        status = "UPDATED"

    profile_result = await db.execute(
        select(FarmerProfile).where(
            FarmerProfile.farmer_id == farmer.id
        )
    )
    profile = profile_result.scalar_one_or_none()

    if profile is None:
        profile = FarmerProfile(
            farmer_id=farmer.id,
            full_name=name,
            current_crop=crop,
        )
        db.add(profile)
    else:
        profile.full_name = name
        profile.current_crop = crop

    await db.flush()

    return {
        "status": status,
        "phone": farmer.phone_number,
        "language": farmer.preferred_language,
        "crop": profile.current_crop,
        "village": village,
        "farmer_id": str(farmer.id),
    }
