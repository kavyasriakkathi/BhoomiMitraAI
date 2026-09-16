"""
BhoomiMitra AI — Push Notification Router

Endpoints for Android devices to register, refresh, and deactivate
FCM device tokens associated with a farmer profile.
"""

from typing import Optional
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from src.core.database import get_db
from src.core.logging import logger
from src.core.models import Farmer, UserAccount
from src.auth.dependencies import get_optional_current_user
from src.notifications.schemas import (
    PushTokenRegisterRequest,
    PushTokenDeactivateRequest,
    PushTokenResponse,
    mask_token,
)
from src.notifications.repository import PushTokenRepository

router = APIRouter(prefix="/notifications", tags=["Notifications"])


@router.post(
    "/tokens",
    response_model=PushTokenResponse,
    status_code=status.HTTP_200_OK,
    summary="Register or refresh an Android FCM device token for a farmer",
)
async def register_push_token(
    payload: PushTokenRegisterRequest,
    db: AsyncSession = Depends(get_db),
    current_user: Optional[UserAccount] = Depends(get_optional_current_user),
):
    """
    Registers an FCM device token for a farmer:
    - If user is authenticated via Bearer token / cookie, links token to current user's farmer profile.
    - If unauthenticated, uses payload.phone_number to link or register the farmer.
    """
    farmer: Optional[Farmer] = None

    if current_user and current_user.phone_number:
        res = await db.execute(select(Farmer).where(Farmer.phone_number == current_user.phone_number))
        farmer = res.scalar_one_or_none()

    if not farmer and payload.phone_number:
        clean_phone = payload.phone_number.strip()
        res = await db.execute(select(Farmer).where(Farmer.phone_number == clean_phone))
        farmer = res.scalar_one_or_none()
        if not farmer:
            farmer = Farmer(phone_number=clean_phone, preferred_language="te", is_active=True)
            db.add(farmer)
            await db.commit()
            await db.refresh(farmer)

    if not farmer:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not identify farmer. Provide a valid phone_number or authenticate with a Bearer token.",
        )

    repo = PushTokenRepository(db)
    token_record = await repo.register_or_update_token(
        farmer_id=farmer.id,
        token=payload.token,
        platform=payload.platform,
        device_model=payload.device_model,
    )

    return PushTokenResponse(
        id=token_record.id,
        farmer_id=token_record.farmer_id,
        token_masked=mask_token(token_record.token),
        platform=token_record.platform,
        is_active=token_record.is_active,
        created_at=token_record.created_at,
        updated_at=token_record.updated_at,
    )


@router.delete(
    "/tokens",
    status_code=status.HTTP_200_OK,
    summary="Deactivate an FCM device token",
)
async def deactivate_push_token(
    payload: PushTokenDeactivateRequest,
    db: AsyncSession = Depends(get_db),
):
    """Deactivate an FCM token when farmer revokes permission or logs out."""
    repo = PushTokenRepository(db)
    deactivated = await repo.deactivate_token(payload.token)
    return {
        "status": "success",
        "deactivated": deactivated,
        "token_masked": mask_token(payload.token),
    }
