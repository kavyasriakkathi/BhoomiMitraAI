"""
Authentication Router for user registration, login, logout, and profile endpoints.
"""

from fastapi import APIRouter, Depends, Response, status
from typing import Optional

from src.auth.dependencies import (
    get_auth_service,
    get_current_active_user,
    get_optional_current_user,
)
from src.auth.schemas import (
    LoginRequest,
    MessageResponse,
    TokenResponse,
    UserRegisterRequest,
    UserResponse,
)
from src.auth.service import AuthService
from src.config import get_settings
from src.core.models import UserAccount

router = APIRouter()


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(
    payload: UserRegisterRequest,
    current_user: Optional[UserAccount] = Depends(get_optional_current_user),
    service: AuthService = Depends(get_auth_service),
):
    """
    Register a new user account:
    - Shop Owners and Experts can self-register with a valid linked shop_id or expert_id.
    - Admin registration requires active Admin credentials or a valid admin_registration_key.
    """
    user = await service.register_user(payload, current_user=current_user)
    return UserResponse.model_validate(user)


@router.post("/login", response_model=TokenResponse)
async def login(
    payload: LoginRequest,
    response: Response,
    service: AuthService = Depends(get_auth_service),
):
    """
    Authenticate user credentials, return a JWT token, and set a secure HttpOnly session cookie.
    """
    settings = get_settings()
    user, access_token = await service.authenticate_user(payload)

    # Set secure HttpOnly cookie for browser sessions
    response.set_cookie(
        key=settings.auth_cookie_name,
        value=access_token,
        httponly=True,
        max_age=settings.access_token_expire_minutes * 60,
        expires=settings.access_token_expire_minutes * 60,
        secure=settings.cookie_secure,
        samesite=settings.auth_cookie_samesite,
        path="/",
    )

    return TokenResponse(
        access_token=access_token,
        token_type="bearer",
        user=UserResponse.model_validate(user),
    )


@router.post("/logout", response_model=MessageResponse)
async def logout(response: Response):
    """
    Log out by clearing the HttpOnly authentication cookie.
    """
    settings = get_settings()
    response.delete_cookie(
        key=settings.auth_cookie_name,
        path="/",
        secure=settings.cookie_secure,
        samesite=settings.auth_cookie_samesite,
    )
    return MessageResponse(
        success=True,
        message="Successfully logged out",
    )


@router.get("/me", response_model=UserResponse)
async def get_current_user_profile(
    current_user: UserAccount = Depends(get_current_active_user),
):
    """
    Get profile information for the currently authenticated user.
    """
    return UserResponse.model_validate(current_user)
