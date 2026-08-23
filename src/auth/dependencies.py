"""
FastAPI dependency injection utilities for authentication and Role-Based Access Control (RBAC).
"""

from typing import Callable, Optional
from uuid import UUID
from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from src.auth.constants import UserRole
from src.auth.repository import AuthRepository
from src.auth.service import AuthService
from src.config import get_settings
from src.core.database import get_db
from src.core.exceptions import BhoomiMitraException
from src.core.models import UserAccount


def get_auth_repository(db: AsyncSession = Depends(get_db)) -> AuthRepository:
    """Provide an AuthRepository instance bound to the request's DB session."""
    return AuthRepository(db)


def get_auth_service(
    repo: AuthRepository = Depends(get_auth_repository),
) -> AuthService:
    """Provide an AuthService instance."""
    return AuthService(repo)


def get_token_from_request(request: Request) -> Optional[str]:
    """
    Extract authentication token from Authorization Bearer header or HttpOnly cookie.
    Bearer header takes precedence for API clients; cookie is used for web browser sessions.
    """
    settings = get_settings()

    # Check Authorization header: "Bearer <token>"
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header[7:].strip()
        if token:
            return token

    # Check HttpOnly cookie
    cookie_token = request.cookies.get(settings.auth_cookie_name)
    if cookie_token:
        return cookie_token

    return None


async def get_current_user(
    token: Optional[str] = Depends(get_token_from_request),
    service: AuthService = Depends(get_auth_service),
) -> UserAccount:
    """
    Dependency that enforces authentication and returns the current user account.
    """
    if not token:
        raise BhoomiMitraException(
            message="Authentication required. Please provide a valid Bearer token or login cookie.",
            status_code=401,
        )

    return await service.get_user_from_token(token)


async def get_optional_current_user(
    token: Optional[str] = Depends(get_token_from_request),
    service: AuthService = Depends(get_auth_service),
) -> Optional[UserAccount]:
    """
    Optional authentication dependency — returns UserAccount if valid token present, else None.
    """
    if not token:
        return None
    try:
        return await service.get_user_from_token(token)
    except Exception:
        return None


async def get_current_active_user(
    current_user: UserAccount = Depends(get_current_user),
) -> UserAccount:
    """
    Dependency verifying the current user is active.
    """
    if not current_user.is_active:
        raise BhoomiMitraException(
            message="User account has been deactivated",
            status_code=403,
        )
    return current_user


def require_roles(*allowed_roles: str) -> Callable:
    """
    Factory creating an RBAC dependency that permits only specified roles (or Admin).
    """
    async def role_checker(
        current_user: UserAccount = Depends(get_current_active_user),
    ) -> UserAccount:
        # Admin has superuser privileges across all routes
        if current_user.role == UserRole.ADMIN.value:
            return current_user

        if current_user.role not in allowed_roles:
            raise BhoomiMitraException(
                message=f"Access forbidden: User role '{current_user.role}' lacks required permissions",
                status_code=403,
            )
        return current_user

    return role_checker


# Pre-configured RBAC dependencies
require_admin = require_roles(UserRole.ADMIN.value)
require_expert = require_roles(UserRole.ADMIN.value, UserRole.EXPERT.value)
require_shop_owner = require_roles(UserRole.ADMIN.value, UserRole.SHOP_OWNER.value)


def verify_shop_access(current_user: UserAccount, shop_id: UUID) -> None:
    """
    Enforce shop-level tenant isolation:
    - Admin can access any shop.
    - Shop Owner can only access their assigned shop.
    - Other roles are forbidden.
    """
    if current_user.role == UserRole.ADMIN.value:
        return

    if current_user.role == UserRole.SHOP_OWNER.value:
        if current_user.shop_id and current_user.shop_id == shop_id:
            return
        raise BhoomiMitraException(
            message="Access forbidden: You do not have permission to access or modify data for another shop",
            status_code=403,
        )

    raise BhoomiMitraException(
        message="Access forbidden: Only shop owners or administrators can perform this action",
        status_code=403,
    )


def verify_expert_access(current_user: UserAccount, expert_id: UUID) -> None:
    """
    Enforce expert-level isolation:
    - Admin can access any expert resource.
    - Expert can only access their assigned expert resource.
    - Other roles are forbidden.
    """
    if current_user.role == UserRole.ADMIN.value:
        return

    if current_user.role == UserRole.EXPERT.value:
        if current_user.expert_id and current_user.expert_id == expert_id:
            return
        raise BhoomiMitraException(
            message="Access forbidden: You do not have permission to access or modify data for another expert",
            status_code=403,
        )

    raise BhoomiMitraException(
        message="Access forbidden: Only agricultural experts or administrators can perform this action",
        status_code=403,
    )
