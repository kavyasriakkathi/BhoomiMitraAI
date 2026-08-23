"""
Authentication Service providing user registration, login verification, and token generation.
"""

from typing import Optional, Tuple
from uuid import UUID
from src.auth.constants import UserRole
from src.auth.repository import AuthRepository
from src.auth.schemas import LoginRequest, UserRegisterRequest
from src.auth.security import create_access_token, decode_access_token, hash_password, verify_password
from src.config import get_settings
from src.core.exceptions import BhoomiMitraException
from src.core.logging import logger
from src.core.models import UserAccount


class AuthService:
    """Business logic service for user authentication and authorization."""

    def __init__(self, repo: AuthRepository):
        self.repo = repo

    async def register_user(
        self,
        payload: UserRegisterRequest,
        current_user: Optional[UserAccount] = None,
    ) -> UserAccount:
        """
        Register a new user account after validating uniqueness and role associations.
        Admin account creation requires an active admin session or a valid admin_registration_key.
        """
        normalized_email = payload.email.lower().strip()

        # Check email uniqueness
        existing_user = await self.repo.get_by_email(normalized_email)
        if existing_user:
            raise BhoomiMitraException(
                message=f"An account with email '{normalized_email}' already exists",
                status_code=400,
            )

        # Protect Admin account creation
        if payload.role == UserRole.ADMIN:
            settings = get_settings()
            is_active_admin = current_user is not None and current_user.role == UserRole.ADMIN.value
            has_valid_admin_key = (
                bool(settings.admin_registration_key)
                and payload.admin_creation_key == settings.admin_registration_key
            )
            if not (is_active_admin or has_valid_admin_key):
                raise BhoomiMitraException(
                    message="Creating an administrator account requires administrator privileges or a valid admin creation key",
                    status_code=403,
                )

        # Validate Expert linking
        if payload.role == UserRole.EXPERT:
            if not payload.expert_id:
                raise BhoomiMitraException(
                    message="Expert user registration requires a valid 'expert_id'",
                    status_code=400,
                )
            expert_exists = await self.repo.check_expert_exists(payload.expert_id)
            if not expert_exists:
                raise BhoomiMitraException(
                    message=f"Expert with ID '{payload.expert_id}' does not exist",
                    status_code=404,
                )
            existing_expert_user = await self.repo.get_by_expert_id(payload.expert_id)
            if existing_expert_user:
                raise BhoomiMitraException(
                    message=f"An account is already associated with Expert ID '{payload.expert_id}'",
                    status_code=400,
                )

        # Validate Shop Owner linking
        if payload.role == UserRole.SHOP_OWNER:
            if not payload.shop_id:
                raise BhoomiMitraException(
                    message="Shop Owner user registration requires a valid 'shop_id'",
                    status_code=400,
                )
            shop_exists = await self.repo.check_shop_exists(payload.shop_id)
            if not shop_exists:
                raise BhoomiMitraException(
                    message=f"Shop with ID '{payload.shop_id}' does not exist",
                    status_code=404,
                )
            existing_shop_user = await self.repo.get_by_shop_id(payload.shop_id)
            if existing_shop_user:
                raise BhoomiMitraException(
                    message=f"An account is already associated with Shop ID '{payload.shop_id}'",
                    status_code=400,
                )

        # Admin should not have expert_id or shop_id
        expert_id = payload.expert_id if payload.role == UserRole.EXPERT else None
        shop_id = payload.shop_id if payload.role == UserRole.SHOP_OWNER else None

        # Securely hash password using Argon2
        hashed_password = hash_password(payload.password)

        new_user = UserAccount(
            email=normalized_email,
            password_hash=hashed_password,
            role=payload.role.value,
            expert_id=expert_id,
            shop_id=shop_id,
            is_active=True,
        )

        user = await self.repo.create_user(new_user)
        logger.info(f"Registered new user '{user.email}' with role '{user.role}'")
        return user

    async def authenticate_user(self, payload: LoginRequest) -> Tuple[UserAccount, str]:
        """
        Verify login credentials, authenticate user, and issue a JWT access token.
        """
        normalized_email = payload.email.lower().strip()
        user = await self.repo.get_by_email(normalized_email)

        if not user or not verify_password(payload.password, user.password_hash):
            logger.warning(f"Failed login attempt for email '{normalized_email}'")
            raise BhoomiMitraException(
                message="Invalid email or password",
                status_code=401,
            )

        if not user.is_active:
            logger.warning(f"Inactive user '{normalized_email}' attempted login")
            raise BhoomiMitraException(
                message="User account is deactivated. Contact an administrator.",
                status_code=403,
            )

        token_data = {
            "sub": str(user.id),
            "email": user.email,
            "role": user.role,
            "expert_id": str(user.expert_id) if user.expert_id else None,
            "shop_id": str(user.shop_id) if user.shop_id else None,
        }

        access_token = create_access_token(token_data)
        logger.info(f"User '{user.email}' ({user.role}) logged in successfully")
        return user, access_token

    async def get_user_from_token(self, token: str) -> UserAccount:
        """
        Validate JWT token and return the associated active user account.
        """
        payload = decode_access_token(token)
        user_id_str = payload.get("sub")
        if not user_id_str:
            raise BhoomiMitraException(
                message="Invalid token payload: missing subject identifier",
                status_code=401,
            )

        try:
            user_id = UUID(user_id_str)
        except ValueError:
            raise BhoomiMitraException(
                message="Invalid user identifier format in token",
                status_code=401,
            )

        user = await self.repo.get_by_id(user_id)
        if not user:
            raise BhoomiMitraException(
                message="User associated with token no longer exists",
                status_code=401,
            )

        if not user.is_active:
            raise BhoomiMitraException(
                message="User account has been deactivated",
                status_code=403,
            )

        return user
