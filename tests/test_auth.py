"""
Unit and Integration Tests for Authentication and Role-Based Access Control (RBAC).
"""

from datetime import timedelta
from uuid import uuid4
import pytest
from fastapi.testclient import TestClient

from src.auth.constants import UserRole
from src.auth.dependencies import get_auth_service, get_current_active_user, get_current_user
from src.auth.schemas import (
    LoginRequest,
    UserRegisterRequest,
    UserResponse,
)
from src.auth.security import (
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)
from src.auth.service import AuthService
from src.core.exceptions import BhoomiMitraException
from src.core.models import UserAccount
from src.main import app

client = TestClient(app)


# =====================================================================
# 1. Argon2 Password Hashing Tests
# =====================================================================

def test_argon2_hashing_and_verification():
    """Verify that Argon2 hashing creates non-reversible secure hashes and validates correctly."""
    plain_password = "SuperSecretPassword123!"
    hashed = hash_password(plain_password)

    # Must not store plaintext
    assert hashed != plain_password
    assert hashed.startswith("$argon2id$")

    # Correct password verifies True
    assert verify_password(plain_password, hashed) is True

    # Incorrect password verifies False
    assert verify_password("WrongPassword123!", hashed) is False
    assert verify_password("", hashed) is False
    assert verify_password(plain_password, "") is False


def test_argon2_empty_password_raises():
    """Empty passwords must raise ValueError."""
    with pytest.raises(ValueError):
        hash_password("")


# =====================================================================
# 2. JWT Token Generation & Validation Tests
# =====================================================================

def test_jwt_create_and_decode():
    """Verify JWT token encoding and decoding."""
    user_id = str(uuid4())
    data = {
        "sub": user_id,
        "email": "farmer.admin@bhoomimitra.ai",
        "role": UserRole.ADMIN.value,
    }

    token = create_access_token(data)
    assert isinstance(token, str)

    payload = decode_access_token(token)
    assert payload["sub"] == user_id
    assert payload["email"] == "farmer.admin@bhoomimitra.ai"
    assert payload["role"] == "admin"
    assert "exp" in payload
    assert "iat" in payload

    # Verify short-lived token expiration (approx 15 mins = 900 secs)
    duration = payload["exp"] - payload["iat"]
    assert 800 <= duration <= 1000

    # Ensure sensitive data is not leaked in token claims
    assert "password" not in payload
    assert "password_hash" not in payload


def test_jwt_expired_token_raises():
    """Expired tokens must raise 401 BhoomiMitraException."""
    user_id = str(uuid4())
    data = {"sub": user_id, "email": "test@bhoomimitra.ai"}

    # Create expired token (-5 minutes)
    expired_token = create_access_token(data, expires_delta=timedelta(minutes=-5))

    with pytest.raises(BhoomiMitraException) as exc_info:
        decode_access_token(expired_token)
    assert exc_info.value.status_code == 401


def test_jwt_invalid_token_raises():
    """Tampered or invalid tokens must raise 401 BhoomiMitraException."""
    with pytest.raises(BhoomiMitraException) as exc_info:
        decode_access_token("invalid.token.signature")
    assert exc_info.value.status_code == 401


# =====================================================================
# 3. Registration Service & API Tests
# =====================================================================

@pytest.mark.asyncio
async def test_unauthenticated_user_cannot_register_as_admin():
    """Arbitrary public user cannot create an admin account without authorization."""
    mock_repo = pytest.importorskip("unittest.mock").AsyncMock()
    mock_repo.get_by_email.return_value = None

    service = AuthService(mock_repo)
    req = UserRegisterRequest(
        email="hacker@bhoomimitra.ai",
        password="ValidPassword123!",
        role=UserRole.ADMIN,
    )

    with pytest.raises(BhoomiMitraException) as exc_info:
        await service.register_user(req, current_user=None)
    assert exc_info.value.status_code == 403
    assert "requires administrator privileges" in exc_info.value.message


@pytest.mark.asyncio
async def test_authenticated_admin_can_register_new_admin():
    """An active admin user can successfully create another admin account."""
    mock_repo = pytest.importorskip("unittest.mock").AsyncMock()
    mock_repo.get_by_email.return_value = None
    mock_repo.create_user.side_effect = lambda u: u

    active_admin = UserAccount(id=uuid4(), email="admin@bhoomimitra.ai", role="admin", is_active=True)
    service = AuthService(mock_repo)
    req = UserRegisterRequest(
        email="newadmin@bhoomimitra.ai",
        password="ValidPassword123!",
        role=UserRole.ADMIN,
    )

    user = await service.register_user(req, current_user=active_admin)
    assert user.email == "newadmin@bhoomimitra.ai"
    assert user.role == "admin"


@pytest.mark.asyncio
async def test_bootstrap_admin_with_valid_creation_key(monkeypatch):
    """An unauthenticated deployment script can bootstrap an admin with a valid secret key."""
    from src.config import get_settings
    settings = get_settings()
    monkeypatch.setattr(settings, "admin_registration_key", "secret-bootstrap-key-2026")

    mock_repo = pytest.importorskip("unittest.mock").AsyncMock()
    mock_repo.get_by_email.return_value = None
    mock_repo.create_user.side_effect = lambda u: u

    service = AuthService(mock_repo)
    req = UserRegisterRequest(
        email="bootstrap@bhoomimitra.ai",
        password="ValidPassword123!",
        role=UserRole.ADMIN,
        admin_creation_key="secret-bootstrap-key-2026",
    )

    user = await service.register_user(req, current_user=None)
    assert user.email == "bootstrap@bhoomimitra.ai"
    assert user.role == "admin"


@pytest.mark.asyncio
async def test_bootstrap_admin_with_invalid_creation_key(monkeypatch):
    """Invalid secret key fails admin registration with 403."""
    from src.config import get_settings
    settings = get_settings()
    monkeypatch.setattr(settings, "admin_registration_key", "secret-bootstrap-key-2026")

    mock_repo = pytest.importorskip("unittest.mock").AsyncMock()
    mock_repo.get_by_email.return_value = None

    service = AuthService(mock_repo)
    req = UserRegisterRequest(
        email="bootstrap@bhoomimitra.ai",
        password="ValidPassword123!",
        role=UserRole.ADMIN,
        admin_creation_key="wrong-key",
    )

    with pytest.raises(BhoomiMitraException) as exc_info:
        await service.register_user(req, current_user=None)
    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
async def test_register_duplicate_email_raises():
    """Registering with an existing email must raise 400."""
    mock_repo = pytest.importorskip("unittest.mock").AsyncMock()
    existing_user = UserAccount(email="existing@bhoomimitra.ai", role="admin")
    mock_repo.get_by_email.return_value = existing_user

    service = AuthService(mock_repo)
    req = UserRegisterRequest(
        email="existing@bhoomimitra.ai",
        password="ValidPassword123!",
        role=UserRole.SHOP_OWNER,
        shop_id=uuid4(),
    )

    with pytest.raises(BhoomiMitraException) as exc_info:
        await service.register_user(req)
    assert exc_info.value.status_code == 400
    assert "already exists" in exc_info.value.message


@pytest.mark.asyncio
async def test_register_shop_owner_missing_shop_id():
    """Shop owner registration without shop_id must raise 400."""
    mock_repo = pytest.importorskip("unittest.mock").AsyncMock()
    mock_repo.get_by_email.return_value = None

    service = AuthService(mock_repo)
    req = UserRegisterRequest(
        email="shop@bhoomimitra.ai",
        password="ValidPassword123!",
        role=UserRole.SHOP_OWNER,
        shop_id=None,
    )

    with pytest.raises(BhoomiMitraException) as exc_info:
        await service.register_user(req)
    assert exc_info.value.status_code == 400
    assert "requires a valid 'shop_id'" in exc_info.value.message


@pytest.mark.asyncio
async def test_register_shop_owner_nonexistent_shop():
    """Shop owner registration with nonexistent shop_id must raise 404."""
    mock_repo = pytest.importorskip("unittest.mock").AsyncMock()
    mock_repo.get_by_email.return_value = None
    mock_repo.check_shop_exists.return_value = False

    service = AuthService(mock_repo)
    target_shop_id = uuid4()
    req = UserRegisterRequest(
        email="shop@bhoomimitra.ai",
        password="ValidPassword123!",
        role=UserRole.SHOP_OWNER,
        shop_id=target_shop_id,
    )

    with pytest.raises(BhoomiMitraException) as exc_info:
        await service.register_user(req)
    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_register_expert_missing_expert_id():
    """Expert registration without expert_id must raise 400."""
    mock_repo = pytest.importorskip("unittest.mock").AsyncMock()
    mock_repo.get_by_email.return_value = None

    service = AuthService(mock_repo)
    req = UserRegisterRequest(
        email="expert@bhoomimitra.ai",
        password="ValidPassword123!",
        role=UserRole.EXPERT,
        expert_id=None,
    )

    with pytest.raises(BhoomiMitraException) as exc_info:
        await service.register_user(req)
    assert exc_info.value.status_code == 400
    assert "requires a valid 'expert_id'" in exc_info.value.message


# =====================================================================
# 4. Login & Authentication API Tests
# =====================================================================

def test_api_register_and_login_flow():
    """Test API endpoint registration, login with HttpOnly cookie, /me, and logout."""
    from unittest.mock import AsyncMock

    mock_service = AsyncMock(spec=AuthService)
    app.dependency_overrides[get_auth_service] = lambda: mock_service

    try:
        user_id = uuid4()
        user_obj = UserAccount(
            id=user_id,
            email="agriexpert@bhoomimitra.ai",
            password_hash=hash_password("ValidPassword123!"),
            role="expert",
            is_active=True,
            expert_id=uuid4(),
        )

        mock_service.register_user.return_value = user_obj
        mock_service.authenticate_user.return_value = (user_obj, "mocked-jwt-token-12345")

        # 1. Register
        reg_payload = {
            "email": "agriexpert@bhoomimitra.ai",
            "password": "ValidPassword123!",
            "role": "expert",
            "expert_id": str(user_obj.expert_id),
        }
        reg_resp = client.post("/auth/register", json=reg_payload)
        assert reg_resp.status_code == 201
        assert reg_resp.json()["email"] == "agriexpert@bhoomimitra.ai"
        assert reg_resp.json()["role"] == "expert"

        # 2. Login
        login_payload = {
            "email": "agriexpert@bhoomimitra.ai",
            "password": "ValidPassword123!",
        }
        login_resp = client.post("/auth/login", json=login_payload)
        assert login_resp.status_code == 200
        data = login_resp.json()
        assert data["access_token"] == "mocked-jwt-token-12345"
        assert data["token_type"] == "bearer"
        assert data["user"]["email"] == "agriexpert@bhoomimitra.ai"

        # Verify HttpOnly Cookie was set
        cookies = login_resp.cookies
        assert "access_token" in cookies
        assert cookies["access_token"] == "mocked-jwt-token-12345"

        # 3. Get /auth/me
        app.dependency_overrides[get_current_active_user] = lambda: user_obj
        me_resp = client.get("/auth/me")
        assert me_resp.status_code == 200
        assert me_resp.json()["id"] == str(user_id)
        assert me_resp.json()["role"] == "expert"

        # 4. Logout
        logout_resp = client.post("/auth/logout")
        assert logout_resp.status_code == 200
        assert logout_resp.json()["success"] is True

    finally:
        app.dependency_overrides.pop(get_auth_service, None)


# =====================================================================
# 5. Role-Based Access Control (RBAC) & Cross-Shop Isolation Tests
# =====================================================================

def test_shop_owner_cannot_access_other_shop_data():
    """Verify that a shop_owner cannot access or update another shop's dashboard/inventory."""
    from unittest.mock import AsyncMock
    from src.inventory.service import InventoryService
    from src.inventory.dependencies import get_inventory_service
    from src.inventory.schemas import InventoryResponse

    shop_a_id = uuid4()
    shop_b_id = uuid4()

    # Logged-in user is owner of Shop A
    shop_a_owner = UserAccount(
        id=uuid4(),
        email="shopa@bhoomimitra.ai",
        role="shop_owner",
        shop_id=shop_a_id,
        is_active=True,
    )

    mock_inv_service = AsyncMock(spec=InventoryService)
    app.dependency_overrides[get_current_user] = lambda: shop_a_owner
    app.dependency_overrides[get_current_active_user] = lambda: shop_a_owner
    app.dependency_overrides[get_inventory_service] = lambda: mock_inv_service

    try:
        # Accessing own shop dashboard -> OK
        mock_inv_service.get_dashboard_summary.return_value = {
            "shop_id": shop_a_id,
            "total_products": 10,
            "available_products_count": 8,
            "low_stock_count": 2,
            "out_of_stock_count": 0,
            "low_stock_items": [],
            "out_of_stock_items": [],
        }
        res_own = client.get(f"/inventory/dashboard/{shop_a_id}")
        assert res_own.status_code == 200

        # Accessing Shop B dashboard -> 403 Forbidden!
        res_other = client.get(f"/inventory/dashboard/{shop_b_id}")
        assert res_other.status_code == 403
        assert "another shop" in res_other.json()["error"]["message"]

        # Modifying a product belonging to Shop B -> 403 Forbidden!
        product_b_id = uuid4()
        mock_inv_service.get_product_by_id.return_value = InventoryResponse(
            id=product_b_id,
            shop_id=shop_b_id,
            product_name="DAP",
            category="Fertilizers",
            brand="IFFCO",
            unit="Bag",
            price=1350.0,
            quantity_in_stock=20,
            minimum_stock_level=5,
            available=True,
            last_updated="2026-08-23T12:00:00Z",
            created_at="2026-08-23T12:00:00Z",
            updated_at="2026-08-23T12:00:00Z",
        )

        res_update_other = client.put(f"/inventory/{product_b_id}", json={"price": 1400.0})
        assert res_update_other.status_code == 403

    finally:
        app.dependency_overrides.pop(get_inventory_service, None)


def test_admin_can_access_any_shop():
    """Verify that an admin has unrestricted access across any shop."""
    from unittest.mock import AsyncMock
    from src.inventory.service import InventoryService
    from src.inventory.dependencies import get_inventory_service

    shop_b_id = uuid4()
    admin_user = UserAccount(
        id=uuid4(),
        email="superadmin@bhoomimitra.ai",
        role="admin",
        is_active=True,
    )

    mock_inv_service = AsyncMock(spec=InventoryService)
    mock_inv_service.get_dashboard_summary.return_value = {
        "shop_id": shop_b_id,
        "total_products": 5,
        "available_products_count": 5,
        "low_stock_count": 0,
        "out_of_stock_count": 0,
        "low_stock_items": [],
        "out_of_stock_items": [],
    }

    app.dependency_overrides[get_current_user] = lambda: admin_user
    app.dependency_overrides[get_current_active_user] = lambda: admin_user
    app.dependency_overrides[get_inventory_service] = lambda: mock_inv_service

    try:
        res = client.get(f"/inventory/dashboard/{shop_b_id}")
        assert res.status_code == 200
        assert res.json()["shop_id"] == str(shop_b_id)
    finally:
        app.dependency_overrides.pop(get_inventory_service, None)


def test_expert_cannot_modify_shop_data():
    """Verify that an expert role cannot access shop owner endpoints."""
    expert_user = UserAccount(
        id=uuid4(),
        email="expert@bhoomimitra.ai",
        role="expert",
        expert_id=uuid4(),
        is_active=True,
    )

    app.dependency_overrides[get_current_user] = lambda: expert_user
    app.dependency_overrides[get_current_active_user] = lambda: expert_user

    shop_id = uuid4()
    try:
        res = client.get(f"/inventory/dashboard/{shop_id}")
        assert res.status_code == 403
    finally:
        app.dependency_overrides.clear()


# =====================================================================
# 6. Comprehensive Authorization Audit & IDOR Tests
# =====================================================================

def test_non_admin_blocked_from_rag_rebuild():
    """Verify shop_owner or expert cannot trigger RAG index rebuild."""
    shop_user = UserAccount(
        id=uuid4(),
        email="shop@bhoomimitra.ai",
        role="shop_owner",
        shop_id=uuid4(),
        is_active=True,
    )
    app.dependency_overrides[get_current_user] = lambda: shop_user
    app.dependency_overrides[get_current_active_user] = lambda: shop_user

    try:
        res = client.post("/rag/rebuild")
        assert res.status_code == 403
    finally:
        app.dependency_overrides.clear()


def test_non_admin_blocked_from_scheme_creation():
    """Verify non-admin cannot inject government schemes."""
    shop_user = UserAccount(
        id=uuid4(),
        email="shop@bhoomimitra.ai",
        role="shop_owner",
        shop_id=uuid4(),
        is_active=True,
    )
    app.dependency_overrides[get_current_user] = lambda: shop_user
    app.dependency_overrides[get_current_active_user] = lambda: shop_user

    payload = {
        "scheme_name": "Unauthorized Subsidy",
        "description": "Fake scheme",
        "category": "Subsidy",
        "benefits_summary": "100% free",
        "state": "All India",
    }
    try:
        res = client.post("/schemes", json=payload)
        assert res.status_code == 403
    finally:
        app.dependency_overrides.clear()


def test_non_admin_blocked_from_farmer_deletion():
    """Verify non-admin cannot delete farmer accounts."""
    expert_user = UserAccount(
        id=uuid4(),
        email="expert@bhoomimitra.ai",
        role="expert",
        expert_id=uuid4(),
        is_active=True,
    )
    app.dependency_overrides[get_current_user] = lambda: expert_user
    app.dependency_overrides[get_current_active_user] = lambda: expert_user

    try:
        res = client.delete(f"/farmers/{uuid4()}")
        assert res.status_code == 403
    finally:
        app.dependency_overrides.clear()


def test_expert_idor_cross_expert_profile_update_blocked():
    """Verify Expert A cannot modify Expert B's profile (IDOR protection)."""
    expert_a_id = uuid4()
    expert_b_id = uuid4()

    expert_a_user = UserAccount(
        id=uuid4(),
        email="expert.a@bhoomimitra.ai",
        role="expert",
        expert_id=expert_a_id,
        is_active=True,
    )
    app.dependency_overrides[get_current_user] = lambda: expert_a_user
    app.dependency_overrides[get_current_active_user] = lambda: expert_a_user

    payload = {
        "full_name": "Hacked Profile",
        "specialty": "Pest",
        "phone_number": "+919876543210",
        "experience_years": 15,
    }
    try:
        # Expert A tries to update Expert B
        res = client.put(f"/escalation/experts/{expert_b_id}", json=payload)
        assert res.status_code == 403
        data = res.json()
        assert "error" in data or "detail" in data
    finally:
        app.dependency_overrides.clear()


def test_shop_owner_blocked_from_escalation_tickets():
    """Verify Shop Owner cannot access farmer escalation consultation records."""
    shop_user = UserAccount(
        id=uuid4(),
        email="shop@bhoomimitra.ai",
        role="shop_owner",
        shop_id=uuid4(),
        is_active=True,
    )
    app.dependency_overrides[get_current_user] = lambda: shop_user
    app.dependency_overrides[get_current_active_user] = lambda: shop_user

    try:
        res = client.get(f"/escalation/tickets/farmer/{uuid4()}")
        assert res.status_code == 403
    finally:
        app.dependency_overrides.clear()

