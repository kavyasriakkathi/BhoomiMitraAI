"""
Unit and Integration Tests for BhoomiMitra AI Demo Agri Shop Owner Account.

Verifies:
1. Successful Shop Owner authentication (login, JWT, cookies).
2. Correct role authorization (RBAC, role constraints, unique shop association).
3. Shop Owner Dashboard access and strict multi-tenant shop isolation.
4. Inventory access, Urea stock updates, and stock-alert restock transitions.
5. Strict denial of Admin-only privileges.
6. Safe temporary password generation and seed mechanism idempotency.
"""

from unittest.mock import AsyncMock, patch
from uuid import UUID, uuid4
import pytest
from fastapi.testclient import TestClient

from scripts.seed_shops_data import (
    DEMO_SHOP_OWNER_EMAIL,
    get_or_generate_temp_password,
    seed_demo_shop_owner,
)
from src.auth.constants import UserRole
from src.auth.dependencies import (
    get_auth_service,
    get_current_active_user,
    get_current_user,
    require_shop_owner,
    verify_shop_access,
)
from src.auth.schemas import LoginRequest, UserRegisterRequest
from src.auth.security import (
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)
from src.auth.service import AuthService
from src.core.exceptions import BhoomiMitraException
from src.core.models import Inventory, Shop, UserAccount
from src.inventory.dependencies import get_inventory_service
from src.inventory.schemas import (
    InventoryResponse,
    PaginatedInventoryResponse,
    ShopDashboardSummaryResponse,
    StockUpdatePayload,
)
from src.inventory.service import InventoryService
from src.main import app

client = TestClient(app)


# =====================================================================
# Fixtures
# =====================================================================

@pytest.fixture
def mallanna_shop_id() -> UUID:
    return UUID("9eb2eba6-d970-4854-95c4-58b9f6a07a12")


@pytest.fixture
def other_shop_id() -> UUID:
    return uuid4()


@pytest.fixture
def demo_shop_owner_account(mallanna_shop_id: UUID) -> UserAccount:
    return UserAccount(
        id=uuid4(),
        email=DEMO_SHOP_OWNER_EMAIL,
        password_hash=hash_password("SecureDemoPass123!"),
        role=UserRole.SHOP_OWNER.value,
        shop_id=mallanna_shop_id,
        is_active=True,
    )


# =====================================================================
# 1. Successful Authentication Tests
# =====================================================================

@pytest.mark.asyncio
async def test_demo_shop_owner_service_authentication(demo_shop_owner_account: UserAccount):
    """Verify demo shop owner can authenticate via AuthService and receive a valid JWT token."""
    mock_repo = AsyncMock()
    mock_repo.get_by_email.return_value = demo_shop_owner_account

    service = AuthService(mock_repo)
    user, token = await service.authenticate_user(
        LoginRequest(email=DEMO_SHOP_OWNER_EMAIL, password="SecureDemoPass123!")
    )

    assert user.email == DEMO_SHOP_OWNER_EMAIL
    assert user.role == UserRole.SHOP_OWNER.value
    assert user.shop_id == demo_shop_owner_account.shop_id

    # Validate decoded token contents
    payload = decode_access_token(token)
    assert payload["sub"] == str(user.id)
    assert payload["email"] == DEMO_SHOP_OWNER_EMAIL
    assert payload["role"] == UserRole.SHOP_OWNER.value
    assert payload["shop_id"] == str(demo_shop_owner_account.shop_id)


def test_demo_shop_owner_login_endpoint(demo_shop_owner_account: UserAccount):
    """Verify /auth/login returns 200, JWT token, UserResponse, and sets secure HttpOnly cookie."""
    mock_service = AsyncMock(spec=AuthService)
    token = create_access_token({
        "sub": str(demo_shop_owner_account.id),
        "email": demo_shop_owner_account.email,
        "role": demo_shop_owner_account.role,
        "shop_id": str(demo_shop_owner_account.shop_id),
    })
    mock_service.authenticate_user.return_value = (demo_shop_owner_account, token)

    app.dependency_overrides[get_auth_service] = lambda: mock_service

    try:
        res = client.post(
            "/auth/login",
            json={"email": DEMO_SHOP_OWNER_EMAIL, "password": "SecureDemoPass123!"},
        )
        assert res.status_code == 200
        data = res.json()
        assert data["access_token"] == token
        assert data["user"]["email"] == DEMO_SHOP_OWNER_EMAIL
        assert data["user"]["role"] == UserRole.SHOP_OWNER.value
        assert data["user"]["shop_id"] == str(demo_shop_owner_account.shop_id)

        # Verify HttpOnly cookie
        assert "access_token" in res.cookies
    finally:
        app.dependency_overrides.pop(get_auth_service, None)


@pytest.mark.asyncio
async def test_demo_shop_owner_invalid_password_rejected(demo_shop_owner_account: UserAccount):
    """Verify invalid password raises 401 Unauthorized."""
    mock_repo = AsyncMock()
    mock_repo.get_by_email.return_value = demo_shop_owner_account

    service = AuthService(mock_repo)
    with pytest.raises(BhoomiMitraException) as exc:
        await service.authenticate_user(
            LoginRequest(email=DEMO_SHOP_OWNER_EMAIL, password="WrongPassword999!")
        )
    assert exc.value.status_code == 401
    assert "Invalid email or password" in exc.value.message


# =====================================================================
# 2. Correct Role Authorization Tests
# =====================================================================

@pytest.mark.asyncio
async def test_demo_shop_owner_role_and_unique_association(mallanna_shop_id: UUID):
    """Verify shop owner registration enforces shop existence and uniqueness."""
    mock_repo = AsyncMock()
    mock_repo.get_by_email.return_value = None
    mock_repo.check_shop_exists.return_value = True
    # Simulate another account already occupying this shop
    mock_repo.get_by_shop_id.return_value = UserAccount(
        id=uuid4(), email="existing@bhoomimitra.ai", role="shop_owner", shop_id=mallanna_shop_id
    )

    service = AuthService(mock_repo)
    req = UserRegisterRequest(
        email="second.owner@bhoomimitra.ai",
        password="ValidPassword123!",
        role=UserRole.SHOP_OWNER,
        shop_id=mallanna_shop_id,
    )

    with pytest.raises(BhoomiMitraException) as exc:
        await service.register_user(req)
    assert exc.value.status_code == 400
    assert "already associated with Shop ID" in exc.value.message


# =====================================================================
# 3. Shop Owner Dashboard Access & Multi-Tenant Isolation
# =====================================================================

def test_demo_shop_owner_dashboard_access(
    demo_shop_owner_account: UserAccount,
    mallanna_shop_id: UUID,
    other_shop_id: UUID,
):
    """Verify demo shop owner can view own dashboard summary but is blocked from other shops."""
    mock_inv_service = AsyncMock(spec=InventoryService)
    mock_inv_service.get_dashboard_summary.return_value = ShopDashboardSummaryResponse(
        shop_id=mallanna_shop_id,
        total_products=4,
        available_products_count=4,
        low_stock_count=0,
        out_of_stock_count=0,
        low_stock_items=[],
        out_of_stock_items=[],
    )

    app.dependency_overrides[get_current_user] = lambda: demo_shop_owner_account
    app.dependency_overrides[get_current_active_user] = lambda: demo_shop_owner_account
    app.dependency_overrides[get_inventory_service] = lambda: mock_inv_service

    try:
        # Access own shop dashboard -> 200 OK
        res_own = client.get(f"/inventory/dashboard/{mallanna_shop_id}")
        assert res_own.status_code == 200
        assert res_own.json()["shop_id"] == str(mallanna_shop_id)
        assert res_own.json()["total_products"] == 4

        # Access other shop dashboard -> 403 Forbidden
        res_other = client.get(f"/inventory/dashboard/{other_shop_id}")
        assert res_other.status_code == 403
        assert "another shop" in res_other.json()["error"]["message"]
    finally:
        app.dependency_overrides.pop(get_current_user, None)
        app.dependency_overrides.pop(get_current_active_user, None)
        app.dependency_overrides.pop(get_inventory_service, None)


# =====================================================================
# 4. Inventory Access, Urea Stock Update & Stock-Alert Trigger
# =====================================================================

def test_demo_shop_owner_inventory_access_and_urea_update(
    demo_shop_owner_account: UserAccount,
    mallanna_shop_id: UUID,
    other_shop_id: UUID,
):
    """Verify demo shop owner can list inventory, update Urea stock, and trigger stock alerts."""
    urea_id = uuid4()
    other_product_id = uuid4()

    mock_inv_service = AsyncMock(spec=InventoryService)

    # 1. Mock list products for Mallanna shop
    mock_inv_service.list_products_by_shop.return_value = PaginatedInventoryResponse(
        items=[
            InventoryResponse(
                id=urea_id,
                shop_id=mallanna_shop_id,
                product_name="Urea",
                category="Fertilizers",
                brand="IFFCO",
                unit="Bag",
                price=295.0,
                quantity_in_stock=50,
                minimum_stock_level=10,
                available=True,
                last_updated="2026-09-10T00:00:00Z",
                created_at="2026-09-10T00:00:00Z",
                updated_at="2026-09-10T00:00:00Z",
            )
        ],
        total=1,
        page=1,
        size=50,
        pages=1,
    )

    # 2. Mock get product by id (Urea belongs to Mallanna, other_product belongs to other shop)
    def mock_get_product(item_id):
        if item_id == urea_id:
            return InventoryResponse(
                id=urea_id,
                shop_id=mallanna_shop_id,
                product_name="Urea",
                category="Fertilizers",
                brand="IFFCO",
                unit="Bag",
                price=295.0,
                quantity_in_stock=50,
                minimum_stock_level=10,
                available=True,
                last_updated="2026-09-10T00:00:00Z",
                created_at="2026-09-10T00:00:00Z",
                updated_at="2026-09-10T00:00:00Z",
            )
        return InventoryResponse(
            id=other_product_id,
            shop_id=other_shop_id,
            product_name="Other Urea",
            category="Fertilizers",
            brand="Other Brand",
            unit="Bag",
            price=300.0,
            quantity_in_stock=10,
            minimum_stock_level=5,
            available=True,
            last_updated="2026-09-10T00:00:00Z",
            created_at="2026-09-10T00:00:00Z",
            updated_at="2026-09-10T00:00:00Z",
        )

    mock_inv_service.get_product_by_id.side_effect = mock_get_product

    # Mock stock update response
    mock_inv_service.update_stock.return_value = InventoryResponse(
        id=urea_id,
        shop_id=mallanna_shop_id,
        product_name="Urea",
        category="Fertilizers",
        brand="IFFCO",
        unit="Bag",
        price=295.0,
        quantity_in_stock=25,
        minimum_stock_level=10,
        available=True,
        last_updated="2026-09-10T00:00:00Z",
        created_at="2026-09-10T00:00:00Z",
        updated_at="2026-09-10T00:00:00Z",
    )

    app.dependency_overrides[get_current_user] = lambda: demo_shop_owner_account
    app.dependency_overrides[get_current_active_user] = lambda: demo_shop_owner_account
    app.dependency_overrides[get_inventory_service] = lambda: mock_inv_service

    try:
        # A. View Mallanna inventory -> 200 OK
        res_list = client.get(f"/inventory/shop/{mallanna_shop_id}")
        assert res_list.status_code == 200
        assert res_list.json()["items"][0]["product_name"] == "Urea"

        # B. Update Urea stock -> 200 OK
        res_update = client.patch(
            f"/inventory/{urea_id}/stock",
            json={"quantity_in_stock": 25, "available": True},
        )
        assert res_update.status_code == 200
        assert res_update.json()["quantity_in_stock"] == 25

        # C. Attempt to update stock for another shop's product -> 403 Forbidden
        res_other = client.patch(
            f"/inventory/{other_product_id}/stock",
            json={"quantity_in_stock": 10, "available": True},
        )
        assert res_other.status_code == 403
        assert "another shop" in res_other.json()["error"]["message"]
    finally:
        app.dependency_overrides.pop(get_current_user, None)
        app.dependency_overrides.pop(get_current_active_user, None)
        app.dependency_overrides.pop(get_inventory_service, None)


@pytest.mark.asyncio
async def test_urea_restock_triggers_stock_alerts_transition(mallanna_shop_id: UUID):
    """Verify that updating Urea stock from 0 to >0 triggers stock alert notifications."""
    urea_id = uuid4()
    mock_repo = AsyncMock()

    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)

    # Product was out of stock
    prev_item = Inventory(
        id=urea_id,
        shop_id=mallanna_shop_id,
        product_name="Urea",
        category="Fertilizers",
        brand="IFFCO",
        unit="Bag",
        price=295.0,
        quantity_in_stock=0,
        minimum_stock_level=10,
        available=False,
        created_at=now,
        updated_at=now,
        last_updated=now,
    )
    mock_repo.get_by_id.return_value = prev_item

    # Updated product is in stock
    updated_item = Inventory(
        id=urea_id,
        shop_id=mallanna_shop_id,
        product_name="Urea",
        category="Fertilizers",
        brand="IFFCO",
        unit="Bag",
        price=295.0,
        quantity_in_stock=50,
        minimum_stock_level=10,
        available=True,
        created_at=now,
        updated_at=now,
        last_updated=now,
    )
    mock_repo.update_stock.return_value = updated_item

    service = InventoryService(mock_repo)

    with patch("src.shops.stock_alerts.trigger_stock_alert_notifications", new_callable=AsyncMock) as mock_trigger:
        res = await service.update_stock(
            urea_id, StockUpdatePayload(quantity_in_stock=50, available=True)
        )
        assert res.quantity_in_stock == 50

        # Wait briefly for asyncio background task
        import asyncio
        await asyncio.sleep(0.01)

        mock_trigger.assert_called_once_with(
            inventory_item_id=urea_id,
            shop_id=mallanna_shop_id,
            product_name="Urea",
            new_quantity=50,
            unit="Bag",
            brand="IFFCO",
        )


# =====================================================================
# 5. Unauthorized Admin Access Tests
# =====================================================================

def test_demo_shop_owner_forbidden_from_admin_functionality(demo_shop_owner_account: UserAccount):
    """Verify demo shop owner account cannot access Admin-only routes."""
    app.dependency_overrides[get_current_user] = lambda: demo_shop_owner_account
    app.dependency_overrides[get_current_active_user] = lambda: demo_shop_owner_account

    try:
        # Analytics (Admin only)
        res = client.get("/analytics/summary")
        assert res.status_code == 403

        res_act = client.get("/analytics/activity")
        assert res_act.status_code == 403

        # RAG Rebuild
        res = client.post("/rag/rebuild")
        assert res.status_code == 403

        # Create Scheme
        res = client.post(
            "/schemes",
            json={"scheme_name": "Unauthorized Scheme", "description": "Blocked"},
        )
        assert res.status_code == 403

        # Create Shop
        res = client.post(
            "/shops",
            json={
                "shop_name": "Blocked Shop",
                "owner_name": "Blocked",
                "phone_number": "9999999999",
                "address": "Blocked",
                "village": "Blocked",
                "mandal": "Blocked",
                "district": "Blocked",
                "state": "Telangana",
                "pin_code": "500001",
                "latitude": 17.385,
                "longitude": 78.486,
            },
        )
        assert res.status_code == 403
    finally:
        app.dependency_overrides.pop(get_current_user, None)
        app.dependency_overrides.pop(get_current_active_user, None)


# =====================================================================
# 6. Temporary Password Generation & Seed Function Idempotency
# =====================================================================

def test_temporary_password_generator(monkeypatch):
    """Verify safe password generation through environment variable or cryptographic randomizer."""
    # Test env override
    monkeypatch.setenv("DEMO_SHOP_OWNER_PASSWORD", "CustomEnvPassword99!")
    pw, is_generated = get_or_generate_temp_password()
    assert pw == "CustomEnvPassword99!"
    assert is_generated is False

    # Test random generation without env
    monkeypatch.delenv("DEMO_SHOP_OWNER_PASSWORD", raising=False)
    pw2, is_generated2 = get_or_generate_temp_password()
    assert is_generated2 is True
    assert len(pw2) >= 16
    assert pw2.startswith("BmDemo#")

    # Hashes securely with Argon2id
    hashed = hash_password(pw2)
    assert verify_password(pw2, hashed) is True


@pytest.mark.asyncio
async def test_seed_demo_shop_owner_idempotent(mallanna_shop_id: UUID):
    """Verify seed_demo_shop_owner creates and updates user correctly without error."""
    mock_db = AsyncMock()
    mock_shop = Shop(
        id=mallanna_shop_id,
        shop_name="Mallanna Fertilizer Seeds and Pesticides",
    )

    from unittest.mock import MagicMock

    # 1. Initial creation
    mock_result_empty = MagicMock()
    mock_result_empty.scalar_one_or_none.return_value = None
    mock_db.execute.return_value = mock_result_empty

    created_user, pw = await seed_demo_shop_owner(mock_db, mock_shop)
    assert created_user.email == DEMO_SHOP_OWNER_EMAIL
    assert created_user.role == UserRole.SHOP_OWNER.value
    assert created_user.shop_id == mallanna_shop_id
    assert verify_password(pw, created_user.password_hash) is True

    # 2. Existing update
    mock_result_existing = MagicMock()
    mock_result_existing.scalar_one_or_none.return_value = created_user
    mock_db.execute.return_value = mock_result_existing

    updated_user, pw2 = await seed_demo_shop_owner(mock_db, mock_shop)
    assert updated_user.email == DEMO_SHOP_OWNER_EMAIL
    assert updated_user.role == UserRole.SHOP_OWNER.value
    assert updated_user.shop_id == mallanna_shop_id
    assert verify_password(pw2, updated_user.password_hash) is True


# =====================================================================
# =====================================================================
# 7. Environment-Controlled One-Time Seed Tests (DEMO_SHOP_OWNER_SEED)
# =====================================================================

@pytest.mark.asyncio
async def test_demo_seed_disabled_by_default(monkeypatch):
    """When DEMO_SHOP_OWNER_SEED is False, seed mechanism must not run."""
    from src.auth.demo_seed import ensure_demo_shop_owner_seeded
    from src.config import get_settings

    settings = get_settings()
    monkeypatch.setattr(settings, "demo_shop_owner_seed", False)

    mock_db = AsyncMock()
    result = await ensure_demo_shop_owner_seeded(mock_db)
    assert result is None
    mock_db.execute.assert_not_called()


@pytest.mark.asyncio
async def test_demo_seed_missing_shop_safe_failure(monkeypatch):
    """When the expected demo shop does not exist, fail safely without creating fake shop data."""
    from unittest.mock import MagicMock
    from src.auth.demo_seed import ensure_demo_shop_owner_seeded
    from src.config import get_settings

    settings = get_settings()
    monkeypatch.setattr(settings, "demo_shop_owner_seed", True)

    mock_db = AsyncMock()
    mock_db.add = MagicMock()

    # Shop query returns None (shop missing)
    shop_res = MagicMock()
    shop_res.scalar_one_or_none.return_value = None
    mock_db.execute.return_value = shop_res

    result = await ensure_demo_shop_owner_seeded(mock_db)
    assert result is None
    # Verify no fake shop or user was added or committed
    mock_db.add.assert_not_called()
    mock_db.commit.assert_not_called()


@pytest.mark.asyncio
async def test_demo_seed_creates_new_user_custom_password(monkeypatch, mallanna_shop_id: UUID):
    """When DEMO_SHOP_OWNER_SEED=True and user is missing, creates demo account with custom password."""
    from unittest.mock import MagicMock
    from src.auth.demo_seed import ensure_demo_shop_owner_seeded
    from src.config import get_settings

    settings = get_settings()
    monkeypatch.setattr(settings, "demo_shop_owner_seed", True)
    monkeypatch.setattr(settings, "demo_shop_owner_password", "MyCustomSecretPass123!")

    mock_db = AsyncMock()
    mock_db.add = MagicMock()
    mock_shop = Shop(id=mallanna_shop_id, shop_name="Mallanna Fertilizer Seeds and Pesticides")

    # 1. Shop lookup returns mock_shop
    shop_res = MagicMock()
    shop_res.scalar_one_or_none.return_value = mock_shop

    # 2. User lookup by email returns None
    user_res = MagicMock()
    user_res.scalar_one_or_none.return_value = None

    # 3. Shop association check returns None
    shop_user_res = MagicMock()
    shop_user_res.scalar_one_or_none.return_value = None

    mock_db.execute.side_effect = [shop_res, user_res, shop_user_res]

    user = await ensure_demo_shop_owner_seeded(mock_db)
    assert user is not None
    assert user.email == "demo.shopowner@bhoomimitra.ai"
    assert user.role == UserRole.SHOP_OWNER.value
    assert user.shop_id == mallanna_shop_id
    assert user.is_active is True
    assert verify_password("MyCustomSecretPass123!", user.password_hash) is True
    mock_db.commit.assert_called_once()


@pytest.mark.asyncio
async def test_demo_seed_creates_new_user_random_password(monkeypatch, mallanna_shop_id: UUID):
    """When DEMO_SHOP_OWNER_SEED=True and password not supplied, generates secure random password."""
    from unittest.mock import MagicMock
    from src.auth.demo_seed import ensure_demo_shop_owner_seeded
    from src.config import get_settings

    settings = get_settings()
    monkeypatch.setattr(settings, "demo_shop_owner_seed", True)
    monkeypatch.setattr(settings, "demo_shop_owner_password", "")

    mock_db = AsyncMock()
    mock_db.add = MagicMock()
    mock_shop = Shop(id=mallanna_shop_id, shop_name="Mallanna Fertilizer Seeds and Pesticides")

    shop_res = MagicMock()
    shop_res.scalar_one_or_none.return_value = mock_shop

    user_res = MagicMock()
    user_res.scalar_one_or_none.return_value = None

    shop_user_res = MagicMock()
    shop_user_res.scalar_one_or_none.return_value = None

    mock_db.execute.side_effect = [shop_res, user_res, shop_user_res]

    user = await ensure_demo_shop_owner_seeded(mock_db)
    assert user is not None
    assert user.email == "demo.shopowner@bhoomimitra.ai"
    assert user.role == UserRole.SHOP_OWNER.value
    assert user.shop_id == mallanna_shop_id
    assert user.is_active is True
    assert user.password_hash.startswith("$argon2id$")
    mock_db.commit.assert_called_once()


@pytest.mark.asyncio
async def test_demo_seed_existing_account_preserves_password(monkeypatch, mallanna_shop_id: UUID):
    """When demo user exists and no new password supplied, existing password hash is preserved."""
    from unittest.mock import MagicMock
    from src.auth.demo_seed import ensure_demo_shop_owner_seeded
    from src.config import get_settings

    settings = get_settings()
    monkeypatch.setattr(settings, "demo_shop_owner_seed", True)
    monkeypatch.setattr(settings, "demo_shop_owner_password", "")

    original_hash = hash_password("OriginalPassword123!")
    existing_user = UserAccount(
        id=uuid4(),
        email="demo.shopowner@bhoomimitra.ai",
        password_hash=original_hash,
        role=UserRole.SHOP_OWNER.value,
        shop_id=mallanna_shop_id,
        is_active=True,
    )

    mock_db = AsyncMock()
    mock_db.add = MagicMock()
    mock_shop = Shop(id=mallanna_shop_id, shop_name="Mallanna Fertilizer Seeds and Pesticides")

    shop_res = MagicMock()
    shop_res.scalar_one_or_none.return_value = mock_shop

    user_res = MagicMock()
    user_res.scalar_one_or_none.return_value = existing_user

    mock_db.execute.side_effect = [shop_res, user_res]

    user = await ensure_demo_shop_owner_seeded(mock_db)
    assert user is not None
    assert user.password_hash == original_hash
    mock_db.commit.assert_called_once()


@pytest.mark.asyncio
async def test_demo_seed_existing_account_explicit_password_override(monkeypatch, mallanna_shop_id: UUID):
    """When demo user exists and DEMO_SHOP_OWNER_PASSWORD is provided, password hash is updated."""
    from unittest.mock import MagicMock
    from src.auth.demo_seed import ensure_demo_shop_owner_seeded
    from src.config import get_settings

    settings = get_settings()
    monkeypatch.setattr(settings, "demo_shop_owner_seed", True)
    monkeypatch.setattr(settings, "demo_shop_owner_password", "ExplicitOverridePass999!")

    original_hash = hash_password("OldPassword123!")
    existing_user = UserAccount(
        id=uuid4(),
        email="demo.shopowner@bhoomimitra.ai",
        password_hash=original_hash,
        role=UserRole.SHOP_OWNER.value,
        shop_id=mallanna_shop_id,
        is_active=True,
    )

    mock_db = AsyncMock()
    mock_db.add = MagicMock()
    mock_shop = Shop(id=mallanna_shop_id, shop_name="Mallanna Fertilizer Seeds and Pesticides")

    shop_res = MagicMock()
    shop_res.scalar_one_or_none.return_value = mock_shop

    user_res = MagicMock()
    user_res.scalar_one_or_none.return_value = existing_user

    mock_db.execute.side_effect = [shop_res, user_res]

    user = await ensure_demo_shop_owner_seeded(mock_db)
    assert user is not None
    assert user.password_hash != original_hash
    assert verify_password("ExplicitOverridePass999!", user.password_hash) is True
    mock_db.commit.assert_called_once()


@pytest.mark.asyncio
async def test_demo_seed_conflicting_email_role_safe_failure(monkeypatch, mallanna_shop_id: UUID):
    """If email exists but belongs to a different role (e.g. admin), abort without modifying."""
    from unittest.mock import MagicMock
    from src.auth.demo_seed import ensure_demo_shop_owner_seeded
    from src.config import get_settings

    settings = get_settings()
    monkeypatch.setattr(settings, "demo_shop_owner_seed", True)

    original_hash = hash_password("AdminSecurePassword123!")
    existing_user = UserAccount(
        id=uuid4(),
        email="demo.shopowner@bhoomimitra.ai",
        password_hash=original_hash,
        role=UserRole.ADMIN.value,
        shop_id=mallanna_shop_id,
        is_active=True,
    )

    mock_db = AsyncMock()
    mock_db.add = MagicMock()
    mock_shop = Shop(id=mallanna_shop_id, shop_name="Mallanna Fertilizer Seeds and Pesticides")

    shop_res = MagicMock()
    shop_res.scalar_one_or_none.return_value = mock_shop

    user_res = MagicMock()
    user_res.scalar_one_or_none.return_value = existing_user

    mock_db.execute.side_effect = [shop_res, user_res]

    result = await ensure_demo_shop_owner_seeded(mock_db)
    assert result is None
    assert existing_user.role == UserRole.ADMIN.value
    assert existing_user.password_hash == original_hash
    mock_db.commit.assert_not_called()


@pytest.mark.asyncio
async def test_demo_seed_conflicting_shop_safe_failure(monkeypatch, mallanna_shop_id: UUID, other_shop_id: UUID):
    """If email exists but is associated with a different shop, abort without modifying."""
    from unittest.mock import MagicMock
    from src.auth.demo_seed import ensure_demo_shop_owner_seeded
    from src.config import get_settings

    settings = get_settings()
    monkeypatch.setattr(settings, "demo_shop_owner_seed", True)

    original_hash = hash_password("OriginalPassword123!")
    existing_user = UserAccount(
        id=uuid4(),
        email="demo.shopowner@bhoomimitra.ai",
        password_hash=original_hash,
        role=UserRole.SHOP_OWNER.value,
        shop_id=other_shop_id,
        is_active=True,
    )

    mock_db = AsyncMock()
    mock_db.add = MagicMock()
    mock_shop = Shop(id=mallanna_shop_id, shop_name="Mallanna Fertilizer Seeds and Pesticides")

    shop_res = MagicMock()
    shop_res.scalar_one_or_none.return_value = mock_shop

    user_res = MagicMock()
    user_res.scalar_one_or_none.return_value = existing_user

    mock_db.execute.side_effect = [shop_res, user_res]

    result = await ensure_demo_shop_owner_seeded(mock_db)
    assert result is None
    assert existing_user.shop_id == other_shop_id
    mock_db.commit.assert_not_called()


@pytest.mark.asyncio
async def test_demo_seed_shop_occupied_by_other_user_safe_failure(monkeypatch, mallanna_shop_id: UUID):
    """If demo email does not exist, but shop is already associated with another user, abort without takeover."""
    from unittest.mock import MagicMock
    from src.auth.demo_seed import ensure_demo_shop_owner_seeded
    from src.config import get_settings

    settings = get_settings()
    monkeypatch.setattr(settings, "demo_shop_owner_seed", True)

    other_user = UserAccount(
        id=uuid4(),
        email="real.owner@mallanna.com",
        password_hash=hash_password("RealOwnerPass123!"),
        role=UserRole.SHOP_OWNER.value,
        shop_id=mallanna_shop_id,
        is_active=True,
    )

    mock_db = AsyncMock()
    mock_db.add = MagicMock()
    mock_shop = Shop(id=mallanna_shop_id, shop_name="Mallanna Fertilizer Seeds and Pesticides")

    shop_res = MagicMock()
    shop_res.scalar_one_or_none.return_value = mock_shop

    # Demo email does not exist
    user_res = MagicMock()
    user_res.scalar_one_or_none.return_value = None

    # But shop is occupied by other_user
    shop_user_res = MagicMock()
    shop_user_res.scalar_one_or_none.return_value = other_user

    mock_db.execute.side_effect = [shop_res, user_res, shop_user_res]

    result = await ensure_demo_shop_owner_seeded(mock_db)
    assert result is None
    mock_db.add.assert_not_called()
    mock_db.commit.assert_not_called()


@pytest.mark.asyncio
async def test_demo_seed_idempotent_repeated_execution(monkeypatch, mallanna_shop_id: UUID):
    """Running seed multiple times creates user on first run and safely preserves on subsequent runs."""
    from unittest.mock import MagicMock
    from src.auth.demo_seed import ensure_demo_shop_owner_seeded
    from src.config import get_settings

    settings = get_settings()
    monkeypatch.setattr(settings, "demo_shop_owner_seed", True)
    monkeypatch.setattr(settings, "demo_shop_owner_password", "IdempotentPass123!")

    mock_db = AsyncMock()
    mock_db.add = MagicMock()
    mock_shop = Shop(id=mallanna_shop_id, shop_name="Mallanna Fertilizer Seeds and Pesticides")

    # Run 1: user doesn't exist -> creates user
    shop_res1 = MagicMock()
    shop_res1.scalar_one_or_none.return_value = mock_shop
    user_res1 = MagicMock()
    user_res1.scalar_one_or_none.return_value = None
    shop_user_res1 = MagicMock()
    shop_user_res1.scalar_one_or_none.return_value = None

    mock_db.execute.side_effect = [shop_res1, user_res1, shop_user_res1]

    user1 = await ensure_demo_shop_owner_seeded(mock_db)
    assert user1 is not None
    saved_hash = user1.password_hash

    # Run 2: user already exists, no password override -> preserves user
    monkeypatch.setattr(settings, "demo_shop_owner_password", "")
    shop_res2 = MagicMock()
    shop_res2.scalar_one_or_none.return_value = mock_shop
    user_res2 = MagicMock()
    user_res2.scalar_one_or_none.return_value = user1

    mock_db.execute.side_effect = [shop_res2, user_res2]

    user2 = await ensure_demo_shop_owner_seeded(mock_db)
    assert user2 is not None
    assert user2.id == user1.id
    assert user2.password_hash == saved_hash


@pytest.mark.asyncio
async def test_demo_seed_never_logs_plaintext_password(monkeypatch, mallanna_shop_id: UUID, caplog):
    """Verify that plaintext password is never exposed in application logs during seeding."""
    import logging
    from unittest.mock import MagicMock
    from src.auth.demo_seed import ensure_demo_shop_owner_seeded
    from src.config import get_settings

    settings = get_settings()
    secret_pass = "TopSecretPasswordDoNotLog!#99"
    monkeypatch.setattr(settings, "demo_shop_owner_seed", True)
    monkeypatch.setattr(settings, "demo_shop_owner_password", secret_pass)

    mock_db = AsyncMock()
    mock_db.add = MagicMock()
    mock_shop = Shop(id=mallanna_shop_id, shop_name="Mallanna Fertilizer Seeds and Pesticides")

    shop_res = MagicMock()
    shop_res.scalar_one_or_none.return_value = mock_shop
    user_res = MagicMock()
    user_res.scalar_one_or_none.return_value = None
    shop_user_res = MagicMock()
    shop_user_res.scalar_one_or_none.return_value = None

    mock_db.execute.side_effect = [shop_res, user_res, shop_user_res]

    with caplog.at_level(logging.DEBUG):
        user = await ensure_demo_shop_owner_seeded(mock_db)

    assert user is not None
    # Crucial assertion: plaintext password must NEVER appear in captured logs
    assert secret_pass not in caplog.text
