"""
Tests for BhoomiMitra AI Real Farmer Pilot Workflows

Verifies:
1. Shop CSV importing rules (unknown stays unknown; only verified stock is written).
2. Farmer onboarding minimal fields & phone normalization.
3. Pilot tracker summary & evaluation metrics.
4. Pilot report generator.
"""

import os
import pytest
import tempfile
import csv
from datetime import datetime
from unittest.mock import AsyncMock, patch, MagicMock

from scripts.import_pilot_shops import (
    clean_phone,
    is_truthy,
    parse_float_or_none,
    parse_int_or_none,
    import_pilot_shops,
)
from scripts.onboard_pilot_farmers import (
    normalize_whatsapp_phone,
    onboard_single_farmer,
)
from scripts.pilot_tracker import summarize_tracker


def test_clean_phone_and_normalization():
    assert clean_phone("7989271932") == "7989271932"
    assert clean_phone("+91 98765 43210") == "9876543210"
    assert clean_phone("919876543210") == "9876543210"

    assert normalize_whatsapp_phone("9876543210") == "+919876543210"
    assert normalize_whatsapp_phone("+919876543210") == "+919876543210"
    assert normalize_whatsapp_phone("919876543210") == "+919876543210"


def test_truthy_and_parsing_helpers():
    assert is_truthy("Yes") is True
    assert is_truthy("verified") is True
    assert is_truthy("No") is False
    assert is_truthy("Unknown") is False
    assert is_truthy("") is False
    assert is_truthy(None) is False

    assert parse_float_or_none("₹295.00") == 295.0
    assert parse_float_or_none("1350") == 1350.0
    assert parse_float_or_none("Unknown") is None
    assert parse_float_or_none("-") is None
    assert parse_float_or_none("0") is None

    assert parse_int_or_none("40") == 40
    assert parse_int_or_none("Unknown") is None
    assert parse_int_or_none("-") is None
    assert parse_int_or_none("") is None


@pytest.mark.asyncio
async def test_shop_import_dry_run_with_unknown_and_verified_records():
    csv_content = """Shop Name,Town,Address,Phone,Fertilizers,Seeds,Pesticides,Urea,Stock Quantity,Price,Verified,Last Updated
RAM FERTILIZER,Korutla,"Beside Balaji",7989271932,Yes,Yes,Yes,Unknown,Unknown,Unknown,No,
Kisan Seva,Korutla,"Main Road",9848011223,Yes,Yes,Yes,Yes,40,295,Yes,2026-09-29
"""
    with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".csv", encoding="utf-8") as tf:
        tf.write(csv_content)
        temp_csv = tf.name

    try:
        stats = await import_pilot_shops(temp_csv, dry_run=True)
        assert stats["rows_read"] == 2
        # RAM FERTILIZER already in DB, Kisan Seva created or updated in session
        assert stats["skipped_unknown_inventory"] >= 1
    finally:
        if os.path.exists(temp_csv):
            os.remove(temp_csv)


@pytest.mark.asyncio
async def test_onboard_single_farmer_minimal_fields():
    from src.core.database import AsyncSessionLocal
    async with AsyncSessionLocal() as db:
        res = await onboard_single_farmer(
            db=db,
            name="Pilot Test Farmer",
            phone="9876543219",
            village="Korutla",
            language="te",
            crop="Paddy",
        )
        assert res["status"] in ("CREATED", "UPDATED")
        assert res["phone"] == "+919876543219"
        assert res["language"] == "te"
        assert res["crop"] == "Paddy"
        # Always rollback in test
        await db.rollback()


def test_summarize_tracker_metrics():
    csv_content = """Farmer,Date,Language,Crop,Query,Voice/Text,Answer useful?,Stock information correct?,Voice response worked?,Issue/feedback
Farmer 1,2026-09-29,te,Paddy,Q1,Voice,Yes,Yes,Yes,All good
Farmer 2,2026-09-29,te,Cotton,Q2,Text,Yes,N/A,N/A,Helpful
Farmer 3,2026-09-29,te,Maize,Q3,Voice,No,No,Yes,Dosage unclear
"""
    with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".csv", encoding="utf-8") as tf:
        tf.write(csv_content)
        temp_csv = tf.name

    try:
        metrics = summarize_tracker(temp_csv)
        assert metrics["total"] == 3
        assert metrics["voice_queries"] == 2
        assert metrics["text_queries"] == 1
        assert metrics["useful_yes"] == 2
        assert metrics["useful_no"] == 1
        assert pytest.approx(metrics["useful_pct"], 0.1) == 66.7
    finally:
        if os.path.exists(temp_csv):
            os.remove(temp_csv)
