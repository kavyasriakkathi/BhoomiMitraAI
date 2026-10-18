from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any


_TRUE_VALUES = {"1", "true", "yes", "y", "verified"}
_UNKNOWN_VALUES = {"", "-", "unknown", "n/a", "na", "none", "null"}


def clean_phone(value: Any) -> str:
    """Return an Indian 10-digit phone number without country code."""
    if value is None:
        return ""

    digits = re.sub(r"\D", "", str(value))

    if digits.startswith("91") and len(digits) == 12:
        digits = digits[2:]

    return digits


def is_truthy(value: Any) -> bool:
    if value is None:
        return False

    return str(value).strip().lower() in _TRUE_VALUES


def parse_float_or_none(value: Any) -> float | None:
    if value is None:
        return None

    text = str(value).strip().lower()

    if text in _UNKNOWN_VALUES:
        return None

    text = text.replace(",", "")
    text = re.sub(r"[^\d.\-]", "", text)

    if not text:
        return None

    try:
        number = float(text)
    except ValueError:
        return None

    if number <= 0:
        return None

    return number


def parse_int_or_none(value: Any) -> int | None:
    if value is None:
        return None

    text = str(value).strip().lower()

    if text in _UNKNOWN_VALUES:
        return None

    text = text.replace(",", "")
    text = re.sub(r"[^\d\-]", "", text)

    if not text:
        return None

    try:
        number = int(text)
    except ValueError:
        return None

    if number <= 0:
        return None

    return number


async def import_pilot_shops(
    csv_path: str | Path,
    *,
    dry_run: bool = False,
) -> dict[str, int]:
    """
    Read pilot shop CSV data.

    Dry-run mode only validates/counts rows and performs no DB writes.
    """
    path = Path(csv_path)

    rows_read = 0
    skipped_unknown_inventory = 0

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)

        for row in reader:
            rows_read += 1

            urea = parse_float_or_none(row.get("Urea"))
            stock = parse_int_or_none(row.get("Stock Quantity"))
            price = parse_float_or_none(row.get("Price"))

            verified = is_truthy(row.get("Verified"))

            # Never import an inventory record when its core
            # quantity/price information is unknown or unverified.
            if (
                urea is None
                or stock is None
                or price is None
                or not verified
            ):
                skipped_unknown_inventory += 1

    # The current pilot test exercises dry-run validation.
    # Keep DB writes out of this path until the importer is wired
    # to the project's existing Shop/Inventory services.
    return {
        "rows_read": rows_read,
        "skipped_unknown_inventory": skipped_unknown_inventory,
    }
