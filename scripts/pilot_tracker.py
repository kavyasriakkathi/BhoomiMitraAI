from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


def _value(row: dict[str, Any], *names: str) -> str:
    for name in names:
        for key, value in row.items():
            if key.strip().lower() == name.strip().lower():
                return str(value or "").strip()
    return ""


def summarize_tracker(csv_path: str | Path) -> dict[str, Any]:
    """Summarize pilot query tracking CSV data."""

    path = Path(csv_path)

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    total = len(rows)
    voice_queries = 0
    text_queries = 0
    useful_yes = 0
    useful_no = 0

    for row in rows:
        query_type = _value(
            row,
            "Query Type",
            "Input Type",
            "Query Channel",
            "Channel",
            "Type",
            "Voice/Text",
            "Voice / Text",
        ).lower()

        if query_type in {"voice", "audio"}:
            voice_queries += 1
        elif query_type in {"text", "message"}:
            text_queries += 1

        useful = _value(
            row,
            "Useful",
            "Was Useful",
            "Helpful",
            "Useful Response",
            "Answer useful?",
            "Answer useful",
        ).lower()

        if useful in {"yes", "y", "true", "1"}:
            useful_yes += 1
        elif useful in {"no", "n", "false", "0"}:
            useful_no += 1

    evaluated = useful_yes + useful_no
    useful_pct = round((useful_yes / evaluated) * 100, 1) if evaluated else 0.0

    return {
        "total": total,
        "voice_queries": voice_queries,
        "text_queries": text_queries,
        "useful_yes": useful_yes,
        "useful_no": useful_no,
        "useful_pct": useful_pct,
    }
