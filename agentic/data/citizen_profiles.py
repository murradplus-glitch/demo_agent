"""Helpers for working with demo CNIC profiles used in the UI."""

from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path
from typing import Any

_DATASET = Path(__file__).with_name("demo_citizen_profiles.csv")


@lru_cache(maxsize=1)
def load_demo_profiles() -> dict[str, dict[str, Any]]:
    """Return all demo CNIC rows indexed by the CNIC string."""

    if not _DATASET.exists():
        return {}
    profiles: dict[str, dict[str, Any]] = {}
    with _DATASET.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            cnic = (row.get("cnic") or "").strip()
            if not cnic:
                continue
            cleaned = {key: (value.strip() if isinstance(value, str) else value) for key, value in row.items()}
            cleaned["cnic"] = cnic
            cleaned["family_size"] = _safe_int(cleaned.get("family_size"))
            cleaned["income_per_month_pkrs"] = _safe_int(cleaned.get("income_per_month_pkrs"))
            cleaned["nser_score"] = _safe_int(cleaned.get("nser_score"))
            cleaned["triage_case_id"] = _safe_int(cleaned.get("triage_case_id"))
            cleaned["sehat_card_eligible"] = (cleaned.get("sehat_card_eligible") or "No").strip().lower() in {
                "yes",
                "true",
                "1",
                "eligible",
            }
            profiles[cnic] = cleaned
    return profiles


def get_profile_by_cnic(cnic: str) -> dict[str, Any] | None:
    """Lookup helper used by the Streamlit front-end."""

    if not cnic:
        return None
    normalized = cnic.strip()
    profiles = load_demo_profiles()
    return profiles.get(normalized)


def list_demo_cnic_examples() -> list[dict[str, Any]]:
    """Return CNICs with short blurbs to render on the login screen."""

    examples: list[dict[str, Any]] = []
    for record in load_demo_profiles().values():
        examples.append(
            {
                "cnic": record["cnic"],
                "name": record.get("name", "Demo Citizen"),
                "sehat_card_eligible": record.get("sehat_card_eligible", False),
                "city": record.get("city", ""),
            }
        )
    return sorted(examples, key=lambda item: item["cnic"])


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
