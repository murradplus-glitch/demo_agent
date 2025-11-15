"""Utility for loading mock triage, facility, and eligibility datasets."""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any


class HealthcareDataRepository:
    """Loads CSV artifacts shipped with the repository."""

    def __init__(
        self,
        *,
        triage_csv: str | Path,
        facility_csv: str | Path,
        eligibility_csv: str | Path,
    ) -> None:
        self.triage_cases = self._load_csv(triage_csv)
        self.facility_rows = self._load_csv(facility_csv)
        self.eligibility_rows = self._load_csv(eligibility_csv)

    # ------------------------------------------------------------------
    # Public helpers consumed by the agents
    # ------------------------------------------------------------------
    def match_triage(self, symptoms: str, top_k: int = 3) -> list[dict[str, Any]]:
        """Return the closest matching historical cases."""

        query_tokens = self._tokenize(symptoms)
        scored: list[tuple[float, dict[str, Any]]] = []
        for row in self.triage_cases:
            case_text = f"{row.get('symptoms_en', '')} {row.get('symptoms_ur', '')}"
            tokens = self._tokenize(case_text)
            overlap = len(query_tokens & tokens)
            ratio = self._sequence_ratio(symptoms, row.get("symptoms_en", ""))
            score = overlap + ratio
            scored.append((score, row))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            {
                "symptoms_en": row.get("symptoms_en", ""),
                "symptoms_ur": row.get("symptoms_ur", ""),
                "classification": row.get("classification", "Self-care"),
                "notes": row.get("notes", "Provide reassurance."),
            }
            for _, row in scored[:top_k]
        ]

    def recommend_facilities(
        self,
        *,
        city: str | None,
        area: str | None,
        severity: str | None,
        limit: int = 3,
    ) -> list[dict[str, Any]]:
        """Filter facilities by geography and simple capacity heuristics."""

        normalized_city = (city or "").strip().lower()
        normalized_area = (area or "").strip().lower()
        severity_weight = 2 if (severity or "").lower() == "emergency" else 1

        def score(row: dict[str, Any]) -> float:
            city_match = 1 if normalized_city and normalized_city in row.get("city", "").lower() else 0
            area_match = 1 if normalized_area and normalized_area in row.get("area", "").lower() else 0
            doctors = self._safe_int(row.get("doctors_available"), default=50)
            return city_match * 2 + area_match + (doctors / 100.0) * severity_weight

        ranked = sorted(self.facility_rows, key=score, reverse=True)
        return [
            {
                "name": row.get("name", "Unknown facility"),
                "city": row.get("city", ""),
                "area": row.get("area", ""),
                "doctors_available": self._safe_int(row.get("doctors_available"), default=0),
                "address": row.get("address", "Not listed"),
                "contact": row.get("contact", "Not listed"),
            }
            for row in ranked[:limit]
        ]

    def evaluate_programs(self, profile: dict[str, Any]) -> dict[str, Any]:
        """Apply basic Sehat Card heuristics and share similar cases."""

        income = self._safe_int(profile.get("income_per_month_pkrs"), default=0)
        family_size = self._safe_int(profile.get("family_size"), default=1)
        nser_score = self._safe_int(profile.get("nser_score"), default=0)
        eligible = False
        reason = "Income exceeds program limits."
        if income < 40000:
            eligible = True
            reason = "Income under PKR 40,000."
        elif income <= 80000 and family_size > 5:
            eligible = True
            reason = "Moderate income but large household size."
        if nser_score and nser_score <= 30:
            eligible = True
            reason = "NSER score below 30 triggers automatic eligibility."
        reference_cases = [
            row
            for row in self.eligibility_rows
            if self._safe_int(row.get("income_per_month_pkrs")) <= income + 5000
            and self._safe_int(row.get("income_per_month_pkrs")) >= max(0, income - 5000)
        ][:5]
        return {
            "eligible": "Yes" if eligible else "No",
            "reason": reason,
            "reference_cases": reference_cases,
        }

    def create_follow_up_plan(
        self,
        profile: dict[str, Any],
        triage_metadata: dict[str, Any],
        facility_metadata: dict[str, Any],
    ) -> dict[str, Any]:
        severity = (triage_metadata.get("severity") or "Self-care").lower()
        facility_name = None
        facilities = facility_metadata.get("facility_options") or facility_metadata.get("facilities")
        if facilities:
            facility_name = facilities[0]["name"]
        reminders: list[str] = []
        monitoring: list[str] = []
        if severity == "emergency":
            reminders.append("Confirm arrival at the recommended emergency facility and share vitals within 30 minutes.")
            monitoring.append("Trigger a welfare call if the citizen does not check in within 2 hours.")
        else:
            reminders.append("Log symptoms twice a day in the mobile app and hydrate adequately.")
            monitoring.append("Escalate to telehealth triage if fever exceeds 38.5°C for 48 hours.")
        reminders.append("Bring CNIC and Sehat Card (if applicable) to the appointment.")
        if facility_name:
            reminders.append(f"Carry previous prescriptions to {facility_name}.")
        monitoring.append("Schedule vaccination review in 30 days if the child is under 5 years old.")
        return {"reminders": reminders, "monitoring": monitoring}

    def calculate_health_trends(self, keyword: str | None) -> dict[str, Any]:
        """Return lightweight analytics for dashboards."""

        keyword = (keyword or "fever").lower()
        keyword_counts = 0
        emergency_cases = 0
        for row in self.triage_cases:
            text = f"{row.get('symptoms_en', '')} {row.get('notes', '')}".lower()
            if keyword in text:
                keyword_counts += 1
            if (row.get("classification") or "").lower() == "emergency":
                emergency_cases += 1
        total = max(len(self.triage_cases), 1)
        return {
            "matching_cases": keyword_counts,
            "emergency_rate": f"{(emergency_cases / total) * 100:.1f}%",
            "dataset_size": total,
        }

    def detect_knowledge_alerts(self) -> list[dict[str, Any]]:
        """Raise alerts when symptom keywords spike."""

        alerts: list[dict[str, Any]] = []
        keyword_groups = {
            "dengue": ["dengue", "mosquito", "platelet"],
            "measles": ["measles", "rash", "spots"],
            "cholera": ["cholera", "diarrhea", "vomiting"],
        }
        text_rows = [
            f"{row.get('symptoms_en', '')} {row.get('notes', '')}".lower()
            for row in self.triage_cases
        ]
        for label, keywords in keyword_groups.items():
            hits = sum(1 for text in text_rows if any(word in text for word in keywords))
            if hits >= 5:
                alerts.append(
                    {
                        "label": label.title(),
                        "description": f"Detected {hits} mock triage cases mentioning {label}. Notify district surveillance.",
                    }
                )
        if not alerts:
            alerts.append(
                {
                    "label": "Baseline",
                    "description": "No outbreaks detected. Continue passive surveillance but collect travel history data.",
                }
            )
        return alerts

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _load_csv(self, path: str | Path) -> list[dict[str, Any]]:
        resolved = self._resolve_path(path)
        if not resolved or not resolved.exists():
            return []
        rows: list[dict[str, Any]] = []
        with resolved.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(line for line in handle if line.strip())
            for row in reader:
                cleaned = {key: value.strip() if isinstance(value, str) else value for key, value in row.items() if key}
                if cleaned:
                    rows.append(cleaned)
        return rows

    def _resolve_path(self, path: str | Path) -> Path | None:
        candidate = Path(path)
        if candidate.exists():
            return candidate
        repo_root = Path(__file__).resolve().parents[2]
        candidate = repo_root / path
        if candidate.exists():
            return candidate
        return None

    def _tokenize(self, text: str) -> set[str]:
        return set(re.findall(r"[\w']+", text.lower()))

    def _sequence_ratio(self, left: str, right: str) -> float:
        if not left or not right:
            return 0.0
        matches = sum(1 for token in self._tokenize(left) if token in self._tokenize(right))
        return matches / max(len(self._tokenize(left)), 1)

    def _safe_int(self, value: Any, *, default: int | None = None) -> int:
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return default if default is not None else 0

