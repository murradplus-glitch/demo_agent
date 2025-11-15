"""Facility finder agent."""

from __future__ import annotations

from typing import Any

from agentic.data.repository import HealthcareDataRepository
from agentic.rag.pipeline import RetrievedContext

from .base import BaseHealthcareAgent


class FacilityFinderAgent(BaseHealthcareAgent):
    """Matches the citizen with open facilities."""

    def __init__(
        self,
        *,
        data_repo: HealthcareDataRepository,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.data_repo = data_repo

    def recommend_facilities(self, profile: dict[str, Any], severity: str | None) -> list[dict[str, Any]]:
        return self.data_repo.recommend_facilities(
            city=profile.get("city"),
            area=profile.get("area"),
            severity=severity,
            limit=3,
        )

    def _build_prompt(
        self,
        user_query: str,
        shared_state: dict[str, Any],
        retrieved_context: RetrievedContext | None,
        extra_payload: dict[str, Any],
    ) -> str:
        profile = shared_state.get("citizen_profile", {})
        triage_metadata = shared_state.get("triage", {})
        facilities = extra_payload.get("facility_options", [])
        formatted_facilities = "\n".join(
            f"- {item['name']} ({item['city']} – {item['area']}), doctors available: {item['doctors_available']}, contact: {item['contact']}"
            for item in facilities
        ) or "- No matching facilities were found near the citizen."
        rag_context = retrieved_context.as_bullet_list() if retrieved_context else "(none)"
        return (
            "You are the facility finder agent. Use the triage severity and the citizen's "
            "location to recommend 1–3 facilities, clarifying why each fits.\n"
            f"Citizen profile: {profile}\n"
            f"Triage summary: {triage_metadata}\n"
            "Candidate facilities:\n"
            f"{formatted_facilities}\n"
            "Knowledge base snippets:\n"
            f"{rag_context}"
        )
