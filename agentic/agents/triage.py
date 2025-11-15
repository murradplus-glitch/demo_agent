"""Symptom triage agent."""

from __future__ import annotations

from typing import Any

from agentic.data.repository import HealthcareDataRepository
from agentic.rag.pipeline import RetrievedContext

from .base import BaseHealthcareAgent


class TriageAgent(BaseHealthcareAgent):
    """Evaluates symptom severity and routing options."""

    def __init__(
        self,
        *,
        data_repo: HealthcareDataRepository,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.data_repo = data_repo

    def analyze_symptoms(self, symptoms: str) -> dict[str, Any]:
        matches = self.data_repo.match_triage(symptoms, top_k=3)
        severity = matches[0]["classification"] if matches else "Self-care"
        recommended = matches[0]["notes"] if matches else "Provide home-care guidance."
        primary_keyword = None
        if matches:
            primary_keyword = matches[0]["symptoms_en"].split()[0].lower()
        return {
            "severity": severity,
            "recommended_action": recommended,
            "triage_matches": matches,
            "primary_keyword": primary_keyword,
        }

    def _build_prompt(
        self,
        user_query: str,
        shared_state: dict[str, Any],
        retrieved_context: RetrievedContext | None,
        extra_payload: dict[str, Any],
    ) -> str:
        matches = extra_payload.get("triage_matches", [])
        formatted_matches = "\n".join(
            f"- {row['symptoms_en']} ({row['classification']}): {row['notes']}"
            for row in matches
        ) or "- No historical cases matched."
        profile = shared_state.get("citizen_profile", {})
        severity = extra_payload.get("severity", "Unknown")
        recommended_action = extra_payload.get(
            "recommended_action", "Provide self-care until a clinician can evaluate."
        )
        rag_context = retrieved_context.as_bullet_list() if retrieved_context else "(none)"
        return (
            "You are the triage agent inside a Pakistani digital health assistant. "
            "Summarize severity, routing level (self-care, BHU, hospital) and key observations.\n"
            f"Patient question: {user_query}\n"
            f"Citizen profile: {profile}\n"
            f"Symptom severity: {severity}\n"
            f"Recommended action: {recommended_action}\n"
            "Historical triage matches:\n"
            f"{formatted_matches}\n"
            "Context from internal knowledge base:\n"
            f"{rag_context}"
        )
