"""Follow-up agent."""

from __future__ import annotations

from typing import Any

from agentic.data.repository import HealthcareDataRepository
from agentic.rag.pipeline import RetrievedContext

from .base import BaseHealthcareAgent


class FollowUpAgent(BaseHealthcareAgent):
    """Schedules reminders and behavioral nudges."""

    def __init__(
        self,
        *,
        data_repo: HealthcareDataRepository,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.data_repo = data_repo

    def plan_follow_up(
        self,
        profile: dict[str, Any],
        triage_metadata: dict[str, Any],
        facility_metadata: dict[str, Any],
    ) -> dict[str, Any]:
        return self.data_repo.create_follow_up_plan(profile, triage_metadata, facility_metadata)

    def _build_prompt(
        self,
        user_query: str,
        shared_state: dict[str, Any],
        retrieved_context: RetrievedContext | None,
        extra_payload: dict[str, Any],
    ) -> str:
        plan = extra_payload.get("follow_up_plan", {})
        triage_metadata = shared_state.get("triage", {})
        profile = shared_state.get("citizen_profile", {})
        rag_context = retrieved_context.as_bullet_list() if retrieved_context else "(none)"
        reminders = "\n".join(f"- {item}" for item in plan.get("reminders", [])) or "- No reminders generated."
        monitoring = "\n".join(f"- {item}" for item in plan.get("monitoring", [])) or "- No monitoring actions."
        return (
            "You are the follow-up and adherence agent. Draft a friendly reminder plan "
            "covering medication, symptom tracking, vaccination, and re-escalation rules.\n"
            f"Citizen profile: {profile}\n"
            f"Triage metadata: {triage_metadata}\n"
            "Reminders:\n"
            f"{reminders}\n"
            "Monitoring actions:\n"
            f"{monitoring}\n"
            "Knowledge base support:\n"
            f"{rag_context}"
        )
