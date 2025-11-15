"""Care planning agent."""

from __future__ import annotations

from agentic.agents.base import BaseHealthcareAgent
from agentic.rag.pipeline import RetrievedContext


class CarePlanAgent(BaseHealthcareAgent):
    """Creates patient-facing plan referencing evidence."""

    def _build_prompt(
        self,
        user_query: str,
        intake_summary: str,
        retrieved_context: RetrievedContext,
        extra_payload: dict[str, str],
    ) -> str:
        research_brief = extra_payload.get("research_brief", "No research brief supplied")
        return (
            "You are a virtual care navigator. Use the research brief to craft a compassionate"
            " plan with three sections: (1) assessment summary, (2) immediate self-care guidance,"
            " (3) follow-up triggers that should prompt escalation to emergency care or a clinician."
            " Reference the internal evidence numbers in parentheses when relevant."
            f"\n\nPatient narrative: {user_query}\n\nIntake summary: {intake_summary}\n\n"
            f"Research brief: {research_brief}"
        )
