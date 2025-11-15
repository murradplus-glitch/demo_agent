"""Patient intake agent."""

from __future__ import annotations

from agentic.agents.base import BaseHealthcareAgent
from agentic.rag.pipeline import RetrievedContext


class PatientIntakeAgent(BaseHealthcareAgent):
    """Collects key facts from the free-form user request."""

    def _build_prompt(
        self,
        user_query: str,
        intake_summary: str,
        retrieved_context: RetrievedContext,
        extra_payload: dict[str, str],
    ) -> str:
        del intake_summary, retrieved_context, extra_payload
        return (
            "Extract the patient's demographics, primary concern, chronic conditions, and "
            "any red flags from the following request. Reply with short bullet points."\
            f"\n\nPatient statement: {user_query.strip()}"
        )
