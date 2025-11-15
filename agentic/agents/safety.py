"""Safety review agent."""

from __future__ import annotations

from agentic.agents.base import BaseHealthcareAgent
from agentic.rag.pipeline import RetrievedContext


class SafetyReviewAgent(BaseHealthcareAgent):
    """Double checks that the care plan is clinically safe."""

    def _build_prompt(
        self,
        user_query: str,
        intake_summary: str,
        retrieved_context: RetrievedContext,
        extra_payload: dict[str, str],
    ) -> str:
        care_plan = extra_payload.get("care_plan", "Care plan missing")
        return (
            "You act as a patient safety reviewer. Audit the proposed plan for tone, accuracy," \
            " and regulatory alignment. Highlight missing assessments, medication conflicts," \
            " or red flags that demand escalation. Suggest concrete fixes when needed."
            f"\n\nIntake summary: {intake_summary}\n\nPlan to review:\n{care_plan}\n\n"
            f"Retrieved evidence: {retrieved_context.as_bullet_list()}"
        )
