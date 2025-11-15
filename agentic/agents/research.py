"""Evidence research agent."""

from __future__ import annotations

from agentic.agents.base import BaseHealthcareAgent
from agentic.rag.pipeline import RetrievedContext


class EvidenceResearchAgent(BaseHealthcareAgent):
    """Maps intake data to structured medical context."""

    def _build_prompt(
        self,
        user_query: str,
        intake_summary: str,
        retrieved_context: RetrievedContext,
        extra_payload: dict[str, str],
    ) -> str:
        context_block = retrieved_context.as_bullet_list() or "- No internal guidance found."
        labs = extra_payload.get("recent_labs", "Not provided")
        return (
            "You review internal clinical guidance for digital triage."
            " Provide a concise research brief that links the intake summary"
            " to the retrieved evidence and to the user's symptoms."
            " Flag at most three differential diagnoses and reference lab data if supplied."
            f"\n\nIntake summary: {intake_summary}\n\nInternal evidence:\n{context_block}\n\n"
            f"Recent labs or vitals: {labs}"
        )
