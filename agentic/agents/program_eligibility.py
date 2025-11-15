"""Program eligibility agent."""

from __future__ import annotations

from typing import Any

from agentic.data.repository import HealthcareDataRepository
from agentic.rag.pipeline import RetrievedContext

from .base import BaseHealthcareAgent


class ProgramEligibilityAgent(BaseHealthcareAgent):
    """Determines whether the citizen qualifies for public health programs."""

    def __init__(
        self,
        *,
        data_repo: HealthcareDataRepository,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.data_repo = data_repo

    def evaluate_profile(self, profile: dict[str, Any]) -> dict[str, Any]:
        return self.data_repo.evaluate_programs(profile)

    def _build_prompt(
        self,
        user_query: str,
        shared_state: dict[str, Any],
        retrieved_context: RetrievedContext | None,
        extra_payload: dict[str, Any],
    ) -> str:
        profile = shared_state.get("citizen_profile", {})
        evaluation = extra_payload.get("eligibility", {})
        reference_cases = "\n".join(
            f"- Income PKR {case['income_per_month_pkrs']} | Family size {case['family_size']} | Status {case['eligible_for_sehat_sahulat']}"
            for case in evaluation.get("reference_cases", [])
        ) or "- No similar applicants were found in the mock dataset."
        rag_context = retrieved_context.as_bullet_list() if retrieved_context else "(none)"
        return (
            "You are the Sehat Card and preventive program eligibility agent. "
            "Report whether the citizen qualifies, what evidence supports it, "
            "and what documents or next steps are required.\n"
            f"Citizen profile: {profile}\n"
            f"Eligibility decision: {evaluation.get('eligible', 'Unknown')}\n"
            f"Reason: {evaluation.get('reason', 'Not provided')}\n"
            "Reference applicants:\n"
            f"{reference_cases}\n"
            "Knowledge base context:\n"
            f"{rag_context}"
        )
