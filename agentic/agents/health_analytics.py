"""Health analytics agent."""

from __future__ import annotations

from typing import Any

from agentic.data.repository import HealthcareDataRepository
from agentic.rag.pipeline import RetrievedContext

from .base import BaseHealthcareAgent


class HealthAnalyticsAgent(BaseHealthcareAgent):
    """Synthesizes RAG context with epidemiological signals."""

    def __init__(
        self,
        *,
        data_repo: HealthcareDataRepository,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.data_repo = data_repo

    def generate_trends(self, triage_metadata: dict[str, Any]) -> dict[str, Any]:
        keyword = triage_metadata.get("primary_keyword")
        return self.data_repo.calculate_health_trends(keyword)

    def _build_prompt(
        self,
        user_query: str,
        shared_state: dict[str, Any],
        retrieved_context: RetrievedContext | None,
        extra_payload: dict[str, Any],
    ) -> str:
        triage_metadata = shared_state.get("triage", {})
        trends = extra_payload.get("trends", {})
        rag_context = retrieved_context.as_bullet_list() if retrieved_context else "(none)"
        analytics_notes = "\n".join(
            f"- {metric}: {value}"
            for metric, value in trends.items()
        ) or "- No analytics signals captured."
        return (
            "You are the health analytics agent. Correlate this citizen's case with "
            "population-level signals, cite high-risk cohorts, and surface transparent "
            "rules of thumb.\n"
            f"Citizen triage summary: {triage_metadata}\n"
            "Analytics features:\n"
            f"{analytics_notes}\n"
            "Knowledge base excerpts:\n"
            f"{rag_context}"
        )
