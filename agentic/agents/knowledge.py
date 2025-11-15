"""Knowledge agent for outbreak monitoring."""

from __future__ import annotations

from typing import Any

from agentic.data.repository import HealthcareDataRepository
from agentic.rag.pipeline import RetrievedContext

from .base import BaseHealthcareAgent


class KnowledgeAgent(BaseHealthcareAgent):
    """Looks across anonymized cases to raise alerts."""

    def __init__(
        self,
        *,
        data_repo: HealthcareDataRepository,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.data_repo = data_repo

    def discover_alerts(self) -> list[dict[str, Any]]:
        return self.data_repo.detect_knowledge_alerts()

    def _build_prompt(
        self,
        user_query: str,
        shared_state: dict[str, Any],
        retrieved_context: RetrievedContext | None,
        extra_payload: dict[str, Any],
    ) -> str:
        alerts = extra_payload.get("alerts", [])
        alert_text = "\n".join(
            f"- {alert['label']}: {alert['description']}"
            for alert in alerts
        ) or "- No unusual patterns were detected this hour."
        rag_context = retrieved_context.as_bullet_list() if retrieved_context else "(none)"
        return (
            "You are the knowledge and outbreak monitoring agent. "
            "Summarize current alerts, data gaps, and recommend what information should be collected next.\n"
            f"Citizen specific prompt: {user_query}\n"
            "Latest alerts:\n"
            f"{alert_text}\n"
            "Knowledge base reinforcement:\n"
            f"{rag_context}"
        )
