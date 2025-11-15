"""Agent abstractions used by the orchestrator."""

from __future__ import annotations

import textwrap
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from agentic.gemini import GeminiClient, GeminiResponse
from agentic.mcp_bridge import MCPToolBridge
from agentic.rag.pipeline import RetrievedContext


@dataclass(slots=True)
class AgentOutput:
    role: str
    summary: str
    evidence: str
    raw_model_output: str
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseHealthcareAgent(ABC):
    """Shared logic between the specialized agents."""

    def __init__(
        self,
        name: str,
        system_instruction: str,
        gemini: GeminiClient,
        mcp_bridge: MCPToolBridge,
    ) -> None:
        self.name = name
        self.system_instruction = system_instruction
        self.gemini = gemini
        self.mcp_bridge = mcp_bridge

    def run(
        self,
        user_query: str,
        shared_state: dict[str, Any] | None = None,
        retrieved_context: RetrievedContext | None = None,
        extra_payload: dict[str, Any] | None = None,
    ) -> AgentOutput:
        shared_state = shared_state or {}
        payload = extra_payload or {}
        prompt = self._build_prompt(user_query, shared_state, retrieved_context, payload)
        mcp_notes = self._format_mcp_observations(user_query)
        combined_prompt = textwrap.dedent(
            f"""{prompt}\n\nMCP observations:\n{mcp_notes}"""
        )
        response = self.gemini.generate(
            combined_prompt,
            system_instruction=self.system_instruction,
        )
        return AgentOutput(
            role=self.name,
            summary=self._summarize(response),
            evidence=self._evidence_block(retrieved_context),
            raw_model_output=response.text,
            metadata=self._build_metadata(response, payload, shared_state),
        )

    def _format_mcp_observations(self, query: str) -> str:
        observations = self.mcp_bridge.gather_observations(query)
        if not observations:
            return "- (offline) MCP bridge not available in this sandbox."
        return "\n".join(f"- {note}" for note in observations)

    def _evidence_block(self, retrieved_context: RetrievedContext | None) -> str:
        if not retrieved_context or not retrieved_context.passages:
            return "No additional RAG context was retrieved for this agent."
        return retrieved_context.as_bullet_list()

    def _summarize(self, response: GeminiResponse) -> str:
        return response.text.strip()

    def _build_metadata(
        self,
        response: GeminiResponse,
        extra_payload: dict[str, Any],
        shared_state: dict[str, Any],
    ) -> dict[str, Any]:
        """Return structured output derived from deterministic signals."""

        return extra_payload

    @abstractmethod
    def _build_prompt(
        self,
        user_query: str,
        shared_state: dict[str, Any],
        retrieved_context: RetrievedContext | None,
        extra_payload: dict[str, Any],
    ) -> str:
        """Return the concrete prompt that should be given to Gemini."""
