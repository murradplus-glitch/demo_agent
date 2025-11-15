"""Agent abstractions used by the orchestrator."""

from __future__ import annotations

import textwrap
from abc import ABC, abstractmethod
from dataclasses import dataclass
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
        intake_summary: str,
        retrieved_context: RetrievedContext,
        extra_payload: dict[str, Any] | None = None,
    ) -> AgentOutput:
        prompt = self._build_prompt(user_query, intake_summary, retrieved_context, extra_payload or {})
        mcp_notes = self._format_mcp_observations(user_query)
        combined_prompt = textwrap.dedent(
            f"""{prompt}\n\nMCP observations:\n{mcp_notes}"""
        )
        response = self.gemini.generate(combined_prompt, system_instruction=self.system_instruction)
        return AgentOutput(
            role=self.name,
            summary=self._summarize(response),
            evidence=self._evidence_block(retrieved_context),
            raw_model_output=response.text,
        )

    def _format_mcp_observations(self, query: str) -> str:
        observations = self.mcp_bridge.gather_observations(query)
        if not observations:
            return "- (offline) MCP bridge not available in this sandbox."
        return "\n".join(f"- {note}" for note in observations)

    def _evidence_block(self, retrieved_context: RetrievedContext) -> str:
        return retrieved_context.as_bullet_list()

    def _summarize(self, response: GeminiResponse) -> str:
        return response.text.strip()

    @abstractmethod
    def _build_prompt(
        self,
        user_query: str,
        intake_summary: str,
        retrieved_context: RetrievedContext,
        extra_payload: dict[str, Any],
    ) -> str:
        """Return the concrete prompt that should be given to Gemini."""
