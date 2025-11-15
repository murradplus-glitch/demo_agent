"""Healthcare multi-agent orchestrator built on top of the template."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

from agentic.agents.base import AgentOutput
from agentic.agents.care_planner import CarePlanAgent
from agentic.agents.patient_intake import PatientIntakeAgent
from agentic.agents.research import EvidenceResearchAgent
from agentic.agents.safety import SafetyReviewAgent
from agentic.config import AgenticSettings
from agentic.gemini import GeminiClient
from agentic.mcp_bridge import MCPToolBridge
from agentic.rag.pipeline import HealthcareRAGPipeline, RetrievedContext


@dataclass(slots=True)
class HealthcareMultiAgentReport:
    """Structured output returned to the caller."""

    patient_query: str
    rag_context: dict[str, Any]
    intake: AgentOutput
    research: AgentOutput
    care_plan: AgentOutput
    safety: AgentOutput

    def to_json(self) -> str:
        return json.dumps(
            {
                "patient_query": self.patient_query,
                "rag_context": self.rag_context,
                "intake": asdict(self.intake),
                "research": asdict(self.research),
                "care_plan": asdict(self.care_plan),
                "safety": asdict(self.safety),
            },
            indent=2,
        )


class HealthcareMultiAgentSystem:
    """Coordinates the RAG pipeline and the specialized agents."""

    def __init__(self, settings: AgenticSettings | None = None) -> None:
        self.settings = settings or AgenticSettings()
        self.gemini = GeminiClient(
            api_key=self.settings.gemini_api_key,
            model=self.settings.gemini_model,
            temperature=self.settings.temperature,
        )
        self.rag = HealthcareRAGPipeline(
            knowledge_base_path=self.settings.knowledge_base_path,
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )
        self.mcp_bridge = MCPToolBridge(self.settings.mcp_servers)

        self.intake_agent = PatientIntakeAgent(
            name="Patient Intake",
            system_instruction=(
                "You are a digital nurse capturing structured data from patient free text."
            ),
            gemini=self.gemini,
            mcp_bridge=self.mcp_bridge,
        )
        self.research_agent = EvidenceResearchAgent(
            name="Clinical Research",
            system_instruction="You map symptoms to evidence and cite the internal library.",
            gemini=self.gemini,
            mcp_bridge=self.mcp_bridge,
        )
        self.care_plan_agent = CarePlanAgent(
            name="Care Planner",
            system_instruction="Craft empathetic action plans that align with virtual care policy.",
            gemini=self.gemini,
            mcp_bridge=self.mcp_bridge,
        )
        self.safety_agent = SafetyReviewAgent(
            name="Safety Review",
            system_instruction="Ensure advice is clinically safe and escalate when needed.",
            gemini=self.gemini,
            mcp_bridge=self.mcp_bridge,
        )

    def run(self, patient_query: str, labs: str | None = None) -> HealthcareMultiAgentReport:
        retrieved_context = self._retrieve_context(patient_query)
        intake = self.intake_agent.run(
            user_query=patient_query,
            intake_summary="",
            retrieved_context=retrieved_context,
        )
        research = self.research_agent.run(
            user_query=patient_query,
            intake_summary=intake.summary,
            retrieved_context=retrieved_context,
            extra_payload={"recent_labs": labs or "Not supplied"},
        )
        care_plan = self.care_plan_agent.run(
            user_query=patient_query,
            intake_summary=intake.summary,
            retrieved_context=retrieved_context,
            extra_payload={"research_brief": research.summary},
        )
        safety = self.safety_agent.run(
            user_query=patient_query,
            intake_summary=intake.summary,
            retrieved_context=retrieved_context,
            extra_payload={"care_plan": care_plan.summary},
        )
        return HealthcareMultiAgentReport(
            patient_query=patient_query,
            rag_context=self.rag.describe(),
            intake=intake,
            research=research,
            care_plan=care_plan,
            safety=safety,
        )

    def _retrieve_context(self, patient_query: str) -> RetrievedContext:
        return self.rag.retrieve(patient_query, top_k=self.settings.top_k)
