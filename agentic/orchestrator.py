"""Healthcare multi-agent orchestrator built on top of the template."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

from agentic.agents import (
    AgentOutput,
    FacilityFinderAgent,
    FollowUpAgent,
    HealthAnalyticsAgent,
    KnowledgeAgent,
    ProgramEligibilityAgent,
    TriageAgent,
)
from agentic.config import AgenticSettings
from agentic.data.repository import HealthcareDataRepository
from agentic.gemini import GeminiClient
from agentic.langgraph_stub import END, StateGraph, describe_langgraph_backend
from agentic.mcp_bridge import MCPToolBridge
from agentic.rag.pipeline import HealthcareRAGPipeline, RetrievedContext


@dataclass(slots=True)
class HealthcareMultiAgentReport:
    """Structured output returned to the caller."""

    patient_query: str
    citizen_profile: dict[str, Any]
    rag_context: dict[str, Any]
    workflow_backend: str
    triage: AgentOutput
    program_eligibility: AgentOutput
    facility_finder: AgentOutput
    follow_up: AgentOutput
    health_analytics: AgentOutput
    knowledge: AgentOutput

    def to_json(self) -> str:
        return json.dumps(
            {
                "patient_query": self.patient_query,
                "citizen_profile": self.citizen_profile,
                "rag_context": self.rag_context,
                "workflow_backend": self.workflow_backend,
                "triage": asdict(self.triage),
                "program_eligibility": asdict(self.program_eligibility),
                "facility_finder": asdict(self.facility_finder),
                "follow_up": asdict(self.follow_up),
                "health_analytics": asdict(self.health_analytics),
                "knowledge": asdict(self.knowledge),
            },
            indent=2,
        )


@dataclass(slots=True)
class HealthcareGraphState:
    """Shared state that flows through the LangGraph pipeline."""

    patient_query: str
    citizen_profile: dict[str, Any]
    retrieved_context: RetrievedContext
    triage: AgentOutput | None = None
    program_eligibility: AgentOutput | None = None
    facility_finder: AgentOutput | None = None
    follow_up: AgentOutput | None = None
    health_analytics: AgentOutput | None = None
    knowledge: AgentOutput | None = None


class HealthcareMultiAgentSystem:
    """Coordinates the RAG pipeline, data repository, and LangGraph workflow."""

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
        self.data_repo = HealthcareDataRepository(
            triage_csv=self.settings.triage_data_path,
            facility_csv=self.settings.facility_data_path,
            eligibility_csv=self.settings.eligibility_data_path,
        )

        self.triage_agent = TriageAgent(
            name="Triage Agent",
            system_instruction=(
                "Classify symptoms as self-care, BHU visit, hospital visit, or emergency "
                "and describe the reasoning in plain Urdu+English."
            ),
            gemini=self.gemini,
            mcp_bridge=self.mcp_bridge,
            data_repo=self.data_repo,
        )
        self.program_eligibility_agent = ProgramEligibilityAgent(
            name="Program Eligibility Agent",
            system_instruction="Explain whether the citizen qualifies for Sehat Card or preventive programmes and cite rules.",
            gemini=self.gemini,
            mcp_bridge=self.mcp_bridge,
            data_repo=self.data_repo,
        )
        self.facility_finder_agent = FacilityFinderAgent(
            name="Facility Finder Agent",
            system_instruction="Recommend nearby facilities that can manage the case and clarify why each was selected.",
            gemini=self.gemini,
            mcp_bridge=self.mcp_bridge,
            data_repo=self.data_repo,
        )
        self.follow_up_agent = FollowUpAgent(
            name="Follow-Up Agent",
            system_instruction="Design reminders, medication adherence nudges, and escalation rules in friendly language.",
            gemini=self.gemini,
            mcp_bridge=self.mcp_bridge,
            data_repo=self.data_repo,
        )
        self.health_analytics_agent = HealthAnalyticsAgent(
            name="Health Analytics Agent",
            system_instruction="Correlate this case with broader analytics, cite datasets, and be transparent about uncertainty.",
            gemini=self.gemini,
            mcp_bridge=self.mcp_bridge,
            data_repo=self.data_repo,
        )
        self.knowledge_agent = KnowledgeAgent(
            name="Knowledge Agent",
            system_instruction="Detect outbreaks, describe data gaps, and outline how to notify authorities transparently.",
            gemini=self.gemini,
            mcp_bridge=self.mcp_bridge,
            data_repo=self.data_repo,
        )

        self.workflow_backend = describe_langgraph_backend()
        self.graph = self._build_graph()

    def run(
        self,
        patient_query: str,
        citizen_profile: dict[str, Any] | None = None,
    ) -> HealthcareMultiAgentReport:
        retrieved_context = self._retrieve_context(patient_query)
        profile = citizen_profile or self._default_profile()
        initial_state = HealthcareGraphState(
            patient_query=patient_query,
            citizen_profile=profile,
            retrieved_context=retrieved_context,
        )
        final_state = self.graph.invoke(initial_state)
        assert final_state.triage and final_state.program_eligibility
        assert final_state.facility_finder and final_state.follow_up
        assert final_state.health_analytics and final_state.knowledge
        return HealthcareMultiAgentReport(
            patient_query=patient_query,
            citizen_profile=profile,
            rag_context=self.rag.describe(),
            workflow_backend=self.workflow_backend,
            triage=final_state.triage,
            program_eligibility=final_state.program_eligibility,
            facility_finder=final_state.facility_finder,
            follow_up=final_state.follow_up,
            health_analytics=final_state.health_analytics,
            knowledge=final_state.knowledge,
        )

    # ------------------------------------------------------------------
    # LangGraph nodes
    # ------------------------------------------------------------------
    def _build_graph(self):
        builder = StateGraph(HealthcareGraphState)
        builder.add_node("triage", self._triage_node)
        builder.add_node("eligibility", self._program_eligibility_node)
        builder.add_node("facility", self._facility_node)
        builder.add_node("follow_up", self._follow_up_node)
        builder.add_node("analytics", self._analytics_node)
        builder.add_node("knowledge", self._knowledge_node)
        builder.add_edge("triage", "eligibility")
        builder.add_edge("eligibility", "facility")
        builder.add_edge("facility", "follow_up")
        builder.add_edge("follow_up", "analytics")
        builder.add_edge("analytics", "knowledge")
        builder.add_edge("knowledge", END)
        builder.set_entry_point("triage")
        return builder.compile()

    def _triage_node(self, state: HealthcareGraphState) -> HealthcareGraphState:
        triage_payload = self.triage_agent.analyze_symptoms(state.patient_query)
        output = self.triage_agent.run(
            user_query=state.patient_query,
            shared_state=self._state_snapshot(state),
            retrieved_context=state.retrieved_context,
            extra_payload=triage_payload,
        )
        state.triage = output
        return state

    def _program_eligibility_node(self, state: HealthcareGraphState) -> HealthcareGraphState:
        evaluation = self.program_eligibility_agent.evaluate_profile(state.citizen_profile)
        output = self.program_eligibility_agent.run(
            user_query=state.patient_query,
            shared_state=self._state_snapshot(state),
            retrieved_context=state.retrieved_context,
            extra_payload={"eligibility": evaluation},
        )
        state.program_eligibility = output
        return state

    def _facility_node(self, state: HealthcareGraphState) -> HealthcareGraphState:
        severity = None
        if state.triage and state.triage.metadata:
            severity = state.triage.metadata.get("severity")
        facilities = self.facility_finder_agent.recommend_facilities(
            state.citizen_profile,
            severity,
        )
        output = self.facility_finder_agent.run(
            user_query=state.patient_query,
            shared_state=self._state_snapshot(state),
            retrieved_context=state.retrieved_context,
            extra_payload={"facility_options": facilities},
        )
        state.facility_finder = output
        return state

    def _follow_up_node(self, state: HealthcareGraphState) -> HealthcareGraphState:
        triage_meta = state.triage.metadata if state.triage else {}
        facility_meta = state.facility_finder.metadata if state.facility_finder else {}
        plan = self.follow_up_agent.plan_follow_up(
            state.citizen_profile,
            triage_meta,
            facility_meta,
        )
        output = self.follow_up_agent.run(
            user_query=state.patient_query,
            shared_state=self._state_snapshot(state),
            retrieved_context=state.retrieved_context,
            extra_payload={"follow_up_plan": plan},
        )
        state.follow_up = output
        return state

    def _analytics_node(self, state: HealthcareGraphState) -> HealthcareGraphState:
        triage_meta = state.triage.metadata if state.triage else {}
        trends = self.health_analytics_agent.generate_trends(triage_meta)
        output = self.health_analytics_agent.run(
            user_query=state.patient_query,
            shared_state=self._state_snapshot(state),
            retrieved_context=state.retrieved_context,
            extra_payload={"trends": trends},
        )
        state.health_analytics = output
        return state

    def _knowledge_node(self, state: HealthcareGraphState) -> HealthcareGraphState:
        alerts = self.knowledge_agent.discover_alerts()
        output = self.knowledge_agent.run(
            user_query=state.patient_query,
            shared_state=self._state_snapshot(state),
            retrieved_context=state.retrieved_context,
            extra_payload={"alerts": alerts},
        )
        state.knowledge = output
        return state

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _state_snapshot(self, state: HealthcareGraphState) -> dict[str, Any]:
        return {
            "citizen_profile": state.citizen_profile,
            "triage": state.triage.metadata if state.triage else {},
            "programs": state.program_eligibility.metadata if state.program_eligibility else {},
            "facilities": state.facility_finder.metadata if state.facility_finder else {},
            "follow_up": state.follow_up.metadata if state.follow_up else {},
        }

    def _default_profile(self) -> dict[str, Any]:
        return {
            "name": "Ayesha",
            "age": 8,
            "city": "Lahore",
            "area": "Johar Town",
            "region": "Punjab",
            "nser_score": 24,
            "income_per_month_pkrs": 28000,
            "family_size": 6,
            "preferred_language": "Urdu",
            "conditions": ["Asthma"],
        }

    def _retrieve_context(self, patient_query: str) -> RetrievedContext:
        return self.rag.retrieve(patient_query, top_k=self.settings.top_k)
