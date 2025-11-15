"""Healthcare multi-agent orchestrator built on top of the template."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
import re
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
    user_summary: str
    citizen_response: str
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
                "user_summary": self.user_summary,
                "citizen_response": self.citizen_response,
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
        profile = dict(citizen_profile) if citizen_profile else self._default_profile()
        profile = self._apply_city_hint(profile, patient_query)
        initial_state = HealthcareGraphState(
            patient_query=patient_query,
            citizen_profile=profile,
            retrieved_context=retrieved_context,
        )
        final_state = self._coerce_graph_state(self.graph.invoke(initial_state))
        assert final_state.triage and final_state.program_eligibility
        assert final_state.facility_finder and final_state.follow_up
        assert final_state.health_analytics and final_state.knowledge
        friendly_summary = self._compose_user_friendly_summary(final_state)
        structured_response = self._compose_structured_response(final_state)
        return HealthcareMultiAgentReport(
            patient_query=patient_query,
            citizen_profile=profile,
            rag_context=self.rag.describe(),
            workflow_backend=self.workflow_backend,
            user_summary=friendly_summary,
            citizen_response=structured_response,
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
        city = (state.citizen_profile.get("city") or "").strip()
        if not city:
            state.facility_finder = AgentOutput(
                role=self.facility_finder_agent.name,
                summary=(
                    "I can’t recommend an exact facility yet because I don’t know your city or town. "
                    "Please tell me your city (for example, 'Multan') so I can share the nearest BHU or hospital."
                ),
                evidence="Facility lookup paused until a city/town is provided.",
                raw_model_output="Facility finder skipped due to missing city.",
                metadata={"facility_options": [], "needs_city": True},
            )
            return state
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

    def _coerce_graph_state(
        self, state: HealthcareGraphState | dict[str, Any]
    ) -> HealthcareGraphState:
        if isinstance(state, HealthcareGraphState):
            return state
        if isinstance(state, dict):
            retrieved_context = state.get("retrieved_context")
            if not isinstance(retrieved_context, RetrievedContext):
                retrieved_context = RetrievedContext(
                    question=str(state.get("patient_query", "")),
                    passages=[],
                )
            citizen_profile_raw = state.get("citizen_profile")
            if isinstance(citizen_profile_raw, dict):
                citizen_profile = dict(citizen_profile_raw)
            elif citizen_profile_raw is None:
                citizen_profile = {}
            elif hasattr(citizen_profile_raw, "__dict__"):
                citizen_profile = dict(vars(citizen_profile_raw))
            else:
                citizen_profile = dict(citizen_profile_raw)
            return HealthcareGraphState(
                patient_query=str(state.get("patient_query", "")),
                citizen_profile=citizen_profile,
                retrieved_context=retrieved_context,
                triage=state.get("triage"),
                program_eligibility=state.get("program_eligibility"),
                facility_finder=state.get("facility_finder"),
                follow_up=state.get("follow_up"),
                health_analytics=state.get("health_analytics"),
                knowledge=state.get("knowledge"),
            )
        raise TypeError(
            "Unsupported state returned by the LangGraph backend: "
            f"{type(state)!r}"
        )

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

    def _apply_city_hint(self, profile: dict[str, Any], patient_query: str) -> dict[str, Any]:
        """Update the profile with a city if the citizen mentions it in the query."""

        if (profile.get("city") or "").strip():
            return profile
        extracted = self._extract_city_from_query(patient_query)
        if not extracted:
            return profile
        profile = dict(profile)
        profile["city"] = extracted
        profile.setdefault("district", extracted)
        return profile

    def _extract_city_from_query(self, patient_query: str) -> str | None:
        lowered = patient_query.strip()
        if not lowered:
            return None
        patterns = [
            r"\bi live in\s+([A-Za-z\s]+)",
            r"\bi am from\s+([A-Za-z\s]+)",
            r"\bmy city is\s+([A-Za-z\s]+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, lowered, flags=re.IGNORECASE)
            if not match:
                continue
            candidate = match.group(1).strip()
            candidate = re.split(r"\b(?:and|but|so|because)\b", candidate, maxsplit=1, flags=re.IGNORECASE)[0]
            candidate = re.split(r"[,\.]+", candidate, maxsplit=1)[0]
            candidate = candidate.strip().strip(",. !?")
            if candidate:
                return candidate.title()
        return None

    def _retrieve_context(self, patient_query: str) -> RetrievedContext:
        return self.rag.retrieve(patient_query, top_k=self.settings.top_k)

    def _compose_user_friendly_summary(self, state: HealthcareGraphState) -> str:
        """Condense the multi-agent output into 2–3 conversational sentences."""

        triage_meta = state.triage.metadata if state.triage else {}
        facility_meta = state.facility_finder.metadata if state.facility_finder else {}
        program_meta = state.program_eligibility.metadata if state.program_eligibility else {}
        follow_up_meta = state.follow_up.metadata if state.follow_up else {}

        sentences: list[str] = []

        severity_sentence = self._build_severity_sentence(triage_meta)
        if severity_sentence:
            sentences.append(severity_sentence)

        facility_sentence = self._build_facility_sentence(
            facility_meta,
            state.citizen_profile,
            triage_meta.get("severity"),
        )
        if facility_sentence:
            sentences.append(facility_sentence)

        eligibility_sentence = self._build_eligibility_sentence(program_meta)
        if eligibility_sentence:
            sentences.append(eligibility_sentence)

        follow_up_sentence = self._build_follow_up_sentence(follow_up_meta)
        if follow_up_sentence:
            sentences.append(follow_up_sentence)

        trimmed = [sentence.strip() for sentence in sentences if sentence.strip()][:3]
        combined = " ".join(trimmed)
        combined = " ".join(combined.split())  # collapse whitespace
        return combined or "Keep monitoring your symptoms and share more details so I can guide you."

    def _compose_structured_response(self, state: HealthcareGraphState) -> str:
        """Return the citizen-facing markdown message that mirrors the SOP layout."""

        triage_meta = state.triage.metadata if state.triage else {}
        severity = str(triage_meta.get("severity") or "").lower()
        facility_meta = state.facility_finder.metadata if state.facility_finder else {}

        sections: list[str] = []
        if severity == "emergency":
            sections.append(
                "⛔ **Possible emergency**\n"
                "These symptoms can be dangerous. Call 1122 or reach the nearest hospital emergency right now."
            )

        triage_text = self._safe_agent_summary(
            state.triage,
            "I need a few more symptom details (duration, fever, bleeding, breathing) to triage you safely.",
        )
        sections.append(f"**Triage result**\n{triage_text}")

        eligibility_text = self._safe_agent_summary(
            state.program_eligibility,
            "Share your family size, city, and income range so I can estimate Sehat Card support.",
        )
        sections.append(f"**Sehat Card / subsidy check**\n{eligibility_text}")

        if facility_meta.get("needs_city"):
            facility_text = "Tell me your city or town so I can list BHUs, RHCs, or hospitals nearby."
        else:
            facility_text = self._safe_agent_summary(
                state.facility_finder,
                "Start with your nearest BHU/clinic for assessment, and escalate to a hospital if danger signs appear.",
            )
        sections.append(f"**Where you can go**\n{facility_text}")

        if severity == "emergency":
            follow_up_text = "No reminder now—focus on emergency care and keep us updated once you’re safe."
        else:
            follow_up_text = self._safe_agent_summary(
                state.follow_up,
                "I can schedule a gentle reminder once you confirm how you're feeling after the visit.",
            )
        sections.append(f"**Follow-up**\n{follow_up_text}")

        return "\n\n".join(section.strip() for section in sections if section.strip())

    def _safe_agent_summary(
        self, agent_output: AgentOutput | None, fallback: str
    ) -> str:
        if agent_output and agent_output.summary and agent_output.summary.strip():
            return agent_output.summary.strip()
        return fallback

    def _build_severity_sentence(self, triage_meta: dict[str, Any]) -> str:
        severity = str(triage_meta.get("severity") or "").strip()
        recommended = str(triage_meta.get("recommended_action") or "").strip()
        severity_map = {
            "emergency": "This sounds like a possible emergency—head to the nearest emergency department or call 1122.",
            "hospital": "A hospital evaluation is safer so a doctor can examine you in person soon.",
            "bhu visit": "Plan a BHU or clinic visit soon so a clinician can check you properly.",
            "self-care": "It appears mild right now; continue home care but monitor for any danger signs.",
        }
        normalized = severity.lower()
        sentence = severity_map.get(normalized)
        if not sentence and severity:
            sentence = f"This appears to need a {severity.lower()} level response, so please seek care accordingly."
        if not sentence:
            sentence = "Keep monitoring your symptoms and seek professional care if they worsen."
        if recommended:
            trimmed = recommended.rstrip(". ")
            if trimmed and trimmed.lower() not in sentence.lower():
                sentence = f"{sentence.rstrip('.')} ({trimmed})."
        return sentence

    def _build_facility_sentence(
        self,
        facility_meta: dict[str, Any],
        profile: dict[str, Any],
        severity: str | None,
    ) -> str:
        if facility_meta.get("needs_city"):
            return "Tell me your city or town so I can list the nearest BHU or hospital."
        facilities = facility_meta.get("facility_options") or facility_meta.get("facilities") or []
        severity_normalized = (severity or "").lower()
        if facilities:
            top = facilities[0]
            location_bits = [part for part in [top.get("area"), top.get("city")] if part]
            location = ", ".join(location_bits)
            prefix = "Head straight to" if severity_normalized == "emergency" else "You can start at"
            sentence = f"{prefix} {top.get('name', 'a nearby facility')}"
            if location:
                sentence += f" in {location}"
            sentence += " for an in-person check."
            return sentence
        if severity_normalized == "emergency":
            return "Go to the closest emergency department or call 1122 immediately."
        if profile.get("city"):
            return f"Visit a BHU or clinic in {profile['city']} so a nurse or doctor can review you."
        return ""

    def _build_eligibility_sentence(self, program_meta: dict[str, Any]) -> str:
        decision = program_meta.get("eligibility") or {}
        eligible_value = str(decision.get("eligible") or "").strip().lower()
        reason = str(decision.get("reason") or "").strip().rstrip(".")
        if eligible_value == "yes":
            detail = f" – {reason}" if reason else ""
            return f"You're likely eligible for Sehat Card support{detail}."
        if eligible_value == "no":
            detail = f" – {reason}" if reason else ""
            return f"Sehat Card support may not apply yet{detail}; still bring your CNIC in case the clinic can re-check."
        return ""

    def _build_follow_up_sentence(self, follow_up_meta: dict[str, Any]) -> str:
        plan = follow_up_meta.get("follow_up_plan") or {}
        reminders = plan.get("reminders") if isinstance(plan, dict) else None
        if reminders:
            first = str(reminders[0]).rstrip(". ")
            if first:
                return f"I'll nudge you to {first.lower()} and check back if things worsen."
        return ""
