"""Streamlit front-end that wraps the multi-agent healthcare system."""

from __future__ import annotations

import json
from typing import Any

import streamlit as st

from agentic import HealthcareMultiAgentSystem, load_settings
from agentic.data.citizen_profiles import (
    get_profile_by_cnic,
    list_demo_cnic_examples,
)

st.set_page_config(page_title="SehatBuddy – Connected Health Assistant", page_icon="🩺", layout="wide")


@st.cache_resource(show_spinner=False)
def get_system() -> HealthcareMultiAgentSystem:
    settings = load_settings()
    return HealthcareMultiAgentSystem(settings=settings)


@st.cache_data(show_spinner=False)
def demo_cnic_examples() -> list[dict[str, Any]]:
    return list_demo_cnic_examples()


def _ensure_session_defaults() -> None:
    st.session_state.setdefault("cnic_entered", "")
    st.session_state.setdefault("profile_known", False)
    st.session_state.setdefault("profile_ready", False)
    st.session_state.setdefault("citizen_profile", None)
    st.session_state.setdefault("latest_report", None)
    st.session_state.setdefault("last_query", "")


def _handle_login(cnic_value: str) -> None:
    profile = get_profile_by_cnic(cnic_value)
    st.session_state["cnic_entered"] = cnic_value.strip()
    st.session_state["profile_known"] = profile is not None
    st.session_state["profile_ready"] = profile is not None
    st.session_state["citizen_profile"] = profile
    st.session_state["latest_report"] = None


def _profile_summary(profile: dict[str, Any], known: bool) -> None:
    st.markdown("### Citizen Profile" + (" (Demo)" if known else ""))
    cols = st.columns(3)
    cols[0].metric("CNIC", profile.get("cnic", "—"))
    cols[1].metric("Name", profile.get("name", "—"))
    cols[2].metric("City", profile.get("city", "—"))
    cols = st.columns(3)
    cols[0].metric("Family size", str(profile.get("family_size", "—")))
    income = profile.get("income_per_month_pkrs")
    income_label = f"Rs {income:,}" if isinstance(income, int) and income else "—"
    cols[1].metric("Monthly income", income_label)
    eligibility = profile.get("sehat_card_eligible")
    emoji = "✅" if eligibility else "❌"
    cols[2].metric("Sehat Card Eligibility", emoji + (" Eligible" if eligibility else " Not eligible"))
    with st.expander("More household details"):
        st.write(
            {
                "District": profile.get("district"),
                "Province": profile.get("province"),
                "Area": profile.get("area"),
                "Rural / Urban": profile.get("rural_or_urban"),
                "NSER score": profile.get("nser_score"),
                "Linked triage case": profile.get("triage_case_id"),
                "Notes": profile.get("notes"),
            }
        )


def _render_unknown_form() -> None:
    st.info(
        "We could not auto-detect the profile for this CNIC."
        " Please provide the minimum household details so the agents can continue."
    )
    with st.form("manual_profile"):
        name = st.text_input("Name", value="")
        city = st.text_input("City / District", value="")
        province = st.text_input("Province", value="")
        area = st.text_input("Area or tehsil", value="")
        family_size = st.number_input("Family size", min_value=1, max_value=20, value=4)
        income = st.number_input("Monthly income (PKR)", min_value=0, max_value=500000, value=45000, step=1000)
        nser_score = st.number_input("NSER score", min_value=0, max_value=100, value=35)
        sehat_card_eligible = st.selectbox("Sehat Card eligibility", ["Unknown", "Yes", "No"], index=0)
        submitted = st.form_submit_button("Save household profile")
        if submitted:
            st.session_state["citizen_profile"] = {
                "cnic": st.session_state.get("cnic_entered", ""),
                "name": name,
                "city": city,
                "district": city,
                "province": province,
                "area": area,
                "family_size": int(family_size),
                "income_per_month_pkrs": int(income),
                "nser_score": int(nser_score),
                "sehat_card_eligible": sehat_card_eligible == "Yes",
            }
            st.session_state["profile_ready"] = True
            st.session_state["profile_known"] = False
            st.success("Profile saved. Scroll down to chat with SehatBuddy.")


def _render_chat() -> None:
    st.markdown("### Ask me about your health, Sehat Card, or nearby facilities")
    query = st.text_area(
        "Citizen question",
        value=st.session_state.get("last_query", ""),
        key="citizen_query",
        placeholder="e.g. My child has a fever and I want to use the Sehat Card. Where should I go?",
    )
    run_clicked = st.button("Run multi-agent assistant", type="primary", use_container_width=True)
    if run_clicked:
        if not query.strip():
            st.warning("Please enter a question so the agents have context.")
            return
        profile = st.session_state.get("citizen_profile") or {}
        system = get_system()
        st.session_state["last_query"] = query
        with st.spinner("Running the RAG-powered multi-agent workflow..."):
            report = system.run(patient_query=query, citizen_profile=profile)
        st.session_state["latest_report"] = json.loads(report.to_json())
        st.success("Multi-agent workflow complete.")


def _render_report() -> None:
    report = st.session_state.get("latest_report")
    if not report:
        return
    st.divider()
    st.markdown("### Agent responses")
    st.caption(f"Workflow backend: {report.get('workflow_backend')}")
    _agent_sections = {
        "Triage": report.get("triage"),
        "Program eligibility": report.get("program_eligibility"),
        "Facility finder": report.get("facility_finder"),
        "Follow-up": report.get("follow_up"),
        "Health analytics": report.get("health_analytics"),
        "Knowledge": report.get("knowledge"),
    }
    for label, payload in _agent_sections.items():
        if not payload:
            continue
        st.markdown(f"#### {label}")
        st.write(payload.get("summary", "No summary returned."))
        evidence = payload.get("evidence")
        if evidence:
            st.caption(evidence)
        metadata = payload.get("metadata")
        if metadata:
            with st.expander("Structured metadata"):
                st.write(metadata)


# ---------------------------------------------------------------------------
# UI start
# ---------------------------------------------------------------------------
_ensure_session_defaults()
st.title("SehatBuddy – Connected Health Assistant (Demo)")
cnic_input = st.text_input("Enter your CNIC", value=st.session_state.get("cnic_entered", ""), placeholder="e.g. 12345-1234567-1")
examples = demo_cnic_examples()
if examples:
    st.caption("Demo CNICs you can use:")
    for example in examples:
        eligibility = "Eligible" if example.get("sehat_card_eligible") else "Not eligible"
        st.write(f"**{example['cnic']}** – {example.get('name')} ({eligibility}, {example.get('city')})")
if st.button("Continue", type="primary"):
    _handle_login(cnic_input)
    if not st.session_state.get("profile_known"):
        st.warning("CNIC not found in demo profiles. Fill in the details below to continue.")

if st.session_state.get("citizen_profile"):
    _profile_summary(st.session_state["citizen_profile"], st.session_state.get("profile_known", False))
elif st.session_state.get("cnic_entered"):
    _render_unknown_form()

if st.session_state.get("profile_ready"):
    _render_chat()
    _render_report()
else:
    st.info("Enter a CNIC above to unlock the SehatBuddy assistant.")
