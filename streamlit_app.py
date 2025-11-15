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


_INCOME_RANGE_OPTIONS = [
    "< Rs 25,000",
    "Rs 25,000 – 50,000",
    "Rs 50,000 – 100,000",
    "> Rs 100,000",
]


_PROVINCE_OPTIONS = [
    "",
    "Punjab",
    "Sindh",
    "KP",
    "Balochistan",
    "GB",
    "AJK",
    "Islamabad",
]


def _handle_login(cnic_value: str) -> None:
    profile = get_profile_by_cnic(cnic_value)
    st.session_state["cnic_entered"] = cnic_value.strip()
    st.session_state["profile_known"] = profile is not None
    st.session_state["profile_ready"] = profile is not None
    st.session_state["citizen_profile"] = profile
    st.session_state["latest_report"] = None


def _profile_summary(profile: dict[str, Any], known: bool) -> None:
    st.markdown("### Citizen Profile" + (" (Demo)" if known else " – Quick registration"))
    cols = st.columns(3)
    cols[0].metric("CNIC", profile.get("cnic", "—"))
    cols[1].metric("Name", profile.get("name", "—") or "—")
    cols[2].metric("City", profile.get("city", "—") or "—")
    cols = st.columns(3)
    cols[0].metric("Family size", str(profile.get("family_size", "—")))
    income = profile.get("income_per_month_pkrs")
    income_label = profile.get("income_range_label")
    if not income_label:
        income_label = f"Rs {income:,}" if isinstance(income, int) and income else "—"
    cols[1].metric("Monthly income", income_label or "—")
    eligibility = profile.get("sehat_card_eligible")
    if eligibility is True:
        cols[2].metric("Sehat Card Eligibility", "✅ Eligible")
    elif eligibility is False:
        cols[2].metric("Sehat Card Eligibility", "❌ Not eligible")
    else:
        cols[2].metric("Sehat Card Eligibility", "❔ Pending assessment")
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


def _income_label_to_value(label: str) -> int:
    mapping = {
        "< Rs 25,000": 20000,
        "Rs 25,000 – 50,000": 37500,
        "Rs 50,000 – 100,000": 75000,
        "> Rs 100,000": 120000,
    }
    return mapping.get(label, 0)


def _render_quick_registration(existing: dict[str, Any] | None = None) -> None:
    st.markdown("### Quick Registration (Demo)")
    st.info(
        "We don’t recognise this CNIC in the demo dataset."
        " Please share household details so agents can guide you properly."
    )
    st.caption("All fields are required so that eligibility, facility finding, and follow-up steps behave exactly like the target workflow.")
    default_income = existing.get("income_range_label") if existing else _INCOME_RANGE_OPTIONS[0]
    default_index = _INCOME_RANGE_OPTIONS.index(default_income) if default_income in _INCOME_RANGE_OPTIONS else 0
    default_province = existing.get("province", "") if existing else ""
    province_index = _PROVINCE_OPTIONS.index(default_province) if default_province in _PROVINCE_OPTIONS else 0
    with st.form("quick_registration_form"):
        family_size = st.number_input(
            "How many people live in your household?",
            min_value=1,
            max_value=20,
            value=int(existing.get("family_size", 4)) if existing else 4,
        )
        income_label = st.selectbox(
            "Monthly household income (range)",
            _INCOME_RANGE_OPTIONS,
            index=default_index,
            help="Approximate range helps us judge eligibility in demo mode.",
        )
        city = st.text_input(
            "Which city/town/village do you live in?",
            value=existing.get("city", "") if existing else "",
        )
        province = st.selectbox(
            "Province (optional)",
            _PROVINCE_OPTIONS,
            index=province_index,
        )
        rural_or_urban = st.radio(
            "Is this rural or urban?",
            ("Rural", "Urban"),
            index=0 if (existing or {}).get("rural_or_urban", "Rural") == "Rural" else 1,
            horizontal=True,
        )
        submitted = st.form_submit_button("Save & Continue")
        if submitted:
            approx_income = _income_label_to_value(income_label)
            profile = {
                "cnic": st.session_state.get("cnic_entered", ""),
                "name": existing.get("name") if existing else "New citizen",
                "family_size": int(family_size),
                "income_range_label": income_label,
                "income_per_month_pkrs": approx_income,
                "city": city.strip(),
                "district": city.strip(),
                "province": province or None,
                "rural_or_urban": rural_or_urban,
                "area": rural_or_urban,
                "sehat_card_eligible": None,
            }
            st.session_state["citizen_profile"] = profile
            st.session_state["profile_ready"] = True
            st.session_state["profile_known"] = False
            st.success("Details saved. You can now chat with SehatBuddy.")


def _render_chat() -> None:
    st.markdown("### Ask me about your health, Sehat Card, or nearby facilities")
    query = st.text_area(
        "Citizen question",
        value=st.session_state.get("last_query", ""),
        key="citizen_query",
        placeholder="How can I help you today?",
    )
    st.caption("Agent order: Triage → Program Eligibility → Facility Finder → Follow-Up → Health Analytics (Knowledge agent shares alerts in the report).")
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
st.markdown(
    """
    **Welcome!** This demo follows the exact workflow outlined in the usability brief:

    1. Enter a CNIC. Known CNICs instantly load Ali Khan or Ayesha Bibi’s household profile.
    2. Unknown CNICs unlock a Quick Registration form (family size, income range, city/town, province, rural/urban) before chatting.
    3. Ask a health question so the agents can run _Triage → Program Eligibility → Facility Finder → Follow-Up → Health Analytics_.
    """
)
cnic_input = st.text_input(
    "Enter your CNIC to continue",
    value=st.session_state.get("cnic_entered", ""),
    placeholder="e.g. 12345-1234567-1",
)
st.caption("If this CNIC is new, we’ll collect a few household details to guide you correctly.")
st.info("Demo CNIC: 12345-1234567-1 (use this to explore the system).")
examples = demo_cnic_examples()
if examples:
    st.caption("Additional demo CNICs you can use:")
    for example in examples:
        eligibility = "Eligible" if example.get("sehat_card_eligible") else "Not eligible"
        st.write(f"**{example['cnic']}** – {example.get('name')} ({eligibility}, {example.get('city')})")
if st.button("Continue", type="primary"):
    _handle_login(cnic_input)
    if not st.session_state.get("profile_known"):
        st.warning(
            "CNIC not found in demo profiles. The Quick Registration form below must be completed before the chat unlocks."
        )

if st.session_state.get("citizen_profile"):
    _profile_summary(st.session_state["citizen_profile"], st.session_state.get("profile_known", False))
    if not st.session_state.get("profile_known"):
        with st.expander("Update Quick Registration (Demo)"):
            _render_quick_registration(st.session_state["citizen_profile"])
elif st.session_state.get("cnic_entered"):
    _render_quick_registration()

if st.session_state.get("profile_ready"):
    _render_chat()
    _render_report()
else:
    if st.session_state.get("cnic_entered"):
        st.info("Complete the Quick Registration form above to unlock the SehatBuddy assistant.")
    else:
        st.info("Enter a CNIC above to unlock the SehatBuddy assistant.")
