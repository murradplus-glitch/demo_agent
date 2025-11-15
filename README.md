# Healthcare Multi-Agent LangGraph System

This repository hosts a healthcare assistant that mirrors the challenge brief: a multi-agent system (built with a LangGraph-styl
e pipeline) that can triage citizens, check Sehat Card eligibility, suggest nearby facilities, schedule follow ups, and surface
population-level alerts. The project extends the OpenAI Agents + MCP template under `openai-agents-mcp-main` and adds:

1. A LangGraph-compatible state machine (with an offline stub when the dependency cannot be installed) that wires together six co
   operative agents.
2. A lightweight RAG stack that chunks guidance extracted from the provided PDF (`agentic/data/pakistan_health_guidelines.md`).
3. CSV-backed mock data sources (`triage_data_large.csv`, `facility_data_large.csv`, `eligibility_data_large.csv`) to emulate nat
   ional registries and BHU directories.

The Gemini client speaks directly to the REST API. When the environment lacks an API key the system returns deterministic placeho
lders so the workflow and tests can run offline.

## Quick start

This repository still ships the exact combination the brief called for: a
retrieval-augmented generation (RAG) stack wired into a LangGraph-style
multi-agent workflow. The `HealthcareMultiAgentSystem` class initialises the
`HealthcareRAGPipeline`, feeds its retrieved passages to each specialised agent,
and drives a `StateGraph` defined in `agentic/orchestrator.py`. Nothing in the
recent CNIC/registration updates changed that plumbing – you still get a
LangGraph-executed chain of six agents whose responses cite the underlying RAG
context.

```bash
python -m pip install -r requirements.txt
cp agentic/config.example.yaml agentic/config.yaml  # optional but handy
export GEMINI_API_KEY=your-key  # or edit the config file
python -m agentic.examples.run_healthcare_system
```

> ✅ **Smoke test:** `python -m compileall agentic streamlit_app.py` compiles the
> orchestrator, agents, and Streamlit UI so you can confirm the environment is
> healthy even before launching Streamlit.

> 💡 **Tip:** Run `python -m agentic.examples.check_setup` to confirm LangGraph is
> installed and that the mock CSV/PDF data files can be found on your machine.

### Streamlit demo UI

Launch the end-to-end experience with Streamlit:

```bash
streamlit run streamlit_app.py
```

The landing screen asks for a CNIC. If you enter one of the demo CNICs listed
below the UI automatically populates the citizen profile (name, city, family
size, NSER score, income, and Sehat Card eligibility) before the RAG-powered
multi-agent workflow starts. Unknown CNICs trigger a lightweight form so the
user can supply the minimum details the agents need.

#### Quick registration & “no city yet” flow

Case B from the usability brief (an unexpected CNIC) is implemented exactly as
described:

1. The login step still accepts any CNIC format but, when the CNIC is missing
   from `agentic/data/demo_citizen_profiles.csv`, the UI renders a **Quick
   Registration (Demo)** form that blocks the assistant until the user submits
   family size, an income bracket, city/town, province (optional), and rural vs
   urban.
2. The Facility Finder agent refuses to hallucinate locations. If the user
   skips the city field (or clears it later), Triage and Program Eligibility
   continue running, but the facility node pauses and politely asks for the
   city before showing BHU or hospital names.
3. When the citizen later types a phrase such as “I live in Multan,” the system
   extracts the city name, updates the profile, and lets the facility
   recommendations resume.

This behaviour is surfaced in `streamlit_app.py` (gating the chat UI and
showing the form/expander) and `agentic/orchestrator.py` (where the LangGraph
node checks for a city before querying the BHU directory).

### Run the project locally (step-by-step)

The commands above are usually all you need, but if you are starting from a
fresh laptop the following checklist walks through the exact flow:

1. **Clone the repo & pick a Python** – install Python 3.10+ and run
   `git clone https://github.com/<your-org>/demo_agent.git && cd demo_agent`.
2. **Create an isolated environment** – `python -m venv .venv && source .venv/bin/activate`
   (or use Conda/uv if you prefer).
3. **Install dependencies** – `python -m pip install --upgrade pip` followed by
   `pip install -r requirements.txt`. The requirements file already bundles
   LangGraph, Streamlit, and the utility libraries used by the multi-agent
   pipeline.
4. **Configure keys & data paths** – copy
   `agentic/config.example.yaml` to `agentic/config.yaml` if you want to persist
   settings, then set `export GEMINI_API_KEY=...` (or fill the config file).
   Leave the CSV/Markdown data files in the repo root so the mock registries can
   be found without editing paths.
5. **Smoke-test the environment** – run
   `python -m compileall agentic streamlit_app.py` to catch syntax/import issues
   early, then `python -m agentic.examples.run_healthcare_system` to confirm the
   LangGraph pipeline and RAG knowledge base initialise correctly.
6. **Launch the Streamlit UI** – `streamlit run streamlit_app.py` starts the
   CNIC → quick registration → chat workflow described in the usability brief.
   Open the printed local URL (e.g. `http://localhost:8501`) in a browser and
   use the demo CNICs listed below.
7. **(Optional) Develop with hot reload** – keep Streamlit running and edit
   Python files; Streamlit will auto-reload the interface so you can iterate on
   prompts, agent logic, or registration UI copy.

Following these steps gives you the exact experience validated in the challenge:
the known/unknown CNIC flows, the five-agent LangGraph pipeline, and the RAG
context feeding every response.

### Agents and workflow

The LangGraph state machine executes the agents in the following order (the
`workflow_backend` field in the JSON report confirms whether the real
`langgraph.graph.StateGraph` engine or the offline fallback executed the run):

1. **Triage Agent** – classifies the symptoms (self-care, BHU, hospital/emergency) using mock historical cases, the MCP bridge, a
   nd RAG guidance.
2. **Program Eligibility Agent** – checks Sehat Card and vaccination program eligibility, pre-filling forms with structured data.
3. **Facility Finder Agent** – ranks nearby BHUs/hospitals based on the citizen’s city/area, doctor availability, and triage seve
   rity.
4. **Follow-Up Agent** – schedules reminders, medication adherence nudges, and escalation rules.
5. **Health Analytics Agent** – correlates the case with national trends (keyword spikes, emergency percentages) plus RAG context
   from the PDF-derived knowledge base.
6. **Knowledge Agent** – scans anonymised cases for outbreaks (dengue/measles/cholera) and explains what to share with health aut
   horities.

Each agent consults the MCP bridge (which logs an explanatory message when the sandbox lacks MCP tooling) so the user can underst
and whether external tools participated.

### Configure Gemini

Set the `GEMINI_API_KEY` environment variable or edit `agentic/config.yaml` to provide `gemini_api_key` and change defaults such
as the model name, chunking strategy, or MCP server list. If the config file is missing, the system falls back to the safe defaul
ts baked into `AgenticSettings`.

### Customize the knowledge base

Add Markdown files to `agentic/data/` and point `knowledge_base_path` to the aggregated file. The default file, `agentic/data/pak
istan_health_guidelines.md`, summarises the PDF shipped with the challenge. The `HealthcareRAGPipeline` automatically chunks and 
indexes the content using a lightweight cosine-similarity vector store that does not require external packages.

### Mock registries

The system reads three CSV files in the repository root:

- `triage_data_large.csv` – historical symptom classifications used by the triage, analytics, and knowledge agents.
- `eligibility_data_large.csv` – sample NSER/Sehat Card data used by the program eligibility agent.
- `facility_data_large.csv` – BHU and hospital directory entries consumed by the facility finder.
- `agentic/data/demo_citizen_profiles.csv` – CNIC-to-profile mappings consumed by the Streamlit UI and linked to the first two
  triage cases, ensuring we have deterministic household metadata for the demo login flow.

### Demo CNICs

Use the following CNICs on the Streamlit login screen:

| CNIC | Name | City | Linked triage case | Sehat Card eligibility |
| --- | --- | --- | --- | --- |
| `12345-1234567-1` | Ali Khan | Lahore | Case #1 – high fever & breathing difficulty | ✅ Eligible |
| `54321-7654321-0` | Ayesha Bibi | Rawalpindi | Case #2 – sore throat & slight fever | ❌ Not eligible |

These records live in `agentic/data/demo_citizen_profiles.csv` so the system can
auto-detect family size, income bracket, and location information without
relying on the `applicant_id` column from `eligibility_data_large.csv`.

Update the paths in `agentic/config.yaml` if you move the files.

### MCP integration

The `MCPToolBridge` detects whether the `mcp_agent` dependency is available. In offline sandboxes the bridge returns explanatory
notes so the multi-agent run remains transparent. When the dependency is supplied the hook can be expanded to call real MCP tools
without changing the orchestrator interface.
