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

```bash
python -m pip install -r requirements.txt
cp agentic/config.example.yaml agentic/config.yaml  # optional but handy
export GEMINI_API_KEY=your-key  # or edit the config file
python -m agentic.examples.run_healthcare_system
```

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
