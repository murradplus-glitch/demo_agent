# Healthcare RAG Multi-Agent Template

This repository wires a healthcare-focused, retrieval-augmented multi-agent workflow on top of the OpenAI Agents + MCP template provided in `openai-agents-mcp-main`. The new `agentic` module exposes a Gemini-compatible pipeline that:

1. Chunks an internal knowledge base and serves it via a lightweight vector store.
2. Spins up four collaborating agents (intake, research, care planning, safety) that each call Gemini with role-specific instructions.
3. Surfaces an MCP bridge so that, when the `mcp_agent` dependency becomes available, the same orchestration can hydrate additional tools.

Because only the Gemini API is available for the challenge, the client is implemented with the public REST endpoint and gracefully degrades to deterministic offline responses when the API key is not configured.

## Quick start

```bash
cp agentic/config.example.yaml agentic/config.yaml  # optional but handy
export GEMINI_API_KEY=your-key  # or edit the config file
python -m agentic.examples.run_healthcare_system
```

### Configure Gemini

Set the `GEMINI_API_KEY` environment variable or edit `agentic/config.yaml` to provide `gemini_api_key` and change defaults such as the model name, chunking strategy, or MCP server list. If the config file is missing, the system falls back to the safe defaults baked into `AgenticSettings`.

### Customize the knowledge base

Add Markdown files to `agentic/data/` and point `knowledge_base_path` to the aggregated file. The `HealthcareRAGPipeline` automatically chunks and indexes the content using a lightweight cosine-similarity vector store that does not require external packages.

### MCP integration

The `MCPToolBridge` detects whether the `mcp_agent` dependency is available. In offline sandboxes the bridge returns explanatory notes so the multi-agent run remains transparent. When the dependency is supplied the hook can be expanded to call real MCP tools without changing the orchestrator interface.
