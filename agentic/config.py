"""Configuration helpers for the healthcare multi-agent system."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import yaml  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    yaml = None  # type: ignore


@dataclass(slots=True)
class AgenticSettings:
    """Runtime configuration for the agentic healthcare system."""

    gemini_model: str = "models/gemini-1.5-flash"
    gemini_api_key: str | None = None
    temperature: float = 0.2
    top_k: int = 4
    knowledge_base_path: str = "agentic/data/medical_guidelines.md"
    chunk_size: int = 450
    chunk_overlap: int = 60
    mcp_servers: list[str] | None = None


DEFAULT_CONFIG = Path("agentic/config.yaml")


def load_settings(path: str | Path | None = None) -> AgenticSettings:
    """Load settings from a YAML file if it exists."""

    config_path = Path(path) if path is not None else DEFAULT_CONFIG
    if not config_path.exists():
        return AgenticSettings()

    if yaml is None:
        raise RuntimeError(
            "PyYAML is required to load configuration files. Install pyyaml or "
            "provide configuration via environment variables."
        )

    with config_path.open("r", encoding="utf-8") as handle:
        payload: dict[str, Any] = yaml.safe_load(handle) or {}

    return AgenticSettings(
        gemini_model=payload.get("gemini_model", AgenticSettings.gemini_model),
        gemini_api_key=payload.get("gemini_api_key"),
        temperature=float(payload.get("temperature", AgenticSettings.temperature)),
        top_k=int(payload.get("top_k", AgenticSettings.top_k)),
        knowledge_base_path=payload.get(
            "knowledge_base_path", AgenticSettings.knowledge_base_path
        ),
        chunk_size=int(payload.get("chunk_size", AgenticSettings.chunk_size)),
        chunk_overlap=int(payload.get("chunk_overlap", AgenticSettings.chunk_overlap)),
        mcp_servers=list(payload.get("mcp_servers", []) or []),
    )
