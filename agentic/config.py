"""Configuration helpers for the healthcare multi-agent system."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import yaml  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    yaml = None  # type: ignore


@dataclass(slots=True)
class AgenticSettings:
    """Runtime configuration for the agentic healthcare system."""

    gemini_model: str = field(
        default_factory=lambda: os.getenv("GEMINI_MODEL")
        or os.getenv("GOOGLE_MODEL")
        or "models/gemini-1.5-flash"
    )
    gemini_api_key: str | None = field(
        default_factory=lambda: os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
    )
    temperature: float = 0.2
    top_k: int = 4
    knowledge_base_path: str = "agentic/data/pakistan_health_guidelines.md"
    chunk_size: int = 450
    chunk_overlap: int = 60
    mcp_servers: list[str] | None = None
    triage_data_path: str = "triage_data_large.csv"
    facility_data_path: str = "facility_data_large.csv"
    eligibility_data_path: str = "eligibility_data_large.csv"


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

    defaults = AgenticSettings()

    return AgenticSettings(
        gemini_model=payload.get("gemini_model", defaults.gemini_model),
        gemini_api_key=payload.get("gemini_api_key", defaults.gemini_api_key),
        temperature=float(payload.get("temperature", defaults.temperature)),
        top_k=int(payload.get("top_k", defaults.top_k)),
        knowledge_base_path=payload.get(
            "knowledge_base_path", defaults.knowledge_base_path
        ),
        chunk_size=int(payload.get("chunk_size", defaults.chunk_size)),
        chunk_overlap=int(payload.get("chunk_overlap", defaults.chunk_overlap)),
        mcp_servers=list(payload.get("mcp_servers", []) or []),
        triage_data_path=payload.get("triage_data_path", defaults.triage_data_path),
        facility_data_path=payload.get("facility_data_path", defaults.facility_data_path),
        eligibility_data_path=payload.get(
            "eligibility_data_path",
            defaults.eligibility_data_path,
        ),
    )
