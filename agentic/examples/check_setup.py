"""Quick diagnostic script to ensure the workflow runs locally."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict

from agentic.config import DEFAULT_CONFIG, load_settings
from agentic.langgraph_stub import (
    LANGGRAPH_IMPORT_ERROR,
    USING_LANGGRAPH,
    describe_langgraph_backend,
)


class FileStatus(TypedDict):
    path: str
    exists: bool
    size_bytes: int


def _file_status(path: str | Path) -> FileStatus:
    resolved = Path(path).expanduser()
    return {
        "path": str(resolved),
        "exists": resolved.exists(),
        "size_bytes": resolved.stat().st_size if resolved.exists() else 0,
    }


def main() -> None:
    settings = load_settings()
    data_files: dict[str, FileStatus] = {
        "triage": _file_status(settings.triage_data_path),
        "facility": _file_status(settings.facility_data_path),
        "eligibility": _file_status(settings.eligibility_data_path),
    }
    knowledge_base = _file_status(settings.knowledge_base_path)
    summary: dict[str, object] = {
        "langgraph_backend": describe_langgraph_backend(),
        "langgraph_ready": USING_LANGGRAPH,
        "config_file": {
            "path": str(DEFAULT_CONFIG),
            "exists": DEFAULT_CONFIG.exists(),
        },
        "data_files": data_files,
        "knowledge_base": knowledge_base,
        "recommendations": [],
    }

    recommendations: list[str] = []
    if not USING_LANGGRAPH:
        msg = (
            "LangGraph is not installed. Install it via 'pip install langgraph' "
            "to run the real workflow backend."
        )
        if LANGGRAPH_IMPORT_ERROR:
            msg += f" Import error: {LANGGRAPH_IMPORT_ERROR}"
        recommendations.append(msg)

    for label, info in data_files.items():
        if not info["exists"]:
            recommendations.append(
                f"Missing {label} dataset at {info['path']}. Keep the CSVs in the repo root or update config."
            )

    if not knowledge_base["exists"]:
        recommendations.append(
            f"Knowledge base file is missing at {knowledge_base['path']}. Add the Markdown file or update `knowledge_base_path`."
        )

    summary["recommendations"] = recommendations

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
