"""Utility that wires MCP tools into the local orchestrator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(slots=True)
class MCPStatus:
    enabled: bool
    reason: str | None = None
    servers: list[str] | None = None


class MCPToolBridge:
    """Thin wrapper that would call MCP tools when dependencies are available."""

    def __init__(self, servers: Iterable[str] | None = None) -> None:
        self.servers = list(servers or [])
        self.status = self._detect_capabilities()

    def _detect_capabilities(self) -> MCPStatus:
        try:
            import importlib.util

            spec = importlib.util.find_spec("mcp_agent")
            if spec is None:
                return MCPStatus(enabled=False, reason="mcp_agent package missing", servers=self.servers)
        except ModuleNotFoundError:
            return MCPStatus(enabled=False, reason="mcp_agent package missing", servers=self.servers)
        return MCPStatus(enabled=True, servers=self.servers)

    def gather_observations(self, query: str) -> list[str]:
        """Return diagnostic notes gathered via MCP tools.

        The sandbox used for evaluation typically does not provide the mcp_agent
        dependency or long-running subprocess permissions. In that case we log a
        helpful explanation so that the calling agent can surface it to the user.
        """

        if not self.status.enabled:
            return [
                "MCP integration disabled: "
                f"{self.status.reason or 'unknown reason'}. Configure servers once dependencies are available."
            ]

        # Placeholder for when the dependency is provided. The logic intentionally
        # keeps the interface stable so that wiring it up later only requires
        # implementing this branch.
        return [
            "(placeholder) MCP tools available but not invoked in this offline build.",
            f"Configured servers: {', '.join(self.servers) if self.servers else 'none'}",
            f"Query fingerprint: {hash(query) & 0xFFFF}",
        ]
