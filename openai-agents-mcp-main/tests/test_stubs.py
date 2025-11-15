"""
Type stubs for tests to help with mypy type checking.
"""

from typing import Any


# Add type stubs for dynamic attributes used in tests
class RunnerContext:
    """Type stub for RunnerContext to satisfy mypy"""

    mcp_server_registry: Any
    mcp_config_path: str
    mcp_config: Any
    custom_attr: str
    another_attr: int
