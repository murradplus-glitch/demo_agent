"""Test configuration for agents_mcp package."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from agents.run_context import RunContextWrapper
from mcp.types import TextContent
from mcp_agent.config import MCPServerSettings, MCPSettings

from agents_mcp import RunnerContext


@pytest.fixture
def mock_mcp_tool():
    """Create a mock MCP tool with necessary attributes."""
    tool = MagicMock()
    tool.name = "mock_tool"
    tool.description = "A mock MCP tool for testing"
    tool.inputSchema = {
        "type": "object",
        "properties": {
            "input": {
                "type": "string",
                "description": "The input for the tool",
            }
        },
        "required": ["input"],
    }
    return tool


@pytest.fixture
def mock_mcp_tools_result():
    """Create a mock ListToolsResult with tools."""
    result = MagicMock()

    tool1 = MagicMock()
    tool1.name = "fetch"
    tool1.description = "Fetch content from a URL"
    tool1.inputSchema = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The URL to fetch",
            }
        },
        "required": ["url"],
    }

    tool2 = MagicMock()
    tool2.name = "read_file"
    tool2.description = "Read a file from the filesystem"
    tool2.inputSchema = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "The path to the file",
            }
        },
        "required": ["path"],
    }

    result.tools = [tool1, tool2]
    return result


@pytest.fixture
def mock_mcp_call_result():
    """Create a mock CallToolResult with text content."""
    result = MagicMock()
    result.isError = False

    # Create content as TextContent
    content = TextContent(type="text", text="Mock tool response content")
    result.content = content

    return result


@pytest.fixture
def mock_mcp_call_error_result():
    """Create a mock CallToolResult with an error."""
    result = MagicMock()
    result.isError = True

    # Create error content
    content = TextContent(type="text", text="Mock tool error message")
    result.content = content

    return result


@pytest.fixture
def mock_mcp_aggregator():
    """Create a mock MCP aggregator with necessary methods."""
    aggregator = AsyncMock()
    aggregator.initialized = True
    aggregator.agent_name = "test_agent"

    # Setup methods
    aggregator.list_tools = AsyncMock()
    aggregator.call_tool = AsyncMock()
    aggregator.__aexit__ = AsyncMock()

    return aggregator


@pytest.fixture
def mock_mcp_settings():
    """Create mock MCP settings with test servers."""
    return MCPSettings(
        servers={
            "fetch": MCPServerSettings(
                command="mock_fetch_command",
                args=["mock_fetch_arg"],
            ),
            "filesystem": MCPServerSettings(
                command="mock_fs_command",
                args=["mock_fs_arg"],
            ),
        }
    )


@pytest.fixture
def mock_runner_context(mock_mcp_settings):
    """Create a RunnerContext with MCP settings."""
    context = RunnerContext(mcp_config=mock_mcp_settings)
    return context


@pytest.fixture
def run_context_wrapper(mock_runner_context):
    """Create a RunContextWrapper with the mock runner context."""

    return RunContextWrapper(context=mock_runner_context)


# Using the built-in pytest-asyncio event_loop fixture instead of defining our own
