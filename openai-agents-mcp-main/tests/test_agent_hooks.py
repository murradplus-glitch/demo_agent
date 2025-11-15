"""Tests for MCP agent hooks."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from agents.lifecycle import AgentHooks

from agents_mcp.agent import Agent
from agents_mcp.agent_hooks import MCPAgentHooks


@pytest.fixture
def mock_agent():
    """Create a mock MCP agent."""
    agent = Agent(
        name="TestAgent",
        instructions="Test instructions",
        mcp_servers=["fetch", "filesystem"],
    )
    # Mock the load_mcp_tools method
    agent.load_mcp_tools = AsyncMock()
    return agent


@pytest.fixture
def mock_original_hooks():
    """Create mock original hooks."""
    hooks = MagicMock(spec=AgentHooks)
    hooks.on_start = AsyncMock()
    hooks.on_end = AsyncMock()
    hooks.on_handoff = AsyncMock()
    hooks.on_tool_start = AsyncMock()
    hooks.on_tool_end = AsyncMock()
    return hooks


@pytest.fixture
def mcp_hooks(mock_agent, mock_original_hooks):
    """Create MCP agent hooks with mocked agent and original hooks."""
    return MCPAgentHooks(agent=mock_agent, original_hooks=mock_original_hooks)


@pytest.mark.asyncio
async def test_mcp_hooks_initialization(mock_agent, mock_original_hooks):
    """Test initialization of MCPAgentHooks."""
    hooks = MCPAgentHooks(agent=mock_agent, original_hooks=mock_original_hooks)

    assert hooks.agent is mock_agent
    assert hooks.original_hooks is mock_original_hooks


@pytest.mark.asyncio
async def test_on_start_loads_mcp_tools(mcp_hooks, run_context_wrapper):
    """Test that on_start loads MCP tools."""
    # Call on_start
    await mcp_hooks.on_start(run_context_wrapper, mcp_hooks.agent)

    # Verify load_mcp_tools was called
    mcp_hooks.agent.load_mcp_tools.assert_called_once_with(run_context_wrapper)

    # Verify original hook was called
    mcp_hooks.original_hooks.on_start.assert_called_once_with(run_context_wrapper, mcp_hooks.agent)


@pytest.mark.asyncio
async def test_on_end_calls_original_hook(mcp_hooks, run_context_wrapper):
    """Test that on_end calls the original hook."""
    # Call on_end
    output = {"final_response": "test output"}
    await mcp_hooks.on_end(run_context_wrapper, mcp_hooks.agent, output)

    # Verify original hook was called
    mcp_hooks.original_hooks.on_end.assert_called_once_with(
        run_context_wrapper, mcp_hooks.agent, output
    )


@pytest.mark.asyncio
async def test_on_handoff_calls_original_hook(mcp_hooks, run_context_wrapper):
    """Test that on_handoff calls the original hook."""
    # Call on_handoff
    source_agent = MagicMock()
    await mcp_hooks.on_handoff(run_context_wrapper, mcp_hooks.agent, source_agent)

    # Verify original hook was called
    mcp_hooks.original_hooks.on_handoff.assert_called_once_with(
        run_context_wrapper, mcp_hooks.agent, source_agent
    )


@pytest.mark.asyncio
async def test_on_tool_start_calls_original_hook(mcp_hooks, run_context_wrapper):
    """Test that on_tool_start calls the original hook."""
    # Call on_tool_start
    tool = MagicMock()
    await mcp_hooks.on_tool_start(run_context_wrapper, mcp_hooks.agent, tool)

    # Verify original hook was called
    mcp_hooks.original_hooks.on_tool_start.assert_called_once_with(
        run_context_wrapper, mcp_hooks.agent, tool
    )


@pytest.mark.asyncio
async def test_on_tool_end_calls_original_hook(mcp_hooks, run_context_wrapper):
    """Test that on_tool_end calls the original hook."""
    # Call on_tool_end
    tool = MagicMock()
    result = "test result"
    await mcp_hooks.on_tool_end(run_context_wrapper, mcp_hooks.agent, tool, result)

    # Verify original hook was called
    mcp_hooks.original_hooks.on_tool_end.assert_called_once_with(
        run_context_wrapper, mcp_hooks.agent, tool, result
    )
