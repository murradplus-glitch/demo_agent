"""Tests for the MCP Agent class."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from agents.run_context import RunContextWrapper

from agents_mcp.agent import Agent


@pytest.mark.asyncio
async def test_agent_initialization():
    """Test MCP Agent initialization and inheritance."""
    # Create a basic agent
    agent = Agent(
        name="TestAgent",
        instructions="Test instructions",
        mcp_servers=["fetch", "filesystem"],
    )

    # Verify agent properties
    assert agent.name == "TestAgent"
    assert agent.instructions == "Test instructions"
    assert agent.mcp_servers == ["fetch", "filesystem"]
    assert agent._mcp_initialized is False
    assert agent._mcp_aggregator is None
    assert agent._mcp_tools == []

    # Verify hooks wrapper was created
    assert agent.hooks is not None
    assert hasattr(agent.hooks, "original_hooks")
    assert agent.hooks.agent is agent


@pytest.mark.asyncio
async def test_agent_load_mcp_tools(
    mock_mcp_aggregator, mock_mcp_tools_result, run_context_wrapper
):
    """Test loading MCP tools into an agent."""
    # Create agent
    agent = Agent(
        name="TestAgent",
        instructions="Test instructions",
        mcp_servers=["fetch", "filesystem"],
    )

    # Mock the initialize_mcp_aggregator function to return our mock aggregator
    with patch("agents_mcp.agent.initialize_mcp_aggregator", return_value=mock_mcp_aggregator):
        # Mock the mcp_list_tools function
        with patch("agents_mcp.agent.mcp_list_tools") as mock_list_tools:
            # Setup the mock to return some tools
            mock_tool1 = MagicMock()
            mock_tool1.name = "fetch_tool"
            mock_tool2 = MagicMock()
            mock_tool2.name = "fs_tool"
            mock_list_tools.return_value = [mock_tool1, mock_tool2]

            # Call load_mcp_tools
            await agent.load_mcp_tools(run_context_wrapper)

            # Verify initialize_mcp_aggregator was called with correct args
            from agents_mcp.agent import initialize_mcp_aggregator

            assert initialize_mcp_aggregator.called

            # Verify mcp_list_tools was called with aggregator
            mock_list_tools.assert_called_once_with(mock_mcp_aggregator)

            # Verify tools were added to agent
            assert len(agent._mcp_tools) == 2
            assert agent._mcp_tools[0] == mock_tool1
            assert agent._mcp_tools[1] == mock_tool2

            # Verify agent state
            assert agent._mcp_initialized is True
            assert agent._mcp_aggregator is mock_mcp_aggregator

            # Verify tools list includes both MCP tools and original tools
            assert len(agent.tools) == 2
            assert agent.tools[0] == mock_tool1
            assert agent.tools[1] == mock_tool2


@pytest.mark.asyncio
async def test_agent_load_mcp_tools_no_servers(run_context_wrapper):
    """Test that no tools are loaded when no MCP servers are specified."""
    # Create agent with no MCP servers
    agent = Agent(
        name="TestAgent",
        instructions="Test instructions",
        mcp_servers=[],  # Empty list
    )

    # Mock functions to verify they're not called
    with patch("agents_mcp.agent.initialize_mcp_aggregator") as mock_init_aggregator:
        with patch("agents_mcp.agent.mcp_list_tools") as mock_list_tools:
            # Call load_mcp_tools
            await agent.load_mcp_tools(run_context_wrapper)

            # Verify functions were not called
            mock_init_aggregator.assert_not_called()
            mock_list_tools.assert_not_called()

            # Verify no tools were added
            assert agent._mcp_tools == []
            assert agent._mcp_initialized is False


@pytest.mark.asyncio
async def test_agent_load_mcp_tools_already_initialized(mock_mcp_aggregator, run_context_wrapper):
    """Test that tools are not reloaded if already initialized."""
    # Create agent
    agent = Agent(
        name="TestAgent",
        instructions="Test instructions",
        mcp_servers=["fetch", "filesystem"],
    )

    # Set agent as already initialized
    agent._mcp_initialized = True
    agent._mcp_aggregator = mock_mcp_aggregator

    # Mock functions to verify they're not called
    with patch("agents_mcp.agent.initialize_mcp_aggregator") as mock_init_aggregator:
        with patch("agents_mcp.agent.mcp_list_tools") as mock_list_tools:
            # Call load_mcp_tools
            await agent.load_mcp_tools(run_context_wrapper)

            # Verify functions were not called
            mock_init_aggregator.assert_not_called()
            mock_list_tools.assert_not_called()


@pytest.mark.asyncio
async def test_agent_load_mcp_tools_force_reload(
    mock_mcp_aggregator, mock_mcp_tools_result, run_context_wrapper
):
    """Test forcing reload of MCP tools even if already initialized."""
    # Create agent
    agent = Agent(
        name="TestAgent",
        instructions="Test instructions",
        mcp_servers=["fetch", "filesystem"],
    )

    # Set agent as already initialized
    agent._mcp_initialized = True
    agent._mcp_aggregator = mock_mcp_aggregator

    # Mock the initialize_mcp_aggregator function
    with patch("agents_mcp.agent.initialize_mcp_aggregator", return_value=mock_mcp_aggregator):
        # Mock the mcp_list_tools function
        with patch("agents_mcp.agent.mcp_list_tools") as mock_list_tools:
            # Setup mock to return some tools
            mock_tool = MagicMock()
            mock_tool.name = "new_tool"
            mock_list_tools.return_value = [mock_tool]

            # Call load_mcp_tools with force=True
            await agent.load_mcp_tools(run_context_wrapper, force=True)

            # Verify initialize_mcp_aggregator was called
            from agents_mcp.agent import initialize_mcp_aggregator

            assert initialize_mcp_aggregator.called

            # Verify tools were reloaded
            mock_list_tools.assert_called_once_with(mock_mcp_aggregator)
            assert len(agent._mcp_tools) == 1
            assert agent._mcp_tools[0] == mock_tool


@pytest.mark.asyncio
async def test_agent_cleanup_resources():
    """Test cleanup of MCP resources."""
    # Create agent
    agent = Agent(
        name="TestAgent",
        instructions="Test instructions",
        mcp_servers=["fetch", "filesystem"],
    )

    # Set up agent with mock aggregator
    mock_aggregator = AsyncMock()
    agent._mcp_aggregator = mock_aggregator
    agent._mcp_initialized = True
    agent._mcp_tools = [MagicMock(), MagicMock()]

    # Call cleanup_resources
    await agent.cleanup_resources()

    # Verify aggregator's __aexit__ was called
    mock_aggregator.__aexit__.assert_called_once_with(None, None, None)

    # Verify agent state was reset
    assert agent._mcp_aggregator is None
    assert agent._mcp_initialized is False
    assert agent._mcp_tools == []


@pytest.mark.asyncio
async def test_agent_as_tool():
    """Test converting an MCP agent to a tool."""
    # Create agent
    agent = Agent(
        name="TestAgent",
        instructions="Test instructions",
        mcp_servers=["fetch", "filesystem"],
    )

    # Convert to tool
    tool = agent.as_tool(
        tool_name="test_tool",
        tool_description="A test tool",
    )

    # Verify tool properties
    assert tool.name == "test_tool"
    assert tool.description == "A test tool"

    # Verify on_invoke_tool is defined (FunctionTool is not directly callable)
    assert hasattr(tool, "on_invoke_tool")
    assert callable(tool.on_invoke_tool)


@pytest.mark.asyncio
async def test_agent_as_tool_with_mcp_servers():
    """Test that agent tool loads MCP tools when invoked."""
    # Create agent with MCP servers
    agent = Agent(
        name="TestAgent",
        instructions="Test instructions",
        mcp_servers=["fetch", "filesystem"],
    )

    # Convert to tool
    tool = agent.as_tool(
        tool_name="test_tool",
        tool_description="A test tool",
    )

    # Create mock context
    mock_context = RunContextWrapper(context=MagicMock())

    # Mock the agent's load_mcp_tools method
    agent.load_mcp_tools = AsyncMock()

    # Mock the Runner.run method
    with patch("agents.run.Runner.run") as mock_run:
        # Setup Runner.run to return a mock result
        mock_result = MagicMock()
        mock_result.new_items = []
        mock_run.return_value = mock_result

        # Call the tool's on_invoke_tool method
        # The arguments would normally be JSON, but we can mock it
        await tool.on_invoke_tool(mock_context, '{"input": "test input"}')

        # Verify load_mcp_tools was called
        agent.load_mcp_tools.assert_called_once_with(mock_context)

        # Verify Runner.run was called
        mock_run.assert_called_once()
