"""Tests for MCP aggregator functionality."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from mcp_agent.context import Context
from mcp_agent.mcp.mcp_aggregator import MCPAggregator
from mcp_agent.mcp_server_registry import ServerRegistry

from agents_mcp.aggregator import create_mcp_aggregator, initialize_mcp_aggregator


@pytest.mark.asyncio
async def test_create_mcp_aggregator_with_empty_servers(run_context_wrapper):
    """Test create_mcp_aggregator with empty server list."""
    # Test with empty server list
    with pytest.raises(RuntimeError) as excinfo:
        create_mcp_aggregator(
            run_context=run_context_wrapper,
            name="test_agent",
            servers=[],
        )
    assert "No MCP servers specified" in str(excinfo.value)


@pytest.mark.asyncio
async def test_create_mcp_aggregator_with_provided_registry(run_context_wrapper):
    """Test create_mcp_aggregator with explicitly provided server registry."""
    # Create a mock server registry
    mock_registry = MagicMock(spec=ServerRegistry)

    # Create a mock MCPAggregator constructor
    mock_aggregator = MagicMock(spec=MCPAggregator)

    # Call the function with explicit registry
    with patch("agents_mcp.aggregator.MCPAggregator") as mock_aggregator_class:
        mock_aggregator_class.return_value = mock_aggregator

        result = create_mcp_aggregator(
            run_context=run_context_wrapper,
            name="test_agent",
            servers=["fetch"],
            server_registry=mock_registry,
        )

        # Verify the aggregator was created with correct parameters
        mock_aggregator_class.assert_called_once()
        kwargs = mock_aggregator_class.call_args.kwargs
        assert kwargs["server_names"] == ["fetch"]
        assert kwargs["name"] == "test_agent"
        assert isinstance(kwargs["context"], Context)
        assert kwargs["context"].server_registry is mock_registry
        assert result is mock_aggregator


@pytest.mark.asyncio
async def test_create_mcp_aggregator_from_context(run_context_wrapper):
    """Test create_mcp_aggregator retrieving registry from context."""
    # Set up a mock server registry in the run context
    mock_registry = MagicMock(spec=ServerRegistry)
    run_context_wrapper.context.mcp_server_registry = mock_registry

    # Call the function with registry from context
    with patch("agents_mcp.aggregator.MCPAggregator") as mock_aggregator_class:
        mock_aggregator = MagicMock(spec=MCPAggregator)
        mock_aggregator_class.return_value = mock_aggregator

        result = create_mcp_aggregator(
            run_context=run_context_wrapper,
            name="test_agent",
            servers=["fetch"],
        )

        # Verify the aggregator was created with correct parameters
        mock_aggregator_class.assert_called_once()
        kwargs = mock_aggregator_class.call_args.kwargs
        assert kwargs["server_names"] == ["fetch"]
        assert kwargs["name"] == "test_agent"
        assert isinstance(kwargs["context"], Context)
        assert kwargs["context"].server_registry is mock_registry
        assert result is mock_aggregator


@pytest.mark.asyncio
async def test_create_mcp_aggregator_no_registry(run_context_wrapper):
    """Test create_mcp_aggregator with no registry available."""
    # Ensure no server registry in context
    if hasattr(run_context_wrapper.context, "mcp_server_registry"):
        delattr(run_context_wrapper.context, "mcp_server_registry")

    # Test with no registry available
    with pytest.raises(RuntimeError) as excinfo:
        create_mcp_aggregator(
            run_context=run_context_wrapper,
            name="test_agent",
            servers=["fetch"],
        )
    assert "No server registry found in run context" in str(excinfo.value)


@pytest.mark.asyncio
async def test_initialize_mcp_aggregator_success(run_context_wrapper):
    """Test successful initialization of an MCP aggregator."""
    # Create a mock aggregator
    mock_aggregator = AsyncMock(spec=MCPAggregator)

    # Mock the create_mcp_aggregator function
    with patch("agents_mcp.aggregator.create_mcp_aggregator", return_value=mock_aggregator):
        # Call initialize_mcp_aggregator
        result = await initialize_mcp_aggregator(
            run_context=run_context_wrapper,
            name="test_agent",
            servers=["fetch"],
        )

        # Verify aggregator was initialized
        mock_aggregator.__aenter__.assert_called_once()
        assert result is mock_aggregator


@pytest.mark.asyncio
async def test_initialize_mcp_aggregator_error(run_context_wrapper):
    """Test error handling during MCP aggregator initialization."""
    # Create a mock aggregator that raises an exception on __aenter__
    mock_aggregator = AsyncMock(spec=MCPAggregator)
    mock_aggregator.__aenter__.side_effect = Exception("Connection error")

    # Mock the create_mcp_aggregator function
    with patch("agents_mcp.aggregator.create_mcp_aggregator", return_value=mock_aggregator):
        # Mock the logger to avoid real logging
        with patch("agents_mcp.aggregator.logger") as mock_logger:
            # Call initialize_mcp_aggregator and expect an exception
            with pytest.raises(Exception) as excinfo:
                await initialize_mcp_aggregator(
                    run_context=run_context_wrapper,
                    name="test_agent",
                    servers=["fetch"],
                )

            # Verify error handling
            assert "Connection error" in str(excinfo.value)
            mock_aggregator.__aexit__.assert_called_once()
            mock_logger.error.assert_called_once()
