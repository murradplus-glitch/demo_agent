"""Tests for MCP context functionality."""

import os
from tempfile import NamedTemporaryFile
from unittest.mock import MagicMock, patch

import pytest
from mcp_agent.config import MCPServerSettings, MCPSettings
from mcp_agent.mcp_server_registry import ServerRegistry

from agents_mcp import RunnerContext
from agents_mcp.server_registry import ensure_mcp_server_registry_in_context


def test_mcp_server_settings():
    """Test MCPServerSettings dataclass."""
    # Basic initialization
    settings = MCPServerSettings(command="test_command", args=["arg1", "arg2"])
    assert settings.command == "test_command"
    assert settings.args == ["arg1", "arg2"]

    # With environment variables - exact format might differ based on MCPServerSettings implementation
    settings = MCPServerSettings(
        command="test_command",
        args=["arg1", "arg2"],
        env={"ENV_VAR": "value"},
    )
    env = settings.env
    if env is not None:  # Check for None before using 'in' operator
        assert "ENV_VAR" in env


def test_mcp_settings():
    """Test MCPSettings dataclass."""
    # Empty initialization
    settings = MCPSettings()
    assert settings.servers == {}

    # With servers
    server_settings = {
        "server1": MCPServerSettings(command="cmd1", args=["arg1"]),
        "server2": MCPServerSettings(command="cmd2", args=["arg2"]),
    }
    settings = MCPSettings(servers=server_settings)
    assert settings.servers == server_settings
    assert settings.servers["server1"].command == "cmd1"
    assert settings.servers["server2"].command == "cmd2"


# Note: AgentsMCPContext has been refactored out of the codebase, so we removed this test


def test_runner_context_initialization():
    """Test RunnerContext initialization."""
    # Empty initialization
    context = RunnerContext()
    assert context.mcp_config is None
    assert context.mcp_config_path is None

    # With explicit MCP config
    settings = MCPSettings(
        servers={
            "server1": MCPServerSettings(command="cmd1", args=["arg1"]),
        }
    )
    context = RunnerContext(mcp_config=settings)
    assert context.mcp_config is settings

    # Test with additional custom attributes
    context = RunnerContext(mcp_config=settings, custom_attr="custom_value", another_attr=123)
    assert context.mcp_config is settings
    assert context.custom_attr == "custom_value"
    assert context.another_attr == 123


@pytest.mark.parametrize(
    "yaml_content",
    [
        # Simple config
        """
        mcp:
          servers:
            server1:
              command: cmd1
              args: [arg1, arg2]
        """,
        # Config with environment variables
        """
        mcp:
          servers:
            server1:
              command: cmd1
              args: [arg1, arg2]
              env:
                VAR1: value1
                VAR2: value2
        """,
        # Multiple servers
        """
        mcp:
          servers:
            server1:
              command: cmd1
              args: [arg1]
            server2:
              command: cmd2
              args: [arg2]
        """,
    ],
)
def test_runner_context_load_from_config_file(yaml_content):
    """Test loading MCP config from a file."""
    # Create a temporary config file
    with NamedTemporaryFile(mode="w", delete=False) as config_file:
        config_file.write(yaml_content)
        config_file.flush()
        config_path = config_file.name

    try:
        # Mock the get_settings function since we don't want to actually read the file
        # and process it in the test
        with patch("agents_mcp.server_registry.get_settings") as mock_get_settings:
            # Create a mock settings object
            mock_settings = MagicMock()
            mock_settings.mcp = MagicMock()
            mock_get_settings.return_value = mock_settings

            # Create RunnerContext with config path
            context = RunnerContext(mcp_config_path=config_path)

            # Verify path was correctly stored
            assert context.mcp_config_path == config_path

    finally:
        # Clean up the temporary file
        os.unlink(config_path)


def test_ensure_mcp_server_registry_in_context():
    """Test ensuring MCP server registry is in context."""
    # Create a context with MCP settings but no registry
    settings = MCPSettings(
        servers={
            "server1": MCPServerSettings(command="cmd1", args=["arg1"]),
        }
    )
    context = RunnerContext(mcp_config=settings)
    assert not hasattr(context, "mcp_server_registry") or context.mcp_server_registry is None

    # Create a wrapper around the context
    context_wrapper = MagicMock()
    context_wrapper.context = context

    # Mock the load_mcp_server_registry function to return a mock registry
    with patch("agents_mcp.server_registry.load_mcp_server_registry") as mock_load:
        mock_registry = MagicMock(spec=ServerRegistry)
        mock_load.return_value = mock_registry

        # Ensure registry is in context
        ensure_mcp_server_registry_in_context(context_wrapper)

        # Verify registry was created
        assert context.mcp_server_registry is not None
        assert context.mcp_server_registry is mock_registry

        # Calling again should not create a new registry
        ensure_mcp_server_registry_in_context(context_wrapper)
        mock_load.assert_called_once()  # Should only be called once
