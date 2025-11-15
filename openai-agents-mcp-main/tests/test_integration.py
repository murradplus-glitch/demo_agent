"""Integration tests for MCP agent functionality."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from agents import function_tool
from agents.run_context import RunContextWrapper
from mcp.types import TextContent
from mcp_agent.config import MCPServerSettings, MCPSettings

from agents_mcp import Agent, RunnerContext


# Define our own MCPToolError for testing
class MCPToolError(RuntimeError):
    """Error raised by MCP tools."""

    def __init__(self, message, error_type="Unknown"):
        super().__init__(message)
        self.error_type = error_type


@pytest.fixture
def mock_mcp_aggregator_factory():
    """Create a factory for mock MCP aggregators."""

    def _create_mock_aggregator(tool_responses=None, error_tools=None):
        """Create a mock aggregator with specified tool responses.

        Args:
            tool_responses: Dict mapping tool names to their success responses
            error_tools: List of tool names that should return errors
        """
        if tool_responses is None:
            tool_responses = {}
        if error_tools is None:
            error_tools = []

        mock_aggregator = AsyncMock()
        mock_aggregator.initialized = True
        mock_aggregator.agent_name = "test_agent"

        # Create a mock ListToolsResult with specified tools
        tools_result = MagicMock()
        tools = []

        # Add fetch tool by default
        fetch_tool = MagicMock()
        fetch_tool.name = "fetch"
        fetch_tool.description = (
            "Fetches a URL from the internet and optionally extracts its contents as markdown."
        )
        fetch_tool.inputSchema = {
            "description": "Parameters for fetching a URL.",
            "properties": {
                "url": {"description": "URL to fetch", "title": "Url", "type": "string"},
                "max_length": {
                    "description": "Maximum number of characters to return.",
                    "title": "Max Length",
                    "type": "integer",
                },
                "start_index": {
                    "description": "On return output starting at this character index, useful if a previous fetch was truncated and more context is required.",
                    "title": "Start Index",
                    "type": "integer",
                },
                "raw": {
                    "description": "Get the actual HTML content if the requested page, without simplification.",
                    "title": "Raw",
                    "type": "boolean",
                },
            },
            "required": ["url", "max_length", "start_index", "raw"],
            "title": "Fetch",
            "type": "object",
        }
        tools.append(fetch_tool)

        # Add read_file tool by default
        file_tool = MagicMock()
        file_tool.name = "read_file"
        file_tool.description = "Read the complete contents of a file from the file system."
        file_tool.inputSchema = {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "The path to the file"}},
            "required": ["path"],
        }
        tools.append(file_tool)

        # Add list_directory tool
        dir_tool = MagicMock()
        dir_tool.name = "list_directory"
        dir_tool.description = (
            "Get a detailed listing of all files and directories in a specified path."
        )
        dir_tool.inputSchema = {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "The path to the directory"}},
            "required": ["path"],
        }
        tools.append(dir_tool)

        tools_result.tools = tools
        mock_aggregator.list_tools.return_value = tools_result

        # Configure call_tool to return specified responses or errors
        async def mock_call_tool(name, arguments):
            result = MagicMock()

            if name in error_tools:
                result.isError = True
                error_text = f"Error executing tool {name}: access denied"
                result.content = TextContent(type="text", text=error_text)
                result.errorType = "AccessDenied"
            else:
                # Format the response to match real MCP tool responses
                if name == "fetch":
                    url = arguments.get("url", "https://example.com")
                    response_text = tool_responses.get(
                        name, f"Contents of {url}:\nThis is mock content from {url}"
                    )
                elif name == "read_file":
                    path = arguments.get("path", "test.txt")
                    response_text = tool_responses.get(
                        name, f"Contents of {path}: Mock file content"
                    )
                elif name == "list_directory":
                    path = arguments.get("path", ".")
                    response_text = tool_responses.get(
                        name, "[DIR] test_dir\n[FILE] test1.txt\n[FILE] test2.txt"
                    )
                else:
                    response_text = tool_responses.get(name, f"Mock response for {name}")

                result.isError = False
                result.content = TextContent(type="text", text=response_text)

            return result

        mock_aggregator.call_tool.side_effect = mock_call_tool
        mock_aggregator.__aexit__.return_value = None

        return mock_aggregator

    return _create_mock_aggregator


@pytest.mark.asyncio
async def test_agent_with_mixed_tools(mock_mcp_aggregator_factory):
    """
    Test a complete agent with both local tools and MCP tools.
    This test simulates the real hello_world_mcp.py example.
    """
    # Create tool responses for the mock aggregator
    tool_responses = {
        "fetch": "Contents of https://example.com:\nThis is the example.com website content. The Example Domain website is intended for illustrative examples in documents. You may use this domain in literature without prior coordination or asking for permission.",
        "read_file": "Contents of test.txt: Hello, world! This is a test file used for demonstration purposes.",
        "list_directory": "[DIR] examples\n[DIR] src\n[DIR] tests\n[FILE] README.md\n[FILE] pyproject.toml",
    }

    # Create a mock aggregator with the expected responses
    mock_aggregator = mock_mcp_aggregator_factory(tool_responses)

    # Define a local tool
    @function_tool
    def get_current_weather(location: str) -> str:
        """
        Get the current weather for a location.

        Args:
            location: The city and state, e.g. "San Francisco, CA"

        Returns:
            The current weather for the requested location
        """
        return f"The weather in {location} is currently sunny and 72 degrees Fahrenheit."

    # Create MCP settings similar to the mcp_agent.config.yaml example
    # mcp_settings = MCPSettings(
    #     servers={
    #         "fetch": MCPServerSettings(
    #             command="mock_fetch",
    #             args=["--arg1", "--arg2"],
    #         ),
    #         "filesystem": MCPServerSettings(
    #             command="mock_filesystem",
    #             args=["--path", "."],
    #         ),
    #     }
    # )

    # # Create a runner context with our settings
    # runner_context = RunnerContext(mcp_config=mcp_settings)

    # Create an agent with both local and MCP tools
    agent = Agent(
        name="MCP Assistant",
        instructions="You are a helpful assistant with access to both local and MCP tools.",
        tools=[get_current_weather],  # Local tool
        mcp_servers=["fetch", "filesystem"],  # MCP tools,
    )

    # Mock the MCP initialization to avoid actual server process spawning
    with patch("agents_mcp.agent.initialize_mcp_aggregator", return_value=mock_aggregator):
        # Mock the loading of MCP tools to track calls
        agent.load_mcp_tools = AsyncMock()

        # Setup expected MCP tools list for when load_mcp_tools is called
        fetch_tool = MagicMock()
        fetch_tool.name = "fetch-fetch"
        fetch_tool.description = (
            "Fetches a URL from the internet and optionally extracts its contents as markdown."
        )

        read_file_tool = MagicMock()
        read_file_tool.name = "filesystem-read_file"
        read_file_tool.description = "Read the complete contents of a file from the file system."

        list_dir_tool = MagicMock()
        list_dir_tool.name = "filesystem-list_directory"
        list_dir_tool.description = (
            "Get a detailed listing of all files and directories in a specified path."
        )

        agent._mcp_tools = [fetch_tool, read_file_tool, list_dir_tool]

        # Verify MCP agent hooks
        assert hasattr(agent.hooks, "on_start")

        # Verify that MCP servers are properly configured
        assert agent.mcp_servers == ["fetch", "filesystem"]

        # Verify that both local and MCP tools are available
        all_tools = agent._openai_tools + agent._mcp_tools
        assert len(all_tools) == 4  # 1 local + 3 MCP tools

        # Verify the names of all available tools
        tool_names = [tool.name for tool in all_tools]
        assert "get_current_weather" in tool_names
        assert "fetch-fetch" in tool_names
        assert "filesystem-read_file" in tool_names
        assert "filesystem-list_directory" in tool_names


@pytest.mark.asyncio
async def test_agent_as_tool_integration(mock_mcp_aggregator_factory):
    """Test using an MCP agent as a tool for another agent."""
    # Create tool responses
    tool_responses = {
        "fetch": "Fetched content from example.com: API documentation",
    }

    # Create mock aggregator
    mock_aggregator = mock_mcp_aggregator_factory(tool_responses)

    # Create MCP settings
    mcp_settings = MCPSettings(
        servers={
            "fetch": MCPServerSettings(command="mock", args=["fetch"]),
        }
    )

    # Create context
    runner_context = RunnerContext(mcp_config=mcp_settings)
    context_wrapper = RunContextWrapper(context=runner_context)

    # Create MCP agent
    mcp_agent = Agent(
        name="WebAgent",
        instructions="You fetch web content for the user.",
        mcp_servers=["fetch"],
    )

    # Convert to tool
    web_tool = mcp_agent.as_tool(
        tool_name="web_search",
        tool_description="Search the web for information",
    )

    # Mock agent's load_mcp_tools method
    original_load_tools = mcp_agent.load_mcp_tools
    mcp_agent.load_mcp_tools = AsyncMock(wraps=original_load_tools)

    # Mock initialize_mcp_aggregator
    with patch("agents_mcp.agent.initialize_mcp_aggregator", return_value=mock_aggregator):
        # Mock ItemHelpers.text_message_outputs to return our expected output
        with patch("agents.items.ItemHelpers.text_message_outputs") as mock_extract:
            mock_extract.return_value = tool_responses["fetch"]

            # Mock Runner.run to avoid LLM calls
            with patch("agents.run.Runner.run") as mock_runner_run:
                # Setup mock to return expected result
                mock_result = MagicMock()
                mock_result.new_items = [MagicMock()]
                mock_runner_run.return_value = mock_result

                # Call the tool using on_invoke_tool with JSON-formatted arguments
                result = await web_tool.on_invoke_tool(
                    context_wrapper, '{"input": "Find information about APIs"}'
                )

                # Verify MCP tools were loaded
                mcp_agent.load_mcp_tools.assert_called_once()

                # Verify runner was called with expected arguments
                mock_runner_run.assert_called_once()
                args, kwargs = mock_runner_run.call_args
                assert kwargs["starting_agent"] is mcp_agent
                assert kwargs["input"] == "Find information about APIs"
                assert kwargs["context"] is runner_context

                # Verify we got the expected result
                assert result == tool_responses["fetch"]


@pytest.mark.asyncio
async def test_agent_with_tool_error_handling(mock_mcp_aggregator_factory):
    """Test that MCP tool errors are properly handled."""
    # Create a mock aggregator where the fetch tool returns errors
    mock_aggregator = mock_mcp_aggregator_factory(
        tool_responses={"read_file": "File contents"}, error_tools=["fetch"]
    )

    # Create MCP settings
    # mcp_settings = MCPSettings(
    #     servers={
    #         "fetch": MCPServerSettings(command="mock", args=["fetch"]),
    #         "filesystem": MCPServerSettings(command="mock", args=["filesystem"]),
    #     }
    # )

    # # Create context
    # runner_context = RunnerContext(mcp_config=mcp_settings)

    # Create agent with MCP tools
    agent = Agent(
        name="ErrorTestAgent",
        instructions="You are a test agent for error handling.",
        mcp_servers=["fetch", "filesystem"],
    )

    # Mock the MCP initialization
    with patch("agents_mcp.agent.initialize_mcp_aggregator", return_value=mock_aggregator):
        # Set up MCP tools
        fetch_tool = MagicMock()
        fetch_tool.name = "fetch-fetch"
        fetch_tool.description = "Fetch content from a URL"
        fetch_tool.on_invoke_tool = AsyncMock()
        fetch_tool.on_invoke_tool.side_effect = MCPToolError(
            "Error executing tool fetch: access denied", "AccessDenied"
        )

        fs_tool = MagicMock()
        fs_tool.name = "filesystem-read_file"
        fs_tool.description = "Read a file from the filesystem"
        fs_tool.on_invoke_tool = AsyncMock(return_value="File contents")

        agent._mcp_tools = [fetch_tool, fs_tool]

        # Test that calling the fetch tool raises MCPToolError
        with pytest.raises(MCPToolError) as excinfo:
            await fetch_tool.on_invoke_tool(MagicMock(), '{"url": "https://example.com"}')

        # Verify error details
        assert "access denied" in str(excinfo.value)
        assert excinfo.value.error_type == "AccessDenied"

        # Test that the filesystem tool works fine
        result = await fs_tool.on_invoke_tool(MagicMock(), '{"path": "test.txt"}')
        assert result == "File contents"


@pytest.mark.asyncio
async def test_mcp_server_initialization_failure():
    """Test handling of MCP server initialization failure."""
    # Create MCP settings with a non-existent server
    mcp_settings = MCPSettings(
        servers={
            "invalid": MCPServerSettings(
                command="non_existent_command",
                args=[],
            ),
        }
    )

    # Create context with invalid settings
    context = RunnerContext(mcp_config=mcp_settings)
    runner_context = RunContextWrapper(context=context)

    # Create agent with invalid server
    agent = Agent(
        name="FailureTestAgent",
        instructions="You are a test agent for error handling.",
        mcp_servers=["invalid"],
    )

    # Initialize _mcp_tools as empty list
    agent._mcp_tools = []

    # Mock initialize_mcp_aggregator to raise an exception
    with patch("agents_mcp.agent.initialize_mcp_aggregator") as mock_init:
        mock_init.side_effect = Exception("Failed to start MCP server")

        # Call load_mcp_tools directly with the expected exception
        with pytest.raises(Exception) as excinfo:
            await agent.load_mcp_tools(runner_context)

        # Verify the exception details
        assert str(excinfo.value) == "Failed to start MCP server"

        # Verify no tools were loaded (tools list should remain empty)
        assert agent._mcp_tools == []
