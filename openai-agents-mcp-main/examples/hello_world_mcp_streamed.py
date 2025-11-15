"""
Example demonstrating how to use an agent with MCP servers.

This example shows how to:
1. Load MCP servers from the config file automatically
2. Create an agent that specifies which MCP servers to use
3. Run the agent to dynamically load and use tools from the specified MCP servers

To use this example:
1. Create an mcp_agent.config.yaml file in this directory or a parent directory
2. Configure your MCP servers in that file
3. Run this example
"""

import asyncio
from typing import TYPE_CHECKING

from openai.types.responses import ResponseTextDeltaEvent

if TYPE_CHECKING:
    pass

from agents import Runner, function_tool

from agents_mcp import Agent, RunnerContext

# enable_verbose_stdout_logging()


# Define a simple local tool to demonstrate combining local and MCP tools
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


async def main():
    # Specify a custom config path if needed, or set to None to use default discovery
    mcp_config_path = None  # Set to a file path if needed

    # Alternatively, define MCP config programmatically
    mcp_config = None
    # mcp_config = MCPSettings(
    #     servers={
    #         "fetch": MCPServerSettings(
    #             command="uvx",
    #             args=["mcp-server-fetch"],
    #         ),
    #         "filesystem": MCPServerSettings(
    #             command="npx",
    #             args=["-y", "@modelcontextprotocol/server-filesystem", "."],
    #         ),
    #     }
    # ),

    # Create a context object containing MCP settings
    context = RunnerContext(mcp_config_path=mcp_config_path, mcp_config=mcp_config)

    # Create an agent with specific MCP servers you want to use
    # These must be defined in your mcp_agent.config.yaml file
    agent: Agent = Agent(
        name="MCP Assistant",
        instructions="""You are a helpful assistant with access to both local tools
            and tools from MCP servers. Use these tools to help the user.""",
        tools=[get_current_weather],  # Local and OpenAI tools
        mcp_servers=[
            "fetch",
            "filesystem",
        ],  # Specify which MCP servers to use (must be defined in your MCP config)
        mcp_server_registry=None,  # Specify a custom MCP server registry per-agent if needed
    )

    # Run the agent - tools from the specified MCP servers will be automatically loaded
    result = Runner.run_streamed(
        agent,
        input="Print the first paragraph of https://openai.github.io/openai-agents-python/",
        context=context,
    )
    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            print(event.data.delta, end="", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
