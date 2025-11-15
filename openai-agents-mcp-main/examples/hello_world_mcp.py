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

if TYPE_CHECKING:
    pass

from agents import Runner, enable_verbose_stdout_logging, function_tool

from agents_mcp import Agent, RunnerContext

enable_verbose_stdout_logging()


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
        tools=[get_current_weather],  # Local tools
        mcp_servers=["fetch", "filesystem"],  # Specify which MCP servers to use
        # These must be defined in your config
    )

    # Run the agent - existing openai tools and local function tools will still work
    result = await Runner.run(
        starting_agent=agent,
        input="What's the weather in Miami?",
        context=context,
    )

    # Print the agent's response
    print("\nInput: What's the weather like in Miami?\nAgent response:")
    print(result.final_output)

    # Tools from the specified MCP servers will be automatically loaded. In this catch fetch will be used
    result = await Runner.run(
        starting_agent=agent,
        input="Print the first paragraph of https://openai.github.io/openai-agents-python/",
        context=context,
    )

    # Print the agent's response
    print(
        "\nInput: Print the first paragraph of https://openai.github.io/openai-agents-python\nAgent response:"
    )
    print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
