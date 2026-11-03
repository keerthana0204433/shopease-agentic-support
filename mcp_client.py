
import asyncio
import sys
from pathlib import Path

from langchain_mcp_adapters.client import MultiServerMCPClient

# Project directory
PROJECT_DIR = Path(__file__).resolve().parent


async def main():

    # Configure connection to our MCP server
    client = MultiServerMCPClient(
        {
            "shopease": {
                "transport": "stdio",
                "command": sys.executable,
                "args": [str(PROJECT_DIR / "mcp_server.py")]
            }
        }
    )

    # Discover tools exposed by the MCP server
    tools = await client.get_tools()

    print("\nConnected to ShopEase MCP server!")
    print("\nAvailable tools:")

    for tool in tools:
        print(f"\nTool name: {tool.name}")
        print(f"Description: {tool.description}")


if __name__ == "__main__":
    asyncio.run(main())
