
import os
import sys
import asyncio
from pathlib import Path

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()

PROJECT_DIR = Path(__file__).resolve().parent


async def main():

    # 1. Connect to our MCP server
    client = MultiServerMCPClient(
        {
            "shopease": {
                "transport": "stdio",
                "command": sys.executable,
                "args": [str(PROJECT_DIR / "mcp_server.py")]
            }
        }
    )

    # 2. Discover MCP tools
    tools = await client.get_tools()

    print("Available MCP tools:")
    for tool in tools:
        print(tool.name)

    # 3. Create our LLM
    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0
    )

    # 4. Give the LLM access to the MCP tools
    llm_with_tools = llm.bind_tools(tools)

    # 5. Ask a question that requires order lookup
    response = llm_with_tools.invoke(
        """
        I want to check my order.

        Use the lookup_order tool with:
        order_id: e481f51cbdc54678b7cc49136f2d6af7
        authenticated_customer_unique_id: 7c396fd4830fd04220f754e42b4e5bff

        Do not answer from your own knowledge.
        """
    )

    # 6. Display the tool call requested by the LLM
    print("\nLLM response:")
    print(response)

    print("\nTool calls requested by the LLM:")
    print(response.tool_calls)


if __name__ == "__main__":
    asyncio.run(main())
