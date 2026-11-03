
import os
import sys
import asyncio
from pathlib import Path
from typing import Annotated

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_mcp_adapters.client import MultiServerMCPClient

from langgraph.graph import StateGraph, START, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

from langchain_core.messages import SystemMessage, HumanMessage

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

    # 3. Initialize LLM and give it access to the tools
    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0
    )

    llm_with_tools = llm.bind_tools(tools)

    # 4. Define the agent node
    async def support_agent(state: MessagesState):

        response = await llm_with_tools.ainvoke(
            state["messages"]
        )

        return {"messages": [response]}

    # 5. Build the LangGraph workflow
    builder = StateGraph(MessagesState)

    builder.add_node("support_agent", support_agent)

    # ToolNode executes the MCP tool requested by the LLM
    builder.add_node("tools", ToolNode(tools))

    builder.add_edge(START, "support_agent")

    # Decide whether to execute a tool or finish
    builder.add_conditional_edges(
        "support_agent",
        tools_condition
    )

    # After tool execution, send the result back to the LLM
    builder.add_edge("tools", "support_agent")

    graph = builder.compile()

    # 6. Run the graph
    result = await graph.ainvoke(
        {
            "messages": [
                SystemMessage(
                    content=(
                        "You are a ShopEase customer support assistant. "
                        "Use the lookup_order tool when order information "
                        "is needed. Never invent order details. "
                        "Use only the supplied customer identity for this test."
                    )
                ),
                HumanMessage(
                    content=(
                        "Please check my order. "
                        "My order ID is e481f51cbdc54678b7cc49136f2d6af7. "
                        "My customer identity for this test is "
                        "7c396fd4830fd04220f754e42b4e5bff. "
                        "Tell me its delivery status and estimated delivery date."
                    )
                )
            ]
        }
    )

    # 7. Display the conversation
    print("\n========== AGENT EXECUTION ==========")

    for message in result["messages"]:
        print(f"\n{message.type.upper()}:")

    if message.content:
        print(message.content)

    if getattr(message, "tool_calls", None):
        print("Tool calls:", message.tool_calls)

    if message.type == "tool":
        print("Tool result:", message.content)


if __name__ == "__main__":
    asyncio.run(main())
