
import os
import sys
import asyncio
from pathlib import Path

from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_mcp_adapters.client import MultiServerMCPClient

from langchain_core.messages import SystemMessage, HumanMessage

from langgraph.graph import StateGraph, START, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition


# --------------------------------------------------
# 1. LOAD ENVIRONMENT VARIABLES
# --------------------------------------------------

load_dotenv()

PROJECT_DIR = Path(__file__).resolve().parent


# --------------------------------------------------
# 2. INITIALIZE THE LLM
# --------------------------------------------------

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)


# --------------------------------------------------
# 3. CONNECT TO THE EXISTING CHROMADB DATABASE
# --------------------------------------------------

DATABASE_PATH = PROJECT_DIR / "data" / "chroma_db"

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    google_api_key=os.getenv("GEMINI_API_KEY"),
)

vector_store = Chroma(
    collection_name="shop_policy",
    persist_directory=str(DATABASE_PATH),
    embedding_function=embeddings,
)

retriever = vector_store.as_retriever(
    search_kwargs={"k": 2}
)


# --------------------------------------------------
# 4. DEFINE THE GRAPH STATE
# --------------------------------------------------

class SupportState(MessagesState):
    retrieved_context: str


# --------------------------------------------------
# 5. POLICY RETRIEVAL NODE
# --------------------------------------------------

def retrieve_policy(state: SupportState):

    # Get the customer's latest message
    question = state["messages"][-1].content

    # Search the vector database
    documents = retriever.invoke(question)

    # Combine the retrieved policy chunks
    context = "\n\n".join(
        f"Source: {doc.metadata.get('source', 'Unknown')}\n"
        f"{doc.page_content}"
        for doc in documents
    )

    print("\n--- Retrieved Policy Context ---")
    print(context)

    return {
        "retrieved_context": context
    }


# --------------------------------------------------
# 6. SUPPORT AGENT NODE
# --------------------------------------------------

async def support_agent(state: SupportState):

    # Give the LLM the retrieved policy context
    system_prompt = SystemMessage(
        content=(
            "You are a helpful customer support assistant for ShopEase. "

            "Answer politely and clearly. "

            "Use the retrieved policy context as the source of truth "
            "for questions about ShopEase policies. "

            "If the policy context does not contain enough information, "
            "do not invent an answer. Recommend contacting support. "

            "Use the lookup_order tool whenever the customer asks "
            "for information about a specific order. "

            "Never invent order details or refund decisions. "

            "The order lookup tool requires an order ID and a customer "
            "identity. For this demonstration, the customer may provide "
            "both values in their message. This is a test identity, "
            "not production authentication. "

            f"\n\nRetrieved policy context:\n"
            f"{state.get('retrieved_context', '')}"
        )
    )

    # Send conversation history to the LLM
    messages = [
        system_prompt,
        *state["messages"]
    ]

    response = await llm_with_tools.ainvoke(messages)

    return {
        "messages": [response]
    }


# --------------------------------------------------
# 7. CONNECT TO THE MCP SERVER
# --------------------------------------------------

async def main():

    client = MultiServerMCPClient(
        {
            "shopease": {
                "transport": "stdio",
                "command": sys.executable,
                "args": [
                    str(PROJECT_DIR / "mcp_server.py")
                ]
            }
        }
    )

    # Discover tools exposed by the MCP server
    mcp_tools = await client.get_tools()

    print("\nConnected to ShopEase MCP server.")

    print("\nAvailable MCP tools:")
    for tool in mcp_tools:
        print(tool.name)

    # Give the MCP tools to the LLM
    global llm_with_tools
    llm_with_tools = llm.bind_tools(mcp_tools)


    # --------------------------------------------------
    # 8. BUILD THE LANGGRAPH WORKFLOW
    # --------------------------------------------------

    builder = StateGraph(SupportState)

    # Register nodes
    builder.add_node("retrieve_policy", retrieve_policy)
    builder.add_node("support_agent", support_agent)
    builder.add_node("tools", ToolNode(mcp_tools))

    # Define workflow
    builder.add_edge(START, "retrieve_policy")

    builder.add_edge(
        "retrieve_policy",
        "support_agent"
    )

    # Route to tools if the LLM requests one
    builder.add_conditional_edges(
        "support_agent",
        tools_condition
    )

    # Return tool results to the LLM
    builder.add_edge(
        "tools",
        "support_agent"
    )

    # Compile the graph
    support_graph = builder.compile()


    # --------------------------------------------------
    # 9. GET CUSTOMER INPUT
    # --------------------------------------------------

    print("\n========== SHOPEASE SUPPORT AGENT ==========")

    customer_message = input("\nCustomer message: ")

    # Execute the graph
    result = await support_graph.ainvoke(
        {
            "messages": [
                HumanMessage(content=customer_message)
            ],
            "retrieved_context": ""
        }
    )


    # --------------------------------------------------
    # 10. DISPLAY THE CONVERSATION
    # --------------------------------------------------

    print("\n========== AGENT EXECUTION ==========")

    for message in result["messages"]:

        print(f"\n{message.type.upper()}:")

        if message.content:
            print(message.content)

        if getattr(message, "tool_calls", None):
            print("Tool calls:", message.tool_calls)

        if message.type == "tool":
            print("Tool result:", message.content)

    print("\n========== FINAL SUPPORT RESPONSE ==========")

    # The final assistant message
    final_message = result["messages"][-1]

    print(final_message.content)


if __name__ == "__main__":
    asyncio.run(main())
