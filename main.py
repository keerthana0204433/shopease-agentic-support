
import os
import sys
import asyncio
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_mcp_adapters.client import MultiServerMCPClient

from langchain_core.messages import SystemMessage, HumanMessage

from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition


# ==================================================
# 1. INITIAL SETUP
# ==================================================

load_dotenv()

PROJECT_DIR = Path(__file__).resolve().parent

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)


# ==================================================
# 2. RAG SETUP
# ==================================================

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


# ==================================================
# 3. GRAPH STATE
# ==================================================

class SupportState(MessagesState):
    route: Literal[
        "POLICY",
        "ORDER",
        "BOTH",
        "GENERAL",
        "UNKNOWN"
    ]

    retrieved_context: str


# ==================================================
# 4. INTENT ROUTER STRUCTURED OUTPUT
# ==================================================

class RouteDecision(BaseModel):

    route: Literal[
        "POLICY",
        "ORDER",
        "BOTH",
        "GENERAL",
        "UNKNOWN"
    ] = Field(
        description=(
            "POLICY: The question needs ShopEase policy information, "
            "but does not require looking up a specific order.\n"

            "ORDER: The question requires information about a specific "
            "order, but does not need policy retrieval.\n"

            "BOTH: The question requires both ShopEase policy information "
            "and a lookup of a specific order.\n"

            "GENERAL: Greetings, thanks, ordinary conversation, or "
            "general questions that do not require policy or order data.\n"

            "UNKNOWN: The request is too ambiguous to determine the "
            "information needed, even after considering its context."
        )
    )


router_llm = llm.with_structured_output(RouteDecision)


# ==================================================
# 5. INTENT ROUTER NODE
# ==================================================

async def intent_router(state: SupportState):

    customer_question = next(
        message.content
        for message in reversed(state["messages"])
        if isinstance(message, HumanMessage)
    )

    router_prompt = [
        SystemMessage(
            content=(
                "You are an intent classification system for ShopEase.\n\n"

                "Your task is to identify what information is needed "
                "to answer the customer's latest question.\n\n"

                "Do not classify based only on keywords. "
                "Understand the meaning and intent of the question.\n\n"

                "ROUTE DEFINITIONS:\n\n"

                "POLICY:\n"
                "Choose when the answer requires ShopEase policy documents, "
                "such as return eligibility, refund conditions, damaged-item "
                "rules, shipping timelines, or product exclusions. "
                "A customer does not need to explicitly mention the word "
                "'policy'.\n\n"

                "ORDER:\n"
                "Choose when the answer requires looking up a specific "
                "order, such as its status, items, payment, or delivery date, "
                "without needing policy information.\n\n"

                "BOTH:\n"
                "Choose when answering requires both policy information "
                "and details from a specific order. For example, determining "
                "whether a particular delivered order qualifies for a refund.\n\n"

                "GENERAL:\n"
                "Choose for greetings, thanks, ordinary conversation, "
                "and questions that do not need policy documents or order data.\n\n"

                "UNKNOWN:\n"
                "Choose when the customer's request is too ambiguous to "
                "identify the required information. Do not use UNKNOWN "
                "merely because an order ID or other required detail is missing. "
                "If the intent is clear but a detail is missing, choose "
                "the appropriate route and let the support agent ask for it.\n\n"

                "IMPORTANT EXAMPLES:\n"
                "1. 'Can I return a personalised product?' -> POLICY\n"
                "2. 'What is happening with my order?' -> ORDER\n"
                "3. 'My order arrived damaged. Can I get a refund?' -> BOTH "
                "if a specific order must be checked; otherwise POLICY.\n"
                "4. 'Hi, how are you?' -> GENERAL\n"
                "5. 'Thank you!' -> GENERAL\n"
                "6. 'Can you help me with that thing?' -> UNKNOWN "
                "if there is no clear context.\n"
                "7. 'My package is late. What does your shipping policy say?' "
                "-> BOTH if the specific order status must also be checked.\n\n"

                "Return exactly one of the five supported categories."
            )
        ),

        HumanMessage(content=str(customer_question)),
    ]

    decision = await router_llm.ainvoke(router_prompt)

    print(f"\n--- Intent Router: {decision.route} ---")

    return {"route": decision.route}


# ==================================================
# 6. CONDITIONAL ROUTING
# ==================================================

def route_request(state: SupportState):

    route = state["route"]

    if route in ["POLICY", "BOTH"]:
        return "retrieve_policy"

    elif route in ["ORDER", "GENERAL", "UNKNOWN"]:
        return "support_agent"

    return "support_agent"


# ==================================================
# 7. RAG RETRIEVAL NODE
# ==================================================

def retrieve_policy(state: SupportState):

    question = next(
        message.content
        for message in reversed(state["messages"])
        if isinstance(message, HumanMessage)
    )

    documents = retriever.invoke(str(question))

    context = "\n\n".join(
        f"Source: {doc.metadata.get('source', 'Unknown')}\n"
        f"{doc.page_content}"
        for doc in documents
    )

    print("\n--- Retrieved Policy Context ---")

    if context:
        print(context)
    else:
        print("No relevant policy documents found.")

    return {"retrieved_context": context}


# ==================================================
# 8. SUPPORT AGENT NODE
# ==================================================

def create_support_agent(llm_with_tools):

    async def support_agent(state: SupportState):

        route = state.get("route", "GENERAL")

        system_prompt = SystemMessage(
            content=(
                "You are a helpful customer support assistant "
                "for ShopEase.\n\n"

                "GENERAL INSTRUCTIONS:\n"
                "Be polite, clear, and concise. "
                "Never invent order details, policies, delivery dates, "
                "or refund decisions.\n\n"

                "CURRENT REQUEST CATEGORY:\n"
                f"{route}\n\n"

                "POLICY INSTRUCTIONS:\n"
                "Use the retrieved policy context as the source of truth "
                "for ShopEase policy questions. "
                "If the context does not contain enough information, "
                "say so and recommend contacting support. "
                "Do not invent policy rules.\n\n"

                "ORDER INSTRUCTIONS:\n"
                "Use the lookup_order tool whenever the customer asks "
                "for information about a specific order. "
                "If the order ID or customer identity is missing, "
                "ask the customer for it. "
                "Never invent these details.\n\n"

                "GENERAL INSTRUCTIONS:\n"
                "For greetings and ordinary conversation, respond naturally. "
                "Do not retrieve or invent ShopEase policy information "
                "unless it is relevant to the question.\n\n"

                "UNKNOWN INSTRUCTIONS:\n"
                "If the request is unclear, politely ask the customer "
                "what they need help with. Do not guess their intent.\n\n"

                "ORDER FIELD INTERPRETATION:\n"
                "carrier_date represents the carrier handover date. "
                "It does not necessarily mean dispatch. "
                "delivered_date represents the recorded delivery date. "
                "estimated_delivery_date represents the estimated delivery date. "
                "Do not describe carrier handover as dispatch unless "
                "the data explicitly confirms it.\n\n"

                "DEMONSTRATION AUTHENTICATION:\n"
                "The customer may provide an order ID and customer identity "
                "for this demonstration. This is a test identity, "
                "not production authentication.\n\n"

                "RETRIEVED POLICY CONTEXT:\n"
                f"{state.get('retrieved_context', '')}"
            )
        )

        messages = [
            system_prompt,
            *state["messages"]
        ]

        response = await llm_with_tools.ainvoke(messages)

        return {"messages": [response]}

    return support_agent


# ==================================================
# 9. BUILD LANGGRAPH
# ==================================================

async def main():

    client = MultiServerMCPClient(
        {
            "shopease": {
                "transport": "stdio",
                "command": sys.executable,
                "args": [
                    str(PROJECT_DIR / "mcp_server.py")
                ],
            }
        }
    )

    mcp_tools = await client.get_tools()

    print("\nConnected to ShopEase MCP server.")

    print("\nAvailable MCP tools:")

    for tool in mcp_tools:
        print(f"- {tool.name}")

    llm_with_tools = llm.bind_tools(mcp_tools)

    support_agent_node = create_support_agent(llm_with_tools)

    builder = StateGraph(SupportState)

    # Register nodes
    builder.add_node("intent_router", intent_router)
    builder.add_node("retrieve_policy", retrieve_policy)
    builder.add_node("support_agent", support_agent_node)
    builder.add_node("tools", ToolNode(mcp_tools))

    # Entry point
    builder.add_edge(START, "intent_router")

    # Conditional routing after classification
    builder.add_conditional_edges(
        "intent_router",
        route_request,
        {
            "retrieve_policy": "retrieve_policy",
            "support_agent": "support_agent",
        },
    )

    # RAG -> Support agent
    builder.add_edge("retrieve_policy", "support_agent")

    # Support agent -> MCP tool or END
    builder.add_conditional_edges(
        "support_agent",
        tools_condition,
    )

    # MCP tool -> Support agent
    builder.add_edge("tools", "support_agent")

    support_graph = builder.compile()

    # ==================================================
    # 10. CUSTOMER INTERACTION LOOP
    # ==================================================

    print("\n========== SHOPEASE CUSTOMER SUPPORT ==========")
    print("Ask your questions below.")
    print("Type 'exit' to close the application.")

    while True:

        customer_message = input("\nCustomer: ").strip()

        if customer_message.lower() in ["exit", "quit"]:
            print("\nThank you for using ShopEase Support!")
            break

        if not customer_message:
            continue

        try:

            result = await support_graph.ainvoke(
                {
                    "messages": [
                        HumanMessage(content=customer_message)
                    ],
                    "retrieved_context": "",
                }
            )

            final_message = result["messages"][-1]

            print("\nShopEase Support:")
            print(final_message.content)

        except Exception as error:

            print("\nAn error occurred while processing your request.")
            print(f"Error: {error}")


# ==================================================
# 11. APPLICATION ENTRY POINT
# ==================================================

if __name__ == "__main__":
    asyncio.run(main())
