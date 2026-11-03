
import json
import os
import sys
import asyncio

from pathlib import Path
from typing import Literal
from typing_extensions import TypedDict

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_mcp_adapters.client import MultiServerMCPClient


from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage
)

from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import InMemorySaver

from refund_workflow import prepare_refund_request
from refund_store import save_refund_request, set_refund_thread_id


# ==================================================
# 1. INITIAL SETUP
# ==================================================

load_dotenv()

PROJECT_DIR = Path(__file__).resolve().parent

DEMO_REQUEST_DATE = "2017-10-20"

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
    search_kwargs={"k": 4}
)


# ==================================================
# 3. GRAPH STATE
# ==================================================

class SupportState(MessagesState, total=False):

    route: Literal[
        "POLICY", "ORDER", "BOTH",
        "GENERAL", "UNKNOWN", "REFUND"
    ]

    retrieved_context: str
    refund_data: dict
    refund_stage: str
    refund_result: dict
    verified_order: dict
    refund_policy_context: str


# ==================================================
# 4. INTENT ROUTER
# ==================================================

class RouteDecision(BaseModel):

    route: Literal[
        "POLICY",
        "ORDER",
        "BOTH",
        "GENERAL",
        "UNKNOWN",
        "REFUND"
    ] = Field(
        description=(
            "POLICY: ShopEase policy information only.\n"
            "ORDER: Specific order information only.\n"
            "BOTH: Requires policy and order information.\n"
            "GENERAL: Greetings and ordinary conversation.\n"
            "UNKNOWN: Ambiguous requests.\n"
            "REFUND: Explicit request to initiate a refund."
        )
    )


router_llm = llm.with_structured_output(RouteDecision)


async def intent_router(state: SupportState):

    refund_data = state.get("refund_data", {})

    if refund_data.get("active"):
        print("\n--- Intent Router: REFUND (continuing) ---")
        return {"route": "REFUND"}

    customer_question = next(
        message.content
        for message in reversed(state["messages"])
        if isinstance(message, HumanMessage)
    )

    prompt = [
        SystemMessage(
            content=(
                "You are the ShopEase intent classifier.\n\n"
                "POLICY: Questions about ShopEase policies.\n"
                "ORDER: Questions about a specific order.\n"
                "BOTH: Questions needing policy and order information.\n"
                "GENERAL: Greetings and ordinary conversation.\n"
                "UNKNOWN: Ambiguous requests.\n"
                "REFUND: Explicit requests to initiate a refund.\n\n"
                "Examples:\n"
                "'What is your return policy?' -> POLICY\n"
                "'Where is my order?' -> ORDER\n"
                "'Is my order eligible for a refund?' -> BOTH\n"
                "'Please initiate a refund.' -> REFUND\n"
                "'Hi!' -> GENERAL\n"
                "'Help me with that thing.' -> UNKNOWN\n\n"
                "Do not classify a question about refund eligibility "
                "as REFUND unless the customer wants to initiate "
                "the refund process."
            )
        ),
        HumanMessage(content=str(customer_question))
    ]

    decision = await router_llm.ainvoke(prompt)

    print(f"\n--- Intent Router: {decision.route} ---")

    return {"route": decision.route}


# ==================================================
# 5. CONDITIONAL ROUTING
# ==================================================

def route_request(state: SupportState):

    route = state["route"]

    if route in ["POLICY", "BOTH"]:
        return "retrieve_policy"

    if route == "REFUND":
        return "refund_intake"

    return "support_agent"


# ==================================================
# 6. NORMAL POLICY RETRIEVAL
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
    print(context if context else "No relevant policy documents found.")

    return {"retrieved_context": context}


# ==================================================
# 7. SUPPORT AGENT
# ==================================================

def create_support_agent(llm_with_tools):

    async def support_agent(state: SupportState):

        route = state.get("route", "GENERAL")

        system_prompt = SystemMessage(
            content=(
                "You are a helpful customer support assistant "
                "for ShopEase.\n\n"
                "Be polite, clear, and concise. Never invent "
                "order details, policies, or decisions.\n\n"
                f"CURRENT CATEGORY: {route}\n\n"
                "POLICY: Use retrieved policy context.\n"
                "ORDER: Use lookup_order for specific order information.\n"
                "BOTH: Use policy context and order lookup when needed.\n"
                "GENERAL: Respond naturally.\n"
                "UNKNOWN: Ask the customer to clarify.\n\n"
                "Do not initiate or approve refunds through this "
                "normal support response.\n\n"
                "The Olist dataset is historical demonstration data, "
                "not live order tracking.\n\n"
                "The customer identity is a test identity, not "
                "production authentication.\n\n"
                "carrier_date means carrier handover, not necessarily "
                "dispatch. delivered_date is recorded delivery. "
                "estimated_delivery_date is the estimated delivery date.\n\n"
                "RETRIEVED POLICY CONTEXT:\n"
                f"{state.get('retrieved_context', '')}"
            )
        )

        response = await llm_with_tools.ainvoke(
            [system_prompt, *state["messages"]]
        )

        return {"messages": [response]}

    return support_agent


# ==================================================
# 8. REFUND INFORMATION EXTRACTION
# ==================================================

class RefundDetails(BaseModel):

    order_id: str | None = None
    customer_unique_id: str | None = None
    reason: str | None = None

    item_unused: bool | None = None
    original_packaging: bool | None = None

    is_personalized_or_digital: bool | None = None
    damage_reported_date: str | None = None


refund_extractor = llm.with_structured_output(RefundDetails)


# ==================================================
# 9. REFUND INTAKE
# ==================================================

def refund_intake(state):

    refund_data = dict(state.get("refund_data", {}))

    if not refund_data.get("active"):
        refund_data = {
            "active": True,
            "pending_question": None
        }

    messages = state.get("messages", [])

    latest_message = next(
        (m for m in reversed(messages) if isinstance(m, HumanMessage)),
        None
    )

    if latest_message is None:
        return {
            "refund_data": refund_data,
            "refund_stage": "COLLECTING",
            "messages": [
                AIMessage(content="Let's start your refund request. What is your order ID?")
            ]
        }

    customer_text = latest_message.content.strip()
    pending = refund_data.get("pending_question")

    if pending:

        answer = customer_text.strip()

        if pending in [
            "item_unused",
            "original_packaging",
            "is_personalized_or_digital"
        ]:

            normalized = answer.lower().rstrip(".!?")

            if normalized in ["yes", "y", "true"]:
                refund_data[pending] = True
                refund_data["pending_question"] = None

            elif normalized in ["no", "n", "false"]:
                refund_data[pending] = False
                refund_data["pending_question"] = None

            else:

                questions = {
                    "item_unused": "Is the item unused? Please answer yes or no.",
                    "original_packaging": "Do you have the original packaging? Please answer yes or no.",
                    "is_personalized_or_digital": "Is this a personalized or digital product? Please answer yes or no."
                }

                return {
                    "refund_data": refund_data,
                    "refund_stage": "COLLECTING",
                    "messages": [
                        AIMessage(content=questions[pending])
                    ]
                }

        elif pending in [
            "order_id",
            "customer_unique_id",
            "reason",
            "damage_reported_date"
        ]:

            refund_data[pending] = answer
            refund_data["pending_question"] = None

    else:

        extracted = refund_extractor.invoke([
            SystemMessage(
                content=(
                    "Extract only information explicitly provided "
                    "by the customer. Do not infer missing information. "
                    "For Boolean fields, use true or false only "
                    "when clearly stated."
                )
            ),
            HumanMessage(content=customer_text)
        ])

        extracted_data = extracted.model_dump(exclude_none=True)

        for key, value in extracted_data.items():
            refund_data[key] = value

    required_fields = [
        ("order_id", "Please provide your order ID."),
        (
            "customer_unique_id",
            "Please provide the customer ID associated with your order."
        ),
        ("reason", "What is the reason for your refund?"),
        ("item_unused", "Is the item unused? Please answer yes or no."),
        (
            "original_packaging",
            "Do you have the original packaging? Please answer yes or no."
        ),
        (
            "is_personalized_or_digital",
            "Is this a personalized or digital product? Please answer yes or no."
        )
    ]

    for field, question in required_fields:

        if refund_data.get(field) is None or refund_data.get(field) == "":

            refund_data["pending_question"] = field

            return {
                "refund_data": refund_data,
                "refund_stage": "COLLECTING",
                "messages": [AIMessage(content=question)]
            }

    reason = str(refund_data.get("reason", "")).lower()

    if (
        ("damage" in reason or "broken" in reason)
        and not refund_data.get("damage_reported_date")
    ):

        refund_data["pending_question"] = "damage_reported_date"

        return {
            "refund_data": refund_data,
            "refund_stage": "COLLECTING",
            "messages": [
                AIMessage(
                    content=(
                        "On what date did you report the damage? "
                        "Please use YYYY-MM-DD."
                    )
                )
            ]
        }

    refund_data["pending_question"] = None

    return {
        "refund_data": refund_data,
        "refund_stage": "READY",
        "messages": [
            AIMessage(
                content=(
                    "Thank you. I have collected the required refund details. "
                    "I'll now verify your order and check the refund policy."
                )
            )
        ]
    }


# ==================================================
# 10. REFUND INTAKE ROUTING
# ==================================================

def route_after_refund_intake(state: SupportState):

    if state.get("refund_stage") == "READY":
        return "refund_lookup_order"

    return END


# ==================================================
# 11. REFUND ORDER LOOKUP THROUGH MCP
# ==================================================

def create_refund_lookup_node(lookup_tool):

    async def refund_lookup_order(state):

        refund_data = state.get("refund_data", {})

        try:

            result = await lookup_tool.ainvoke({
                "order_id": refund_data["order_id"],
                "authenticated_customer_unique_id": (
                    refund_data["customer_unique_id"]
                )
            })

            if isinstance(result, list):

                text_parts = []

                for item in result:

                    if isinstance(item, str):
                        text_parts.append(item)

                    elif isinstance(item, dict) and "text" in item:
                        text_parts.append(item["text"])

                    elif hasattr(item, "text"):
                        text_parts.append(item.text)

                result = "\n".join(text_parts)

            if isinstance(result, str):
                result = json.loads(result)

            if not isinstance(result, dict):
                raise ValueError(
                    "Unexpected response format from MCP order lookup."
                )

            if not result.get("success"):

                return {
                    "refund_result": {
                        "success": False,
                        "message": result.get(
                            "message",
                            "Order lookup failed."
                        )
                    },
                    "refund_stage": "FAILED"
                }

            return {
                "verified_order": result["order"],
                "refund_stage": "ORDER_VERIFIED"
            }

        except Exception as error:

            return {
                "refund_result": {
                    "success": False,
                    "message": f"Order lookup failed: {str(error)}"
                },
                "refund_stage": "FAILED"
            }

    return refund_lookup_order


# ==================================================
# 12. REFUND POLICY RETRIEVAL
# ==================================================

def refund_policy_retrieval(state: SupportState):

    query = (
        "ShopEase refund and return policy: 30 calendar day return "
        "window after delivery, unused items, original packaging, "
        "damaged product reporting within 7 calendar days, "
        "personalized and digital product exclusions, "
        "and refund processing after approval."
    )

    documents = retriever.invoke(query)

    context = "\n\n".join(
        f"Source: {doc.metadata.get('source', 'Unknown')}\n"
        f"{doc.page_content}"
        for doc in documents
    )

    print("\n--- Refund Policy Evidence ---")
    print(context if context else "No refund policy evidence found.")

    if not context.strip():

        return {
            "refund_policy_context": "",
            "refund_stage": "POLICY_NOT_FOUND",
            "refund_result": {
                "success": False,
                "message": (
                    "Refund policy evidence could not be retrieved. "
                    "The request requires manual review."
                )
            }
        }

    return {
        "refund_policy_context": context,
        "refund_stage": "POLICY_RETRIEVED"
    }


# ==================================================
# 13. REFUND PREPARATION + APPROVAL INTERRUPT
# ==================================================

def refund_preparation(state: SupportState):

    data = state["refund_data"]

    result = prepare_refund_request(
        order=state["verified_order"],
        customer_unique_id=data["customer_unique_id"],
        reason=data["reason"],
        request_date=DEMO_REQUEST_DATE,
        policy_context=state["refund_policy_context"],
        item_unused=data["item_unused"],
        original_packaging=data["original_packaging"],
        damage_reported_date=data.get("damage_reported_date"),
        is_personalized_or_digital=data["is_personalized_or_digital"]
    )

    if not result.get("success"):

        return {
            "refund_result": result,
            "refund_stage": "PREPARATION_FAILED"
        }

    refund = result["refund_request"]

    if refund["eligibility_status"] != "ELIGIBLE":

        return {
            "refund_result": result,
            "refund_stage": "NOT_ELIGIBLE"
        }

    # Save the refund request to SQLite.
    request_id = save_refund_request(refund)

    # The thread ID is generated and stored by refund_store.py.
    thread_id = f"refund-{request_id}"
    set_refund_thread_id(request_id, thread_id)

    result = dict(result)
    result["request_id"] = request_id
    result["thread_id"] = thread_id

    # Start the separate persistent approval graph.
    from approval_service import get_approval_graph

    with get_approval_graph() as approval_graph:

        approval_result = approval_graph.invoke(
            {"refund_request": refund},
            config={
                "configurable": {
                    "thread_id": thread_id
                }
            }
        )

    if not approval_result.get("__interrupt__"):

        raise RuntimeError(
            "The refund approval workflow did not pause for human review."
        )

    result["approval_details"] = (
        approval_result["__interrupt__"][0].value
    )

    print("\n--- Refund workflow paused for human approval ---")
    print("Request ID:", request_id)
    print("Thread ID:", thread_id)

    return {
        "refund_result": result,
        "refund_stage": "AWAITING_HUMAN_APPROVAL"
    }


# ==================================================
# 14. CUSTOMER REFUND RESPONSE
# ==================================================

def refund_response(state: SupportState):

    result = state.get("refund_result", {})

    if not result.get("success"):

        message = (
            "I could not prepare your refund request.\n\n"
            + result.get("message", "Please contact support.")
        )

    else:

        refund = result["refund_request"]
        status = refund["eligibility_status"]

        if status == "ELIGIBLE":

            message = (
                "Your refund request has passed the automated "
                "eligibility checks.\n\n"
                f"Request ID: {result['request_id']}\n"
                f"Order ID: {refund['order_id']}\n"
                f"Refund amount: {refund['currency']} "
                f"{refund['refund_amount']}\n"
                f"Demo request date: {refund['request_date']}\n\n"
                "Status: PENDING HUMAN APPROVAL\n\n"
                "Your request has been saved, and the LangGraph "
                "approval workflow is paused until a human reviewer "
                "makes a decision.\n\n"
                "No refund has been executed."
            )

        else:

            message = (
                f"Your refund request has status: {status}.\n\n"
                f"Reasons: {'; '.join(refund['eligibility_reasons'])}\n\n"
                "No refund has been executed. The request cannot "
                "proceed to automatic approval."
            )

    data = dict(state.get("refund_data", {}))
    data["active"] = False

    return {
        "refund_data": data,
        "refund_result": result,
        "messages": [AIMessage(content=message)]
    }


# ==================================================
# 15. BUILD APPLICATION
# ==================================================

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

    mcp_tools = await client.get_tools()

    print("\nConnected to ShopEase MCP server.")

    for tool in mcp_tools:
        print(f"- {tool.name}")

    lookup_tool = next(
        tool for tool in mcp_tools
        if tool.name == "lookup_order"
    )

    llm_with_tools = llm.bind_tools(mcp_tools)

    support_agent_node = create_support_agent(llm_with_tools)

    builder = StateGraph(SupportState)

    # Support nodes
    builder.add_node("intent_router", intent_router)
    builder.add_node("retrieve_policy", retrieve_policy)
    builder.add_node("support_agent", support_agent_node)
    builder.add_node("tools", ToolNode(mcp_tools))

    # Refund nodes
    builder.add_node("refund_intake", refund_intake)

    builder.add_node(
        "refund_lookup_order",
        create_refund_lookup_node(lookup_tool)
    )

    builder.add_node(
        "refund_policy_retrieval",
        refund_policy_retrieval
    )

    builder.add_node(
        "refund_preparation",
        refund_preparation
    )

    builder.add_node(
        "refund_response",
        refund_response
    )

    builder.add_edge(START, "intent_router")

    builder.add_conditional_edges(
        "intent_router",
        route_request,
        {
            "retrieve_policy": "retrieve_policy",
            "support_agent": "support_agent",
            "refund_intake": "refund_intake"
        }
    )

    # Normal support flow
    builder.add_edge("retrieve_policy", "support_agent")

    builder.add_conditional_edges(
        "support_agent",
        tools_condition
    )

    builder.add_edge("tools", "support_agent")

    # Refund intake flow
    builder.add_conditional_edges(
        "refund_intake",
        route_after_refund_intake,
        {
            "refund_lookup_order": "refund_lookup_order",
            END: END
        }
    )

    builder.add_conditional_edges(
        "refund_lookup_order",
        lambda state: (
            "refund_policy_retrieval"
            if state.get("refund_stage") == "ORDER_VERIFIED"
            else "refund_response"
        ),
        {
            "refund_policy_retrieval": "refund_policy_retrieval",
            "refund_response": "refund_response"
        }
    )

    builder.add_conditional_edges(
        "refund_policy_retrieval",
        lambda state: (
            "refund_preparation"
            if state.get("refund_stage") == "POLICY_RETRIEVED"
            else "refund_response"
        ),
        {
            "refund_preparation": "refund_preparation",
            "refund_response": "refund_response"
        }
    )

    builder.add_edge("refund_preparation", "refund_response")
    builder.add_edge("refund_response", END)

    # Customer conversation checkpoint.
    # Refund approval has its own persistent SQLite checkpointer.
    checkpointer = InMemorySaver()

    support_graph = builder.compile(
        checkpointer=checkpointer
    )

    # ==================================================
    # 16. CUSTOMER INTERACTION
    # ==================================================

    print("\n========== SHOPEASE CUSTOMER SUPPORT ==========")
    print("Historical Olist demonstration mode.")
    print(f"Refund demo date: {DEMO_REQUEST_DATE}")
    print("Type 'exit' to close the application.")

    config = {
        "configurable": {
            "thread_id": "shopease-customer-session-001"
        }
    }

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
                    ]
                },
                config=config
            )

            final_message = result["messages"][-1]

            print("\nShopEase Support:")

            print(final_message.content)

        except Exception as error:

            print("\nAn error occurred.")
            print(f"Error: {error}")


# ==================================================
# 17. APPLICATION ENTRY POINT
# ==================================================

if __name__ == "__main__":
    asyncio.run(main())
