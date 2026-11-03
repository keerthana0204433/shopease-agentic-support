
from typing import TypedDict, Literal

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command


# 1. Define the state shared across workflow nodes

class RefundState(TypedDict, total=False):
    refund_request: dict
    approval_decision: Literal["APPROVE", "REJECT"]
    approval_reason: str
    execution_status: str
    final_message: str


# 2. Pause the workflow and request human approval

def human_approval(state: RefundState):

    refund = state["refund_request"]

    # Only eligible requests can reach the approval stage
    if refund["eligibility_status"] != "ELIGIBLE":
        return {
            "execution_status": "NOT_EXECUTED",
            "final_message": "Refund is not eligible for approval."
        }

    decision = interrupt({
        "message": "Human approval is required before processing this refund.",
        "refund_details": refund,
        "available_actions": ["APPROVE", "REJECT"]
    })

    approval = decision.get("decision", "").upper()

    if approval not in ["APPROVE", "REJECT"]:
        return {
            "approval_decision": "REJECT",
            "approval_reason": "Invalid approval decision."
        }

    return {
        "approval_decision": approval,
        "approval_reason": decision.get("reason", "")
    }


# 3. Decide what happens after human review

def route_after_approval(state: RefundState):

    if state.get("approval_decision") == "APPROVE":
        return "execute_refund"

    return "reject_refund"


# 4. Simulate refund execution after approval

def execute_refund(state: RefundState):

    refund = state["refund_request"]

    return {
        "execution_status": "SIMULATED_SUCCESS",
        "final_message": (
            f"Simulated refund of {refund['currency']} "
            f"{refund['refund_amount']} for order "
            f"{refund['order_id']} was executed after human approval."
        )
    }


# 5. Handle rejected requests

def reject_refund(state: RefundState):

    return {
        "execution_status": "REJECTED",
        "final_message": (
            "Refund request was rejected by the human reviewer. "
            "No refund was executed. Reason: "
            + state.get("approval_reason", "Not provided")
        )
    }


# 6. Construct the LangGraph workflow

def build_refund_approval_graph():

    builder = StateGraph(RefundState)

    builder.add_node("human_approval", human_approval)
    builder.add_node("execute_refund", execute_refund)
    builder.add_node("reject_refund", reject_refund)

    builder.add_edge(START, "human_approval")

    builder.add_conditional_edges(
        "human_approval",
        route_after_approval,
        {
            "execute_refund": "execute_refund",
            "reject_refund": "reject_refund"
        }
    )

    builder.add_edge("execute_refund", END)
    builder.add_edge("reject_refund", END)

    # Retain workflow state while waiting for approval
    checkpointer = InMemorySaver()

    return builder.compile(checkpointer=checkpointer)


# 7. Run a standalone approval demonstration

def run_approval_demo():

    graph = build_refund_approval_graph()

    refund_request = {
        "order_id": "TEST-001",
        "refund_amount": 29.99,
        "currency": "BRL",
        "eligibility_status": "ELIGIBLE",
        "approval_status": "PENDING",
        "execution_status": "NOT_EXECUTED"
    }

    config = {
        "configurable": {
            "thread_id": "refund-demo-001"
        }
    }

    print("\n========== REFUND APPROVAL WORKFLOW ==========")

    # First invocation pauses at the human approval node
    result = graph.invoke(
        {"refund_request": refund_request},
        config=config
    )

    print("\n--- HUMAN APPROVAL REQUIRED ---")
    print(result["__interrupt__"][0].value)

    decision = input("\nEnter APPROVE or REJECT: ").strip().upper()
    reason = input("Enter reviewer reason: ").strip()

    # Resume the paused workflow with the human decision
    result = graph.invoke(
        Command(
            resume={
                "decision": decision,
                "reason": reason
            }
        ),
        config=config
    )

    print("\n========== FINAL RESULT ==========")
    print(result["final_message"])
    print("Execution status:", result["execution_status"])


if __name__ == "__main__":
    run_approval_demo()
