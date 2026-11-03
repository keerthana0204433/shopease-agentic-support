
from refund_workflow import prepare_refund_request
from refund_approval import build_refund_approval_graph
from langgraph.types import Command


class RefundAgent:

    def __init__(self):
        # Build the LangGraph approval workflow
        self.approval_graph = build_refund_approval_graph()

    # Step 1: Prepare and validate a refund request
    def prepare_request(
        self,
        order_id,
        customer_unique_id,
        reason,
        request_date,
        item_unused=None,
        original_packaging=None,
        damage_reported_date=None,
        is_personalized_or_digital=None
    ):

        result = prepare_refund_request(
            order_id=order_id,
            customer_unique_id=customer_unique_id,
            reason=reason,
            request_date=request_date,
            item_unused=item_unused,
            original_packaging=original_packaging,
            damage_reported_date=damage_reported_date,
            is_personalized_or_digital=is_personalized_or_digital
        )

        if not result.get("success"):
            return result

        refund_request = result["refund_request"]

        if refund_request["eligibility_status"] != "ELIGIBLE":
            return {
                "success": True,
                "status": refund_request["eligibility_status"],
                "refund_request": refund_request,
                "message": (
                    "This request cannot proceed to automatic approval. "
                    "Additional review or information may be required."
                )
            }

        return {
            "success": True,
            "status": "READY_FOR_APPROVAL",
            "refund_request": refund_request
        }

    # Step 2: Submit an eligible request to the approval workflow
    def request_approval(self, refund_request, thread_id):

        config = {
            "configurable": {
                "thread_id": thread_id
            }
        }

        result = self.approval_graph.invoke(
            {"refund_request": refund_request},
            config=config
        )

        if result.get("__interrupt__"):

            return {
                "success": True,
                "status": "AWAITING_HUMAN_APPROVAL",
                "thread_id": thread_id,
                "approval_details": result["__interrupt__"][0].value
            }

        return {
            "success": True,
            "status": result.get("execution_status"),
            "message": result.get("final_message")
        }

    # Step 3: Resume the workflow after a human decision
    def submit_human_decision(self, thread_id, decision, reason=""):

        decision = decision.upper().strip()

        if decision not in ["APPROVE", "REJECT"]:
            return {
                "success": False,
                "message": "Decision must be APPROVE or REJECT."
            }

        config = {
            "configurable": {
                "thread_id": thread_id
            }
        }

        result = self.approval_graph.invoke(
            Command(
                resume={
                    "decision": decision,
                    "reason": reason
                }
            ),
            config=config
        )

        return {
            "success": True,
            "status": result.get("execution_status"),
            "message": result.get("final_message")
        }


# Step 4: Test the complete refund journey
if __name__ == "__main__":

    agent = RefundAgent()

    # Prepare a refund request using our Olist test order
    preparation = agent.prepare_request(
        order_id="e481f51cbdc54678b7cc49136f2d6af7",
        customer_unique_id="7c396fd4830fd04220f754e42b4e5bff",
        reason="Changed my mind",
        request_date="2017-10-20",
        item_unused=True,
        original_packaging=True,
        is_personalized_or_digital=False
    )

    print("\n========== REFUND PREPARATION ==========")
    print(preparation)

    # Continue only if the request is eligible
    if preparation.get("status") == "READY_FOR_APPROVAL":

        approval = agent.request_approval(
            refund_request=preparation["refund_request"],
            thread_id="shopease-refund-test-001"
        )

        print("\n========== APPROVAL REQUEST ==========")
        print(approval)

        if approval.get("status") == "AWAITING_HUMAN_APPROVAL":

            decision = input(
                "\nEnter APPROVE or REJECT: "
            ).strip()

            reason = input(
                "Enter reviewer reason: "
            ).strip()

            final_result = agent.submit_human_decision(
                thread_id=approval["thread_id"],
                decision=decision,
                reason=reason
            )

            print("\n========== FINAL REFUND RESULT ==========")
            print(final_result)
