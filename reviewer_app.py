
import streamlit as st
import json

from langgraph.types import Command

from refund_store import (
    initialize_database,
    get_pending_refunds,
    get_refund_by_id,
    get_all_refunds
)

from approval_service import get_approval_graph


st.set_page_config(
    page_title="ShopEase | Refund Review",
    page_icon="🛍️",
    layout="wide"
)

initialize_database()

st.title("🛍️ ShopEase")
st.subheader("Refund Review Dashboard")

st.caption(
    "Human-in-the-loop review portal for AI-prepared refund requests."
)

reviewer_id = st.sidebar.text_input(
    "Reviewer ID",
    value="demo_reviewer"
)

st.sidebar.divider()

st.sidebar.info(
    "Review each request carefully. Approval triggers simulated "
    "execution only. No real payment is made."
)


def process_decision(refund, decision, reviewer_reason):

    request_id = refund["request_id"]

    # Reload the latest database record.
    saved_request = get_refund_by_id(request_id)

    if not saved_request:
        raise ValueError("Refund request was not found.")

    if saved_request["approval_status"] != "PENDING":
        raise ValueError("This refund has already been reviewed.")

    thread_id = saved_request.get("thread_id")

    if not thread_id:
        raise ValueError(
            "This is an older refund request without a LangGraph "
            "approval checkpoint. Create a new request to test."
        )

    # Resume the exact paused LangGraph workflow.
    with get_approval_graph() as graph:

        result = graph.invoke(
            Command(
                resume={
                    "decision": decision,
                    "reason": reviewer_reason
                }
            ),
            config={
                "configurable": {
                    "thread_id": thread_id
                }
            }
        )

    if result.get("__interrupt__"):
        raise RuntimeError(
            "The approval workflow is still waiting for input."
        )

    execution_status = result.get("execution_status")

    if decision == "APPROVE":

        if execution_status != "SIMULATED_SUCCESS":
            raise RuntimeError(
                "The approval graph did not complete simulated execution."
            )

        approval_status = "APPROVED"

    else:

        if execution_status != "REJECTED":
            raise RuntimeError(
                "The approval graph did not complete the rejection."
            )

        approval_status = "REJECTED"

    # Update the dashboard database only after the graph completes.
    import sqlite3
    from datetime import datetime, timezone
    from refund_store import get_connection

    reviewed_at = datetime.now(timezone.utc).isoformat()

    with get_connection() as connection:

        cursor = connection.execute(
            """
            UPDATE refund_requests
            SET approval_status = ?,
                execution_status = ?,
                reviewer_id = ?,
                reviewer_reason = ?,
                reviewed_at = ?
            WHERE request_id = ?
              AND approval_status = 'PENDING'
            """,
            (
                approval_status,
                execution_status,
                reviewer_id.strip(),
                reviewer_reason,
                reviewed_at,
                request_id
            )
        )

        if cursor.rowcount == 0:
            raise ValueError(
                "The request was already reviewed or updated."
            )

    return result


# ==================================================
# DASHBOARD
# ==================================================

pending_refunds = get_pending_refunds()

col1, col2 = st.columns(2)

with col1:
    st.metric("Pending requests", len(pending_refunds))

with col2:
    st.metric(
        "Requests requiring review",
        sum(
            1 for refund in pending_refunds
            if refund["eligibility_status"] == "ELIGIBLE"
        )
    )

st.divider()
st.header("Pending refund requests")


if not pending_refunds:

    st.success("There are currently no pending refund requests.")

else:

    for refund in pending_refunds:

        with st.expander(
            f"Order: {refund['order_id']} | "
            f"{refund['currency']} {refund['refund_amount']:.2f}"
        ):

            request_id = refund["request_id"]

            st.write("**Request ID:**", request_id)
            st.write("**Customer ID:**", refund["customer_unique_id"])
            st.write("**Reason:**", refund["reason"])

            st.write(
                "**Eligibility:**",
                refund["eligibility_status"]
            )

            st.write(
                "**Approval status:**",
                refund["approval_status"]
            )

            st.write("**Eligibility reasons:**")

            reasons = json.loads(refund["eligibility_reasons"])

            for reason in reasons:
                st.write(f"- {reason}")

            with st.expander("View refund policy evidence"):
                st.text(refund["policy_evidence"])

            if not refund.get("thread_id"):

                st.warning(
                    "Legacy request: no LangGraph approval checkpoint. "
                    "Create a new request to test the integrated workflow."
                )

            st.warning(
                "Approval runs simulated execution only. "
                "No real payment or refund is made."
            )

            approval_reason = st.text_area(
                "Reviewer comments",
                key=f"reason_{request_id}",
                placeholder="Enter your review comments..."
            )

            approve_col, reject_col = st.columns(2)

            with approve_col:

                if st.button(
                    "Approve refund",
                    key=f"approve_{request_id}",
                    type="primary",
                    disabled=(
                        refund["eligibility_status"] != "ELIGIBLE"
                        or not reviewer_id.strip()
                        or not refund.get("thread_id")
                    )
                ):

                    try:

                        result = process_decision(
                            refund,
                            "APPROVE",
                            approval_reason
                        )

                        st.success(
                            result.get(
                                "final_message",
                                "Refund approved."
                            )
                        )

                        st.rerun()

                    except Exception as error:
                        st.error(f"Approval failed: {error}")

            with reject_col:

                if st.button(
                    "Reject refund",
                    key=f"reject_{request_id}",
                    disabled=(
                        not reviewer_id.strip()
                        or not refund.get("thread_id")
                    )
                ):

                    try:

                        result = process_decision(
                            refund,
                            "REJECT",
                            approval_reason
                        )

                        st.warning(
                            result.get(
                                "final_message",
                                "Refund rejected."
                            )
                        )

                        st.rerun()

                    except Exception as error:
                        st.error(f"Rejection failed: {error}")


st.divider()

if st.button("Refresh requests"):
    st.rerun()
st.divider()

st.header("Refund History")

all_refunds = get_all_refunds()

if not all_refunds:
    st.info("No refund requests found.")
else:
    history_data = []

    for refund in all_refunds:
        history_data.append({
            "Request ID": refund["request_id"],
            "Order ID": refund["order_id"],
            "Reason": refund["reason"],
            "Amount": f'{refund["currency"]} {refund["refund_amount"]:.2f}',
            "Approval Status": refund["approval_status"],
            "Execution Status": refund["execution_status"],
            "Reviewer": refund["reviewer_id"] or "—",
            "Created At": refund["created_at"]
        })

    st.dataframe(
        history_data,
        use_container_width=True,
        hide_index=True
    )
