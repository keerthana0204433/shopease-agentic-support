
from datetime import date

from order_tools import get_order_by_id
from refund_tools import check_refund_eligibility


def prepare_refund_request(
    order_id: str,
    customer_unique_id: str,
    reason: str,
    request_date: str,
    item_unused: bool | None = None,
    original_packaging: bool | None = None,
    damage_reported_date: str | None = None,
    is_personalized_or_digital: bool | None = None,
):
    """
    Fetches an order, checks refund eligibility,
    and prepares a request for human approval.

    This function does not approve or execute refunds.
    """

    # 1. Fetch and authorize the order
    order_result = get_order_by_id(
        order_id=order_id,
        authenticated_customer_unique_id=customer_unique_id
    )

    if not order_result.get("success"):
        return {
            "success": False,
            "message": order_result.get(
                "message",
                "Unable to retrieve order."
            )
        }

    order = order_result["order"]

    # 2. Check eligibility
    eligibility = check_refund_eligibility(
        order=order,
        reason=reason,
        request_date=request_date,
        item_unused=item_unused,
        original_packaging=original_packaging,
        damage_reported_date=damage_reported_date,
        is_personalized_or_digital=is_personalized_or_digital,
    )

    # 3. Calculate item subtotal
    try:
        item_subtotal = sum(
            float(item["price"])
            for item in order.get("items", [])
        )
    except (TypeError, ValueError, KeyError):
        return {
            "success": False,
            "message": "Unable to calculate the order amount."
        }

    # 4. Prepare refund request
    refund_request = {
        "order_id": order["order_id"],
        "customer_unique_id": customer_unique_id,
        "reason": reason,
        "request_date": request_date,
        "refund_amount": round(item_subtotal, 2),
        "currency": "BRL",
        "eligibility_status": eligibility["status"],
        "eligibility_reasons": eligibility["reasons"],
        "approval_status": "PENDING",
        "execution_status": "NOT_EXECUTED"
    }

    return {
        "success": True,
        "refund_request": refund_request
    }
