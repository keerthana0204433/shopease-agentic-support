
from refund_tools import check_refund_eligibility


def prepare_refund_request(
    order: dict,
    customer_unique_id: str,
    reason: str,
    request_date: str,
    policy_context: str,
    item_unused: bool | None = None,
    original_packaging: bool | None = None,
    damage_reported_date: str | None = None,
    is_personalized_or_digital: bool | None = None,
):
    if not order:
        return {
            "success": False,
            "message": "Order information is unavailable."
        }

    if not policy_context or not policy_context.strip():
        return {
            "success": False,
            "message": "Refund policy evidence could not be retrieved."
        }

    eligibility = check_refund_eligibility(
        order=order,
        reason=reason,
        request_date=request_date,
        item_unused=item_unused,
        original_packaging=original_packaging,
        damage_reported_date=damage_reported_date,
        is_personalized_or_digital=is_personalized_or_digital,
    )

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

    refund_request = {
        "order_id": order["order_id"],
        "customer_unique_id": customer_unique_id,
        "reason": reason,
        "request_date": request_date,
        "refund_amount": round(item_subtotal, 2),
        "currency": "BRL",
        "eligibility_status": eligibility["status"],
        "eligibility_reasons": eligibility["reasons"],
        "policy_evidence": policy_context,
        "approval_status": "PENDING",
        "execution_status": "NOT_EXECUTED"
    }

    return {
        "success": True,
        "refund_request": refund_request
    }
