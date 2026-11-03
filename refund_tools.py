
from datetime import date, datetime, timedelta


def check_refund_eligibility(
    order: dict,
    reason: str,
    request_date: str,
    item_unused: bool | None = None,
    original_packaging: bool | None = None,
    damage_reported_date: str | None = None,
    is_personalized_or_digital: bool | None = None,
):
    """
    Checks a refund request against ShopEase's return policy.

    This function only assesses eligibility.
    It does NOT approve or execute a refund.
    """

    if not order:
        return {
            "status": "INELIGIBLE",
            "reasons": ["Order information is unavailable."]
        }

    # Parse dates
    try:
        requested_on = date.fromisoformat(request_date)

        delivered_value = order.get("delivered_date")

        if not delivered_value:
            return {
                "status": "NEEDS_REVIEW",
                "reasons": [
                    "The order does not have a recorded delivery date."
                ]
            }

        delivered_on = datetime.fromisoformat(
            delivered_value
        ).date()

    except (ValueError, TypeError):
        return {
            "status": "NEEDS_REVIEW",
            "reasons": ["A required date is missing or invalid."]
        }

    reasons = []
    review_reasons = []

    # Rule 1: Order must have been delivered
    if order.get("order_status") != "delivered":
        return {
            "status": "INELIGIBLE",
            "reasons": [
                "The order is not recorded as delivered."
            ]
        }

    # Rule 2: Return request must be within 30 calendar days
    days_since_delivery = (requested_on - delivered_on).days

    if days_since_delivery < 0:
        return {
            "status": "NEEDS_REVIEW",
            "reasons": [
                "The request date is before the recorded delivery date."
            ]
        }

    if days_since_delivery > 30:
        return {
            "status": "INELIGIBLE",
            "reasons": [
                "The return request is outside the 30-calendar-day window."
            ]
        }

    # Rule 3: Personalized and digital products are excluded
    if is_personalized_or_digital is True:
        return {
            "status": "INELIGIBLE",
            "reasons": [
                "Personalized and digital products are excluded "
                "under the return policy, subject to applicable law."
            ]
        }

    if is_personalized_or_digital is None:
        review_reasons.append(
            "Product type has not been verified."
        )

    # Rule 4: Item must be unused
    if item_unused is False:
        return {
            "status": "INELIGIBLE",
            "reasons": [
                "The item does not satisfy the unused-item requirement."
            ]
        }

    if item_unused is None:
        review_reasons.append(
            "Item condition has not been verified."
        )

    # Rule 5: Original packaging is required
    if original_packaging is False:
        return {
            "status": "INELIGIBLE",
            "reasons": [
                "Original packaging requirement is not satisfied."
            ]
        }

    if original_packaging is None:
        review_reasons.append(
            "Original packaging has not been verified."
        )

    # Rule 6: Damaged items must be reported within 7 days
    if "damage" in reason.lower() or "broken" in reason.lower():

        if not damage_reported_date:
            review_reasons.append(
                "The damage report date has not been provided."
            )
        else:
            try:
                reported_on = date.fromisoformat(
                    damage_reported_date
                )

                days_to_report = (reported_on - delivered_on).days

                if days_to_report < 0:
                    review_reasons.append(
                        "The damage report date is before delivery."
                    )

                elif days_to_report > 7:
                    return {
                        "status": "INELIGIBLE",
                        "reasons": [
                            "The damage was reported outside the "
                            "7-calendar-day reporting window."
                        ]
                    }

            except ValueError:
                review_reasons.append(
                    "The damage report date is invalid."
                )

    # Final assessment
    if review_reasons:
        return {
            "status": "NEEDS_REVIEW",
            "reasons": review_reasons
        }

    return {
        "status": "ELIGIBLE",
        "reasons": [
            "The available information satisfies the "
            "automated policy checks."
        ],
        "days_since_delivery": days_since_delivery
    }
