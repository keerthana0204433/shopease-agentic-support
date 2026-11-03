
from refund_tools import check_refund_eligibility


# Sample delivered order for testing
# This is a controlled test case, not a new Olist record.

test_order = {
    "order_id": "TEST-001",
    "order_status": "delivered",
    "delivered_date": "2026-10-01T12:00:00"
}


# Test 1: All required conditions are satisfied
result = check_refund_eligibility(
    order=test_order,
    reason="Changed my mind",
    request_date="2026-10-15",
    item_unused=True,
    original_packaging=True,
    is_personalized_or_digital=False
)

print("\nTEST 1: Eligible request")
print(result)


# Test 2: Request is outside the 30-day window
result = check_refund_eligibility(
    order=test_order,
    reason="Changed my mind",
    request_date="2026-11-05",
    item_unused=True,
    original_packaging=True,
    is_personalized_or_digital=False
)

print("\nTEST 2: Outside return window")
print(result)


# Test 3: Missing information
result = check_refund_eligibility(
    order=test_order,
    reason="Changed my mind",
    request_date="2026-10-15"
)

print("\nTEST 3: Needs review")
print(result)


# Test 4: Damaged item reported within 7 days
result = check_refund_eligibility(
    order=test_order,
    reason="Product arrived damaged",
    request_date="2026-10-05",
    item_unused=True,
    original_packaging=True,
    damage_reported_date="2026-10-05",
    is_personalized_or_digital=False
)

print("\nTEST 4: Damaged item")
print(result)


# Test 5: Personalized product
result = check_refund_eligibility(
    order=test_order,
    reason="Changed my mind",
    request_date="2026-10-15",
    item_unused=True,
    original_packaging=True,
    is_personalized_or_digital=True
)

print("\nTEST 5: Personalized product")
print(result)
