
from refund_workflow import prepare_refund_request


result = prepare_refund_request(
    order_id="e481f51cbdc54678b7cc49136f2d6af7",

    customer_unique_id="7c396fd4830fd04220f754e42b4e5bff",

    reason="Changed my mind",

    request_date="2017-10-20",

    item_unused=True,

    original_packaging=True,

    is_personalized_or_digital=False
)


print("\n========== REFUND REQUEST RESULT ==========")

print(result)

if result.get("success"):

    refund = result["refund_request"]

    print("\n========== REFUND SUMMARY ==========")

    print("Order ID:", refund["order_id"])
    print("Reason:", refund["reason"])
    print("Refund amount:", refund["refund_amount"], refund["currency"])
    print("Eligibility:", refund["eligibility_status"])
    print("Approval:", refund["approval_status"])
    print("Execution:", refund["execution_status"])
