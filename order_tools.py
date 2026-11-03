
from load_olist_data import load_csv


def get_order_by_id(order_id: str, authenticated_customer_unique_id: str):
    """
    Retrieve an order and its related information.

    The authenticated customer identity must match the
    customer_unique_id associated with the requested order.
    """

    if not order_id or not authenticated_customer_unique_id:
        return {
            "success": False,
            "message": "Order ID and customer identity are required."
        }

    # Load only the datasets needed for order lookup.
    orders = load_csv("olist_orders_dataset.csv")
    customers = load_csv("olist_customers_dataset.csv")
    order_items = load_csv("olist_order_items_dataset.csv")
    products = load_csv("olist_products_dataset.csv")
    payments = load_csv("olist_order_payments_dataset.csv")
    reviews = load_csv("olist_order_reviews_dataset.csv")
    translations = load_csv("product_category_name_translation.csv")

    # Find the requested order.
    order = next(
        (
            row for row in orders
            if row["order_id"].lower() == order_id.strip().lower()
        ),
        None
    )

    if order is None:
        return {
            "success": False,
            "message": "Order not found."
        }

    # Find the customer associated with the order.
    customer = next(
        (
            row for row in customers
            if row["customer_id"] == order["customer_id"]
        ),
        None
    )

    if customer is None:
        return {
            "success": False,
            "message": "Customer information not found."
        }

    # Verify customer ownership before retrieving order details.
    if customer["customer_unique_id"] != authenticated_customer_unique_id:
        return {
            "success": False,
            "message": "You are not authorized to access this order."
        }

    # Create lookup dictionaries for efficient joins.
    product_lookup = {
        row["product_id"]: row
        for row in products
    }

    category_lookup = {
        row["product_category_name"]: row["product_category_name_english"]
        for row in translations
    }

    # Retrieve all items belonging to the order.
    items = []

    for item in order_items:
        if item["order_id"] == order["order_id"]:

            product = product_lookup.get(item["product_id"], {})

            category = product.get("product_category_name", "")

            items.append({
                "product_id": item["product_id"],
                "category": category_lookup.get(category, category),
                "seller_id": item["seller_id"],
                "price": item["price"],
                "freight_value": item["freight_value"]
            })

    # Retrieve payments.
    order_payments = [
        {
            "payment_type": payment["payment_type"],
            "payment_installments": payment["payment_installments"],
            "payment_value": payment["payment_value"]
        }
        for payment in payments
        if payment["order_id"] == order["order_id"]
    ]

    # Retrieve reviews.
    order_reviews = [
        {
            "review_score": review["review_score"],
            "review_comment_title": review["review_comment_title"],
            "review_comment_message": review["review_comment_message"],
            "review_creation_date": review["review_creation_date"]
        }
        for review in reviews
        if review["order_id"] == order["order_id"]
    ]

    # Return structured order information.
    return {
        "success": True,
        "order": {
            "order_id": order["order_id"],
            "order_status": order["order_status"],
            "purchase_date": order["order_purchase_timestamp"],
            "approved_at": order["order_approved_at"],
            "carrier_date": order["order_delivered_carrier_date"],
            "delivered_date": order["order_delivered_customer_date"],
            "estimated_delivery_date": order["order_estimated_delivery_date"],
            "items": items,
            "payments": order_payments,
            "reviews": order_reviews
        }
    }
