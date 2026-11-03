
from mcp.server.fastmcp import FastMCP
from order_tools import get_order_by_id

# Create the MCP server
mcp = FastMCP("ShopEase Order Server")


@mcp.tool()
def lookup_order(
    order_id: str,
    authenticated_customer_unique_id: str
) -> dict:
    """
    Retrieve historical order information.

    Requires the order ID and authenticated customer identity.
    Verifies customer ownership before returning order details.
    """

    return get_order_by_id(
        order_id,
        authenticated_customer_unique_id
    )


if __name__ == "__main__":
    mcp.run()
