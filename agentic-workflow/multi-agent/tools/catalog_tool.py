"""
Product catalog tool — exposes the in-memory sunglasses inventory to agents.

Usage::

    tool = ProductCatalogTool()
    rows = tool.run(max_items=5)
"""
from __future__ import annotations

from typing import Any

from data.inventory import create_inventory_dataframe

from .base_tool import BaseTool, ToolDefinition


class ProductCatalogTool(BaseTool):
    """
    Returns rows from the sunglasses product catalog.

    The catalog is created once at construction time and is immutable
    from the tool's perspective (agents read only).
    """

    def __init__(self, seed: int = 42) -> None:
        self._df = create_inventory_dataframe(seed=seed)

    # ── BaseTool interface ────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "product_catalog_tool"

    @property
    def definition(self) -> ToolDefinition:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (
                    "Retrieve products from the internal sunglasses inventory catalog. "
                    "Returns name, item_id, description, quantity_in_stock, and price."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "max_items": {
                            "type": "integer",
                            "description": "Maximum number of catalog items to return (default 10)",
                            "default": 10,
                        },
                    },
                },
            },
        }

    def run(self, max_items: int = 10) -> list[dict[str, Any]]:
        """
        Return up to *max_items* rows from the product catalog.

        Args:
            max_items: How many products to include.

        Returns:
            List of dicts with keys: name, item_id, description,
            quantity_in_stock, price.
        """
        return self._df.head(max_items).to_dict(orient="records")
