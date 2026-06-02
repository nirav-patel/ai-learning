"""
Sunglasses inventory data layer.

Provides factory functions that return clean pandas DataFrames for:
  - Product catalog  (inventory)
  - Cash register    (transactions)
  - Sales ledger     (audit log)

No side-effects; all state is caller-owned.
"""
from __future__ import annotations

import random
from datetime import datetime

import pandas as pd

# ─── Catalog ──────────────────────────────────────────────────────────────────

CATALOG_SEED = 42

CATALOG_ITEMS: list[dict] = [
    {
        "name": "Aviator",
        "item_id": "SG001",
        "description": (
            "Originally designed for pilots, these teardrop-shaped lenses with thin metal "
            "frames offer timeless appeal. Large lenses provide excellent coverage while the "
            "lightweight construction ensures comfort during long wear."
        ),
    },
    {
        "name": "Wayfarer",
        "item_id": "SG002",
        "description": (
            "Featuring thick, angular frames that make a statement, these sunglasses combine "
            "retro charm with modern edge. The rectangular lenses and sturdy acetate "
            "construction create a confident look."
        ),
    },
    {
        "name": "Mystique",
        "item_id": "SG003",
        "description": (
            "Inspired by 1950s glamour, these frames sweep upward at the outer corners to "
            "create an elegant, feminine silhouette. Subtle curves and embellished temples "
            "add sophistication to any outfit."
        ),
    },
    {
        "name": "Sport",
        "item_id": "SG004",
        "description": (
            "Designed for active lifestyles, these wraparound sunglasses feature a single "
            "curved lens that provides maximum coverage and wind protection. Lightweight, "
            "flexible frames include rubber grips."
        ),
    },
    {
        "name": "Round",
        "item_id": "SG005",
        "description": (
            "Circular lenses set in minimalist frames create a thoughtful, artistic "
            "appearance, evoking a scholarly or creative vibe while remaining "
            "effortlessly stylish."
        ),
    },
]


def create_inventory_dataframe(seed: int = CATALOG_SEED) -> pd.DataFrame:
    """
    Return the sunglasses product catalog as a DataFrame.

    Columns: name, item_id, description, quantity_in_stock, price

    Args:
        seed: Random seed used for reproducible stock quantities / prices.

    Returns:
        DataFrame with one row per product.
    """
    rng = random.Random(seed)
    rows = [
        {
            **item,
            "quantity_in_stock": rng.randint(3, 25),
            "price": rng.randint(75, 150),
        }
        for item in CATALOG_ITEMS
    ]
    return pd.DataFrame(rows, columns=["name", "item_id", "description", "quantity_in_stock", "price"])


# ─── Transactions ─────────────────────────────────────────────────────────────


def create_transaction_dataframe(opening_balance: float = 500.00) -> pd.DataFrame:
    """
    Return a cash-register transaction DataFrame seeded with an opening balance.

    Columns: transaction_id, customer_name, transaction_summary,
             transaction_amount, balance_after_transaction

    Args:
        opening_balance: Starting cash balance for the day.

    Returns:
        DataFrame with a single opening-balance row.
    """
    return pd.DataFrame(
        {
            "transaction_id": ["TXN001"],
            "customer_name": ["OPENING_BALANCE"],
            "transaction_summary": ["Daily opening register balance"],
            "transaction_amount": [opening_balance],
            "balance_after_transaction": [opening_balance],
        }
    )


# ─── Ledger ───────────────────────────────────────────────────────────────────


def create_ledger_dataframe() -> pd.DataFrame:
    """
    Return an empty sales-ledger DataFrame.

    Columns: transaction_date, item_id, quantity, transaction_type
    """
    return pd.DataFrame(
        columns=["transaction_date", "item_id", "quantity", "transaction_type"]
    )


# ─── Inventory helpers ────────────────────────────────────────────────────────


def get_item_names(df: pd.DataFrame) -> list[str]:
    """Return a list of all product names in the inventory DataFrame."""
    return df["name"].tolist()


def check_stock_by_name(df: pd.DataFrame, item_name: str) -> int:
    """
    Return the stock quantity for *item_name* (case-insensitive).

    Returns:
        Quantity in stock, or ``-1`` if the item is not found.
    """
    match = df[df["name"].str.lower() == item_name.lower()]
    if match.empty:
        return -1
    return int(match.iloc[0]["quantity_in_stock"])


def update_stock(
    df: pd.DataFrame,
    item_name: str,
    transaction_type: str,
    quantity: int,
) -> bool:
    """
    Mutate *df* in-place to reflect a sale or return.

    Args:
        df: Inventory DataFrame (mutated in-place).
        item_name: Case-insensitive product name.
        transaction_type: ``"sale"`` decrements stock; ``"return"`` increments it.
        quantity: Units to add / remove (must be > 0).

    Returns:
        ``True`` on success, ``False`` if inputs are invalid or item not found.
    """
    if quantity <= 0:
        return False
    if transaction_type.lower() not in {"sale", "return"}:
        return False

    mask = df["name"].str.lower() == item_name.lower()
    if not mask.any():
        return False

    if transaction_type.lower() == "sale":
        df.loc[mask, "quantity_in_stock"] -= quantity
    else:
        df.loc[mask, "quantity_in_stock"] += quantity

    df.loc[mask, "quantity_in_stock"] = df.loc[mask, "quantity_in_stock"].clip(lower=0)
    return True


def record_ledger_entry(
    ledger: pd.DataFrame,
    item_id: str,
    quantity: int,
    transaction_type: str,
) -> pd.DataFrame:
    """
    Append a new entry to the sales ledger and return the updated DataFrame.

    Args:
        ledger: Existing ledger DataFrame (not mutated; new DataFrame returned).
        item_id: Product identifier.
        quantity: Units transacted.
        transaction_type: ``"sale"`` or ``"return"``.

    Returns:
        New DataFrame with the appended row.
    """
    new_row = pd.DataFrame(
        [
            {
                "transaction_date": datetime.now().isoformat(),
                "item_id": item_id,
                "quantity": quantity,
                "transaction_type": transaction_type,
            }
        ]
    )
    return pd.concat([ledger, new_row], ignore_index=True)
