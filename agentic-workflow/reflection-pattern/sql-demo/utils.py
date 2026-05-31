"""
Database utilities for the SQL reflection pattern demo.

Provides helpers to:
  - Create a synthetic event-sourced transactions SQLite database.
  - Inspect the table schema.
  - Execute a SQL query and return a pandas DataFrame.
"""

import random
import sqlite3

import pandas as pd


# ---------------------------------------------------------------------------
# Database creation
# ---------------------------------------------------------------------------

def create_transactions_db(
    db_name: str = "products.db",
    n_products: int = 100,
    n_txns_per_product: int = 50,
) -> None:
    """
    Create an SQLite DB with a single 'transactions' table (event-sourced).
    All analytics must be derived from this table — no views or extra tables.

    Each row is one of four event types:
      insert       — initial stock with opening price
      restock      — positive qty_delta, no price
      sale         — negative qty_delta, price at time of sale
      price_update — qty_delta = 0, new unit_price
    """
    conn = sqlite3.connect(db_name)
    cur = conn.cursor()

    cur.execute("DROP TABLE IF EXISTS transactions")
    cur.execute("""
        CREATE TABLE transactions (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            product_id   INTEGER NOT NULL,
            product_name TEXT    NOT NULL,
            brand        TEXT    NOT NULL,
            category     TEXT    NOT NULL,
            color        TEXT    NOT NULL,
            action       TEXT    NOT NULL,
            qty_delta    INTEGER DEFAULT 0,
            unit_price   REAL,
            notes        TEXT,
            ts           DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    brands     = ["Nike", "Adidas", "Puma", "Reebok", "New Balance"]
    categories = ["shoes", "hoodie", "t-shirt", "hat", "backpack"]
    colors     = ["black", "white", "red", "blue", "green"]

    rng = random.Random(42)
    catalog = [
        (
            pid,
            f"{rng.choice(brands)} {rng.choice(categories)}",
            None,  # brand extracted below
            None,  # category extracted below
            rng.choice(colors),
            round(rng.uniform(20.0, 150.0), 2),
        )
        for pid in range(1, n_products + 1)
    ]
    # Fill brand / category from generated name
    catalog = [
        (pid, name, name.split()[0], name.split()[1], color, price)
        for pid, name, _, _, color, price in catalog
    ]

    for pid, name, brand, category, color, base_price in catalog:
        initial_stock = rng.randint(5, 50)
        cur.execute(
            """
            INSERT INTO transactions
                (product_id, product_name, brand, category, color,
                 action, qty_delta, unit_price, notes)
            VALUES (?, ?, ?, ?, ?, 'insert', ?, ?, ?)
            """,
            (pid, name, brand, category, color, initial_stock, base_price,
             f"Initial insert: stock={initial_stock}, price={base_price}"),
        )

        current_price = base_price
        for _ in range(n_txns_per_product - 1):
            event = rng.choices(
                ["restock", "sale", "price_update"],
                weights=[0.25, 0.60, 0.15],
                k=1,
            )[0]

            if event == "restock":
                qty = rng.randint(1, 25)
                cur.execute(
                    """
                    INSERT INTO transactions
                        (product_id, product_name, brand, category, color,
                         action, qty_delta, unit_price, notes)
                    VALUES (?, ?, ?, ?, ?, 'restock', ?, NULL, ?)
                    """,
                    (pid, name, brand, category, color, qty, f"Restock +{qty}"),
                )

            elif event == "sale":
                qty = -rng.randint(1, 10)
                cur.execute(
                    """
                    INSERT INTO transactions
                        (product_id, product_name, brand, category, color,
                         action, qty_delta, unit_price, notes)
                    VALUES (?, ?, ?, ?, ?, 'sale', ?, ?, ?)
                    """,
                    (pid, name, brand, category, color, qty, current_price,
                     f"Sale {-qty} units @ {current_price}"),
                )

            else:  # price_update
                delta = round(rng.uniform(-5.0, 5.0), 2)
                current_price = max(1.0, round(current_price + delta, 2))
                cur.execute(
                    """
                    INSERT INTO transactions
                        (product_id, product_name, brand, category, color,
                         action, qty_delta, unit_price, notes)
                    VALUES (?, ?, ?, ?, ?, 'price_update', 0, ?, ?)
                    """,
                    (pid, name, brand, category, color, current_price,
                     f"Price → {current_price}"),
                )

    conn.commit()
    conn.close()
    print(f"✅ Created '{db_name}' — {n_products} products × {n_txns_per_product} events.")


# ---------------------------------------------------------------------------
# Schema inspection
# ---------------------------------------------------------------------------

def get_schema(db_path: str) -> str:
    """Return a human-readable schema string for the 'transactions' table."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(transactions)")
    rows = cur.fetchall()
    conn.close()
    lines = ["Table name: transactions"] + [f"  {r[1]} ({r[2]})" for r in rows]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Query execution
# ---------------------------------------------------------------------------

def execute_sql(query: str, db_path: str) -> pd.DataFrame:
    """
    Execute a SELECT query against the transactions table and return a DataFrame.
    Strips markdown SQL fences if present.
    On error, returns a single-row DataFrame with an 'error' column.
    """
    q = query.strip().removeprefix("```sql").removesuffix("```").strip()
    conn = sqlite3.connect(db_path)
    try:
        return pd.read_sql_query(q, conn)
    except Exception as exc:
        return pd.DataFrame({"error": [str(exc)]})
    finally:
        conn.close()
