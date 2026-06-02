"""
Unit tests for the data/inventory.py module.
"""
from __future__ import annotations

import pandas as pd
import pytest

from data.inventory import (
    CATALOG_ITEMS,
    check_stock_by_name,
    create_inventory_dataframe,
    create_ledger_dataframe,
    create_transaction_dataframe,
    get_item_names,
    record_ledger_entry,
    update_stock,
)


class TestCreateInventoryDataframe:
    def test_returns_dataframe(self):
        df = create_inventory_dataframe()
        assert isinstance(df, pd.DataFrame)

    def test_expected_columns(self):
        df = create_inventory_dataframe()
        assert set(df.columns) == {"name", "item_id", "description", "quantity_in_stock", "price"}

    def test_row_count_matches_catalog(self):
        df = create_inventory_dataframe()
        assert len(df) == len(CATALOG_ITEMS)

    def test_seed_reproducibility(self):
        df1 = create_inventory_dataframe(seed=42)
        df2 = create_inventory_dataframe(seed=42)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_differ(self):
        df1 = create_inventory_dataframe(seed=1)
        df2 = create_inventory_dataframe(seed=99)
        assert not df1["quantity_in_stock"].equals(df2["quantity_in_stock"])

    def test_quantities_in_range(self):
        df = create_inventory_dataframe()
        assert (df["quantity_in_stock"] >= 3).all()
        assert (df["quantity_in_stock"] <= 25).all()

    def test_prices_in_range(self):
        df = create_inventory_dataframe()
        assert (df["price"] >= 75).all()
        assert (df["price"] <= 150).all()


class TestCreateTransactionDataframe:
    def test_single_opening_row(self):
        df = create_transaction_dataframe()
        assert len(df) == 1

    def test_opening_balance_default(self):
        df = create_transaction_dataframe()
        assert df.iloc[0]["balance_after_transaction"] == 500.00

    def test_custom_opening_balance(self):
        df = create_transaction_dataframe(opening_balance=1000.0)
        assert df.iloc[0]["balance_after_transaction"] == 1000.0

    def test_columns_present(self):
        df = create_transaction_dataframe()
        for col in ("transaction_id", "customer_name", "transaction_summary", "transaction_amount", "balance_after_transaction"):
            assert col in df.columns


class TestCreateLedgerDataframe:
    def test_returns_empty_dataframe(self):
        df = create_ledger_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_expected_columns(self):
        df = create_ledger_dataframe()
        assert set(df.columns) == {"transaction_date", "item_id", "quantity", "transaction_type"}


class TestGetItemNames:
    def test_returns_list(self, inventory_df):
        names = get_item_names(inventory_df)
        assert isinstance(names, list)

    def test_all_catalog_names_present(self, inventory_df):
        names = get_item_names(inventory_df)
        for item in CATALOG_ITEMS:
            assert item["name"] in names


class TestCheckStockByName:
    def test_found_item(self, inventory_df):
        qty = check_stock_by_name(inventory_df, "Aviator")
        assert qty >= 0

    def test_case_insensitive(self, inventory_df):
        qty_lower = check_stock_by_name(inventory_df, "aviator")
        qty_upper = check_stock_by_name(inventory_df, "AVIATOR")
        assert qty_lower == qty_upper

    def test_not_found_returns_minus_one(self, inventory_df):
        qty = check_stock_by_name(inventory_df, "NonExistentModel")
        assert qty == -1


class TestUpdateStock:
    def test_sale_decrements_stock(self, inventory_df):
        initial = check_stock_by_name(inventory_df, "Aviator")
        result = update_stock(inventory_df, "Aviator", "sale", 1)
        assert result is True
        assert check_stock_by_name(inventory_df, "Aviator") == initial - 1

    def test_return_increments_stock(self, inventory_df):
        initial = check_stock_by_name(inventory_df, "Wayfarer")
        update_stock(inventory_df, "Wayfarer", "return", 2)
        assert check_stock_by_name(inventory_df, "Wayfarer") == initial + 2

    def test_stock_does_not_go_below_zero(self, inventory_df):
        # Force stock to 0
        update_stock(inventory_df, "Sport", "sale", 9999)
        assert check_stock_by_name(inventory_df, "Sport") == 0

    def test_invalid_quantity_returns_false(self, inventory_df):
        assert update_stock(inventory_df, "Aviator", "sale", 0) is False
        assert update_stock(inventory_df, "Aviator", "sale", -5) is False

    def test_invalid_type_returns_false(self, inventory_df):
        assert update_stock(inventory_df, "Aviator", "steal", 1) is False

    def test_unknown_item_returns_false(self, inventory_df):
        assert update_stock(inventory_df, "UnknownModel", "sale", 1) is False

    def test_case_insensitive(self, inventory_df):
        initial = check_stock_by_name(inventory_df, "Round")
        update_stock(inventory_df, "round", "sale", 1)
        assert check_stock_by_name(inventory_df, "Round") == initial - 1


class TestRecordLedgerEntry:
    def test_appends_row(self):
        ledger = create_ledger_dataframe()
        new_ledger = record_ledger_entry(ledger, "SG001", 2, "sale")
        assert len(new_ledger) == 1

    def test_original_not_mutated(self):
        ledger = create_ledger_dataframe()
        record_ledger_entry(ledger, "SG001", 1, "sale")
        assert len(ledger) == 0  # original unchanged

    def test_multiple_entries(self):
        ledger = create_ledger_dataframe()
        ledger = record_ledger_entry(ledger, "SG001", 2, "sale")
        ledger = record_ledger_entry(ledger, "SG002", 1, "return")
        assert len(ledger) == 2
        assert ledger.iloc[1]["item_id"] == "SG002"
