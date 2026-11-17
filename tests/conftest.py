"""Shared fixtures and builders for the test suite."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from src.config import AnalyticsConfig
from src.data_validator import validate_inventory
from src.inventory_engine import build_inventory_metrics

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_INVENTORY = ROOT / "data" / "sample_inventory.csv"
SAMPLE_HISTORY = ROOT / "data" / "sample_sales_history.csv"
ERROR_SAMPLE = ROOT / "data" / "sample_inventory_with_errors.csv"
AS_OF = date(2026, 11, 15)

DEFAULT_ROW = {
    "SKU_ID": "SKU-1",
    "Product_Name": "Widget",
    "Category": "Gadgets",
    "Current_Stock": 100,
    "Units_Sold": 60,
    "Unit_Price": 10.0,
    "Reorder_Level": 20,
    "Supplier_Lead_Time": 7,
    "Last_Restock_Date": "2026-10-01",
}


def make_raw(rows: list[dict]) -> pd.DataFrame:
    """Raw (all-text) frame like ``load_csv`` would return. Missing keys use defaults."""
    records = []
    for i, overrides in enumerate(rows):
        record = {**DEFAULT_ROW, "SKU_ID": f"SKU-{i + 1}", **overrides}
        records.append({k: (None if v is None else str(v)) for k, v in record.items()})
    return pd.DataFrame(records, dtype=object)


def make_metrics(rows: list[dict], config: AnalyticsConfig | None = None) -> pd.DataFrame:
    """Validate rows and run the full analytics pipeline."""
    config = config or AnalyticsConfig(as_of_date=AS_OF)
    result = validate_inventory(make_raw(rows), reference_date=AS_OF)
    assert result.is_valid, [i.message for i in result.errors]
    return build_inventory_metrics(result.data, config)


@pytest.fixture
def config() -> AnalyticsConfig:
    return AnalyticsConfig(as_of_date=AS_OF)


@pytest.fixture(scope="session")
def sample_metrics() -> pd.DataFrame:
    from src.data_loader import load_csv
    from src.data_validator import validate_sales_history

    cfg = AnalyticsConfig(as_of_date=AS_OF)
    inventory = validate_inventory(load_csv(SAMPLE_INVENTORY), reference_date=AS_OF)
    history = validate_sales_history(load_csv(SAMPLE_HISTORY), inventory.data["SKU_ID"], AS_OF)
    return build_inventory_metrics(inventory.data, cfg, history.data)
