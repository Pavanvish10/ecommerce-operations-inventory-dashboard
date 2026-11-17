"""CSV report generation and filter-respecting exports."""

from __future__ import annotations

import io
from datetime import date

import pandas as pd
import pytest

from src.export_manager import REPORT_COLUMNS, prepare_report, report_filename, to_csv_bytes
from src.filters import FilterCriteria, apply_filters


def read_back(table: pd.DataFrame) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(to_csv_bytes(table)), dtype={"SKU_ID": str})


def test_full_report_columns_and_rows(sample_metrics):
    table = prepare_report(sample_metrics, "full")
    assert len(table) == len(sample_metrics)
    assert list(table.columns) == [c for c in REPORT_COLUMNS["full"] if c in sample_metrics.columns]
    assert table.columns[0] == "SKU_ID"


def test_filtered_export_contains_exactly_filtered_records(sample_metrics):
    criteria = FilterCriteria(categories=("Books", "Grocery"), health_classes=("Healthy", "Moderate"))
    filtered = apply_filters(sample_metrics, criteria)
    exported = read_back(prepare_report(filtered, "filtered"))
    assert len(exported) == len(filtered) > 0
    assert set(exported["SKU_ID"]) == set(filtered["SKU_ID"])
    assert set(exported["Category"]) <= {"Books", "Grocery"}
    assert set(exported["Health_Class"]) <= {"Healthy", "Moderate"}


def test_risk_report_excludes_healthy_and_is_ranked(sample_metrics):
    table = prepare_report(sample_metrics, "risk")
    assert "Healthy" not in set(table["Risk_Category"])
    assert len(table) == (sample_metrics["Risk_Category"] != "Healthy").sum()
    assert {"Recommended_Action", "Action_Rationale"} <= set(table.columns)


def test_reorder_report_only_needed_items_most_urgent_first(sample_metrics):
    table = prepare_report(sample_metrics, "reorder")
    assert len(table) == int(sample_metrics["Needs_Reorder"].sum())
    priority = {"Critical": 0, "High": 1, "Medium": 2, "Healthy": 3}
    ranks = table["Alert_Level"].map(priority)
    assert ranks.is_monotonic_increasing
    assert (table["Reorder_Quantity"] > 0).all()


def test_reorder_report_respects_filters(sample_metrics):
    books = apply_filters(sample_metrics, FilterCriteria(categories=("Books",)))
    table = prepare_report(books, "reorder")
    assert set(table["Category"]) <= {"Books"}


def test_csv_round_trip_preserves_values_and_types(sample_metrics):
    table = prepare_report(sample_metrics, "full")
    back = read_back(table)
    assert list(back.columns) == list(table.columns)
    assert len(back) == len(table)
    assert pd.api.types.is_numeric_dtype(back["Current_Stock"])
    assert pd.api.types.is_numeric_dtype(back["Health_Score"])
    pd.testing.assert_series_equal(back["Current_Stock"], table["Current_Stock"], check_dtype=False)
    assert back["Needs_Reorder"].isin([True, False]).all()


def test_dates_are_iso_formatted(sample_metrics):
    back = read_back(prepare_report(sample_metrics, "full"))
    dates = back["Last_Restock_Date"].dropna().astype(str)
    assert dates.str.fullmatch(r"\d{4}-\d{2}-\d{2}").all()
    stockout = back["Estimated_Stockout_Date"].dropna().astype(str)
    assert stockout.str.fullmatch(r"\d{4}-\d{2}-\d{2}").all()


def test_infinite_cover_is_explicit(sample_metrics):
    table = prepare_report(sample_metrics, "full")
    dead = sample_metrics["Units_Sold"].eq(0) & sample_metrics["Current_Stock"].gt(0)
    assert dead.any()
    assert (table.loc[dead.to_numpy(), "Days_Of_Inventory"] == "No demand").all()
    assert "inf" not in to_csv_bytes(table).decode()


def test_derived_values_are_rounded(sample_metrics):
    table = prepare_report(sample_metrics, "full")
    assert (table["Avg_Daily_Sales"].round(3) == table["Avg_Daily_Sales"]).all()


def test_export_of_empty_selection_has_header_only(sample_metrics):
    empty = apply_filters(sample_metrics, FilterCriteria(search="no-such-sku"))
    content = to_csv_bytes(prepare_report(empty, "full")).decode().strip().splitlines()
    assert len(content) == 1
    assert content[0].startswith("SKU_ID,")


def test_unknown_report_type(sample_metrics):
    with pytest.raises(ValueError):
        prepare_report(sample_metrics, "weekly")


def test_report_filename():
    assert report_filename("risk", True, date(2026, 11, 17)) == "inventory_risk_report_filtered_2026-11-17.csv"
    assert report_filename("full", False, date(2026, 11, 17)) == "inventory_full_report_2026-11-17.csv"
