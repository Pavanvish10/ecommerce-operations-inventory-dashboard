"""Search, combined filtering and sorting."""

from __future__ import annotations

import pandas as pd
import pytest

from src.filters import SORT_OPTIONS, FilterCriteria, apply_filters, sort_products
from tests.conftest import make_metrics


@pytest.fixture
def m(config) -> pd.DataFrame:
    return make_metrics([
        {"SKU_ID": "ELE-1", "Product_Name": "Wireless Earbuds", "Category": "Electronics",
         "Current_Stock": 0, "Units_Sold": 90},
        {"SKU_ID": "ELE-2", "Product_Name": "USB-C (2m) Cable", "Category": "Electronics",
         "Current_Stock": 500, "Units_Sold": 30},
        {"SKU_ID": "BKS-1", "Product_Name": "Python Guide", "Category": "Books",
         "Current_Stock": 60, "Units_Sold": 30},
        {"SKU_ID": "BKS-2", "Product_Name": "Mystery Novel", "Category": "Books",
         "Current_Stock": 40, "Units_Sold": 0},
        {"SKU_ID": "BAD-1", "Product_Name": "Broken Row", "Category": "Books",
         "Current_Stock": "??", "Units_Sold": 10},
    ], config)


def skus(frame: pd.DataFrame) -> list[str]:
    return sorted(frame["SKU_ID"])


def test_no_criteria_returns_everything(m):
    assert len(apply_filters(m, FilterCriteria())) == len(m)
    assert FilterCriteria().active_count == 0


@pytest.mark.parametrize("term, expected", [
    ("ele-1", ["ELE-1"]),            # SKU, case-insensitive
    ("EARBUDS", ["ELE-1"]),          # product name
    ("  python ", ["BKS-1"]),        # whitespace trimmed
    ("(2m)", ["ELE-2"]),             # regex characters are literal
    ("zzz", []),
])
def test_search(m, term, expected):
    assert skus(apply_filters(m, FilterCriteria(search=term))) == expected


def test_category_filter(m):
    assert skus(apply_filters(m, FilterCriteria(categories=("Electronics",)))) == ["ELE-1", "ELE-2"]


def test_combined_filters_are_anded(m):
    criteria = FilterCriteria(categories=("Books",), stock_statuses=("In Stock",), search="o")
    result = apply_filters(m, criteria)
    assert skus(result) == ["BKS-1", "BKS-2"]
    assert criteria.active_count == 3
    narrower = FilterCriteria(categories=("Books",), risk_categories=("Dead Stock",))
    assert skus(apply_filters(m, narrower)) == ["BKS-2"]


def test_health_and_risk_filters(m):
    assert skus(apply_filters(m, FilterCriteria(health_classes=("Critical",)))) == ["ELE-1"]
    assert skus(apply_filters(m, FilterCriteria(health_classes=("Insufficient Data",)))) == ["BAD-1"]
    assert "ELE-1" in skus(apply_filters(m, FilterCriteria(risk_levels=("Critical",))))
    assert skus(apply_filters(m, FilterCriteria(alert_levels=("Critical",)))) == ["ELE-1"]


def test_abc_filter(m):
    a_class = apply_filters(m, FilterCriteria(abc_classes=("A",)))
    assert len(a_class) >= 1
    assert (a_class["ABC_Class"] == "A").all()


def test_health_range_excludes_unscored_rows(m):
    full = apply_filters(m, FilterCriteria(health_score_range=(0.0, 100.0)))
    assert "BAD-1" in skus(full)
    partial = apply_filters(m, FilterCriteria(health_score_range=(0.0, 99.0)))
    assert "BAD-1" not in skus(partial)
    assert partial["Health_Score"].le(99).all()
    boundary = apply_filters(m, FilterCriteria(health_score_range=(0.0, 0.0)))
    assert skus(boundary) == ["ELE-1"]  # inclusive bounds


def test_filtering_does_not_modify_input(m):
    before = m.copy()
    apply_filters(m, FilterCriteria(search="x", categories=("Books",)))
    pd.testing.assert_frame_equal(m, before)


def test_no_matches_returns_empty_frame_with_columns(m):
    result = apply_filters(m, FilterCriteria(search="no-such-product"))
    assert result.empty
    assert list(result.columns) == list(m.columns)


@pytest.mark.parametrize("option, first", [
    ("Highest sales", "ELE-1"),
    ("Lowest sales", "BKS-2"),
    ("Highest inventory", "ELE-2"),
    ("Lowest inventory", "ELE-1"),
    ("Highest risk", "ELE-1"),
    ("Lowest health score", "ELE-1"),
])
def test_sort_options(m, option, first):
    assert sort_products(m, option)["SKU_ID"].iloc[0] == first


@pytest.mark.parametrize("option", list(SORT_OPTIONS))
def test_missing_values_sort_last(m, option):
    result = sort_products(m, option)
    columns = SORT_OPTIONS[option][0]
    if m.loc[m["SKU_ID"] == "BAD-1", columns].isna().any(axis=None):
        assert result["SKU_ID"].iloc[-1] == "BAD-1"


def test_unknown_sort_option(m):
    with pytest.raises(ValueError):
        sort_products(m, "Alphabetical")
