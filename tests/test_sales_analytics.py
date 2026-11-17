"""Sales velocity, ABC classification, category performance and demand trends."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from src.config import AnalyticsConfig
from src.sales_analytics import (
    INSUFFICIENT_HISTORY,
    category_performance,
    classify_abc,
    compute_demand_trends,
    days_of_inventory,
    safe_divide,
    sales_time_series,
    slowest_movers,
    top_products,
)
from tests.conftest import AS_OF, make_metrics


def test_velocity_from_assumed_window(config):
    row = make_metrics([{"Units_Sold": 90, "Unit_Price": 2.5}], config).iloc[0]
    assert row["Avg_Daily_Sales"] == pytest.approx(3.0)
    assert row["Avg_Weekly_Sales"] == pytest.approx(21.0)
    assert row["Avg_Monthly_Sales"] == pytest.approx(90.0)
    assert row["Revenue"] == pytest.approx(225.0)
    assert row["Velocity_Basis"] == "Assumed 30-day window"


def test_configurable_window_changes_velocity():
    cfg = AnalyticsConfig(as_of_date=AS_OF, sales_window_days=90)
    row = make_metrics([{"Units_Sold": 90}], cfg).iloc[0]
    assert row["Avg_Daily_Sales"] == pytest.approx(1.0)
    assert row["Velocity_Basis"] == "Assumed 90-day window"


def test_days_of_inventory_edge_cases():
    stock = pd.Series([10.0, 0.0, 10.0, 0.0, np.nan, 10.0])
    daily = pd.Series([2.0, 2.0, 0.0, 0.0, 1.0, np.nan])
    result = days_of_inventory(stock, daily)
    assert result.iloc[0] == 5.0
    assert result.iloc[1] == 0.0
    assert np.isinf(result.iloc[2])
    assert result.iloc[3] == 0.0
    assert np.isnan(result.iloc[4]) and np.isnan(result.iloc[5])


def test_safe_divide_never_raises():
    result = safe_divide(pd.Series([1.0, 1.0, 0.0]), pd.Series([0.0, np.nan, 0.0]))
    assert result.isna().all()


@pytest.mark.parametrize(
    "revenue, expected",
    [
        ([80, 15, 5], ["A", "B", "C"]),
        ([50, 30, 15, 5], ["A", "A", "B", "C"]),
        ([100], ["A"]),
        ([0, 0], ["C", "C"]),
        ([90, 10, 0], ["A", "B", "C"]),
    ],
)
def test_abc_classification(revenue, expected):
    assert classify_abc(pd.Series(revenue, dtype=float)).tolist() == expected


def test_abc_preserves_unknown_revenue():
    result = classify_abc(pd.Series([100.0, np.nan]))
    assert result.iloc[0] == "A"
    assert pd.isna(result.iloc[1])


def test_category_performance(config):
    m = make_metrics([
        {"Category": "A", "Units_Sold": 30, "Unit_Price": 10, "Current_Stock": 5},
        {"Category": "A", "Units_Sold": 60, "Unit_Price": 10, "Current_Stock": 5},
        {"Category": "B", "Units_Sold": 30, "Unit_Price": 10, "Current_Stock": 20},
    ], config)
    perf = category_performance(m).set_index("Category")
    assert perf.loc["A", "Products"] == 2
    assert perf.loc["A", "Revenue"] == 900
    assert perf.loc["A", "Avg_Daily_Sales"] == pytest.approx(3.0)
    assert perf.loc["B", "Inventory_Units"] == 20
    assert perf["Revenue_Share"].sum() == pytest.approx(1.0)
    assert list(perf.index) == ["A", "B"]  # sorted by revenue


def test_category_performance_empty():
    assert category_performance(pd.DataFrame()).empty


def test_top_products_deterministic_ties(config):
    m = make_metrics([{"SKU_ID": s, "Units_Sold": 10} for s in ["C", "A", "B"]], config)
    assert top_products(m, "Units_Sold", 2)["SKU_ID"].tolist() == ["A", "B"]


def test_slowest_movers_excludes_out_of_stock(config):
    m = make_metrics([
        {"SKU_ID": "ZERO-STOCK", "Units_Sold": 0, "Current_Stock": 0},
        {"SKU_ID": "DEAD", "Units_Sold": 0, "Current_Stock": 10},
        {"SKU_ID": "SLOW", "Units_Sold": 3, "Current_Stock": 10},
        {"SKU_ID": "FAST", "Units_Sold": 300, "Current_Stock": 10},
    ], config)
    assert slowest_movers(m, 2)["SKU_ID"].tolist() == ["DEAD", "SLOW"]


def _history(daily_values: dict[str, list[float]], start=date(2026, 9, 1)) -> pd.DataFrame:
    rows = []
    for sku, values in daily_values.items():
        for offset, units in enumerate(values):
            rows.append({"SKU_ID": sku, "Date": pd.Timestamp(start + timedelta(days=offset)),
                         "Units_Sold": float(units)})
    return pd.DataFrame(rows)


def test_demand_trends_classification(config):
    days = 28
    history = _history({
        "UP": [10 + i for i in range(days)],
        "DOWN": [40 - i for i in range(days)],
        "FLAT": [10] * days,
        "NONE": [0] * days,
    })
    trends = compute_demand_trends(history, config).set_index("SKU_ID")
    assert trends.loc["UP", "Demand_Trend"] == "Growing"
    assert trends.loc["DOWN", "Demand_Trend"] == "Declining"
    assert trends.loc["FLAT", "Demand_Trend"] == "Stable"
    assert trends.loc["FLAT", "Trend_Pct_Per_Week"] == pytest.approx(0.0)
    assert trends.loc["NONE", "Demand_Trend"] == INSUFFICIENT_HISTORY  # no demand → no trend
    # UP: slope 1/day, mean 23.5 → 7/23.5 = 29.8 %/week
    assert trends.loc["UP", "Trend_Pct_Per_Week"] == pytest.approx(7 / 23.5 * 100)


def test_short_history_is_insufficient(config):
    trends = compute_demand_trends(_history({"A": [1, 2, 3, 4, 5]}), config)
    assert (trends["Demand_Trend"] == INSUFFICIENT_HISTORY).all()


def test_missing_days_count_as_zero_sales(config):
    history = _history({"A": [5] * 20, "B": [5]})  # B only sold on day 1
    trends = compute_demand_trends(history, config).set_index("SKU_ID")
    assert trends.loc["B", "History_Days"] == 20
    assert trends.loc["B", "History_Avg_Daily"] == pytest.approx(5 / 20)


def test_weekly_series_drops_partial_weeks():
    # 2026-09-07 is a Monday; 17 days → 2 complete weeks + 3 days
    history = _history({"A": [1] * 17}, start=date(2026, 9, 7))
    weekly = sales_time_series(history, ["A"], freq="W")
    assert weekly["Units_Sold"].tolist() == [7.0, 7.0]
    daily = sales_time_series(history, ["A"], freq="D")
    assert len(daily) == 17


def test_time_series_for_unknown_sku_is_empty():
    history = _history({"A": [1] * 14})
    assert sales_time_series(history, ["ZZZ"]).empty


def test_sample_trends_are_present(sample_metrics):
    assert set(sample_metrics["Demand_Trend"]) <= {"Growing", "Stable", "Declining", INSUFFICIENT_HISTORY}
    assert sample_metrics["Trend_Pct_Per_Week"].notna().any()
