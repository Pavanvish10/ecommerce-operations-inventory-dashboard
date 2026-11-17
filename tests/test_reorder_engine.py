"""Reorder engine: thresholds, alert levels, quantities, stockout dates."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.config import AnalyticsConfig
from src.reorder_engine import effective_threshold, reorder_recommendations
from tests.conftest import AS_OF, make_metrics


def one(row: dict, config: AnalyticsConfig | None = None) -> pd.Series:
    return make_metrics([row], config).iloc[0]


def test_hand_calculated_reorder_quantity(config):
    # d = 63/30 = 2.1, L = 22, SS = 14.7, ROP = 60.9 > RL 45 → T = 60.9
    # TS = max(2.1*(22+14) + 14.7, 60.9 + 2.1*14) = 90.3 → Q = ceil(90.3 - 27) = 64
    row = one({"Current_Stock": 27, "Units_Sold": 63, "Reorder_Level": 45, "Supplier_Lead_Time": 22}, config)
    assert row["Safety_Stock"] == pytest.approx(14.7)
    assert row["Reorder_Point"] == pytest.approx(60.9)
    assert row["Effective_Reorder_Threshold"] == pytest.approx(60.9)
    assert row["Target_Stock"] == pytest.approx(90.3)
    assert row["Reorder_Quantity"] == 64
    assert row["Reorder_Value"] == pytest.approx(640.0)
    assert row["Alert_Level"] == "High"  # 12.9 days of cover < 22-day lead time
    assert bool(row["Needs_Reorder"])


def test_out_of_stock_with_demand_is_critical(config):
    row = one({"Current_Stock": 0, "Units_Sold": 30}, config)
    assert row["Alert_Level"] == "Critical"
    assert row["Alert_Reason"] == "Out of stock with active demand"
    assert row["Reorder_Quantity"] > 0
    assert row["Estimated_Stockout_Date"] == pd.Timestamp(AS_OF)


@pytest.mark.parametrize("stock, expected", [(3, "Critical"), (4, "High")])
def test_critical_cover_boundary(config, stock, expected):
    # d = 1/day, critical ≤ 3 days, lead time 7 days
    assert one({"Current_Stock": stock, "Units_Sold": 30, "Supplier_Lead_Time": 7}, config)["Alert_Level"] == expected


def test_high_when_cover_shorter_than_lead_time(config):
    row = one({"Current_Stock": 10, "Units_Sold": 30, "Supplier_Lead_Time": 14, "Reorder_Level": 5}, config)
    assert row["Alert_Level"] == "High"
    assert "lead time" in row["Alert_Reason"]


def test_lead_time_boundary_is_not_high(config):
    # DOI exactly equal to lead time → stock lasts until delivery → not High
    row = one({"Current_Stock": 14, "Units_Sold": 30, "Supplier_Lead_Time": 14, "Reorder_Level": 0},
              AnalyticsConfig(as_of_date=AS_OF, threshold_mode="dataset"))
    assert row["Alert_Level"] != "High"


def test_medium_at_threshold_requires_reorder(config):
    # d=1, L=2 → ROP = 2 + 7 = 9; RL = 40 → T = 40; stock 40 → at threshold
    row = one({"Current_Stock": 40, "Units_Sold": 30, "Supplier_Lead_Time": 2, "Reorder_Level": 40}, config)
    assert row["Alert_Level"] == "Medium"
    assert bool(row["Needs_Reorder"])
    assert row["Reorder_Quantity"] == math.ceil(40 + 14 - 40)  # T + d*R − S


def test_medium_approaching_threshold_does_not_reorder_yet(config):
    # T = 40, buffer 20% → approaching up to 48
    row = one({"Current_Stock": 45, "Units_Sold": 30, "Supplier_Lead_Time": 2, "Reorder_Level": 40}, config)
    assert row["Alert_Level"] == "Medium"
    assert row["Alert_Reason"] == "Approaching reorder threshold"
    assert not bool(row["Needs_Reorder"])
    assert row["Reorder_Quantity"] == 0


def test_healthy_product(config):
    row = one({"Current_Stock": 100, "Units_Sold": 30, "Supplier_Lead_Time": 5, "Reorder_Level": 20}, config)
    assert row["Alert_Level"] == "Healthy"
    assert not bool(row["Needs_Reorder"])
    assert row["Reorder_Quantity"] == 0
    assert row["Estimated_Stockout_Date"] == pd.Timestamp(AS_OF) + pd.Timedelta(days=100)


def test_zero_sales_zero_stock_is_not_a_false_critical(config):
    row = one({"Current_Stock": 0, "Units_Sold": 0, "Reorder_Level": 12}, config)
    assert row["Alert_Level"] == "Medium"
    assert "no sales recorded" in row["Alert_Reason"]
    assert row["Reorder_Quantity"] == 12  # restores the reorder level only
    assert "restores the reorder level" in row["Reorder_Note"]
    assert pd.isna(row["Estimated_Stockout_Date"])


def test_zero_sales_with_stock_never_stocks_out(config):
    row = one({"Current_Stock": 50, "Units_Sold": 0, "Reorder_Level": 10}, config)
    assert row["Alert_Level"] == "Healthy"
    assert np.isinf(row["Days_Of_Inventory"])
    assert pd.isna(row["Estimated_Stockout_Date"])


def test_insufficient_data_raises_no_alert(config):
    row = one({"Current_Stock": "unknown"}, config)
    assert row["Alert_Level"] == "Insufficient Data"
    assert not bool(row["Needs_Reorder"])
    assert row["Reorder_Quantity"] == 0


def test_missing_lead_time_is_explained(config):
    row = one({"Current_Stock": 10, "Units_Sold": 30, "Supplier_Lead_Time": None, "Reorder_Level": 20}, config)
    assert row["Alert_Level"] != "High"
    assert "lead time missing" in row["Reorder_Note"]
    assert bool(row["Needs_Reorder"])  # still below the dataset reorder level
    assert row["Reorder_Quantity"] == math.ceil(20 + 14 - 10)


def test_assumed_window_disclosed_in_note(config):
    row = one({"Current_Stock": 0, "Units_Sold": 30}, config)
    assert "30-day window" in row["Reorder_Note"]


@pytest.mark.parametrize("mode, expected", [("dataset", [50.0, 5.0]), ("dynamic", [10.0, 20.0]),
                                            ("max", [50.0, 20.0])])
def test_threshold_modes(mode, expected):
    rl = pd.Series([50.0, 5.0])
    rop = pd.Series([10.0, 20.0])
    assert effective_threshold(rl, rop, mode).tolist() == expected


def test_max_mode_falls_back_when_one_input_missing():
    result = effective_threshold(pd.Series([np.nan, 7.0]), pd.Series([3.0, np.nan]), "max")
    assert result.tolist() == [3.0, 7.0]


def test_invalid_threshold_mode():
    with pytest.raises(ValueError):
        effective_threshold(pd.Series([1.0]), pd.Series([1.0]), "median")


def test_higher_safety_stock_increases_quantity():
    row = {"Current_Stock": 5, "Units_Sold": 60, "Supplier_Lead_Time": 10, "Reorder_Level": 0}
    low = one(row, AnalyticsConfig(as_of_date=AS_OF, safety_stock_days=0))
    high = one(row, AnalyticsConfig(as_of_date=AS_OF, safety_stock_days=14))
    assert high["Reorder_Quantity"] - low["Reorder_Quantity"] == pytest.approx(2 * 14)


def test_reported_sales_period_drives_reorder_quantity(config):
    """Same Units_Sold over 7 reported days means 30/7× the demand of the assumed 30 days."""
    assumed = one({"Current_Stock": 0, "Units_Sold": 70, "Supplier_Lead_Time": 7}, config)
    reported = one({"Current_Stock": 0, "Units_Sold": 70, "Supplier_Lead_Time": 7,
                    "Sales_Period_Days": 7}, config)
    assert assumed["Avg_Daily_Sales"] == pytest.approx(70 / 30)
    assert reported["Avg_Daily_Sales"] == pytest.approx(10.0)
    assert reported["Velocity_Basis"] == "Reported period"
    assert reported["Reorder_Quantity"] == math.ceil(10 * (7 + 14) + 10 * 7)  # d(L+R) + SS
    assert "window" not in reported["Reorder_Note"]


def test_recommendations_sorted_by_urgency(config):
    m = make_metrics([
        {"SKU_ID": "MED", "Current_Stock": 40, "Units_Sold": 30, "Supplier_Lead_Time": 2, "Reorder_Level": 40},
        {"SKU_ID": "CRIT", "Current_Stock": 0, "Units_Sold": 30},
        {"SKU_ID": "OK", "Current_Stock": 500, "Units_Sold": 30, "Reorder_Level": 5},
        {"SKU_ID": "HIGH", "Current_Stock": 10, "Units_Sold": 30, "Supplier_Lead_Time": 14},
    ], config)
    plan = reorder_recommendations(m)
    assert plan["SKU_ID"].tolist() == ["CRIT", "HIGH", "MED"]
