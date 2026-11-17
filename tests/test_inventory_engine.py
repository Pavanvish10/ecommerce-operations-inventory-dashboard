"""Pipeline orchestration, KPIs, and slow-moving / high-risk detection."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import AnalyticsConfig
from src.inventory_engine import build_inventory_metrics, compute_kpis
from src.risk_analysis import (
    ACTION_INVESTIGATE,
    ACTION_MONITOR,
    ACTION_PRICING,
    ACTION_REDUCE,
    ACTION_REPLENISH,
    ACTION_REPLENISH_URGENT,
    excess_by_category,
    high_risk_products,
    risk_summary,
)
from tests.conftest import AS_OF, make_metrics


class TestPipeline:
    def test_one_row_per_sku(self, sample_metrics):
        assert len(sample_metrics) == 160
        assert sample_metrics["SKU_ID"].is_unique

    def test_clean_data_has_no_insufficient_rows(self, sample_metrics):
        assert sample_metrics["Health_Score"].notna().all()
        assert (sample_metrics["Alert_Level"] != "Insufficient Data").all()

    def test_pipeline_is_deterministic(self, sample_metrics):
        from src.data_loader import load_csv
        from src.data_validator import validate_inventory, validate_sales_history
        from tests.conftest import SAMPLE_HISTORY, SAMPLE_INVENTORY

        inventory = validate_inventory(load_csv(SAMPLE_INVENTORY), reference_date=AS_OF)
        history = validate_sales_history(load_csv(SAMPLE_HISTORY), reference_date=AS_OF)
        again = build_inventory_metrics(inventory.data, AnalyticsConfig(as_of_date=AS_OF), history.data)
        pd.testing.assert_frame_equal(again, sample_metrics)

    def test_input_frame_not_mutated(self, config):
        from src.data_validator import validate_inventory
        from tests.conftest import make_raw

        data = validate_inventory(make_raw([{}])).data
        before = data.copy()
        build_inventory_metrics(data, config)
        pd.testing.assert_frame_equal(data, before)

    def test_days_since_restock(self, config):
        row = make_metrics([{"Last_Restock_Date": "2026-11-05"}], config).iloc[0]
        assert row["Days_Since_Restock"] == 10

    def test_no_history_marks_trend(self, config):
        assert make_metrics([{}], config).iloc[0]["Demand_Trend"] == "No History"


class TestKpis:
    def test_kpis_match_manual_calculation(self, sample_metrics):
        k = compute_kpis(sample_metrics)
        assert k.total_products == 160
        assert k.total_inventory_units == sample_metrics["Current_Stock"].sum()
        assert k.total_units_sold == sample_metrics["Units_Sold"].sum()
        assert k.total_revenue == pytest.approx((sample_metrics["Units_Sold"] * sample_metrics["Unit_Price"]).sum())
        assert k.avg_daily_sales == pytest.approx((sample_metrics["Units_Sold"] / 30).mean())
        assert k.products_requiring_reorder == int(sample_metrics["Needs_Reorder"].sum())
        assert k.low_stock_products == int(
            (sample_metrics["Current_Stock"] <= sample_metrics["Reorder_Level"]).sum()
        )
        assert k.slow_moving_products == int((sample_metrics["Units_Sold"] / 30 < 0.5).sum())

    def test_kpis_follow_filters(self, sample_metrics):
        subset = sample_metrics[sample_metrics["Category"] == "Books"]
        k = compute_kpis(subset)
        assert k.total_products == len(subset)
        assert k.total_revenue == pytest.approx(subset["Revenue"].sum())

    def test_empty_dataset_kpis(self):
        k = compute_kpis(pd.DataFrame())
        assert k.total_products == 0
        assert k.total_revenue == 0
        assert np.isnan(k.avg_daily_sales)

    def test_kpis_ignore_missing_values(self, config):
        m = make_metrics([{"Current_Stock": 10}, {"Current_Stock": "bad"}], config)
        k = compute_kpis(m)
        assert k.total_inventory_units == 10
        assert k.insufficient_data_products == 1


class TestRiskClassification:
    @pytest.fixture
    def m(self, config):
        rows = [
            {"SKU_ID": "OOS", "Current_Stock": 0, "Units_Sold": 30},
            {"SKU_ID": "OOS-NODEMAND", "Current_Stock": 0, "Units_Sold": 0, "Reorder_Level": 0},
            {"SKU_ID": "STOCKOUT", "Current_Stock": 2, "Units_Sold": 30},
            {"SKU_ID": "LOW", "Current_Stock": 30, "Units_Sold": 30, "Supplier_Lead_Time": 2, "Reorder_Level": 40},
            {"SKU_ID": "DEAD", "Current_Stock": 40, "Units_Sold": 0, "Unit_Price": 5, "Reorder_Level": 5},
            {"SKU_ID": "EXCESS-FAST", "Current_Stock": 2000, "Units_Sold": 300, "Unit_Price": 2},
            {"SKU_ID": "EXCESS-SLOW", "Current_Stock": 100, "Units_Sold": 6, "Reorder_Level": 1},
            {"SKU_ID": "SLOW", "Current_Stock": 12, "Units_Sold": 9, "Reorder_Level": 1, "Supplier_Lead_Time": 3},
            {"SKU_ID": "HEALTHY", "Current_Stock": 100, "Units_Sold": 60, "Reorder_Level": 20},
            {"SKU_ID": "UNKNOWN", "Units_Sold": "?"},
        ]
        return make_metrics(rows, config).set_index("SKU_ID")

    @pytest.mark.parametrize(
        "sku, category, level, action",
        [
            ("OOS", "Out of Stock", "Critical", ACTION_REPLENISH_URGENT),
            ("OOS-NODEMAND", "Out of Stock", "Medium", ACTION_INVESTIGATE),
            ("STOCKOUT", "Stockout Risk", "Critical", ACTION_REPLENISH_URGENT),
            ("LOW", "Low Stock", "Medium", ACTION_REPLENISH),
            ("DEAD", "Dead Stock", "High", ACTION_INVESTIGATE),
            ("EXCESS-FAST", "Excess Inventory", "Medium", ACTION_REDUCE),
            ("EXCESS-SLOW", "Excess Inventory", "Medium", ACTION_PRICING),
            ("SLOW", "Slow-Moving", "Low", ACTION_MONITOR),
            ("HEALTHY", "Healthy", "Low", ACTION_MONITOR),
            ("UNKNOWN", "Insufficient Data", "Unknown", "Fix data quality"),
        ],
    )
    def test_categories_levels_and_actions(self, m, sku, category, level, action):
        row = m.loc[sku]
        assert row["Risk_Category"] == category
        assert row["Risk_Level"] == level
        assert row["Recommended_Action"] == action
        assert row["Action_Rationale"]

    def test_slow_moving_flag(self, m):
        assert bool(m.loc["SLOW", "Is_Slow_Moving"])       # 0.3/day < 0.5
        assert bool(m.loc["DEAD", "Is_Slow_Moving"])       # zero sales is slow
        assert not bool(m.loc["HEALTHY", "Is_Slow_Moving"])  # 2/day
        assert not bool(m.loc["UNKNOWN", "Is_Slow_Moving"])  # unknown is not flagged

    def test_slow_moving_threshold_boundary(self):
        cfg = AnalyticsConfig(as_of_date=AS_OF, slow_moving_velocity=0.5)
        m = make_metrics([{"Units_Sold": 15}, {"Units_Sold": 14}], cfg)  # 0.5/day vs 0.467/day
        assert m["Is_Slow_Moving"].tolist() == [False, True]

    def test_excess_units_and_tied_up_capital(self, m):
        # EXCESS-FAST: d = 10, overstock 120 days → 1200 needed, 800 excess × $2
        assert m.loc["EXCESS-FAST", "Excess_Units"] == 800
        assert m.loc["EXCESS-FAST", "Excess_Inventory_Value"] == pytest.approx(1600)
        # DEAD: no demand → all stock is excess
        assert m.loc["DEAD", "Excess_Inventory_Value"] == pytest.approx(200)
        assert m.loc["HEALTHY", "Excess_Units"] == 0

    def test_risk_summary_and_ranking(self, m):
        summary = risk_summary(m.reset_index())
        assert summary["Products"].sum() == len(m)
        assert summary["Risk_Category"].iloc[0] == "Out of Stock"
        flagged = high_risk_products(m.reset_index())
        assert "HEALTHY" not in set(flagged["SKU_ID"])
        assert flagged["Risk_Priority"].is_monotonic_increasing

    def test_excess_by_category(self, m):
        result = excess_by_category(m.reset_index())
        assert result["Excess_Inventory_Value"].sum() == pytest.approx(m["Excess_Inventory_Value"].sum())


class TestConfig:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"sales_window_days": 0},
            {"safety_stock_days": -1},
            {"overstock_cover_days": 20, "target_cover_days": 30},
            {"threshold_mode": "average"},
            {"duplicate_strategy": "merge"},
            {"min_history_days": 1},
        ],
    )
    def test_invalid_configuration_rejected(self, kwargs):
        with pytest.raises(ValueError):
            AnalyticsConfig(**kwargs)

    def test_round_trip_dict(self):
        cfg = AnalyticsConfig(as_of_date=AS_OF, safety_stock_days=3)
        assert AnalyticsConfig.from_dict(cfg.to_dict()) == cfg
