"""Cost/margin analysis, statistical safety stock and backtested forecasting."""

from __future__ import annotations

from datetime import date, timedelta
from statistics import NormalDist

import numpy as np
import pandas as pd
import pytest

from src.config import AnalyticsConfig
from src.data_validator import validate_inventory
from src.forecasting import METHODS, backtest_forecasts, forecast_demand
from src.inventory_engine import build_inventory_metrics, compute_kpis
from tests.conftest import AS_OF, make_metrics, make_raw


def _history(series: dict[str, list[float]], start: date = date(2026, 8, 1)) -> pd.DataFrame:
    rows = [
        {"SKU_ID": sku, "Date": pd.Timestamp(start + timedelta(days=i)), "Units_Sold": float(v)}
        for sku, values in series.items() for i, v in enumerate(values)
    ]
    return pd.DataFrame(rows)


def _metrics_with(rows, config, history=None, extra_columns=None):
    raw = make_raw(rows)
    for column, values in (extra_columns or {}).items():
        raw[column] = values
    data = validate_inventory(raw).data
    return build_inventory_metrics(data, config, history)


# --------------------------------------------------------------------------- #
# Purchase cost & gross margin
# --------------------------------------------------------------------------- #


class TestCostAndMargin:
    def test_margin_metrics(self, config):
        m = _metrics_with([{"Units_Sold": 30, "Unit_Price": 20, "Current_Stock": 10}], config,
                          extra_columns={"Unit_Cost": ["12"]}).iloc[0]
        assert m["Gross_Profit"] == pytest.approx(30 * 8)
        assert m["Gross_Margin_Pct"] == pytest.approx(0.4)
        assert m["Inventory_Cost_Value"] == pytest.approx(120)
        assert m["Inventory_Value"] == pytest.approx(200)  # retail value unchanged
        assert m["Valuation_Basis"] == "Cost"

    def test_capital_measures_use_cost_with_price_fallback(self, config):
        rows = [
            {"SKU_ID": "COST", "Current_Stock": 40, "Units_Sold": 0, "Unit_Price": 10},
            {"SKU_ID": "NOCOST", "Current_Stock": 40, "Units_Sold": 0, "Unit_Price": 10},
            {"SKU_ID": "OOS", "Current_Stock": 0, "Units_Sold": 30, "Unit_Price": 10},
        ]
        m = _metrics_with(rows, config, extra_columns={"Unit_Cost": ["4", "", "4"]}).set_index("SKU_ID")
        assert m.loc["COST", "Excess_Inventory_Value"] == pytest.approx(160)   # 40 × cost 4
        assert m.loc["NOCOST", "Excess_Inventory_Value"] == pytest.approx(400)  # 40 × price 10
        assert m.loc["NOCOST", "Valuation_Basis"] == "Price (cost missing)"
        assert m.loc["OOS", "Reorder_Value"] == pytest.approx(m.loc["OOS", "Reorder_Quantity"] * 4)

    def test_without_cost_column_everything_is_price_based(self, config):
        m = make_metrics([{"Current_Stock": 40, "Units_Sold": 0, "Unit_Price": 5}], config).iloc[0]
        assert m["Valuation_Basis"] == "Price (no cost data)"
        assert m["Excess_Inventory_Value"] == pytest.approx(200)
        assert "Gross_Profit" not in m.index
        assert np.isnan(compute_kpis(make_metrics([{}], config)).gross_margin_pct)

    def test_margin_kpi_only_over_costed_products(self, config):
        m = _metrics_with(
            [{"Units_Sold": 10, "Unit_Price": 10}, {"Units_Sold": 10, "Unit_Price": 1000}],
            config, extra_columns={"Unit_Cost": ["6", ""]},
        )
        k = compute_kpis(m)
        assert k.total_gross_profit == pytest.approx(40)
        assert k.gross_margin_pct == pytest.approx(0.4)  # 40 / 100, not 40 / 10,100

    def test_zero_price_margin_is_undefined_not_error(self, config):
        m = _metrics_with([{"Unit_Price": 0}], config, extra_columns={"Unit_Cost": ["1"]}).iloc[0]
        assert np.isnan(m["Gross_Margin_Pct"])


# --------------------------------------------------------------------------- #
# Statistical safety stock
# --------------------------------------------------------------------------- #


class TestStatisticalSafetyStock:
    @pytest.fixture
    def stat_config(self):
        return AnalyticsConfig(as_of_date=AS_OF, safety_stock_method="statistical", service_level=0.95)

    def test_formula(self, stat_config):
        daily = [2, 4] * 15  # mean 3, sample std computed below
        history = _history({"A": daily})
        sigma_d = float(np.std(daily, ddof=1))
        m = _metrics_with(
            [{"SKU_ID": "A", "Units_Sold": 90, "Supplier_Lead_Time": 4, "Current_Stock": 50}],
            stat_config, history, extra_columns={"Lead_Time_Std_Days": ["1.5"]},
        ).iloc[0]
        d, lead, sigma_l = 3.0, 4.0, 1.5
        z = NormalDist().inv_cdf(0.95)
        expected = z * np.sqrt(lead * sigma_d**2 + d**2 * sigma_l**2)
        assert m["Safety_Stock"] == pytest.approx(expected)
        assert m["Safety_Stock_Method"].startswith("Statistical (95.0%")
        assert m["Reorder_Point"] == pytest.approx(d * lead + expected)

    def test_higher_service_level_means_more_safety_stock(self):
        history = _history({"A": [1, 5] * 15})
        row = [{"SKU_ID": "A", "Units_Sold": 90, "Supplier_Lead_Time": 6}]
        low = _metrics_with(row, AnalyticsConfig(as_of_date=AS_OF, safety_stock_method="statistical",
                                                 service_level=0.90), history).iloc[0]
        high = _metrics_with(row, AnalyticsConfig(as_of_date=AS_OF, safety_stock_method="statistical",
                                                  service_level=0.99), history).iloc[0]
        assert high["Safety_Stock"] > low["Safety_Stock"]

    def test_fallback_without_history(self, stat_config, config):
        row = [{"SKU_ID": "A", "Units_Sold": 90}]
        stat = _metrics_with(row, stat_config).iloc[0]
        days = _metrics_with(row, config).iloc[0]
        assert stat["Safety_Stock"] == pytest.approx(days["Safety_Stock"])
        assert "fallback" in stat["Safety_Stock_Method"]

    def test_fallback_for_short_history_and_missing_lead_time(self, stat_config):
        short = _metrics_with([{"SKU_ID": "A"}], stat_config, _history({"A": [1, 2, 3]})).iloc[0]
        assert "fallback" in short["Safety_Stock_Method"]
        no_lead = _metrics_with([{"SKU_ID": "A", "Supplier_Lead_Time": None}], stat_config,
                                _history({"A": [1, 3] * 15})).iloc[0]
        assert "fallback" in no_lead["Safety_Stock_Method"]

    def test_constant_demand_needs_no_statistical_buffer(self, stat_config):
        m = _metrics_with([{"SKU_ID": "A", "Units_Sold": 90}], stat_config, _history({"A": [3] * 30})).iloc[0]
        assert m["Safety_Stock"] == pytest.approx(0.0)

    def test_health_availability_uses_actual_safety_stock(self, stat_config):
        # Statistical SS is small here, so stock of 5 fully covers it (A = 100)
        m = _metrics_with([{"SKU_ID": "A", "Units_Sold": 90, "Current_Stock": 5, "Supplier_Lead_Time": 2}],
                          stat_config, _history({"A": [3, 3, 3, 4] * 8})).iloc[0]
        assert m["Safety_Stock"] < 5
        assert m["Score_Stock_Availability"] == 100.0

    def test_sample_dataset_statistical_mode_runs(self, sample_metrics):
        from src.data_loader import load_csv
        from src.data_validator import validate_sales_history
        from tests.conftest import SAMPLE_HISTORY, SAMPLE_INVENTORY

        cfg = AnalyticsConfig(as_of_date=AS_OF, safety_stock_method="statistical")
        inv = validate_inventory(load_csv(SAMPLE_INVENTORY), reference_date=AS_OF)
        hist = validate_sales_history(load_csv(SAMPLE_HISTORY), reference_date=AS_OF)
        m = build_inventory_metrics(inv.data, cfg, hist.data)
        assert m["Safety_Stock_Method"].str.startswith("Statistical").all()
        assert m["Safety_Stock"].notna().all() and (m["Safety_Stock"] >= 0).all()


# --------------------------------------------------------------------------- #
# Forecasting with backtest
# --------------------------------------------------------------------------- #


class TestForecasting:
    def test_each_method_wins_on_its_own_pattern(self, config):
        days = 56
        history = _history({
            "TREND": [10 + i for i in range(days)],
        })
        assert backtest_forecasts(history, config).iloc[0]["Method"] == "Linear trend"
        history = _history({"WEEKLY": [1, 1, 1, 1, 10, 10, 1] * 8})
        result = backtest_forecasts(history, config).set_index("Method")
        assert result.loc["Seasonal naive (weekly)", "WAPE"] == pytest.approx(0.0)
        history = _history({"FLAT": [5] * days})
        result = backtest_forecasts(history, config).set_index("Method")
        assert result.loc["Moving average", "WAPE"] == pytest.approx(0.0)
        assert result.loc["Moving average", "Bias"] == pytest.approx(0.0)

    def test_metrics_are_computed_on_the_holdout_only(self, config):
        # Train period flat at 5, holdout at 10 → MA forecasts 5, WAPE = 0.5, bias = −0.5
        history = _history({"A": [5] * 42 + [10] * 14})
        result = backtest_forecasts(history, config).set_index("Method")
        assert result.loc["Moving average", "WAPE"] == pytest.approx(0.5)
        assert result.loc["Moving average", "Bias"] == pytest.approx(-0.5)
        assert result.loc["Moving average", "Holdout_Days"] == 14

    def test_insufficient_history_returns_empty(self, config):
        assert backtest_forecasts(_history({"A": [1] * 20}), config).empty
        assert backtest_forecasts(_history({"A": [1] * 42 + [0] * 14}), config).empty  # no holdout sales

    def test_forecasts_are_non_negative_and_complete(self, config):
        history = _history({"DOWN": [60 - i for i in range(56)], "UP": list(range(56))})
        for method in METHODS:
            fc = forecast_demand(history, config, method, horizon=30)
            assert len(fc) == 2
            assert (fc["Forecast_Daily"] >= 0).all()
            assert fc["Forecast_Total"].tolist() == pytest.approx((fc["Forecast_Daily"] * 30).tolist())

    def test_unknown_method(self, config):
        with pytest.raises(ValueError):
            forecast_demand(_history({"A": [1] * 40}), config, "ARIMA")

    def test_sample_backtest_reports_real_numbers(self, config):
        from src.data_loader import load_csv
        from src.data_validator import validate_sales_history
        from tests.conftest import SAMPLE_HISTORY

        history = validate_sales_history(load_csv(SAMPLE_HISTORY), reference_date=AS_OF).data
        result = backtest_forecasts(history, config)
        assert list(result["Method"]) and set(result["Method"]) == set(METHODS)
        assert result["WAPE"].is_monotonic_increasing
        assert (result["WAPE"] > 0).all()  # real noisy data: no method is perfect
