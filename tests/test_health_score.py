"""Inventory Health Score: formulas, classification boundaries and edge cases."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

from src.config import AnalyticsConfig, HealthWeights
from src.health_score import (
    classify_health,
    explain_health_score,
    inventory_coverage_score,
    lead_time_coverage_score,
    reorder_compliance_score,
    stock_availability_score,
)
from tests.conftest import AS_OF, make_metrics


def test_hand_calculated_example(config):
    """S=27, sold=63 over 30d (d=2.1), RL=45, L=22 → 64.5 (see docs for derivation)."""
    m = make_metrics([{"Current_Stock": 27, "Units_Sold": 63, "Reorder_Level": 45,
                       "Supplier_Lead_Time": 22}], config).iloc[0]
    # A = min(1, 27 / (2.1*7)) = 1
    assert m["Score_Stock_Availability"] == pytest.approx(100.0)
    # C = DOI/T = (27/2.1)/30 = 0.4286
    assert m["Score_Inventory_Coverage"] == pytest.approx(42.9, abs=0.05)
    # R = 27 / (45*1.2) = 0.5
    assert m["Score_Reorder_Compliance"] == pytest.approx(50.0)
    # LT = 12.857 / (22 + 7) = 0.4433
    assert m["Score_Lead_Time_Coverage"] == pytest.approx(44.3, abs=0.05)
    expected = 100 * (0.35 * 1 + 0.30 * (27 / 2.1 / 30) + 0.20 * 0.5 + 0.15 * (27 / 2.1 / 29))
    assert m["Health_Score"] == pytest.approx(round(expected, 1))
    assert m["Health_Class"] == "Moderate"


def test_perfectly_stocked_product_scores_100(config):
    # d = 2/day, DOI = 60 (within 30–120), well above reorder level and lead time
    m = make_metrics([{"Current_Stock": 120, "Units_Sold": 60, "Reorder_Level": 20,
                       "Supplier_Lead_Time": 7}], config).iloc[0]
    assert m["Health_Score"] == 100.0
    assert m["Health_Class"] == "Healthy"


def test_zero_stock_scores_zero(config):
    m = make_metrics([{"Current_Stock": 0, "Units_Sold": 90}], config).iloc[0]
    assert m["Health_Score"] == 0.0
    assert m["Health_Class"] == "Critical"


def test_zero_sales_with_stock_is_dead_stock_profile(config):
    m = make_metrics([{"Current_Stock": 50, "Units_Sold": 0, "Reorder_Level": 10}], config).iloc[0]
    assert m["Score_Stock_Availability"] == 100.0  # no demand → buffer trivially met
    assert m["Score_Inventory_Coverage"] == 0.0    # infinite cover = maximal overstock
    assert m["Score_Reorder_Compliance"] == 100.0
    assert m["Score_Lead_Time_Coverage"] == 100.0
    assert m["Health_Score"] == 70.0


def test_zero_stock_and_zero_sales(config):
    m = make_metrics([{"Current_Stock": 0, "Units_Sold": 0}], config).iloc[0]
    assert m["Health_Score"] == 0.0
    assert not np.isnan(m["Health_Score"])


@pytest.mark.parametrize("column", ["Current_Stock", "Units_Sold"])
def test_missing_core_input_gives_insufficient_data(config, column):
    m = make_metrics([{column: "n/a"}], config).iloc[0]
    assert np.isnan(m["Health_Score"])
    assert m["Health_Class"] == "Insufficient Data"
    assert "not computed" in m["Health_Note"]


def test_missing_reorder_level_reweights_remaining_components(config):
    m = make_metrics([{"Current_Stock": 27, "Units_Sold": 63, "Reorder_Level": None,
                       "Supplier_Lead_Time": 22}], config).iloc[0]
    assert np.isnan(m["Score_Reorder_Compliance"])
    a, c, lt = 1.0, 27 / 2.1 / 30, 27 / 2.1 / 29
    expected = 100 * (0.35 * a + 0.30 * c + 0.15 * lt) / (0.35 + 0.30 + 0.15)
    assert m["Health_Score"] == pytest.approx(round(expected, 1))
    assert "Reorder_Level missing" in m["Health_Note"]


def test_missing_lead_time_flagged_partial(config):
    m = make_metrics([{"Supplier_Lead_Time": None}], config).iloc[0]
    assert np.isnan(m["Score_Lead_Time_Coverage"])
    assert not np.isnan(m["Health_Score"])
    assert "Supplier_Lead_Time" in m["Health_Note"]


@pytest.mark.parametrize(
    "score, label",
    [(100, "Healthy"), (80.0, "Healthy"), (79.9, "Moderate"), (60.0, "Moderate"),
     (59.9, "At Risk"), (40.0, "At Risk"), (39.9, "Critical"), (0.0, "Critical"),
     (np.nan, "Insufficient Data")],
)
def test_classification_boundaries(score, label):
    assert classify_health(pd.Series([score])).iloc[0] == label


@pytest.mark.parametrize(
    "doi, expected",
    [(0, 0.0), (15, 0.5), (30, 1.0), (60, 1.0), (120, 1.0), (180, 0.5), (240, 0.0), (1000, 0.0),
     (np.inf, 0.0)],
)
def test_coverage_piecewise(doi, expected):
    assert inventory_coverage_score(pd.Series([float(doi)]), 30, 120).iloc[0] == pytest.approx(expected)


def test_component_functions_handle_zero_denominators():
    stock = pd.Series([0.0, 5.0, 5.0, np.nan])
    daily = pd.Series([0.0, 0.0, 1.0, 1.0])
    assert stock_availability_score(stock, daily, 0).tolist()[:3] == [0.0, 1.0, 1.0]
    rl = pd.Series([0.0, 0.0, 10.0, 10.0])
    assert reorder_compliance_score(stock, rl, 0.2).tolist()[:3] == [0.0, 1.0, pytest.approx(5 / 12)]
    doi = pd.Series([0.0, np.inf, 5.0, np.nan])
    lead = pd.Series([0.0, 0.0, 0.0, 3.0])
    lt = lead_time_coverage_score(stock, daily, doi, lead, 0)
    assert lt.tolist()[:3] == [0.0, 1.0, 1.0]
    assert np.isnan(lt.iloc[3])


def test_scores_always_within_bounds(config):
    """Grid over extreme inputs: score is always NaN-free and within [0, 100]."""
    values = [0, 1, 7, 1_000_000]
    rows = [
        {"Current_Stock": s, "Units_Sold": u, "Reorder_Level": r, "Supplier_Lead_Time": l}
        for s, u, r, l in itertools.product(values, values, values, [0, 3, 365])
    ]
    m = make_metrics(rows, config)
    assert m["Health_Score"].notna().all()
    assert m["Health_Score"].between(0, 100).all()
    for column in ["Score_Stock_Availability", "Score_Inventory_Coverage",
                   "Score_Reorder_Compliance", "Score_Lead_Time_Coverage"]:
        assert m[column].between(0, 100).all()


def test_custom_weights_change_score():
    rows = [{"Current_Stock": 50, "Units_Sold": 0, "Reorder_Level": 10}]
    coverage_only = AnalyticsConfig(as_of_date=AS_OF, health_weights=HealthWeights(0, 1, 0, 0))
    availability_only = AnalyticsConfig(as_of_date=AS_OF, health_weights=HealthWeights(1, 0, 0, 0))
    assert make_metrics(rows, coverage_only).iloc[0]["Health_Score"] == 0.0
    assert make_metrics(rows, availability_only).iloc[0]["Health_Score"] == 100.0


def test_weights_are_normalised_and_validated():
    assert sum(HealthWeights(2, 2, 2, 2).normalised().values()) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        HealthWeights(0, 0, 0, 0).normalised()
    with pytest.raises(ValueError):
        HealthWeights(-1, 1, 1, 1).normalised()


def test_explanation_sums_to_score(config):
    m = make_metrics([{"Current_Stock": 27, "Units_Sold": 63, "Reorder_Level": 45,
                       "Supplier_Lead_Time": 22}, {"Reorder_Level": None}], config)
    for _, row in m.iterrows():
        table = explain_health_score(row, config)
        assert table["Weighted Points"].sum() == pytest.approx(row["Health_Score"], abs=0.2)
        assert table["Effective Weight"].sum() == pytest.approx(1.0)


def test_scores_are_deterministic(sample_metrics, config):
    from src.health_score import compute_health_scores

    again = compute_health_scores(sample_metrics, config)
    pd.testing.assert_series_equal(again["Health_Score"], sample_metrics["Health_Score"])
