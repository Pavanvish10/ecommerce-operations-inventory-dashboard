"""Regression tests for defects found in the production-readiness audit.

Each test reproduces a defect that existed before the audit fixes (see
docs/technical_documentation.md §6, "Audit findings").
"""

from __future__ import annotations

import io
import itertools
import math
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest

from src.config import AnalyticsConfig, STOCK_STATUSES
from src.data_loader import DataLoadError, load_csv
from src.data_validator import MAX_NUMERIC_VALUE, validate_inventory
from src.export_manager import neutralize_formulas, prepare_report, to_csv_bytes
from src.reorder_engine import MAX_STOCKOUT_HORIZON_DAYS
from src.risk_analysis import RISK_CATEGORIES
from src.utils import join_nonempty
from tests.conftest import AS_OF, make_metrics, make_raw


# --------------------------------------------------------------------------- #
# A1 – Decimal commas were silently multiplied (e.g. "12,50" → 1250)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "raw, expected",
    [("12,50", None), ("1,2,3", None), ("1,234.5", 1234.5), ("$12,345", 12345.0),
     ("1,000,000", 1_000_000.0), ("12.50", 12.5)],
)
def test_ambiguous_commas_are_rejected_not_guessed(raw, expected):
    result = validate_inventory(make_raw([{"Unit_Price": raw}]))
    value = result.data.loc[0, "Unit_Price"]
    if expected is None:
        assert np.isnan(value)
        assert any("thousands separators" in i.message for i in result.warnings)
    else:
        assert value == expected


# --------------------------------------------------------------------------- #
# A2 – Implausible magnitudes overflowed totals (1e308 → inventory value inf)
# --------------------------------------------------------------------------- #


def test_implausibly_large_values_are_rejected():
    result = validate_inventory(make_raw([
        {"Current_Stock": "1e308"}, {"Current_Stock": str(int(MAX_NUMERIC_VALUE))},
    ]))
    assert np.isnan(result.data.loc[0, "Current_Stock"])
    assert result.data.loc[1, "Current_Stock"] == MAX_NUMERIC_VALUE  # boundary accepted
    assert any("plausible maximum" in i.message for i in result.warnings)


# --------------------------------------------------------------------------- #
# A3 – Timezone-aware dates crashed validation (ValueError: Mixed timezones)
# --------------------------------------------------------------------------- #


def test_timezone_aware_and_mixed_dates_do_not_crash():
    result = validate_inventory(
        make_raw([{"Last_Restock_Date": "2026-01-01T10:00:00Z"},
                  {"Last_Restock_Date": "2026-01-05T08:00:00+05:30"},
                  {"Last_Restock_Date": "2026-02-01"}]),
        reference_date=AS_OF,
    )
    dates = result.data["Last_Restock_Date"]
    assert dates.dt.tz is None
    assert dates.tolist() == [pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-05"),
                              pd.Timestamp("2026-02-01")]


def test_ambiguous_day_month_dates_are_flagged():
    result = validate_inventory(
        make_raw([{"Last_Restock_Date": "03/04/2026"}, {"Last_Restock_Date": "25/04/2026"},
                  {"Last_Restock_Date": "2026-04-03"}]),
        reference_date=AS_OF,
    )
    note = next(i for i in result.infos if "ambiguous" in i.message)
    assert note.rows == [2]  # only 03/04 is ambiguous; 25/04 and ISO are not


# --------------------------------------------------------------------------- #
# A4 – SKUs differing only by case were silently treated as different products
# --------------------------------------------------------------------------- #


def test_case_variant_skus_are_warned():
    result = validate_inventory(make_raw([{"SKU_ID": "abc-1"}, {"SKU_ID": "ABC-1"}, {"SKU_ID": "x"}]))
    assert len(result.data) == 3  # kept: they may legitimately differ
    warning = next(i for i in result.warnings if "letter case" in i.message)
    assert warning.rows == [2, 3]


# --------------------------------------------------------------------------- #
# A5 – Floating-point noise changed integer results
# --------------------------------------------------------------------------- #


def test_reorder_quantity_not_inflated_by_float_noise(config):
    # d = 0.1, L = 9, R = 14, SS = 0.7 → TS = 3 exactly, but 3.0000000000000004 in floats
    row = make_metrics([{"Current_Stock": 0, "Units_Sold": 3, "Supplier_Lead_Time": 9,
                         "Reorder_Level": 0}], config).iloc[0]
    assert row["Reorder_Quantity"] == 3


def test_exact_cover_equal_to_lead_time_is_not_high_alert(config):
    # d = 23/30, S = 23 → exactly 30 days of cover (29.999999999999996 in floats)
    row = make_metrics([{"Current_Stock": 23, "Units_Sold": 23, "Supplier_Lead_Time": 30,
                         "Reorder_Level": 0}],
                       AnalyticsConfig(as_of_date=AS_OF, threshold_mode="dataset")).iloc[0]
    assert row["Days_Of_Inventory"] == 30.0
    assert row["Alert_Level"] != "High"
    assert row["Estimated_Stockout_Date"] == pd.Timestamp(AS_OF) + pd.Timedelta(days=30)


def _exact_reference(S, U, L, RL, cfg: AnalyticsConfig):
    """Reference implementation in exact rational arithmetic (days method, max mode)."""
    d = Fraction(U, cfg.sales_window_days)
    ss = d * Fraction(cfg.safety_stock_days).limit_denominator()
    rop = d * L + ss
    threshold = max(Fraction(RL), rop)
    review = Fraction(cfg.review_period_days).limit_denominator()
    target = max(d * (L + review) + ss, threshold + d * review)
    doi = Fraction(S) / d if d else None
    critical = d > 0 and (S == 0 or doi <= Fraction(cfg.critical_cover_days).limit_denominator())
    high = d > 0 and not critical and doi < L
    needs = critical or high or (threshold > 0 and S <= threshold)
    qty = math.ceil(max(Fraction(0), target - S)) if needs else 0
    excess = math.floor(max(Fraction(0), S - d * Fraction(cfg.overstock_cover_days).limit_denominator()))
    return qty, excess, critical, high


def test_engine_matches_exact_rational_reference(config):
    """Property test: float engine == exact arithmetic over 1,000+ input combinations."""
    grid = list(itertools.product([0, 1, 7, 23, 46, 120], [1, 3, 7, 23, 29, 60, 91],
                                  [1, 3, 9, 14, 30], [0, 5, 40], ))
    rows = [{"SKU_ID": f"S{i}", "Current_Stock": s, "Units_Sold": u, "Supplier_Lead_Time": l,
             "Reorder_Level": rl} for i, (s, u, l, rl) in enumerate(grid)]
    metrics = make_metrics(rows, config)
    mismatches = []
    for (s, u, l, rl), (_, row) in zip(grid, metrics.iterrows()):
        qty, excess, critical, high = _exact_reference(s, u, l, rl, config)
        got = (row["Reorder_Quantity"], row["Excess_Units"],
               row["Alert_Level"] == "Critical", row["Alert_Level"] == "High")
        if got != (qty, excess, critical, high):
            mismatches.append(((s, u, l, rl), got, (qty, excess, critical, high)))
    assert not mismatches, mismatches[:5]
    assert len(grid) >= 600


# --------------------------------------------------------------------------- #
# A6 – Very long cover overflowed the stockout-date calculation (crash)
# --------------------------------------------------------------------------- #


def test_extreme_cover_does_not_crash_and_is_explained(config):
    row = make_metrics([{"Current_Stock": 5_000_000_000, "Units_Sold": 1}], config).iloc[0]
    assert pd.isna(row["Estimated_Stockout_Date"])
    assert "not projected" in row["Reorder_Note"]
    within = make_metrics([{"Current_Stock": MAX_STOCKOUT_HORIZON_DAYS, "Units_Sold": 30}], config).iloc[0]
    assert within["Estimated_Stockout_Date"] == pd.Timestamp(AS_OF) + pd.Timedelta(
        days=MAX_STOCKOUT_HORIZON_DAYS)


# --------------------------------------------------------------------------- #
# A7 – CSV / formula injection in exports
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "name, exported",
    [('=HYPERLINK("http://x","y")', '\'=HYPERLINK("http://x","y")'), ("+1+cmd", "'+1+cmd"),
     ("@SUM(A1)", "'@SUM(A1)"), ("-2+3", "'-2+3"), ("\tTab", "'\tTab"), ("Normal name", "Normal name"),
     ("-4", "-4"), ("-4.5", "-4.5")],
)
def test_formula_injection_is_neutralised(name, exported):
    frame = pd.DataFrame({"Product_Name": [name], "Current_Stock": [-1.0]})
    back = pd.read_csv(io.BytesIO(to_csv_bytes(frame)), dtype=str, keep_default_na=False)
    assert back.loc[0, "Product_Name"] == exported
    assert back.loc[0, "Current_Stock"] == "-1.0"  # numeric columns untouched


def test_reports_are_sanitised_end_to_end(config):
    m = make_metrics([{"Product_Name": "=1+1"}], config)
    content = to_csv_bytes(prepare_report(m, "full")).decode()
    assert ",'=1+1," in content


def test_neutralize_does_not_mutate_input():
    frame = pd.DataFrame({"a": ["=x"]})
    neutralize_formulas(frame)
    assert frame.loc[0, "a"] == "=x"


# --------------------------------------------------------------------------- #
# A8 – Uploaded text was rendered as raw HTML in the product drill-down (XSS)
# --------------------------------------------------------------------------- #


def test_product_heading_escapes_uploaded_text():
    from views.explorer import product_heading_html

    row = pd.Series({"Product_Name": "<img src=x onerror=alert(1)>", "SKU_ID": "<b>1</b>",
                     "Category": "A&B", "Health_Class": "Healthy", "Alert_Level": "High"})
    html = product_heading_html(row)
    assert "<img" not in html and "<b>" not in html
    assert "&lt;img src=x onerror=alert(1)&gt;" in html
    assert "A&amp;B" in html


def test_markdown_escape_neutralises_links_and_formatting():
    from components.alerts import md_escape

    escaped = md_escape(r"[click](https://x) **b** `c` $x$ \ ok.csv")
    for raw in ("[click]", "**b**", "`c`", "$x$"):
        assert raw not in escaped
    assert escaped.startswith(r"\[click\]")
    assert r"\\ ok.csv" in escaped  # the backslash itself is escaped
    assert md_escape("plain-name_v2.csv") == r"plain-name\_v2.csv"


# --------------------------------------------------------------------------- #
# A9 – "Low Stock" meant two different things (risk category vs stock status)
# --------------------------------------------------------------------------- #


def test_risk_categories_do_not_reuse_stock_status_labels_with_new_meaning():
    shared = set(RISK_CATEGORIES) & set(STOCK_STATUSES)
    assert shared == {"Out of Stock"}


def test_out_of_stock_label_is_consistent(sample_metrics):
    risk_oos = sample_metrics["Risk_Category"] == "Out of Stock"
    status_oos = sample_metrics["Stock_Status"] == "Out of Stock"
    assert (risk_oos == status_oos).all()


# --------------------------------------------------------------------------- #
# A10 – Upload size limit
# --------------------------------------------------------------------------- #


def test_upload_size_limit_enforced(monkeypatch):
    import src.data_loader as loader

    monkeypatch.setattr(loader, "MAX_FILE_BYTES", 100)
    with pytest.raises(DataLoadError, match="maximum supported size"):
        load_csv(b"a,b\n" + b"1,2\n" * 50)


# --------------------------------------------------------------------------- #
# A11 – Row-wise string joining was the main performance bottleneck
# --------------------------------------------------------------------------- #


def test_join_nonempty_matches_rowwise_reference():
    rng = np.random.default_rng(0)
    choices = np.array(["", "a", "bb", None], dtype=object)
    parts = [pd.Series(rng.choice(choices, 500)) for _ in range(4)]
    expected = pd.concat(parts, axis=1).fillna("").apply(
        lambda r: "; ".join(p for p in r if p), axis=1)
    assert join_nonempty(parts).tolist() == expected.tolist()


def test_pipeline_scales_to_20k_products(config):
    """Generous guard (not a benchmark): 20k products must score in well under 10 s."""
    import time

    n = 20_000
    raw = pd.DataFrame({
        "SKU_ID": [f"S{i}" for i in range(n)], "Product_Name": "P", "Category": "C",
        "Current_Stock": (np.arange(n) % 300).astype(str), "Units_Sold": (np.arange(n) % 97).astype(str),
        "Unit_Price": "9.99", "Reorder_Level": "20", "Supplier_Lead_Time": "7",
    })
    from src.inventory_engine import build_inventory_metrics

    data = validate_inventory(raw).data
    start = time.perf_counter()
    metrics = build_inventory_metrics(data, config)
    assert time.perf_counter() - start < 10
    assert len(metrics) == n
