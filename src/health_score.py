"""Inventory Health Score engine (0–100, transparent and explainable).

Each product receives four component scores in [0, 1]. ``S`` is current
stock, ``d`` average daily sales, ``DOI = S / d`` days of inventory, ``RL`` the
dataset reorder level, ``L`` the supplier lead time.

1. **Stock availability** (default weight 35 %) – depth of stock relative to
   the safety buffer ``SS = d × safety_stock_days``::

       A = 0                 if S = 0
         = 1                 if S > 0 and SS = 0   (no measurable demand)
         = min(1, S / SS)    otherwise

2. **Inventory coverage** (30 %) – rewards coverage near the target and
   penalises both under- and over-stocking (``T`` = target cover days,
   ``O`` = overstock cover days)::

       C = DOI / T                      if DOI ≤ T
         = 1                            if T < DOI ≤ O
         = max(0, 1 − (DOI − O) / O)    if DOI > O    (0 at 2·O; 0 for dead stock)

3. **Reorder threshold compliance** (20 %) – stock relative to the
   operator's reorder level plus buffer ``B``::

       R = min(1, S / (RL × (1 + B)))   if RL > 0
         = 1 if S > 0 else 0            if RL = 0

4. **Lead-time coverage** (15 %) – can stock last until an order placed today
   arrives, plus the safety buffer::

       LT = 0                                       if S = 0
          = 1                                       if d = 0
          = min(1, DOI / (L + safety_stock_days))   otherwise

Final score::

    Health = 100 × Σ wᵢ·cᵢ / Σ wᵢ      (sum over components that could be computed)

If stock or sales velocity is unknown the score is NaN ("Insufficient Data").
If only ``Reorder_Level`` or ``Supplier_Lead_Time`` is missing, that component
is skipped, the remaining weights are re-normalised, and the row is flagged
as a partial score.

Classification: 80–100 Healthy · 60–79.9 Moderate · 40–59.9 At Risk · 0–39.9 Critical.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import INSUFFICIENT_DATA, AnalyticsConfig

COMPONENT_COLUMNS: dict[str, str] = {
    "stock_availability": "Score_Stock_Availability",
    "inventory_coverage": "Score_Inventory_Coverage",
    "reorder_compliance": "Score_Reorder_Compliance",
    "lead_time_coverage": "Score_Lead_Time_Coverage",
}

HEALTH_BANDS: tuple[tuple[float, str], ...] = (
    (80.0, "Healthy"),
    (60.0, "Moderate"),
    (40.0, "At Risk"),
    (0.0, "Critical"),
)


def stock_availability_score(
    stock: pd.Series, daily: pd.Series, safety_days: float
) -> pd.Series:
    """Component A – stock relative to the safety-stock buffer."""
    safety = daily * safety_days
    ratio = (stock / safety.where(safety > 0)).clip(upper=1.0)
    score = ratio.mask(safety == 0, 1.0).mask(stock == 0, 0.0)
    return score.mask(stock.isna() | daily.isna())


def inventory_coverage_score(
    doi: pd.Series, target_days: float, overstock_days: float
) -> pd.Series:
    """Component C – piecewise coverage score around the target window."""
    under = (doi / target_days).clip(upper=1.0)
    over = (1 - (doi - overstock_days) / overstock_days).clip(lower=0.0)
    score = pd.Series(
        np.select(
            [doi <= target_days, doi <= overstock_days],
            [under, 1.0],
            default=over,
        ),
        index=doi.index,
        dtype="float64",
    )
    score = score.mask(np.isinf(doi), 0.0)
    return score.mask(doi.isna())


def reorder_compliance_score(
    stock: pd.Series, reorder_level: pd.Series, buffer: float
) -> pd.Series:
    """Component R – stock against the dataset reorder level (+ buffer)."""
    required = reorder_level * (1 + buffer)
    score = (stock / required.where(required > 0)).clip(upper=1.0)
    zero_level = reorder_level == 0
    score = score.mask(zero_level, (stock > 0).astype("float64"))
    return score.mask(stock.isna() | reorder_level.isna())


def lead_time_coverage_score(
    stock: pd.Series,
    daily: pd.Series,
    doi: pd.Series,
    lead_time: pd.Series,
    safety_days: float,
) -> pd.Series:
    """Component LT – days of inventory against lead time plus safety buffer."""
    horizon = lead_time + safety_days
    score = (doi / horizon.where(horizon > 0)).clip(upper=1.0)
    score = score.mask(horizon == 0, (stock > 0).astype("float64"))
    score = score.mask(daily == 0, 1.0).mask(stock == 0, 0.0)
    return score.mask(stock.isna() | daily.isna() | lead_time.isna())


def classify_health(score: pd.Series) -> pd.Series:
    """Map numeric scores to Healthy / Moderate / At Risk / Critical."""
    conditions = [score >= floor for floor, _ in HEALTH_BANDS]
    labels = [label for _, label in HEALTH_BANDS]
    result = np.select(conditions, labels, default=INSUFFICIENT_DATA)
    return pd.Series(result, index=score.index).mask(score.isna(), INSUFFICIENT_DATA)


def compute_health_scores(metrics: pd.DataFrame, config: AnalyticsConfig) -> pd.DataFrame:
    """Add component scores, ``Health_Score``, ``Health_Class`` and ``Health_Note``.

    ``metrics`` must contain ``Current_Stock``, ``Avg_Daily_Sales``,
    ``Days_Of_Inventory``, ``Reorder_Level`` and ``Supplier_Lead_Time``.
    Component columns are expressed on a 0–100 scale for readability.
    """
    out = metrics.copy()
    stock = out["Current_Stock"].astype("float64")
    daily = out["Avg_Daily_Sales"].astype("float64")
    doi = out["Days_Of_Inventory"].astype("float64")

    components = {
        "stock_availability": stock_availability_score(stock, daily, config.safety_stock_days),
        "inventory_coverage": inventory_coverage_score(
            doi, config.target_cover_days, config.overstock_cover_days
        ),
        "reorder_compliance": reorder_compliance_score(
            stock, out["Reorder_Level"].astype("float64"), config.approaching_threshold_buffer
        ),
        "lead_time_coverage": lead_time_coverage_score(
            stock,
            daily,
            doi,
            out["Supplier_Lead_Time"].astype("float64"),
            config.safety_stock_days,
        ),
    }
    weights = config.health_weights.normalised()

    weighted_sum = pd.Series(0.0, index=out.index)
    weight_used = pd.Series(0.0, index=out.index)
    for name, values in components.items():
        available = values.notna()
        weighted_sum += values.fillna(0.0) * weights[name]
        weight_used += available * weights[name]
        out[COMPONENT_COLUMNS[name]] = (values * 100).round(1)

    core_known = stock.notna() & daily.notna()
    score = (100 * weighted_sum / weight_used.where(weight_used > 0)).where(core_known)
    out["Health_Score"] = score.round(1)
    out["Health_Class"] = classify_health(out["Health_Score"])

    note = pd.Series("", index=out.index, dtype=object)
    note = note.mask(~core_known, "Stock or sales velocity missing – score not computed")
    partial_parts = []
    for name, label in (
        ("reorder_compliance", "Reorder_Level"),
        ("lead_time_coverage", "Supplier_Lead_Time"),
    ):
        partial_parts.append(
            pd.Series(np.where(core_known & components[name].isna(), label, ""), index=out.index)
        )
    missing_inputs = pd.concat(partial_parts, axis=1).apply(
        lambda row: ", ".join(p for p in row if p), axis=1
    )
    note = note.mask(
        missing_inputs != "",
        "Partial score (re-weighted): " + missing_inputs + " missing",
    )
    out["Health_Note"] = note
    return out


def explain_health_score(row: pd.Series, config: AnalyticsConfig) -> pd.DataFrame:
    """Break one product's score into its weighted contributions (for the UI)."""
    weights = config.health_weights.normalised()
    available = [n for n, c in COMPONENT_COLUMNS.items() if pd.notna(row.get(c))]
    used_total = sum(weights[n] for n in available)
    records = []
    for name, column in COMPONENT_COLUMNS.items():
        value = row.get(column)
        effective = weights[name] / used_total if name in available and used_total > 0 else 0.0
        records.append(
            {
                "Component": name.replace("_", " ").title(),
                "Component Score (0-100)": value,
                "Effective Weight": effective,
                "Weighted Points": None if pd.isna(value) else value * effective,
            }
        )
    return pd.DataFrame(records)
