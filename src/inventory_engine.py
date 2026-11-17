"""Pipeline orchestration and executive KPIs.

:func:`build_inventory_metrics` runs every engine in dependency order:

    validated data
      → sales velocity, revenue, margin       (sales_analytics)
      → ABC classification                    (sales_analytics)
      → demand trends & variability           (sales_analytics, if history supplied)
      → reorder plan & alerts                 (reorder_engine; uses demand variability
                                               for statistical safety stock)
      → health score                          (health_score)
      → risk classification                   (risk_analysis)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from src.config import AnalyticsConfig
from src.health_score import compute_health_scores
from src.reorder_engine import compute_reorder_plan
from src.risk_analysis import classify_risk
from src.sales_analytics import (
    NO_HISTORY,
    classify_abc,
    compute_demand_trends,
    compute_sales_velocity,
)


def build_inventory_metrics(
    data: pd.DataFrame,
    config: AnalyticsConfig,
    history: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute every product-level metric used by the dashboard.

    Args:
        data: Validated inventory data (``ValidationResult.data``).
        config: Business parameters.
        history: Optional validated daily sales history.

    Returns:
        One row per SKU with all derived metrics appended.
    """
    metrics = compute_sales_velocity(data, config)
    metrics["ABC_Class"] = classify_abc(metrics["Revenue"])

    if history is not None and not history.empty:
        trends = compute_demand_trends(history, config)
        metrics = metrics.merge(trends, on="SKU_ID", how="left")  # preserves row order
        metrics["Demand_Trend"] = metrics["Demand_Trend"].fillna(NO_HISTORY)
    else:
        metrics["Demand_Trend"] = NO_HISTORY

    metrics = compute_reorder_plan(metrics, config)
    metrics = compute_health_scores(metrics, config)
    metrics = classify_risk(metrics, config)

    if "Last_Restock_Date" in metrics.columns:
        reference = pd.Timestamp(config.reference_date)
        metrics["Days_Since_Restock"] = (reference - metrics["Last_Restock_Date"]).dt.days
    return metrics


@dataclass(frozen=True)
class KPISummary:
    """Executive KPIs computed from the (filtered) metrics table."""

    total_products: int
    total_inventory_units: float
    total_inventory_value: float
    total_revenue: float
    total_units_sold: float
    avg_daily_sales: float
    products_requiring_reorder: int
    low_stock_products: int
    out_of_stock_products: int
    slow_moving_products: int
    avg_health_score: float
    insufficient_data_products: int
    total_gross_profit: float = float("nan")
    """Σ Units_Sold × (price − cost); NaN when no cost data is available."""
    gross_margin_pct: float = float("nan")
    """Gross profit ÷ revenue over products with known cost."""

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def _nan_safe(value: float) -> float:
    return float(value) if pd.notna(value) else float("nan")


def compute_kpis(metrics: pd.DataFrame) -> KPISummary:
    """Calculate dashboard KPIs. Missing values are excluded from sums/means."""
    if metrics.empty:
        return KPISummary(0, 0.0, 0.0, 0.0, 0.0, float("nan"), 0, 0, 0, 0, float("nan"), 0)
    gross_profit = margin = float("nan")
    if "Gross_Profit" in metrics.columns and metrics["Gross_Profit"].notna().any():
        with_cost = metrics["Gross_Profit"].notna()
        gross_profit = float(metrics.loc[with_cost, "Gross_Profit"].sum())
        costed_revenue = float(metrics.loc[with_cost, "Revenue"].sum())
        margin = gross_profit / costed_revenue if costed_revenue > 0 else float("nan")
    return KPISummary(
        total_products=int(len(metrics)),
        total_inventory_units=float(np.nansum(metrics["Current_Stock"])),
        total_inventory_value=float(np.nansum(metrics["Inventory_Value"])),
        total_revenue=float(np.nansum(metrics["Revenue"])),
        total_units_sold=float(np.nansum(metrics["Units_Sold"])),
        avg_daily_sales=_nan_safe(metrics["Avg_Daily_Sales"].mean()),
        products_requiring_reorder=int(metrics["Needs_Reorder"].sum()),
        low_stock_products=int(metrics["Stock_Status"].isin(["Low Stock", "Out of Stock"]).sum()),
        out_of_stock_products=int((metrics["Stock_Status"] == "Out of Stock").sum()),
        slow_moving_products=int(metrics["Is_Slow_Moving"].sum()),
        avg_health_score=_nan_safe(metrics["Health_Score"].mean()),
        insufficient_data_products=int((metrics["Health_Class"] == "Insufficient Data").sum()),
        total_gross_profit=gross_profit,
        gross_margin_pct=margin,
    )
