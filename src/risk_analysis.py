"""Slow-moving, excess and stockout-risk detection with recommended actions.

Risk categories are assigned in priority order (the first match wins):

=================  =====================================================  ==========
Category           Rule                                                   Risk level
=================  =====================================================  ==========
Insufficient Data  stock or sales velocity unknown                        Unknown
Out of Stock       ``S = 0``                                              Critical (Medium if no demand)
Stockout Risk      reorder alert is Critical or High                      Critical / High
Reorder Point      at or below the effective reorder threshold            Medium
Dead Stock         ``Units_Sold = 0`` and ``S > 0``                        High
Excess Inventory   ``DOI > overstock_cover_days``                         Medium
Slow-Moving        ``d < slow_moving_velocity``                           Low
Healthy            none of the above                                      Low
=================  =====================================================  ==========

Tied-up working capital is measured as the value of stock beyond the overstock
horizon: ``Excess_Units = floor(max(0, S − d × overstock_cover_days))``, valued at
``Unit_Value`` (purchase cost when a ``Unit_Cost`` column is supplied,
otherwise selling price — see ``Valuation_Basis``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import INSUFFICIENT_DATA, RISK_LEVELS, AnalyticsConfig
from src.utils import safe_floor

RISK_CATEGORIES: tuple[str, ...] = (
    "Out of Stock",
    "Stockout Risk",
    "Reorder Point",
    "Dead Stock",
    "Excess Inventory",
    "Slow-Moving",
    "Healthy",
    INSUFFICIENT_DATA,
)
UNKNOWN_RISK = "Unknown"
RISK_PRIORITY: dict[str, int] = {
    **{level: rank for rank, level in enumerate(RISK_LEVELS)},
    UNKNOWN_RISK: len(RISK_LEVELS),
}

ACTION_REPLENISH_URGENT = "Replenish inventory (urgent)"
ACTION_REPLENISH = "Replenish inventory"
ACTION_MONITOR = "Monitor inventory"
ACTION_REDUCE = "Reduce purchasing"
ACTION_PRICING = "Review pricing"
ACTION_INVESTIGATE = "Investigate demand"
ACTION_FIX_DATA = "Fix data quality"


def stock_status(stock: pd.Series, reorder_level: pd.Series) -> pd.Series:
    """Out of Stock / Low Stock (≤ Reorder_Level) / In Stock / Unknown."""
    result = np.select(
        [stock.isna(), stock == 0, reorder_level.notna() & (stock <= reorder_level)],
        ["Unknown", "Out of Stock", "Low Stock"],
        default="In Stock",
    )
    return pd.Series(result, index=stock.index)


def _fmt(series: pd.Series, decimals: int = 1) -> pd.Series:
    return series.round(decimals).map(lambda v: f"{v:,.{decimals}f}" if pd.notna(v) else "n/a")


def classify_risk(plan: pd.DataFrame, config: AnalyticsConfig) -> pd.DataFrame:
    """Add stock status, slow-moving/excess flags, risk category and action.

    ``plan`` must contain the outputs of the sales, reorder and health engines.
    """
    out = plan.copy()
    stock = out["Current_Stock"].astype("float64")
    daily = out["Avg_Daily_Sales"].astype("float64")
    doi = out["Days_Of_Inventory"].astype("float64")
    sold = out["Units_Sold"].astype("float64")
    unit_value = (out["Unit_Value"] if "Unit_Value" in out.columns else out["Unit_Price"]).astype("float64")
    alert = out["Alert_Level"]
    overstock = config.overstock_cover_days

    known = stock.notna() & daily.notna()
    out["Stock_Status"] = stock_status(stock, out["Reorder_Level"].astype("float64"))
    out["Is_Slow_Moving"] = known & (daily < config.slow_moving_velocity)
    out["Is_Excess"] = known & (doi > overstock)
    excess_units = (stock - daily * overstock).clip(lower=0).where(known)
    out["Excess_Units"] = safe_floor(excess_units)
    out["Excess_Inventory_Value"] = out["Excess_Units"] * unit_value

    out_of_stock = known & (stock == 0)
    stockout_risk = known & alert.isin(["Critical", "High"]) & ~out_of_stock
    low_stock = known & out["Needs_Reorder"] & ~out_of_stock & ~stockout_risk
    remaining = known & ~(out_of_stock | stockout_risk | low_stock)
    dead = remaining & (sold == 0) & (stock > 0)
    excess = remaining & ~dead & out["Is_Excess"]
    slow = remaining & ~dead & ~excess & out["Is_Slow_Moving"]

    out["Risk_Category"] = np.select(
        [~known, out_of_stock, stockout_risk, low_stock, dead, excess, slow],
        [INSUFFICIENT_DATA, "Out of Stock", "Stockout Risk", "Reorder Point", "Dead Stock",
         "Excess Inventory", "Slow-Moving"],
        default="Healthy",
    )
    out["Risk_Level"] = np.select(
        [
            ~known,
            out_of_stock & (daily > 0),
            out_of_stock,
            stockout_risk & (alert == "Critical"),
            stockout_risk,
            dead,
            low_stock | excess,
        ],
        [UNKNOWN_RISK, "Critical", "Medium", "Critical", "High", "High", "Medium"],
        default="Low",
    )
    out["Risk_Priority"] = out["Risk_Level"].map(RISK_PRIORITY).astype(int)

    action = np.select(
        [
            ~known,
            out_of_stock & (daily > 0),
            out_of_stock,
            stockout_risk & (alert == "Critical"),
            stockout_risk | low_stock,
            dead,
            excess & out["Is_Slow_Moving"],
            excess,
        ],
        [ACTION_FIX_DATA, ACTION_REPLENISH_URGENT, ACTION_INVESTIGATE, ACTION_REPLENISH_URGENT,
         ACTION_REPLENISH, ACTION_INVESTIGATE, ACTION_PRICING, ACTION_REDUCE],
        default=ACTION_MONITOR,
    )
    out["Recommended_Action"] = action

    out["Action_Rationale"] = _build_rationale(
        out, known, out_of_stock, stockout_risk, low_stock, dead, excess, slow, config
    )
    return out


def _build_rationale(
    out: pd.DataFrame,
    known: pd.Series,
    out_of_stock: pd.Series,
    stockout_risk: pd.Series,
    low_stock: pd.Series,
    dead: pd.Series,
    excess: pd.Series,
    slow: pd.Series,
    config: AnalyticsConfig,
) -> pd.Series:
    """Explain each classification with the product's own numbers.

    Each branch formats numbers only for the rows it applies to; formatting
    every column for every row was the main cost of the risk engine.
    """
    daily = out["Avg_Daily_Sales"].astype("float64")
    rationale = pd.Series("", index=out.index, dtype=object)

    def assign(mask: pd.Series, build) -> None:
        if mask.any():
            rationale.loc[mask] = build(out.loc[mask])

    if "Data_Quality_Notes" in out.columns:
        assign(~known, lambda f: "Stock or sales data missing (" + f["Data_Quality_Notes"].fillna("") + ")")
    else:
        assign(~known, lambda f: pd.Series("Stock or sales data missing", index=f.index))
    assign(
        out_of_stock & (daily > 0),
        lambda f: "0 units on hand while selling " + _fmt(f["Avg_Daily_Sales"], 2) + "/day; order "
        + _fmt(f["Reorder_Quantity"], 0) + " units",
    )
    assign(
        out_of_stock & (daily == 0),
        lambda f: "Out of stock with no sales in the last " + _fmt(f["Sales_Window_Days"], 0)
        + " days – confirm demand before reordering",
    )
    assign(
        stockout_risk | low_stock,
        lambda f: _fmt(f["Days_Of_Inventory"]) + " days of cover vs " + _fmt(f["Supplier_Lead_Time"])
        + "-day lead time; order " + _fmt(f["Reorder_Quantity"], 0) + " units",
    )
    assign(
        dead,
        lambda f: "No sales in the last " + _fmt(f["Sales_Window_Days"], 0) + " days; "
        + _fmt(f["Inventory_Value"], 2) + " of inventory value tied up",
    )
    assign(
        excess,
        lambda f: _fmt(f["Days_Of_Inventory"], 0) + f" days of cover (> {config.overstock_cover_days:g}); "
        + _fmt(f["Excess_Inventory_Value"], 2) + " in excess stock",
    )
    assign(
        slow,
        lambda f: "Sells " + _fmt(f["Avg_Daily_Sales"], 2)
        + f"/day (below {config.slow_moving_velocity:g}/day)",
    )
    assign(
        rationale == "",
        lambda f: _fmt(f["Days_Of_Inventory"], 0).where(np.isfinite(f["Days_Of_Inventory"]), "∞")
        + " days of cover; health score " + _fmt(f["Health_Score"]),
    )
    return rationale


def risk_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    """Product count and inventory value per risk category (in canonical order)."""
    grouped = metrics.groupby("Risk_Category").agg(
        Products=("SKU_ID", "count"),
        Inventory_Value=("Inventory_Value", "sum"),
        Excess_Inventory_Value=("Excess_Inventory_Value", "sum"),
    )
    order = [c for c in RISK_CATEGORIES if c in grouped.index]
    return grouped.reindex(order).reset_index()


def excess_by_category(metrics: pd.DataFrame) -> pd.DataFrame:
    """Tied-up working capital (excess inventory value) per category, largest first."""
    grouped = (
        metrics.groupby("Category")
        .agg(
            Excess_Inventory_Value=("Excess_Inventory_Value", "sum"),
            Excess_Units=("Excess_Units", "sum"),
            Products=("SKU_ID", "count"),
        )
        .reset_index()
    )
    return grouped.sort_values("Excess_Inventory_Value", ascending=False, kind="mergesort").reset_index(
        drop=True
    )


def high_risk_products(metrics: pd.DataFrame) -> pd.DataFrame:
    """Products whose risk category is anything other than Healthy, worst first."""
    flagged = metrics[metrics["Risk_Category"] != "Healthy"]
    return flagged.sort_values(
        ["Risk_Priority", "Health_Score", "SKU_ID"], kind="mergesort", na_position="last"
    ).reset_index(drop=True)
