"""Slow-moving, excess and stockout-risk detection with recommended actions.

Risk categories are assigned in priority order (the first match wins):

=================  =====================================================  ==========
Category           Rule                                                   Risk level
=================  =====================================================  ==========
Insufficient Data  stock or sales velocity unknown                        Unknown
Out of Stock       ``S = 0``                                              Critical (Medium if no demand)
Stockout Risk      reorder alert is Critical or High                      Critical / High
Low Stock          at or below the effective reorder threshold            Medium
Dead Stock         ``Units_Sold = 0`` and ``S > 0``                        High
Excess Inventory   ``DOI > overstock_cover_days``                         Medium
Slow-Moving        ``d < slow_moving_velocity``                           Low
Healthy            none of the above                                      Low
=================  =====================================================  ==========

Tied-up working capital is measured as the value of stock beyond the overstock
horizon: ``Excess_Units = max(0, S − d × overstock_cover_days)``, valued at
``Unit_Price``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import INSUFFICIENT_DATA, RISK_LEVELS, AnalyticsConfig

RISK_CATEGORIES: tuple[str, ...] = (
    "Out of Stock",
    "Stockout Risk",
    "Low Stock",
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
    price = out["Unit_Price"].astype("float64")
    alert = out["Alert_Level"]
    overstock = config.overstock_cover_days

    known = stock.notna() & daily.notna()
    out["Stock_Status"] = stock_status(stock, out["Reorder_Level"].astype("float64"))
    out["Is_Slow_Moving"] = known & (daily < config.slow_moving_velocity)
    out["Is_Excess"] = known & (doi > overstock)
    excess_units = (stock - daily * overstock).clip(lower=0).where(known)
    out["Excess_Units"] = np.floor(excess_units)
    out["Excess_Inventory_Value"] = out["Excess_Units"] * price

    out_of_stock = known & (stock == 0)
    stockout_risk = known & alert.isin(["Critical", "High"]) & ~out_of_stock
    low_stock = known & out["Needs_Reorder"] & ~out_of_stock & ~stockout_risk
    remaining = known & ~(out_of_stock | stockout_risk | low_stock)
    dead = remaining & (sold == 0) & (stock > 0)
    excess = remaining & ~dead & out["Is_Excess"]
    slow = remaining & ~dead & ~excess & out["Is_Slow_Moving"]

    out["Risk_Category"] = np.select(
        [~known, out_of_stock, stockout_risk, low_stock, dead, excess, slow],
        [INSUFFICIENT_DATA, "Out of Stock", "Stockout Risk", "Low Stock", "Dead Stock",
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

    qty = _fmt(out["Reorder_Quantity"], 0)
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

    window = _fmt(out["Sales_Window_Days"], 0)
    rationale = pd.Series("", index=out.index, dtype=object)
    if "Data_Quality_Notes" in out.columns:
        missing_text = "Stock or sales data missing (" + out["Data_Quality_Notes"].fillna("") + ")"
    else:
        missing_text = pd.Series("Stock or sales data missing", index=out.index)
    rationale = rationale.mask(~known, missing_text)
    rationale = rationale.mask(
        out_of_stock & (daily > 0),
        "0 units on hand while selling " + _fmt(daily, 2) + "/day; order " + qty + " units",
    )
    rationale = rationale.mask(
        out_of_stock & (daily == 0),
        "Out of stock with no sales in the last " + window + " days – confirm demand before reordering",
    )
    rationale = rationale.mask(
        stockout_risk | low_stock,
        _fmt(doi) + " days of cover vs " + _fmt(out["Supplier_Lead_Time"]) + "-day lead time; order "
        + qty + " units",
    )
    rationale = rationale.mask(
        dead,
        "No sales in the last " + window + " days; " + _fmt(out["Inventory_Value"], 2)
        + " of inventory value tied up",
    )
    rationale = rationale.mask(
        excess,
        _fmt(doi, 0) + " days of cover (> " + f"{overstock:g}" + "); "
        + _fmt(out["Excess_Inventory_Value"], 2) + " in excess stock",
    )
    rationale = rationale.mask(
        slow,
        "Sells " + _fmt(daily, 2) + "/day (below " + f"{config.slow_moving_velocity:g}" + "/day)",
    )
    rationale = rationale.mask(
        rationale == "",
        _fmt(doi, 0).where(np.isfinite(doi), "∞") + " days of cover; health score "
        + _fmt(out["Health_Score"]),
    )
    out["Action_Rationale"] = rationale
    return out


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
