"""Executive KPI cards."""

from __future__ import annotations

import math

import streamlit as st

from src.config import AnalyticsConfig
from src.inventory_engine import KPISummary


def format_number(value: float, decimals: int = 0) -> str:
    """Thousands-separated number, or an em dash for missing values."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    return f"{value:,.{decimals}f}"


def format_currency(value: float) -> str:
    """Compact currency: $1.24M, $608.2K, $950."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    magnitude = abs(value)
    if magnitude >= 1e9:
        return f"${value / 1e9:,.2f}B"
    if magnitude >= 1e6:
        return f"${value / 1e6:,.2f}M"
    if magnitude >= 1e4:
        return f"${value / 1e3:,.1f}K"
    return f"${value:,.0f}"


def _card(label: str, value: str, help_text: str, caption: str | None = None) -> None:
    with st.container(border=True):
        st.metric(label, value, help=help_text)
        if caption:
            st.caption(caption)


def render_kpi_grid(kpis: KPISummary, config: AnalyticsConfig) -> None:
    """Render the eight executive KPI cards in two rows of four."""
    share = (
        f"{kpis.products_requiring_reorder / kpis.total_products:.0%} of products"
        if kpis.total_products
        else None
    )
    row1 = st.columns(4)
    with row1[0]:
        _card(
            "Total Products", format_number(kpis.total_products),
            "Number of unique SKUs after validation and filtering.",
            f"Avg health score {format_number(kpis.avg_health_score, 1)}",
        )
    with row1[1]:
        _card(
            "Total Inventory Units", format_number(kpis.total_inventory_units),
            "Sum of Current_Stock across products.",
            f"Valued at {format_currency(kpis.total_inventory_value)} (stock × unit price)",
        )
    with row1[2]:
        _card(
            "Total Revenue", format_currency(kpis.total_revenue),
            "Σ Units_Sold × Unit_Price over the sales window.",
            "Over the sales window",
        )
    with row1[3]:
        _card(
            "Total Units Sold", format_number(kpis.total_units_sold),
            "Sum of Units_Sold across products.",
            "Over the sales window",
        )
    row2 = st.columns(4)
    with row2[0]:
        _card(
            "Avg Sales Velocity", f"{format_number(kpis.avg_daily_sales, 2)} /day",
            "Mean of per-product average daily sales (Units_Sold ÷ sales window days).",
            "Units per product per day",
        )
    with row2[1]:
        _card(
            "Products Requiring Reorder", format_number(kpis.products_requiring_reorder),
            "Products with a Critical/High alert or stock at/below the effective reorder threshold.",
            share,
        )
    with row2[2]:
        _card(
            "Low Stock Products", format_number(kpis.low_stock_products),
            "Products with stock at or below their dataset Reorder_Level (including out of stock).",
            f"{format_number(kpis.out_of_stock_products)} out of stock",
        )
    with row2[3]:
        _card(
            "Slow-Moving Products", format_number(kpis.slow_moving_products),
            f"Products selling fewer than {config.slow_moving_velocity:g} units/day "
            "(configurable in Settings).",
            f"Threshold: < {config.slow_moving_velocity:g} units/day",
        )
