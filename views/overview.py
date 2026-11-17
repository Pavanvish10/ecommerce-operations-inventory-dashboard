"""Overview: executive KPIs, alerts and headline charts."""

from __future__ import annotations

import streamlit as st

from components import charts
from components.alerts import page_header, render_alert_summary
from components.kpi_cards import render_kpi_grid
from components.state import get_context
from src.inventory_engine import compute_kpis
from src.risk_analysis import risk_summary
from src.sales_analytics import category_performance, top_products
from views.common import context_bar, plot, require_data


def render() -> None:
    page_header(
        "Operations Overview",
        "Executive snapshot of sales performance, inventory position and replenishment risk.",
    )
    ctx = get_context()
    if not require_data(ctx) or not context_bar(ctx):
        return
    data = ctx.filtered

    render_kpi_grid(compute_kpis(data), ctx.config)
    st.write("")
    render_alert_summary(data)

    left, right = st.columns(2)
    with left:
        plot(
            charts.category_bar(
                category_performance(data), "Revenue", "Sales by Category", "Revenue", money=True
            ),
            "ov_sales_cat",
        )
    with right:
        plot(charts.risk_distribution(risk_summary(data)), "ov_risk")

    left, right = st.columns(2)
    with left:
        plot(
            charts.top_products_bar(
                top_products(data, "Units_Sold", 10), "Units_Sold",
                "Top 10 Best-Selling Products", "Units sold",
            ),
            "ov_top10",
        )
    with right:
        plot(
            charts.stock_vs_sales(data, ctx.config.target_cover_days, ctx.config.overstock_cover_days),
            "ov_stock_sales",
        )
