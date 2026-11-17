"""Inventory Analytics: stock distribution and the Inventory Health Score."""

from __future__ import annotations

import streamlit as st

from components import charts
from components.alerts import page_header
from components.kpi_cards import format_currency, format_number
from components.state import get_context
from components.tables import render_table
from src.sales_analytics import category_performance
from views.common import context_bar, plot, require_data

HEALTH_TABLE_COLUMNS = [
    "SKU_ID", "Product_Name", "Category", "Current_Stock", "Avg_Daily_Sales", "Days_Of_Inventory",
    "Score_Stock_Availability", "Score_Inventory_Coverage", "Score_Reorder_Compliance",
    "Score_Lead_Time_Coverage", "Health_Score", "Health_Class", "Health_Note",
]


def render_health_methodology(config) -> None:
    """Explain the scoring formula with the *current* configured weights."""
    weights = config.health_weights.normalised()
    with st.expander("How the Inventory Health Score is calculated", icon="🧮"):
        st.markdown(
            "Each product gets four component scores between 0 and 1, combined with "
            "configurable weights (Settings page). `S` = stock, `d` = average daily sales, "
            "`DOI = S / d`, `RL` = reorder level, `L` = lead time."
        )
        st.markdown(
            f"""<div class="formula">A  Stock availability  ({weights['stock_availability']:.0%})  = min(1, S / SS),  SS = d × {config.safety_stock_days:g} safety days
C  Inventory coverage  ({weights['inventory_coverage']:.0%})  = DOI / {config.target_cover_days:g}      if DOI ≤ {config.target_cover_days:g}
                                  = 1                  if DOI ≤ {config.overstock_cover_days:g}
                                  = max(0, 1 − (DOI − {config.overstock_cover_days:g}) / {config.overstock_cover_days:g})   otherwise
R  Reorder compliance  ({weights['reorder_compliance']:.0%})  = min(1, S / (RL × {1 + config.approaching_threshold_buffer:g}))
LT Lead-time coverage  ({weights['lead_time_coverage']:.0%})  = min(1, DOI / (L + {config.safety_stock_days:g}))

Health Score = 100 × Σ wᵢ·cᵢ / Σ wᵢ   (over components that can be computed)</div>""",
            unsafe_allow_html=True,
        )
        st.markdown(
            "**Classes:** 80–100 🟢 Healthy · 60–79.9 🟡 Moderate · 40–59.9 🟠 At Risk · "
            "0–39.9 🔴 Critical.  \n"
            "**Edge cases:** zero stock scores 0 on availability, coverage, compliance and lead "
            "time; zero sales with stock gives full availability and lead-time scores but zero "
            "coverage (dead stock); missing stock or sales → *Insufficient Data* (no score); "
            "a missing reorder level or lead time drops that component and re-weights the rest."
        )


def render() -> None:
    page_header(
        "Inventory Analytics",
        "Where inventory sits, how healthy it is, and which products drag the score down.",
    )
    ctx = get_context()
    if not require_data(ctx) or not context_bar(ctx):
        return
    data = ctx.filtered
    perf = category_performance(data)

    cols = st.columns(4)
    cols[0].metric("Inventory value", format_currency(data["Inventory_Value"].sum()),
                   help="Σ Current_Stock × Unit_Price")
    cols[1].metric("Average health score", format_number(data["Health_Score"].mean(), 1),
                   help="Mean Inventory Health Score of products with a computable score.")
    healthy_share = (data["Health_Class"] == "Healthy").mean()
    cols[2].metric("Healthy products", f"{healthy_share:.0%}", help="Share of products scoring ≥ 80.")
    median_doi = data.loc[data["Current_Stock"] > 0, "Days_Of_Inventory"].median()
    cols[3].metric("Median days of cover", format_number(median_doi, 1),
                   help="Median of Current_Stock ÷ average daily sales, for products in stock.")

    render_health_methodology(ctx.config)

    measure = st.segmented_control(
        "Inventory distribution measure", ["Units", "Value"], default="Units", key="inv_measure",
    ) or "Units"
    left, right = st.columns(2)
    with left:
        if measure == "Units":
            fig = charts.category_bar(perf, "Inventory_Units", "Inventory Distribution by Category",
                                      "Units on hand")
        else:
            fig = charts.category_bar(perf, "Inventory_Value", "Inventory Distribution by Category",
                                      "Inventory value", money=True)
        plot(fig, "inv_dist")
    with right:
        plot(charts.health_histogram(data), "inv_hist")

    left, right = st.columns(2)
    with left:
        plot(charts.health_by_category(data), "inv_health_cat")
    with right:
        plot(
            charts.stock_vs_sales(data, ctx.config.target_cover_days, ctx.config.overstock_cover_days),
            "inv_stock_sales",
        )

    st.subheader("Lowest health scores")
    st.caption("Component scores are on a 0–100 scale; see the methodology above.")
    worst = data.sort_values(["Health_Score", "SKU_ID"], na_position="first", kind="mergesort").head(25)
    render_table(worst, HEALTH_TABLE_COLUMNS, key="inv_worst_table")
