"""Sales Analytics: velocity, revenue concentration and demand trends."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from components import charts
from components.alerts import page_header
from components.kpi_cards import format_currency, format_number
from components.state import get_context
from components.tables import render_table
from src.sales_analytics import (
    INSUFFICIENT_HISTORY,
    NO_HISTORY,
    category_performance,
    sales_time_series,
    slowest_movers,
    top_products,
)
from views.common import context_bar, plot, require_data

VELOCITY_COLUMNS = [
    "SKU_ID", "Product_Name", "Category", "Units_Sold", "Avg_Daily_Sales", "Avg_Weekly_Sales",
    "Avg_Monthly_Sales", "Days_Of_Inventory", "Revenue", "ABC_Class", "Demand_Trend",
    "Trend_Pct_Per_Week", "Velocity_Basis",
]


def _velocity_disclosure(data: pd.DataFrame, window_days: int) -> None:
    assumed = int(data["Velocity_Basis"].str.startswith("Assumed").sum())
    if assumed:
        st.info(
            f"**How velocity is calculated:** the inventory file provides aggregate `Units_Sold`. "
            f"For {assumed:,} product(s) the sales period is not supplied, so velocity assumes "
            f"`Units_Sold` covers the configured **{window_days}-day** window (Settings). Daily, weekly and monthly figures are average rates derived from "
            "that total — **not observed daily trends**. Demand trends are only shown when a daily "
            "sales history is uploaded.",
            icon="ℹ️",
        )
    else:
        st.caption("Velocity uses the sales period reported in the `Sales_Period_Days` column.")


def _render_trends(ctx, data: pd.DataFrame) -> None:
    st.subheader("Demand trends")
    if not ctx.has_history:
        st.caption(
            "No daily sales history loaded. Upload one on the **Data Sources** page "
            "(columns: Date, SKU_ID, Units_Sold) to see real demand trends."
        )
        return
    history = ctx.history.data
    skus = data["SKU_ID"].astype(str).tolist()
    st.caption(
        f"From the daily sales history ({ctx.history_name}): {history['Date'].min():%d %b %Y} – "
        f"{history['Date'].max():%d %b %Y}. Days without records count as zero sales. Trend = "
        f"least-squares slope, as % of mean daily demand per week (±{ctx.config.trend_threshold_pct:g}% "
        "threshold)."
    )
    left, right = st.columns([2, 1])
    with left:
        options = ["All filtered products", *sorted(
            (data["Product_Name"] + " (" + data["SKU_ID"] + ")").tolist()
        )]
        choice = st.selectbox("Series", options, key="sales_trend_series")
        if choice == options[0]:
            series = sales_time_series(history, skus, freq="W")
            title = "Weekly Units Sold (complete weeks)"
        else:
            sku = choice.rsplit("(", 1)[1].rstrip(")")
            series = sales_time_series(history, [sku], freq="W")
            title = f"Weekly Units Sold — {choice}"
        plot(charts.sales_trend_line(series, title), "sales_trend")
    with right:
        counts = (
            data["Demand_Trend"].value_counts()
            .reindex(["Growing", "Stable", "Declining", INSUFFICIENT_HISTORY, NO_HISTORY])
            .dropna().astype(int).rename("Products").rename_axis("Trend").reset_index()
        )
        st.dataframe(counts, hide_index=True, width="stretch")
        movers = data.dropna(subset=["Trend_Pct_Per_Week"]) if "Trend_Pct_Per_Week" in data else data.iloc[0:0]
        if not movers.empty:
            st.markdown("**Fastest growing**")
            render_table(
                movers.nlargest(5, "Trend_Pct_Per_Week"),
                ["Product_Name", "Trend_Pct_Per_Week"], key="sales_growing",
            )
            st.markdown("**Fastest declining**")
            render_table(
                movers.nsmallest(5, "Trend_Pct_Per_Week"),
                ["Product_Name", "Trend_Pct_Per_Week"], key="sales_declining",
            )


def render() -> None:
    page_header(
        "Sales Analytics",
        "Sales velocity, revenue concentration and product demand.",
    )
    ctx = get_context()
    if not require_data(ctx) or not context_bar(ctx):
        return
    data = ctx.filtered
    _velocity_disclosure(data, ctx.config.sales_window_days)

    cols = st.columns(4)
    cols[0].metric("Revenue", format_currency(data["Revenue"].sum()))
    cols[1].metric("Units sold", format_number(data["Units_Sold"].sum()))
    cols[2].metric("Fleet daily sales", f"{format_number(data['Avg_Daily_Sales'].sum(), 1)} /day",
                   help="Sum of average daily sales across the selected products.")
    a_share = (data["ABC_Class"] == "A").mean()
    cols[3].metric("Class A products", f"{a_share:.0%}",
                   help="Share of products generating the first 80% of revenue (ABC analysis on the full dataset).")

    st.subheader("Sales performance by category")
    perf = category_performance(data)
    st.dataframe(
        perf,
        hide_index=True,
        width="stretch",
        column_config={
            "Units_Sold": st.column_config.NumberColumn("Units Sold", format="%,.0f"),
            "Revenue": st.column_config.NumberColumn("Revenue", format="$%,.0f"),
            "Inventory_Units": st.column_config.NumberColumn("Inventory Units", format="%,.0f"),
            "Inventory_Value": st.column_config.NumberColumn("Inventory Value", format="$%,.0f"),
            "Avg_Daily_Sales": st.column_config.NumberColumn("Daily Sales (units)", format="%,.1f"),
            "Revenue_Share": st.column_config.ProgressColumn(
                "Revenue Share", min_value=0, max_value=1, format="percent"
            ),
        },
    )

    left, right = st.columns(2)
    with left:
        plot(
            charts.top_products_bar(top_products(data, "Units_Sold", 10), "Units_Sold",
                                    "Top 10 Best-Selling Products", "Units sold"),
            "sales_top10",
        )
    with right:
        plot(
            charts.top_products_bar(slowest_movers(data, 10), "Avg_Daily_Sales",
                                    "Top 10 Slow-Moving Products (in stock)", "Avg daily sales",
                                    decimals=2),
            "sales_slow10",
        )

    left, right = st.columns([3, 2])
    with left:
        plot(
            charts.top_products_bar(top_products(data, "Revenue", 20), "Revenue",
                                    "Revenue by Product (Top 20)", "Revenue", money=True),
            "sales_revenue_product",
        )
    with right:
        plot(charts.pareto_curve(data), "sales_pareto")

    _render_trends(ctx, data)

    st.subheader("Product velocity")
    render_table(
        data.sort_values(["Avg_Daily_Sales", "SKU_ID"], ascending=[False, True]),
        VELOCITY_COLUMNS, key="sales_velocity_table", height=420,
    )
