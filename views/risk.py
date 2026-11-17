"""Risk Analysis: slow-moving, excess and stockout-risk products."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from components import charts
from components.alerts import page_header
from components.kpi_cards import format_currency, format_number
from components.state import get_context
from components.tables import render_table
from components.theme import PRIMARY
from src.risk_analysis import RISK_CATEGORIES, excess_by_category, high_risk_products, risk_summary
from views.common import context_bar, plot, require_data

RISK_COLUMNS = [
    "Risk_Level", "Risk_Category", "SKU_ID", "Product_Name", "Category", "Current_Stock",
    "Avg_Daily_Sales", "Days_Of_Inventory", "Inventory_Value", "Excess_Inventory_Value",
    "Health_Score", "Recommended_Action", "Action_Rationale",
]


def _excess_chart(data) -> go.Figure:
    excess = excess_by_category(data)
    excess = excess[excess["Excess_Inventory_Value"] > 0]
    if excess.empty:
        return charts.empty_figure("No excess inventory for the current filters")
    excess = excess.iloc[::-1]
    fig = go.Figure(
        go.Bar(
            x=excess["Excess_Inventory_Value"], y=excess["Category"], orientation="h",
            marker_color=PRIMARY, text=excess["Excess_Inventory_Value"],
            texttemplate="%{text:$,.0f}", textposition="outside", cliponaxis=False,
            customdata=excess["Excess_Units"],
            hovertemplate="<b>%{y}</b><br>Excess value: %{x:$,.0f}<br>Excess units: %{customdata:,.0f}<extra></extra>",
        )
    )
    fig.update_layout(title="Tied-up Working Capital by Category", height=max(320, 90 + 28 * len(excess)))
    fig.update_xaxes(title="Excess inventory value", tickformat="$~s", showline=False, showgrid=True)
    fig.update_yaxes(showgrid=False)
    charts.room_for_labels(fig, excess["Excess_Inventory_Value"].max(), pad=0.3)
    return fig


def render() -> None:
    page_header(
        "Risk Analysis",
        "Products with stockout risk, excess or dead stock, and the action each one needs.",
    )
    ctx = get_context()
    if not require_data(ctx) or not context_bar(ctx):
        return
    data = ctx.filtered
    category_counts = data["Risk_Category"].value_counts()

    cols = st.columns(4)
    cols[0].metric(
        "Stockout risk", format_number(category_counts.get("Out of Stock", 0) + category_counts.get("Stockout Risk", 0)),
        help="Out of stock, or a Critical/High reorder alert.",
    )
    cols[1].metric("Dead stock", format_number(category_counts.get("Dead Stock", 0)),
                   help="In stock but no units sold during the sales window.")
    cols[2].metric("Slow-moving", format_number(int(data["Is_Slow_Moving"].sum())),
                   help=f"Average daily sales below {ctx.config.slow_moving_velocity:g} units.")
    cols[3].metric("Tied-up capital", format_currency(data["Excess_Inventory_Value"].sum()),
                   help=f"Value of stock beyond {ctx.config.overstock_cover_days:g} days of cover "
                        "(all stock for products with no sales).")

    with st.expander("How products are classified", icon="🧮"):
        st.markdown(
            "Categories are assigned in priority order — the first matching rule wins:\n\n"
            "| Category | Rule | Default action |\n|---|---|---|\n"
            "| Out of Stock | stock = 0 | Replenish (urgent) — or investigate demand if nothing sold |\n"
            "| Stockout Risk | Critical/High reorder alert | Replenish |\n"
            "| Reorder Point | at/below the effective reorder threshold, not yet a stockout risk | Replenish |\n"
            "| Dead Stock | no sales in the window, stock > 0 | Investigate demand |\n"
            f"| Excess Inventory | > {ctx.config.overstock_cover_days:g} days of cover | Reduce purchasing (review pricing if also slow-moving) |\n"
            f"| Slow-Moving | < {ctx.config.slow_moving_velocity:g} units/day | Monitor inventory |\n"
            "| Healthy | none of the above | Monitor inventory |\n"
        )

    left, right = st.columns(2)
    with left:
        plot(charts.risk_distribution(risk_summary(data)), "risk_dist")
    with right:
        plot(_excess_chart(data), "risk_excess")

    st.subheader("Products requiring attention")
    flagged = high_risk_products(data)
    options = [c for c in RISK_CATEGORIES if c in set(flagged["Risk_Category"])]
    selected = st.pills("Risk category", options, selection_mode="multi", key="risk_categories")
    if selected:
        flagged = flagged[flagged["Risk_Category"].isin(selected)]
    if flagged.empty:
        st.success("No at-risk products for the current selection.", icon="✅")
        return
    action_counts = flagged["Recommended_Action"].value_counts()
    st.caption(" · ".join(f"**{action}**: {count}" for action, count in action_counts.items()))
    render_table(flagged, RISK_COLUMNS, key="risk_table", height=480)
