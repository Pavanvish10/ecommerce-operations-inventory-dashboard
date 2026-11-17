"""Product Explorer: search, combined filters, sorting and product drill-down."""

from __future__ import annotations

from dataclasses import replace

import pandas as pd
import streamlit as st

from components.alerts import page_header, status_badge
from components.kpi_cards import format_currency, format_number
from components.state import get_context, restore_widget, save_widget
from components.tables import format_days, render_table
from src.config import ALERT_LEVELS, INSUFFICIENT_DATA
from src.export_manager import prepare_report, report_filename, to_csv_bytes
from src.filters import FULL_SCORE_RANGE, SORT_OPTIONS, FilterCriteria, apply_filters, sort_products
from src.health_score import explain_health_score
from src.risk_analysis import RISK_CATEGORIES
from views.common import context_bar, require_data

EXPLORER_COLUMNS = [
    "SKU_ID", "Product_Name", "Category", "Current_Stock", "Units_Sold", "Avg_Daily_Sales",
    "Days_Of_Inventory", "Inventory_Value", "Health_Score", "Health_Class", "Alert_Level",
    "Risk_Category", "Recommended_Action", "Stock_Status", "ABC_Class",
]
EXPLORER_CRITERIA_KEY = "explorer_criteria"
_KEYS = ("ex_search", "ex_risk_categories", "ex_alerts", "ex_health_range", "ex_sort")


def _reset_explorer() -> None:
    defaults = {"ex_search": "", "ex_risk_categories": [], "ex_alerts": [],
                "ex_health_range": FULL_SCORE_RANGE, "ex_sort": "Highest risk"}
    for key, value in defaults.items():
        st.session_state[key] = value
        st.session_state[f"_persist_{key}"] = value


def explorer_criteria(global_criteria: FilterCriteria) -> FilterCriteria:
    """Global criteria combined with the explorer's own filters (last values used)."""
    stored = st.session_state.get(EXPLORER_CRITERIA_KEY)
    if stored is None:
        return global_criteria
    return replace(
        global_criteria,
        search=stored.search,
        risk_categories=stored.risk_categories,
        alert_levels=stored.alert_levels,
        health_score_range=stored.health_score_range,
    )


def _filter_controls(metrics: pd.DataFrame) -> tuple[FilterCriteria, str]:
    risk_options = [c for c in RISK_CATEGORIES if c in set(metrics["Risk_Category"])]
    alert_options = [a for a in [*ALERT_LEVELS, INSUFFICIENT_DATA] if a in set(metrics["Alert_Level"])]
    restore_widget("ex_search", "")
    restore_widget("ex_risk_categories", [], risk_options)
    restore_widget("ex_alerts", [], alert_options)
    restore_widget("ex_health_range", FULL_SCORE_RANGE)
    restore_widget("ex_sort", "Highest risk")

    top = st.columns([3, 2])
    top[0].text_input("Search by product name or SKU", key="ex_search",
                      placeholder="e.g. earbuds or ELE-1001")
    top[1].selectbox("Sort by", list(SORT_OPTIONS), key="ex_sort")
    row = st.columns([2, 2, 2])
    row[0].multiselect("Risk category", risk_options, key="ex_risk_categories", placeholder="All")
    row[1].multiselect("Alert level", alert_options, key="ex_alerts", placeholder="All")
    row[2].slider("Health score range", 0.0, 100.0, step=1.0, key="ex_health_range")
    for key in _KEYS:
        save_widget(key)

    state = st.session_state
    local = FilterCriteria(
        search=state["ex_search"],
        risk_categories=tuple(state["ex_risk_categories"]),
        alert_levels=tuple(state["ex_alerts"]),
        health_score_range=tuple(state["ex_health_range"]),
    )
    state[EXPLORER_CRITERIA_KEY] = local
    return local, state["ex_sort"]


def _product_detail(row: pd.Series, config) -> None:
    st.markdown(
        f"### {row['Product_Name']}  \n`{row['SKU_ID']}` · {row['Category']} &nbsp; "
        f"{status_badge(row['Health_Class'])} {status_badge(row['Alert_Level'])}",
        unsafe_allow_html=True,
    )
    cols = st.columns(6)
    cols[0].metric("Stock", format_number(row["Current_Stock"]))
    cols[1].metric("Daily sales", format_number(row["Avg_Daily_Sales"], 2))
    cols[2].metric("Days of cover", format_days(pd.Series([row["Days_Of_Inventory"]])).iloc[0])
    cols[3].metric("Health score", format_number(row["Health_Score"], 1))
    cols[4].metric("Reorder qty", format_number(row["Reorder_Quantity"]))
    cols[5].metric("Inventory value", format_currency(row["Inventory_Value"]))

    left, right = st.columns([3, 2])
    with left:
        st.markdown("**Health score breakdown**")
        breakdown = explain_health_score(row, config)
        st.dataframe(
            breakdown, hide_index=True, width="stretch",
            column_config={
                "Component Score (0-100)": st.column_config.ProgressColumn(
                    "Component score", min_value=0, max_value=100, format="%.0f"),
                "Effective Weight": st.column_config.NumberColumn(format="percent"),
                "Weighted Points": st.column_config.NumberColumn(format="%.1f"),
            },
        )
        if row.get("Health_Note"):
            st.caption(row["Health_Note"])
    with right:
        st.markdown("**Assessment**")
        st.markdown(
            f"- **Risk:** {row['Risk_Category']} ({row['Risk_Level']})\n"
            f"- **Action:** {row['Recommended_Action']}\n"
            f"- **Why:** {row['Action_Rationale']}\n"
            f"- **Alert:** {row['Alert_Reason']}\n"
            f"- **Velocity basis:** {row['Velocity_Basis']}\n"
            f"- **Demand trend:** {row['Demand_Trend']}"
        )
        if row.get("Reorder_Note"):
            st.caption(row["Reorder_Note"])
        if row.get("Data_Quality_Notes"):
            st.warning(f"Data quality: {row['Data_Quality_Notes']}", icon="⚠️")


def render() -> None:
    page_header(
        "Product Explorer",
        "Search, combine filters and drill into any product. Sidebar filters also apply.",
    )
    ctx = get_context()
    if not require_data(ctx) or not context_bar(ctx):
        return

    local, sort_option = _filter_controls(ctx.metrics)
    results = sort_products(apply_filters(ctx.filtered, local), sort_option)

    info, actions = st.columns([3, 2])
    info.caption(
        f"**{len(results):,}** product(s) match · {local.active_count} explorer filter(s) · "
        "select a row to see its details"
    )
    with actions:
        b1, b2 = st.columns(2)
        b1.button("Clear explorer filters", on_click=_reset_explorer, width="stretch",
                  disabled=local.active_count == 0 and sort_option == "Highest risk")
        b2.download_button(
            "⬇️ Export results",
            data=to_csv_bytes(prepare_report(results, "filtered")),
            file_name=report_filename("explorer", filtered=True),
            mime="text/csv",
            width="stretch",
            disabled=results.empty,
            key="ex_download",
        )

    if results.empty:
        st.info("No products match these filters.")
        return
    selected = render_table(results, EXPLORER_COLUMNS, key="ex_table",
                            height=min(440, 38 + 35 * (len(results) + 1)), selectable=True)
    st.divider()
    if selected is not None and selected < len(results):
        _product_detail(results.iloc[selected], ctx.config)
    else:
        st.caption("Select a product in the table to see its health score breakdown and recommendation.")
