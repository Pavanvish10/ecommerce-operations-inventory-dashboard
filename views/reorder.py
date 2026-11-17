"""Reorder Management: dynamic alerts and replenishment recommendations."""

from __future__ import annotations

import streamlit as st

from components import charts
from components.alerts import page_header
from components.kpi_cards import format_currency, format_number
from components.state import get_context
from components.tables import render_table
from src.export_manager import prepare_report, report_filename, to_csv_bytes
from src.reorder_engine import reorder_recommendations
from views.common import context_bar, plot, require_data

REORDER_COLUMNS = [
    "Alert_Level", "SKU_ID", "Product_Name", "Category", "Current_Stock", "Avg_Daily_Sales",
    "Days_Of_Inventory", "Supplier_Lead_Time", "Effective_Reorder_Threshold", "Target_Stock",
    "Reorder_Quantity", "Reorder_Value", "Estimated_Stockout_Date", "Alert_Reason", "Reorder_Note",
]
WATCH_COLUMNS = [
    "Alert_Level", "SKU_ID", "Product_Name", "Current_Stock", "Effective_Reorder_Threshold",
    "Days_Of_Inventory", "Supplier_Lead_Time", "Alert_Reason",
]


def _methodology(config) -> None:
    with st.expander("How reorder alerts and quantities are calculated", icon="🧮"):
        mode_text = {
            "dataset": "the dataset Reorder_Level",
            "dynamic": "the demand-based reorder point (ROP)",
            "max": "the higher of Reorder_Level and the demand-based reorder point",
        }[config.threshold_mode]
        st.markdown(
            f"""<div class="formula">Safety stock        SS  = d × {config.safety_stock_days:g} days
Reorder point       ROP = d × L + SS
Reorder threshold   T   = {mode_text}
Target stock        TS  = max(d × (L + {config.review_period_days:g}) + SS,  T + d × {config.review_period_days:g})
Reorder quantity    Q   = ceil(max(0, TS − Stock))        (only when a reorder is needed)
Stockout date           = as-of date + Stock / d          (assumes constant demand)</div>""",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"- 🔴 **Critical** — demand exists and stock is 0 or covers ≤ {config.critical_cover_days:g} days.\n"
            "- 🟠 **High** — stock runs out before an order placed today arrives (days of cover < lead time).\n"
            f"- 🟡 **Medium** — stock at or within {config.approaching_threshold_buffer:.0%} of the reorder "
            "threshold, or out of stock with no recorded demand.\n"
            "- 🟢 **Healthy** — sufficient inventory.\n"
            "- ⚪ **Insufficient Data** — stock or sales missing; no alert is raised rather than a false one.\n\n"
            "`d` is average daily sales; when the sales period is not supplied it is based on the "
            "assumed sales window, which is disclosed in the *Notes* column. All parameters are "
            "adjustable on the Settings page."
        )


def render() -> None:
    page_header(
        "Reorder Management",
        "Dynamic reorder alerts, stockout estimates and recommended replenishment quantities.",
    )
    ctx = get_context()
    if not require_data(ctx) or not context_bar(ctx):
        return
    data = ctx.filtered
    plan = reorder_recommendations(data)
    counts = data["Alert_Level"].value_counts()

    cols = st.columns(5)
    cols[0].metric("🔴 Critical", int(counts.get("Critical", 0)), help="Out of stock or imminent stockout.")
    cols[1].metric("🟠 High", int(counts.get("High", 0)), help="Stock won't last the supplier lead time.")
    cols[2].metric("🟡 Medium", int(counts.get("Medium", 0)), help="At or approaching the reorder threshold.")
    cols[3].metric("Units to reorder", format_number(plan["Reorder_Quantity"].sum()),
                   help="Sum of recommended reorder quantities.")
    cols[4].metric("Reorder value", format_currency(plan["Reorder_Value"].sum()),
                   help="Reorder quantity × unit price (selling price; purchase cost is not in the dataset).")
    st.caption(f"Stockout dates are estimated from the as-of date **{ctx.config.reference_date:%d %b %Y}**.")

    _methodology(ctx.config)

    left, right = st.columns([3, 2])
    with left:
        # Out-of-stock items have zero cover by definition and are listed in the table below;
        # the chart ranks in-stock items by how far their cover falls short of the lead time.
        at_risk = plan[plan["Alert_Level"].isin(["Critical", "High"]) & (plan["Current_Stock"] > 0)]
        gap = at_risk["Supplier_Lead_Time"] - at_risk["Days_Of_Inventory"]
        urgent = at_risk.assign(_gap=gap).sort_values(["_gap", "SKU_ID"], ascending=[False, True]).head(20)
        plot(charts.cover_vs_lead_time(urgent, "Largest Cover Shortfalls (in stock): Days of Cover vs Lead Time"),
             "ro_cover")
    with right:
        plot(charts.alert_level_bar(data), "ro_alerts")

    st.subheader(f"Reorder recommendations ({len(plan):,})")
    if plan.empty:
        st.success("No products currently need replenishment.", icon="✅")
    else:
        levels = st.pills(
            "Alert level", ["Critical", "High", "Medium", "Healthy"], selection_mode="multi",
            key="ro_levels", help="Narrow the recommendation list by alert level.",
        )
        shown = plan[plan["Alert_Level"].isin(levels)] if levels else plan
        render_table(shown, REORDER_COLUMNS, key="ro_table", height=460)
        st.download_button(
            "⬇️ Download reorder report (CSV)",
            data=to_csv_bytes(prepare_report(shown, "reorder")),
            file_name=report_filename("reorder", filtered=bool(levels) or ctx.criteria.active_count > 0),
            mime="text/csv",
            key="ro_download",
        )

    watch = data[(data["Alert_Level"] == "Medium") & ~data["Needs_Reorder"]]
    if not watch.empty:
        st.subheader(f"Watch list — approaching threshold ({len(watch):,})")
        st.caption("Not yet below the threshold, so no order is recommended yet.")
        render_table(watch.sort_values("Days_Of_Inventory"), WATCH_COLUMNS, key="ro_watch")

    insufficient = data[data["Alert_Level"] == "Insufficient Data"]
    if not insufficient.empty:
        with st.expander(f"Products not evaluated — insufficient data ({len(insufficient):,})"):
            render_table(insufficient, ["SKU_ID", "Product_Name", "Data_Quality_Notes"], key="ro_insufficient")
