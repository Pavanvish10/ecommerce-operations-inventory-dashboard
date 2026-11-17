"""Reports & Exports: CSV downloads that respect the active filters."""

from __future__ import annotations

import streamlit as st

from components.alerts import page_header
from components.state import get_context
from src.export_manager import prepare_report, report_filename, to_csv_bytes
from src.filters import apply_filters
from views.common import context_bar, require_data
from views.explorer import explorer_criteria

REPORTS = [
    ("full", "📋 Complete inventory report",
     "Every product with all input fields and calculated metrics (velocity, health score "
     "components, reorder plan, risk classification, data-quality notes)."),
    ("risk", "⚠️ Inventory risk report",
     "Products in any risk category other than Healthy, worst first, with recommended action "
     "and rationale."),
    ("reorder", "🚚 Reorder recommendation report",
     "Products that need replenishment, most urgent first, with quantities, stockout dates "
     "and calculation notes."),
]


def _describe_filters(criteria) -> str:
    parts = []
    labels = {
        "categories": "Category", "health_classes": "Health", "risk_levels": "Risk level",
        "stock_statuses": "Stock", "abc_classes": "ABC", "risk_categories": "Risk category",
        "alert_levels": "Alert",
    }
    for attribute, label in labels.items():
        values = getattr(criteria, attribute)
        if values:
            parts.append(f"{label}: {', '.join(values)}")
    if criteria.search.strip():
        parts.append(f"Search: “{criteria.search.strip()}”")
    if tuple(criteria.health_score_range) != (0.0, 100.0):
        low, high = criteria.health_score_range
        parts.append(f"Health score {low:g}–{high:g}")
    return " · ".join(parts) if parts else "None"


def render() -> None:
    page_header(
        "Reports & Exports",
        "Download CSV reports. Exports contain exactly the products selected by the active filters.",
    )
    ctx = get_context()
    if not require_data(ctx) or not context_bar(ctx):
        return

    scope = st.radio(
        "Export scope",
        ["Global filters (sidebar)", "Product Explorer view (sidebar + explorer filters)", "All products (ignore filters)"],
        key="rep_scope",
        horizontal=True,
    )
    if scope.startswith("All"):
        source, criteria_text, filtered = ctx.metrics, "None (all products)", False
    elif scope.startswith("Product Explorer"):
        combined = explorer_criteria(ctx.criteria)
        source = apply_filters(ctx.metrics, combined)
        criteria_text, filtered = _describe_filters(combined), combined.active_count > 0
    else:
        source = ctx.filtered
        criteria_text, filtered = _describe_filters(ctx.criteria), ctx.criteria.active_count > 0
    st.caption(f"**Active filters:** {criteria_text} — **{len(source):,}** of {len(ctx.metrics):,} products in scope.")

    for report, title, description in REPORTS:
        table = prepare_report(source, report)
        with st.container(border=True):
            left, right = st.columns([4, 1.4])
            with left:
                st.markdown(f"**{title}**")
                st.caption(f"{description}  \n{len(table):,} rows · {len(table.columns)} columns")
            with right:
                st.download_button(
                    "⬇️ Download CSV",
                    data=to_csv_bytes(table),
                    file_name=report_filename(report, filtered),
                    mime="text/csv",
                    key=f"rep_dl_{report}",
                    disabled=table.empty,
                    width="stretch",
                )
            with st.expander("Preview first 20 rows"):
                preview = table.head(20)
                if "Days_Of_Inventory" in preview.columns:  # numbers mixed with "No demand"
                    preview = preview.astype({"Days_Of_Inventory": str})
                st.dataframe(preview, hide_index=True, width="stretch")

    if ctx.inventory is not None and not ctx.inventory.rejected.empty:
        with st.container(border=True):
            st.markdown("**🧾 Data validation exceptions**")
            st.caption("Rows excluded during validation, with the reason for each.")
            st.download_button(
                "⬇️ Download excluded rows",
                data=to_csv_bytes(ctx.inventory.rejected),
                file_name="excluded_rows.csv",
                mime="text/csv",
                key="rep_dl_rejected",
            )
