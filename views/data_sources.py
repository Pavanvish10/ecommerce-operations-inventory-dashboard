"""Data Sources: upload, validate, preview and replace datasets."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from components.alerts import page_header, render_validation_report
from components.state import (
    get_context,
    load_sample_data,
    set_history,
    set_inventory,
    validate_history_bytes,
    validate_inventory_bytes,
)
from src.config import HISTORY_REQUIRED_COLUMNS, OPTIONAL_COLUMNS, REQUIRED_COLUMNS
from src.export_manager import to_csv_bytes

_TYPE_LABELS = {
    "text": "Text", "non_negative_number": "Number ≥ 0", "positive_number": "Number > 0",
    "date": "Date (YYYY-MM-DD)",
}


_PREVIEW_CONFIG = {
    "Last_Restock_Date": st.column_config.DateColumn("Last_Restock_Date", format="YYYY-MM-DD"),
}


def _schema_table() -> pd.DataFrame:
    rows = [
        {"Column": name, "Required": "Yes", "Type": _TYPE_LABELS[kind], "Description": text}
        for name, (kind, text) in REQUIRED_COLUMNS.items()
    ] + [
        {"Column": name, "Required": "Optional", "Type": _TYPE_LABELS[kind], "Description": text}
        for name, (kind, text) in OPTIONAL_COLUMNS.items()
    ]
    return pd.DataFrame(rows)


def _template_csv() -> bytes:
    template = pd.DataFrame(
        [["SKU-0001", "Example Product", "Example Category", 120, 90, 19.99, 40, 7, "2026-10-01"]],
        columns=[*REQUIRED_COLUMNS, "Last_Restock_Date"],
    )
    return to_csv_bytes(template)


def _upload_inventory(ctx) -> None:
    st.subheader("Inventory dataset")
    uploaded = st.file_uploader("Upload inventory CSV", type=["csv"], key="inventory_upload")
    if uploaded is None:
        return
    raw = uploaded.getvalue()
    result = validate_inventory_bytes(raw, ctx.config.duplicate_strategy, ctx.config.reference_date)
    st.markdown(f"**Validation of `{uploaded.name}`**")
    render_validation_report(result, "Validation details", key="upload_inventory")
    if result.is_valid:
        st.markdown("**Preview** (first 20 cleaned rows)")
        st.dataframe(result.data.head(20), hide_index=True, width="stretch",
                     column_config=_PREVIEW_CONFIG)
        is_active = raw == st.session_state.get("inventory_bytes")
        if is_active:
            st.success("This file is the active dataset.", icon="✅")
        elif st.button(f"Use `{uploaded.name}` for analysis", type="primary", key="apply_inventory"):
            set_inventory(raw, uploaded.name)
            st.rerun()
    else:
        st.error(
            "This file cannot be used until the errors above are fixed. The current dataset "
            "remains active.",
            icon="⛔",
        )


def _upload_history(ctx) -> None:
    st.subheader("Daily sales history (optional)")
    st.caption(
        "Enables real demand-trend analysis. Columns: "
        + ", ".join(f"`{c}`" for c in HISTORY_REQUIRED_COLUMNS)
        + ". One row per SKU per day (multiple rows for the same day are summed)."
    )
    uploaded = st.file_uploader("Upload sales history CSV", type=["csv"], key="history_upload")
    if uploaded is not None and ctx.inventory is not None and ctx.inventory.is_valid:
        raw = uploaded.getvalue()
        skus = tuple(ctx.inventory.data["SKU_ID"].astype(str))
        result = validate_history_bytes(raw, skus, ctx.config.reference_date)
        render_validation_report(result, "History validation details", key="upload_history")
        if result.is_valid:
            if raw == st.session_state.get("history_bytes"):
                st.success("This history file is active.", icon="✅")
            elif st.button(f"Use `{uploaded.name}` as sales history", type="primary", key="apply_history"):
                set_history(raw, uploaded.name)
                st.rerun()
    if st.session_state.get("history_bytes"):
        if st.button("Remove sales history", key="remove_history"):
            set_history(None, None)
            st.rerun()


def render() -> None:
    page_header(
        "Data Sources",
        "Upload, validate and replace the datasets behind the dashboard.",
    )
    ctx = get_context()

    with st.container(border=True):
        left, right = st.columns([3, 1.3])
        with left:
            st.markdown("**Active dataset**")
            label = "🧪 " if ctx.is_sample else "📄 "
            st.markdown(f"{label}{ctx.source_name}" + (" — *synthetic sample data*" if ctx.is_sample else ""))
            if ctx.history_name:
                st.caption(f"Sales history: {ctx.history_name}")
            if ctx.inventory is not None and ctx.inventory.is_valid:
                summary = ctx.inventory.summary()
                st.caption(
                    f"{summary['accepted_rows']:,} products accepted · {summary['rejected_rows']:,} rows "
                    f"excluded · {summary['warnings']:,} warning(s)"
                )
        with right:
            if st.button("Reload sample data", width="stretch", key="reload_sample"):
                load_sample_data()
                st.rerun()

    if ctx.inventory is not None and not ctx.inventory.is_valid:
        st.error("The active dataset is not valid:", icon="⛔")
        render_validation_report(ctx.inventory, "Active dataset", key="active_invalid")
    elif ctx.inventory is not None and (ctx.inventory.issues or not ctx.inventory.rejected.empty):
        with st.expander("Validation report for the active dataset"):
            render_validation_report(ctx.inventory, "Active dataset", key="active_report")

    _upload_inventory(ctx)
    with st.expander("Expected file format", icon="📋", expanded=ctx.is_sample or not ctx.has_data):
        st.dataframe(
            _schema_table(), hide_index=True, width="stretch",
            column_config={"Description": st.column_config.TextColumn(width="large")},
        )
        st.caption(
            "Column names are case-insensitive; spaces or hyphens are accepted in place of "
            "underscores. Extra columns are kept but ignored. Currency symbols and thousands "
            "separators in numbers are accepted."
        )
        st.download_button("⬇️ Download CSV template", _template_csv(), "inventory_template.csv",
                           "text/csv", key="template_download")

    st.divider()
    _upload_history(ctx)

    if ctx.has_data:
        st.divider()
        st.subheader("Active data preview")
        st.dataframe(ctx.inventory.data.head(50), hide_index=True, width="stretch",
                     column_config=_PREVIEW_CONFIG)
        st.caption(
            f"Showing up to 50 of {len(ctx.inventory.data):,} validated rows. "
            "`Source_Row` is the line number in the original CSV."
        )
