"""Interactive product tables with consistent column formatting."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from components.theme import STATUS_ICONS

_MONEY = "$%,.2f"

COLUMN_CONFIG: dict[str, Any] = {
    "SKU_ID": st.column_config.TextColumn("SKU", width="small"),
    "Product_Name": st.column_config.TextColumn("Product", width="medium"),
    "Category": st.column_config.TextColumn("Category"),
    "Current_Stock": st.column_config.NumberColumn("Stock", format="%,.0f"),
    "Units_Sold": st.column_config.NumberColumn("Units Sold", format="%,.0f"),
    "Unit_Price": st.column_config.NumberColumn("Unit Price", format=_MONEY),
    "Unit_Cost": st.column_config.NumberColumn("Unit Cost", format=_MONEY),
    "Inventory_Cost_Value": st.column_config.NumberColumn("Inventory @ Cost", format=_MONEY),
    "Gross_Profit": st.column_config.NumberColumn("Gross Profit", format=_MONEY),
    "Gross_Margin_Pct": st.column_config.NumberColumn("Margin", format="percent"),
    "Valuation_Basis": st.column_config.TextColumn("Valued At"),
    "Safety_Stock_Method": st.column_config.TextColumn("Safety Stock Method"),
    "Demand_Std_Daily": st.column_config.NumberColumn("Demand σ/day", format="%.2f"),
    "Lead_Time_Std_Days": st.column_config.NumberColumn("Lead Time σ (d)", format="%.1f"),
    "Reorder_Level": st.column_config.NumberColumn("Reorder Level", format="%,.0f"),
    "Supplier_Lead_Time": st.column_config.NumberColumn("Lead Time (d)", format="%,.0f"),
    "Last_Restock_Date": st.column_config.DateColumn("Last Restock", format="YYYY-MM-DD"),
    "Days_Since_Restock": st.column_config.NumberColumn("Days Since Restock", format="%,.0f"),
    "Avg_Daily_Sales": st.column_config.NumberColumn("Daily Sales", format="%.2f"),
    "Avg_Weekly_Sales": st.column_config.NumberColumn("Weekly Sales", format="%.1f"),
    "Avg_Monthly_Sales": st.column_config.NumberColumn("Monthly Sales", format="%.1f"),
    "Velocity_Basis": st.column_config.TextColumn("Velocity Basis"),
    "Revenue": st.column_config.NumberColumn("Revenue", format=_MONEY),
    "Inventory_Value": st.column_config.NumberColumn("Inventory Value", format=_MONEY),
    "Days_Of_Inventory": st.column_config.TextColumn("Days of Cover"),
    "ABC_Class": st.column_config.TextColumn("ABC", width="small"),
    "Demand_Trend": st.column_config.TextColumn("Demand Trend"),
    "Trend_Pct_Per_Week": st.column_config.NumberColumn("Trend %/wk", format="%+.1f%%"),
    "Health_Score": st.column_config.ProgressColumn(
        "Health Score", min_value=0, max_value=100, format="%.1f"
    ),
    "Score_Stock_Availability": st.column_config.NumberColumn("Availability", format="%.0f"),
    "Score_Inventory_Coverage": st.column_config.NumberColumn("Coverage", format="%.0f"),
    "Score_Reorder_Compliance": st.column_config.NumberColumn("Reorder Compl.", format="%.0f"),
    "Score_Lead_Time_Coverage": st.column_config.NumberColumn("Lead-Time Cov.", format="%.0f"),
    "Health_Class": st.column_config.TextColumn("Health"),
    "Health_Note": st.column_config.TextColumn("Health Note"),
    "Stock_Status": st.column_config.TextColumn("Stock Status"),
    "Safety_Stock": st.column_config.NumberColumn("Safety Stock", format="%,.1f"),
    "Reorder_Point": st.column_config.NumberColumn("Reorder Point", format="%,.1f"),
    "Effective_Reorder_Threshold": st.column_config.NumberColumn("Reorder Threshold", format="%,.1f"),
    "Target_Stock": st.column_config.NumberColumn("Target Stock", format="%,.1f"),
    "Alert_Level": st.column_config.TextColumn("Alert"),
    "Alert_Reason": st.column_config.TextColumn("Alert Reason", width="large"),
    "Needs_Reorder": st.column_config.CheckboxColumn("Reorder?"),
    "Reorder_Quantity": st.column_config.NumberColumn("Reorder Qty", format="%,.0f"),
    "Reorder_Value": st.column_config.NumberColumn("Reorder Value", format=_MONEY),
    "Estimated_Stockout_Date": st.column_config.DateColumn("Est. Stockout", format="YYYY-MM-DD"),
    "Reorder_Note": st.column_config.TextColumn("Notes", width="large"),
    "Is_Slow_Moving": st.column_config.CheckboxColumn("Slow?"),
    "Is_Excess": st.column_config.CheckboxColumn("Excess?"),
    "Excess_Units": st.column_config.NumberColumn("Excess Units", format="%,.0f"),
    "Excess_Inventory_Value": st.column_config.NumberColumn("Excess Value", format=_MONEY),
    "Risk_Category": st.column_config.TextColumn("Risk Category"),
    "Risk_Level": st.column_config.TextColumn("Risk Level"),
    "Recommended_Action": st.column_config.TextColumn("Recommended Action"),
    "Action_Rationale": st.column_config.TextColumn("Rationale", width="large"),
    "Data_Quality_Notes": st.column_config.TextColumn("Data Quality Notes", width="large"),
}

_STATUS_COLUMNS = ("Alert_Level", "Health_Class", "Risk_Level")


def format_days(series: pd.Series) -> pd.Series:
    """Days of cover as display text: '12.3', '∞ (no sales)' or 'n/a'."""
    def _fmt(value: float) -> str:
        if pd.isna(value):
            return "n/a"
        if np.isinf(value):
            return "∞ (no sales)"
        return f"{value:,.1f}"

    return series.map(_fmt)


def display_frame(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Prepare a display copy: column subset, icons on status labels, readable days."""
    present = [c for c in columns if c in frame.columns]
    view = frame.loc[:, present].copy()
    for column in _STATUS_COLUMNS:
        if column in view.columns:
            view[column] = view[column].map(lambda v: f"{STATUS_ICONS.get(v, '')} {v}".strip())
    if "Days_Of_Inventory" in view.columns:
        view["Days_Of_Inventory"] = format_days(view["Days_Of_Inventory"])
    return view.reset_index(drop=True)


def render_table(
    frame: pd.DataFrame,
    columns: list[str],
    key: str,
    height: int | None = None,
    selectable: bool = False,
) -> int | None:
    """Render a formatted table. Returns the selected row position if selectable."""
    view = display_frame(frame, columns)
    config = {c: COLUMN_CONFIG[c] for c in view.columns if c in COLUMN_CONFIG}
    kwargs: dict[str, Any] = dict(
        hide_index=True, column_config=config, key=key, width="stretch",
    )
    if height is not None:
        kwargs["height"] = height
    if selectable:
        event = st.dataframe(view, on_select="rerun", selection_mode="single-row", **kwargs)
        rows = event.selection.rows if event is not None else []
        return rows[0] if rows else None
    st.dataframe(view, **kwargs)
    return None
