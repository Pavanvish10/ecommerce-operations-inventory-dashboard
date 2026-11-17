"""CSV report generation.

Every report is built from the DataFrame the caller passes in, so when the UI
passes the *filtered* metrics, exports automatically respect active filters.
"""

from __future__ import annotations

import re
from datetime import date

import numpy as np
import pandas as pd

from src.reorder_engine import reorder_recommendations
from src.risk_analysis import high_risk_products

INPUT_COLUMNS = [
    "SKU_ID", "Product_Name", "Category", "Current_Stock", "Units_Sold", "Unit_Price",
    "Unit_Cost", "Reorder_Level", "Supplier_Lead_Time", "Lead_Time_Std_Days",
    "Last_Restock_Date", "Sales_Period_Days",
]

REPORT_COLUMNS: dict[str, list[str]] = {
    "full": [
        *INPUT_COLUMNS,
        "Sales_Window_Days", "Velocity_Basis", "Avg_Daily_Sales", "Avg_Weekly_Sales",
        "Avg_Monthly_Sales", "Revenue", "Inventory_Value", "Inventory_Cost_Value", "Gross_Profit",
        "Gross_Margin_Pct", "Valuation_Basis", "ABC_Class", "Days_Of_Inventory",
        "Days_Since_Restock", "Demand_Trend", "Trend_Pct_Per_Week", "Demand_Std_Daily",
        "Score_Stock_Availability", "Score_Inventory_Coverage", "Score_Reorder_Compliance",
        "Score_Lead_Time_Coverage", "Health_Score", "Health_Class", "Health_Note",
        "Stock_Status", "Safety_Stock", "Safety_Stock_Method", "Reorder_Point",
        "Effective_Reorder_Threshold",
        "Target_Stock", "Alert_Level", "Alert_Reason", "Needs_Reorder", "Reorder_Quantity",
        "Reorder_Value", "Estimated_Stockout_Date", "Reorder_Note", "Is_Slow_Moving",
        "Is_Excess", "Excess_Units", "Excess_Inventory_Value", "Risk_Category", "Risk_Level",
        "Recommended_Action", "Action_Rationale", "Data_Quality_Notes",
    ],
    "risk": [
        "SKU_ID", "Product_Name", "Category", "Current_Stock", "Avg_Daily_Sales",
        "Days_Of_Inventory", "Inventory_Value", "Excess_Inventory_Value", "Valuation_Basis",
        "Health_Score", "Health_Class", "Risk_Category", "Risk_Level", "Recommended_Action",
        "Action_Rationale",
    ],
    "reorder": [
        "SKU_ID", "Product_Name", "Category", "Alert_Level", "Alert_Reason", "Current_Stock",
        "Avg_Daily_Sales", "Days_Of_Inventory", "Supplier_Lead_Time", "Safety_Stock",
        "Safety_Stock_Method", "Effective_Reorder_Threshold", "Target_Stock", "Reorder_Quantity",
        "Unit_Price", "Unit_Cost", "Reorder_Value", "Valuation_Basis", "Estimated_Stockout_Date",
        "Velocity_Basis", "Reorder_Note",
    ],
}

#: Rounding applied to derived float columns in exports (inputs are left untouched).
_ROUNDING: dict[str, int] = {
    "Sales_Window_Days": 0, "Avg_Daily_Sales": 3, "Avg_Weekly_Sales": 2, "Avg_Monthly_Sales": 2,
    "Revenue": 2, "Inventory_Value": 2, "Inventory_Cost_Value": 2, "Gross_Profit": 2,
    "Gross_Margin_Pct": 4, "Demand_Std_Daily": 3, "Days_Of_Inventory": 1, "Trend_Pct_Per_Week": 1,
    "Safety_Stock": 1, "Reorder_Point": 1, "Effective_Reorder_Threshold": 1,
    "Target_Stock": 1, "Reorder_Value": 2, "Excess_Inventory_Value": 2,
}


def prepare_report(metrics: pd.DataFrame, report: str) -> pd.DataFrame:
    """Select, order and format the columns of a named report.

    Args:
        metrics: The metrics table (already filtered by the caller).
        report: ``"full"``, ``"risk"``, ``"reorder"`` or ``"filtered"`` (the
            filtered view uses the full column set).
    """
    if report == "risk":
        source = high_risk_products(metrics)
    elif report == "reorder":
        source = reorder_recommendations(metrics)
    elif report in {"full", "filtered"}:
        source = metrics
    else:
        raise ValueError(f"Unknown report type: {report!r}")

    key = "full" if report == "filtered" else report
    columns = [c for c in REPORT_COLUMNS[key] if c in source.columns]
    table = source.loc[:, columns].copy()

    for column, decimals in _ROUNDING.items():
        if column in table.columns:
            table[column] = table[column].round(decimals)
    if "Days_Of_Inventory" in table.columns:
        # 'inf' is not portable in CSV consumers; make the meaning explicit.
        doi = table["Days_Of_Inventory"]
        table["Days_Of_Inventory"] = doi.astype(object).where(~np.isinf(doi.astype(float)), "No demand")
    for column in table.columns:
        if pd.api.types.is_datetime64_any_dtype(table[column]):
            table[column] = table[column].dt.strftime("%Y-%m-%d")
    return table.reset_index(drop=True)


_FORMULA_TRIGGERS = ("=", "+", "@", "\t", "\r")
_PLAIN_NUMBER = re.compile(r"^-\d+(\.\d+)?$")


def neutralize_formulas(frame: pd.DataFrame) -> pd.DataFrame:
    """Defuse CSV/formula injection (OWASP) in text cells.

    Spreadsheet apps execute cells that start with ``=``, ``+``, ``-`` or
    ``@`` (and tab/CR variants). Uploaded product names are untrusted, so any
    such text cell is prefixed with an apostrophe, which spreadsheets display
    but do not evaluate. Plain negative numbers stored as text (``-4``) are
    left unchanged. Numeric columns are never touched.
    """
    out = frame.copy()
    for column in out.columns:
        series = out[column]
        if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
            continue
        # Vectorised: non-string cells (floats, dates, booleans) can never
        # stringify to a formula trigger, so no per-cell type check is needed.
        text = series.astype("string")
        risky = (
            text.str.startswith(_FORMULA_TRIGGERS)
            | (text.str.startswith("-") & ~text.str.fullmatch(_PLAIN_NUMBER))
        ).fillna(False).astype(bool)
        if risky.any():
            out[column] = series.astype(object).mask(risky, "'" + text)
    return out


def to_csv_bytes(frame: pd.DataFrame, sanitize: bool = True) -> bytes:
    """Serialise a DataFrame to UTF-8 CSV bytes (no index).

    With ``sanitize`` (the default) text cells are protected against CSV
    formula injection, see :func:`neutralize_formulas`.
    """
    data = neutralize_formulas(frame) if sanitize else frame
    return data.to_csv(index=False).encode("utf-8")


def report_filename(report: str, filtered: bool, today: date | None = None) -> str:
    """Build a descriptive, date-stamped file name for a report."""
    stamp = (today or date.today()).isoformat()
    suffix = "_filtered" if filtered else ""
    return f"inventory_{report}_report{suffix}_{stamp}.csv"
