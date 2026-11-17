"""Sales velocity, revenue, ABC classification and demand-trend analytics.

Velocity from aggregate data
----------------------------
The inventory file only provides an aggregate ``Units_Sold`` figure. Average
daily sales are therefore computed as::

    Avg_Daily_Sales = Units_Sold / Sales_Window_Days

where ``Sales_Window_Days`` comes from the optional ``Sales_Period_Days``
column, or from the configurable assumed window (default 30 days). The
``Velocity_Basis`` column records which source was used for every row so the
UI can disclose it. This is an *average rate*, not an observed daily trend.

Demand trends
-------------
Real trends are only reported when a daily sales-history file is uploaded.
For each SKU, a least-squares line is fitted to daily units over the history
calendar (days without a record count as zero sales), and the slope is
expressed as a weekly percentage of the mean daily demand.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import ABC_CLASSES, AnalyticsConfig
from src.utils import clean_float

DAYS_PER_WEEK = 7
DAYS_PER_MONTH = 30
ABC_A_SHARE = 0.80
ABC_B_SHARE = 0.95
NO_HISTORY = "No History"
INSUFFICIENT_HISTORY = "Insufficient History"


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Element-wise division returning NaN where the denominator is 0 or NaN."""
    denominator = denominator.astype("float64")
    return numerator.astype("float64") / denominator.where(denominator != 0)


def days_of_inventory(stock: pd.Series, daily_sales: pd.Series) -> pd.Series:
    """Estimated days until stock runs out at the current sales rate.

    * stock == 0                 -> 0
    * stock > 0, daily_sales == 0 -> +inf (no demand, stock never depletes)
    * any input missing          -> NaN
    """
    stock = stock.astype("float64")
    daily_sales = daily_sales.astype("float64")
    result = clean_float(safe_divide(stock, daily_sales))  # 29.999999999999996 → 30.0
    result = result.mask((daily_sales == 0) & (stock > 0), np.inf)
    result = result.mask(stock == 0, 0.0)
    return result.mask(stock.isna() | daily_sales.isna())


def compute_sales_velocity(frame: pd.DataFrame, config: AnalyticsConfig) -> pd.DataFrame:
    """Add velocity, revenue and coverage columns to a validated inventory frame.

    Added columns: ``Sales_Window_Days``, ``Velocity_Basis``, ``Avg_Daily_Sales``,
    ``Avg_Weekly_Sales``, ``Avg_Monthly_Sales``, ``Revenue``, ``Inventory_Value``,
    ``Days_Of_Inventory``, ``Unit_Value``, ``Valuation_Basis`` and — when a
    ``Unit_Cost`` column is supplied — ``Inventory_Cost_Value``, ``Gross_Profit``
    and ``Gross_Margin_Pct``.

    ``Inventory_Value`` is always at selling price (retail value). ``Unit_Value``
    is the per-unit value used for working-capital measures (excess stock,
    reorder value): purchase cost where known, otherwise selling price, with
    the basis recorded in ``Valuation_Basis``.
    """
    out = frame.copy()
    assumed = float(config.sales_window_days)
    if "Sales_Period_Days" in out.columns:
        reported = out["Sales_Period_Days"].astype("float64")
        out["Sales_Window_Days"] = reported.fillna(assumed)
        out["Velocity_Basis"] = np.where(
            reported.notna(), "Reported period", f"Assumed {config.sales_window_days}-day window"
        )
    else:
        out["Sales_Window_Days"] = assumed
        out["Velocity_Basis"] = f"Assumed {config.sales_window_days}-day window"

    daily = safe_divide(out["Units_Sold"], out["Sales_Window_Days"])
    out["Avg_Daily_Sales"] = daily
    out["Avg_Weekly_Sales"] = daily * DAYS_PER_WEEK
    out["Avg_Monthly_Sales"] = daily * DAYS_PER_MONTH
    out["Revenue"] = out["Units_Sold"] * out["Unit_Price"]
    out["Inventory_Value"] = out["Current_Stock"] * out["Unit_Price"]
    out["Days_Of_Inventory"] = days_of_inventory(out["Current_Stock"], daily)

    if "Unit_Cost" in out.columns:
        cost = out["Unit_Cost"].astype("float64")
        price = out["Unit_Price"].astype("float64")
        out["Inventory_Cost_Value"] = out["Current_Stock"] * cost
        out["Gross_Profit"] = out["Units_Sold"] * (price - cost)
        out["Gross_Margin_Pct"] = (price - cost) / price.where(price > 0)
        out["Unit_Value"] = cost.fillna(price)
        out["Valuation_Basis"] = np.where(cost.notna(), "Cost", "Price (cost missing)")
    else:
        out["Unit_Value"] = out["Unit_Price"]
        out["Valuation_Basis"] = "Price (no cost data)"
    return out


def classify_abc(revenue: pd.Series) -> pd.Series:
    """Pareto (ABC) classification by revenue.

    Products are ranked by revenue; those making up the first 80 % of total
    revenue are class A, the next 15 % class B and the remainder class C.
    Products with unknown revenue are left as NaN. If total revenue is zero,
    every product is class C.
    """
    result = pd.Series(pd.NA, index=revenue.index, dtype="object")
    known = revenue.dropna().astype("float64")
    if known.empty:
        return result
    total = known.sum()
    if total <= 0:
        result.loc[known.index] = ABC_CLASSES[2]
        return result
    ordered = known.sort_values(ascending=False, kind="mergesort")
    share_before = (ordered.cumsum() - ordered) / total
    classes = np.select(
        [share_before < ABC_A_SHARE, share_before < ABC_B_SHARE],
        [ABC_CLASSES[0], ABC_CLASSES[1]],
        default=ABC_CLASSES[2],
    )
    classes = np.where(ordered.to_numpy() <= 0, ABC_CLASSES[2], classes)
    result.loc[ordered.index] = classes
    return result


def category_performance(metrics: pd.DataFrame) -> pd.DataFrame:
    """Aggregate sales and inventory KPIs per category.

    ``Avg_Daily_Sales`` here is the *sum* of product daily rates (the category's
    total daily demand). ``Gross_Profit`` is added when cost data is present.
    """
    if metrics.empty:
        return pd.DataFrame(
            columns=[
                "Category", "Products", "Units_Sold", "Revenue", "Inventory_Units",
                "Inventory_Value", "Avg_Daily_Sales", "Revenue_Share",
            ]
        )
    grouped = (
        metrics.groupby("Category", dropna=False)
        .agg(
            Products=("SKU_ID", "count"),
            Units_Sold=("Units_Sold", "sum"),
            Revenue=("Revenue", "sum"),
            Inventory_Units=("Current_Stock", "sum"),
            Inventory_Value=("Inventory_Value", "sum"),
            Avg_Daily_Sales=("Avg_Daily_Sales", "sum"),
        )
        .reset_index()
    )
    if "Gross_Profit" in metrics.columns:
        grouped = grouped.merge(
            metrics.groupby("Category", dropna=False)["Gross_Profit"].sum(min_count=1).reset_index(),
            on="Category",
        )
    total_revenue = grouped["Revenue"].sum()
    grouped["Revenue_Share"] = (
        grouped["Revenue"] / total_revenue if total_revenue > 0 else 0.0
    )
    return grouped.sort_values("Revenue", ascending=False, kind="mergesort").reset_index(
        drop=True
    )


def top_products(
    metrics: pd.DataFrame, column: str, n: int = 10, ascending: bool = False
) -> pd.DataFrame:
    """Return the ``n`` products with the highest (or lowest) value of ``column``.

    Rows where ``column`` is missing are excluded. Ties are broken by SKU so the
    result is deterministic.
    """
    valid = metrics.dropna(subset=[column])
    return valid.sort_values(
        [column, "SKU_ID"], ascending=[ascending, True], kind="mergesort"
    ).head(n)


def slowest_movers(metrics: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Products *holding stock* with the lowest average daily sales.

    Products with zero stock are excluded (they are a replenishment issue, not
    a slow-moving one). Ties are broken by higher inventory value first, since
    those tie up the most capital.
    """
    holding = metrics[(metrics["Current_Stock"] > 0) & metrics["Avg_Daily_Sales"].notna()]
    return holding.sort_values(
        ["Avg_Daily_Sales", "Inventory_Value", "SKU_ID"],
        ascending=[True, False, True],
        kind="mergesort",
    ).head(n)


# --------------------------------------------------------------------------- #
# Daily history
# --------------------------------------------------------------------------- #


def history_matrix(history: pd.DataFrame) -> pd.DataFrame:
    """Pivot history into a SKU x Date matrix covering every calendar day.

    Days without a record for a SKU are filled with zero sales.
    """
    if history.empty:
        return pd.DataFrame()
    full_range = pd.date_range(history["Date"].min(), history["Date"].max(), freq="D")
    matrix = history.pivot_table(
        index="SKU_ID", columns="Date", values="Units_Sold", aggfunc="sum", fill_value=0.0
    )
    return matrix.reindex(columns=full_range, fill_value=0.0)


def compute_demand_trends(history: pd.DataFrame, config: AnalyticsConfig) -> pd.DataFrame:
    """Fit a linear demand trend per SKU from daily sales history.

    Returns one row per SKU with ``History_Days``, ``History_Units``,
    ``History_Avg_Daily``, ``Trend_Pct_Per_Week`` and ``Demand_Trend``
    (Growing / Stable / Declining / Insufficient History).
    """
    columns = [
        "SKU_ID", "History_Days", "History_Units", "History_Avg_Daily", "Demand_Std_Daily",
        "Trend_Pct_Per_Week", "Demand_Trend",
    ]
    matrix = history_matrix(history)
    if matrix.empty:
        return pd.DataFrame(columns=columns)

    values = matrix.to_numpy(dtype="float64")
    n_days = values.shape[1]
    mean_daily = values.mean(axis=1)
    result = pd.DataFrame(
        {
            "SKU_ID": matrix.index.astype(str),
            "History_Days": n_days,
            "History_Units": values.sum(axis=1),
            "History_Avg_Daily": mean_daily,
            "Demand_Std_Daily": values.std(axis=1, ddof=1) if n_days > 1 else np.nan,
        }
    )

    if n_days < config.min_history_days:
        result["Trend_Pct_Per_Week"] = np.nan
        result["Demand_Trend"] = INSUFFICIENT_HISTORY
        return result[columns]

    x = np.arange(n_days, dtype="float64")
    x_centered = x - x.mean()
    slope = (values - mean_daily[:, None]) @ x_centered / np.sum(x_centered**2)
    with np.errstate(divide="ignore", invalid="ignore"):
        pct_per_week = np.where(mean_daily > 0, slope * DAYS_PER_WEEK / mean_daily * 100, np.nan)
    threshold = config.trend_threshold_pct
    trend = np.select(
        [np.isnan(pct_per_week), pct_per_week > threshold, pct_per_week < -threshold],
        [INSUFFICIENT_HISTORY, "Growing", "Declining"],
        default="Stable",
    )
    result["Trend_Pct_Per_Week"] = pct_per_week
    result["Demand_Trend"] = trend
    return result[columns]


def sales_time_series(
    history: pd.DataFrame, skus: list[str] | None = None, freq: str = "W"
) -> pd.DataFrame:
    """Total units sold per period (``D``aily or ``W``eekly) for the given SKUs."""
    if history.empty:
        return pd.DataFrame(columns=["Date", "Units_Sold"])
    subset = history if skus is None else history[history["SKU_ID"].isin(skus)]
    if subset.empty:
        return pd.DataFrame(columns=["Date", "Units_Sold"])
    daily = subset.groupby("Date")["Units_Sold"].sum()
    full_range = pd.date_range(history["Date"].min(), history["Date"].max(), freq="D")
    daily = daily.reindex(full_range, fill_value=0.0)
    if freq == "D":
        series = daily
    else:
        # Only complete 7-day weeks are kept so partial edge weeks don't look like dips.
        weekly = daily.resample("W-MON", label="left", closed="left")
        series = weekly.sum()[weekly.count() == DAYS_PER_WEEK]
    return series.rename_axis("Date").reset_index(name="Units_Sold")
