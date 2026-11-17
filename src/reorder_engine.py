"""Dynamic reorder alert and replenishment engine.

Notation (per product)
----------------------
* ``d``  – average daily sales (``Avg_Daily_Sales``)
* ``L``  – supplier lead time in days
* ``S``  – current stock
* ``RL`` – dataset reorder level
* ``SSd`` / ``R`` / ``B`` / ``Cd`` – safety-stock days, review period,
  approaching-threshold buffer and critical-cover days from the config.

Formulas
--------
* Safety stock         ``SS  = d × SSd``  (default "days of cover" method), or
  ``SS = z × √(L·σd² + d²·σL²)`` (statistical method, see :func:`safety_stock`)
* Reorder point        ``ROP = d × L + SS``
* Effective threshold  ``T`` = ``RL`` (dataset mode), ``ROP`` (dynamic mode)
  or ``max(RL, ROP)`` (max mode; a missing input falls back to the other)
* Target stock         ``TS  = max(d × (L + R) + SS,  T + d × R)``
* Reorder quantity     ``Q   = ceil(max(0, TS − S))`` when a reorder is needed
* Days of inventory    ``DOI = S / d``; stockout date = as-of date + DOI

Alert levels (evaluated in order)
---------------------------------
1. **Insufficient Data** – stock or sales velocity unknown; no alert is raised.
2. **Critical** – ``d > 0`` and (``S = 0`` or ``DOI ≤ Cd``).
3. **High** – ``d > 0`` and ``DOI < L`` (stock runs out before an order placed
   today can arrive).
4. **Medium** – ``S ≤ T × (1 + B)`` (at, below, or approaching the threshold),
   including out-of-stock products with no recorded demand.
5. **Healthy** – none of the above.

A reorder is recommended for Critical/High products and any product at or
below its threshold ``T``.
"""

from __future__ import annotations

from datetime import timedelta
from statistics import NormalDist

import numpy as np
import pandas as pd

from src.config import ALERT_LEVELS, INSUFFICIENT_DATA, AnalyticsConfig
from src.utils import join_nonempty, safe_ceil, safe_floor

#: Stockout dates further out than this are not projected: constant-demand
#: extrapolation is meaningless at that range and would overflow date types.
MAX_STOCKOUT_HORIZON_DAYS = 3650
METHOD_DAYS = "Days of cover"

ALERT_PRIORITY: dict[str, int] = {
    **{level: rank for rank, level in enumerate(ALERT_LEVELS)},
    INSUFFICIENT_DATA: len(ALERT_LEVELS),
}


def safety_stock(metrics: pd.DataFrame, config: AnalyticsConfig) -> tuple[pd.Series, pd.Series]:
    """Safety stock per product and the method used for it.

    * **Days of cover** (default): ``SS = d × safety_stock_days``.
    * **Statistical** (``config.safety_stock_method == "statistical"``)::

          SS = z × √(L·σd² + d²·σL²)

      ``z`` is the standard-normal quantile of ``config.service_level``,
      ``σd`` the standard deviation of daily demand from the uploaded sales
      history (``Demand_Std_Daily``), ``σL`` the optional
      ``Lead_Time_Std_Days`` column (0 if absent). This is the standard
      formula for independent demand and lead-time variability. Products
      without enough history, or without a lead time, fall back to the
      days-of-cover method and are labelled accordingly.
    """
    daily = metrics["Avg_Daily_Sales"].astype("float64")
    days_based = daily * config.safety_stock_days
    method = pd.Series(METHOD_DAYS, index=metrics.index, dtype=object)
    if config.safety_stock_method != "statistical":
        return days_based, method

    lead = metrics["Supplier_Lead_Time"].astype("float64")
    sigma_d = metrics.get("Demand_Std_Daily", pd.Series(np.nan, index=metrics.index)).astype("float64")
    history_days = metrics.get("History_Days", pd.Series(np.nan, index=metrics.index)).astype("float64")
    sigma_l = metrics.get("Lead_Time_Std_Days", pd.Series(0.0, index=metrics.index))
    sigma_l = sigma_l.astype("float64").fillna(0.0)
    z = NormalDist().inv_cdf(config.service_level)
    statistical = z * np.sqrt(lead * sigma_d**2 + daily**2 * sigma_l**2)
    eligible = (
        statistical.notna()
        & daily.notna()
        & (history_days >= config.min_history_days)
    )
    method = method.mask(eligible, f"Statistical ({config.service_level:.1%} service level)")
    method = method.mask(~eligible, f"{METHOD_DAYS} (fallback: no usable history or lead time)")
    return statistical.where(eligible, days_based), method


def effective_threshold(
    reorder_level: pd.Series, reorder_point: pd.Series, mode: str
) -> pd.Series:
    """Select the reorder threshold according to ``mode``."""
    if mode == "dataset":
        return reorder_level.astype("float64")
    if mode == "dynamic":
        return reorder_point.astype("float64")
    if mode == "max":
        return pd.Series(
            np.fmax(reorder_level.to_numpy("float64"), reorder_point.to_numpy("float64")),
            index=reorder_level.index,
        )
    raise ValueError(f"Unknown threshold mode: {mode!r}")


def compute_reorder_plan(metrics: pd.DataFrame, config: AnalyticsConfig) -> pd.DataFrame:
    """Add reorder thresholds, alert levels and replenishment quantities.

    ``metrics`` must already contain the columns produced by
    :func:`src.sales_analytics.compute_sales_velocity`.

    Added columns: ``Safety_Stock``, ``Lead_Time_Demand``, ``Reorder_Point``,
    ``Effective_Reorder_Threshold``, ``Target_Stock``, ``Needs_Reorder``,
    ``Reorder_Quantity``, ``Reorder_Value``, ``Estimated_Stockout_Date``,
    ``Alert_Level``, ``Alert_Priority``, ``Alert_Reason``, ``Reorder_Note``.
    """
    out = metrics.copy()
    stock = out["Current_Stock"].astype("float64")
    daily = out["Avg_Daily_Sales"].astype("float64")
    lead = out["Supplier_Lead_Time"].astype("float64")
    reorder_level = out["Reorder_Level"].astype("float64")
    doi = out["Days_Of_Inventory"].astype("float64")
    review = config.review_period_days

    safety, safety_method = safety_stock(out, config)
    lead_demand = daily * lead
    reorder_point = lead_demand + safety
    threshold = effective_threshold(reorder_level, reorder_point, config.threshold_mode)
    demand_target = daily * (lead + review) + safety
    target = pd.Series(
        np.fmax(demand_target.to_numpy(), (threshold + daily * review).to_numpy()),
        index=out.index,
    )

    known = stock.notna() & daily.notna()
    has_demand = known & (daily > 0)
    lead_known = lead.notna()
    threshold_known = threshold.notna() & (threshold > 0)

    critical = has_demand & ((stock == 0) | (doi <= config.critical_cover_days))
    high = has_demand & lead_known & (doi < lead) & ~critical
    at_threshold = known & threshold_known & (stock <= threshold)
    approaching = known & threshold_known & (stock <= threshold * (1 + config.approaching_threshold_buffer))
    dormant_out = known & (stock == 0) & (daily == 0)
    medium = (approaching | dormant_out) & ~critical & ~high

    alert = np.select(
        [~known, critical, high, medium],
        [INSUFFICIENT_DATA, "Critical", "High", "Medium"],
        default="Healthy",
    )
    out["Safety_Stock"] = safety
    out["Safety_Stock_Method"] = safety_method
    out["Lead_Time_Demand"] = lead_demand
    out["Reorder_Point"] = reorder_point
    out["Effective_Reorder_Threshold"] = threshold
    out["Target_Stock"] = target
    out["Alert_Level"] = alert
    out["Alert_Priority"] = out["Alert_Level"].map(ALERT_PRIORITY).astype(int)

    needs = known & (critical | high | at_threshold)
    shortfall = (target - stock).clip(lower=0)
    quantity = safe_ceil(shortfall).where(needs, 0.0)
    out["Needs_Reorder"] = needs
    out["Reorder_Quantity"] = quantity.where(~(needs & target.isna()))  # NaN = cannot compute
    unit_value = out["Unit_Value"] if "Unit_Value" in out.columns else out["Unit_Price"]
    out["Reorder_Value"] = out["Reorder_Quantity"] * unit_value

    reference = pd.Timestamp(config.reference_date)
    projectable = has_demand & np.isfinite(doi) & (doi <= MAX_STOCKOUT_HORIZON_DAYS)
    finite_doi = safe_floor(doi).where(projectable)
    out["Estimated_Stockout_Date"] = reference + pd.to_timedelta(finite_doi, unit="D")

    # Human-readable explanation of each alert
    fmt_doi = doi.round(1).astype(str)
    reason = pd.Series("", index=out.index, dtype=object)
    reason = reason.mask(~known, "Stock or sales data missing – alert not evaluated")
    reason = reason.mask(critical & (stock == 0), "Out of stock with active demand")
    reason = reason.mask(
        critical & (stock > 0),
        "Only " + fmt_doi + " day(s) of stock left (critical ≤ "
        + f"{config.critical_cover_days:g} days)",
    )
    reason = reason.mask(
        high,
        fmt_doi + " day(s) of stock vs " + lead.round(1).astype(str) + "-day supplier lead time",
    )
    reason = reason.mask(medium & at_threshold, "At or below reorder threshold")
    reason = reason.mask(medium & ~at_threshold & approaching, "Approaching reorder threshold")
    reason = reason.mask(
        dormant_out & medium,
        "Out of stock but no sales recorded in the window – verify the product is active",
    )
    reason = reason.mask(reason == "", "Sufficient inventory")
    out["Alert_Reason"] = reason

    note_lead = pd.Series(
        np.where(known & ~lead_known, "Supplier lead time missing – lead-time risk not evaluated", ""),
        index=out.index,
    )
    note_no_demand = pd.Series(
        np.where(
            needs & (daily == 0),
            "No sales in window – quantity only restores the reorder level",
            "",
        ),
        index=out.index,
    )
    note_qty = pd.Series(
        np.where(needs & target.isna(), "Reorder quantity cannot be computed (missing inputs)", ""),
        index=out.index,
    )
    note_basis = pd.Series(
        np.where(
            needs & out["Velocity_Basis"].str.startswith("Assumed"),
            "Demand rate assumes " + out["Velocity_Basis"].str.replace("Assumed ", "", regex=False),
            "",
        ),
        index=out.index,
    )
    note_horizon = pd.Series(
        np.where(
            has_demand & np.isfinite(doi) & (doi > MAX_STOCKOUT_HORIZON_DAYS),
            f"Stockout beyond {MAX_STOCKOUT_HORIZON_DAYS // 365} years – date not projected",
            "",
        ),
        index=out.index,
    )
    out["Reorder_Note"] = join_nonempty([note_lead, note_no_demand, note_qty, note_basis, note_horizon])
    return out


def reorder_recommendations(plan: pd.DataFrame) -> pd.DataFrame:
    """Products that need replenishment, most urgent first."""
    needed = plan[plan["Needs_Reorder"]]
    return needed.sort_values(
        ["Alert_Priority", "Days_Of_Inventory", "SKU_ID"], kind="mergesort"
    ).reset_index(drop=True)


def stockout_horizon_days(plan: pd.DataFrame, reference: pd.Timestamp) -> pd.Series:
    """Days from ``reference`` until each product's estimated stockout."""
    return (plan["Estimated_Stockout_Date"] - reference) / timedelta(days=1)
