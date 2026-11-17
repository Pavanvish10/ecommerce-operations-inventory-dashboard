"""Demand forecasting from daily sales history, with holdout backtesting.

Three transparent baseline methods are provided. Accuracy is never assumed:
:func:`backtest_forecasts` refits each method on the history *excluding* the
last ``forecast_holdout_days`` days, forecasts that period, and scores the
forecast against what actually sold.

Methods (``W`` = ``forecast_window_days``, applied per SKU)
------------------------------------------------------------
* **Moving average** – mean daily units over the last ``W`` days, held flat.
* **Linear trend** – least-squares line over the last ``W`` days,
  extrapolated and clipped at zero.
* **Seasonal naive (weekly)** – repeats the last observed 7-day pattern.

Accuracy metrics (over all SKU-days in the holdout)
---------------------------------------------------
* ``WAPE = Σ|actual − forecast| / Σ actual`` – volume-weighted error; robust
  to the many zero-sales days that make MAPE undefined.
* ``Bias = (Σ forecast − Σ actual) / Σ actual`` – positive = over-forecast.
* ``MAE`` – mean absolute error in units per SKU per day.

The forecasts are informational: reorder quantities still use the aggregate
sales rate so that every recommendation stays traceable to the inventory file.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import AnalyticsConfig
from src.sales_analytics import history_matrix

DAYS_PER_WEEK = 7
METHODS: tuple[str, ...] = ("Moving average", "Linear trend", "Seasonal naive (weekly)")


def _forecast(values: np.ndarray, horizon: int, window: int, method: str) -> np.ndarray:
    """Daily forecasts of shape (n_sku, horizon) from a (n_sku, n_days) matrix."""
    recent = values[:, -window:]
    n_sku, n_days = recent.shape
    if method == "Moving average":
        return np.repeat(recent.mean(axis=1, keepdims=True), horizon, axis=1)
    if method == "Linear trend":
        x = np.arange(n_days, dtype="float64")
        x_centered = x - x.mean()
        mean = recent.mean(axis=1)
        slope = (recent - mean[:, None]) @ x_centered / np.sum(x_centered**2)
        future_x = np.arange(n_days, n_days + horizon) - x.mean()
        return np.clip(mean[:, None] + slope[:, None] * future_x[None, :], 0, None)
    if method == "Seasonal naive (weekly)":
        last_week = values[:, -DAYS_PER_WEEK:]
        reps = int(np.ceil(horizon / DAYS_PER_WEEK))
        return np.tile(last_week, reps)[:, :horizon]
    raise ValueError(f"Unknown forecasting method: {method!r}")


def required_history_days(config: AnalyticsConfig) -> int:
    """Minimum history length for a backtest (fit window + holdout)."""
    return config.forecast_window_days + config.forecast_holdout_days


def backtest_forecasts(history: pd.DataFrame, config: AnalyticsConfig) -> pd.DataFrame:
    """Score every method on a holdout period at the end of the history.

    Returns one row per method with ``WAPE``, ``Bias``, ``MAE`` and the
    evaluation window, sorted best (lowest WAPE) first. Returns an empty frame
    when the history is too short or contains no sales in the holdout.
    """
    columns = ["Method", "WAPE", "Bias", "MAE", "SKUs", "Holdout_Days", "Holdout_Start", "Holdout_End"]
    matrix = history_matrix(history)
    if matrix.empty or matrix.shape[1] < required_history_days(config):
        return pd.DataFrame(columns=columns)

    values = matrix.to_numpy(dtype="float64")
    holdout = config.forecast_holdout_days
    train, actual = values[:, :-holdout], values[:, -holdout:]
    total_actual = actual.sum()
    if total_actual <= 0:
        return pd.DataFrame(columns=columns)

    rows = []
    for method in METHODS:
        predicted = _forecast(train, holdout, config.forecast_window_days, method)
        error = np.abs(actual - predicted)
        rows.append(
            {
                "Method": method,
                "WAPE": error.sum() / total_actual,
                "Bias": (predicted.sum() - total_actual) / total_actual,
                "MAE": error.mean(),
                "SKUs": values.shape[0],
                "Holdout_Days": holdout,
                "Holdout_Start": matrix.columns[-holdout],
                "Holdout_End": matrix.columns[-1],
            }
        )
    return pd.DataFrame(rows, columns=columns).sort_values("WAPE", kind="mergesort").reset_index(drop=True)


def forecast_demand(
    history: pd.DataFrame, config: AnalyticsConfig, method: str, horizon: int = 30
) -> pd.DataFrame:
    """Forecast the next ``horizon`` days per SKU using all available history.

    Returns ``SKU_ID``, ``Forecast_Daily`` (mean forecast units/day over the
    horizon) and ``Forecast_Total``. Empty when the history is shorter than
    the fitting window.
    """
    columns = ["SKU_ID", "Forecast_Daily", "Forecast_Total"]
    matrix = history_matrix(history)
    if matrix.empty or matrix.shape[1] < config.forecast_window_days:
        return pd.DataFrame(columns=columns)
    predicted = _forecast(matrix.to_numpy(dtype="float64"), horizon, config.forecast_window_days, method)
    return pd.DataFrame(
        {
            "SKU_ID": matrix.index.astype(str),
            "Forecast_Daily": predicted.mean(axis=1),
            "Forecast_Total": predicted.sum(axis=1),
        }
    )
