"""Settings: every business parameter used by the analytics engines."""

from __future__ import annotations

from datetime import date

import streamlit as st

from components.alerts import page_header
from components.state import get_context
from src.config import AnalyticsConfig, HealthWeights

THRESHOLD_MODES = {
    "max": "Higher of dataset Reorder_Level and demand-based reorder point (recommended)",
    "dataset": "Dataset Reorder_Level only",
    "dynamic": "Demand-based reorder point only",
}


def render() -> None:
    page_header(
        "Settings",
        "Tune the business rules. Every KPI, score, alert and report recalculates immediately.",
    )
    config: AnalyticsConfig = get_context().config
    weights = config.health_weights

    with st.form("settings_form"):
        st.subheader("Sales velocity")
        window = st.number_input(
            "Assumed sales window (days)", 1, 365, int(config.sales_window_days),
            help="Days covered by Units_Sold when the file has no Sales_Period_Days column.",
        )
        use_today = st.checkbox("Use today's date as the as-of date", value=config.as_of_date is None)
        as_of = st.date_input(
            "As-of date (used when the box above is unchecked)", value=config.reference_date,
            help="Reference date for stockout estimates and days since restock.",
        )

        st.subheader("Reorder engine")
        c1, c2, c3 = st.columns(3)
        safety = c1.number_input("Safety stock (days of demand)", 0.0, 120.0, float(config.safety_stock_days), 1.0)
        review = c2.number_input("Review period (days)", 0.0, 180.0, float(config.review_period_days), 1.0,
                                 help="Days of demand an order should cover beyond the lead time.")
        critical = c3.number_input("Critical cover (days)", 0.0, 60.0, float(config.critical_cover_days), 0.5,
                                   help="Days of stock at or below which stockout is imminent.")
        c1, c2 = st.columns(2)
        ss_method = c1.radio(
            "Safety stock method", ["days", "statistical"],
            index=["days", "statistical"].index(config.safety_stock_method),
            format_func=lambda v: {
                "days": "Days of cover (d × safety days)",
                "statistical": "Statistical z·√(L·σd² + d²·σL²) — needs sales history",
            }[v],
            help="Statistical uses demand variability from the daily sales history and the optional "
                 "Lead_Time_Std_Days column. Products without enough history fall back to days of cover.",
        )
        service_pct = c2.slider(
            "Service level (statistical method)", 50.0, 99.9, float(config.service_level * 100), 0.5,
            format="%.1f%%", help="Probability of not stocking out during a replenishment cycle.",
        )
        c1, c2 = st.columns(2)
        buffer_pct = c1.slider("'Approaching threshold' buffer (%)", 0, 100,
                               int(round(config.approaching_threshold_buffer * 100)), 5)
        mode = c2.selectbox("Reorder threshold", list(THRESHOLD_MODES), format_func=THRESHOLD_MODES.get,
                            index=list(THRESHOLD_MODES).index(config.threshold_mode))

        st.subheader("Inventory Health Score")
        c1, c2 = st.columns(2)
        target = c1.number_input("Target coverage (days)", 1.0, 365.0, float(config.target_cover_days), 1.0)
        overstock = c2.number_input("Overstock threshold (days)", 2.0, 1000.0, float(config.overstock_cover_days), 5.0)
        st.caption("Component weights (normalised automatically to 100%)")
        w1, w2, w3, w4 = st.columns(4)
        w_avail = w1.slider("Stock availability", 0, 100, int(round(weights.stock_availability * 100)))
        w_cover = w2.slider("Inventory coverage", 0, 100, int(round(weights.inventory_coverage * 100)))
        w_reorder = w3.slider("Reorder compliance", 0, 100, int(round(weights.reorder_compliance * 100)))
        w_lead = w4.slider("Lead-time coverage", 0, 100, int(round(weights.lead_time_coverage * 100)))

        st.subheader("Risk detection & trends")
        c1, c2, c3 = st.columns(3)
        slow = c1.number_input("Slow-moving below (units/day)", 0.0, 1000.0, float(config.slow_moving_velocity), 0.1)
        trend = c2.number_input("Trend threshold (% per week)", 0.0, 100.0, float(config.trend_threshold_pct), 0.5)
        min_hist = c3.number_input("Minimum history (days)", 2, 365, int(config.min_history_days))
        c1, c2 = st.columns(2)
        fc_window = c1.number_input("Forecast fitting window (days)", 7, 365, int(config.forecast_window_days),
                                    help="Most recent days of history each forecasting method uses.")
        fc_holdout = c2.number_input("Forecast backtest holdout (days)", 7, 120, int(config.forecast_holdout_days),
                                     help="Days held out at the end of the history to measure accuracy.")

        st.subheader("Data handling")
        dup = st.radio(
            "When a SKU appears more than once with different values, keep the…",
            ["first", "last"], index=["first", "last"].index(config.duplicate_strategy),
            format_func=lambda v: f"{v} occurrence", horizontal=True,
        )
        submitted = st.form_submit_button("Apply settings", type="primary")

    if submitted:
        try:
            new_config = AnalyticsConfig(
                sales_window_days=int(window),
                safety_stock_days=float(safety),
                review_period_days=float(review),
                critical_cover_days=float(critical),
                approaching_threshold_buffer=buffer_pct / 100,
                threshold_mode=mode,
                safety_stock_method=ss_method,
                service_level=float(service_pct) / 100,
                target_cover_days=float(target),
                overstock_cover_days=float(overstock),
                health_weights=HealthWeights(w_avail / 100, w_cover / 100, w_reorder / 100, w_lead / 100),
                slow_moving_velocity=float(slow),
                trend_threshold_pct=float(trend),
                min_history_days=int(min_hist),
                forecast_window_days=int(fc_window),
                forecast_holdout_days=int(fc_holdout),
                duplicate_strategy=dup,
                as_of_date=None if use_today else as_of,
            )
        except ValueError as exc:
            st.error(f"Settings not applied: {exc}", icon="⛔")
        else:
            st.session_state.config = new_config
            st.session_state["_settings_saved"] = True
            st.rerun()

    if st.session_state.pop("_settings_saved", False):
        st.success("Settings applied — all pages now use the new parameters.", icon="✅")

    if st.button("Reset to defaults"):
        st.session_state.config = AnalyticsConfig()
        st.session_state["_settings_saved"] = True
        st.rerun()

    with st.expander("Current configuration (JSON)"):
        payload = config.to_dict()
        payload["as_of_date"] = (config.as_of_date or "today")
        if isinstance(payload["as_of_date"], date):
            payload["as_of_date"] = payload["as_of_date"].isoformat()
        st.json(payload)
