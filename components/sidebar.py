"""Sidebar: dataset status and global filters applied to every page."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from components.state import restore_widget, save_widget
from src.config import ABC_CLASSES, HEALTH_CLASSES, INSUFFICIENT_DATA, RISK_LEVELS, STOCK_STATUSES
from src.filters import FilterCriteria

GLOBAL_FILTER_KEYS = ("gf_categories", "gf_health", "gf_risk", "gf_stock", "gf_abc")


def _reset_filters() -> None:
    for key in GLOBAL_FILTER_KEYS:
        st.session_state[key] = []
        st.session_state[f"_persist_{key}"] = []


def _multiselect(label: str, key: str, options: list[str], help_text: str) -> tuple[str, ...]:
    restore_widget(key, [], options)
    st.multiselect(label, options, key=key, placeholder="All", help=help_text)
    save_widget(key)
    return tuple(st.session_state[key])


def render_global_filters(metrics: pd.DataFrame | None) -> FilterCriteria:
    """Render the global filter widgets and return the resulting criteria."""
    with st.sidebar:
        st.markdown("#### Global filters")
        if metrics is None:
            st.caption("Load a valid dataset to enable filters.")
            return FilterCriteria()
        categories = sorted(metrics["Category"].dropna().unique().tolist())
        health_options = [c for c in [*HEALTH_CLASSES[::-1], INSUFFICIENT_DATA]
                          if c in set(metrics["Health_Class"])]
        risk_options = [r for r in [*RISK_LEVELS, "Unknown"] if r in set(metrics["Risk_Level"])]
        stock_options = [s for s in [*STOCK_STATUSES, "Unknown"] if s in set(metrics["Stock_Status"])]
        abc_options = [a for a in ABC_CLASSES if a in set(metrics["ABC_Class"].dropna())]

        criteria = FilterCriteria(
            categories=_multiselect("Category", "gf_categories", categories,
                                    "Show only these product categories."),
            health_classes=_multiselect("Inventory health", "gf_health", health_options,
                                        "Filter by Inventory Health Score class."),
            risk_levels=_multiselect("Risk level", "gf_risk", risk_options,
                                     "Filter by risk level from the risk engine."),
            stock_statuses=_multiselect("Stock availability", "gf_stock", stock_options,
                                        "Out of Stock, Low Stock (≤ reorder level) or In Stock."),
            abc_classes=_multiselect("Sales performance (ABC)", "gf_abc", abc_options,
                                     "A = top 80% of revenue, B = next 15%, C = last 5%."),
        )
        if criteria.active_count:
            st.button("Reset filters", on_click=_reset_filters, width="stretch")
        return criteria
