"""Session state, caching and the per-run page context.

Raw uploaded bytes are kept in ``st.session_state``; validation and metric
computation are memoised with ``st.cache_data`` keyed on those bytes and on the
configuration, so changing a filter never re-parses or re-scores the dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import AnalyticsConfig
from src.data_loader import DataLoadError, load_csv
from src.data_validator import (
    ValidationIssue,
    ValidationResult,
    validate_inventory,
    validate_sales_history,
)
from src.filters import FilterCriteria, apply_filters
from src.inventory_engine import build_inventory_metrics

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_INVENTORY = ROOT / "data" / "sample_inventory.csv"
SAMPLE_HISTORY = ROOT / "data" / "sample_sales_history.csv"
SAMPLE_NAME = "Sample dataset (synthetic)"

_CONTEXT_KEY = "_page_context"


@dataclass
class PageContext:
    """Everything a page needs for the current run."""

    config: AnalyticsConfig
    inventory: ValidationResult | None
    history: ValidationResult | None
    metrics: pd.DataFrame | None
    filtered: pd.DataFrame | None
    criteria: FilterCriteria
    source_name: str
    is_sample: bool
    history_name: str | None

    @property
    def has_data(self) -> bool:
        return self.metrics is not None and not self.metrics.empty

    @property
    def has_history(self) -> bool:
        return self.history is not None and self.history.is_valid


def _load_error_result(message: str) -> ValidationResult:
    return ValidationResult(
        data=pd.DataFrame(), rejected=pd.DataFrame(),
        issues=[ValidationIssue("error", message)], input_rows=0,
    )


@st.cache_data(show_spinner="Validating dataset…", max_entries=8)
def validate_inventory_bytes(raw: bytes, duplicate_strategy: str, reference: date) -> ValidationResult:
    """Parse and validate inventory CSV bytes (cached)."""
    try:
        frame = load_csv(raw)
    except DataLoadError as exc:
        return _load_error_result(str(exc))
    return validate_inventory(frame, duplicate_strategy, reference)


@st.cache_data(show_spinner="Validating sales history…", max_entries=8)
def validate_history_bytes(
    raw: bytes, known_skus: tuple[str, ...], reference: date
) -> ValidationResult:
    """Parse and validate sales-history CSV bytes (cached)."""
    try:
        frame = load_csv(raw)
    except DataLoadError as exc:
        return _load_error_result(str(exc))
    return validate_sales_history(frame, known_skus, reference)


@st.cache_data(show_spinner="Computing inventory analytics…", max_entries=16)
def compute_metrics(
    data: pd.DataFrame, config_dict: dict, history: pd.DataFrame | None
) -> pd.DataFrame:
    """Run the full analytics pipeline (cached on data + configuration)."""
    return build_inventory_metrics(data, AnalyticsConfig.from_dict(config_dict), history)


def init_session() -> None:
    """Initialise configuration and load the sample dataset on first visit."""
    state = st.session_state
    if "config" not in state:
        state.config = AnalyticsConfig()
    if "inventory_bytes" not in state:
        load_sample_data()


def load_sample_data() -> None:
    """Replace the current datasets with the bundled sample data."""
    state = st.session_state
    state.inventory_bytes = SAMPLE_INVENTORY.read_bytes()
    state.inventory_name = SAMPLE_NAME
    state.is_sample = True
    state.history_bytes = SAMPLE_HISTORY.read_bytes() if SAMPLE_HISTORY.exists() else None
    state.history_name = "Sample sales history (synthetic)" if state.history_bytes else None


def set_inventory(raw: bytes, name: str) -> None:
    """Make an uploaded (already validated) inventory file the active dataset."""
    state = st.session_state
    state.inventory_bytes = raw
    state.inventory_name = name
    if state.get("is_sample"):
        # Sample history SKUs would not match a user's catalogue.
        state.history_bytes = None
        state.history_name = None
    state.is_sample = False


def set_history(raw: bytes | None, name: str | None) -> None:
    """Set or clear the daily sales-history dataset."""
    st.session_state.history_bytes = raw
    st.session_state.history_name = name


def build_context(criteria_factory) -> PageContext:
    """Validate data, compute metrics, render sidebar filters and store the context.

    Args:
        criteria_factory: Callable receiving the full metrics DataFrame (or
            ``None``) and returning the active :class:`FilterCriteria`; it is
            responsible for rendering the sidebar filter widgets.
    """
    state = st.session_state
    config: AnalyticsConfig = state.config
    reference = config.reference_date
    inventory = validate_inventory_bytes(state.inventory_bytes, config.duplicate_strategy, reference)

    history = None
    metrics = None
    if inventory.is_valid:
        if state.get("history_bytes"):
            skus = tuple(inventory.data["SKU_ID"].astype(str))
            history = validate_history_bytes(state.history_bytes, skus, reference)
        history_data = history.data if history is not None and history.is_valid else None
        metrics = compute_metrics(inventory.data, config.to_dict(), history_data)

    criteria = criteria_factory(metrics)
    filtered = None
    if metrics is not None:

        filtered = apply_filters(metrics, criteria)

    context = PageContext(
        config=config,
        inventory=inventory,
        history=history,
        metrics=metrics,
        filtered=filtered,
        criteria=criteria,
        source_name=state.inventory_name,
        is_sample=bool(state.get("is_sample")),
        history_name=state.get("history_name"),
    )
    state[_CONTEXT_KEY] = context
    return context


def get_context() -> PageContext:
    """Context built by ``app.py`` for the current run."""
    return st.session_state[_CONTEXT_KEY]


def restore_widget(key: str, default, options: list | None = None) -> None:
    """Re-seed a widget's value from persistent storage before it is rendered.

    Streamlit discards widget state when a page is not rendered; this keeps
    selections when users navigate away and back. Values no longer present in
    ``options`` (e.g. after a dataset change) are dropped.
    """
    store_key = f"_persist_{key}"
    if key in st.session_state:
        value = st.session_state[key]
    else:
        value = st.session_state.get(store_key, default)
    if options is not None and isinstance(value, list):
        value = [v for v in value if v in options]
    st.session_state[key] = value


def save_widget(key: str) -> None:
    """Persist a widget's current value (call after rendering it)."""
    st.session_state[f"_persist_{key}"] = st.session_state[key]
