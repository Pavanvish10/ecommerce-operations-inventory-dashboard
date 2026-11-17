"""Central configuration: dataset schema and tunable business parameters.

All business thresholds live in :class:`AnalyticsConfig` so they can be adjusted
from the Settings page instead of being hard-coded in calculation functions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, replace
from datetime import date
from typing import Any

# --------------------------------------------------------------------------- #
# Dataset schema
# --------------------------------------------------------------------------- #

#: Column name -> (kind, description). ``kind`` drives type coercion.
REQUIRED_COLUMNS: dict[str, tuple[str, str]] = {
    "SKU_ID": ("text", "Unique product identifier (stock keeping unit)."),
    "Product_Name": ("text", "Human-readable product name."),
    "Category": ("text", "Product category used for grouping."),
    "Current_Stock": ("non_negative_number", "Units currently on hand."),
    "Units_Sold": ("non_negative_number", "Units sold during the sales window."),
    "Unit_Price": ("non_negative_number", "Selling price per unit."),
    "Reorder_Level": ("non_negative_number", "Stock level at which a reorder is triggered."),
    "Supplier_Lead_Time": ("non_negative_number", "Days between placing and receiving an order."),
}

OPTIONAL_COLUMNS: dict[str, tuple[str, str]] = {
    "Last_Restock_Date": ("date", "Date of the most recent replenishment (YYYY-MM-DD)."),
    "Sales_Period_Days": (
        "positive_number",
        "Number of days Units_Sold covers. If absent, the configurable assumed "
        "sales window is used.",
    ),
}

#: Columns of the optional daily sales-history file used for demand trends.
HISTORY_REQUIRED_COLUMNS: dict[str, tuple[str, str]] = {
    "Date": ("date", "Calendar date of the sales record."),
    "SKU_ID": ("text", "Product identifier matching the inventory file."),
    "Units_Sold": ("non_negative_number", "Units sold for that SKU on that date."),
}

NUMERIC_COLUMNS: tuple[str, ...] = tuple(
    name
    for name, (kind, _) in {**REQUIRED_COLUMNS, **OPTIONAL_COLUMNS}.items()
    if kind.endswith("number")
)

# Ordinal labels used across the app (ordered from worst to best).
HEALTH_CLASSES: tuple[str, ...] = ("Critical", "At Risk", "Moderate", "Healthy")
INSUFFICIENT_DATA = "Insufficient Data"
ALERT_LEVELS: tuple[str, ...] = ("Critical", "High", "Medium", "Healthy")
RISK_LEVELS: tuple[str, ...] = ("Critical", "High", "Medium", "Low")
STOCK_STATUSES: tuple[str, ...] = ("Out of Stock", "Low Stock", "In Stock")
ABC_CLASSES: tuple[str, ...] = ("A", "B", "C")


# --------------------------------------------------------------------------- #
# Tunable parameters
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class HealthWeights:
    """Relative weights of the four Inventory Health Score components.

    Weights do not need to sum to 1; they are normalised before use.
    """

    stock_availability: float = 0.35
    inventory_coverage: float = 0.30
    reorder_compliance: float = 0.20
    lead_time_coverage: float = 0.15

    def normalised(self) -> dict[str, float]:
        """Return the weights scaled so that they sum to 1."""
        raw = asdict(self)
        if any(value < 0 for value in raw.values()):
            raise ValueError("Health score weights must be non-negative.")
        total = sum(raw.values())
        if total <= 0:
            raise ValueError("At least one health score weight must be positive.")
        return {name: value / total for name, value in raw.items()}


@dataclass(frozen=True)
class AnalyticsConfig:
    """All business parameters used by the analytics engines."""

    # Sales velocity
    sales_window_days: int = 30
    """Assumed number of days covered by ``Units_Sold`` when no
    ``Sales_Period_Days`` column is supplied."""

    # Reorder engine
    safety_stock_days: float = 7.0
    """Safety stock expressed as days of average demand."""
    review_period_days: float = 14.0
    """Order-cycle length: how many days of demand a replenishment should cover
    beyond the lead time."""
    critical_cover_days: float = 3.0
    """Days of inventory at or below which stockout is considered imminent."""
    approaching_threshold_buffer: float = 0.20
    """Stock within (1 + buffer) x reorder threshold counts as 'approaching'."""
    threshold_mode: str = "max"
    """How the effective reorder threshold is chosen: ``dataset`` (Reorder_Level
    column only), ``dynamic`` (demand-based reorder point only) or ``max`` (the
    higher of the two)."""

    # Health score
    target_cover_days: float = 30.0
    """Days of inventory regarded as ideal coverage."""
    overstock_cover_days: float = 120.0
    """Days of inventory above which stock is considered excessive."""
    health_weights: HealthWeights = field(default_factory=HealthWeights)

    # Risk detection
    slow_moving_velocity: float = 0.5
    """Average daily sales below which a product is considered slow-moving."""

    # Demand trends (only used when daily sales history is uploaded)
    trend_threshold_pct: float = 5.0
    """Weekly % change in fitted demand beyond which a trend is Growing/Declining."""
    min_history_days: int = 14
    """Minimum calendar days of history required before a trend is reported."""

    # Data handling
    duplicate_strategy: str = "first"
    """Which record to keep for conflicting duplicate SKUs: ``first`` or ``last``."""

    as_of_date: date | None = None
    """Reference date for stockout-date estimates. ``None`` means today."""

    def __post_init__(self) -> None:
        if self.sales_window_days <= 0:
            raise ValueError("sales_window_days must be positive.")
        for name in (
            "safety_stock_days",
            "review_period_days",
            "critical_cover_days",
            "approaching_threshold_buffer",
            "slow_moving_velocity",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative.")
        if self.trend_threshold_pct < 0:
            raise ValueError("trend_threshold_pct must be non-negative.")
        if self.min_history_days < 2:
            raise ValueError("min_history_days must be at least 2.")
        if self.target_cover_days <= 0:
            raise ValueError("target_cover_days must be positive.")
        if self.overstock_cover_days <= self.target_cover_days:
            raise ValueError("overstock_cover_days must exceed target_cover_days.")
        if self.threshold_mode not in {"dataset", "dynamic", "max"}:
            raise ValueError("threshold_mode must be 'dataset', 'dynamic' or 'max'.")
        if self.duplicate_strategy not in {"first", "last"}:
            raise ValueError("duplicate_strategy must be 'first' or 'last'.")
        self.health_weights.normalised()  # validates weights

    @property
    def reference_date(self) -> date:
        """The date used as 'today' in time-based estimates."""
        return self.as_of_date or date.today()

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain (hashable-by-Streamlit) dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AnalyticsConfig":
        """Build a config from :meth:`to_dict` output, ignoring unknown keys."""
        known = {f.name for f in fields(cls)}
        kwargs = {key: value for key, value in data.items() if key in known}
        weights = kwargs.get("health_weights")
        if isinstance(weights, dict):
            kwargs["health_weights"] = HealthWeights(**weights)
        return cls(**kwargs)

    def with_updates(self, **changes: Any) -> "AnalyticsConfig":
        """Return a validated copy with the given fields replaced."""
        return replace(self, **changes)
