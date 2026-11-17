"""Combined product filtering and sorting (pure functions, UI-independent)."""

from __future__ import annotations

from dataclasses import dataclass, fields

import pandas as pd

FULL_SCORE_RANGE: tuple[float, float] = (0.0, 100.0)


@dataclass(frozen=True)
class FilterCriteria:
    """Filter selections. Empty tuples mean "no restriction" for that field."""

    search: str = ""
    categories: tuple[str, ...] = ()
    health_classes: tuple[str, ...] = ()
    risk_levels: tuple[str, ...] = ()
    risk_categories: tuple[str, ...] = ()
    stock_statuses: tuple[str, ...] = ()
    abc_classes: tuple[str, ...] = ()
    alert_levels: tuple[str, ...] = ()
    health_score_range: tuple[float, float] = FULL_SCORE_RANGE

    @property
    def active_count(self) -> int:
        """Number of filters that currently restrict the result."""
        count = 0
        for f in fields(self):
            value = getattr(self, f.name)
            if f.name == "health_score_range":
                count += tuple(value) != FULL_SCORE_RANGE
            elif f.name == "search":
                count += bool(value.strip())
            else:
                count += bool(value)
        return count


_MEMBERSHIP_FILTERS: dict[str, str] = {
    "categories": "Category",
    "health_classes": "Health_Class",
    "risk_levels": "Risk_Level",
    "risk_categories": "Risk_Category",
    "stock_statuses": "Stock_Status",
    "abc_classes": "ABC_Class",
    "alert_levels": "Alert_Level",
}


def apply_filters(metrics: pd.DataFrame, criteria: FilterCriteria) -> pd.DataFrame:
    """Return the rows that satisfy *all* criteria (logical AND).

    * ``search`` is a case-insensitive literal substring match on SKU_ID or
      Product_Name (special regex characters are not interpreted).
    * A restricted health-score range excludes rows without a score.
    """
    mask = pd.Series(True, index=metrics.index)

    term = criteria.search.strip().lower()
    if term:
        sku = metrics["SKU_ID"].astype(str).str.lower().str.contains(term, regex=False)
        name = metrics["Product_Name"].astype(str).str.lower().str.contains(term, regex=False)
        mask &= sku | name

    for attribute, column in _MEMBERSHIP_FILTERS.items():
        selected = getattr(criteria, attribute)
        if selected and column in metrics.columns:
            mask &= metrics[column].isin(selected)

    low, high = criteria.health_score_range
    if (low, high) != FULL_SCORE_RANGE:
        score = metrics["Health_Score"]
        mask &= score.notna() & (score >= low) & (score <= high)

    return metrics.loc[mask]


#: Sort option label -> (columns, ascending flags). Ties always fall back to SKU_ID.
SORT_OPTIONS: dict[str, tuple[list[str], list[bool]]] = {
    "Highest risk": (["Risk_Priority", "Health_Score"], [True, True]),
    "Lowest health score": (["Health_Score"], [True]),
    "Highest sales": (["Units_Sold"], [False]),
    "Lowest sales": (["Units_Sold"], [True]),
    "Highest revenue": (["Revenue"], [False]),
    "Highest inventory": (["Current_Stock"], [False]),
    "Lowest inventory": (["Current_Stock"], [True]),
    "Fewest days of cover": (["Days_Of_Inventory"], [True]),
    "Highest excess value": (["Excess_Inventory_Value"], [False]),
}


def sort_products(metrics: pd.DataFrame, option: str) -> pd.DataFrame:
    """Sort by a named option; missing values always go last."""
    if option not in SORT_OPTIONS:
        raise ValueError(f"Unknown sort option: {option!r}")
    columns, ascending = SORT_OPTIONS[option]
    return metrics.sort_values(
        [*columns, "SKU_ID"],
        ascending=[*ascending, True],
        na_position="last",
        kind="mergesort",
    )
