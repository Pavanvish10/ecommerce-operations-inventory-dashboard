"""Plotly figure builders.

Each function takes already-computed (and already-filtered) data and returns a
``go.Figure``; none of them perform business calculations, which keeps charts
consistent with tables and KPIs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from components.theme import (
    ALERT_COLORS,
    HEALTH_COLORS,
    PRIMARY,
    RISK_CATEGORY_COLORS,
    STATUS_UNKNOWN,
    TEXT_MUTED,
    TEXT_SECONDARY,
)
from src.config import ALERT_LEVELS, HEALTH_CLASSES, INSUFFICIENT_DATA

_LABEL_MAX = 38


def product_label(frame: pd.DataFrame) -> pd.Series:
    """'Product name (SKU)' truncated for axis readability."""
    names = frame["Product_Name"].astype(str)
    names = names.where(names.str.len() <= _LABEL_MAX, names.str.slice(0, _LABEL_MAX - 1) + "…")
    return names + " (" + frame["SKU_ID"].astype(str) + ")"


def empty_figure(message: str = "No data for the current filters") -> go.Figure:
    """Placeholder figure shown when there is nothing to plot."""
    fig = go.Figure()
    fig.add_annotation(
        text=message, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False,
        font=dict(color=TEXT_MUTED, size=14),
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(height=320)
    return fig


def _height_for(rows: int, minimum: int = 320, per_row: int = 28) -> int:
    return max(minimum, 90 + per_row * rows)


def room_for_labels(fig: go.Figure, max_value: float, pad: float = 0.22) -> None:
    """Extend the value axis so outside bar labels are never clipped."""
    if pd.notna(max_value) and max_value > 0:
        fig.update_xaxes(range=[0, float(max_value) * (1 + pad)])


def category_bar(
    category_perf: pd.DataFrame, value: str, title: str, value_label: str, money: bool = False
) -> go.Figure:
    """Horizontal bar of a category-level measure, largest at the top."""
    if category_perf.empty:
        return empty_figure()
    data = category_perf.sort_values(value, ascending=True)
    fmt = "$,.0f" if money else ",.0f"
    fig = go.Figure(
        go.Bar(
            x=data[value],
            y=data["Category"],
            orientation="h",
            marker_color=PRIMARY,
            text=data[value],
            texttemplate=f"%{{text:{fmt}}}",
            textposition="outside",
            cliponaxis=False,
            customdata=np.stack([data["Products"], data["Revenue_Share"] * 100], axis=-1),
            hovertemplate=(
                "<b>%{y}</b><br>" + value_label + ": %{x:" + fmt + "}"
                "<br>Products: %{customdata[0]}<br>Revenue share: %{customdata[1]:.1f}%<extra></extra>"
            ),
        )
    )
    fig.update_layout(title=title, height=_height_for(len(data)), showlegend=False)
    fig.update_xaxes(title=value_label, showgrid=True, showline=False, tickformat="$~s" if money else "~s")
    room_for_labels(fig, data[value].max())
    fig.update_yaxes(showgrid=False, title=None)
    return fig


def top_products_bar(
    products: pd.DataFrame,
    value: str,
    title: str,
    value_label: str,
    money: bool = False,
    decimals: int = 0,
) -> go.Figure:
    """Horizontal bar of individual products for ``value`` (rank order preserved)."""
    if products.empty:
        return empty_figure()
    data = products.iloc[::-1]  # first row at the top
    fmt = "$,.2f" if money else f",.{decimals}f"
    fig = go.Figure(
        go.Bar(
            x=data[value],
            y=product_label(data),
            orientation="h",
            marker_color=PRIMARY,
            text=data[value],
            texttemplate=f"%{{text:{fmt}}}",
            textposition="outside",
            cliponaxis=False,
            customdata=np.stack(
                [data["Category"], data["Current_Stock"].fillna(-1), data["Health_Class"]], axis=-1
            ),
            hovertemplate=(
                "<b>%{y}</b><br>" + value_label + ": %{x:" + fmt + "}"
                "<br>Category: %{customdata[0]}<br>Stock: %{customdata[1]:,.0f}"
                "<br>Health: %{customdata[2]}<extra></extra>"
            ),
        )
    )
    fig.update_layout(title=title, height=_height_for(len(data)), showlegend=False)
    fig.update_xaxes(title=value_label, showgrid=True, showline=False)
    fig.update_yaxes(showgrid=False, title=None, tickfont=dict(size=11))
    room_for_labels(fig, data[value].max(), pad=0.3 if money else 0.18)
    return fig


def risk_distribution(summary: pd.DataFrame) -> go.Figure:
    """Products per risk category, coloured by severity (labels carry identity)."""
    if summary.empty:
        return empty_figure()
    data = summary.iloc[::-1]
    colors = [RISK_CATEGORY_COLORS.get(c, STATUS_UNKNOWN) for c in data["Risk_Category"]]
    fig = go.Figure(
        go.Bar(
            x=data["Products"],
            y=data["Risk_Category"],
            orientation="h",
            marker_color=colors,
            text=data["Products"],
            textposition="outside",
            cliponaxis=False,
            customdata=data[["Inventory_Value", "Excess_Inventory_Value"]].to_numpy(),
            hovertemplate=(
                "<b>%{y}</b><br>Products: %{x}<br>Inventory value: %{customdata[0]:$,.0f}"
                "<br>Excess value: %{customdata[1]:$,.0f}<extra></extra>"
            ),
        )
    )
    fig.update_layout(title="Inventory Risk Distribution", height=_height_for(len(data)), showlegend=False)
    fig.update_xaxes(title="Products", showgrid=True, showline=False)
    fig.update_yaxes(showgrid=False, title=None)
    room_for_labels(fig, data["Products"].max(), pad=0.12)
    return fig


def status_count_bar(
    counts: pd.Series, order: list[str], colors: dict[str, str], title: str
) -> go.Figure:
    """Vertical bar of counts per ordered status label."""
    counts = counts.reindex(order).fillna(0).astype(int)
    if counts.sum() == 0:
        return empty_figure()
    fig = go.Figure(
        go.Bar(
            x=counts.index,
            y=counts.values,
            marker_color=[colors.get(label, STATUS_UNKNOWN) for label in counts.index],
            text=counts.values,
            textposition="outside",
            cliponaxis=False,
            hovertemplate="<b>%{x}</b><br>Products: %{y}<extra></extra>",
        )
    )
    fig.update_layout(title=title, height=340, showlegend=False)
    fig.update_yaxes(title="Products")
    fig.update_xaxes(title=None)
    return fig


def health_histogram(metrics: pd.DataFrame) -> go.Figure:
    """Distribution of health scores in 10-point bins coloured by band."""
    scores = metrics["Health_Score"].dropna()
    if scores.empty:
        return empty_figure("No health scores available")
    edges = np.arange(0, 110, 10)
    counts, _ = np.histogram(scores.clip(0, 100), bins=edges)
    labels = [f"{lo}–{lo + 10 if lo < 90 else 100}" for lo in edges[:-1]]
    band = ["Critical"] * 4 + ["At Risk"] * 2 + ["Moderate"] * 2 + ["Healthy"] * 2
    fig = go.Figure(
        go.Bar(
            x=labels,
            y=counts,
            marker_color=[HEALTH_COLORS[b] for b in band],
            customdata=band,
            hovertemplate="Score %{x}<br>%{customdata}<br>Products: %{y}<extra></extra>",
        )
    )
    fig.update_layout(title="Inventory Health Score Distribution", height=340, bargap=0.08)
    fig.update_xaxes(title="Health score")
    fig.update_yaxes(title="Products")
    return fig


def health_by_category(metrics: pd.DataFrame) -> go.Figure:
    """Stacked bar: number of products in each health class per category."""
    if metrics.empty:
        return empty_figure()
    table = pd.crosstab(metrics["Category"], metrics["Health_Class"])
    order = [c for c in [*HEALTH_CLASSES, INSUFFICIENT_DATA] if c in table.columns]
    table = table[order]
    table = table.loc[table.sum(axis=1).sort_values().index]
    fig = go.Figure()
    for health_class in order:
        fig.add_trace(
            go.Bar(
                name=health_class,
                y=table.index,
                x=table[health_class],
                orientation="h",
                marker=dict(color=HEALTH_COLORS[health_class], line=dict(color="#ffffff", width=2)),
                hovertemplate="<b>%{y}</b><br>" + health_class + ": %{x}<extra></extra>",
            )
        )
    fig.update_layout(
        barmode="stack", title="Health Class by Category", height=_height_for(len(table)) + 30, margin=dict(t=84),
        barcornerradius=0,
    )
    fig.update_xaxes(title="Products", showgrid=True, showline=False)
    fig.update_yaxes(showgrid=False, title=None)
    return fig


def stock_vs_sales(metrics: pd.DataFrame, target_days: float, overstock_days: float) -> go.Figure:
    """Scatter of daily sales vs stock, with constant days-of-cover guide lines."""
    data = metrics.dropna(subset=["Avg_Daily_Sales", "Current_Stock"])
    if data.empty:
        return empty_figure()
    fig = go.Figure()
    max_x = float(data["Avg_Daily_Sales"].max()) or 1.0
    y_top = float(data["Current_Stock"].max()) * 1.08 or 1.0
    for days, label in ((target_days, "target cover"), (overstock_days, "overstock")):
        end_x = min(max_x, y_top / days)  # stop the guide line at the top of the plot
        fig.add_trace(
            go.Scatter(
                x=[0, end_x], y=[0, end_x * days], mode="lines",
                line=dict(color=TEXT_MUTED, width=1, dash="dot"), hoverinfo="skip",
                showlegend=False,
            )
        )
        hits_top = end_x < max_x
        fig.add_annotation(
            x=end_x, y=end_x * days, text=f"{days:g} days {label}", showarrow=False,
            xanchor="left" if hits_top else "right", yanchor="top" if hits_top else "bottom",
            xshift=6 if hits_top else 0, font=dict(size=11, color=TEXT_SECONDARY),
        )
    for health_class in [*HEALTH_CLASSES, INSUFFICIENT_DATA]:
        subset = data[data["Health_Class"] == health_class]
        if subset.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=subset["Avg_Daily_Sales"],
                y=subset["Current_Stock"],
                mode="markers",
                name=health_class,
                marker=dict(
                    color=HEALTH_COLORS[health_class], size=9, opacity=0.85,
                    line=dict(color="#ffffff", width=1.5),
                ),
                text=product_label(subset),
                customdata=subset[["Days_Of_Inventory", "Health_Score"]].replace(np.inf, np.nan).to_numpy(),
                hovertemplate=(
                    "<b>%{text}</b><br>Daily sales: %{x:.2f}<br>Stock: %{y:,.0f}"
                    "<br>Days of cover: %{customdata[0]:.0f}<br>Health score: %{customdata[1]:.1f}"
                    "<extra>" + health_class + "</extra>"
                ),
            )
        )
    y_cap = float(data["Current_Stock"].max()) * 1.08 or 1.0
    fig.update_layout(title="Stock vs Sales Velocity", height=460, margin=dict(t=84))
    fig.update_xaxes(title="Average daily sales (units)", showgrid=True, rangemode="tozero")
    fig.update_yaxes(title="Current stock (units)", range=[0, y_cap])
    return fig


def pareto_curve(metrics: pd.DataFrame) -> go.Figure:
    """Cumulative revenue share by product rank (ABC / Pareto view)."""
    revenue = metrics["Revenue"].dropna().sort_values(ascending=False).reset_index(drop=True)
    total = revenue.sum()
    if revenue.empty or total <= 0:
        return empty_figure("No revenue to analyse")
    cumulative = revenue.cumsum() / total * 100
    rank_pct = (np.arange(1, len(revenue) + 1) / len(revenue)) * 100
    fig = go.Figure(
        go.Scatter(
            x=rank_pct, y=cumulative, mode="lines", line=dict(color=PRIMARY, width=2),
            hovertemplate="Top %{x:.0f}% of products<br>%{y:.1f}% of revenue<extra></extra>",
        )
    )
    for level, label in ((80, "A | B"), (95, "B | C")):
        fig.add_hline(y=level, line=dict(color=TEXT_MUTED, width=1, dash="dot"),
                      annotation_text=label, annotation_position="bottom right",
                      annotation_font=dict(size=11, color=TEXT_SECONDARY))
    fig.update_layout(title="Revenue Concentration (Pareto)", height=360, hovermode="x unified")
    fig.update_xaxes(title="% of products (ranked by revenue)", ticksuffix="%", range=[0, 100])
    fig.update_yaxes(title="Cumulative % of revenue", ticksuffix="%", range=[0, 102])
    return fig


def sales_trend_line(series: pd.DataFrame, title: str) -> go.Figure:
    """Line chart of units sold over time (from uploaded daily history)."""
    if series.empty:
        return empty_figure("No sales history available")
    fig = go.Figure(
        go.Scatter(
            x=series["Date"], y=series["Units_Sold"], mode="lines+markers",
            line=dict(color=PRIMARY, width=2), marker=dict(size=8, color=PRIMARY,
                                                           line=dict(color="#ffffff", width=2)),
            hovertemplate="%{x|%d %b %Y}<br>Units sold: %{y:,.0f}<extra></extra>",
        )
    )
    fig.update_layout(title=title, height=340, hovermode="x unified")
    fig.update_yaxes(title="Units sold", rangemode="tozero")
    fig.update_xaxes(title=None)
    return fig


def cover_vs_lead_time(products: pd.DataFrame, title: str) -> go.Figure:
    """Days of inventory (bar) against supplier lead time (marker) per product."""
    if products.empty:
        return empty_figure("No products need attention")
    data = products.iloc[::-1].copy()
    data["Days_Of_Inventory"] = data["Days_Of_Inventory"].replace(np.inf, np.nan)
    labels = product_label(data)
    colors = [ALERT_COLORS.get(level, STATUS_UNKNOWN) for level in data["Alert_Level"]]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Days of inventory", x=data["Days_Of_Inventory"], y=labels, orientation="h",
            marker_color=colors, customdata=data["Alert_Level"],
            hovertemplate="<b>%{y}</b><br>Days of inventory: %{x:.1f}<br>Alert: %{customdata}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            name="Supplier lead time", x=data["Supplier_Lead_Time"], y=labels, mode="markers",
            marker=dict(symbol="line-ns", size=16, line=dict(width=3, color="#0b0b0b")),
            hovertemplate="<b>%{y}</b><br>Lead time: %{x:.0f} days<extra></extra>",
        )
    )
    fig.update_layout(title=title, height=_height_for(len(data)) + 30, barmode="overlay", margin=dict(t=84))
    fig.update_xaxes(title="Days", showgrid=True, showline=False, rangemode="tozero")
    fig.update_yaxes(showgrid=False, title=None, tickfont=dict(size=11))
    return fig


def alert_level_bar(metrics: pd.DataFrame) -> go.Figure:
    """Products per reorder alert level."""
    counts = metrics["Alert_Level"].value_counts()
    order = [*ALERT_LEVELS, INSUFFICIENT_DATA] if counts.get(INSUFFICIENT_DATA, 0) else list(ALERT_LEVELS)
    return status_count_bar(
        counts,
        order,
        ALERT_COLORS,
        "Reorder Alert Levels",
    )
