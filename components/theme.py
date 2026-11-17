"""Visual design tokens shared by charts, cards and tables.

Colors follow a validated, colour-vision-deficiency-safe reference palette.
Status colours are reserved for health / alert / risk states and are always
shown together with a text label, never on their own.
"""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

FONT_FAMILY = 'system-ui, -apple-system, "Segoe UI", Roboto, sans-serif'
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
PRIMARY = SERIES[0]
SECONDARY = SERIES[1]
SEQUENTIAL_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

STATUS_GOOD = "#0ca30c"
STATUS_WARNING = "#fab219"
STATUS_SERIOUS = "#ec835a"
STATUS_CRITICAL = "#d03b3b"
STATUS_UNKNOWN = "#b5b3ab"

HEALTH_COLORS = {
    "Healthy": STATUS_GOOD,
    "Moderate": STATUS_WARNING,
    "At Risk": STATUS_SERIOUS,
    "Critical": STATUS_CRITICAL,
    "Insufficient Data": STATUS_UNKNOWN,
}
ALERT_COLORS = {
    "Healthy": STATUS_GOOD,
    "Medium": STATUS_WARNING,
    "High": STATUS_SERIOUS,
    "Critical": STATUS_CRITICAL,
    "Insufficient Data": STATUS_UNKNOWN,
}
RISK_LEVEL_COLORS = {
    "Low": STATUS_GOOD,
    "Medium": STATUS_WARNING,
    "High": STATUS_SERIOUS,
    "Critical": STATUS_CRITICAL,
    "Unknown": STATUS_UNKNOWN,
}
RISK_CATEGORY_COLORS = {
    "Out of Stock": STATUS_CRITICAL,
    "Stockout Risk": STATUS_SERIOUS,
    "Low Stock": STATUS_WARNING,
    "Dead Stock": STATUS_SERIOUS,
    "Excess Inventory": STATUS_WARNING,
    "Slow-Moving": STATUS_WARNING,
    "Healthy": STATUS_GOOD,
    "Insufficient Data": STATUS_UNKNOWN,
}
STATUS_ICONS = {
    "Critical": "🔴", "High": "🟠", "Medium": "🟡", "Healthy": "🟢", "Low": "🟢",
    "Moderate": "🟡", "At Risk": "🟠", "Insufficient Data": "⚪", "Unknown": "⚪",
}

TEMPLATE_NAME = "inventory"


def register_plotly_template() -> None:
    """Register (once) and activate the dashboard's Plotly template."""
    if TEMPLATE_NAME in pio.templates:
        pio.templates.default = TEMPLATE_NAME
        return
    axis = dict(
        showgrid=True,
        gridcolor=GRID,
        gridwidth=1,
        zeroline=False,
        linecolor=AXIS,
        ticks="",
        tickfont=dict(color=TEXT_MUTED, size=12),
        title=dict(font=dict(color=TEXT_SECONDARY, size=12)),
        automargin=True,
        ticklabelstandoff=8,
    )
    template = go.layout.Template(
        layout=go.Layout(
            font=dict(family=FONT_FAMILY, color=TEXT_PRIMARY, size=13),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            colorway=SERIES,
            xaxis={**axis, "showgrid": False, "showline": True},
            yaxis=axis,
            margin=dict(l=8, r=16, t=48, b=8),
            title=dict(font=dict(size=15, color=TEXT_PRIMARY), x=0, xanchor="left"),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0,
                font=dict(color=TEXT_SECONDARY, size=12), title=dict(text=""),
            ),
            hoverlabel=dict(
                bgcolor="#ffffff", bordercolor=AXIS, font=dict(family=FONT_FAMILY, color=TEXT_PRIMARY)
            ),
            hovermode="closest",
            bargap=0.28,
            barcornerradius=4,
        )
    )
    pio.templates[TEMPLATE_NAME] = template
    pio.templates.default = TEMPLATE_NAME


APP_CSS = """
<style>
.block-container {padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1400px;}
h1 {font-weight: 650; letter-spacing: -0.01em;}
h2, h3 {font-weight: 600; letter-spacing: -0.005em;}
[data-testid="stMetric"] {padding: 2px 4px;}
[data-testid="stMetricLabel"] p {font-size: 0.82rem; color: #52514e; font-weight: 500;}
[data-testid="stMetricValue"] {font-size: 1.65rem; font-weight: 650;}
.page-subtitle {color: #52514e; margin-top: -0.6rem; margin-bottom: 1.2rem; font-size: 0.95rem;}
.badge {display:inline-block; padding: 2px 10px; border-radius: 999px; font-size: 0.78rem;
        font-weight: 600; border: 1px solid rgba(11,11,11,0.10); background: #f3f2ee; color:#0b0b0b;}
.sample-banner {background:#fff7e0; border:1px solid #f2d27a; border-radius:8px; padding:8px 14px;
        font-size:0.88rem; color:#5a4300; margin-bottom: 1rem;}
.formula {font-family: ui-monospace, SFMono-Regular, Consolas, monospace; background:#f3f2ee;
        padding: 10px 14px; border-radius: 6px; font-size: 0.85rem; white-space: pre-wrap;}
</style>
"""
