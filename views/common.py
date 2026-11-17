"""Helpers shared by the dashboard pages."""

from __future__ import annotations

import streamlit as st

from components.alerts import sample_data_banner
from components.state import PageContext



def plot(fig, key: str) -> None:
    """Render a Plotly figure with the dashboard's own template."""
    st.plotly_chart(fig, theme=None, key=key, config={"displaylogo": False})


def require_data(ctx: PageContext) -> bool:
    """Show guidance and return False when no valid dataset is loaded."""
    if ctx.has_data:
        return True
    st.warning(
        "No valid dataset is loaded. Go to **Data Sources** to upload a CSV file or reload "
        "the sample dataset.",
        icon="📂",
    )
    return False


def context_bar(ctx: PageContext) -> bool:
    """Sample-data banner plus a 'showing X of Y products' line.

    Returns False (after informing the user) when filters match no products.
    """
    sample_data_banner(ctx.is_sample, ctx.source_name)
    total = len(ctx.metrics)
    shown = len(ctx.filtered)
    if ctx.criteria.active_count:
        st.caption(
            f"🔎 Showing **{shown:,} of {total:,}** products · "
            f"{ctx.criteria.active_count} global filter(s) active (sidebar)"
        )
    if shown == 0:
        st.info("No products match the current filters. Adjust or reset them in the sidebar.")
        return False
    return True
