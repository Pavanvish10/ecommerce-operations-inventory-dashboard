"""E-commerce Operations & Inventory Analytics Dashboard — Streamlit entry point.

Run with::

    streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

from components.sidebar import render_global_filters
from components.state import build_context, init_session
from components.theme import APP_CSS, register_plotly_template
from views import (
    data_sources,
    explorer,
    inventory,
    overview,
    reorder,
    reports,
    risk,
    sales,
    settings,
)

st.set_page_config(
    page_title="Inventory Analytics",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)
register_plotly_template()
st.markdown(APP_CSS, unsafe_allow_html=True)

init_session()

navigation = st.navigation(
    {
        "Analytics": [
            st.Page(overview.render, title="Overview", icon=":material/dashboard:",
                    default=True),
            st.Page(inventory.render, title="Inventory Analytics", icon=":material/inventory_2:",
                    url_path="inventory"),
            st.Page(sales.render, title="Sales Analytics", icon=":material/trending_up:",
                    url_path="sales"),
        ],
        "Operations": [
            st.Page(reorder.render, title="Reorder Management", icon=":material/local_shipping:",
                    url_path="reorder"),
            st.Page(risk.render, title="Risk Analysis", icon=":material/warning:", url_path="risk"),
            st.Page(explorer.render, title="Product Explorer", icon=":material/search:",
                    url_path="explorer"),
            st.Page(reports.render, title="Reports & Exports", icon=":material/download:",
                    url_path="reports"),
        ],
        "Data": [
            st.Page(data_sources.render, title="Data Sources", icon=":material/upload_file:",
                    url_path="data"),
            st.Page(settings.render, title="Settings", icon=":material/tune:", url_path="settings"),
        ],
    }
)

with st.sidebar:
    st.markdown("### 📦 Inventory Analytics")
    st.caption("E-commerce operations & replenishment")

context = build_context(render_global_filters)

with st.sidebar:
    st.divider()
    if context.inventory is not None and context.inventory.is_valid:
        badge = "🧪 Sample data" if context.is_sample else f"📄 {context.source_name}"
        st.caption(f"{badge} · {len(context.metrics):,} products")
    else:
        st.caption("⛔ No valid dataset loaded")

navigation.run()
