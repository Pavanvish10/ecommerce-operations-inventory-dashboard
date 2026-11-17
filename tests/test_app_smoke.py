"""End-to-end UI checks with Streamlit's AppTest (headless, no browser)."""

from __future__ import annotations

import pytest
from streamlit.testing.v1 import AppTest

from tests.conftest import ROOT

PAGES = ["overview", "inventory", "sales", "reorder", "risk", "explorer", "reports",
         "data_sources", "settings"]
TIMEOUT = 90


def _page_harness(page_name: str) -> None:
    import importlib

    from components.sidebar import render_global_filters
    from components.state import build_context, init_session
    from components.theme import register_plotly_template

    register_plotly_template()
    init_session()
    build_context(render_global_filters)
    importlib.import_module(f"views.{page_name}").render()


def run_page(page: str) -> AppTest:
    at = AppTest.from_function(_page_harness, args=(page,), default_timeout=TIMEOUT)
    at.run()
    return at


def metric(at: AppTest, label: str) -> str:
    return next(m.value for m in at.metric if m.label == label)


def test_app_entrypoint_starts():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=TIMEOUT)
    at.run()
    assert not at.exception
    assert at.title[0].value == "Operations Overview"
    assert metric(at, "Total Products") == "160"


@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders_without_exceptions(page):
    at = run_page(page)
    assert not at.exception, [e.value for e in at.exception]
    assert at.title, f"{page} rendered no title"


def test_sidebar_filter_updates_kpis():
    at = run_page("overview")
    assert metric(at, "Total Products") == "160"
    at.multiselect(key="gf_categories").select("Books").run()
    assert not at.exception
    assert metric(at, "Total Products") == "20"
    assert any("Showing **20 of 160**" in c.value for c in at.caption)


def test_filters_with_no_matches_show_message():
    at = run_page("overview")
    at.multiselect(key="gf_categories").select("Books").run()
    at.multiselect(key="gf_stock").select("Out of Stock").run()
    at.multiselect(key="gf_abc").select("A").run()
    at.multiselect(key="gf_health").select("Healthy").run()
    assert not at.exception
    assert any("No products match" in i.value for i in at.info)


def test_explorer_search_filters_results():
    at = run_page("explorer")
    at.text_input(key="ex_search").input("ELE-1001").run()
    assert not at.exception
    assert any("**1** product(s) match" in c.value for c in at.caption)


def test_settings_change_recalculates_metrics():
    at = run_page("reorder")
    before = metric(at, "Units to reorder")
    # Raise the safety stock: reorder quantities must increase.
    at.session_state["config"] = at.session_state["config"].with_updates(safety_stock_days=21.0)
    at.run()
    after = metric(at, "Units to reorder")
    assert not at.exception
    assert int(after.replace(",", "")) > int(before.replace(",", ""))


def test_settings_form_rejects_invalid_values():
    at = run_page("settings")
    at.number_input[5].set_value(10.0)  # overstock threshold below target coverage (30)
    at.button[0].click().run()  # form submit button
    assert not at.exception
    assert any("Settings not applied" in e.value for e in at.error)
