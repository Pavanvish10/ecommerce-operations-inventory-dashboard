"""Alert indicators, banners and validation reports."""

from __future__ import annotations

from html import escape

import pandas as pd
import streamlit as st

from components.theme import STATUS_ICONS
from src.data_validator import ValidationResult
from src.export_manager import to_csv_bytes


_MARKDOWN_SPECIALS = set(r"\`*_[]<>#|~$")


def md_escape(text: str) -> str:
    """Escape Markdown/LaTeX syntax in untrusted text (file names, CSV headers).

    Streamlit already strips scripts from Markdown, but unescaped user text
    could still render links, images or formatting.
    """
    return "".join("\\" + ch if ch in _MARKDOWN_SPECIALS else ch for ch in str(text))


def page_header(title: str, subtitle: str) -> None:
    """Page title with a muted subtitle line."""
    st.title(title)
    st.markdown(f'<div class="page-subtitle">{subtitle}</div>', unsafe_allow_html=True)


def sample_data_banner(is_sample: bool, source_name: str) -> None:
    """Make it unmistakable when the dashboard shows demonstration data."""
    if is_sample:
        st.markdown(
            '<div class="sample-banner">📎 <b>Sample data</b> — you are viewing a synthetic '
            "demonstration dataset. Upload your own CSV on the <b>Data Sources</b> page.</div>",
            unsafe_allow_html=True,
        )
    else:
        st.caption(f"Dataset: **{source_name}**")


def render_alert_summary(metrics: pd.DataFrame) -> None:
    """Callouts for Critical and High reorder alerts."""
    counts = metrics["Alert_Level"].value_counts()
    critical = int(counts.get("Critical", 0))
    high = int(counts.get("High", 0))
    insufficient = int(counts.get("Insufficient Data", 0))
    if critical:
        st.error(
            f"**{critical} critical alert(s):** products out of stock or with imminent stockout. "
            "See **Reorder Management**.",
            icon="🚨",
        )
    if high:
        st.warning(
            f"**{high} high alert(s):** stock will run out before a new order can arrive.",
            icon="⚠️",
        )
    if insufficient:
        st.info(
            f"{insufficient} product(s) have incomplete data and were not evaluated for alerts.",
            icon="ℹ️",
        )
    if not (critical or high):
        st.success("No critical or high reorder alerts for the current selection.", icon="✅")


def status_badge(label: str) -> str:
    """Inline HTML badge with icon + text (colour is never the only cue). Escaped."""
    return f'<span class="badge">{STATUS_ICONS.get(label, "")} {escape(str(label))}</span>'


def render_validation_report(result: ValidationResult, title: str, key: str) -> None:
    """Show the outcome of validating a dataset, including rejected rows."""
    summary = result.summary()
    cols = st.columns(4)
    cols[0].metric("Rows in file", f"{summary['input_rows']:,}")
    cols[1].metric("Rows accepted", f"{summary['accepted_rows']:,}")
    cols[2].metric("Rows excluded", f"{summary['rejected_rows']:,}")
    cols[3].metric("Warnings", f"{summary['warnings']:,}")

    for issue in result.errors:
        st.error(md_escape(issue.message), icon="⛔")
    if result.warnings or result.infos:
        with st.expander(
            f"{title}: {len(result.warnings)} warning(s), {len(result.infos)} note(s)",
            expanded=bool(result.warnings),
        ):
            for issue in [*result.warnings, *result.infos]:
                rows = issue.rows_preview()
                suffix = f" — CSV line(s): {rows}" if rows else ""
                icon = "⚠️" if issue.severity == "warning" else "ℹ️"
                st.markdown(f"{icon} {md_escape(issue.message)}{suffix}")
    elif not result.errors:
        st.success("No data quality issues found.", icon="✅")

    if not result.rejected.empty:
        with st.expander(f"Excluded rows ({len(result.rejected)})"):
            st.dataframe(result.rejected, hide_index=True, width="stretch")
            st.download_button(
                "Download excluded rows (CSV)",
                data=to_csv_bytes(result.rejected),
                file_name="excluded_rows.csv",
                mime="text/csv",
                key=f"{key}_rejected_download",
            )
