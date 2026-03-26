from __future__ import annotations

import streamlit as st


APP_SHELL_CSS = """
<style>
section[data-testid="stSidebar"] {display: none;}
div[data-testid="stSidebarNav"] {display: none;}
div[data-testid="collapsedControl"] {display: none;}
.app-shell {
    padding: 0.2rem 0.2rem 0.15rem 0.2rem;
}
.app-shell__top {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 0.45rem;
}
.app-shell__eyebrow {
    margin: 0;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #5b6472;
}
.app-shell__current {
    display: inline-block;
    margin: 0;
    padding: 0.26rem 0.65rem;
    border-radius: 999px;
    background: #eef3f8;
    color: #1f2937;
    font-size: 0.8rem;
    font-weight: 600;
    white-space: nowrap;
}
.app-shell__subtitle {
    margin: 0 0 0.75rem 0;
    color: #4b5563;
    font-size: 0.9rem;
    line-height: 1.4;
}
.app-shell__active {
    display: block;
    width: 100%;
    padding: 0.45rem 0.7rem;
    border: 1px solid rgba(37, 99, 235, 0.18);
    border-radius: 0.8rem;
    background: linear-gradient(180deg, rgba(239, 246, 255, 0.98), rgba(248, 250, 252, 0.98));
    color: #1d4ed8;
    font-size: 0.88rem;
    font-weight: 700;
    text-align: center;
}
div[data-testid="stVerticalBlock"] > div:has(> div[data-testid="stVerticalBlockBorderWrapper"]) {
    margin-bottom: 0.15rem;
}
div[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 0.95rem;
    border-color: rgba(49, 51, 63, 0.12);
    background: #ffffff;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.03);
}
div[data-testid="stAlert"] {
    border-radius: 0.8rem;
}
</style>
"""


def is_export_bundle_built(bundle: dict | None) -> bool:
    if not isinstance(bundle, dict):
        return False
    artifact_keys = ("html", "png", "csv", "stats_csv")
    return any(bundle.get(key) is not None for key in artifact_keys)


def render_top_nav(current_page: str, subtitle: str | None = None) -> None:
    st.markdown(APP_SHELL_CSS, unsafe_allow_html=True)
    pages = [
        ("Home", "app.py", ":material/home:"),
        ("Upload", "pages/01_upload_and_mapping.py", ":material/upload_file:"),
        ("Analysis", "pages/02_analysis.py", ":material/monitoring:"),
        ("Export", "pages/03_export.py", ":material/download:"),
    ]
    with st.container(border=True):
        st.markdown(
            f"""
            <div class="app-shell">
                <div class="app-shell__top">
                    <p class="app-shell__eyebrow">PD Analysis Workspace</p>
                    <p class="app-shell__current">Current: {current_page}</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if subtitle:
            st.markdown(f'<p class="app-shell__subtitle">{subtitle}</p>', unsafe_allow_html=True)
        nav_cols = st.columns(4)
        for col, (label, path, icon) in zip(nav_cols, pages):
            with col:
                if label == current_page:
                    st.markdown(f'<div class="app-shell__active">{label}</div>', unsafe_allow_html=True)
                else:
                    st.page_link(path, label=label, icon=icon)
