from __future__ import annotations

import streamlit as st


def render_top_nav(current_page: str, subtitle: str | None = None) -> None:
    with st.container(border=True):
        st.caption(f"Current: {current_page}")
        nav_cols = st.columns([1, 1, 1, 1, 2])
        with nav_cols[0]:
            st.page_link("app.py", label="Home", icon=":material/home:")
        with nav_cols[1]:
            st.page_link("pages/01_upload_and_mapping.py", label="Upload", icon=":material/upload_file:")
        with nav_cols[2]:
            st.page_link("pages/02_analysis.py", label="Analysis", icon=":material/monitoring:")
        with nav_cols[3]:
            st.page_link("pages/03_export.py", label="Export", icon=":material/download:")
        with nav_cols[4]:
            if subtitle:
                st.caption(subtitle)
