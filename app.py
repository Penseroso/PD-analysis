from __future__ import annotations

import streamlit as st

from utils.state import init_session_state


def main() -> None:
    st.set_page_config(
        page_title="PD Data Analysis",
        page_icon=":bar_chart:",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    init_session_state()

    st.title("PD Data Analysis App")
    st.caption(
        "Paste preclinical study tables, confirm mapping, run statistical analysis, and export figures and results."
    )
    st.info("Use the pages in the sidebar to upload data, run analysis, and export outputs.")


if __name__ == "__main__":
    main()
