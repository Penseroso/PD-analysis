from __future__ import annotations

import streamlit as st

from utils.state import init_session_state


CHROME_CSS = """
<style>
#MainMenu {visibility: hidden;}
header[data-testid=\"stHeader\"] {display: none;}
footer {visibility: hidden;}
div[data-testid=\"stToolbar\"] {display: none;}
div[data-testid=\"stDecoration\"] {display: none;}
div[data-testid=\"stStatusWidget\"] {display: none;}
div[data-testid=\"stMainBlockContainer\"] {
    padding-top: 0.75rem;
    padding-bottom: 1.5rem;
}
div.block-container {
    padding-top: 0.75rem;
    padding-bottom: 1.5rem;
}
h1, h2, h3 {
    margin-top: 0;
}
.workspace-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 1rem;
    padding: 1rem 1.1rem;
    margin-bottom: 1rem;
    border: 1px solid rgba(49, 51, 63, 0.12);
    border-radius: 0.9rem;
    background: linear-gradient(180deg, rgba(248, 250, 252, 0.96), rgba(255, 255, 255, 0.98));
}
.workspace-header__eyebrow {
    margin: 0 0 0.35rem 0;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #5b6472;
}
.workspace-header__title {
    margin: 0;
    font-size: 1.8rem;
    font-weight: 700;
    color: #111827;
}
.workspace-header__subtitle {
    margin: 0.35rem 0 0 0;
    color: #4b5563;
    font-size: 0.98rem;
}
.workspace-header__status {
    flex-shrink: 0;
    text-align: right;
}
.workspace-status-label {
    margin: 0 0 0.35rem 0;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #5b6472;
}
.workspace-status-badge {
    display: inline-block;
    padding: 0.38rem 0.72rem;
    border-radius: 999px;
    background: #e8f0e8;
    color: #1f5130;
    font-size: 0.84rem;
    font-weight: 600;
    white-space: nowrap;
}
.workflow-section {
    margin-bottom: 1rem;
}
.workflow-section__label {
    margin: 0 0 0.6rem 0;
    font-size: 0.74rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #5b6472;
}
.workflow-section__intro {
    margin: 0 0 0.9rem 0;
    color: #4b5563;
    font-size: 0.94rem;
}
.workflow-panel {
    padding: 1rem 1.05rem;
    border: 1px solid rgba(49, 51, 63, 0.12);
    border-radius: 0.9rem;
    background: #ffffff;
    min-height: 100%;
}
.workflow-panel__eyebrow {
    margin: 0 0 0.35rem 0;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #5b6472;
}
.workflow-panel__title {
    margin: 0 0 0.35rem 0;
    font-size: 1.05rem;
    font-weight: 700;
    color: #111827;
}
.workflow-panel__copy {
    margin: 0 0 0.9rem 0;
    color: #4b5563;
    font-size: 0.9rem;
    line-height: 1.45;
}
</style>
"""


def main() -> None:
    st.set_page_config(
        page_title="PD Data Analysis",
        page_icon=":bar_chart:",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    init_session_state()
    st.markdown(CHROME_CSS, unsafe_allow_html=True)

    status_text = "Ready"
    if st.session_state.analysis_results:
        status_text = "Analysis complete"
    elif st.session_state.normalized_df is not None:
        status_text = "Input loaded"
    elif st.session_state.raw_df is not None:
        status_text = "Input parsed"

    st.markdown(
        f"""
        <section class="workspace-header">
            <div class="workspace-header__body">
                <p class="workspace-header__eyebrow">PD Analysis Workspace</p>
                <h1 class="workspace-header__title">PD Data Analysis App</h1>
                <p class="workspace-header__subtitle">Paste preclinical study tables, confirm mapping, run statistical analysis, and export figures and results.</p>
            </div>
            <div class="workspace-header__status">
                <p class="workspace-status-label">Status</p>
                <span class="workspace-status-badge">{status_text}</span>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <section class="workflow-section">
            <p class="workflow-section__label">Workflow</p>
            <p class="workflow-section__intro">Start in the main workspace, move through setup in order, and return here whenever you need to jump between stages.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )
    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.markdown('<p class="workflow-panel__eyebrow">Data Input</p>', unsafe_allow_html=True)
            st.markdown(
                '<p class="workflow-panel__title">Upload And Mapping</p>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<p class="workflow-panel__copy">Paste source tables, parse the dataset, and review the normalized structure before analysis.</p>',
                unsafe_allow_html=True,
            )
            st.page_link("pages/01_upload_and_mapping.py", label="Open Upload Workspace", icon=":material/upload_file:")
    with col2:
        with st.container(border=True):
            st.markdown('<p class="workflow-panel__eyebrow">Analysis Options</p>', unsafe_allow_html=True)
            st.markdown(
                '<p class="workflow-panel__title">Analysis</p>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<p class="workflow-panel__copy">Choose biomarkers, review the method configuration, and run the inferential workflow.</p>',
                unsafe_allow_html=True,
            )
            st.page_link("pages/02_analysis.py", label="Open Analysis Workspace", icon=":material/monitoring:")
    with st.container(border=True):
        st.markdown('<p class="workflow-panel__eyebrow">Run / Actions</p>', unsafe_allow_html=True)
        st.markdown('<p class="workflow-panel__title">Export</p>', unsafe_allow_html=True)
        st.markdown(
            '<p class="workflow-panel__copy">Build the export bundle and download figures, normalized data, and statistics tables after review.</p>',
            unsafe_allow_html=True,
        )
        st.page_link("pages/03_export.py", label="Open Export Workspace", icon=":material/download:")


if __name__ == "__main__":
    main()
