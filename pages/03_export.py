from __future__ import annotations

import streamlit as st

from utils.export import build_export_bundle
from utils.state import init_session_state
from utils.ui import render_top_nav


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
.section-eyebrow {
    margin: 0 0 0.35rem 0;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #5b6472;
}
.summary-label {
    margin: 0 0 0.2rem 0;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: #6b7280;
}
.summary-value {
    margin: 0;
    font-size: 1rem;
    font-weight: 600;
    color: #111827;
}
.kpi-helper {
    margin: 0.35rem 0 0 0;
    font-size: 0.82rem;
    color: #6b7280;
}
</style>
"""


init_session_state()
st.markdown(CHROME_CSS, unsafe_allow_html=True)
render_top_nav("Export", "Build the current bundle and download reporting artifacts.")
st.title("Export")
st.caption("Build the export bundle and download figures, normalized data, and statistics outputs for reporting.")

analysis_results = st.session_state.analysis_results or {}
normalized_df = st.session_state.normalized_df
figure_objects = st.session_state.figure_objects or {}
value_display_map = st.session_state.value_display_map or {}
bundle = st.session_state.export_bundle

if not analysis_results:
    with st.container(border=True):
        st.subheader("Analysis Required")
        if normalized_df is None:
            st.info("Normalize the dataset and run analysis before opening the export workbench.")
        else:
            st.info("Run analysis before opening the export workbench.")
    st.stop()


def _label_for_result(dv_col: str, result: dict) -> str:
    return str(result.get("dv_label") or value_display_map.get(dv_col) or dv_col)


result_labels = [_label_for_result(dv_col, result) for dv_col, result in analysis_results.items()]
if len(figure_objects) == 0:
    png_mode = "None"
elif len(figure_objects) == 1:
    png_mode = "Single PNG"
else:
    png_mode = "ZIP"
warning_count = len(bundle.get("warnings", [])) if bundle else 0

with st.container(border=True):
    st.markdown('<p class="section-eyebrow">Export Status</p>', unsafe_allow_html=True)
    st.subheader("Export Context")
    kpi_rows = [
        [
            ("Normalized data", "Available" if normalized_df is not None else "Missing", "Normalized dataset state"),
            ("Analysis results", str(len(analysis_results)), "Result entries available"),
            ("Figures", str(len(figure_objects)), "Figure objects ready for export"),
        ],
        [
            ("Bundle state", "Built" if bundle else "Not built", "Current session bundle state"),
            ("PNG packaging", png_mode, "Expected PNG packaging mode"),
            ("Export warnings", str(warning_count), "Warnings in current bundle"),
        ],
    ]
    for row in kpi_rows:
        cols = st.columns(3)
        for col, (label, value, helper) in zip(cols, row):
            with col:
                with st.container(border=True):
                    st.markdown(f'<p class="summary-label">{label}</p>', unsafe_allow_html=True)
                    st.markdown(f'<p class="summary-value">{value}</p>', unsafe_allow_html=True)
                    st.markdown(f'<p class="kpi-helper">{helper}</p>', unsafe_allow_html=True)
    if result_labels:
        if len(result_labels) <= 4:
            st.caption("Biomarkers ready for export: " + ", ".join(result_labels))
        else:
            st.caption(f"Biomarkers ready for export: {len(result_labels)} selected across the current analysis results.")
    if normalized_df is None:
        st.warning("Normalized data is not currently available in session state. Data CSV export may be unavailable until the dataset is restored.")
    if not figure_objects:
        st.info("No figure objects are currently available. CSV and statistics exports can still be built if analysis results are present.")

with st.container(border=True):
    st.markdown('<p class="section-eyebrow">Build</p>', unsafe_allow_html=True)
    st.subheader("Build Export Bundle")
    st.caption("Compile the current figures, normalized data, and statistics tables into downloadable export artifacts.")
    if st.button("Build Export Bundle", type="primary"):
        st.session_state.export_bundle = build_export_bundle(
            normalized_df=st.session_state.normalized_df,
            analysis_results=st.session_state.analysis_results,
            figure_objects=st.session_state.figure_objects,
            value_display_map=st.session_state.value_display_map,
        )
        bundle = st.session_state.export_bundle
        st.success("Export bundle refreshed for the current session state.")

bundle = st.session_state.export_bundle
with st.container(border=True):
    st.markdown('<p class="section-eyebrow">Downloads</p>', unsafe_allow_html=True)
    st.subheader("Export Outputs")
    if not bundle:
        st.caption("Build the export bundle to populate the available artifacts and download actions.")
    artifact_defs = [
        {
            "title": "Interactive Figures (HTML)",
            "description": "Interactive Plotly figures for browser-based review and reporting.",
            "available": bundle is not None and bundle.get("html") is not None,
            "button_label": "Download HTML",
            "data": None if bundle is None else bundle.get("html"),
            "file_name": "figures.html",
            "mime": "text/html",
            "availability_text": "Available" if bundle is not None and bundle.get("html") is not None else "Missing",
        },
        {
            "title": "PNG Figure Export",
            "description": "Rendered figure image output. Multiple figures are packaged into a ZIP when more than one PNG is available."
            if bundle is not None and (bundle.get("png_mime") == "application/zip" or bundle.get("png_name") == "figures.zip")
            else "Rendered figure image output for the current analysis figures.",
            "available": bundle is not None and bundle.get("png") is not None,
            "button_label": "Download PNG Export",
            "data": None if bundle is None else bundle.get("png"),
            "file_name": None if bundle is None else (bundle.get("png_name") or "figure.png"),
            "mime": None if bundle is None else (bundle.get("png_mime") or "image/png"),
            "availability_text": "ZIP of multiple PNGs"
            if bundle is not None and (bundle.get("png_mime") == "application/zip" or bundle.get("png_name") == "figures.zip")
            else ("Single PNG" if bundle is not None and bundle.get("png") is not None else "Missing"),
        },
        {
            "title": "Normalized Data CSV",
            "description": "Normalized long-format dataset used as the export-ready analysis input.",
            "available": bundle is not None and bundle.get("csv") is not None,
            "button_label": "Download Data CSV",
            "data": None if bundle is None else bundle.get("csv"),
            "file_name": "normalized_data.csv",
            "mime": "text/csv",
            "availability_text": "Available" if bundle is not None and bundle.get("csv") is not None else "Missing",
        },
        {
            "title": "Stats Results CSV",
            "description": "Analysis result summaries plus any available omnibus, posthoc, fixed-effect, and contrast tables.",
            "available": bundle is not None and bundle.get("stats_csv") is not None,
            "button_label": "Download Stats CSV",
            "data": None if bundle is None else bundle.get("stats_csv"),
            "file_name": "stats_results.csv",
            "mime": "text/csv",
            "availability_text": "Available" if bundle is not None and bundle.get("stats_csv") is not None else "Missing",
        },
    ]

    row1 = st.columns(2)
    row2 = st.columns(2)
    artifact_cols = row1 + row2
    for col, artifact in zip(artifact_cols, artifact_defs):
        with col:
            with st.container(border=True):
                st.markdown(f"**{artifact['title']}**")
                st.caption(artifact["description"])
                st.write(f"Availability: {artifact['availability_text']}")
                if artifact["available"]:
                    st.download_button(
                        artifact["button_label"],
                        data=artifact["data"],
                        file_name=artifact["file_name"],
                        mime=artifact["mime"],
                    )
                else:
                    st.caption("This artifact will appear here after a bundle is built and the required source content is available.")

with st.container(border=True):
    st.markdown('<p class="section-eyebrow">Review Notes</p>', unsafe_allow_html=True)
    st.subheader("Export Warnings / Notes")
    if not bundle:
        st.caption("Warnings and notes will appear here after the export bundle is generated.")
    else:
        warnings = list(dict.fromkeys(bundle.get("warnings", [])))
        if warnings:
            for warning in warnings:
                st.warning(warning)
        else:
            st.success("No export warnings were reported for the current bundle.")
