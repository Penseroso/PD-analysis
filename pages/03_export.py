from __future__ import annotations

import streamlit as st

from utils.export import build_export_bundle


st.title("Export")

if not st.session_state.analysis_results:
    st.info("Run analysis first.")
    st.stop()

if st.button("Build Export Bundle", type="primary"):
    st.session_state.export_bundle = build_export_bundle(
        normalized_df=st.session_state.normalized_df,
        analysis_results=st.session_state.analysis_results,
        figure_objects=st.session_state.figure_objects,
    )

bundle = st.session_state.export_bundle
if bundle:
    if bundle.get("html") is not None:
        st.download_button(
            "Download HTML", data=bundle["html"], file_name="figures.html", mime="text/html"
        )
    if bundle.get("png") is not None:
        st.download_button(
            "Download PNG", data=bundle["png"], file_name="figure.png", mime="image/png"
        )
    if bundle.get("csv") is not None:
        st.download_button(
            "Download Data CSV",
            data=bundle["csv"],
            file_name="normalized_data.csv",
            mime="text/csv",
        )
    if bundle.get("stats_csv") is not None:
        st.download_button(
            "Download Stats CSV",
            data=bundle["stats_csv"],
            file_name="stats_results.csv",
            mime="text/csv",
        )
