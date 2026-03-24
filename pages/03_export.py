from __future__ import annotations

import streamlit as st

from utils.export import build_export_bundle


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
</style>
"""


st.markdown(CHROME_CSS, unsafe_allow_html=True)
st.title("Export")

if not st.session_state.analysis_results:
    st.info("Run analysis first.")
    st.stop()

if st.button("Build Export Bundle", type="primary"):
    st.session_state.export_bundle = build_export_bundle(
        normalized_df=st.session_state.normalized_df,
        analysis_results=st.session_state.analysis_results,
        figure_objects=st.session_state.figure_objects,
        value_display_map=st.session_state.value_display_map,
    )

bundle = st.session_state.export_bundle
if bundle:
    if bundle.get("html") is not None:
        st.download_button(
            "Download HTML", data=bundle["html"], file_name="figures.html", mime="text/html"
        )
    if bundle.get("png") is not None:
        file_name = bundle.get("png_name") or "figure.png"
        mime = bundle.get("png_mime") or "image/png"
        st.download_button(
            "Download PNG", data=bundle["png"], file_name=file_name, mime=mime
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
    for warning in bundle.get("warnings", []):
        st.warning(warning)

