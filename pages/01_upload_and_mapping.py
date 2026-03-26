from __future__ import annotations

import pandas as pd
import streamlit as st

from config import SUPPORTED_FORMATS
from utils.parser import detect_schema_candidates, normalize_to_long, parse_pasted_table
from utils.state import init_session_state, reset_analysis_state
from utils.ui import render_top_nav
from utils.validators import validate_normalized_df


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
.kpi-label {
    margin: 0 0 0.2rem 0;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: #6b7280;
}
.kpi-value {
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
render_top_nav("Upload", "Prepare input data, confirm mapping, and normalize before analysis.")
st.title("Upload And Mapping")
st.caption("Prepare a pasted dataset, review the detected structure, configure column mapping, and verify normalization before continuing.")

raw_df = st.session_state.raw_df
parse_metadata = st.session_state.parse_metadata or {}
detected_schema = st.session_state.detected_schema or {}
if raw_df is None:
    parse_status = "Not parsed"
elif parse_metadata.get("parse_mode") == "raw_lines":
    parse_status = "Raw fallback"
else:
    parse_status = "Structured"
raw_rows = str(len(raw_df)) if raw_df is not None else "0"
format_guess_text = detected_schema.get("format_type") or "Unknown"
if st.session_state.normalized_df is not None:
    normalization_status = "Ready"
elif st.session_state.analysis_status == "blocked":
    normalization_status = "Blocked"
else:
    normalization_status = "Not run"
warnings_count = len(st.session_state.warnings or [])
blocking_count = len(st.session_state.blocking_reasons or [])
kpi_rows = [
    [
        ("Parse status", parse_status, "Current parse state"),
        ("Raw rows", raw_rows, "Rows in current preview"),
        ("Format guess", format_guess_text, "Detected source layout"),
    ],
    [
        ("Normalization", normalization_status, "Current normalization state"),
        ("Warnings", str(warnings_count), "Review items present"),
        ("Blocking items", str(blocking_count), "Issues requiring action"),
    ],
]
for row in kpi_rows:
    cols = st.columns(3)
    for col, (label, value, helper) in zip(cols, row):
        with col:
            with st.container(border=True):
                st.markdown(f'<p class="kpi-label">{label}</p>', unsafe_allow_html=True)
                st.markdown(f'<p class="kpi-value">{value}</p>', unsafe_allow_html=True)
                st.markdown(f'<p class="kpi-helper">{helper}</p>', unsafe_allow_html=True)

with st.container(border=True):
    st.subheader("Input Data")
    st.caption("Paste a tabular block copied from Excel or a similar source, then parse it into a working preview.")
    raw_input = st.text_area(
        "Paste a tabular dataset copied from Excel",
        value=st.session_state.raw_input_text,
        height=240,
    )

    action_col, _ = st.columns([1, 5])
    with action_col:
        if st.button("Parse Input", type="primary"):
            st.session_state.raw_input_text = raw_input
            reset_analysis_state()
            parse_result = parse_pasted_table(raw_input)
            st.session_state.raw_df = parse_result["raw_df"]
            st.session_state.preview_df = (
                parse_result["raw_df"].copy() if parse_result["raw_df"] is not None else None
            )
            st.session_state.parse_metadata = {
                "structured_parse_succeeded": parse_result["structured_parse_succeeded"],
                "parse_mode": parse_result["parse_mode"],
                "metadata": parse_result["metadata"],
            }
            schema_input = parse_result["structured_df"] if parse_result["structured_df"] is not None else parse_result["raw_df"]
            schema_result = detect_schema_candidates(schema_input)
            st.session_state.detected_schema = schema_result["detected_schema"]
            st.session_state.analysis_status = parse_result["analysis_status"]
            st.session_state.warnings = sorted(set(parse_result["warnings"] + schema_result["warnings"]))
            st.session_state.normalized_df = None
            st.session_state.value_display_map = {}
            st.session_state.time_order = []
            st.session_state.time_order_metadata = {}
            st.session_state.normalization_metadata = {}
            st.session_state.replicate_preserved = False

parse_metadata = st.session_state.parse_metadata or {}
if st.session_state.raw_df is not None:
    with st.container(border=True):
        st.subheader("Raw Preview")
        st.caption("Inspect the parsed table before configuring mapping and normalization.")
        if parse_metadata.get("parse_mode") == "raw_lines":
            st.warning("Structured parsing failed. The preview below is only a raw single-column fallback.")
        st.dataframe(st.session_state.raw_df, use_container_width=True)

    detected = st.session_state.detected_schema or {}
    format_guess = detected.get("format_type")
    default_format_idx = list(SUPPORTED_FORMATS).index(format_guess) if format_guess in SUPPORTED_FORMATS else 0
    columns = list(st.session_state.raw_df.columns)
    none_plus_cols = [None] + columns
    numeric_default = detected.get("numeric_candidates", [])
    wide_default = detected.get("wide_time_cols") or detected.get("replicate_cols") or numeric_default

    with st.container(border=True):
        st.subheader("Mapping Setup")
        st.caption("Confirm the source layout and assign analysis roles to each relevant column.")

        format_col, values_col = st.columns([1, 2])
        with format_col:
            format_type = st.selectbox(
                "Format Type",
                options=list(SUPPORTED_FORMATS.keys()),
                index=default_format_idx,
                format_func=lambda key: SUPPORTED_FORMATS[key],
            )
        with values_col:
            selected_value_cols = st.multiselect("Value columns", columns, default=numeric_default)

        group_col, subject_col, time_col, factor2_col = st.columns(4)
        with group_col:
            group_value = st.selectbox(
                "Group column",
                none_plus_cols,
                index=none_plus_cols.index(detected.get("group_col")) if detected.get("group_col") in none_plus_cols else 0,
            )
        with subject_col:
            subject_value = st.selectbox(
                "Subject column",
                none_plus_cols,
                index=none_plus_cols.index(detected.get("subject_col")) if detected.get("subject_col") in none_plus_cols else 0,
            )
        with time_col:
            time_value = st.selectbox(
                "Time column",
                none_plus_cols,
                index=none_plus_cols.index(detected.get("time_col")) if detected.get("time_col") in none_plus_cols else 0,
            )
        with factor2_col:
            factor2_value = st.selectbox(
                "Factor2 column",
                none_plus_cols,
                index=none_plus_cols.index(detected.get("factor2_col")) if detected.get("factor2_col") in none_plus_cols else 0,
            )

        column_mapping = {
            "group": group_value,
            "subject": subject_value,
            "time": time_value,
            "factor2": factor2_value,
            "value_cols": selected_value_cols,
        }

        replicate_strategy = st.session_state.replicate_strategy
        if format_type in {"wide_time", "replicate"}:
            st.caption("Wide-format controls")
            column_mapping["wide_value_cols"] = st.multiselect(
                "Wide/replicate columns",
                columns,
                default=[col for col in wide_default if col in columns],
            )
        if format_type == "replicate":
            strategy_col, _ = st.columns([1, 2])
            with strategy_col:
                strategy_options = ["mean", "median", "keep_long"]
                replicate_strategy = st.selectbox(
                    "Replicate collapse strategy",
                    options=strategy_options,
                    index=strategy_options.index(st.session_state.replicate_strategy) if st.session_state.replicate_strategy in strategy_options else 0,
                )
            if replicate_strategy == "keep_long":
                st.warning(
                    "keep_long preserves technical replicate rows for exploratory preview/export only. Inferential analysis will be blocked until replicates are collapsed."
                )
        st.session_state.replicate_strategy = replicate_strategy

        st.caption("Next step")
        if st.button("Confirm Mapping And Normalize", type="primary"):
            st.session_state.column_mapping = column_mapping
            if parse_metadata.get("parse_mode") == "raw_lines":
                st.session_state.normalized_df = None
                st.session_state.analysis_status = "blocked"
                st.session_state.value_display_map = {}
                st.session_state.time_order = []
                st.session_state.time_order_metadata = {}
                st.session_state.normalization_metadata = {}
                st.session_state.replicate_preserved = False
                st.session_state.blocking_reasons = ["Normalization is blocked because structured parsing failed and only a raw single-column preview is available."]
                st.session_state.suggested_actions = ["Fix the pasted delimiters or paste tabular data with real columns, then parse again."]
                st.session_state.warnings = sorted(set(st.session_state.warnings + ["Structured parsing must succeed before normalization."]))
            else:
                normalize_result = normalize_to_long(
                    st.session_state.raw_df,
                    column_mapping,
                    format_type,
                    replicate_strategy=replicate_strategy,
                )
                st.session_state.detected_schema = normalize_result["detected_schema"]
                st.session_state.warnings = normalize_result["warnings"]
                st.session_state.blocking_reasons = normalize_result.get("blocking_reasons", [])
                st.session_state.suggested_actions = normalize_result.get("suggested_actions", [])
                st.session_state.analysis_status = normalize_result["analysis_status"]
                st.session_state.value_display_map = normalize_result.get("value_display_map", {})
                st.session_state.time_order = normalize_result.get("time_order", [])
                st.session_state.time_order_metadata = normalize_result.get("time_order_metadata", {})
                st.session_state.normalization_metadata = normalize_result.get("normalization_metadata", {})
                st.session_state.replicate_preserved = normalize_result.get("replicate_preserved", False)

                if normalize_result["analysis_status"] == "blocked":
                    st.session_state.normalized_df = None
                else:
                    st.session_state.normalized_df = normalize_result["normalized_df"]
                    dv_cols = [col for col in st.session_state.normalized_df.columns if col.startswith("value_")]
                    validation_result = validate_normalized_df(
                        df=st.session_state.normalized_df,
                        data_type="longitudinal"
                        if "time" in st.session_state.normalized_df.columns
                        and st.session_state.normalized_df["time"].notna().any()
                        else "cross",
                        selected_dv_cols=dv_cols,
                        between_factors=["group"],
                        factor2_col="factor2" if "factor2" in st.session_state.normalized_df.columns else None,
                        replicate_preserved=st.session_state.replicate_preserved,
                        normalization_metadata=st.session_state.normalization_metadata,
                    )
                    st.session_state.analysis_status = validation_result["analysis_status"]
                    st.session_state.blocking_reasons = sorted(set(st.session_state.blocking_reasons + validation_result["blocking_reasons"]))
                    st.session_state.suggested_actions = sorted(set(st.session_state.suggested_actions + validation_result["suggested_actions"]))
                    st.session_state.warnings = sorted(
                        set(st.session_state.warnings + validation_result["warnings"])
                    )

with st.container(border=True):
    st.subheader("Review / Validation")
    st.caption("Review the normalized output, inferred labels, warnings, and any actions required before proceeding.")

    if st.session_state.normalized_df is not None:
        st.markdown("**Normalized Preview**")
        preview_df = st.session_state.normalized_df.copy()
        preview_labels = st.session_state.value_display_map or {}
        preview_df = preview_df.rename(columns={col: f"{col} ({preview_labels[col]})" for col in preview_labels if col in preview_df.columns})
        st.dataframe(preview_df, use_container_width=True)
        if st.session_state.value_display_map:
            st.caption(
                "Biomarker labels: " + ", ".join(f"{key} = {value}" for key, value in st.session_state.value_display_map.items())
            )
        if st.session_state.time_order:
            st.caption("Inferred time order: " + " -> ".join(st.session_state.time_order))
            if st.session_state.time_order_metadata.get("ambiguous"):
                st.warning(st.session_state.time_order_metadata.get("warning", "Ambiguous time labels fell back to first-seen order."))
        if st.session_state.replicate_preserved:
            replicate_id_col = (st.session_state.normalization_metadata or {}).get("replicate_id_col", "replicate_id")
            st.warning(
                f"Technical replicates were preserved in column '{replicate_id_col}'. This normalized dataset can be previewed or exported, but inferential analysis is blocked until replicates are collapsed."
            )
    else:
        st.caption("Normalized output will appear here after mapping is confirmed.")

    if st.session_state.warnings:
        st.markdown("**Warnings**")
        for warning in st.session_state.warnings:
            st.warning(warning)

    if st.session_state.blocking_reasons:
        st.markdown("**Blocking Reasons**")
        st.error("Normalization or analysis setup is blocked")
        st.write(pd.DataFrame({"blocking_reasons": st.session_state.blocking_reasons}))

    if st.session_state.suggested_actions:
        st.markdown("**Suggested Actions**")
        st.write(pd.DataFrame({"suggested_actions": st.session_state.suggested_actions}))
