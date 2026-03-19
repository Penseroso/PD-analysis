from __future__ import annotations

import pandas as pd
import streamlit as st

from config import SUPPORTED_FORMATS
from utils.parser import detect_schema_candidates, normalize_to_long, parse_pasted_table
from utils.state import reset_analysis_state
from utils.validators import validate_normalized_df


st.title("Upload And Mapping")

raw_input = st.text_area(
    "Paste a tabular dataset copied from Excel",
    value=st.session_state.raw_input_text,
    height=240,
)

if st.button("Parse Input", type="primary"):
    st.session_state.raw_input_text = raw_input
    st.session_state.raw_df = parse_pasted_table(raw_input)
    st.session_state.preview_df = (
        st.session_state.raw_df.copy() if st.session_state.raw_df is not None else None
    )
    schema_result = detect_schema_candidates(st.session_state.raw_df)
    st.session_state.detected_schema = schema_result["detected_schema"]
    st.session_state.analysis_status = schema_result["analysis_status"]
    st.session_state.warnings = schema_result["warnings"]
    reset_analysis_state()

if st.session_state.raw_df is not None:
    st.subheader("Raw Preview")
    st.dataframe(st.session_state.raw_df, use_container_width=True)

    detected = st.session_state.detected_schema or {}
    format_guess = detected.get("format_type")
    default_format_idx = list(SUPPORTED_FORMATS).index(format_guess) if format_guess in SUPPORTED_FORMATS else 0
    format_type = st.selectbox(
        "Format Type",
        options=list(SUPPORTED_FORMATS.keys()),
        index=default_format_idx,
        format_func=lambda key: SUPPORTED_FORMATS[key],
    )

    columns = list(st.session_state.raw_df.columns)
    none_plus_cols = [None] + columns
    numeric_default = detected.get("numeric_candidates", [])
    selected_value_cols = st.multiselect("Value columns", columns, default=numeric_default)

    column_mapping = {
        "group": st.selectbox(
            "Group column",
            none_plus_cols,
            index=none_plus_cols.index(detected.get("group_col")) if detected.get("group_col") in none_plus_cols else 0,
        ),
        "subject": st.selectbox(
            "Subject column",
            none_plus_cols,
            index=none_plus_cols.index(detected.get("subject_col")) if detected.get("subject_col") in none_plus_cols else 0,
        ),
        "time": st.selectbox(
            "Time column",
            none_plus_cols,
            index=none_plus_cols.index(detected.get("time_col")) if detected.get("time_col") in none_plus_cols else 0,
        ),
        "factor2": st.selectbox(
            "Factor2 column",
            none_plus_cols,
            index=none_plus_cols.index(detected.get("factor2_col")) if detected.get("factor2_col") in none_plus_cols else 0,
        ),
        "value_cols": selected_value_cols,
    }

    if format_type in {"wide_time", "replicate"}:
        column_mapping["wide_value_cols"] = st.multiselect(
            "Wide/replicate columns",
            columns,
            default=selected_value_cols,
        )

    if st.button("Confirm Mapping And Normalize", type="primary"):
        st.session_state.column_mapping = column_mapping
        normalize_result = normalize_to_long(st.session_state.raw_df, column_mapping, format_type)
        st.session_state.normalized_df = normalize_result["normalized_df"]
        st.session_state.detected_schema = normalize_result["detected_schema"]
        st.session_state.warnings = normalize_result["warnings"]
        st.session_state.analysis_status = normalize_result["analysis_status"]

        if st.session_state.normalized_df is not None:
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
            )
            st.session_state.analysis_status = validation_result["analysis_status"]
            st.session_state.blocking_reasons = validation_result["blocking_reasons"]
            st.session_state.suggested_actions = validation_result["suggested_actions"]
            st.session_state.warnings = sorted(
                set(st.session_state.warnings + validation_result["warnings"])
            )

if st.session_state.normalized_df is not None:
    st.subheader("Normalized Preview")
    st.dataframe(st.session_state.normalized_df, use_container_width=True)

if st.session_state.warnings:
    for warning in st.session_state.warnings:
        st.warning(warning)

if st.session_state.blocking_reasons:
    st.error("Analysis blocked")
    st.write(pd.DataFrame({"blocking_reasons": st.session_state.blocking_reasons}))
