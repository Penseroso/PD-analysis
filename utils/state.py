from __future__ import annotations

from typing import Any

import streamlit as st


def get_state_defaults() -> dict[str, Any]:
    return {
        "raw_input_text": "",
        "raw_df": None,
        "preview_df": None,
        "detected_schema": {},
        "column_mapping": {},
        "normalized_df": None,
        "analysis_status": "needs_user_confirmation",
        "data_type": None,
        "between_factors": ["group"],
        "factor2_col": None,
        "selected_dv_cols": [],
        "control_group": None,
        "method_override": None,
        "analysis_results": {},
        "figure_objects": {},
        "export_bundle": {"html": None, "png": None, "csv": None, "stats_csv": None},
        "warnings": [],
        "blocking_reasons": [],
        "suggested_actions": [],
    }


def init_session_state() -> None:
    for key, value in get_state_defaults().items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_analysis_state() -> None:
    st.session_state.analysis_results = {}
    st.session_state.figure_objects = {}
    st.session_state.export_bundle = {"html": None, "png": None, "csv": None, "stats_csv": None}
    st.session_state.warnings = []
    st.session_state.blocking_reasons = []
    st.session_state.suggested_actions = []


def reset_all_state() -> None:
    defaults = get_state_defaults()
    for key, value in defaults.items():
        st.session_state[key] = value
