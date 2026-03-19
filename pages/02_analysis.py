from __future__ import annotations

import streamlit as st

from config import ANALYSIS_METHODS, DEFAULT_FIGURE_CONFIG
from utils.state import reset_analysis_state
from utils.stats_cross import compute_cross_assumptions, run_cross_sectional
from utils.stats_longitudinal import compute_longitudinal_assumptions, run_longitudinal
from utils.stats_mixedlm import run_mixedlm
from utils.stats_selector import build_analysis_plan, select_method
from utils.validators import validate_normalized_df
from utils.viz_cross import make_figure as make_cross_figure
from utils.viz_longitudinal import make_figure as make_longitudinal_figure


st.title("Analysis")

df = st.session_state.normalized_df
if df is None:
    st.info("Normalize data in the upload page first.")
    st.stop()

all_dv_cols = [col for col in df.columns if col.startswith("value_")]
if not all_dv_cols:
    st.error("No numeric value columns were detected.")
    st.stop()

data_type = st.radio("Data type", options=["cross", "longitudinal"], horizontal=True)
st.session_state.data_type = data_type

default_dv = st.session_state.selected_dv_cols or all_dv_cols[:1]
selected_dv_cols = st.multiselect("Biomarkers", all_dv_cols, default=default_dv)
st.session_state.selected_dv_cols = selected_dv_cols

group_levels = sorted(df["group"].dropna().astype(str).unique().tolist()) if "group" in df.columns else []
control_group = st.selectbox("Control group", options=[None] + group_levels, index=0)
st.session_state.control_group = control_group

between_factors = ["group"]
factor2_col = None
factor_candidates = [
    col for col in df.columns if col not in {"subject", "group", "time"} and not col.startswith("value_")
]
if factor_candidates:
    factor2_col = st.selectbox("Optional factor2", options=[None] + factor_candidates, index=0)
    if factor2_col:
        between_factors.append(factor2_col)
st.session_state.between_factors = between_factors
st.session_state.factor2_col = factor2_col

method_override = st.selectbox(
    "Method override",
    options=ANALYSIS_METHODS[data_type],
    index=0,
)
st.session_state.method_override = None if method_override == "auto" else method_override

if st.button("Run Analysis", type="primary"):
    reset_analysis_state()
    validation_result = validate_normalized_df(
        df=df,
        data_type=data_type,
        selected_dv_cols=selected_dv_cols,
        between_factors=between_factors,
        factor2_col=factor2_col,
        control_group=control_group,
    )
    st.session_state.analysis_status = validation_result["analysis_status"]
    st.session_state.blocking_reasons = validation_result["blocking_reasons"]
    st.session_state.suggested_actions = validation_result["suggested_actions"]
    st.session_state.warnings = validation_result["warnings"]

    if validation_result["analysis_status"] == "blocked":
        st.rerun()

    analysis_results: dict[str, dict] = {}
    figure_objects: dict[str, object] = {}

    for dv_col in selected_dv_cols:
        if data_type == "cross":
            assumptions = compute_cross_assumptions(df=df, dv_col=dv_col, group_col="group")
            selector_result = select_method(
                data_type=data_type,
                normality=assumptions.get("normality", {}),
                sphericity=None,
                levene=assumptions.get("levene", {}),
                balance_info=validation_result["balance_info"],
                between_factors=between_factors,
                n_per_group=validation_result["n_per_group"],
            )
            plan = build_analysis_plan(validation_result, selector_result, st.session_state.method_override)
            result = run_cross_sectional(
                df=df,
                dv_col=dv_col,
                group_col="group",
                control_group=control_group,
                method=plan["final_method"],
            )
            result["selector"] = selector_result
            analysis_results[dv_col] = result
            figure_objects[dv_col] = make_cross_figure(df=df, result=result, config=DEFAULT_FIGURE_CONFIG)
        else:
            assumptions = compute_longitudinal_assumptions(
                df=df,
                dv_col=dv_col,
                group_col="group",
                subject_col="subject",
                time_col="time",
                between_factors=between_factors,
            )
            selector_result = select_method(
                data_type=data_type,
                normality=assumptions.get("normality", {}),
                sphericity=assumptions.get("sphericity"),
                levene={},
                balance_info=validation_result["balance_info"],
                between_factors=between_factors,
                n_per_group=validation_result["n_per_group"],
            )
            plan = build_analysis_plan(validation_result, selector_result, st.session_state.method_override)
            if plan["engine"] == "statsmodels":
                result = run_mixedlm(
                    df=df,
                    dv_col=dv_col,
                    subject_col="subject",
                    time_col="time",
                    group_col="group",
                    factor2_col=factor2_col,
                    formula_mode="default",
                )
            else:
                result = run_longitudinal(
                    df=df,
                    dv_col=dv_col,
                    group_col="group",
                    subject_col="subject",
                    time_col="time",
                    control_group=control_group,
                    between_factors=between_factors,
                    factor2_col=factor2_col,
                    method=plan["final_method"],
                )
            result["selector"] = selector_result
            analysis_results[dv_col] = result
            figure_objects[dv_col] = make_longitudinal_figure(
                df=df, result=result, config=DEFAULT_FIGURE_CONFIG
            )

    st.session_state.analysis_results = analysis_results
    st.session_state.figure_objects = figure_objects

if st.session_state.analysis_results:
    for dv_col, result in st.session_state.analysis_results.items():
        st.subheader(dv_col)
        st.json(
            {
                "analysis_status": result.get("analysis_status"),
                "used_method": result.get("used_method", result.get("used_formula")),
                "warnings": result.get("warnings", []),
            }
        )
        fig = st.session_state.figure_objects.get(dv_col)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)

if st.session_state.blocking_reasons:
    for reason in st.session_state.blocking_reasons:
        st.error(reason)
