from __future__ import annotations

import streamlit as st

from config import ANALYSIS_METHODS, DEFAULT_FIGURE_CONFIG
from utils.state import reset_analysis_state
from utils.stats_cross import compute_cross_assumptions, run_cross_sectional
from utils.stats_longitudinal import compute_longitudinal_assumptions, run_longitudinal
from utils.stats_mixedlm import run_mixedlm
from utils.stats_selector import build_analysis_plan, select_method
from utils.validators import detect_repeated_structure, validate_normalized_df
from utils.viz_cross import make_figure as make_cross_figure
from utils.viz_longitudinal import make_figure as make_longitudinal_figure


KEEP_LONG_BLOCKING_REASON = (
    "Inferential analysis is blocked because technical replicates were preserved with keep_long and the app does not model replicate structure explicitly."
)
KEEP_LONG_SUGGESTED_ACTION = "Rerun normalization with mean or median replicate collapse before running inferential analysis."
REPEATED_STRUCTURE_BLOCKING_REASON = (
    "Cross-sectional inferential analysis is blocked because the normalized dataset contains repeated-measures structure (subject plus time)."
)


st.title("Analysis")

df = st.session_state.normalized_df
if df is None:
    st.info("Normalize data in the upload page first.")
    st.stop()

all_dv_cols = [col for col in df.columns if col.startswith("value_")]
if not all_dv_cols:
    st.error("No numeric value columns were detected.")
    st.stop()

value_display_map = st.session_state.value_display_map or {}
time_order = st.session_state.time_order or []
time_order_metadata = st.session_state.time_order_metadata or {}
normalization_metadata = st.session_state.normalization_metadata or {}
replicate_preserved = bool(st.session_state.replicate_preserved)
repeated_structure_info = detect_repeated_structure(df)


def _label_for_dv(column: str) -> str:
    return value_display_map.get(column, column)


preferred_data_type = "longitudinal" if repeated_structure_info.get("detected") else "cross"
current_data_type = st.session_state.data_type if st.session_state.data_type in {"cross", "longitudinal"} else preferred_data_type
data_type = st.radio(
    "Data type",
    options=["cross", "longitudinal"],
    horizontal=True,
    index=0 if current_data_type == "cross" else 1,
)
st.session_state.data_type = data_type

default_dv = [col for col in (st.session_state.selected_dv_cols or all_dv_cols[:1]) if col in all_dv_cols] or all_dv_cols[:1]
selected_dv_cols = st.multiselect("Biomarkers", all_dv_cols, default=default_dv, format_func=_label_for_dv)
st.session_state.selected_dv_cols = selected_dv_cols

if repeated_structure_info.get("detected"):
    st.info("Detected repeated-measures structure; longitudinal analysis is recommended.")
    if data_type == "cross":
        st.warning(REPEATED_STRUCTURE_BLOCKING_REASON)

if data_type == "longitudinal" and time_order:
    st.caption("Time order: " + " -> ".join(time_order))
    if time_order_metadata.get("ambiguous"):
        st.warning(time_order_metadata.get("warning", "Ambiguous time labels fell back to first-seen order."))

if replicate_preserved:
    replicate_id_col = normalization_metadata.get("replicate_id_col", "replicate_id")
    st.error(KEEP_LONG_BLOCKING_REASON)
    st.caption(
        f"Preserved replicate column: {replicate_id_col}. Exploratory preview/export is available, but inferential tests are disabled for this normalized dataset."
    )

group_levels = sorted(df["group"].dropna().astype(str).unique().tolist()) if "group" in df.columns else []

between_factors = ["group"]
factor2_col = None
factor_candidates = [
    col for col in df.columns if col not in {"subject", "group", "time", "replicate", "replicate_id"} and not col.startswith("value_")
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

preview_validation = validate_normalized_df(
    df=df,
    data_type=data_type,
    selected_dv_cols=selected_dv_cols,
    between_factors=between_factors,
    factor2_col=factor2_col,
    control_group=None,
    replicate_preserved=replicate_preserved,
    normalization_metadata=normalization_metadata,
)
preview_effective_method = st.session_state.method_override
if data_type == "longitudinal" and selected_dv_cols:
    preview_dv = selected_dv_cols[0]
    preview_assumptions = compute_longitudinal_assumptions(
        df=df,
        dv_col=preview_dv,
        group_col="group",
        subject_col="subject",
        time_col="time",
        between_factors=between_factors,
        time_order=time_order,
    )
    preview_selector = select_method(
        data_type=data_type,
        normality=preview_assumptions.get("normality", {}),
        sphericity=preview_assumptions.get("sphericity"),
        levene={},
        balance_info=preview_validation["balance_info"],
        between_factors=between_factors,
        n_per_group=preview_validation["n_per_group"],
    )
    preview_effective_method = st.session_state.method_override or preview_selector["recommended_method"]
else:
    preview_selector = None

control_group = None
reference_group = None
if data_type == "cross":
    control_group = st.selectbox("Control group", options=[None] + group_levels, index=0)
    st.session_state.control_group = control_group
    st.session_state.reference_group = None
else:
    st.session_state.control_group = None
    if group_levels and preview_effective_method == "mixedlm":
        reference_group = st.selectbox("Reference group for contrasts", options=[None] + group_levels, index=0)
    st.session_state.reference_group = reference_group

if st.button("Run Analysis", type="primary"):
    reset_analysis_state()
    validation_result = validate_normalized_df(
        df=df,
        data_type=data_type,
        selected_dv_cols=selected_dv_cols,
        between_factors=between_factors,
        factor2_col=factor2_col,
        control_group=control_group,
        replicate_preserved=replicate_preserved,
        normalization_metadata=normalization_metadata,
    )
    st.session_state.analysis_status = validation_result["analysis_status"]
    st.session_state.blocking_reasons = validation_result["blocking_reasons"]
    st.session_state.suggested_actions = validation_result["suggested_actions"]
    st.session_state.warnings = validation_result["warnings"]

    if validation_result["analysis_status"] == "blocked":
        if replicate_preserved:
            st.session_state.blocking_reasons = sorted(set(st.session_state.blocking_reasons + [KEEP_LONG_BLOCKING_REASON]))
            st.session_state.suggested_actions = sorted(set(st.session_state.suggested_actions + [KEEP_LONG_SUGGESTED_ACTION]))
        st.rerun()

    analysis_results: dict[str, dict] = {}
    figure_objects: dict[str, object] = {}
    aggregated_blocking: list[str] = []

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
            if plan["analysis_status"] == "blocked":
                result = {
                    "analysis_status": "blocked",
                    "used_method": plan["final_method"],
                    "warnings": plan.get("warnings", []),
                    "blocking_reasons": plan.get("blocking_reasons", []),
                    "suggested_actions": plan.get("suggested_actions", validation_result.get("suggested_actions", [])),
                    "star_map": [],
                    "omnibus": None,
                    "posthoc_table": None,
                    "dv_col": dv_col,
                    "selector": selector_result,
                }
            else:
                result = run_cross_sectional(
                    df=df,
                    dv_col=dv_col,
                    group_col="group",
                    control_group=control_group,
                    method=plan["final_method"],
                )
                result["warnings"] = sorted(set(result.get("warnings", []) + plan.get("warnings", [])))
                result["selector"] = selector_result
            analysis_results[dv_col] = result
            if result.get("analysis_status") == "ready":
                figure_objects[dv_col] = make_cross_figure(df=df, result=result, config=DEFAULT_FIGURE_CONFIG)
            else:
                aggregated_blocking.extend(result.get("blocking_reasons", []))
        else:
            assumptions = compute_longitudinal_assumptions(
                df=df,
                dv_col=dv_col,
                group_col="group",
                subject_col="subject",
                time_col="time",
                between_factors=between_factors,
                time_order=time_order,
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
            if plan["analysis_status"] == "blocked":
                result = {
                    "analysis_status": "blocked",
                    "used_method": plan["final_method"],
                    "warnings": plan.get("warnings", []),
                    "blocking_reasons": plan.get("blocking_reasons", []),
                    "suggested_actions": plan.get("suggested_actions", validation_result.get("suggested_actions", [])),
                    "star_map": [],
                    "omnibus": None,
                    "posthoc_table": None,
                    "dv_col": dv_col,
                    "selector": selector_result,
                    "time_order": time_order,
                }
            elif plan["engine"] == "statsmodels":
                result = run_mixedlm(
                    df=df,
                    dv_col=dv_col,
                    subject_col="subject",
                    time_col="time",
                    group_col="group",
                    factor2_col=factor2_col,
                    formula_mode="default",
                    reference_group=reference_group,
                    time_order=time_order,
                )
                result["warnings"] = sorted(set(result.get("warnings", []) + plan.get("warnings", [])))
            else:
                result = run_longitudinal(
                    df=df,
                    dv_col=dv_col,
                    group_col="group",
                    subject_col="subject",
                    time_col="time",
                    control_group=None,
                    between_factors=between_factors,
                    factor2_col=factor2_col,
                    method=plan["final_method"],
                    time_order=time_order,
                )
                result["warnings"] = sorted(set(result.get("warnings", []) + plan.get("warnings", [])))
            result["selector"] = selector_result
            result["time_order"] = time_order
            analysis_results[dv_col] = result
            if result.get("analysis_status") == "ready":
                figure_objects[dv_col] = make_longitudinal_figure(
                    df=df, result=result, config=DEFAULT_FIGURE_CONFIG, time_order=time_order
                )
            else:
                aggregated_blocking.extend(result.get("blocking_reasons", []))

    st.session_state.analysis_results = analysis_results
    st.session_state.figure_objects = figure_objects
    st.session_state.blocking_reasons = sorted(set(st.session_state.blocking_reasons + aggregated_blocking))

if st.session_state.analysis_results:
    for dv_col, result in st.session_state.analysis_results.items():
        st.subheader(_label_for_dv(dv_col))
        st.caption(f"Internal column: {dv_col}")
        st.json(
            {
                "analysis_status": result.get("analysis_status"),
                "used_method": result.get("used_method"),
                "used_formula": result.get("used_formula"),
                "time_order": result.get("time_order", []),
                "warnings": result.get("warnings", []),
                "blocking_reasons": result.get("blocking_reasons", []),
            }
        )
        fig = st.session_state.figure_objects.get(dv_col)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)

if st.session_state.blocking_reasons:
    for reason in st.session_state.blocking_reasons:
        st.error(reason)

if st.session_state.suggested_actions:
    for action in st.session_state.suggested_actions:
        st.info(action)
