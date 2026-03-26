from __future__ import annotations

import streamlit as st

from config import ANALYSIS_METHODS, DEFAULT_FIGURE_CONFIG
from utils.state import init_session_state, reset_analysis_state
from utils.ui import render_top_nav
from utils.stats_cross import compute_cross_assumptions, run_cross_sectional
from utils.stats_longitudinal import compute_longitudinal_assumptions, run_longitudinal
from utils.stats_mixedlm import run_mixedlm
from utils.stats_selector import build_analysis_plan, select_method
from utils.validators import detect_repeated_structure, validate_normalized_df
from utils.viz_cross import make_figure as make_cross_figure
from utils.viz_longitudinal import make_figure as make_longitudinal_figure


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


KEEP_LONG_BLOCKING_REASON = (
    "Inferential analysis is blocked because technical replicates were preserved with keep_long and the app does not model replicate structure explicitly."
)
KEEP_LONG_SUGGESTED_ACTION = "Rerun normalization with mean or median replicate collapse before running inferential analysis."
REPEATED_STRUCTURE_BLOCKING_REASON = (
    "Cross-sectional inferential analysis is blocked because the normalized dataset contains repeated-measures structure (subject plus time)."
)


init_session_state()
st.markdown(CHROME_CSS, unsafe_allow_html=True)
render_top_nav("Analysis", "Review readiness, configure the inferential plan, and run the analysis workbench.")
st.title("Analysis")
st.caption("Review normalized inputs, configure the inferential plan, run the analysis, and inspect results before export.")

df = st.session_state.normalized_df
if df is None:
    with st.container(border=True):
        st.subheader("Dataset Required")
        st.info("Normalize data on the upload page before opening the analysis workbench.")
    st.stop()

all_dv_cols = [col for col in df.columns if col.startswith("value_")]
if not all_dv_cols:
    with st.container(border=True):
        st.subheader("Dataset Required")
        st.error("No numeric value columns were detected in the normalized dataset.")
    st.stop()

value_display_map = st.session_state.value_display_map or {}
time_order = st.session_state.time_order or []
time_order_metadata = st.session_state.time_order_metadata or {}
normalization_metadata = st.session_state.normalization_metadata or {}
replicate_preserved = bool(st.session_state.replicate_preserved)
repeated_structure_info = detect_repeated_structure(df)
analysis_result_count = len(st.session_state.analysis_results or {})
figure_count = len(st.session_state.figure_objects or {})
blocking_count = len(st.session_state.blocking_reasons or [])


def _label_for_dv(column: str) -> str:
    return value_display_map.get(column, column)


preferred_data_type = "longitudinal" if repeated_structure_info.get("detected") else "cross"
current_data_type = st.session_state.data_type if st.session_state.data_type in {"cross", "longitudinal"} else preferred_data_type
group_levels = sorted(df["group"].dropna().astype(str).unique().tolist()) if "group" in df.columns else []
default_dv = [col for col in (st.session_state.selected_dv_cols or all_dv_cols[:1]) if col in all_dv_cols] or all_dv_cols[:1]
selected_dv_count = len(default_dv)

with st.container(border=True):
    st.markdown('<p class="section-eyebrow">Dataset Status</p>', unsafe_allow_html=True)
    st.subheader("Analysis Context")
    kpi_rows = [
        [
            ("Structure", "Repeated-measures" if repeated_structure_info.get("detected") else "Cross-sectional", "Detected dataset structure"),
            ("Biomarkers available", str(len(all_dv_cols)), "Value columns ready for analysis"),
            ("Biomarkers selected", str(selected_dv_count), "Current biomarker selection"),
            ("Group levels", str(len(group_levels)), "Distinct group labels"),
        ],
        [
            ("Replicates", "Preserved" if replicate_preserved else "Collapsed", "Technical replicate state"),
            ("Analysis results", str(analysis_result_count), "Results stored in session"),
            ("Figures", str(figure_count), "Generated figure objects"),
            ("Blocking items", str(blocking_count), "Current blocking reasons"),
        ],
    ]
    for row in kpi_rows:
        cols = st.columns(4)
        for col, (label, value, helper) in zip(cols, row):
            with col:
                with st.container(border=True):
                    st.markdown(f'<p class="summary-label">{label}</p>', unsafe_allow_html=True)
                    st.markdown(f'<p class="summary-value">{value}</p>', unsafe_allow_html=True)
                    st.markdown(f'<p class="kpi-helper">{helper}</p>', unsafe_allow_html=True)
    if time_order:
        st.caption("Inferred time order: " + " -> ".join(time_order))
        if time_order_metadata.get("ambiguous"):
            st.warning(time_order_metadata.get("warning", "Ambiguous time labels fell back to first-seen order."))

between_factors = ["group"]
factor2_col = None
factor_candidates = [
    col for col in df.columns if col not in {"subject", "group", "time", "replicate", "replicate_id"} and not col.startswith("value_")
]

with st.container(border=True):
    st.markdown('<p class="section-eyebrow">Configuration</p>', unsafe_allow_html=True)
    st.subheader("Analysis Setup")

    row1_col1, row1_col2 = st.columns([1, 2])
    with row1_col1:
        data_type = st.radio(
            "Data type",
            options=["cross", "longitudinal"],
            horizontal=True,
            index=0 if current_data_type == "cross" else 1,
        )
    st.session_state.data_type = data_type

    with row1_col2:
        selected_dv_cols = st.multiselect("Biomarkers", all_dv_cols, default=default_dv, format_func=_label_for_dv)
    st.session_state.selected_dv_cols = selected_dv_cols

    row2_col1, row2_col2 = st.columns(2)
    with row2_col1:
        if factor_candidates:
            factor2_col = st.selectbox("Optional factor2", options=[None] + factor_candidates, index=0)
            if factor2_col:
                between_factors.append(factor2_col)
        else:
            st.selectbox("Optional factor2", options=[None], index=0, disabled=True)
            factor2_col = None
    st.session_state.between_factors = between_factors
    st.session_state.factor2_col = factor2_col

    with row2_col2:
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

    row3_col1, _ = st.columns([1, 2])
    control_group = None
    reference_group = None
    with row3_col1:
        if data_type == "cross":
            control_group = st.selectbox("Control group", options=[None] + group_levels, index=0)
            st.session_state.control_group = control_group
            st.session_state.reference_group = None
        else:
            st.session_state.control_group = None
            if group_levels and preview_effective_method == "mixedlm":
                reference_group = st.selectbox("Reference group for contrasts", options=[None] + group_levels, index=0)
            st.session_state.reference_group = reference_group

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

selected_labels = [_label_for_dv(col) for col in selected_dv_cols]
method_preview_text = preview_effective_method if data_type == "longitudinal" else (st.session_state.method_override or "auto")
constraint_lines = []
for warning in preview_validation.get("warnings", []):
    constraint_lines.append(f"Warning: {warning}")
for reason in preview_validation.get("blocking_reasons", []):
    constraint_lines.append(f"Blocking: {reason}")
if replicate_preserved and KEEP_LONG_BLOCKING_REASON not in preview_validation.get("blocking_reasons", []):
    constraint_lines.append(f"Blocking: {KEEP_LONG_BLOCKING_REASON}")

with st.container(border=True):
    st.markdown('<p class="section-eyebrow">Pre-run Check</p>', unsafe_allow_html=True)
    st.subheader("Pre-run Validation / Method Preview")
    summary_cols = st.columns(3)
    with summary_cols[0]:
        st.markdown(f"**Selected data type**\n\n{data_type}")
    with summary_cols[1]:
        st.markdown("**Biomarkers**\n\n" + (", ".join(selected_labels) if selected_labels else "None selected"))
    with summary_cols[2]:
        st.markdown("**Effective method preview**\n\n" + (method_preview_text or "auto"))
    detail_cols = st.columns(2)
    with detail_cols[0]:
        st.markdown("**Between-factors**")
        st.write(", ".join(between_factors) if between_factors else "None")
        if factor2_col:
            st.markdown(f"**Optional factor2**\n\n{factor2_col}")
    with detail_cols[1]:
        status_text = preview_validation.get("analysis_status", "ready")
        st.markdown(f"**Validation status**\n\n{status_text}")
        if constraint_lines:
            for line in constraint_lines:
                st.write(f"- {line}")
        else:
            st.write("No pre-run warnings or blocking constraints detected for the current selections.")

with st.container(border=True):
    st.markdown('<p class="section-eyebrow">Execution</p>', unsafe_allow_html=True)
    st.subheader("Run Analysis")
    st.caption("Execute inferential analysis for the selected biomarkers using the current configuration.")
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
            dv_label = _label_for_dv(dv_col)
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
                        "dv_label": dv_label,
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
                    result["dv_label"] = dv_label
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
                        "dv_label": dv_label,
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
                    result["dv_label"] = dv_label
                else:
                    result = run_longitudinal(
                        df=df,
                        dv_col=dv_col,
                        group_col="group",
                        subject_col="subject",
                        time_col="time",
                        between_factors=between_factors,
                        factor2_col=factor2_col,
                        method=plan["final_method"],
                        time_order=time_order,
                    )
                    result["warnings"] = sorted(set(result.get("warnings", []) + plan.get("warnings", [])))
                    result["dv_label"] = dv_label
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
    with st.container(border=True):
        st.markdown('<p class="section-eyebrow">Outputs</p>', unsafe_allow_html=True)
        st.subheader("Results Review")
        for dv_col, result in st.session_state.analysis_results.items():
            with st.container(border=True):
                st.markdown(f"**{_label_for_dv(dv_col)}**")
                st.caption(f"Internal column: {dv_col}")
                summary_lines = [
                    f"Analysis status: {result.get('analysis_status')}",
                    f"Used method: {result.get('used_method')}",
                ]
                if result.get("used_formula"):
                    summary_lines.append(f"Used formula: {result.get('used_formula')}")
                if result.get("time_order"):
                    summary_lines.append("Time order: " + " -> ".join(result.get("time_order", [])))
                for line in summary_lines:
                    st.write(f"- {line}")
                if result.get("warnings"):
                    st.markdown("**Warnings**")
                    for warning in result.get("warnings", []):
                        st.warning(warning)
                if result.get("blocking_reasons"):
                    st.markdown("**Blocking reasons**")
                    for reason in result.get("blocking_reasons", []):
                        st.error(reason)
                fig = st.session_state.figure_objects.get(dv_col)
                if fig is not None:
                    st.plotly_chart(fig, use_container_width=True)

if st.session_state.blocking_reasons or st.session_state.suggested_actions:
    with st.container(border=True):
        st.markdown('<p class="section-eyebrow">Review Notes</p>', unsafe_allow_html=True)
        st.subheader("Action Items")
        if st.session_state.blocking_reasons:
            st.markdown("**Blocking reasons**")
            for reason in st.session_state.blocking_reasons:
                st.error(reason)
        if st.session_state.suggested_actions:
            st.markdown("**Suggested actions**")
            for action in st.session_state.suggested_actions:
                st.info(action)
