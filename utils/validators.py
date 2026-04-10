from __future__ import annotations

import pandas as pd

from utils.stats.diagnostics.design_inspector import inspect_repeated_structure, summarize_balance as summarize_design_balance
from utils.stats.validation.dataset_validator import validate_dataset


KEEP_LONG_BLOCKING_MESSAGE = (
    "Technical replicates are preserved in long form. They are not valid independent inferential units in the current app."
)
KEEP_LONG_BLOCKING_REASON = (
    "Inferential analysis is blocked because technical replicates were preserved with keep_long and the app does not model replicate structure explicitly."
)
KEEP_LONG_SUGGESTED_ACTION = "Rerun normalization with mean or median replicate collapse before running inferential analysis."
REPEATED_STRUCTURE_RECOMMENDATION = "Detected repeated-measures structure; longitudinal analysis is recommended."
REPEATED_STRUCTURE_BLOCKING_REASON = (
    "Cross-sectional inferential analysis is blocked because the normalized dataset contains repeated-measures structure (subject plus time)."
)
REPEATED_STRUCTURE_SUGGESTED_ACTION = "Switch Data type to longitudinal before running inferential analysis."



def detect_repeated_structure(
    df: pd.DataFrame,
    subject_col: str = "subject",
    time_col: str = "time",
) -> dict:
    return inspect_repeated_structure(df=df, subject_col=subject_col, time_col=time_col).to_dict()



def validate_normalized_df(
    df: pd.DataFrame,
    data_type: str,
    selected_dv_cols: list[str],
    between_factors: list[str],
    subject_col: str = "subject",
    group_col: str = "group",
    time_col: str = "time",
    factor2_col: str | None = None,
    control_group: str | None = None,
    replicate_preserved: bool = False,
    normalization_metadata: dict | None = None,
) -> dict:
    validation_payload = validate_dataset(
        df=df,
        data_type=data_type,
        selected_dv_cols=selected_dv_cols,
        between_factors=between_factors,
        subject_col=subject_col,
        group_col=group_col,
        time_col=time_col,
        factor2_col=factor2_col,
    )
    warnings = list(validation_payload.get("warnings", []))
    blocking_reasons = list(validation_payload.get("blocking_reasons", []))
    suggested_actions = list(validation_payload.get("suggested_actions", []))
    normalization_metadata = normalization_metadata or {}

    repeated_structure_info = detect_repeated_structure(df=df, subject_col=subject_col, time_col=time_col)
    if repeated_structure_info["detected"]:
        warnings.append(REPEATED_STRUCTURE_RECOMMENDATION)
        if data_type == "cross":
            blocking_reasons.append(REPEATED_STRUCTURE_BLOCKING_REASON)
            suggested_actions.append(REPEATED_STRUCTURE_SUGGESTED_ACTION)

    if control_group is None and data_type == "cross":
        warnings.append("Control-based post-hoc comparisons are unavailable until a control group is selected.")

    balance_info = summarize_balance(
        df=df,
        subject_col=subject_col,
        time_col=time_col if data_type == "longitudinal" else None,
        group_col=group_col if group_col in df.columns else None,
    )
    n_per_group = _count_observations_per_group(df, group_col, selected_dv_cols)
    subject_counts_per_group = balance_info.get("n_subjects_per_group", {})

    if replicate_preserved:
        warnings.append(KEEP_LONG_BLOCKING_MESSAGE)
        blocking_reasons.append(KEEP_LONG_BLOCKING_REASON)
        suggested_actions.append(KEEP_LONG_SUGGESTED_ACTION)
        replicate_id_col = normalization_metadata.get("replicate_id_col")
        if replicate_id_col and replicate_id_col in df.columns:
            warnings.append(
                f"Replicate identifier column '{replicate_id_col}' is present for exploratory preview/export, but replicate rows are not modeled as nested or repeated technical measurements."
            )

    if data_type == "longitudinal":
        if subject_counts_per_group and any(count < 2 for count in subject_counts_per_group.values()):
            warnings.append("Some groups have fewer than 2 unique subjects, which may block repeated-measures inference.")
        complete_counts = balance_info.get("n_complete_subjects_per_group", {})
        if complete_counts and any(count < 2 for count in complete_counts.values()):
            warnings.append("Some groups have fewer than 2 complete subjects across all time levels.")
        if balance_info.get("has_missing_repeated_cells", False):
            warnings.append("Some subjects are missing repeated cells across the expected time levels.")
    elif n_per_group and any(count < 2 for count in n_per_group.values()):
        warnings.append("Some groups have fewer than 2 observations, which may block inferential tests.")

    analysis_status = "blocked" if blocking_reasons else "ready"
    if analysis_status != "blocked" and (not between_factors or group_col not in between_factors):
        warnings.append("The default group factor is missing from between_factors.")

    return {
        "analysis_status": analysis_status,
        "warnings": sorted(set(warnings)),
        "blocking_reasons": sorted(set(blocking_reasons)),
        "suggested_actions": sorted(set(suggested_actions)),
        "balance_info": balance_info,
        "missingness_info": balance_info.get("missingness_info", {}),
        "n_per_group": n_per_group,
        "data_type": data_type,
        "between_factors": between_factors,
        "factor2_col": factor2_col,
        "control_group": control_group,
        "replicate_preserved": replicate_preserved,
        "normalization_metadata": normalization_metadata,
        "repeated_structure_info": repeated_structure_info,
        "recommended_data_type": repeated_structure_info.get("recommended_data_type", "cross"),
    }



def check_blocking_conditions(**kwargs) -> dict:
    return validate_normalized_df(**kwargs)



def summarize_balance(
    df: pd.DataFrame,
    subject_col: str,
    time_col: str | None,
    group_col: str | None = "group",
) -> dict:
    return summarize_design_balance(
        df=df,
        subject_col=subject_col,
        time_col=time_col,
        group_col=group_col,
    ).to_dict()



def _count_observations_per_group(df: pd.DataFrame, group_col: str, selected_dv_cols: list[str]) -> dict:
    if df is None or df.empty:
        return {}
    if group_col not in df.columns:
        return {}
    count_col = next((column for column in selected_dv_cols if column in df.columns), None)
    if count_col:
        return df.groupby(group_col)[count_col].count().to_dict()
    return df.groupby(group_col).size().to_dict()
