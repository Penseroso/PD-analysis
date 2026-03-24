from __future__ import annotations

import pandas as pd


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
) -> dict:
    warnings: list[str] = []
    blocking_reasons: list[str] = []
    suggested_actions: list[str] = []

    if df is None or df.empty:
        blocking_reasons.append("No normalized data is available.")
        suggested_actions.append("Return to Upload And Mapping and normalize the input data.")

    if not selected_dv_cols:
        blocking_reasons.append("No biomarker columns were selected.")
        suggested_actions.append("Select at least one value column in Analysis.")

    if group_col not in df.columns:
        blocking_reasons.append("The normalized dataset does not contain a group column.")
    elif df[group_col].nunique(dropna=True) < 2 and data_type == "cross":
        blocking_reasons.append("At least two groups are required for cross-sectional comparison.")
        suggested_actions.append("Verify the group mapping or choose a different analysis type.")

    if data_type == "longitudinal":
        if subject_col not in df.columns or df[subject_col].isna().all():
            blocking_reasons.append("Repeated-measures analysis requires a subject column.")
            suggested_actions.append("Map the subject ID column and normalize again.")
        if time_col not in df.columns or df[time_col].isna().all():
            blocking_reasons.append("Longitudinal analysis requires a time column.")
            suggested_actions.append("Map the time column and normalize again.")
        elif df[time_col].nunique(dropna=True) < 2:
            blocking_reasons.append("Longitudinal analysis requires at least two time levels.")
            suggested_actions.append("Confirm that the time variable was mapped correctly.")

    if factor2_col and factor2_col not in df.columns:
        warnings.append("The selected factor2 column is not present in the normalized dataset.")

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
        "blocking_reasons": blocking_reasons,
        "suggested_actions": suggested_actions,
        "balance_info": balance_info,
        "missingness_info": balance_info.get("missingness_info", {}),
        "n_per_group": n_per_group,
        "data_type": data_type,
        "between_factors": between_factors,
        "factor2_col": factor2_col,
        "control_group": control_group,
    }



def check_blocking_conditions(**kwargs) -> dict:
    return validate_normalized_df(**kwargs)



def summarize_balance(
    df: pd.DataFrame,
    subject_col: str,
    time_col: str | None,
    group_col: str | None = "group",
) -> dict:
    if df is None or df.empty or time_col is None or time_col not in df.columns or subject_col not in df.columns:
        return {
            "is_balanced": True,
            "has_missing_repeated_cells": False,
            "subject_observation_counts": {},
            "n_subjects_per_group": {},
            "n_complete_subjects_per_group": {},
            "missingness_info": {},
            "expected_time_levels": 0,
            "incomplete_subjects_by_group": {},
        }

    working_df = df[[subject_col, time_col] + ([group_col] if group_col and group_col in df.columns else [])].copy()
    working_df[subject_col] = working_df[subject_col].astype(str)
    working_df[time_col] = working_df[time_col].astype(str)
    if group_col and group_col in working_df.columns:
        working_df[group_col] = working_df[group_col].astype(str)

    counts = working_df.groupby(subject_col)[time_col].nunique(dropna=True).to_dict()
    expected_levels = int(working_df[time_col].nunique(dropna=True))
    is_balanced = len(set(counts.values())) <= 1 and all(count == expected_levels for count in counts.values())
    has_missing = any(count < expected_levels for count in counts.values())

    n_subjects_per_group: dict[str, int] = {}
    n_complete_subjects_per_group: dict[str, int] = {}
    incomplete_subjects_by_group: dict[str, list[str]] = {}
    if group_col and group_col in working_df.columns:
        n_subjects_per_group = (
            working_df[[group_col, subject_col]]
            .drop_duplicates()
            .groupby(group_col, sort=False)[subject_col]
            .nunique(dropna=True)
            .astype(int)
            .to_dict()
        )
        subject_time_counts = (
            working_df.groupby([group_col, subject_col], sort=False)[time_col]
            .nunique(dropna=True)
            .reset_index(name="n_time_levels")
        )
        n_complete_subjects_per_group = (
            subject_time_counts.loc[subject_time_counts["n_time_levels"] >= expected_levels]
            .groupby(group_col, sort=False)[subject_col]
            .nunique(dropna=True)
            .astype(int)
            .to_dict()
        )
        for group_name in n_subjects_per_group:
            n_complete_subjects_per_group.setdefault(group_name, 0)
        incomplete_subjects_by_group = (
            subject_time_counts.loc[subject_time_counts["n_time_levels"] < expected_levels]
            .groupby(group_col, sort=False)[subject_col]
            .apply(lambda series: [str(item) for item in series.tolist()])
            .to_dict()
        )

    return {
        "is_balanced": is_balanced,
        "has_missing_repeated_cells": has_missing,
        "subject_observation_counts": counts,
        "n_subjects_per_group": n_subjects_per_group,
        "n_complete_subjects_per_group": n_complete_subjects_per_group,
        "expected_time_levels": expected_levels,
        "incomplete_subjects_by_group": incomplete_subjects_by_group,
        "missingness_info": {
            "expected_time_levels": expected_levels,
            "subjects_with_missing": [subject for subject, count in counts.items() if count < expected_levels],
        },
    }



def _count_observations_per_group(df: pd.DataFrame, group_col: str, selected_dv_cols: list[str]) -> dict:
    if group_col not in df.columns:
        return {}
    count_col = next((column for column in selected_dv_cols if column in df.columns), None)
    if count_col:
        return df.groupby(group_col)[count_col].count().to_dict()
    return df.groupby(group_col).size().to_dict()
