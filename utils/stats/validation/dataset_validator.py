from __future__ import annotations

import pandas as pd


def validate_dataset(
    df: pd.DataFrame,
    data_type: str,
    selected_dv_cols: list[str],
    between_factors: list[str],
    subject_col: str = "subject",
    group_col: str = "group",
    time_col: str = "time",
    factor2_col: str | None = None,
) -> dict:
    warnings: list[str] = []
    blocking_reasons: list[str] = []
    suggested_actions: list[str] = []

    if df is None or df.empty:
        blocking_reasons.append("No normalized data is available.")
        suggested_actions.append("Return to Upload And Mapping and normalize the input data.")
        return _build_response(
            analysis_status="blocked",
            warnings=warnings,
            blocking_reasons=blocking_reasons,
            suggested_actions=suggested_actions,
        )

    if not selected_dv_cols:
        blocking_reasons.append("No biomarker columns were selected.")
        suggested_actions.append("Select at least one value column in Analysis.")
    else:
        missing_dv = [column for column in selected_dv_cols if column not in df.columns]
        if missing_dv:
            blocking_reasons.append(f"Selected biomarker columns are missing: {', '.join(missing_dv)}.")
            suggested_actions.append("Refresh the biomarker selection or rerun normalization.")

    if group_col not in df.columns:
        blocking_reasons.append("The normalized dataset does not contain a group column.")
    elif data_type == "cross" and df[group_col].nunique(dropna=True) < 2:
        blocking_reasons.append("At least two groups are required for cross-sectional comparison.")
        suggested_actions.append("Verify the group mapping or choose a different analysis type.")

    if factor2_col and factor2_col not in df.columns:
        warnings.append("The selected factor2 column is not present in the normalized dataset.")

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

    if analysis_ready_without_method(df=df, between_factors=between_factors, group_col=group_col):
        warnings.append("The default group factor is missing from between_factors.")

    return _build_response(
        analysis_status="blocked" if blocking_reasons else "ready",
        warnings=warnings,
        blocking_reasons=blocking_reasons,
        suggested_actions=suggested_actions,
    )


def analysis_ready_without_method(df: pd.DataFrame, between_factors: list[str], group_col: str) -> bool:
    return df is not None and not df.empty and (not between_factors or group_col not in between_factors)


def _build_response(
    analysis_status: str,
    warnings: list[str],
    blocking_reasons: list[str],
    suggested_actions: list[str],
) -> dict:
    return {
        "analysis_status": analysis_status,
        "warnings": sorted(set(warnings)),
        "blocking_reasons": sorted(set(blocking_reasons)),
        "suggested_actions": sorted(set(suggested_actions)),
    }
