from __future__ import annotations

import pandas as pd

from utils.stats.contracts.diagnostics import BalanceSummary, RepeatedStructureSummary


def inspect_repeated_structure(
    df: pd.DataFrame,
    subject_col: str = "subject",
    time_col: str = "time",
) -> RepeatedStructureSummary:
    if df is None or df.empty:
        return RepeatedStructureSummary(reason="No normalized data is available.")

    has_subject_column = subject_col in df.columns and df[subject_col].notna().any()
    has_time_column = time_col in df.columns and df[time_col].notna().any()
    if not has_subject_column or not has_time_column:
        return RepeatedStructureSummary(
            has_subject_column=has_subject_column,
            has_time_column=has_time_column,
            reason="Repeated-measures structure requires both subject and time columns.",
        )

    working_df = df[[subject_col, time_col]].dropna().copy()
    if working_df.empty:
        return RepeatedStructureSummary(
            has_subject_column=has_subject_column,
            has_time_column=has_time_column,
            reason="Subject/time columns are present but contain no analyzable repeated rows.",
        )

    working_df[subject_col] = working_df[subject_col].astype(str)
    working_df[time_col] = working_df[time_col].astype(str)
    n_time_levels = int(working_df[time_col].nunique(dropna=True))
    per_subject = working_df.groupby(subject_col, sort=False)[time_col].nunique(dropna=True)
    n_multi = int((per_subject >= 2).sum())
    detected = n_time_levels >= 2 and n_multi >= 1
    reason = (
        "Subject and time columns with repeated observations across at least two time levels were detected."
        if detected
        else "Subject/time columns are present, but repeated observations across multiple time levels were not clearly detected."
    )
    return RepeatedStructureSummary(
        detected=detected,
        has_subject_column=has_subject_column,
        has_time_column=has_time_column,
        n_time_levels=n_time_levels,
        n_subjects_with_multiple_timepoints=n_multi,
        recommended_data_type="longitudinal" if detected else "cross",
        reason=reason,
    )


def summarize_balance(
    df: pd.DataFrame,
    subject_col: str,
    time_col: str | None,
    group_col: str | None = "group",
) -> BalanceSummary:
    if df is None or df.empty or time_col is None or time_col not in df.columns or subject_col not in df.columns:
        return BalanceSummary()

    keep_cols = [subject_col, time_col] + ([group_col] if group_col and group_col in df.columns else [])
    working_df = df[keep_cols].dropna(subset=[subject_col, time_col]).copy()
    if working_df.empty:
        return BalanceSummary()

    working_df[subject_col] = working_df[subject_col].astype(str)
    working_df[time_col] = working_df[time_col].astype(str)
    if group_col and group_col in working_df.columns:
        working_df[group_col] = working_df[group_col].astype(str)

    expected_levels = working_df[time_col].dropna().drop_duplicates().tolist()
    expected_time_levels = len(expected_levels)
    subject_observation_counts = (
        working_df.groupby(subject_col, sort=False)[time_col].nunique(dropna=True).astype(int).to_dict()
    )
    is_balanced = len(set(subject_observation_counts.values())) <= 1 and all(
        count == expected_time_levels for count in subject_observation_counts.values()
    )
    missing_subjects = [subject for subject, count in subject_observation_counts.items() if count < expected_time_levels]

    n_subjects_per_group: dict[str, int] = {}
    n_complete_subjects_per_group: dict[str, int] = {}
    incomplete_subjects_by_group: dict[str, list[str]] = {}
    missing_repeated_cells: list[dict[str, str]] = []

    if group_col and group_col in working_df.columns:
        n_subjects_per_group = (
            working_df[[group_col, subject_col]]
            .drop_duplicates()
            .groupby(group_col, sort=False)[subject_col]
            .nunique(dropna=True)
            .astype(int)
            .to_dict()
        )
        subject_time = (
            working_df.groupby([group_col, subject_col], sort=False)[time_col]
            .agg(lambda values: {str(item) for item in values.dropna().tolist()})
            .reset_index(name="observed_time_levels")
        )
        for row in subject_time.itertuples(index=False):
            observed = set(row.observed_time_levels)
            missing = [level for level in expected_levels if level not in observed]
            if not missing:
                n_complete_subjects_per_group[row[0]] = n_complete_subjects_per_group.get(row[0], 0) + 1
                continue
            incomplete_subjects_by_group.setdefault(row[0], []).append(str(row[1]))
            for level in missing:
                missing_repeated_cells.append(
                    {"group": str(row[0]), "subject": str(row[1]), "time": str(level)}
                )
        for group_name in n_subjects_per_group:
            n_complete_subjects_per_group.setdefault(group_name, 0)
    else:
        observed_by_subject = (
            working_df.groupby(subject_col, sort=False)[time_col]
            .agg(lambda values: {str(item) for item in values.dropna().tolist()})
            .to_dict()
        )
        for subject, observed in observed_by_subject.items():
            for level in expected_levels:
                if level not in observed:
                    missing_repeated_cells.append({"group": "", "subject": str(subject), "time": str(level)})

    return BalanceSummary(
        is_balanced=is_balanced,
        has_missing_repeated_cells=bool(missing_repeated_cells),
        subject_observation_counts=subject_observation_counts,
        n_subjects_per_group=n_subjects_per_group,
        n_complete_subjects_per_group=n_complete_subjects_per_group,
        missingness_info={
            "expected_time_levels": expected_time_levels,
            "subjects_with_missing": missing_subjects,
        },
        expected_time_levels=expected_time_levels,
        incomplete_subjects_by_group=incomplete_subjects_by_group,
        missing_repeated_cells=missing_repeated_cells,
    )
