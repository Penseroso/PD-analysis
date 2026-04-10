from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
from scipy import stats

from utils.stats.contracts.diagnostics import OutlierFlag, OutlierSummary
from utils.stats.registry.outliers import get_outlier_method_metadata


def detect_outliers(
    *,
    df: pd.DataFrame,
    dv_col: str,
    method: str = "modified_zscore",
    group_col: str = "group",
    subject_col: str | None = "subject",
    handling_mode: str = "include_all",
    alpha: float = 0.05,
    modified_z_threshold: float = 3.5,
    iqr_multiplier: float = 1.5,
) -> OutlierSummary:
    metadata = get_outlier_method_metadata(method)
    if metadata is None:
        return OutlierSummary(
            detected=False,
            method=method,
            label=None,
            flagged_count=0,
            sensitivity_run_available=False,
            handling_mode=handling_mode,
            warnings=[f"Outlier method '{method}' is not recognized."],
        )

    clean_df = df.copy()
    clean_df[dv_col] = pd.to_numeric(clean_df[dv_col], errors="coerce")
    clean_df = clean_df.loc[clean_df[dv_col].notna()].copy()
    if clean_df.empty:
        return OutlierSummary(
            detected=False,
            method=metadata.id,
            label=metadata.label,
            flagged_count=0,
            sensitivity_run_available=False,
            handling_mode=handling_mode,
            warnings=["No numeric observations were available for outlier screening."],
        )

    warnings: list[str] = []
    flags: list[OutlierFlag] = []
    grouping_column = group_col if group_col in clean_df.columns else None
    grouped = clean_df.groupby(grouping_column, sort=False, dropna=False) if grouping_column else [(None, clean_df)]

    for group_name, group_df in grouped:
        values = group_df[dv_col].to_numpy(dtype=float)
        if len(values) < metadata.min_n:
            label = f"group '{group_name}'" if group_name is not None else "the dataset"
            warnings.append(
                f"{metadata.label} was not applied to {label} because at least {metadata.min_n} observations are required."
            )
            continue
        if metadata.id == "grubbs":
            flag = _detect_grubbs(group_df, dv_col, subject_col=subject_col, group_name=group_name, alpha=alpha)
            if flag is not None:
                flags.append(flag)
        elif metadata.id == "modified_zscore":
            flags.extend(
                _detect_modified_zscore(
                    group_df,
                    dv_col,
                    subject_col=subject_col,
                    group_name=group_name,
                    threshold=modified_z_threshold,
                )
            )
        elif metadata.id == "iqr":
            flags.extend(
                _detect_iqr(
                    group_df,
                    dv_col,
                    subject_col=subject_col,
                    group_name=group_name,
                    multiplier=iqr_multiplier,
                )
            )

    return OutlierSummary(
        detected=bool(flags),
        method=metadata.id,
        label=metadata.label,
        flagged_count=len(flags),
        sensitivity_run_available=bool(flags),
        handling_mode=handling_mode,
        flags=flags,
        warnings=warnings,
    )


def filter_flagged_outliers(df: pd.DataFrame, summary: OutlierSummary) -> pd.DataFrame:
    if not summary.flags:
        return df.copy()
    flagged_row_ids = {str(flag.row_id) for flag in summary.flags}
    keep_mask = ~df.index.map(lambda index: str(index)).isin(flagged_row_ids)
    return df.loc[keep_mask].copy()


def update_summary_handling(summary: OutlierSummary, handling_mode: str) -> OutlierSummary:
    return replace(summary, handling_mode=handling_mode, sensitivity_run_available=bool(summary.flags))


def _detect_grubbs(
    group_df: pd.DataFrame,
    dv_col: str,
    *,
    subject_col: str | None,
    group_name: object,
    alpha: float,
) -> OutlierFlag | None:
    values = group_df[dv_col].to_numpy(dtype=float)
    mean_value = float(np.mean(values))
    std_value = float(np.std(values, ddof=1))
    if len(values) < 3 or std_value == 0:
        return None
    distances = np.abs(values - mean_value)
    max_position = int(np.argmax(distances))
    statistic = float(distances[max_position] / std_value)
    threshold = _grubbs_critical_value(len(values), alpha)
    if threshold is None or statistic <= threshold:
        return None
    row = group_df.iloc[max_position]
    return OutlierFlag(
        row_id=str(group_df.index[max_position]),
        subject_id=_subject_id(row, subject_col),
        group=_group_name(group_name),
        value=_safe_float(row[dv_col]),
        method="grubbs",
        statistic=_safe_float(statistic),
        threshold=_safe_float(threshold),
        pvalue=_grubbs_pvalue(statistic, len(values)),
        flag_reason="Extreme value exceeded the Grubbs critical threshold for a single-outlier test.",
    )


def _detect_modified_zscore(
    group_df: pd.DataFrame,
    dv_col: str,
    *,
    subject_col: str | None,
    group_name: object,
    threshold: float,
) -> list[OutlierFlag]:
    values = group_df[dv_col].to_numpy(dtype=float)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    flags: list[OutlierFlag] = []
    if mad == 0:
        for index, row in group_df.iterrows():
            value = float(row[dv_col])
            if value != median:
                flags.append(
                    OutlierFlag(
                        row_id=str(index),
                        subject_id=_subject_id(row, subject_col),
                        group=_group_name(group_name),
                        value=_safe_float(value),
                        method="modified_zscore",
                        statistic=None,
                        threshold=threshold,
                        pvalue=None,
                        flag_reason="Median absolute deviation was zero and the value differed from the group median.",
                    )
                )
        return flags

    scores = 0.6745 * (values - median) / mad
    for position, score in enumerate(scores):
        if abs(score) > threshold:
            row = group_df.iloc[position]
            flags.append(
                OutlierFlag(
                    row_id=str(group_df.index[position]),
                    subject_id=_subject_id(row, subject_col),
                    group=_group_name(group_name),
                    value=_safe_float(row[dv_col]),
                    method="modified_zscore",
                    statistic=_safe_float(score),
                    threshold=threshold,
                    pvalue=None,
                    flag_reason="Absolute modified Z-score exceeded the review threshold.",
                )
            )
    return flags


def _detect_iqr(
    group_df: pd.DataFrame,
    dv_col: str,
    *,
    subject_col: str | None,
    group_name: object,
    multiplier: float,
) -> list[OutlierFlag]:
    values = group_df[dv_col].to_numpy(dtype=float)
    q1 = float(np.quantile(values, 0.25))
    q3 = float(np.quantile(values, 0.75))
    iqr = q3 - q1
    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr
    flags: list[OutlierFlag] = []
    for index, row in group_df.iterrows():
        value = float(row[dv_col])
        if value < lower or value > upper:
            flags.append(
                OutlierFlag(
                    row_id=str(index),
                    subject_id=_subject_id(row, subject_col),
                    group=_group_name(group_name),
                    value=_safe_float(value),
                    method="iqr",
                    statistic=None,
                    threshold=f"[{_safe_float(lower)}, {_safe_float(upper)}]",
                    pvalue=None,
                    flag_reason="Value fell outside the Tukey IQR fences.",
                )
            )
    return flags


def _grubbs_critical_value(n: int, alpha: float) -> float | None:
    if n < 3:
        return None
    t_value = stats.t.ppf(1.0 - alpha / (2.0 * n), n - 2)
    numerator = (n - 1) * np.sqrt(t_value**2)
    denominator = np.sqrt(n) * np.sqrt(n - 2 + t_value**2)
    if denominator == 0:
        return None
    return float(numerator / denominator)


def _grubbs_pvalue(statistic: float, n: int) -> float | None:
    if n < 3:
        return None

    def _critical(alpha: float) -> float:
        return _grubbs_critical_value(n, alpha) or np.inf

    low, high = 1e-12, 0.999999
    if statistic <= _critical(low):
        return 1.0
    for _ in range(60):
        mid = (low + high) / 2.0
        if statistic > _critical(mid):
            high = mid
        else:
            low = mid
    return _safe_float(high)


def _subject_id(row: pd.Series, subject_col: str | None) -> str | None:
    if subject_col and subject_col in row.index and pd.notna(row[subject_col]):
        return str(row[subject_col])
    return None


def _group_name(group_name: object) -> str | None:
    if group_name is None:
        return None
    return str(group_name)


def _safe_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    return float(value)
