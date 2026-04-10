from __future__ import annotations

import pandas as pd
import pingouin as pg
from scipy import stats


def run_one_way_anova(df: pd.DataFrame, dv_col: str, group_col: str) -> dict:
    anova = pg.anova(data=df, dv=dv_col, between=group_col, detailed=True)
    between_row = anova.loc[anova["Source"] != "Within"].iloc[0]
    within_row = anova.loc[anova["Source"] == "Within"].iloc[0]
    omega_sq = _omega_squared(
        ss_effect=between_row.get("SS"),
        df_effect=between_row.get("DF"),
        ms_error=within_row.get("MS"),
        ss_total=float(anova["SS"].sum(skipna=True)),
    )
    omnibus = pd.DataFrame(
        [
            {
                "term": str(between_row["Source"]),
                "test": "one_way_anova",
                "statistic": _safe_float(between_row.get("F")),
                "pvalue": _safe_float(between_row.get("p_unc")),
                "df1": _safe_float(between_row.get("DF")),
                "df2": _safe_float(within_row.get("DF")),
                "ss": _safe_float(between_row.get("SS")),
                "ms": _safe_float(between_row.get("MS")),
                "effect_estimate": omega_sq,
                "effect_metric": "omega_squared",
                "interpretation_basis": "one_way_anova",
            }
        ]
    )
    effect_sizes = omnibus[["term", "effect_estimate", "effect_metric", "interpretation_basis"]].copy()
    return {"omnibus_table": omnibus, "warnings": [], "metadata": {"effect_sizes": {"omnibus": effect_sizes}}}


def run_welch_anova(df: pd.DataFrame, dv_col: str, group_col: str) -> dict:
    warnings = ["Omega-squared is not reported for Welch ANOVA because a robust estimate is not implemented with the current dependency set."]
    welch = pg.welch_anova(data=df, dv=dv_col, between=group_col).iloc[0]
    omnibus = pd.DataFrame(
        [
            {
                "term": group_col,
                "test": "welch_anova",
                "statistic": _safe_float(welch.get("F")),
                "pvalue": _safe_float(welch.get("p_unc")),
                "df1": _safe_float(welch.get("ddof1")),
                "df2": _safe_float(welch.get("ddof2")),
                "ss": None,
                "ms": None,
                "effect_estimate": None,
                "effect_metric": "omega_squared",
                "interpretation_basis": "welch_anova",
            }
        ]
    )
    effect_sizes = omnibus[["term", "effect_estimate", "effect_metric", "interpretation_basis"]].copy()
    return {"omnibus_table": omnibus, "warnings": warnings, "metadata": {"effect_sizes": {"omnibus": effect_sizes}}}


def run_kruskal(df: pd.DataFrame, dv_col: str, group_col: str) -> dict:
    grouped_values = [group_df[dv_col].to_numpy(dtype=float) for _, group_df in df.groupby(group_col, sort=False)]
    statistic, pvalue = stats.kruskal(*grouped_values)
    n_total = len(df)
    n_groups = int(df[group_col].nunique(dropna=True))
    epsilon_sq = _safe_float((statistic - n_groups + 1.0) / (n_total - n_groups)) if n_total > n_groups else None
    omnibus = pd.DataFrame(
        [
            {
                "term": group_col,
                "test": "kruskal",
                "statistic": _safe_float(statistic),
                "pvalue": _safe_float(pvalue),
                "df1": float(n_groups - 1),
                "df2": None,
                "ss": None,
                "ms": None,
                "effect_estimate": epsilon_sq,
                "effect_metric": "epsilon_squared",
                "interpretation_basis": "kruskal",
            }
        ]
    )
    effect_sizes = omnibus[["term", "effect_estimate", "effect_metric", "interpretation_basis"]].copy()
    return {"omnibus_table": omnibus, "warnings": [], "metadata": {"effect_sizes": {"omnibus": effect_sizes}}}


def _omega_squared(ss_effect: float | None, df_effect: float | None, ms_error: float | None, ss_total: float | None) -> float | None:
    if any(value is None or pd.isna(value) for value in (ss_effect, df_effect, ms_error, ss_total)):
        return None
    estimate = (float(ss_effect) - float(df_effect) * float(ms_error)) / (float(ss_total) + float(ms_error))
    return _safe_float(max(estimate, 0.0))


def _safe_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    return float(value)
