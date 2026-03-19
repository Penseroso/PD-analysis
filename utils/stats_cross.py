from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
import pingouin as pg
from scipy import stats


PVALUE_COLUMNS = ("p_corr", "pvalue", "p_unc", "pval")


def compute_cross_assumptions(
    df: pd.DataFrame,
    dv_col: str,
    group_col: str,
) -> dict:
    clean_df = _prepare_cross_df(df, dv_col, group_col)
    normality: dict = {}
    for group_name, group_df in clean_df.groupby(group_col, sort=False):
        values = group_df[dv_col].to_numpy(dtype=float)
        if len(values) >= 3:
            stat, pvalue = stats.shapiro(values)
            normality[str(group_name)] = {
                "stat": float(stat),
                "pvalue": float(pvalue),
                "is_normal": bool(pvalue >= 0.05),
                "n": int(len(values)),
            }
        else:
            normality[str(group_name)] = {
                "stat": None,
                "pvalue": None,
                "is_normal": True,
                "n": int(len(values)),
            }

    grouped_values = [group_df[dv_col].to_numpy(dtype=float) for _, group_df in clean_df.groupby(group_col, sort=False)]
    if len(grouped_values) >= 2 and all(len(values) >= 2 for values in grouped_values):
        stat, pvalue = stats.levene(*grouped_values)
        levene = {"stat": float(stat), "pvalue": float(pvalue), "equal_variance": bool(pvalue >= 0.05)}
    else:
        levene = {"stat": None, "pvalue": None, "equal_variance": True}

    return {"normality": normality, "levene": levene}


def run_cross_sectional(
    df: pd.DataFrame,
    dv_col: str,
    group_col: str,
    control_group: str,
    method: str,
) -> dict:
    clean_df = _prepare_cross_df(df, dv_col, group_col)
    assumptions = compute_cross_assumptions(clean_df, dv_col, group_col)
    warnings: list[str] = []
    blocking_reasons: list[str] = []
    suggested_actions: list[str] = []

    if clean_df.empty:
        blocking_reasons.append("No valid numeric observations are available for the selected biomarker.")
        suggested_actions.append("Check the selected biomarker column and remove non-numeric values.")
    elif clean_df[group_col].nunique(dropna=True) < 2:
        blocking_reasons.append("At least two groups are required.")
        suggested_actions.append("Verify the group mapping or choose a different design.")

    if blocking_reasons:
        return {
            "analysis_status": "blocked",
            "assumptions": assumptions,
            "omnibus": None,
            "posthoc_table": None,
            "star_map": [],
            "effect_sizes": {"omnibus": None, "pairwise": None},
            "used_method": method,
            "warnings": warnings,
            "blocking_reasons": blocking_reasons,
            "suggested_actions": suggested_actions,
            "dv_col": dv_col,
        }

    resolved_method = _resolve_cross_method(method, assumptions)
    if method == "auto":
        warnings.append(f"Auto-selected method: {resolved_method}.")

    if resolved_method == "one_way_anova":
        omnibus, omnibus_effects = _run_one_way_anova(clean_df, dv_col, group_col)
        posthoc_table, pairwise_effects, extra_warnings = _run_dunnett_posthoc(clean_df, dv_col, group_col, control_group)
        warnings.extend(extra_warnings)
    elif resolved_method == "welch_anova":
        omnibus, omnibus_effects, extra_warnings = _run_welch_anova(clean_df, dv_col, group_col)
        posthoc_table, pairwise_effects = _run_games_howell(clean_df, dv_col, group_col)
        warnings.extend(extra_warnings)
    elif resolved_method == "kruskal":
        omnibus, omnibus_effects = _run_kruskal(clean_df, dv_col, group_col)
        posthoc_table, pairwise_effects = _run_pairwise_mannwhitney(clean_df, dv_col, group_col)
    else:
        warnings.append(f"Unsupported method '{resolved_method}' requested. Falling back to auto resolution.")
        fallback_method = _resolve_cross_method("auto", assumptions)
        return run_cross_sectional(clean_df, dv_col, group_col, control_group, fallback_method)

    return {
        "analysis_status": "ready",
        "assumptions": assumptions,
        "omnibus": omnibus,
        "posthoc_table": posthoc_table,
        "star_map": _build_star_map(posthoc_table),
        "effect_sizes": {
            "omnibus": omnibus_effects,
            "pairwise": pairwise_effects,
        },
        "used_method": resolved_method,
        "warnings": sorted(set(warnings)),
        "blocking_reasons": blocking_reasons,
        "suggested_actions": suggested_actions,
        "dv_col": dv_col,
    }


def _prepare_cross_df(df: pd.DataFrame, dv_col: str, group_col: str) -> pd.DataFrame:
    clean_df = df[[group_col, dv_col]].copy()
    clean_df[dv_col] = pd.to_numeric(clean_df[dv_col], errors="coerce")
    clean_df[group_col] = clean_df[group_col].astype(str)
    return clean_df.dropna(subset=[group_col, dv_col])


def _resolve_cross_method(method: str, assumptions: dict) -> str:
    if method != "auto":
        return method
    any_non_normal = any(not item.get("is_normal", True) for item in assumptions.get("normality", {}).values())
    equal_variance = assumptions.get("levene", {}).get("equal_variance", True)
    if any_non_normal:
        return "kruskal"
    if not equal_variance:
        return "welch_anova"
    return "one_way_anova"


def _run_one_way_anova(df: pd.DataFrame, dv_col: str, group_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    return omnibus, effect_sizes


def _run_dunnett_posthoc(
    df: pd.DataFrame,
    dv_col: str,
    group_col: str,
    control_group: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    warnings: list[str] = []
    columns = [
        "group_a",
        "group_b",
        "comparison",
        "test",
        "statistic",
        "pvalue",
        "p_adjust",
        "ci_low",
        "ci_high",
        "effect_estimate",
        "effect_metric",
        "interpretation_basis",
    ]
    if control_group is None:
        warnings.append("Control group was not selected. Dunnett post-hoc comparisons were skipped.")
        empty = pd.DataFrame(columns=columns)
        return empty, empty[["group_a", "group_b", "effect_estimate", "effect_metric", "interpretation_basis"]].copy(), warnings

    group_levels = df[group_col].astype(str).unique().tolist()
    if control_group not in group_levels:
        warnings.append("Selected control group was not found in the data. Dunnett post-hoc comparisons were skipped.")
        empty = pd.DataFrame(columns=columns)
        return empty, empty[["group_a", "group_b", "effect_estimate", "effect_metric", "interpretation_basis"]].copy(), warnings

    control_values = df.loc[df[group_col] == control_group, dv_col].to_numpy(dtype=float)
    comparison_groups = [group for group in group_levels if group != control_group]
    if not comparison_groups:
        warnings.append("Only the control group is present. Dunnett post-hoc comparisons were skipped.")
        empty = pd.DataFrame(columns=columns)
        return empty, empty[["group_a", "group_b", "effect_estimate", "effect_metric", "interpretation_basis"]].copy(), warnings

    sample_arrays = [df.loc[df[group_col] == group, dv_col].to_numpy(dtype=float) for group in comparison_groups]
    dunnett_result = stats.dunnett(*sample_arrays, control=control_values)
    ci_result = dunnett_result.confidence_interval()
    rows: list[dict] = []
    for index, group in enumerate(comparison_groups):
        sample_values = sample_arrays[index]
        hedges_g = _hedges_g(sample_values, control_values, paired=False)
        rows.append(
            {
                "group_a": control_group,
                "group_b": group,
                "comparison": f"{control_group} vs {group}",
                "test": "dunnett",
                "statistic": _safe_float(np.atleast_1d(dunnett_result.statistic)[index]),
                "pvalue": _safe_float(np.atleast_1d(dunnett_result.pvalue)[index]),
                "p_adjust": "dunnett",
                "ci_low": _safe_float(np.atleast_1d(ci_result.low)[index]),
                "ci_high": _safe_float(np.atleast_1d(ci_result.high)[index]),
                "effect_estimate": hedges_g,
                "effect_metric": "hedges_g",
                "interpretation_basis": "pairwise_parametric",
            }
        )

    posthoc = pd.DataFrame(rows)
    pairwise_effects = posthoc[["group_a", "group_b", "effect_estimate", "effect_metric", "interpretation_basis"]].copy()
    return posthoc, pairwise_effects, warnings


def _run_welch_anova(df: pd.DataFrame, dv_col: str, group_col: str) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
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
    return omnibus, effect_sizes, warnings


def _run_games_howell(df: pd.DataFrame, dv_col: str, group_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    gh = pg.pairwise_gameshowell(data=df, dv=dv_col, between=group_col)
    posthoc = gh.rename(
        columns={
            "A": "group_a",
            "B": "group_b",
            "T": "statistic",
            "pval": "pvalue",
            "hedges": "effect_estimate",
        }
    ).copy()
    posthoc["comparison"] = posthoc["group_a"].astype(str) + " vs " + posthoc["group_b"].astype(str)
    posthoc["test"] = "games_howell"
    posthoc["p_adjust"] = "games_howell"
    posthoc["ci_low"] = None
    posthoc["ci_high"] = None
    posthoc["effect_metric"] = "hedges_g"
    posthoc["interpretation_basis"] = "pairwise_parametric"
    ordered_columns = [
        "group_a",
        "group_b",
        "comparison",
        "test",
        "statistic",
        "pvalue",
        "p_adjust",
        "ci_low",
        "ci_high",
        "effect_estimate",
        "effect_metric",
        "interpretation_basis",
    ]
    posthoc = posthoc[ordered_columns]
    pairwise_effects = posthoc[["group_a", "group_b", "effect_estimate", "effect_metric", "interpretation_basis"]].copy()
    return posthoc, pairwise_effects


def _run_kruskal(df: pd.DataFrame, dv_col: str, group_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    return omnibus, effect_sizes


def _run_pairwise_mannwhitney(df: pd.DataFrame, dv_col: str, group_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    groups = [(str(group_name), group_df[dv_col].to_numpy(dtype=float)) for group_name, group_df in df.groupby(group_col, sort=False)]
    rows: list[dict] = []
    n_tests = max(1, len(groups) * (len(groups) - 1) // 2)
    for (group_a, values_a), (group_b, values_b) in combinations(groups, 2):
        statistic, pvalue = stats.mannwhitneyu(values_a, values_b, alternative="two-sided")
        corrected = min(pvalue * n_tests, 1.0)
        rows.append(
            {
                "group_a": group_a,
                "group_b": group_b,
                "comparison": f"{group_a} vs {group_b}",
                "test": "mannwhitneyu",
                "statistic": _safe_float(statistic),
                "pvalue": _safe_float(corrected),
                "p_unc": _safe_float(pvalue),
                "p_adjust": "bonferroni",
                "ci_low": None,
                "ci_high": None,
                "effect_estimate": _rank_biserial_from_u(statistic, len(values_a), len(values_b)),
                "effect_metric": "rank_biserial",
                "interpretation_basis": "pairwise_nonparametric_bonferroni",
            }
        )

    posthoc = pd.DataFrame(rows)
    if posthoc.empty:
        posthoc = pd.DataFrame(
            columns=[
                "group_a",
                "group_b",
                "comparison",
                "test",
                "statistic",
                "pvalue",
                "p_unc",
                "p_adjust",
                "ci_low",
                "ci_high",
                "effect_estimate",
                "effect_metric",
                "interpretation_basis",
            ]
        )
    pairwise_effects = posthoc[["group_a", "group_b", "effect_estimate", "effect_metric", "interpretation_basis"]].copy()
    return posthoc, pairwise_effects


def _omega_squared(ss_effect: float | None, df_effect: float | None, ms_error: float | None, ss_total: float | None) -> float | None:
    if any(value is None or pd.isna(value) for value in (ss_effect, df_effect, ms_error, ss_total)):
        return None
    estimate = (float(ss_effect) - float(df_effect) * float(ms_error)) / (float(ss_total) + float(ms_error))
    return _safe_float(max(estimate, 0.0))


def _hedges_g(values_a: np.ndarray, values_b: np.ndarray, paired: bool) -> float | None:
    try:
        return _safe_float(pg.compute_effsize(values_a, values_b, paired=paired, eftype="hedges"))
    except Exception:
        return None


def _rank_biserial_from_u(statistic: float, n_a: int, n_b: int) -> float | None:
    if n_a <= 0 or n_b <= 0:
        return None
    return _safe_float((2.0 * float(statistic) / float(n_a * n_b)) - 1.0)


def _safe_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    return float(value)


def _build_star_map(posthoc_table: pd.DataFrame | None) -> list[dict]:
    if posthoc_table is None or posthoc_table.empty:
        return []

    pvalue_col = next((column for column in PVALUE_COLUMNS if column in posthoc_table.columns), None)
    if pvalue_col is None:
        return []

    star_map: list[dict] = []
    for row in posthoc_table.itertuples(index=False):
        pvalue = getattr(row, pvalue_col, None)
        if pvalue is None or pd.isna(pvalue):
            continue
        label = _pvalue_to_label(float(pvalue))
        if label is None:
            continue
        record = {
            "comparison": getattr(row, "comparison", f"{getattr(row, 'group_a', '')} vs {getattr(row, 'group_b', '')}"),
            "group_a": getattr(row, "group_a", None),
            "group_b": getattr(row, "group_b", None),
            "pvalue": float(pvalue),
            "label": label,
        }
        if hasattr(row, "time"):
            record["time"] = getattr(row, "time")
        if hasattr(row, "factor2"):
            record["factor2"] = getattr(row, "factor2")
        star_map.append(record)
    return star_map


def _pvalue_to_label(pvalue: float) -> str | None:
    if pvalue < 0.001:
        return "***"
    if pvalue < 0.01:
        return "**"
    if pvalue < 0.05:
        return "*"
    return None
