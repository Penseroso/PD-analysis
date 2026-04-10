from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
import pingouin as pg
from scipy import stats


def run_dunnett(df: pd.DataFrame, dv_col: str, group_col: str, control_group: str | None) -> dict:
    warnings: list[str] = []
    if control_group is None:
        warnings.append("Control group was not selected. Dunnett post-hoc comparisons were skipped.")
        return _payload(pd.DataFrame(), warnings)

    group_levels = df[group_col].astype(str).unique().tolist()
    if control_group not in group_levels:
        warnings.append("Selected control group was not found in the data. Dunnett post-hoc comparisons were skipped.")
        return _payload(pd.DataFrame(), warnings)

    control_values = df.loc[df[group_col] == control_group, dv_col].to_numpy(dtype=float)
    comparison_groups = [group for group in group_levels if group != control_group]
    if not comparison_groups:
        warnings.append("Only the control group is present. Dunnett post-hoc comparisons were skipped.")
        return _payload(pd.DataFrame(), warnings)

    sample_arrays = [df.loc[df[group_col] == group, dv_col].to_numpy(dtype=float) for group in comparison_groups]
    dunnett_result = stats.dunnett(*sample_arrays, control=control_values)
    ci_result = dunnett_result.confidence_interval()
    rows: list[dict] = []
    for index, group in enumerate(comparison_groups):
        sample_values = sample_arrays[index]
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
                "effect_estimate": _hedges_g(sample_values, control_values),
                "effect_metric": "hedges_g",
                "interpretation_basis": "pairwise_parametric",
            }
        )
    return _payload(pd.DataFrame(rows), warnings)


def run_games_howell(df: pd.DataFrame, dv_col: str, group_col: str) -> dict:
    gh = pg.pairwise_gameshowell(data=df, dv=dv_col, between=group_col)
    posthoc = gh.rename(
        columns={"A": "group_a", "B": "group_b", "T": "statistic", "pval": "pvalue", "hedges": "effect_estimate"}
    ).copy()
    posthoc["comparison"] = posthoc["group_a"].astype(str) + " vs " + posthoc["group_b"].astype(str)
    posthoc["test"] = "games_howell"
    posthoc["p_adjust"] = "games_howell"
    posthoc["ci_low"] = None
    posthoc["ci_high"] = None
    posthoc["effect_metric"] = "hedges_g"
    posthoc["interpretation_basis"] = "pairwise_parametric"
    return _payload(posthoc, [])


def run_pairwise_mannwhitney(
    df: pd.DataFrame,
    dv_col: str,
    group_col: str,
    *,
    multiplicity_method: str | None = "bonferroni",
) -> dict:
    groups = [(str(group_name), group_df[dv_col].to_numpy(dtype=float)) for group_name, group_df in df.groupby(group_col, sort=False)]
    rows: list[dict] = []
    for (group_a, values_a), (group_b, values_b) in combinations(groups, 2):
        statistic, pvalue = stats.mannwhitneyu(values_a, values_b, alternative="two-sided")
        rows.append(
            {
                "group_a": group_a,
                "group_b": group_b,
                "comparison": f"{group_a} vs {group_b}",
                "test": "mannwhitneyu",
                "statistic": _safe_float(statistic),
                "pvalue": _safe_float(pvalue),
                "p_unc": _safe_float(pvalue),
                "p_adjust": multiplicity_method,
                "ci_low": None,
                "ci_high": None,
                "effect_estimate": _rank_biserial_from_u(statistic, len(values_a), len(values_b)),
                "effect_metric": "rank_biserial",
                "interpretation_basis": "pairwise_nonparametric",
            }
        )
    posthoc = pd.DataFrame(rows)
    posthoc = _apply_pairwise_multiplicity(posthoc, multiplicity_method)
    return _payload(posthoc, [])


def _payload(table: pd.DataFrame, warnings: list[str]) -> dict:
    if table.empty:
        table = pd.DataFrame(
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
    pairwise_effects = table[[column for column in ["group_a", "group_b", "effect_estimate", "effect_metric", "interpretation_basis"] if column in table.columns]].copy()
    return {"pairwise_table": table, "warnings": warnings, "metadata": {"effect_sizes": {"pairwise": pairwise_effects}}}


def _apply_pairwise_multiplicity(table: pd.DataFrame, multiplicity_method: str | None) -> pd.DataFrame:
    if table.empty or multiplicity_method in {None, "none"}:
        return table
    adjusted = table.copy()
    if multiplicity_method == "bonferroni" and "p_unc" in adjusted.columns:
        n_tests = max(1, len(adjusted))
        adjusted["pvalue"] = adjusted["p_unc"].apply(lambda value: _safe_float(min(float(value) * n_tests, 1.0)))
        adjusted["p_adjust"] = "bonferroni"
    return adjusted


def _hedges_g(values_a: np.ndarray, values_b: np.ndarray) -> float | None:
    try:
        return _safe_float(pg.compute_effsize(values_a, values_b, paired=False, eftype="hedges"))
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
