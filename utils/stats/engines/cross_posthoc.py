from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
import pingouin as pg
from scipy import stats
from scipy.stats import rankdata
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from utils.stats.engines.multiplicity import apply_multiplicity_to_pairwise_table, build_correction_metadata


def run_dunnett(df: pd.DataFrame, dv_col: str, group_col: str, control_group: str | None) -> dict:
    warnings: list[str] = []
    if control_group is None:
        warnings.append("Control group was not selected. Dunnett post-hoc comparisons were skipped.")
        return _payload(pd.DataFrame(), warnings, correction_method=None, correction_source="internal")

    group_levels = df[group_col].astype(str).unique().tolist()
    if control_group not in group_levels:
        warnings.append("Selected control group was not found in the data. Dunnett post-hoc comparisons were skipped.")
        return _payload(pd.DataFrame(), warnings, correction_method=None, correction_source="internal")

    control_values = df.loc[df[group_col] == control_group, dv_col].to_numpy(dtype=float)
    comparison_groups = [group for group in group_levels if group != control_group]
    if not comparison_groups:
        warnings.append("Only the control group is present. Dunnett post-hoc comparisons were skipped.")
        return _payload(pd.DataFrame(), warnings, correction_method=None, correction_source="internal")

    sample_arrays = [df.loc[df[group_col] == group, dv_col].to_numpy(dtype=float) for group in comparison_groups]
    dunnett_result = stats.dunnett(*sample_arrays, control=control_values)
    ci_result = dunnett_result.confidence_interval()
    rows: list[dict] = []
    for index, group in enumerate(comparison_groups):
        sample_values = sample_arrays[index]
        adjusted_p = _safe_float(np.atleast_1d(dunnett_result.pvalue)[index])
        rows.append(
            {
                "group_a": control_group,
                "group_b": group,
                "comparison": f"{control_group} vs {group}",
                "test": "dunnett",
                "statistic": _safe_float(np.atleast_1d(dunnett_result.statistic)[index]),
                "pvalue": adjusted_p,
                "p_unc": adjusted_p,
                "p_adjust": "dunnett",
                "ci_low": _safe_float(np.atleast_1d(ci_result.low)[index]),
                "ci_high": _safe_float(np.atleast_1d(ci_result.high)[index]),
                "effect_estimate": _hedges_g(sample_values, control_values),
                "effect_metric": "hedges_g",
                "interpretation_basis": "pairwise_parametric",
            }
        )
    return _payload(pd.DataFrame(rows), warnings, correction_method=None, correction_source="internal")


def run_games_howell(df: pd.DataFrame, dv_col: str, group_col: str) -> dict:
    gh = pg.pairwise_gameshowell(data=df, dv=dv_col, between=group_col)
    posthoc = gh.rename(
        columns={"A": "group_a", "B": "group_b", "T": "statistic", "pval": "pvalue", "hedges": "effect_estimate"}
    ).copy()
    posthoc["comparison"] = posthoc["group_a"].astype(str) + " vs " + posthoc["group_b"].astype(str)
    posthoc["test"] = "games_howell"
    posthoc["p_unc"] = posthoc["pvalue"]
    posthoc["p_adjust"] = "games_howell"
    posthoc["ci_low"] = None
    posthoc["ci_high"] = None
    posthoc["effect_metric"] = "hedges_g"
    posthoc["interpretation_basis"] = "pairwise_parametric"
    return _payload(posthoc, [], correction_method=None, correction_source="internal")


def run_tukey_hsd(df: pd.DataFrame, dv_col: str, group_col: str) -> dict:
    result = pairwise_tukeyhsd(endog=df[dv_col].to_numpy(dtype=float), groups=df[group_col].astype(str).to_numpy())
    summary_rows = result.summary().data[1:]
    rows: list[dict] = []
    for group_a, group_b, mean_diff, p_adj, ci_low, ci_high, reject in summary_rows:
        values_a = df.loc[df[group_col].astype(str) == str(group_a), dv_col].to_numpy(dtype=float)
        values_b = df.loc[df[group_col].astype(str) == str(group_b), dv_col].to_numpy(dtype=float)
        adjusted_p = _safe_float(p_adj)
        rows.append(
            {
                "group_a": str(group_a),
                "group_b": str(group_b),
                "comparison": f"{group_a} vs {group_b}",
                "test": "tukey_hsd",
                "statistic": None,
                "pvalue": adjusted_p,
                "p_unc": adjusted_p,
                "p_adjust": "tukey_hsd",
                "ci_low": _safe_float(ci_low),
                "ci_high": _safe_float(ci_high),
                "mean_difference": _safe_float(mean_diff),
                "reject_null": bool(reject),
                "effect_estimate": _hedges_g(values_a, values_b),
                "effect_metric": "hedges_g",
                "interpretation_basis": "pairwise_parametric",
            }
        )
    return _payload(pd.DataFrame(rows), [], correction_method=None, correction_source="internal")


def run_dunn(
    df: pd.DataFrame,
    dv_col: str,
    group_col: str,
    *,
    multiplicity_method: str | None = "bonferroni",
) -> dict:
    groups = [(str(group_name), group_df[dv_col].to_numpy(dtype=float)) for group_name, group_df in df.groupby(group_col, sort=False)]
    all_values = df[dv_col].to_numpy(dtype=float)
    all_groups = df[group_col].astype(str).to_numpy()
    ranks = rankdata(all_values)
    n_total = float(len(all_values))
    tie_correction = _dunn_tie_correction(all_values)
    base_variance = (n_total * (n_total + 1.0) / 12.0) * tie_correction if n_total > 0 else np.nan
    rows: list[dict] = []
    for (group_a, values_a), (group_b, values_b) in combinations(groups, 2):
        rank_a = ranks[all_groups == group_a]
        rank_b = ranks[all_groups == group_b]
        n_a = len(rank_a)
        n_b = len(rank_b)
        if n_a == 0 or n_b == 0 or base_variance <= 0:
            continue
        z_value = (rank_a.mean() - rank_b.mean()) / np.sqrt(base_variance * ((1.0 / n_a) + (1.0 / n_b)))
        p_unc = _safe_float(2.0 * stats.norm.sf(abs(z_value)))
        u_statistic, _ = stats.mannwhitneyu(values_a, values_b, alternative="two-sided")
        rows.append(
            {
                "group_a": group_a,
                "group_b": group_b,
                "comparison": f"{group_a} vs {group_b}",
                "test": "dunn",
                "statistic": _safe_float(z_value),
                "pvalue": p_unc,
                "p_unc": p_unc,
                "p_adjust": multiplicity_method,
                "ci_low": None,
                "ci_high": None,
                "effect_estimate": _rank_biserial_from_u(u_statistic, len(values_a), len(values_b)),
                "effect_metric": "rank_biserial",
                "interpretation_basis": "pairwise_nonparametric",
            }
        )
    posthoc = apply_multiplicity_to_pairwise_table(pd.DataFrame(rows), multiplicity_method)
    return _payload(posthoc, [], correction_method=multiplicity_method, correction_source="external")


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
    posthoc = apply_multiplicity_to_pairwise_table(pd.DataFrame(rows), multiplicity_method)
    return _payload(posthoc, [], correction_method=multiplicity_method, correction_source="external")


def run_group_pairwise_by_factor(
    df: pd.DataFrame,
    dv_col: str,
    group_col: str,
    factor2_col: str,
    *,
    multiplicity_method: str | None = "bonferroni",
) -> dict:
    rows: list[dict] = []
    warnings: list[str] = []
    for factor_level, subset in df.groupby(factor2_col, sort=False):
        if subset[group_col].nunique(dropna=True) < 2:
            continue
        pairwise = pg.pairwise_tests(
            data=subset,
            dv=dv_col,
            between=group_col,
            parametric=True,
            padjust="none",
            effsize="hedges",
        )
        formatted = pairwise.rename(
            columns={"A": "group_a", "B": "group_b", "T": "statistic", "hedges": "effect_estimate"}
        ).copy()
        formatted["comparison"] = formatted["group_a"].astype(str) + " vs " + formatted["group_b"].astype(str) + f" | {factor2_col}=" + str(factor_level)
        formatted["test"] = "group_pairwise_by_factor"
        formatted["pvalue"] = formatted.get("p_unc")
        formatted["p_adjust"] = multiplicity_method
        formatted["ci_low"] = None
        formatted["ci_high"] = None
        formatted["effect_metric"] = "hedges_g"
        formatted["interpretation_basis"] = "pairwise_parametric"
        formatted["annotation_type"] = "cross_group_pair"
        formatted["group"] = str(factor_level)
        formatted["factor2_level"] = str(factor_level)
        rows.extend(formatted.to_dict(orient="records"))
    if not rows:
        warnings.append("Two-way ANOVA post-hoc comparisons were skipped because no factor2 level contained at least two groups.")
        return _payload(pd.DataFrame(), warnings, correction_method=multiplicity_method, correction_source="external")
    posthoc = apply_multiplicity_to_pairwise_table(pd.DataFrame(rows), multiplicity_method)
    return _payload(posthoc, warnings, correction_method=multiplicity_method, correction_source="external")


def _payload(table: pd.DataFrame, warnings: list[str], *, correction_method: str | None, correction_source: str | None) -> dict:
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
    pairwise_effects = table[
        [column for column in ["group_a", "group_b", "effect_estimate", "effect_metric", "interpretation_basis"] if column in table.columns]
    ].copy()
    metadata = {
        "effect_sizes": {"pairwise": pairwise_effects},
        "correction_applied": build_correction_metadata(multiplicity_method=correction_method, source=correction_source or "external"),
    }
    return {"pairwise_table": table, "warnings": warnings, "metadata": metadata}


def _dunn_tie_correction(values: np.ndarray) -> float:
    _, counts = np.unique(values, return_counts=True)
    if values.size < 2:
        return 1.0
    tie_sum = float(np.sum(counts**3 - counts))
    denom = float(values.size**3 - values.size)
    if denom == 0:
        return 1.0
    return max(0.0, 1.0 - (tie_sum / denom))


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
