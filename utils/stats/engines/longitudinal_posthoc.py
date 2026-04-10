from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
import pingouin as pg
from scipy import stats


def run_pairwise_time_tests(
    df: pd.DataFrame,
    dv_col: str,
    subject_col: str,
    time_col: str,
    *,
    interpretation_basis: str = "pairwise_parametric",
    multiplicity_method: str | None = "bonferroni",
) -> dict:
    pairwise = pg.pairwise_tests(
        data=df,
        dv=dv_col,
        within=time_col,
        subject=subject_col,
        parametric=True,
        padjust=_resolve_pingouin_padjust(multiplicity_method),
        effsize="hedges",
    )
    posthoc = _format_pairwise_time_tests(pairwise, interpretation_basis=interpretation_basis)
    posthoc["p_adjust"] = multiplicity_method
    return _payload(posthoc, [])


def run_pairwise_wilcoxon(
    df: pd.DataFrame,
    dv_col: str,
    subject_col: str,
    time_col: str,
    *,
    multiplicity_method: str | None = "bonferroni",
) -> dict:
    wide = df.pivot_table(index=subject_col, columns=time_col, values=dv_col, aggfunc="mean")
    wide = wide.dropna(axis=0, how="any")
    rows: list[dict] = []
    for time_a, time_b in combinations(wide.columns.tolist(), 2):
        paired = wide[[time_a, time_b]].dropna()
        if paired.empty:
            continue
        diff = paired[time_a] - paired[time_b]
        wilcoxon = stats.wilcoxon(paired[time_a], paired[time_b], zero_method="wilcox", alternative="two-sided")
        rows.append(
            {
                "annotation_type": "longitudinal_time_pair_within_group",
                "time_a": str(time_a),
                "time_b": str(time_b),
                "comparison": f"{time_a} vs {time_b}",
                "test": "wilcoxon",
                "statistic": _safe_float(wilcoxon.statistic),
                "pvalue": _safe_float(wilcoxon.pvalue),
                "p_unc": _safe_float(wilcoxon.pvalue),
                "p_adjust": multiplicity_method,
                "effect_estimate": _paired_rank_biserial(diff.to_numpy(dtype=float)),
                "effect_metric": "rank_biserial",
                "interpretation_basis": "pairwise_nonparametric",
            }
        )
    posthoc = _apply_pairwise_multiplicity(pd.DataFrame(rows), multiplicity_method)
    return _payload(posthoc, [])


def run_pairwise_group_at_time_tests(
    df: pd.DataFrame,
    dv_col: str,
    subject_col: str,
    time_col: str,
    group_col: str,
    *,
    multiplicity_method: str | None = "bonferroni",
) -> dict:
    pairwise = pg.pairwise_tests(
        data=df,
        dv=dv_col,
        within=time_col,
        between=group_col,
        subject=subject_col,
        parametric=True,
        padjust=_resolve_pingouin_padjust(multiplicity_method),
        effsize="hedges",
        interaction=True,
    )
    posthoc = _format_pairwise_mixed_tests(pairwise, time_col, group_col)
    posthoc["p_adjust"] = multiplicity_method
    return _payload(posthoc, [])


def _payload(table: pd.DataFrame, warnings: list[str]) -> dict:
    if table.empty:
        table = pd.DataFrame(
            columns=[
                "annotation_type",
                "Contrast",
                "time",
                "group",
                "group_a",
                "group_b",
                "time_a",
                "time_b",
                "comparison",
                "test",
                "statistic",
                "pvalue",
                "p_unc",
                "p_adjust",
                "effect_estimate",
                "effect_metric",
                "interpretation_basis",
            ]
        )
    pairwise_effects = table[[column for column in ["comparison", "time_a", "time_b", "effect_estimate", "effect_metric", "interpretation_basis"] if column in table.columns]].copy()
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


def _resolve_pingouin_padjust(multiplicity_method: str | None) -> str:
    if multiplicity_method in {None, "none"}:
        return "none"
    if multiplicity_method == "bonferroni":
        return "bonf"
    return "none"


def _format_pairwise_time_tests(pairwise: pd.DataFrame, interpretation_basis: str) -> pd.DataFrame:
    posthoc = pairwise.rename(
        columns={"A": "time_a", "B": "time_b", "T": "statistic", "p_corr": "pvalue", "hedges": "effect_estimate"}
    ).copy()
    if "pvalue" not in posthoc.columns:
        posthoc["pvalue"] = posthoc.get("p_unc")
    posthoc["annotation_type"] = "longitudinal_time_pair_within_group"
    posthoc["comparison"] = posthoc["time_a"].astype(str) + " vs " + posthoc["time_b"].astype(str)
    posthoc["test"] = "pairwise_ttests"
    posthoc["effect_metric"] = "hedges_g"
    posthoc["interpretation_basis"] = interpretation_basis
    return posthoc


def _format_pairwise_mixed_tests(pairwise: pd.DataFrame, time_col: str, group_col: str) -> pd.DataFrame:
    posthoc = pairwise.rename(columns={"T": "statistic", "p_corr": "pvalue", "hedges": "effect_estimate"}).copy()
    if "pvalue" not in posthoc.columns:
        posthoc["pvalue"] = posthoc.get("p_unc")
    rows: list[dict] = []
    for _, row in posthoc.iterrows():
        contrast = str(row.get("Contrast", ""))
        record = {
            "Contrast": contrast,
            "test": "pairwise_tests",
            "statistic": _safe_float(row.get("statistic")),
            "pvalue": _safe_float(row.get("pvalue")),
            "p_unc": _safe_float(row.get("p_unc")),
            "p_adjust": row.get("p_adjust"),
            "effect_estimate": _safe_float(row.get("effect_estimate")),
            "effect_metric": "hedges_g",
            "interpretation_basis": "pairwise_parametric_interaction" if "*" in contrast else "pairwise_parametric",
        }
        lower_contrast = contrast.lower()
        if "*" in lower_contrast and pd.notna(row.get(time_col)):
            record.update(
                {
                    "annotation_type": "longitudinal_group_pair_at_time",
                    "time": str(row.get(time_col)),
                    "group_a": str(row.get("A")),
                    "group_b": str(row.get("B")),
                    "comparison": f"{row.get(time_col)}: {row.get('A')} vs {row.get('B')}",
                }
            )
        elif lower_contrast == time_col.lower():
            record.update(
                {
                    "annotation_type": "longitudinal_time_pair_within_group",
                    "time_a": str(row.get("A")),
                    "time_b": str(row.get("B")),
                    "comparison": f"{row.get('A')} vs {row.get('B')}",
                }
            )
            if group_col in row.index and pd.notna(row.get(group_col)):
                record["group"] = str(row.get(group_col))
        elif lower_contrast == group_col.lower():
            record.update(
                {
                    "annotation_type": "cross_group_pair",
                    "group_a": str(row.get("A")),
                    "group_b": str(row.get("B")),
                    "comparison": f"{row.get('A')} vs {row.get('B')}",
                }
            )
            if time_col in row.index and pd.notna(row.get(time_col)):
                record["time"] = str(row.get(time_col))
        else:
            record.update({"annotation_type": "cross_group_pair", "comparison": f"{row.get('A')} vs {row.get('B')}"})
        rows.append(record)
    return pd.DataFrame(rows)


def _paired_rank_biserial(diff: np.ndarray) -> float | None:
    diff = diff[np.isfinite(diff)]
    diff = diff[diff != 0]
    if diff.size == 0:
        return None
    ranks = stats.rankdata(np.abs(diff))
    pos = float(ranks[diff > 0].sum())
    neg = float(ranks[diff < 0].sum())
    denom = pos + neg
    if denom == 0:
        return None
    return _safe_float((pos - neg) / denom)


def _safe_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    return float(value)
