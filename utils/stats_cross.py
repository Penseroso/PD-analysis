from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def compute_cross_assumptions(
    df: pd.DataFrame,
    dv_col: str,
    group_col: str,
) -> dict:
    normality: dict = {}
    for group_name, group_df in df.groupby(group_col):
        values = pd.to_numeric(group_df[dv_col], errors="coerce").dropna()
        if len(values) >= 3:
            stat, p = stats.shapiro(values)
            normality[str(group_name)] = {"stat": float(stat), "pvalue": float(p), "is_normal": bool(p >= 0.05)}
        else:
            normality[str(group_name)] = {"stat": None, "pvalue": None, "is_normal": True}

    grouped_values = [pd.to_numeric(group_df[dv_col], errors="coerce").dropna().values for _, group_df in df.groupby(group_col)]
    if len(grouped_values) >= 2 and all(len(values) >= 2 for values in grouped_values):
        stat, p = stats.levene(*grouped_values)
        levene = {"stat": float(stat), "pvalue": float(p), "equal_variance": bool(p >= 0.05)}
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
    assumptions = compute_cross_assumptions(df, dv_col, group_col)
    omnibus = _build_placeholder_omnibus(df=df, dv_col=dv_col, group_col=group_col, method=method)
    posthoc_table = _build_placeholder_posthoc(df=df, group_col=group_col, control_group=control_group)
    effect_sizes = {
        "omnibus": omnibus[["term", "effect_estimate", "effect_metric", "interpretation_basis"]].copy(),
        "pairwise": posthoc_table[["group_a", "group_b", "effect_estimate", "effect_metric", "interpretation_basis"]].copy()
        if not posthoc_table.empty
        else pd.DataFrame(columns=["group_a", "group_b", "effect_estimate", "effect_metric", "interpretation_basis"]),
    }

    if df[group_col].nunique(dropna=True) < 2:
        return {
            "analysis_status": "blocked",
            "assumptions": assumptions,
            "omnibus": None,
            "posthoc_table": None,
            "star_map": [],
            "effect_sizes": {"omnibus": None, "pairwise": None},
            "used_method": method,
            "warnings": [],
            "blocking_reasons": ["At least two groups are required."],
            "suggested_actions": ["Verify the group mapping or choose a different design."],
        }

    warnings = []
    if control_group is None:
        warnings.append("Control group was not selected. Control-based post-hoc output is omitted.")

    return {
        "analysis_status": "ready",
        "assumptions": assumptions,
        "omnibus": omnibus,
        "posthoc_table": posthoc_table,
        "star_map": _build_star_map(posthoc_table),
        "effect_sizes": effect_sizes,
        "used_method": method,
        "warnings": warnings,
        "blocking_reasons": [],
        "suggested_actions": [],
    }


def _build_placeholder_omnibus(df: pd.DataFrame, dv_col: str, group_col: str, method: str) -> pd.DataFrame:
    groups = [pd.to_numeric(group_df[dv_col], errors="coerce").dropna().values for _, group_df in df.groupby(group_col)]
    f_stat = np.nan
    pvalue = np.nan
    if len(groups) >= 2 and all(len(values) >= 2 for values in groups):
        try:
            f_stat, pvalue = stats.f_oneway(*groups)
        except Exception:
            pass
    return pd.DataFrame(
        [
            {
                "term": group_col,
                "test": method,
                "statistic": f_stat,
                "pvalue": pvalue,
                "effect_estimate": None,
                "effect_metric": "omega_squared" if "anova" in method else "rank_effect",
                "interpretation_basis": method,
            }
        ]
    )


def _build_placeholder_posthoc(df: pd.DataFrame, group_col: str, control_group: str | None) -> pd.DataFrame:
    if control_group is None:
        return pd.DataFrame(columns=["group_a", "group_b", "pvalue", "effect_estimate", "effect_metric", "interpretation_basis"])

    groups = sorted(df[group_col].dropna().astype(str).unique().tolist())
    rows = []
    for group in groups:
        if group == control_group:
            continue
        rows.append(
            {
                "group_a": control_group,
                "group_b": group,
                "pvalue": None,
                "effect_estimate": None,
                "effect_metric": "hedges_g",
                "interpretation_basis": "pairwise_parametric",
            }
        )
    return pd.DataFrame(rows)


def _build_star_map(posthoc_table: pd.DataFrame | None) -> list[dict]:
    if posthoc_table is None or posthoc_table.empty:
        return []
    return [
        {
            "comparison": f"{row.group_a} vs {row.group_b}",
            "label": "ns" if pd.isna(row.pvalue) else "*" if row.pvalue < 0.05 else "ns",
        }
        for row in posthoc_table.itertuples(index=False)
    ]
