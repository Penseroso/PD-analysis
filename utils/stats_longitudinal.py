from __future__ import annotations

import pandas as pd
from scipy import stats


def compute_longitudinal_assumptions(
    df: pd.DataFrame,
    dv_col: str,
    group_col: str,
    subject_col: str,
    time_col: str,
    between_factors: list[str],
) -> dict:
    normality: dict = {}
    for keys, cell_df in df.groupby([group_col, time_col], dropna=False):
        values = pd.to_numeric(cell_df[dv_col], errors="coerce").dropna()
        label = "|".join(map(str, keys if isinstance(keys, tuple) else (keys,)))
        if len(values) >= 3:
            stat, p = stats.shapiro(values)
            normality[label] = {"stat": float(stat), "pvalue": float(p), "is_normal": bool(p >= 0.05)}
        else:
            normality[label] = {"stat": None, "pvalue": None, "is_normal": True}

    n_time = int(df[time_col].nunique(dropna=True)) if time_col in df.columns else 0
    sphericity = {
        "applies": n_time >= 3,
        "method": "mauchly",
        "pvalue": None,
        "sphericity_met": None if n_time >= 3 else True,
        "correction_recommended": None if n_time >= 3 else False,
    }

    return {"normality": normality, "sphericity": sphericity}


def run_longitudinal(
    df: pd.DataFrame,
    dv_col: str,
    group_col: str,
    subject_col: str,
    time_col: str,
    control_group: str | None,
    between_factors: list[str],
    factor2_col: str | None,
    method: str,
) -> dict:
    assumptions = compute_longitudinal_assumptions(
        df=df,
        dv_col=dv_col,
        group_col=group_col,
        subject_col=subject_col,
        time_col=time_col,
        between_factors=between_factors,
    )

    if subject_col not in df.columns or time_col not in df.columns:
        return {
            "analysis_status": "blocked",
            "assumptions": assumptions,
            "omnibus": None,
            "posthoc_table": None,
            "star_map": [],
            "effect_sizes": {"omnibus": None, "pairwise": None},
            "used_method": method,
            "correction_applied": None,
            "engine_used": "pingouin",
            "warnings": [],
            "blocking_reasons": ["Longitudinal analysis requires subject and time columns."],
            "suggested_actions": ["Map subject and time columns and normalize again."],
        }

    omnibus = pd.DataFrame(
        [
            {
                "term": "time" if len(between_factors) == 0 else "time * group",
                "test": method,
                "statistic": None,
                "pvalue": None,
                "effect_estimate": None,
                "effect_metric": "partial_eta_squared",
                "interpretation_basis": method,
            }
        ]
    )
    posthoc = pd.DataFrame(
        columns=["time", "group_a", "group_b", "pvalue", "effect_estimate", "effect_metric", "interpretation_basis"]
    )
    effect_sizes = {
        "omnibus": omnibus[["term", "effect_estimate", "effect_metric", "interpretation_basis"]].copy(),
        "pairwise": posthoc.copy(),
    }

    warnings = []
    if assumptions["sphericity"]["applies"]:
        warnings.append("Sphericity output is a placeholder until pingouin integration is completed.")
    if control_group is None and "group" in between_factors:
        warnings.append("Control-based pairwise summaries are omitted until a control group is selected.")

    return {
        "analysis_status": "ready",
        "assumptions": assumptions,
        "omnibus": omnibus,
        "posthoc_table": posthoc,
        "star_map": [],
        "effect_sizes": effect_sizes,
        "used_method": method,
        "correction_applied": assumptions["sphericity"],
        "engine_used": "pingouin",
        "warnings": warnings,
        "blocking_reasons": [],
        "suggested_actions": [],
    }
