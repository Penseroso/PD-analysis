from __future__ import annotations

import pandas as pd
from scipy import stats

from utils.stats.contracts.diagnostics import AssumptionSummary, CombinedDiagnosticsSummary
from utils.stats.planning.execution_bridge import execute_and_normalize
from utils.stats.planning.plan_builder import build_analysis_plan_contract
from utils.stats.formatting.result_normalizer import to_legacy_cross_payload


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
    elif resolved_method not in {"one_way_anova", "welch_anova", "kruskal"}:
        warnings.append(f"Unsupported method '{resolved_method}' requested. Falling back to auto resolution.")
        resolved_method = _resolve_cross_method("auto", assumptions)

    diagnostics = CombinedDiagnosticsSummary(
        assumptions=AssumptionSummary(
            normality=assumptions.get("normality", {}),
            levene=assumptions.get("levene", {}),
            sphericity=None,
        )
    )
    plan = build_analysis_plan_contract(
        data_type="cross",
        omnibus_method=resolved_method,
        control_group=control_group,
        warnings=warnings,
    )
    result = execute_and_normalize(
        plan,
        diagnostics=diagnostics,
        df=clean_df,
        dv_col=dv_col,
        group_col=group_col,
        control_group=control_group,
    )
    return to_legacy_cross_payload(result, assumptions=assumptions, dv_col=dv_col)


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
