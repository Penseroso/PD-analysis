from __future__ import annotations

import pandas as pd
from scipy import stats

from utils.stats.contracts.diagnostics import AssumptionSummary, CombinedDiagnosticsSummary
from utils.stats.planning.execution_bridge import execute_and_normalize
from utils.stats.planning.plan_builder import build_analysis_plan_contract
from utils.stats.formatting.result_normalizer import to_legacy_longitudinal_payload


def compute_longitudinal_assumptions(
    df: pd.DataFrame,
    dv_col: str,
    group_col: str,
    subject_col: str,
    time_col: str,
    between_factors: list[str],
    time_order: list[str] | None = None,
) -> dict:
    clean_df = _prepare_longitudinal_df(df, dv_col, group_col, subject_col, time_col, time_order=time_order)
    normality: dict = {}

    grouping_cols = [time_col]
    if group_col in clean_df.columns and clean_df[group_col].nunique(dropna=True) > 1:
        grouping_cols = [group_col, time_col]

    for keys, cell_df in clean_df.groupby(grouping_cols, dropna=False, sort=False):
        label = "|".join(map(str, keys if isinstance(keys, tuple) else (keys,)))
        values = cell_df[dv_col].to_numpy(dtype=float)
        if len(values) >= 3:
            stat, pvalue = stats.shapiro(values)
            normality[label] = {
                "stat": float(stat),
                "pvalue": float(pvalue),
                "is_normal": bool(pvalue >= 0.05),
                "n": int(len(values)),
            }
        else:
            normality[label] = {
                "stat": None,
                "pvalue": None,
                "is_normal": True,
                "n": int(len(values)),
            }

    sphericity = _compute_sphericity(clean_df, dv_col, subject_col, time_col, group_col)
    return {"normality": normality, "sphericity": sphericity}


def run_longitudinal(
    df: pd.DataFrame,
    dv_col: str,
    group_col: str,
    subject_col: str,
    time_col: str,
    between_factors: list[str],
    factor2_col: str | None,
    method: str,
    time_order: list[str] | None = None,
) -> dict:
    clean_df = _prepare_longitudinal_df(df, dv_col, group_col, subject_col, time_col, factor2_col, time_order=time_order)
    assumptions = compute_longitudinal_assumptions(
        df=clean_df,
        dv_col=dv_col,
        group_col=group_col,
        subject_col=subject_col,
        time_col=time_col,
        between_factors=between_factors,
        time_order=time_order,
    )
    warnings: list[str] = []
    blocking_reasons: list[str] = []
    suggested_actions: list[str] = []

    if subject_col not in clean_df.columns or clean_df[subject_col].isna().all():
        blocking_reasons.append("Longitudinal analysis requires a subject column.")
        suggested_actions.append("Map the subject identifier and normalize again.")
    if time_col not in clean_df.columns or clean_df[time_col].isna().all():
        blocking_reasons.append("Longitudinal analysis requires a time column.")
        suggested_actions.append("Map the time variable and normalize again.")
    elif clean_df[time_col].nunique(dropna=True) < 2:
        blocking_reasons.append("Longitudinal analysis requires at least two time levels.")
        suggested_actions.append("Confirm the time mapping and input format.")
    if factor2_col is not None or len(between_factors) >= 2:
        blocking_reasons.append(
            "Designs with two or more between-subject factors must use the MixedLM engine because pingouin.mixed_anova() supports only one between-subject factor."
        )
        suggested_actions.append("Use the MixedLM route for Three-way Mixed ANOVA style designs.")

    if blocking_reasons:
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
            "warnings": warnings,
            "blocking_reasons": blocking_reasons,
            "suggested_actions": suggested_actions,
            "dv_col": dv_col,
            "time_order": _resolve_time_order(clean_df, time_col, time_order),
        }

    resolved_method = _resolve_longitudinal_method(method, assumptions, clean_df, group_col)
    if method == "auto":
        warnings.append(f"Auto-selected method: {resolved_method}.")

    if clean_df[group_col].nunique(dropna=True) <= 1:
        if resolved_method == "friedman":
            pass
        else:
            if resolved_method not in {"rm_anova", "auto"}:
                warnings.append(f"Method '{resolved_method}' is not valid for a single-group repeated design. Using rm_anova.")
            resolved_method = "rm_anova"
    else:
        if resolved_method == "friedman":
            warnings.append(
                "A full nonparametric omnibus for group-by-time mixed designs is not available in the current dependency set. Using mixed_anova for the balanced mixed-design path."
            )
            resolved_method = "mixed_anova"
        if any(not item.get("is_normal", True) for item in assumptions["normality"].values()):
            warnings.append(
                "At least one group-by-time cell failed Shapiro-Wilk. Mixed ANOVA is still reported because no full nonparametric mixed-design omnibus is available with the current dependencies."
            )

    diagnostics = CombinedDiagnosticsSummary(
        assumptions=AssumptionSummary(
            normality=assumptions.get("normality", {}),
            levene={},
            sphericity=assumptions.get("sphericity"),
        )
    )
    plan = build_analysis_plan_contract(
        data_type="longitudinal",
        omnibus_method=resolved_method,
        factor2_col=factor2_col,
        warnings=warnings,
    )
    result = execute_and_normalize(
        plan,
        diagnostics=diagnostics,
        df=clean_df,
        dv_col=dv_col,
        group_col=group_col,
        subject_col=subject_col,
        time_col=time_col,
    )
    return to_legacy_longitudinal_payload(
        result,
        assumptions=assumptions,
        dv_col=dv_col,
        time_order=_resolve_time_order(clean_df, time_col, time_order),
    )


def _prepare_longitudinal_df(
    df: pd.DataFrame,
    dv_col: str,
    group_col: str,
    subject_col: str,
    time_col: str,
    factor2_col: str | None = None,
    time_order: list[str] | None = None,
) -> pd.DataFrame:
    keep_cols = [column for column in [dv_col, group_col, subject_col, time_col, factor2_col] if column and column in df.columns]
    clean_df = df[keep_cols].copy()
    clean_df[dv_col] = pd.to_numeric(clean_df[dv_col], errors="coerce")
    for column in [group_col, subject_col, factor2_col]:
        if column and column in clean_df.columns:
            clean_df[column] = clean_df[column].astype(str)
    if time_col in clean_df.columns:
        clean_df[time_col] = clean_df[time_col].astype(str)
        resolved_time_order = _resolve_time_order(clean_df, time_col, time_order)
        clean_df[time_col] = pd.Categorical(clean_df[time_col], categories=resolved_time_order, ordered=True)
    return clean_df.dropna(subset=[dv_col, subject_col, time_col])


def _resolve_longitudinal_method(method: str, assumptions: dict, df: pd.DataFrame, group_col: str) -> str:
    if method != "auto":
        return method
    any_non_normal = any(not item.get("is_normal", True) for item in assumptions.get("normality", {}).values())
    if df[group_col].nunique(dropna=True) <= 1:
        return "friedman" if any_non_normal else "rm_anova"
    return "mixed_anova"


def _compute_sphericity(
    df: pd.DataFrame,
    dv_col: str,
    subject_col: str,
    time_col: str,
    group_col: str,
) -> dict:
    n_time = int(df[time_col].nunique(dropna=True)) if time_col in df.columns else 0
    if n_time < 2:
        return {
            "applies": False,
            "method": "mauchly",
            "tested": False,
            "sphericity_met": None,
            "W": None,
            "chi2": None,
            "dof": None,
            "pvalue": None,
            "by_group": {},
        }
    if n_time == 2:
        return {
            "applies": False,
            "method": "mauchly",
            "tested": False,
            "sphericity_met": True,
            "W": None,
            "chi2": None,
            "dof": None,
            "pvalue": None,
            "by_group": {},
            "note": "Sphericity is automatically satisfied with two time levels.",
        }

    output = {
        "applies": True,
        "method": "mauchly",
        "tested": False,
        "sphericity_met": None,
        "W": None,
        "chi2": None,
        "dof": None,
        "pvalue": None,
        "by_group": {},
    }

    try:
        import pingouin as pg

        result = pg.sphericity(df, dv=dv_col, within=time_col, subject=subject_col, method="mauchly")
        output.update(
            {
                "tested": True,
                "sphericity_met": bool(result.spher),
                "W": _safe_float(result.W),
                "chi2": _safe_float(result.chi2),
                "dof": _safe_float(result.dof),
                "pvalue": _safe_float(result.pval),
            }
        )
    except Exception as exc:
        output["error"] = str(exc)

    if group_col in df.columns and df[group_col].nunique(dropna=True) > 1:
        try:
            import pingouin as pg

            for group_name, group_df in df.groupby(group_col, sort=False):
                try:
                    result = pg.sphericity(group_df, dv=dv_col, within=time_col, subject=subject_col, method="mauchly")
                    output["by_group"][str(group_name)] = {
                        "tested": True,
                        "sphericity_met": bool(result.spher),
                        "W": _safe_float(result.W),
                        "chi2": _safe_float(result.chi2),
                        "dof": _safe_float(result.dof),
                        "pvalue": _safe_float(result.pval),
                    }
                except Exception as exc:
                    output["by_group"][str(group_name)] = {
                        "tested": False,
                        "sphericity_met": None,
                        "W": None,
                        "chi2": None,
                        "dof": None,
                        "pvalue": None,
                        "error": str(exc),
                    }
        except Exception:
            pass
    return output


def _resolve_time_order(df: pd.DataFrame, time_col: str, time_order: list[str] | None) -> list[str]:
    explicit = [str(item) for item in (time_order or []) if item is not None]
    if explicit:
        seen = set(df[time_col].astype(str).dropna().unique().tolist())
        ordered = [item for item in explicit if item in seen]
        leftovers = [item for item in df[time_col].astype(str).dropna().unique().tolist() if item not in ordered]
        return ordered + leftovers
    return df[time_col].astype(str).dropna().drop_duplicates().tolist()


def _safe_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, pd.Series):
        if value.empty:
            return None
        value = value.iloc[0]
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    return float(value)
