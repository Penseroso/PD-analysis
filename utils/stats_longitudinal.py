from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
import pingouin as pg
from scipy import stats


PVALUE_COLUMNS = ("p_corr", "pvalue", "p_unc", "pval")



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
            omnibus, posthoc_table, omnibus_effects, pairwise_effects = _run_friedman(
                clean_df, dv_col, subject_col, time_col
            )
            correction_applied = assumptions["sphericity"]
        else:
            if resolved_method not in {"rm_anova", "auto"}:
                warnings.append(f"Method '{resolved_method}' is not valid for a single-group repeated design. Using rm_anova.")
            omnibus, posthoc_table, omnibus_effects, pairwise_effects, correction_applied = _run_rm_anova(
                clean_df, dv_col, subject_col, time_col
            )
            resolved_method = "rm_anova"
        engine_used = "pingouin"
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
        omnibus, posthoc_table, omnibus_effects, pairwise_effects, correction_applied = _run_mixed_anova(
            clean_df, dv_col, subject_col, time_col, group_col
        )
        engine_used = "pingouin"
        resolved_method = "mixed_anova"

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
        "correction_applied": correction_applied,
        "engine_used": engine_used,
        "warnings": sorted(set(warnings)),
        "blocking_reasons": [],
        "suggested_actions": [],
        "dv_col": dv_col,
        "time_order": _resolve_time_order(clean_df, time_col, time_order),
    }



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
    return output



def _run_rm_anova(
    df: pd.DataFrame,
    dv_col: str,
    subject_col: str,
    time_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    anova = pg.rm_anova(
        data=df,
        dv=dv_col,
        within=time_col,
        subject=subject_col,
        correction="auto",
        detailed=True,
        effsize="np2",
    )
    effect_row = anova.loc[anova["Source"] != "Error"].copy()
    omnibus = effect_row.rename(
        columns={
            "Source": "term",
            "F": "statistic",
            "p_unc": "pvalue",
            "DF": "df1",
            "SS": "ss",
            "MS": "ms",
            "np2": "effect_estimate",
        }
    )
    omnibus["df2"] = _safe_float(anova.loc[anova["Source"] == "Error", "DF"].iloc[0]) if (anova["Source"] == "Error").any() else None
    omnibus["test"] = "rm_anova"
    omnibus["effect_metric"] = "partial_eta_squared"
    omnibus["interpretation_basis"] = "rm_anova"
    omnibus = omnibus[["term", "test", "statistic", "pvalue", "df1", "df2", "ss", "ms", "effect_estimate", "effect_metric", "interpretation_basis"]]

    pairwise = pg.pairwise_tests(
        data=df,
        dv=dv_col,
        within=time_col,
        subject=subject_col,
        parametric=True,
        padjust="bonf",
        effsize="hedges",
    )
    posthoc = _format_pairwise_time_tests(pairwise, interpretation_basis="pairwise_parametric")
    pairwise_effects = posthoc[["time_a", "time_b", "effect_estimate", "effect_metric", "interpretation_basis"]].copy()
    omnibus_effects = omnibus[["term", "effect_estimate", "effect_metric", "interpretation_basis"]].copy()
    correction_applied = {
        "method": "auto",
        "sphericity_tested": _bool_or_none(effect_row.get("sphericity")),
        "sphericity_met": _bool_or_none(effect_row.get("sphericity")),
        "greenhouse_geisser_applied": _safe_float(effect_row.get("p_GG_corr")) is not None,
        "pvalue_uncorrected": _safe_float(effect_row.get("p_unc")),
        "pvalue_corrected": _safe_float(effect_row.get("p_GG_corr")),
        "epsilon": _safe_float(effect_row.get("eps")),
        "W": _safe_float(effect_row.get("W_spher")),
        "p_spher": _safe_float(effect_row.get("p_spher")),
    }
    return omnibus, posthoc, omnibus_effects, pairwise_effects, correction_applied



def _run_friedman(
    df: pd.DataFrame,
    dv_col: str,
    subject_col: str,
    time_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    wide = df.pivot_table(index=subject_col, columns=time_col, values=dv_col, aggfunc="mean")
    wide = wide.dropna(axis=0, how="any")
    statistic, pvalue = stats.friedmanchisquare(*[wide[column].to_numpy(dtype=float) for column in wide.columns])
    n_subjects = wide.shape[0]
    n_time = wide.shape[1]
    kendalls_w = _safe_float(statistic / (n_subjects * (n_time - 1))) if n_subjects > 0 and n_time > 1 else None
    omnibus = pd.DataFrame(
        [
            {
                "term": time_col,
                "test": "friedman",
                "statistic": _safe_float(statistic),
                "pvalue": _safe_float(pvalue),
                "df1": float(n_time - 1),
                "df2": None,
                "ss": None,
                "ms": None,
                "effect_estimate": kendalls_w,
                "effect_metric": "kendalls_w",
                "interpretation_basis": "friedman",
            }
        ]
    )
    rows: list[dict] = []
    n_tests = max(1, n_time * (n_time - 1) // 2)
    for time_a, time_b in combinations(wide.columns.tolist(), 2):
        paired = wide[[time_a, time_b]].dropna()
        if paired.empty:
            continue
        diff = paired[time_a] - paired[time_b]
        wilcoxon = stats.wilcoxon(paired[time_a], paired[time_b], zero_method="wilcox", alternative="two-sided")
        p_corr = min(float(wilcoxon.pvalue) * n_tests, 1.0)
        rows.append(
            {
                "annotation_type": "longitudinal_time_pair_within_group",
                "time_a": str(time_a),
                "time_b": str(time_b),
                "comparison": f"{time_a} vs {time_b}",
                "test": "wilcoxon",
                "statistic": _safe_float(wilcoxon.statistic),
                "pvalue": p_corr,
                "p_unc": _safe_float(wilcoxon.pvalue),
                "p_adjust": "bonferroni",
                "effect_estimate": _paired_rank_biserial(diff.to_numpy(dtype=float)),
                "effect_metric": "rank_biserial",
                "interpretation_basis": "pairwise_nonparametric",
            }
        )
    posthoc = pd.DataFrame(rows)
    if posthoc.empty:
        posthoc = pd.DataFrame(
            columns=[
                "annotation_type",
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
    omnibus_effects = omnibus[["term", "effect_estimate", "effect_metric", "interpretation_basis"]].copy()
    pairwise_effects = posthoc[["time_a", "time_b", "effect_estimate", "effect_metric", "interpretation_basis"]].copy()
    return omnibus, posthoc, omnibus_effects, pairwise_effects



def _run_mixed_anova(
    df: pd.DataFrame,
    dv_col: str,
    subject_col: str,
    time_col: str,
    group_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    anova = pg.mixed_anova(
        data=df,
        dv=dv_col,
        within=time_col,
        subject=subject_col,
        between=group_col,
        correction="auto",
        effsize="np2",
    )
    omnibus = anova.rename(
        columns={
            "Source": "term",
            "F": "statistic",
            "p_unc": "pvalue",
            "DF1": "df1",
            "DF2": "df2",
            "SS": "ss",
            "MS": "ms",
            "np2": "effect_estimate",
        }
    ).copy()
    omnibus["test"] = "mixed_anova"
    omnibus["effect_metric"] = "partial_eta_squared"
    omnibus["interpretation_basis"] = "mixed_anova"
    if "ms" not in omnibus.columns:
        omnibus["ms"] = None
    omnibus = omnibus[["term", "test", "statistic", "pvalue", "df1", "df2", "ss", "ms", "effect_estimate", "effect_metric", "interpretation_basis"]]

    pairwise = pg.pairwise_tests(
        data=df,
        dv=dv_col,
        within=time_col,
        between=group_col,
        subject=subject_col,
        parametric=True,
        padjust="bonf",
        effsize="hedges",
        interaction=True,
    )
    posthoc = _format_pairwise_mixed_tests(pairwise, time_col, group_col)
    omnibus_effects = omnibus[["term", "effect_estimate", "effect_metric", "interpretation_basis"]].copy()
    pairwise_effects = posthoc[["comparison", "effect_estimate", "effect_metric", "interpretation_basis"]].copy()

    time_row = anova.loc[anova["Source"].astype(str).str.lower() == time_col.lower()]
    if time_row.empty:
        time_row = anova.loc[anova["Source"].astype(str).str.lower() == "time"]
    epsilon = _safe_float(time_row.iloc[0].get("eps")) if not time_row.empty else None
    correction_applied = {
        "method": "auto",
        "sphericity_tested": True if epsilon is not None and df[time_col].nunique(dropna=True) >= 3 else False,
        "greenhouse_geisser_applied": epsilon is not None and epsilon < 1.0,
        "epsilon": epsilon,
        "pvalue_corrected": None,
    }
    return omnibus, posthoc, omnibus_effects, pairwise_effects, correction_applied



def _format_pairwise_time_tests(pairwise: pd.DataFrame, interpretation_basis: str) -> pd.DataFrame:
    posthoc = pairwise.rename(
        columns={
            "A": "time_a",
            "B": "time_b",
            "T": "statistic",
            "p_corr": "pvalue",
            "hedges": "effect_estimate",
        }
    ).copy()
    if "pvalue" not in posthoc.columns:
        posthoc["pvalue"] = posthoc.get("p_unc")
    posthoc["annotation_type"] = "longitudinal_time_pair_within_group"
    posthoc["comparison"] = posthoc["time_a"].astype(str) + " vs " + posthoc["time_b"].astype(str)
    posthoc["test"] = "pairwise_ttests"
    posthoc["effect_metric"] = "hedges_g"
    posthoc["interpretation_basis"] = interpretation_basis
    ordered = [
        "annotation_type",
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
    return posthoc[[column for column in ordered if column in posthoc.columns]]



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
            record.update(
                {
                    "annotation_type": "cross_group_pair",
                    "comparison": f"{row.get('A')} vs {row.get('B')}",
                }
            )
            if pd.notna(row.get(time_col)):
                record["time"] = str(row.get(time_col))
            if pd.notna(row.get("A")):
                record["group_a"] = str(row.get("A"))
            if pd.notna(row.get("B")):
                record["group_b"] = str(row.get("B"))
        rows.append(record)

    ordered = [
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
    formatted = pd.DataFrame(rows)
    if formatted.empty:
        return pd.DataFrame(columns=ordered)
    for column in ordered:
        if column not in formatted.columns:
            formatted[column] = None
    return formatted[ordered]



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
            "annotation_type": getattr(row, "annotation_type", _infer_annotation_type(row)),
            "comparison": getattr(row, "comparison", None),
            "pvalue": float(pvalue),
            "label": label,
        }
        for field in ("time", "time_a", "time_b", "group", "group_a", "group_b", "factor2", "Contrast"):
            if hasattr(row, field):
                record[field] = getattr(row, field)
        star_map.append(record)
    return star_map



def _infer_annotation_type(row) -> str:
    if hasattr(row, "time_a") and hasattr(row, "time_b"):
        return "longitudinal_time_pair_within_group"
    if hasattr(row, "time") and hasattr(row, "group_a") and hasattr(row, "group_b"):
        return "longitudinal_group_pair_at_time"
    if hasattr(row, "group_a") and hasattr(row, "group_b"):
        return "cross_group_pair"
    return "cross_group_pair"



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



def _bool_or_none(value: object) -> bool | None:
    if isinstance(value, pd.Series):
        if value.empty:
            return None
        value = value.iloc[0]
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    return bool(value)



def _pvalue_to_label(pvalue: float) -> str | None:
    if pvalue < 0.001:
        return "***"
    if pvalue < 0.01:
        return "**"
    if pvalue < 0.05:
        return "*"
    return None
