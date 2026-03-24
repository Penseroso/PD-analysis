from __future__ import annotations

import numpy as np
import pandas as pd
from patsy import build_design_matrices
import statsmodels.formula.api as smf


ANNOTATION_TYPE = "mixedlm_reference_contrast"


def build_mixedlm_formula(
    dv_col: str,
    time_col: str,
    group_col: str,
    factor2_col: str | None,
    formula_mode: str,
) -> str:
    if factor2_col:
        return f"{dv_col} ~ C({time_col}) * C({group_col}) * C({factor2_col})"
    return f"{dv_col} ~ C({time_col}) * C({group_col})"



def run_mixedlm(
    df: pd.DataFrame,
    dv_col: str,
    subject_col: str,
    time_col: str,
    group_col: str,
    factor2_col: str | None,
    formula_mode: str,
    reference_group: str | None = None,
) -> dict:
    warnings: list[str] = []
    blocking_reasons: list[str] = []
    suggested_actions: list[str] = []

    required_cols = [dv_col, subject_col, time_col, group_col]
    missing_cols = [column for column in required_cols if column not in df.columns]
    if missing_cols:
        blocking_reasons.append(f"MixedLM requires columns: {', '.join(missing_cols)}.")
        suggested_actions.append("Verify the subject, time, and group mappings before running MixedLM.")
    if factor2_col and factor2_col not in df.columns:
        blocking_reasons.append(f"Selected factor2 column '{factor2_col}' is not present in the normalized data.")
        suggested_actions.append("Choose a valid factor2 column or rerun normalization.")

    if blocking_reasons:
        return {
            "analysis_status": "blocked",
            "model_summary": None,
            "fixed_effects": None,
            "contrast_table": None,
            "star_map": [],
            "effect_sizes": None,
            "used_method": "mixedlm",
            "used_formula": "",
            "engine_used": "statsmodels",
            "warnings": warnings,
            "blocking_reasons": blocking_reasons,
            "suggested_actions": suggested_actions,
            "dv_col": dv_col,
        }

    fit_df = _prepare_mixedlm_df(df, dv_col, subject_col, time_col, group_col, factor2_col)
    if fit_df.empty:
        return {
            "analysis_status": "blocked",
            "model_summary": None,
            "fixed_effects": None,
            "contrast_table": None,
            "star_map": [],
            "effect_sizes": None,
            "used_method": "mixedlm",
            "used_formula": "",
            "engine_used": "statsmodels",
            "warnings": warnings,
            "blocking_reasons": ["No complete observations are available for MixedLM fitting."],
            "suggested_actions": ["Check missing values and verify the selected biomarker."],
            "dv_col": dv_col,
        }

    used_formula = build_mixedlm_formula(dv_col, time_col, group_col, factor2_col, formula_mode)
    try:
        model = smf.mixedlm(used_formula, data=fit_df, groups=fit_df[subject_col], re_formula="1")
        result = _fit_mixedlm(model)
    except Exception as exc:
        return {
            "analysis_status": "blocked",
            "model_summary": None,
            "fixed_effects": None,
            "contrast_table": None,
            "star_map": [],
            "effect_sizes": None,
            "used_method": "mixedlm",
            "used_formula": used_formula,
            "engine_used": "statsmodels",
            "warnings": warnings,
            "blocking_reasons": [f"MixedLM fitting failed: {exc}"],
            "suggested_actions": ["Inspect missingness, small cell sizes, or simplify the design."],
            "dv_col": dv_col,
        }

    if not bool(getattr(result, "converged", False)):
        warnings.append("MixedLM did not report convergence. Coefficients are returned, but interpret them with caution.")

    fixed_effects = _build_fixed_effects_table(result)
    effect_sizes = _build_effect_sizes_table(fixed_effects)
    contrast_table, contrast_warnings, reference_group_used = _build_contrast_table(
        result=result,
        model=model,
        fit_df=fit_df,
        time_col=time_col,
        group_col=group_col,
        factor2_col=factor2_col,
        reference_group=reference_group,
    )
    warnings.extend(contrast_warnings)

    return {
        "analysis_status": "ready",
        "model_summary": result.summary().as_text(),
        "fixed_effects": fixed_effects,
        "contrast_table": contrast_table,
        "star_map": _build_star_map(contrast_table),
        "effect_sizes": effect_sizes,
        "used_method": "mixedlm",
        "used_formula": used_formula,
        "engine_used": "statsmodels",
        "reference_group_used": reference_group_used,
        "warnings": sorted(set(warnings)),
        "blocking_reasons": [],
        "suggested_actions": [],
        "dv_col": dv_col,
    }



def _prepare_mixedlm_df(
    df: pd.DataFrame,
    dv_col: str,
    subject_col: str,
    time_col: str,
    group_col: str,
    factor2_col: str | None,
) -> pd.DataFrame:
    keep_cols = [column for column in [dv_col, subject_col, time_col, group_col, factor2_col] if column and column in df.columns]
    fit_df = df[keep_cols].copy()
    fit_df[dv_col] = pd.to_numeric(fit_df[dv_col], errors="coerce")
    fit_df = fit_df.dropna(subset=[dv_col, subject_col, time_col, group_col])

    fit_df[subject_col] = fit_df[subject_col].astype(str)
    fit_df[time_col] = pd.Categorical(fit_df[time_col].astype(str), categories=sorted(fit_df[time_col].astype(str).unique()), ordered=True)
    fit_df[group_col] = pd.Categorical(fit_df[group_col].astype(str), categories=sorted(fit_df[group_col].astype(str).unique()), ordered=True)
    if factor2_col and factor2_col in fit_df.columns:
        fit_df[factor2_col] = pd.Categorical(
            fit_df[factor2_col].astype(str),
            categories=sorted(fit_df[factor2_col].astype(str).unique()),
            ordered=True,
        )
    return fit_df



def _fit_mixedlm(model):
    last_error: Exception | None = None
    for kwargs in (
        {"reml": False, "method": "lbfgs", "disp": False},
        {"reml": False, "method": "powell", "disp": False},
    ):
        try:
            return model.fit(**kwargs)
        except Exception as exc:
            last_error = exc
    if last_error is None:
        raise RuntimeError("Unknown MixedLM fitting failure")
    raise last_error



def _build_fixed_effects_table(result) -> pd.DataFrame:
    ci = result.conf_int()
    rows: list[dict] = []
    for term in result.fe_params.index:
        rows.append(
            {
                "term": str(term),
                "beta": _safe_float(result.fe_params[term]),
                "se": _safe_float(result.bse_fe[term]),
                "ci_low": _safe_float(ci.loc[term, 0]) if term in ci.index else None,
                "ci_high": _safe_float(ci.loc[term, 1]) if term in ci.index else None,
                "pvalue": _safe_float(result.pvalues.get(term)),
            }
        )
    return pd.DataFrame(rows)



def _build_effect_sizes_table(fixed_effects: pd.DataFrame) -> pd.DataFrame:
    effect_sizes = fixed_effects.rename(columns={"beta": "effect_estimate", "se": "standard_error"}).copy()
    effect_sizes["effect_metric"] = "beta_se_ci"
    effect_sizes["interpretation_basis"] = "mixedlm"
    return effect_sizes



def _build_contrast_table(
    result,
    model,
    fit_df: pd.DataFrame,
    time_col: str,
    group_col: str,
    factor2_col: str | None,
    reference_group: str | None,
) -> tuple[pd.DataFrame, list[str], str | None]:
    warnings: list[str] = []
    group_levels = fit_df[group_col].cat.categories.tolist() if hasattr(fit_df[group_col], "cat") else sorted(fit_df[group_col].astype(str).unique())
    time_levels = fit_df[time_col].cat.categories.tolist() if hasattr(fit_df[time_col], "cat") else sorted(fit_df[time_col].astype(str).unique())
    factor2_levels = [None]
    if factor2_col and factor2_col in fit_df.columns:
        factor2_levels = fit_df[factor2_col].cat.categories.tolist() if hasattr(fit_df[factor2_col], "cat") else sorted(fit_df[factor2_col].astype(str).unique())

    if len(group_levels) < 2:
        return pd.DataFrame(columns=["contrast", "estimate", "se", "ci_low", "ci_high", "pvalue"]), warnings, None

    reference_group_used = reference_group if reference_group in group_levels else group_levels[0]
    if reference_group and reference_group not in group_levels:
        warnings.append(
            f"Reference group '{reference_group}' was not found in the fitted data. Falling back to '{reference_group_used}'."
        )
    elif reference_group is None:
        warnings.append(f"Reference group was not selected. Falling back to '{reference_group_used}'.")

    rows: list[dict] = []
    comparison_groups = [group_level for group_level in group_levels if group_level != reference_group_used]
    for time_level in time_levels:
        for factor2_level in factor2_levels:
            base_row = {time_col: time_level, group_col: reference_group_used}
            if factor2_col:
                base_row[factor2_col] = factor2_level
            base_design = np.asarray(build_design_matrices([model.data.design_info], pd.DataFrame([base_row]))[0][0], dtype=float)
            for group_level in comparison_groups:
                comp_row = {time_col: time_level, group_col: group_level}
                if factor2_col:
                    comp_row[factor2_col] = factor2_level
                comp_design = np.asarray(build_design_matrices([model.data.design_info], pd.DataFrame([comp_row]))[0][0], dtype=float)
                contrast = np.atleast_2d(comp_design - base_design)
                test_result = result.t_test(contrast)
                estimate = _safe_float(np.asarray(test_result.effect).squeeze())
                se = _safe_float(np.asarray(test_result.sd).squeeze())
                ci = np.asarray(test_result.conf_int())
                pvalue = _safe_float(np.asarray(test_result.pvalue).squeeze())
                row = {
                    "annotation_type": ANNOTATION_TYPE,
                    "contrast": f"{reference_group_used} vs {group_level} at {time_level}",
                    "time": str(time_level),
                    "group_a": str(reference_group_used),
                    "group_b": str(group_level),
                    "estimate": estimate,
                    "se": se,
                    "ci_low": _safe_float(ci.squeeze()[0]) if ci.size >= 2 else None,
                    "ci_high": _safe_float(ci.squeeze()[1]) if ci.size >= 2 else None,
                    "pvalue": pvalue,
                    "effect_estimate": estimate,
                    "effect_metric": "beta_se_ci",
                    "interpretation_basis": "mixedlm_contrast",
                    "comparison": f"{reference_group_used} vs {group_level}",
                    "reference_group": str(reference_group_used),
                }
                if factor2_col:
                    row["factor2"] = str(factor2_level)
                    row["contrast"] = f"{reference_group_used} vs {group_level} at {time_level} | {factor2_col}={factor2_level}"
                rows.append(row)

    if not rows:
        warnings.append("No reference-cell contrasts were generated for the fitted MixedLM.")
        return pd.DataFrame(columns=["contrast", "estimate", "se", "ci_low", "ci_high", "pvalue"]), warnings, reference_group_used

    contrast_table = pd.DataFrame(rows)
    return contrast_table, warnings, reference_group_used



def _build_star_map(contrast_table: pd.DataFrame | None) -> list[dict]:
    if contrast_table is None or contrast_table.empty or "pvalue" not in contrast_table.columns:
        return []
    star_map: list[dict] = []
    for row in contrast_table.itertuples(index=False):
        pvalue = getattr(row, "pvalue", None)
        if pvalue is None or pd.isna(pvalue):
            continue
        label = _pvalue_to_label(float(pvalue))
        if label is None:
            continue
        record = {
            "annotation_type": getattr(row, "annotation_type", ANNOTATION_TYPE),
            "comparison": getattr(row, "comparison", getattr(row, "contrast", None)),
            "time": getattr(row, "time", None),
            "group_a": getattr(row, "group_a", None),
            "group_b": getattr(row, "group_b", None),
            "pvalue": float(pvalue),
            "label": label,
        }
        if hasattr(row, "factor2"):
            record["factor2"] = getattr(row, "factor2")
        if hasattr(row, "reference_group"):
            record["reference_group"] = getattr(row, "reference_group")
        star_map.append(record)
    return star_map



def _safe_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    return float(value)



def _pvalue_to_label(pvalue: float) -> str | None:
    if pvalue < 0.001:
        return "***"
    if pvalue < 0.01:
        return "**"
    if pvalue < 0.05:
        return "*"
    return None
