from __future__ import annotations

import pandas as pd


def build_mixedlm_formula(
    dv_col: str,
    time_col: str,
    group_col: str,
    factor2_col: str | None,
    formula_mode: str,
) -> str:
    if factor2_col:
        return f"{dv_col} ~ {time_col} * {group_col} * {factor2_col}"
    return f"{dv_col} ~ {time_col} * {group_col}"


def run_mixedlm(
    df: pd.DataFrame,
    dv_col: str,
    subject_col: str,
    time_col: str,
    group_col: str,
    factor2_col: str | None,
    formula_mode: str,
) -> dict:
    if subject_col not in df.columns:
        return {
            "analysis_status": "blocked",
            "model_summary": None,
            "fixed_effects": None,
            "contrast_table": None,
            "star_map": [],
            "effect_sizes": None,
            "used_formula": "",
            "warnings": [],
            "blocking_reasons": ["MixedLM requires a subject column for random effects."],
            "suggested_actions": ["Map the subject identifier and normalize again."],
        }

    formula = build_mixedlm_formula(dv_col, time_col, group_col, factor2_col, formula_mode)
    fixed_effects = pd.DataFrame(
        [
            {
                "term": "Intercept",
                "beta": None,
                "se": None,
                "ci_low": None,
                "ci_high": None,
            }
        ]
    )
    contrast_table = pd.DataFrame(columns=["contrast", "estimate", "se", "ci_low", "ci_high", "pvalue"])
    effect_sizes = fixed_effects.rename(columns={"beta": "effect_estimate", "se": "standard_error"})

    return {
        "analysis_status": "ready",
        "model_summary": "MixedLM scaffold result. Integrate statsmodels MixedLM fitting here.",
        "fixed_effects": fixed_effects,
        "contrast_table": contrast_table,
        "star_map": [],
        "effect_sizes": effect_sizes,
        "used_formula": formula,
        "warnings": ["MixedLM fitting is scaffolded but not yet numerically implemented."],
        "blocking_reasons": [],
        "suggested_actions": [],
    }
