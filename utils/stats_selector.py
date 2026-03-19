from __future__ import annotations


def select_method(
    data_type: str,
    normality: dict,
    sphericity: dict | None,
    levene: dict,
    balance_info: dict,
    between_factors: list[str],
    n_per_group: dict,
) -> dict:
    rationale: list[str] = []
    fallback_reason: str | None = None

    any_non_normal = any(not item.get("is_normal", True) for item in normality.values()) if normality else False
    unequal_variance = not levene.get("equal_variance", True) if levene else False
    repeated_missing = balance_info.get("has_missing_repeated_cells", False)
    is_balanced = balance_info.get("is_balanced", True)
    n_groups = len(n_per_group) if n_per_group else 0

    if data_type == "cross":
        if any_non_normal:
            recommended_method = "kruskal"
            recommended_engine = "scipy"
            rationale.append("At least one group failed normality screening, so a rank-based omnibus test is preferred.")
        elif unequal_variance:
            recommended_method = "welch_anova"
            recommended_engine = "pingouin"
            rationale.append("Levene's test suggests unequal variances, so Welch ANOVA is preferred.")
        else:
            recommended_method = "one_way_anova"
            recommended_engine = "pingouin"
            rationale.append("Normality and equal-variance checks support a standard One-way ANOVA.")
    else:
        if len(between_factors) >= 2:
            recommended_method = "mixedlm"
            recommended_engine = "statsmodels"
            fallback_reason = (
                "Pingouin mixed_anova() supports only one between-subject factor. "
                "Designs such as group + factor2 + time must use MixedLM."
            )
            rationale.append("Two or more between-subject factors require the MixedLM engine.")
        elif repeated_missing or not is_balanced:
            recommended_method = "mixedlm"
            recommended_engine = "statsmodels"
            fallback_reason = (
                "Repeated cells are missing or unbalanced. MixedLM is preferred because RM-ANOVA and Mixed ANOVA would rely on listwise deletion or balanced-data assumptions."
            )
            rationale.append("The repeated-measures structure is incomplete or unbalanced.")
        elif n_groups <= 1:
            recommended_method = "friedman" if any_non_normal else "rm_anova"
            recommended_engine = "pingouin"
            rationale.append("This is a single-group repeated-time design.")
            if any_non_normal:
                fallback_reason = "Friedman is preferred because at least one time cell failed normality screening."
            elif sphericity and sphericity.get("applies"):
                rationale.append("Sphericity will be checked and Greenhouse-Geisser correction metadata will be returned when needed.")
        else:
            recommended_method = "mixed_anova"
            recommended_engine = "pingouin"
            rationale.append("Balanced group-by-time repeated structure supports Two-way Mixed ANOVA.")
            if any_non_normal:
                fallback_reason = (
                    "A full nonparametric mixed-design omnibus is not available in the current dependency set. "
                    "Mixed ANOVA is still the executable engine for balanced group-by-time designs."
                )
                rationale.append("Normality concerns remain, so the result should be interpreted with caution.")
            elif sphericity and sphericity.get("applies"):
                rationale.append("Sphericity metadata will be evaluated for the within-subject time factor.")

    if any(count < 2 for count in n_per_group.values()) if n_per_group else False:
        rationale.append("Some groups have fewer than two observations, which weakens assumption tests and post-hoc stability.")

    return {
        "recommended_method": recommended_method,
        "recommended_engine": recommended_engine,
        "rationale": rationale,
        "can_override": True,
        "fallback_reason": fallback_reason,
    }


def build_analysis_plan(
    validation_result: dict,
    selector_result: dict,
    method_override: str | None,
) -> dict:
    if validation_result["analysis_status"] == "blocked":
        return {
            "final_method": "blocked",
            "engine": "none",
            "analysis_status": "blocked",
            "rationale": selector_result.get("rationale", []),
            "warnings": validation_result.get("warnings", []),
        }

    final_method = method_override or selector_result["recommended_method"]
    engine = selector_result["recommended_engine"]
    if final_method == "mixedlm":
        engine = "statsmodels"

    return {
        "final_method": final_method,
        "engine": engine,
        "analysis_status": validation_result["analysis_status"],
        "rationale": selector_result.get("rationale", []),
        "warnings": validation_result.get("warnings", []),
    }
