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

    if data_type == "cross":
        if any_non_normal:
            recommended_method = "kruskal"
            recommended_engine = "scipy"
            rationale.append("At least one group failed normality screening.")
        elif unequal_variance:
            recommended_method = "welch_anova"
            recommended_engine = "pingouin"
            rationale.append("Variance heterogeneity favors Welch ANOVA.")
        else:
            recommended_method = "one_way_anova"
            recommended_engine = "pingouin"
            rationale.append("Distribution and variance checks support a parametric one-way comparison.")
    else:
        if len(between_factors) >= 2:
            recommended_method = "mixedlm"
            recommended_engine = "statsmodels"
            fallback_reason = "Pingouin mixed_anova supports only one between-subject factor. Multiple between factors require MixedLM."
            rationale.append("Repeated-measures design with two between factors is routed to MixedLM.")
        elif repeated_missing or not is_balanced:
            recommended_method = "mixedlm"
            recommended_engine = "statsmodels"
            fallback_reason = "Missing or unbalanced repeated measures make ANOVA listwise deletion undesirable."
            rationale.append("Repeated structure is incomplete or unbalanced.")
        elif any_non_normal:
            recommended_method = "mixed_anova"
            recommended_engine = "pingouin"
            rationale.append("Normality issues were detected; nonparametric fallback guidance should be shown to the user.")
        else:
            recommended_method = "mixed_anova"
            recommended_engine = "pingouin"
            if sphericity and sphericity.get("applies", False):
                rationale.append("Sphericity will be handled through auto-correction when needed.")
            else:
                rationale.append("Balanced repeated-measures structure supports ANOVA-based analysis.")

    if any(count < 2 for count in n_per_group.values()) if n_per_group else False:
        rationale.append("Small group sizes reduce reliability of assumption tests and post-hoc comparisons.")

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
