from __future__ import annotations


METHOD_TO_ENGINE = {
    "one_way_anova": "pingouin",
    "welch_anova": "pingouin",
    "kruskal": "scipy",
    "rm_anova": "pingouin",
    "mixed_anova": "pingouin",
    "friedman": "pingouin",
    "mixedlm": "statsmodels",
}


LONGITUDINAL_METHOD_RULES = {
    "rm_anova": {
        "requires_single_group": True,
        "requires_complete_repeated": True,
        "max_between_factors": 1,
    },
    "friedman": {
        "requires_single_group": True,
        "requires_complete_repeated": True,
        "max_between_factors": 1,
    },
    "mixed_anova": {
        "requires_multiple_groups": True,
        "requires_complete_repeated": True,
        "max_between_factors": 1,
    },
    "mixedlm": {
        "requires_longitudinal": True,
    },
}


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
    longitudinal_n_per_group = balance_info.get("n_subjects_per_group") or n_per_group
    complete_n_per_group = balance_info.get("n_complete_subjects_per_group") or {}
    n_groups = len(longitudinal_n_per_group) if longitudinal_n_per_group else 0

    if data_type == "cross":
        if any_non_normal:
            recommended_method = "kruskal"
            rationale.append("At least one group failed normality screening, so a rank-based omnibus test is preferred.")
        elif unequal_variance:
            recommended_method = "welch_anova"
            rationale.append("Levene's test suggests unequal variances, so Welch ANOVA is preferred.")
        else:
            recommended_method = "one_way_anova"
            rationale.append("Normality and equal-variance checks support a standard One-way ANOVA.")
    else:
        if len(between_factors) >= 2:
            recommended_method = "mixedlm"
            fallback_reason = (
                "Pingouin mixed_anova() supports only one between-subject factor. "
                "Designs such as group + factor2 + time must use MixedLM."
            )
            rationale.append("Two or more between-subject factors require the MixedLM engine.")
        elif repeated_missing or not is_balanced:
            recommended_method = "mixedlm"
            fallback_reason = (
                "Repeated cells are missing or unbalanced. MixedLM is preferred because RM-ANOVA and Mixed ANOVA would rely on listwise deletion or balanced-data assumptions."
            )
            rationale.append("The repeated-measures structure is incomplete or unbalanced.")
        elif n_groups <= 1:
            recommended_method = "friedman" if any_non_normal else "rm_anova"
            rationale.append("This is a single-group repeated-time design.")
            if any_non_normal:
                fallback_reason = "Friedman is preferred because at least one time cell failed normality screening."
            elif sphericity and sphericity.get("applies"):
                rationale.append("Sphericity will be checked and Greenhouse-Geisser correction metadata will be returned when needed.")
        else:
            recommended_method = "mixed_anova"
            rationale.append("Balanced group-by-time repeated structure supports Two-way Mixed ANOVA.")
            if any_non_normal:
                fallback_reason = (
                    "A full nonparametric mixed-design omnibus is not available in the current dependency set. "
                    "Mixed ANOVA is still the executable engine for balanced group-by-time designs."
                )
                rationale.append("Normality concerns remain, so the result should be interpreted with caution.")
            elif sphericity and sphericity.get("applies"):
                rationale.append("Sphericity metadata will be evaluated for the within-subject time factor.")

        if longitudinal_n_per_group and any(count < 2 for count in longitudinal_n_per_group.values()):
            rationale.append("Some groups have fewer than two unique subjects, which weakens repeated-measures inference.")
        if complete_n_per_group and any(count < 2 for count in complete_n_per_group.values()):
            rationale.append("Complete-case subject counts are small in at least one group.")

    recommended_engine = METHOD_TO_ENGINE[recommended_method]
    if data_type == "cross" and n_per_group and any(count < 2 for count in n_per_group.values()):
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
    warnings = list(validation_result.get("warnings", []))
    rationale = list(selector_result.get("rationale", []))

    if validation_result["analysis_status"] == "blocked":
        return {
            "final_method": "blocked",
            "engine": "none",
            "analysis_status": "blocked",
            "rationale": rationale,
            "warnings": warnings,
            "blocking_reasons": validation_result.get("blocking_reasons", []),
            "selector_method": selector_result.get("recommended_method"),
        }

    final_method = method_override or selector_result["recommended_method"]
    blocking_reasons = _validate_override_compatibility(
        validation_result=validation_result,
        final_method=final_method,
    )
    if blocking_reasons:
        return {
            "final_method": final_method,
            "engine": METHOD_TO_ENGINE.get(final_method, "none"),
            "analysis_status": "blocked",
            "rationale": rationale,
            "warnings": warnings,
            "blocking_reasons": blocking_reasons,
            "selector_method": selector_result.get("recommended_method"),
        }

    if method_override and method_override != selector_result["recommended_method"]:
        warnings.append(
            f"Method override applied: running '{final_method}' instead of selector recommendation '{selector_result['recommended_method']}'."
        )

    return {
        "final_method": final_method,
        "engine": METHOD_TO_ENGINE.get(final_method, "none"),
        "analysis_status": validation_result["analysis_status"],
        "rationale": rationale,
        "warnings": sorted(set(warnings)),
        "blocking_reasons": [],
        "selector_method": selector_result.get("recommended_method"),
    }



def _validate_override_compatibility(validation_result: dict, final_method: str) -> list[str]:
    data_type = validation_result.get("data_type")
    if final_method == "auto" or final_method not in METHOD_TO_ENGINE:
        return []
    if data_type != "longitudinal":
        if final_method in LONGITUDINAL_METHOD_RULES:
            return [f"Method '{final_method}' is only valid for longitudinal designs."]
        return []

    rules = LONGITUDINAL_METHOD_RULES.get(final_method)
    if not rules:
        return []

    balance_info = validation_result.get("balance_info", {})
    n_subjects_per_group = balance_info.get("n_subjects_per_group") or {}
    n_complete_subjects_per_group = balance_info.get("n_complete_subjects_per_group") or {}
    n_groups = len(n_subjects_per_group) if n_subjects_per_group else 0
    repeated_missing = balance_info.get("has_missing_repeated_cells", False)
    is_balanced = balance_info.get("is_balanced", True)
    between_factors = validation_result.get("between_factors", [])

    blocking_reasons: list[str] = []
    if rules.get("requires_single_group") and n_groups > 1:
        blocking_reasons.append(
            f"Method '{final_method}' requires a single-group repeated-measures design, but {n_groups} groups are present."
        )
    if rules.get("requires_multiple_groups") and n_groups <= 1:
        blocking_reasons.append(
            f"Method '{final_method}' requires at least two groups in the longitudinal design."
        )
    if rules.get("requires_complete_repeated") and (repeated_missing or not is_balanced):
        blocking_reasons.append(
            f"Method '{final_method}' requires complete repeated cells for each subject across all expected time levels."
        )
    if rules.get("max_between_factors") is not None and len(between_factors) > rules["max_between_factors"]:
        blocking_reasons.append(
            f"Method '{final_method}' supports at most {rules['max_between_factors']} between-subject factor(s), but {len(between_factors)} were provided."
        )
    if final_method in {"rm_anova", "friedman", "mixed_anova"} and n_complete_subjects_per_group:
        too_small = [group for group, count in n_complete_subjects_per_group.items() if count < 2]
        if too_small:
            blocking_reasons.append(
                f"Method '{final_method}' requires at least 2 complete subjects per group; too few complete subjects were found in: {', '.join(map(str, too_small))}."
            )
    return blocking_reasons
