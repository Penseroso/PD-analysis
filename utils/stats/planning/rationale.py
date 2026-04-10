from __future__ import annotations


def build_rationale(
    *,
    data_type: str,
    selected_method: str,
    normality: dict,
    sphericity: dict | None,
    levene: dict,
    balance_info: dict,
    between_factors: list[str],
    n_per_group: dict,
) -> tuple[list[str], str | None]:
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
        if selected_method == "two_way_anova":
            rationale.append("Two between-subject factors are present, so a native cross-sectional two-way ANOVA path is preferred.")
            if any_non_normal:
                rationale.append("Normality concerns remain because a robust nonparametric two-way omnibus is not available in the current engine set.")
            elif unequal_variance:
                rationale.append("Variance heterogeneity should be reviewed because a Welch-style two-way alternative is not implemented in the current engine set.")
        elif selected_method == "kruskal":
            rationale.append("At least one group failed normality screening, so a rank-based omnibus test is preferred.")
        elif selected_method == "welch_anova":
            rationale.append("Levene's test suggests unequal variances, so Welch ANOVA is preferred.")
        else:
            rationale.append("Normality and equal-variance checks support a standard One-way ANOVA.")
        if n_per_group and any(count < 2 for count in n_per_group.values()):
            rationale.append("Some groups have fewer than two observations, which weakens assumption tests and post-hoc stability.")
        return rationale, fallback_reason

    if len(between_factors) >= 2:
        rationale.append("Two or more between-subject factors require the MixedLM engine.")
        fallback_reason = (
            "Pingouin mixed_anova() supports only one between-subject factor. "
            "Designs such as group + factor2 + time must use MixedLM."
        )
    elif repeated_missing or not is_balanced:
        rationale.append("The repeated-measures structure is incomplete or unbalanced.")
        fallback_reason = (
            "Repeated cells are missing or unbalanced. MixedLM is preferred because RM-ANOVA and Mixed ANOVA would rely on listwise deletion or balanced-data assumptions."
        )
    elif n_groups <= 1:
        rationale.append("This is a single-group repeated-time design.")
        if selected_method == "friedman":
            fallback_reason = "Friedman is preferred because at least one time cell failed normality screening."
        elif sphericity and sphericity.get("applies"):
            rationale.append("Sphericity will be checked and Greenhouse-Geisser correction metadata will be returned when needed.")
    else:
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
    return rationale, fallback_reason
