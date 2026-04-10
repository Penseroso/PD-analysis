from __future__ import annotations

from utils.stats.registry.defaults import (
    build_effect_size_policy,
    get_default_multiplicity_method,
    get_default_posthoc_method,
    resolve_engine,
)


def recommend_omnibus_method(
    *,
    data_type: str,
    normality: dict,
    sphericity: dict | None,
    levene: dict,
    balance_info: dict,
    between_factors: list[str],
    n_per_group: dict,
) -> str:
    any_non_normal = any(not item.get("is_normal", True) for item in normality.values()) if normality else False
    unequal_variance = not levene.get("equal_variance", True) if levene else False
    repeated_missing = balance_info.get("has_missing_repeated_cells", False)
    is_balanced = balance_info.get("is_balanced", True)
    longitudinal_n_per_group = balance_info.get("n_subjects_per_group") or n_per_group
    n_groups = len(longitudinal_n_per_group) if longitudinal_n_per_group else 0

    if data_type == "cross":
        if any_non_normal:
            return "kruskal"
        if unequal_variance:
            return "welch_anova"
        return "one_way_anova"

    if len(between_factors) >= 2:
        return "mixedlm"
    if repeated_missing or not is_balanced:
        return "mixedlm"
    if n_groups <= 1:
        return "friedman" if any_non_normal else "rm_anova"
    return "mixed_anova"


def build_policy_defaults(omnibus_method: str) -> dict:
    posthoc_method = get_default_posthoc_method(omnibus_method)
    return {
        "omnibus_method": omnibus_method,
        "posthoc_method": posthoc_method,
        "multiplicity_method": get_default_multiplicity_method(posthoc_method),
        "engine": resolve_engine(omnibus_method),
        "effect_size_policy": build_effect_size_policy(omnibus_method),
    }
