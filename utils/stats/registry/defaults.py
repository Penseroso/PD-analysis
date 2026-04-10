from __future__ import annotations

from utils.stats.registry.effect_sizes import get_default_effect_size_key
from utils.stats.registry.methods import get_engine_for_method, get_method_metadata


DEFAULT_OMNIBUS_METHODS = {
    "cross": "one_way_anova",
    "longitudinal": "rm_anova",
}

DEFAULT_POSTHOC_ASSOCIATIONS = {
    "one_way_anova": "dunnett",
    "welch_anova": "games_howell",
    "kruskal": "mannwhitney_bonferroni",
    "rm_anova": "pairwise_ttests_bonferroni",
    "mixed_anova": "pairwise_tests_bonferroni",
    "friedman": "wilcoxon_bonferroni",
    "mixedlm": "reference_contrasts",
}

MULTIPLICITY_DEFAULTS = {
    "dunnett": "dunnett",
    "games_howell": "games_howell",
    "mannwhitney_bonferroni": "bonferroni",
    "pairwise_ttests_bonferroni": "bonferroni",
    "pairwise_tests_bonferroni": "bonferroni",
    "wilcoxon_bonferroni": "bonferroni",
    "reference_contrasts": None,
}


def get_default_omnibus_method(data_type: str) -> str | None:
    return DEFAULT_OMNIBUS_METHODS.get(data_type)


def get_default_posthoc_method(method_id: str) -> str | None:
    metadata = get_method_metadata(method_id)
    if metadata and metadata.default_posthoc is not None:
        return metadata.default_posthoc
    return DEFAULT_POSTHOC_ASSOCIATIONS.get(method_id)


def get_default_multiplicity_method(posthoc_method: str | None) -> str | None:
    if posthoc_method is None:
        return None
    return MULTIPLICITY_DEFAULTS.get(posthoc_method)


def resolve_engine(method_id: str) -> str:
    return get_engine_for_method(method_id)


def build_effect_size_policy(method_id: str) -> str | None:
    return get_default_effect_size_key(method_id)
