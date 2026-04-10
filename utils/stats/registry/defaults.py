from __future__ import annotations

from utils.stats.registry.effect_sizes import get_default_effect_size_key
from utils.stats.registry.methods import get_engine_for_method, get_method_metadata
from utils.stats.registry.multiplicity import normalize_multiplicity_method
from utils.stats.registry.posthocs import get_posthoc_metadata, resolve_posthoc_id


DEFAULT_OMNIBUS_METHODS = {
    "cross": "one_way_anova",
    "longitudinal": "rm_anova",
}

DEFAULT_POSTHOC_ASSOCIATIONS = {
    "one_way_anova": "tukey_hsd",
    "welch_anova": "games_howell",
    "kruskal": "dunn",
    "two_way_anova": "group_pairwise_by_factor",
    "rm_anova": "pairwise_ttests",
    "mixed_anova": "pairwise_tests",
    "friedman": "pairwise_wilcoxon",
    "mixedlm": "reference_contrasts",
}

MULTIPLICITY_DEFAULTS = {
    "dunnett": None,
    "tukey_hsd": None,
    "games_howell": None,
    "mannwhitney_pairwise": "bonferroni",
    "dunn": "holm",
    "group_pairwise_by_factor": "bonferroni",
    "pairwise_ttests": "bonferroni",
    "pairwise_tests": "bonferroni",
    "pairwise_wilcoxon": "bonferroni",
    "reference_contrasts": None,
}


def get_default_omnibus_method(data_type: str) -> str | None:
    return DEFAULT_OMNIBUS_METHODS.get(data_type)


def get_default_posthoc_method(method_id: str, *, control_group: str | None = None) -> str | None:
    metadata = get_method_metadata(method_id)
    if method_id == "one_way_anova" and control_group is not None:
        return "dunnett"
    if metadata and metadata.default_posthoc is not None:
        return resolve_posthoc_id(metadata.default_posthoc)
    return resolve_posthoc_id(DEFAULT_POSTHOC_ASSOCIATIONS.get(method_id))


def get_default_multiplicity_method(posthoc_method: str | None) -> str | None:
    resolved_posthoc = resolve_posthoc_id(posthoc_method)
    if resolved_posthoc is None:
        return None
    metadata = get_posthoc_metadata(resolved_posthoc)
    if metadata and metadata.default_multiplicity_method is not None:
        return normalize_multiplicity_method(metadata.default_multiplicity_method)
    return normalize_multiplicity_method(MULTIPLICITY_DEFAULTS.get(resolved_posthoc))


def resolve_engine(method_id: str) -> str:
    return get_engine_for_method(method_id)


def build_effect_size_policy(method_id: str) -> str | None:
    return get_default_effect_size_key(method_id)
