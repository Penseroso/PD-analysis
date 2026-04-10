from __future__ import annotations

from utils.stats.registry.methods import get_method_metadata


EFFECT_SIZE_DEFAULTS = {
    "one_way_anova": "omega_squared",
    "welch_anova": "omega_squared",
    "kruskal": "epsilon_squared",
    "two_way_anova": "omega_squared",
    "rm_anova": "partial_eta_squared",
    "mixed_anova": "partial_eta_squared",
    "friedman": "kendalls_w",
    "pairwise_parametric": "hedges_g",
    "pairwise_nonparametric": "rank_biserial",
    "mixedlm": "beta_se_ci",
}


def get_default_effect_size_key(method_id: str) -> str | None:
    metadata = get_method_metadata(method_id)
    if metadata and metadata.default_effect_size_key in EFFECT_SIZE_DEFAULTS:
        return EFFECT_SIZE_DEFAULTS[metadata.default_effect_size_key]
    return EFFECT_SIZE_DEFAULTS.get(method_id)
