from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PosthocMetadata:
    id: str
    label: str
    compatible_omnibus_methods: tuple[str, ...]
    compatible_data_types: tuple[str, ...]
    comparison_mode: str | None
    supports_control_group: bool
    default_multiplicity_method: str | None
    execution_family: str
    internally_controls_error: bool = False


POSTHOC_ALIASES = {
    "mannwhitney_bonferroni": "mannwhitney_pairwise",
    "pairwise_ttests_bonferroni": "pairwise_ttests",
    "pairwise_tests_bonferroni": "pairwise_tests",
    "wilcoxon_bonferroni": "pairwise_wilcoxon",
}


POSTHOC_REGISTRY: dict[str, PosthocMetadata] = {
    "dunnett": PosthocMetadata(
        id="dunnett",
        label="Dunnett",
        compatible_omnibus_methods=("one_way_anova",),
        compatible_data_types=("cross",),
        comparison_mode="control_based",
        supports_control_group=True,
        default_multiplicity_method=None,
        execution_family="scipy",
        internally_controls_error=True,
    ),
    "games_howell": PosthocMetadata(
        id="games_howell",
        label="Games-Howell",
        compatible_omnibus_methods=("welch_anova",),
        compatible_data_types=("cross",),
        comparison_mode="all_pairs",
        supports_control_group=False,
        default_multiplicity_method=None,
        execution_family="pingouin",
        internally_controls_error=True,
    ),
    "mannwhitney_pairwise": PosthocMetadata(
        id="mannwhitney_pairwise",
        label="Pairwise Mann-Whitney",
        compatible_omnibus_methods=("kruskal",),
        compatible_data_types=("cross",),
        comparison_mode="all_pairs",
        supports_control_group=False,
        default_multiplicity_method="bonferroni",
        execution_family="scipy",
    ),
    "pairwise_ttests": PosthocMetadata(
        id="pairwise_ttests",
        label="Pairwise t-tests",
        compatible_omnibus_methods=("rm_anova",),
        compatible_data_types=("longitudinal",),
        comparison_mode="all_pairs",
        supports_control_group=False,
        default_multiplicity_method="bonferroni",
        execution_family="pingouin",
    ),
    "pairwise_tests": PosthocMetadata(
        id="pairwise_tests",
        label="Pairwise mixed tests",
        compatible_omnibus_methods=("mixed_anova",),
        compatible_data_types=("longitudinal",),
        comparison_mode="all_pairs",
        supports_control_group=False,
        default_multiplicity_method="bonferroni",
        execution_family="pingouin",
    ),
    "pairwise_wilcoxon": PosthocMetadata(
        id="pairwise_wilcoxon",
        label="Pairwise Wilcoxon",
        compatible_omnibus_methods=("friedman",),
        compatible_data_types=("longitudinal",),
        comparison_mode="all_pairs",
        supports_control_group=False,
        default_multiplicity_method="bonferroni",
        execution_family="scipy",
    ),
    "reference_contrasts": PosthocMetadata(
        id="reference_contrasts",
        label="Reference contrasts",
        compatible_omnibus_methods=("mixedlm",),
        compatible_data_types=("longitudinal",),
        comparison_mode="reference_based",
        supports_control_group=False,
        default_multiplicity_method=None,
        execution_family="statsmodels",
    ),
}


def resolve_posthoc_id(posthoc_id: str | None) -> str | None:
    if posthoc_id is None:
        return None
    return POSTHOC_ALIASES.get(posthoc_id, posthoc_id)


def get_posthoc_metadata(posthoc_id: str | None) -> PosthocMetadata | None:
    resolved = resolve_posthoc_id(posthoc_id)
    if resolved is None:
        return None
    return POSTHOC_REGISTRY.get(resolved)
