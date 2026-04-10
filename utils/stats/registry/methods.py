from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MethodMetadata:
    id: str
    label: str
    family: str
    engine: str
    compatible_data_types: tuple[str, ...]
    analysis_class: str
    requires_repeated_measures: bool
    requires_complete_repeated: bool
    max_between_subject_factors: int | None
    default_posthoc: str | None
    default_effect_size_key: str | None
    requires_single_group: bool = False
    requires_multiple_groups: bool = False


METHOD_REGISTRY: dict[str, MethodMetadata] = {
    "one_way_anova": MethodMetadata(
        id="one_way_anova",
        label="One-way ANOVA",
        family="cross",
        engine="pingouin",
        compatible_data_types=("cross",),
        analysis_class="omnibus",
        requires_repeated_measures=False,
        requires_complete_repeated=False,
        max_between_subject_factors=1,
        default_posthoc="dunnett",
        default_effect_size_key="one_way_anova",
    ),
    "welch_anova": MethodMetadata(
        id="welch_anova",
        label="Welch ANOVA",
        family="cross",
        engine="pingouin",
        compatible_data_types=("cross",),
        analysis_class="omnibus",
        requires_repeated_measures=False,
        requires_complete_repeated=False,
        max_between_subject_factors=1,
        default_posthoc="games_howell",
        default_effect_size_key="welch_anova",
    ),
    "kruskal": MethodMetadata(
        id="kruskal",
        label="Kruskal-Wallis",
        family="cross",
        engine="scipy",
        compatible_data_types=("cross",),
        analysis_class="omnibus",
        requires_repeated_measures=False,
        requires_complete_repeated=False,
        max_between_subject_factors=1,
        default_posthoc="mannwhitney_pairwise",
        default_effect_size_key="kruskal",
    ),
    "rm_anova": MethodMetadata(
        id="rm_anova",
        label="Repeated-measures ANOVA",
        family="longitudinal",
        engine="pingouin",
        compatible_data_types=("longitudinal",),
        analysis_class="omnibus",
        requires_repeated_measures=True,
        requires_complete_repeated=True,
        max_between_subject_factors=1,
        default_posthoc="pairwise_ttests",
        default_effect_size_key="rm_anova",
        requires_single_group=True,
    ),
    "mixed_anova": MethodMetadata(
        id="mixed_anova",
        label="Mixed ANOVA",
        family="longitudinal",
        engine="pingouin",
        compatible_data_types=("longitudinal",),
        analysis_class="omnibus",
        requires_repeated_measures=True,
        requires_complete_repeated=True,
        max_between_subject_factors=1,
        default_posthoc="pairwise_tests",
        default_effect_size_key="mixed_anova",
        requires_multiple_groups=True,
    ),
    "friedman": MethodMetadata(
        id="friedman",
        label="Friedman test",
        family="longitudinal",
        engine="pingouin",
        compatible_data_types=("longitudinal",),
        analysis_class="omnibus",
        requires_repeated_measures=True,
        requires_complete_repeated=True,
        max_between_subject_factors=1,
        default_posthoc="pairwise_wilcoxon",
        default_effect_size_key="friedman",
        requires_single_group=True,
    ),
    "mixedlm": MethodMetadata(
        id="mixedlm",
        label="Mixed Linear Model",
        family="longitudinal",
        engine="statsmodels",
        compatible_data_types=("longitudinal",),
        analysis_class="model",
        requires_repeated_measures=True,
        requires_complete_repeated=False,
        max_between_subject_factors=None,
        default_posthoc="reference_contrasts",
        default_effect_size_key="mixedlm",
    ),
}


def get_method_metadata(method_id: str) -> MethodMetadata | None:
    return METHOD_REGISTRY.get(method_id)


def get_engine_for_method(method_id: str) -> str:
    metadata = get_method_metadata(method_id)
    return metadata.engine if metadata else "none"


def get_supported_methods_for_data_type(data_type: str) -> list[str]:
    return [method_id for method_id, metadata in METHOD_REGISTRY.items() if data_type in metadata.compatible_data_types]
