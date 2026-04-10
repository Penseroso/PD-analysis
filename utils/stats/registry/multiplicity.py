from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MultiplicityMetadata:
    id: str
    label: str
    compatible_posthocs: tuple[str, ...]
    allows_none: bool = False


MULTIPLICITY_REGISTRY: dict[str, MultiplicityMetadata] = {
    "bonferroni": MultiplicityMetadata(
        id="bonferroni",
        label="Bonferroni",
        compatible_posthocs=("mannwhitney_pairwise", "pairwise_ttests", "pairwise_tests", "pairwise_wilcoxon"),
    ),
    "none": MultiplicityMetadata(
        id="none",
        label="No extra correction",
        compatible_posthocs=("dunnett", "games_howell", "reference_contrasts"),
        allows_none=True,
    ),
}


def normalize_multiplicity_method(multiplicity_method: str | None) -> str | None:
    if multiplicity_method in {None, "none"}:
        return None
    return multiplicity_method


def get_multiplicity_metadata(multiplicity_method: str | None) -> MultiplicityMetadata | None:
    normalized = normalize_multiplicity_method(multiplicity_method)
    if normalized is None:
        return MULTIPLICITY_REGISTRY["none"]
    return MULTIPLICITY_REGISTRY.get(normalized)
