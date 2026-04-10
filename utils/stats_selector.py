from __future__ import annotations

from utils.stats.planning.legacy_bridge import legacy_build_analysis_plan, legacy_select_method


def select_method(
    data_type: str,
    normality: dict,
    sphericity: dict | None,
    levene: dict,
    balance_info: dict,
    between_factors: list[str],
    n_per_group: dict,
) -> dict:
    """Compatibility facade for the legacy page-level selector API."""
    return legacy_select_method(
        data_type=data_type,
        normality=normality,
        sphericity=sphericity,
        levene=levene,
        balance_info=balance_info,
        between_factors=between_factors,
        n_per_group=n_per_group,
    )


def build_analysis_plan(
    validation_result: dict,
    selector_result: dict,
    method_override: str | None,
) -> dict:
    """Compatibility facade that returns the legacy plan payload expected by the page."""
    return legacy_build_analysis_plan(
        validation_result=validation_result,
        selector_result=selector_result,
        method_override=method_override,
    )
