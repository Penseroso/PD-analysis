from __future__ import annotations

from utils.stats.contracts.analysis_plan import AnalysisPlan
from utils.stats.registry.methods import get_method_metadata


def validate_plan_compatibility(
    plan: AnalysisPlan,
    validation_result: dict,
    diagnostics_context: dict | None = None,
) -> list[str]:
    """Validate a resolved AnalysisPlan against the inspected dataset design."""
    diagnostics_context = diagnostics_context or {}
    data_type = validation_result.get("data_type")

    if plan.data_type != data_type:
        return [f"Plan data type '{plan.data_type}' is not compatible with validated data type '{data_type}'."]

    metadata = get_method_metadata(plan.omnibus_method)
    if metadata is None:
        return []

    if plan.engine != metadata.engine:
        return [f"Plan engine '{plan.engine}' is not valid for omnibus method '{plan.omnibus_method}'."]

    if data_type not in metadata.compatible_data_types:
        return [f"Method '{plan.omnibus_method}' is only valid for {metadata.family} designs."]

    if plan.design_family != data_type:
        return [f"Plan design family '{plan.design_family}' is not compatible with data type '{data_type}'."]

    if data_type != "longitudinal":
        return []

    balance_info = validation_result.get("balance_info", {})
    n_subjects_per_group = balance_info.get("n_subjects_per_group") or {}
    n_complete_subjects_per_group = balance_info.get("n_complete_subjects_per_group") or {}
    n_groups = len(n_subjects_per_group) if n_subjects_per_group else 0
    repeated_missing = balance_info.get("has_missing_repeated_cells", False)
    is_balanced = balance_info.get("is_balanced", True)
    between_factors = validation_result.get("between_factors", [])

    blocking_reasons: list[str] = []
    if metadata.requires_single_group and n_groups > 1:
        blocking_reasons.append(
            f"Method '{plan.omnibus_method}' requires a single-group repeated-measures design, but {n_groups} groups are present."
        )
    if metadata.requires_multiple_groups and n_groups <= 1:
        blocking_reasons.append(
            f"Method '{plan.omnibus_method}' requires at least two groups in the longitudinal design."
        )
    if metadata.requires_complete_repeated and (repeated_missing or not is_balanced):
        blocking_reasons.append(
            f"Method '{plan.omnibus_method}' requires complete repeated cells for each subject across all expected time levels."
        )
    if metadata.max_between_subject_factors is not None and len(between_factors) > metadata.max_between_subject_factors:
        blocking_reasons.append(
            f"Method '{plan.omnibus_method}' supports at most {metadata.max_between_subject_factors} between-subject factor(s), but {len(between_factors)} were provided."
        )
    if plan.omnibus_method in {"rm_anova", "friedman", "mixed_anova"} and n_complete_subjects_per_group:
        too_small = [group for group, count in n_complete_subjects_per_group.items() if count < 2]
        if too_small:
            blocking_reasons.append(
                f"Method '{plan.omnibus_method}' requires at least 2 complete subjects per group; too few complete subjects were found in: {', '.join(map(str, too_small))}."
            )
    return blocking_reasons


def validate_resolved_plan_compatibility(validation_result: dict, final_method: str) -> list[str]:
    if final_method == "auto":
        return []
    metadata = get_method_metadata(final_method)
    if metadata is None:
        return []
    plan = AnalysisPlan(
        data_type=validation_result.get("data_type", metadata.family),
        design_family=validation_result.get("data_type", metadata.family),
        omnibus_method=final_method,
        posthoc_method=metadata.default_posthoc,
        multiplicity_method=None,
        engine=metadata.engine,
        effect_size_policy=metadata.default_effect_size_key,
        control_group=validation_result.get("control_group"),
        reference_group=validation_result.get("reference_group"),
        factor2_col=validation_result.get("factor2_col"),
        warnings=[],
    )
    return validate_plan_compatibility(plan=plan, validation_result=validation_result)
