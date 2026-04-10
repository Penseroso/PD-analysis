from __future__ import annotations

from utils.stats.planning.plan_builder import apply_method_override, build_analysis_plan
from utils.stats.planning.policy_defaults import build_policy_defaults, recommend_omnibus_method
from utils.stats.planning.rationale import build_rationale
from utils.stats.validation.compatibility import validate_plan_compatibility


def recommend_analysis_plan(
    *,
    validation_result: dict,
    diagnostics_context: dict,
    method_override: str | None = None,
    control_group: str | None = None,
    reference_group: str | None = None,
    factor2_col: str | None = None,
) -> dict:
    """Recommend, resolve, and validate an AnalysisPlan from diagnostics and legacy inputs."""
    data_type = validation_result.get("data_type", "cross")
    normality = diagnostics_context.get("normality", {})
    sphericity = diagnostics_context.get("sphericity")
    levene = diagnostics_context.get("levene", {})
    balance_info = validation_result.get("balance_info", {})
    between_factors = validation_result.get("between_factors", [])
    n_per_group = validation_result.get("n_per_group", {})
    warnings = list(validation_result.get("warnings", []))

    recommended_method = recommend_omnibus_method(
        data_type=data_type,
        normality=normality,
        sphericity=sphericity,
        levene=levene,
        balance_info=balance_info,
        between_factors=between_factors,
        n_per_group=n_per_group,
    )
    rationale, fallback_reason = build_rationale(
        data_type=data_type,
        selected_method=recommended_method,
        normality=normality,
        sphericity=sphericity,
        levene=levene,
        balance_info=balance_info,
        between_factors=between_factors,
        n_per_group=n_per_group,
    )
    defaults = build_policy_defaults(recommended_method)
    recommended_plan = build_analysis_plan(
        data_type=data_type,
        design_family=data_type,
        omnibus_method=defaults["omnibus_method"],
        posthoc_method=defaults["posthoc_method"],
        multiplicity_method=defaults["multiplicity_method"],
        factor2_col=factor2_col if factor2_col is not None else validation_result.get("factor2_col"),
        control_group=control_group if control_group is not None else validation_result.get("control_group"),
        reference_group=reference_group,
        warnings=warnings,
    )

    resolved_plan = apply_method_override(
        recommended_plan,
        method_override=method_override,
        control_group=control_group,
        reference_group=reference_group,
        factor2_col=factor2_col,
        warnings=[],
    )
    if method_override and method_override != recommended_method:
        warnings.append(
            f"Method override applied: running '{resolved_plan.omnibus_method}' instead of selector recommendation '{recommended_method}'."
        )
        resolved_plan = apply_method_override(
            recommended_plan,
            method_override=method_override,
            control_group=control_group,
            reference_group=reference_group,
            factor2_col=factor2_col,
            warnings=warnings,
        )

    blocking_reasons = validate_plan_compatibility(
        plan=resolved_plan,
        validation_result=validation_result,
        diagnostics_context=diagnostics_context,
    )
    return {
        "recommended_plan": recommended_plan,
        "resolved_plan": resolved_plan,
        "rationale": rationale,
        "fallback_reason": fallback_reason,
        "warnings": sorted(set(resolved_plan.warnings)),
        "blocking_reasons": blocking_reasons,
        "suggested_actions": validation_result.get("suggested_actions", []),
        "selector_metadata": {
            "recommended_method": recommended_method,
            "recommended_engine": recommended_plan.engine,
            "can_override": True,
        },
    }
