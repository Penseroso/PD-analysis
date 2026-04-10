from __future__ import annotations

from utils.stats.planning.plan_builder import apply_method_override
from utils.stats.planning.selector import recommend_analysis_plan
from utils.stats.validation.compatibility import validate_plan_compatibility


KEEP_LONG_BLOCKING_REASON = (
    "Inferential analysis is blocked because technical replicates were preserved with keep_long and no explicit replicate model is implemented."
)
KEEP_LONG_SUGGESTED_ACTION = "Rerun normalization with mean or median replicate collapse before running inferential analysis."
REPEATED_STRUCTURE_BLOCKING_REASON = (
    "Cross-sectional inferential analysis is blocked because the normalized dataset contains repeated-measures structure (subject plus time)."
)
REPEATED_STRUCTURE_SUGGESTED_ACTION = "Switch Data type to longitudinal before running inferential analysis."


def legacy_select_method(
    *,
    data_type: str,
    normality: dict,
    sphericity: dict | None,
    levene: dict,
    balance_info: dict,
    between_factors: list[str],
    n_per_group: dict,
) -> dict:
    selection = recommend_analysis_plan(
        validation_result={
            "analysis_status": "ready",
            "warnings": [],
            "blocking_reasons": [],
            "suggested_actions": [],
            "balance_info": balance_info,
            "n_per_group": n_per_group,
            "data_type": data_type,
            "between_factors": between_factors,
            "factor2_col": None,
            "control_group": None,
            "replicate_preserved": False,
            "repeated_structure_info": {"detected": False},
        },
        diagnostics_context={
            "normality": normality,
            "sphericity": sphericity,
            "levene": levene,
        },
    )
    plan = selection["recommended_plan"]
    return {
        "recommended_method": plan.omnibus_method,
        "recommended_engine": plan.engine,
        "recommended_plan": plan,
        "rationale": selection["rationale"],
        "can_override": selection["selector_metadata"]["can_override"],
        "fallback_reason": selection["fallback_reason"],
    }


def legacy_build_analysis_plan(
    *,
    validation_result: dict,
    selector_result: dict,
    method_override: str | None,
) -> dict:
    recommended_plan = selector_result.get("recommended_plan")
    if recommended_plan is None:
        raise ValueError("legacy_build_analysis_plan requires selector_result['recommended_plan'].")
    warnings = list(validation_result.get("warnings", []))
    resolved_plan = apply_method_override(
        recommended_plan,
        method_override=method_override,
        control_group=validation_result.get("control_group"),
        reference_group=validation_result.get("reference_group"),
        factor2_col=validation_result.get("factor2_col"),
        warnings=warnings,
    )
    if method_override and method_override != recommended_plan.omnibus_method:
        warnings.append(
            f"Method override applied: running '{resolved_plan.omnibus_method}' instead of selector recommendation '{recommended_plan.omnibus_method}'."
        )
        resolved_plan = apply_method_override(
            recommended_plan,
            method_override=method_override,
            control_group=validation_result.get("control_group"),
            reference_group=validation_result.get("reference_group"),
            factor2_col=validation_result.get("factor2_col"),
            warnings=warnings,
        )

    if validation_result["analysis_status"] == "blocked":
        return {
            "final_method": "blocked",
            "engine": "none",
            "analysis_status": "blocked",
            "rationale": selector_result.get("rationale", []),
            "warnings": validation_result.get("warnings", []),
            "blocking_reasons": validation_result.get("blocking_reasons", []),
            "suggested_actions": validation_result.get("suggested_actions", []),
            "selector_method": recommended_plan.omnibus_method,
            "plan": recommended_plan,
        }

    blocking_reasons = validate_plan_compatibility(
        plan=resolved_plan,
        validation_result=validation_result,
    )
    suggested_actions = list(validation_result.get("suggested_actions", []))
    if validation_result.get("repeated_structure_info", {}).get("detected") and validation_result.get("data_type") == "cross":
        if REPEATED_STRUCTURE_BLOCKING_REASON not in blocking_reasons:
            blocking_reasons.append(REPEATED_STRUCTURE_BLOCKING_REASON)
        if REPEATED_STRUCTURE_SUGGESTED_ACTION not in suggested_actions:
            suggested_actions.append(REPEATED_STRUCTURE_SUGGESTED_ACTION)
    if validation_result.get("replicate_preserved"):
        if KEEP_LONG_BLOCKING_REASON not in blocking_reasons:
            blocking_reasons.append(KEEP_LONG_BLOCKING_REASON)
        if KEEP_LONG_SUGGESTED_ACTION not in suggested_actions:
            suggested_actions.append(KEEP_LONG_SUGGESTED_ACTION)

    status = "blocked" if blocking_reasons else validation_result.get("analysis_status", "ready")
    return {
        "final_method": resolved_plan.omnibus_method,
        "engine": resolved_plan.engine,
        "analysis_status": status,
        "rationale": selector_result.get("rationale", []),
        "warnings": sorted(set(resolved_plan.warnings)),
        "blocking_reasons": sorted(set(blocking_reasons)),
        "suggested_actions": sorted(set(suggested_actions)),
        "selector_method": recommended_plan.omnibus_method,
        "plan": resolved_plan,
    }
