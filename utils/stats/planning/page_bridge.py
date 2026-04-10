from __future__ import annotations

from utils.stats.contracts.diagnostics import BalanceSummary, CombinedDiagnosticsSummary, RepeatedStructureSummary
from utils.stats.diagnostics.assumptions import inspect_cross_assumptions, inspect_longitudinal_assumptions
from utils.stats.formatting.result_normalizer import (
    to_legacy_cross_payload,
    to_legacy_longitudinal_payload,
    to_legacy_mixedlm_payload,
)
from utils.stats.planning.execution_bridge import execute_and_normalize
from utils.stats.planning.selector import recommend_analysis_plan


def preview_plan(
    *,
    df,
    dv_col: str,
    validation_result: dict,
    data_type: str,
    between_factors: list[str],
    method_override: str | None,
    control_group: str | None,
    reference_group: str | None,
    factor2_col: str | None,
    time_order: list[str] | None = None,
) -> dict:
    assumptions_summary = _inspect_assumptions(
        df=df,
        dv_col=dv_col,
        data_type=data_type,
        between_factors=between_factors,
        time_order=time_order,
    )
    diagnostics_context = assumptions_summary.to_dict()
    selection = recommend_analysis_plan(
        validation_result=validation_result,
        diagnostics_context=diagnostics_context,
        method_override=method_override,
        control_group=control_group,
        reference_group=reference_group,
        factor2_col=factor2_col,
    )
    return {"selection": selection, "diagnostics_context": diagnostics_context}


def run_page_analysis_for_dv(
    *,
    df,
    dv_col: str,
    dv_label: str,
    validation_result: dict,
    data_type: str,
    between_factors: list[str],
    method_override: str | None,
    control_group: str | None,
    reference_group: str | None,
    factor2_col: str | None,
    time_order: list[str] | None = None,
) -> dict:
    preview = preview_plan(
        df=df,
        dv_col=dv_col,
        validation_result=validation_result,
        data_type=data_type,
        between_factors=between_factors,
        method_override=method_override,
        control_group=control_group,
        reference_group=reference_group,
        factor2_col=factor2_col,
        time_order=time_order,
    )
    selection = preview["selection"]
    diagnostics_context = preview["diagnostics_context"]
    diagnostics = CombinedDiagnosticsSummary(
        repeated_structure=RepeatedStructureSummary(**validation_result.get("repeated_structure_info", {})),
        balance=BalanceSummary(**validation_result.get("balance_info", {})),
        assumptions=_inspect_assumptions(
            df=df,
            dv_col=dv_col,
            data_type=data_type,
            between_factors=between_factors,
            time_order=time_order,
        ),
        warnings=list(selection.get("warnings", [])),
    )

    if selection["blocking_reasons"] or validation_result.get("analysis_status") == "blocked":
        return _build_blocked_page_payload(
            selection=selection,
            validation_result=validation_result,
            dv_col=dv_col,
            dv_label=dv_label,
            time_order=time_order or [],
        )

    result = execute_and_normalize(
        selection["resolved_plan"],
        diagnostics=diagnostics,
        df=df,
        dv_col=dv_col,
        group_col="group",
        subject_col="subject",
        time_col="time",
        factor2_col=factor2_col,
        control_group=control_group,
        reference_group=reference_group,
        formula_mode="default",
    )
    page_payload = _to_page_payload(
        result=result,
        diagnostics_context=diagnostics_context,
        dv_col=dv_col,
        time_order=time_order or [],
    )
    page_payload["warnings"] = sorted(set(page_payload.get("warnings", []) + selection.get("warnings", [])))
    page_payload["dv_label"] = dv_label
    page_payload["selector"] = {
        "recommended_method": selection["recommended_plan"].omnibus_method,
        "recommended_engine": selection["recommended_plan"].engine,
        "recommended_plan": selection["recommended_plan"],
        "rationale": selection["rationale"],
        "can_override": selection["selector_metadata"]["can_override"],
        "fallback_reason": selection["fallback_reason"],
        "resolved_plan": selection["resolved_plan"],
    }
    return page_payload


def _inspect_assumptions(*, df, dv_col: str, data_type: str, between_factors: list[str], time_order: list[str] | None):
    if data_type == "cross":
        return inspect_cross_assumptions(df=df, dv_col=dv_col, group_col="group")
    return inspect_longitudinal_assumptions(
        df=df,
        dv_col=dv_col,
        group_col="group",
        subject_col="subject",
        time_col="time",
        between_factors=between_factors,
        time_order=time_order,
    )


def _to_page_payload(*, result, diagnostics_context: dict, dv_col: str, time_order: list[str]) -> dict:
    if result.plan.engine == "statsmodels":
        return to_legacy_mixedlm_payload(result, dv_col=dv_col, time_order=result.metadata.get("time_order", time_order))
    if result.plan.data_type == "cross":
        return to_legacy_cross_payload(result, assumptions=diagnostics_context, dv_col=dv_col)
    return to_legacy_longitudinal_payload(
        result,
        assumptions=diagnostics_context,
        dv_col=dv_col,
        time_order=result.metadata.get("time_order", time_order),
    )


def _build_blocked_page_payload(
    *,
    selection: dict,
    validation_result: dict,
    dv_col: str,
    dv_label: str,
    time_order: list[str],
) -> dict:
    blocking_reasons = sorted(set(validation_result.get("blocking_reasons", []) + selection.get("blocking_reasons", [])))
    suggested_actions = sorted(set(validation_result.get("suggested_actions", []) + selection.get("suggested_actions", [])))
    return {
        "analysis_status": "blocked",
        "used_method": selection["resolved_plan"].omnibus_method,
        "warnings": selection.get("warnings", []),
        "blocking_reasons": blocking_reasons,
        "suggested_actions": suggested_actions,
        "star_map": [],
        "omnibus": None,
        "posthoc_table": None,
        "dv_col": dv_col,
        "dv_label": dv_label,
        "selector": {
            "recommended_method": selection["recommended_plan"].omnibus_method,
            "recommended_engine": selection["recommended_plan"].engine,
            "recommended_plan": selection["recommended_plan"],
            "rationale": selection["rationale"],
            "can_override": selection["selector_metadata"]["can_override"],
            "fallback_reason": selection["fallback_reason"],
            "resolved_plan": selection["resolved_plan"],
        },
        "time_order": time_order,
    }
