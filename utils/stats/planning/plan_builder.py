from __future__ import annotations

from dataclasses import replace

from utils.stats.contracts.analysis_plan import AnalysisPlan
from utils.stats.registry.defaults import (
    build_effect_size_policy,
    get_default_multiplicity_method,
    get_default_posthoc_method,
    resolve_engine,
)


def build_analysis_plan(
    *,
    data_type: str,
    design_family: str,
    omnibus_method: str,
    posthoc_method: str | None = None,
    multiplicity_method: str | None = None,
    factor2_col: str | None = None,
    control_group: str | None = None,
    reference_group: str | None = None,
    warnings: list[str] | None = None,
) -> AnalysisPlan:
    """Construct a normalized AnalysisPlan from explicit planning decisions."""
    resolved_posthoc = posthoc_method if posthoc_method is not None else get_default_posthoc_method(omnibus_method)
    resolved_multiplicity = (
        multiplicity_method if multiplicity_method is not None else get_default_multiplicity_method(resolved_posthoc)
    )
    return AnalysisPlan(
        data_type=data_type,
        design_family=design_family,
        omnibus_method=omnibus_method,
        posthoc_method=resolved_posthoc,
        multiplicity_method=resolved_multiplicity,
        engine=resolve_engine(omnibus_method),
        effect_size_policy=build_effect_size_policy(omnibus_method),
        control_group=control_group,
        reference_group=reference_group,
        factor2_col=factor2_col,
        warnings=sorted(set(warnings or [])),
    )


def build_analysis_plan_contract(
    data_type: str,
    omnibus_method: str,
    factor2_col: str | None = None,
    control_group: str | None = None,
    reference_group: str | None = None,
    warnings: list[str] | None = None,
) -> AnalysisPlan:
    return build_analysis_plan(
        data_type=data_type,
        design_family=data_type,
        omnibus_method=omnibus_method,
        control_group=control_group,
        reference_group=reference_group,
        factor2_col=factor2_col,
        warnings=warnings,
    )


def apply_method_override(
    plan: AnalysisPlan,
    *,
    method_override: str | None,
    control_group: str | None = None,
    reference_group: str | None = None,
    factor2_col: str | None = None,
    warnings: list[str] | None = None,
) -> AnalysisPlan:
    """Translate the legacy method override into a fully resolved AnalysisPlan."""
    if not method_override:
        merged_warnings = sorted(set(plan.warnings + list(warnings or [])))
        return replace(
            plan,
            control_group=control_group if control_group is not None else plan.control_group,
            reference_group=reference_group if reference_group is not None else plan.reference_group,
            factor2_col=factor2_col if factor2_col is not None else plan.factor2_col,
            warnings=merged_warnings,
        )

    return build_analysis_plan(
        data_type=plan.data_type,
        design_family=plan.design_family,
        omnibus_method=method_override,
        factor2_col=factor2_col if factor2_col is not None else plan.factor2_col,
        control_group=control_group if control_group is not None else plan.control_group,
        reference_group=reference_group if reference_group is not None else plan.reference_group,
        warnings=sorted(set(plan.warnings + list(warnings or []))),
    )
