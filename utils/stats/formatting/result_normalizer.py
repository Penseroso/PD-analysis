from __future__ import annotations

import pandas as pd

from utils.stats.contracts.analysis_result import AnalysisResult
from utils.stats.contracts.diagnostics import CombinedDiagnosticsSummary
from utils.stats.formatting.tables import (
    canonicalize_model_table,
    canonicalize_omnibus_table,
    canonicalize_pairwise_table,
)


def normalize_execution_result(
    *,
    plan,
    diagnostics: CombinedDiagnosticsSummary,
    execution_payload: dict,
) -> AnalysisResult:
    """Convert engine payloads into the canonical AnalysisResult contract."""
    return AnalysisResult(
        analysis_status=execution_payload.get("analysis_status", "blocked"),
        plan=plan,
        diagnostics=diagnostics,
        omnibus_table=canonicalize_omnibus_table(execution_payload.get("omnibus_table")),
        pairwise_table=canonicalize_pairwise_table(execution_payload.get("pairwise_table")),
        model_table=canonicalize_model_table(execution_payload.get("model_table")),
        warnings=sorted(set(execution_payload.get("warnings", []))),
        blocking_reasons=sorted(set(execution_payload.get("blocking_reasons", []))),
        suggested_actions=sorted(set(execution_payload.get("suggested_actions", []))),
        metadata=dict(execution_payload.get("metadata", {})),
    )


def to_legacy_cross_payload(result: AnalysisResult, *, assumptions: dict, dv_col: str) -> dict:
    metadata = result.metadata
    return {
        "analysis_status": result.analysis_status,
        "assumptions": assumptions,
        "omnibus": result.omnibus_table,
        "posthoc_table": result.pairwise_table,
        "star_map": metadata.get("star_map", []),
        "effect_sizes": metadata.get("effect_sizes", {"omnibus": None, "pairwise": None}),
        "used_method": result.plan.omnibus_method,
        "warnings": result.warnings,
        "blocking_reasons": result.blocking_reasons,
        "suggested_actions": result.suggested_actions,
        "dv_col": dv_col,
    }


def to_legacy_longitudinal_payload(
    result: AnalysisResult,
    *,
    assumptions: dict,
    dv_col: str,
    time_order: list[str],
) -> dict:
    metadata = result.metadata
    return {
        "analysis_status": result.analysis_status,
        "assumptions": assumptions,
        "omnibus": result.omnibus_table,
        "posthoc_table": result.pairwise_table,
        "star_map": metadata.get("star_map", []),
        "effect_sizes": metadata.get("effect_sizes", {"omnibus": None, "pairwise": None}),
        "used_method": result.plan.omnibus_method,
        "correction_applied": metadata.get("correction_applied"),
        "engine_used": result.plan.engine,
        "warnings": result.warnings,
        "blocking_reasons": result.blocking_reasons,
        "suggested_actions": result.suggested_actions,
        "dv_col": dv_col,
        "time_order": time_order,
    }


def to_legacy_mixedlm_payload(result: AnalysisResult, *, dv_col: str, time_order: list[str]) -> dict:
    metadata = result.metadata
    return {
        "analysis_status": result.analysis_status,
        "model_summary": metadata.get("model_summary"),
        "fixed_effects": result.model_table,
        "contrast_table": result.pairwise_table,
        "star_map": metadata.get("star_map", []),
        "effect_sizes": metadata.get("effect_sizes"),
        "used_method": result.plan.omnibus_method,
        "used_formula": metadata.get("used_formula", ""),
        "engine_used": result.plan.engine,
        "reference_group_used": metadata.get("reference_group_used"),
        "warnings": result.warnings,
        "blocking_reasons": result.blocking_reasons,
        "suggested_actions": result.suggested_actions,
        "dv_col": dv_col,
        "time_order": time_order,
    }


def empty_table(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)
