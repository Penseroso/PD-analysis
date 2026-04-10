from __future__ import annotations

from utils.stats.contracts.diagnostics import CombinedDiagnosticsSummary
from utils.stats.engines.dispatch import execute_plan
from utils.stats.formatting.result_normalizer import normalize_execution_result
from utils.stats.formatting.star_map import build_star_map


def execute_and_normalize(plan, *, diagnostics: CombinedDiagnosticsSummary, **execution_kwargs):
    """Bridge planning outputs to execution and normalization for legacy wrappers."""
    execution_payload = execute_plan(plan, **execution_kwargs)
    execution_payload.setdefault("metadata", {})
    execution_payload["metadata"]["star_map"] = build_star_map(execution_payload.get("pairwise_table"))
    return normalize_execution_result(plan=plan, diagnostics=diagnostics, execution_payload=execution_payload)
