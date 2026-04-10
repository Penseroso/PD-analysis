from __future__ import annotations

import pandas as pd

from utils.stats.contracts.diagnostics import CombinedDiagnosticsSummary
from utils.stats.formatting.result_normalizer import to_legacy_mixedlm_payload
from utils.stats.planning.execution_bridge import execute_and_normalize
from utils.stats.planning.plan_builder import build_analysis_plan_contract


ANNOTATION_TYPE = "mixedlm_reference_contrast"


def run_mixedlm(
    df: pd.DataFrame,
    dv_col: str,
    subject_col: str,
    time_col: str,
    group_col: str,
    factor2_col: str | None,
    formula_mode: str,
    reference_group: str | None = None,
    time_order: list[str] | None = None,
) -> dict:
    plan = build_analysis_plan_contract(
        data_type="longitudinal",
        omnibus_method="mixedlm",
        factor2_col=factor2_col,
        reference_group=reference_group,
    )
    result = execute_and_normalize(
        plan,
        diagnostics=CombinedDiagnosticsSummary(),
        df=df,
        dv_col=dv_col,
        subject_col=subject_col,
        time_col=time_col,
        group_col=group_col,
        factor2_col=factor2_col,
        formula_mode=formula_mode,
        reference_group=reference_group,
    )
    resolved_time_order = result.metadata.get("time_order")
    if not resolved_time_order:
        resolved_time_order = _resolve_time_order(df, time_col, time_order)
    return to_legacy_mixedlm_payload(result, dv_col=dv_col, time_order=resolved_time_order)


def _resolve_time_order(df: pd.DataFrame, time_col: str, time_order: list[str] | None) -> list[str]:
    explicit = [str(item) for item in (time_order or []) if item is not None]
    observed = df[time_col].astype(str).dropna().drop_duplicates().tolist() if time_col in df.columns else []
    if explicit:
        ordered = [item for item in explicit if item in observed]
        leftovers = [item for item in observed if item not in ordered]
        return ordered + leftovers
    return observed
