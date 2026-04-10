from __future__ import annotations

import pandas as pd

from utils.stats.contracts.diagnostics import AssumptionSummary
from utils.stats_cross import compute_cross_assumptions
from utils.stats_longitudinal import compute_longitudinal_assumptions


def inspect_cross_assumptions(
    df: pd.DataFrame,
    dv_col: str,
    group_col: str,
) -> AssumptionSummary:
    payload = compute_cross_assumptions(df=df, dv_col=dv_col, group_col=group_col)
    return AssumptionSummary(
        normality=payload.get("normality", {}),
        levene=payload.get("levene", {}),
        sphericity=None,
    )


def inspect_longitudinal_assumptions(
    df: pd.DataFrame,
    dv_col: str,
    group_col: str,
    subject_col: str,
    time_col: str,
    between_factors: list[str],
    time_order: list[str] | None = None,
) -> AssumptionSummary:
    payload = compute_longitudinal_assumptions(
        df=df,
        dv_col=dv_col,
        group_col=group_col,
        subject_col=subject_col,
        time_col=time_col,
        between_factors=between_factors,
        time_order=time_order,
    )
    return AssumptionSummary(
        normality=payload.get("normality", {}),
        levene={},
        sphericity=payload.get("sphericity"),
    )
