from __future__ import annotations

import pandas as pd


OMNIBUS_COLUMNS = [
    "term",
    "test",
    "statistic",
    "pvalue",
    "df1",
    "df2",
    "ss",
    "ms",
    "effect_estimate",
    "effect_metric",
    "interpretation_basis",
]

PAIRWISE_COLUMNS = [
    "annotation_type",
    "Contrast",
    "time",
    "group",
    "group_a",
    "group_b",
    "time_a",
    "time_b",
    "comparison",
    "test",
    "statistic",
    "pvalue",
    "p_unc",
    "p_adjust",
    "ci_low",
    "ci_high",
    "effect_estimate",
    "effect_metric",
    "interpretation_basis",
]

MODEL_COLUMNS = [
    "term",
    "beta",
    "se",
    "ci_low",
    "ci_high",
    "pvalue",
]


def canonicalize_omnibus_table(table: pd.DataFrame | None) -> pd.DataFrame | None:
    return _canonicalize(table, OMNIBUS_COLUMNS)


def canonicalize_pairwise_table(table: pd.DataFrame | None) -> pd.DataFrame | None:
    return _canonicalize(table, PAIRWISE_COLUMNS)


def canonicalize_model_table(table: pd.DataFrame | None) -> pd.DataFrame | None:
    return _canonicalize(table, MODEL_COLUMNS)


def _canonicalize(table: pd.DataFrame | None, ordered_columns: list[str]) -> pd.DataFrame | None:
    if table is None:
        return None
    if table.empty:
        return pd.DataFrame(columns=ordered_columns)
    formatted = table.copy()
    for column in ordered_columns:
        if column not in formatted.columns:
            formatted[column] = None
    trailing_columns = [column for column in formatted.columns if column not in ordered_columns]
    return formatted[ordered_columns + trailing_columns]
