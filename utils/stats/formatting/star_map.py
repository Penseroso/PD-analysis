from __future__ import annotations

import pandas as pd


PVALUE_COLUMNS = ("p_corr", "pvalue", "p_unc", "pval")


def build_star_map(table: pd.DataFrame | None) -> list[dict]:
    if table is None or table.empty:
        return []

    pvalue_col = next((column for column in PVALUE_COLUMNS if column in table.columns), None)
    if pvalue_col is None:
        return []

    star_map: list[dict] = []
    for row in table.itertuples(index=False):
        pvalue = getattr(row, pvalue_col, None)
        if pvalue is None or pd.isna(pvalue):
            continue
        label = pvalue_to_label(float(pvalue))
        if label is None:
            continue

        record = {
            "annotation_type": getattr(row, "annotation_type", infer_annotation_type(row)),
            "comparison": getattr(row, "comparison", getattr(row, "contrast", None)),
            "pvalue": float(pvalue),
            "label": label,
        }
        for field in ("time", "time_a", "time_b", "group", "group_a", "group_b", "factor2", "Contrast", "reference_group"):
            if hasattr(row, field):
                record[field] = getattr(row, field)
        star_map.append(record)
    return star_map


def infer_annotation_type(row) -> str:
    if hasattr(row, "annotation_type") and getattr(row, "annotation_type") is not None:
        return getattr(row, "annotation_type")
    if hasattr(row, "time_a") and hasattr(row, "time_b"):
        return "longitudinal_time_pair_within_group"
    if hasattr(row, "time") and hasattr(row, "group_a") and hasattr(row, "group_b"):
        return "longitudinal_group_pair_at_time"
    if hasattr(row, "group_a") and hasattr(row, "group_b"):
        return "cross_group_pair"
    return "cross_group_pair"


def pvalue_to_label(pvalue: float) -> str | None:
    if pvalue < 0.001:
        return "***"
    if pvalue < 0.01:
        return "**"
    if pvalue < 0.05:
        return "*"
    return None
