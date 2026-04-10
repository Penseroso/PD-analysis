from __future__ import annotations

import pandas as pd
from statsmodels.stats.multitest import multipletests

from utils.stats.registry.multiplicity import normalize_multiplicity_method


PINGOUIN_PADJUST_MAP = {
    None: "none",
    "none": "none",
    "bonferroni": "bonf",
    "holm": "holm",
    "fdr_bh": "fdr_bh",
}

MULTIPLETESTS_METHOD_MAP = {
    "bonferroni": "bonferroni",
    "holm": "holm",
    "fdr_bh": "fdr_bh",
}


def apply_multiplicity_to_pairwise_table(table: pd.DataFrame, multiplicity_method: str | None) -> pd.DataFrame:
    normalized = normalize_multiplicity_method(multiplicity_method)
    if table.empty or normalized is None:
        adjusted = table.copy()
        if "p_adjust" not in adjusted.columns:
            adjusted["p_adjust"] = None
        return adjusted

    adjusted = table.copy()
    if "p_unc" not in adjusted.columns:
        adjusted["p_unc"] = adjusted.get("pvalue")
    valid_mask = adjusted["p_unc"].notna()
    if not valid_mask.any():
        adjusted["p_adjust"] = normalized
        return adjusted

    method = MULTIPLETESTS_METHOD_MAP.get(normalized)
    if method is None:
        adjusted["p_adjust"] = normalized
        return adjusted

    pvalues = adjusted.loc[valid_mask, "p_unc"].astype(float).to_numpy()
    corrected = multipletests(pvalues, method=method)[1]
    adjusted.loc[valid_mask, "pvalue"] = corrected
    adjusted["p_adjust"] = normalized
    return adjusted


def resolve_pingouin_padjust(multiplicity_method: str | None) -> str:
    normalized = normalize_multiplicity_method(multiplicity_method)
    return PINGOUIN_PADJUST_MAP.get(normalized, "none")


def build_correction_metadata(*, multiplicity_method: str | None, source: str) -> dict:
    normalized = normalize_multiplicity_method(multiplicity_method)
    return {"method": normalized, "source": source}
