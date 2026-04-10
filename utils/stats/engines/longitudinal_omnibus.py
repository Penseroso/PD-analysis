from __future__ import annotations

import pandas as pd
import pingouin as pg
from scipy import stats


def run_rm_anova(df: pd.DataFrame, dv_col: str, subject_col: str, time_col: str) -> dict:
    anova = pg.rm_anova(
        data=df,
        dv=dv_col,
        within=time_col,
        subject=subject_col,
        correction="auto",
        detailed=True,
        effsize="np2",
    )
    effect_row = anova.loc[anova["Source"] != "Error"].copy()
    omnibus = effect_row.rename(
        columns={
            "Source": "term",
            "F": "statistic",
            "p_unc": "pvalue",
            "DF": "df1",
            "SS": "ss",
            "MS": "ms",
            "np2": "effect_estimate",
        }
    )
    omnibus["df2"] = _safe_float(anova.loc[anova["Source"] == "Error", "DF"].iloc[0]) if (anova["Source"] == "Error").any() else None
    omnibus["test"] = "rm_anova"
    omnibus["effect_metric"] = "partial_eta_squared"
    omnibus["interpretation_basis"] = "rm_anova"
    omnibus = omnibus[["term", "test", "statistic", "pvalue", "df1", "df2", "ss", "ms", "effect_estimate", "effect_metric", "interpretation_basis"]]
    effect_sizes = omnibus[["term", "effect_estimate", "effect_metric", "interpretation_basis"]].copy()
    correction_applied = {
        "method": "auto",
        "sphericity_tested": _bool_or_none(effect_row.get("sphericity")),
        "sphericity_met": _bool_or_none(effect_row.get("sphericity")),
        "greenhouse_geisser_applied": _safe_float(effect_row.get("p_GG_corr")) is not None,
        "pvalue_uncorrected": _safe_float(effect_row.get("p_unc")),
        "pvalue_corrected": _safe_float(effect_row.get("p_GG_corr")),
        "epsilon": _safe_float(effect_row.get("eps")),
        "W": _safe_float(effect_row.get("W_spher")),
        "p_spher": _safe_float(effect_row.get("p_spher")),
    }
    return {
        "omnibus_table": omnibus,
        "warnings": [],
        "metadata": {"effect_sizes": {"omnibus": effect_sizes}, "correction_applied": correction_applied},
    }


def run_friedman(df: pd.DataFrame, dv_col: str, subject_col: str, time_col: str) -> dict:
    wide = df.pivot_table(index=subject_col, columns=time_col, values=dv_col, aggfunc="mean")
    wide = wide.dropna(axis=0, how="any")
    statistic, pvalue = stats.friedmanchisquare(*[wide[column].to_numpy(dtype=float) for column in wide.columns])
    n_subjects = wide.shape[0]
    n_time = wide.shape[1]
    kendalls_w = _safe_float(statistic / (n_subjects * (n_time - 1))) if n_subjects > 0 and n_time > 1 else None
    omnibus = pd.DataFrame(
        [
            {
                "term": time_col,
                "test": "friedman",
                "statistic": _safe_float(statistic),
                "pvalue": _safe_float(pvalue),
                "df1": float(n_time - 1),
                "df2": None,
                "ss": None,
                "ms": None,
                "effect_estimate": kendalls_w,
                "effect_metric": "kendalls_w",
                "interpretation_basis": "friedman",
            }
        ]
    )
    effect_sizes = omnibus[["term", "effect_estimate", "effect_metric", "interpretation_basis"]].copy()
    return {"omnibus_table": omnibus, "warnings": [], "metadata": {"effect_sizes": {"omnibus": effect_sizes}}}


def run_mixed_anova(df: pd.DataFrame, dv_col: str, subject_col: str, time_col: str, group_col: str) -> dict:
    anova = pg.mixed_anova(
        data=df,
        dv=dv_col,
        within=time_col,
        subject=subject_col,
        between=group_col,
        correction="auto",
        effsize="np2",
    )
    omnibus = anova.rename(
        columns={
            "Source": "term",
            "F": "statistic",
            "p_unc": "pvalue",
            "DF1": "df1",
            "DF2": "df2",
            "SS": "ss",
            "MS": "ms",
            "np2": "effect_estimate",
        }
    ).copy()
    omnibus["test"] = "mixed_anova"
    omnibus["effect_metric"] = "partial_eta_squared"
    omnibus["interpretation_basis"] = "mixed_anova"
    if "ms" not in omnibus.columns:
        omnibus["ms"] = None
    omnibus = omnibus[["term", "test", "statistic", "pvalue", "df1", "df2", "ss", "ms", "effect_estimate", "effect_metric", "interpretation_basis"]]
    effect_sizes = omnibus[["term", "effect_estimate", "effect_metric", "interpretation_basis"]].copy()

    time_row = anova.loc[anova["Source"].astype(str).str.lower() == time_col.lower()]
    if time_row.empty:
        time_row = anova.loc[anova["Source"].astype(str).str.lower() == "time"]
    epsilon = _safe_float(time_row.iloc[0].get("eps")) if not time_row.empty else None
    correction_applied = {
        "method": "auto",
        "sphericity_tested": True if epsilon is not None and df[time_col].nunique(dropna=True) >= 3 else False,
        "greenhouse_geisser_applied": epsilon is not None and epsilon < 1.0,
        "epsilon": epsilon,
        "pvalue_corrected": None,
    }
    return {
        "omnibus_table": omnibus,
        "warnings": [],
        "metadata": {"effect_sizes": {"omnibus": effect_sizes}, "correction_applied": correction_applied},
    }


def _safe_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, pd.Series):
        if value.empty:
            return None
        value = value.iloc[0]
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    return float(value)


def _bool_or_none(value: object) -> bool | None:
    if isinstance(value, pd.Series):
        if value.empty:
            return None
        value = value.iloc[0]
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    return bool(value)
