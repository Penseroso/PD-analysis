from __future__ import annotations

from io import StringIO
import re

import pandas as pd


def parse_pasted_table(raw_text: str) -> pd.DataFrame | None:
    if not raw_text.strip():
        return None

    separators = [r"\t", r"\s{2,}", r","]
    for sep in separators:
        try:
            df = pd.read_csv(StringIO(raw_text), sep=sep, engine="python")
            if len(df.columns) > 1:
                return df
        except Exception:
            continue
    return pd.DataFrame({"raw": [line for line in raw_text.splitlines() if line.strip()]})


def detect_schema_candidates(raw_df: pd.DataFrame | None) -> dict:
    if raw_df is None or raw_df.empty:
        return {
            "detected_schema": {},
            "confidence": {},
            "warnings": ["No data was parsed from the pasted input."],
            "analysis_status": "blocked",
        }

    columns = list(raw_df.columns)
    numeric_candidates = [col for col in columns if _numeric_success_rate(raw_df[col]) >= 0.8]

    subject_col = _pick_first(columns, ["mouse", "animal", "subject", "id"])
    group_col = _pick_first(columns, ["group", "treat", "cohort"])
    time_col = _pick_first(columns, ["time", "week", "day", "month"])
    factor2_col = _pick_first(columns, ["sex", "strain", "batch"])
    replicate_cols = [col for col in columns if re.match(r"(?i)rep\d+$", str(col))]

    format_type = "long_time" if time_col else "long_single"
    if replicate_cols:
        format_type = "replicate"
    elif _looks_wide_time(columns):
        format_type = "wide_time"

    confidence = {
        "format_type": 0.9 if format_type else 0.0,
        "group": 0.9 if group_col else 0.2,
        "subject": 0.85 if subject_col else 0.2,
        "time": 0.85 if time_col else 0.2,
        "numeric": min(1.0, len(numeric_candidates) / max(len(columns), 1) + 0.2),
    }

    warnings: list[str] = []
    if not numeric_candidates:
        warnings.append("No strong numeric candidate columns were detected.")
    if format_type in {"wide_time", "replicate"}:
        warnings.append("Confirm wide or replicate columns before normalization.")

    analysis_status = "ready" if group_col and numeric_candidates else "needs_user_confirmation"

    return {
        "detected_schema": {
            "format_type": format_type,
            "group_col": group_col,
            "subject_col": subject_col,
            "time_col": time_col,
            "factor2_col": factor2_col,
            "numeric_candidates": numeric_candidates,
            "replicate_cols": replicate_cols,
        },
        "confidence": confidence,
        "warnings": warnings,
        "analysis_status": analysis_status,
    }


def infer_format_type(raw_df: pd.DataFrame) -> dict:
    result = detect_schema_candidates(raw_df)
    detected = result["detected_schema"]
    return {
        "format_type": detected.get("format_type"),
        "confidence_score": result["confidence"].get("format_type", 0.0),
        "candidate_columns": {
            "group": detected.get("group_col"),
            "subject": detected.get("subject_col"),
            "time": detected.get("time_col"),
            "value_cols": detected.get("numeric_candidates", []),
        },
        "warnings": result["warnings"],
    }


def normalize_to_long(
    raw_df: pd.DataFrame,
    column_mapping: dict[str, str | list[str] | None],
    format_type: str,
) -> dict:
    warnings: list[str] = []
    schema_result = detect_schema_candidates(raw_df)
    detected_schema = schema_result["detected_schema"]

    if raw_df is None or raw_df.empty:
        return {
            "normalized_df": None,
            "detected_schema": detected_schema,
            "confidence": {},
            "warnings": ["Raw dataframe is empty."],
            "analysis_status": "blocked",
        }

    group_col = column_mapping.get("group")
    subject_col = column_mapping.get("subject")
    time_col = column_mapping.get("time")
    factor2_col = column_mapping.get("factor2")
    value_cols = column_mapping.get("value_cols") or []
    wide_value_cols = column_mapping.get("wide_value_cols") or value_cols

    base = pd.DataFrame(index=raw_df.index)
    base["group"] = raw_df[group_col].astype(str) if group_col else "all"
    base["subject"] = raw_df[subject_col].astype(str) if subject_col else [f"row_{i}" for i in raw_df.index]
    base["factor2"] = raw_df[factor2_col].astype(str) if factor2_col else pd.NA

    if not group_col:
        warnings.append("No group column selected. Defaulted to a single group.")

    normalized_df: pd.DataFrame | None
    if format_type == "long_single":
        base["time"] = pd.NA
        for idx, value_col in enumerate(value_cols, start=1):
            base[f"value_{idx}"] = pd.to_numeric(raw_df[value_col], errors="coerce")
        normalized_df = base
    elif format_type == "long_time":
        base["time"] = raw_df[time_col].astype(str) if time_col else pd.NA
        for idx, value_col in enumerate(value_cols, start=1):
            base[f"value_{idx}"] = pd.to_numeric(raw_df[value_col], errors="coerce")
        normalized_df = base
    elif format_type == "wide_time":
        id_vars = ["group", "subject", "factor2"]
        wide = base[id_vars].copy()
        for col in wide_value_cols:
            wide[col] = pd.to_numeric(raw_df[col], errors="coerce")
        normalized_df = wide.melt(id_vars=id_vars, value_vars=wide_value_cols, var_name="time", value_name="value_1")
    elif format_type == "replicate":
        base["time"] = pd.NA
        base["value_1"] = raw_df[list(wide_value_cols)].apply(pd.to_numeric, errors="coerce").mean(axis=1)
        normalized_df = base
    else:
        normalized_df = None
        warnings.append("Format type is not supported.")

    analysis_status = "ready" if normalized_df is not None else "blocked"
    if normalized_df is not None and not value_cols and format_type in {"long_single", "long_time"}:
        analysis_status = "needs_user_confirmation"
        warnings.append("No value columns were selected.")

    return {
        "normalized_df": normalized_df,
        "detected_schema": detected_schema,
        "confidence": schema_result["confidence"],
        "warnings": warnings,
        "analysis_status": analysis_status,
    }


def _pick_first(columns: list[str], patterns: list[str]) -> str | None:
    for pattern in patterns:
        for col in columns:
            if pattern in str(col).lower():
                return col
    return None


def _numeric_success_rate(series: pd.Series) -> float:
    coerced = pd.to_numeric(series, errors="coerce")
    return float(coerced.notna().mean())


def _looks_wide_time(columns: list[str]) -> bool:
    time_like = [col for col in columns if re.match(r"(?i)^(w|week|day|d)\d+$", str(col))]
    return len(time_like) >= 2
