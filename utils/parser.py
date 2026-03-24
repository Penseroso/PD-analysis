from __future__ import annotations

from io import StringIO
import re

import pandas as pd


STRUCTURED_SEPARATORS = [
    (r"\t", "tab"),
    (r",", "comma"),
    (r";", "semicolon"),
    (r"\s{2,}", "multi_space"),
]
TIME_HEADER_PATTERNS = [
    re.compile(r"(?i)^baseline$"),
    re.compile(r"(?i)^bl$"),
    re.compile(r"(?i)^pre$"),
    re.compile(r"(?i)^post$"),
    re.compile(r"(?i)^follow\s*up$"),
    re.compile(r"(?i)^d\s*_?-?\s*\d+$"),
    re.compile(r"(?i)^day\s*_?-?\s*\d+$"),
    re.compile(r"(?i)^wk\s*_?-?\s*\d+$"),
    re.compile(r"(?i)^week\s*_?-?\s*\d+$"),
    re.compile(r"(?i)^month\s*_?-?\s*\d+$"),
    re.compile(r"(?i)^m\s*_?-?\s*\d+$"),
    re.compile(r"(?i)^t\s*_?-?\s*\d+$"),
    re.compile(r"(?i)^\d+\s*(h|hr|hrs|hour|hours)$"),
    re.compile(r"(?i)^\d+\s*(m|min|mins|minute|minutes)$"),
]
BASELINE_PATTERN = re.compile(r"(?i)^(baseline|bl|pre)$")
DAY_PATTERN = re.compile(r"(?i)^(?:d|day|t)\s*_?-?\s*(\d+(?:\.\d+)?)$")
WEEK_PATTERN = re.compile(r"(?i)^(?:wk|week)\s*_?-?\s*(\d+(?:\.\d+)?)$")
MONTH_PATTERN = re.compile(r"(?i)^(?:m|month)\s*_?-?\s*(\d+(?:\.\d+)?)$")
HOUR_PATTERN = re.compile(r"(?i)^(\d+(?:\.\d+)?)\s*(h|hr|hrs|hour|hours)$")
MINUTE_PATTERN = re.compile(r"(?i)^(\d+(?:\.\d+)?)\s*(m|min|mins|minute|minutes)$")
GENERIC_NUMBER_PATTERN = re.compile(r"(?i)^(\d+(?:\.\d+)?)$")


KEEP_LONG_BLOCKING_MESSAGE = (
    "Technical replicates were preserved with keep_long. Inferential analysis is blocked to avoid pseudo-replication."
)
KEEP_LONG_SUGGESTED_ACTION = "Rerun normalization with mean or median replicate collapse before running inferential analysis."


def parse_pasted_table(raw_text: str) -> dict:
    if not raw_text.strip():
        return {
            "raw_df": None,
            "structured_df": None,
            "structured_parse_succeeded": False,
            "parse_mode": "empty",
            "warnings": ["No input text was provided."],
            "metadata": {"attempts": []},
            "analysis_status": "blocked",
        }

    attempts: list[dict] = []
    for sep, label in STRUCTURED_SEPARATORS:
        try:
            df = pd.read_csv(StringIO(raw_text), sep=sep, engine="python")
            attempts.append({"separator": label, "columns": int(len(df.columns)), "rows": int(len(df))})
            if len(df.columns) > 1:
                return {
                    "raw_df": df,
                    "structured_df": df.copy(),
                    "structured_parse_succeeded": True,
                    "parse_mode": "structured",
                    "warnings": [],
                    "metadata": {"attempts": attempts, "successful_separator": label},
                    "analysis_status": "ready",
                }
        except Exception as exc:
            attempts.append({"separator": label, "error": str(exc)})

    fallback_df = pd.DataFrame({"raw": [line for line in raw_text.splitlines() if line.strip()]})
    return {
        "raw_df": fallback_df,
        "structured_df": None,
        "structured_parse_succeeded": False,
        "parse_mode": "raw_lines",
        "warnings": [
            "Structured parsing failed. Showing a raw single-column preview only.",
            "Normalization is blocked until the pasted input is parsed into real columns.",
        ],
        "metadata": {"attempts": attempts},
        "analysis_status": "needs_user_confirmation",
    }


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
    replicate_cols = [col for col in columns if re.match(r"(?i)^rep(?:licate)?[_\s-]*\d+$", str(col))]
    wide_time_cols = [col for col in columns if _is_time_like_header(col)]

    format_type = "long_time" if time_col else "long_single"
    if replicate_cols:
        format_type = "replicate"
    elif len(wide_time_cols) >= 2:
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
            "wide_time_cols": wide_time_cols,
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
    replicate_strategy: str = "mean",
) -> dict:
    warnings: list[str] = []
    blocking_reasons: list[str] = []
    suggested_actions: list[str] = []
    schema_result = detect_schema_candidates(raw_df)
    detected_schema = schema_result["detected_schema"]

    if raw_df is None or raw_df.empty:
        return {
            "normalized_df": None,
            "detected_schema": detected_schema,
            "confidence": {},
            "warnings": ["Raw dataframe is empty."],
            "blocking_reasons": ["Normalization requires parsed tabular input."],
            "suggested_actions": ["Paste a table with real columns and parse it again."],
            "analysis_status": "blocked",
            "value_display_map": {},
            "time_order": [],
            "time_order_metadata": {},
            "normalization_metadata": {},
            "replicate_preserved": False,
        }

    group_col = column_mapping.get("group")
    subject_col = column_mapping.get("subject")
    time_col = column_mapping.get("time")
    factor2_col = column_mapping.get("factor2")
    value_cols = _as_existing_columns(raw_df, column_mapping.get("value_cols") or [])
    wide_value_cols = _as_existing_columns(raw_df, column_mapping.get("wide_value_cols") or value_cols)

    missing_mapping_reasons = _validate_required_mappings(
        raw_df=raw_df,
        format_type=format_type,
        group_col=group_col,
        subject_col=subject_col,
        time_col=time_col,
        factor2_col=factor2_col,
        value_cols=value_cols,
        wide_value_cols=wide_value_cols,
    )
    if missing_mapping_reasons:
        blocking_reasons.extend(missing_mapping_reasons)
        suggested_actions.append("Complete the required column mappings for the selected format before normalizing.")

    if format_type == "replicate" and replicate_strategy not in {"mean", "median", "keep_long"}:
        blocking_reasons.append("Replicate strategy must be one of: mean, median, keep_long.")

    if blocking_reasons:
        return {
            "normalized_df": None,
            "detected_schema": detected_schema,
            "confidence": schema_result["confidence"],
            "warnings": warnings,
            "blocking_reasons": blocking_reasons,
            "suggested_actions": suggested_actions,
            "analysis_status": "blocked",
            "value_display_map": {},
            "time_order": [],
            "time_order_metadata": {},
            "normalization_metadata": {},
            "replicate_preserved": False,
        }

    base = pd.DataFrame(index=raw_df.index)
    base["group"] = raw_df[group_col].astype(str) if group_col else "all"
    base["factor2"] = raw_df[factor2_col].astype(str) if factor2_col else pd.NA
    if subject_col:
        base["subject"] = raw_df[subject_col].astype(str)
    elif format_type == "long_single":
        base["subject"] = [f"row_{i}" for i in raw_df.index]
        warnings.append("No subject column was selected for long_single data. Synthetic subject IDs were created for row tracking only.")

    normalized_df: pd.DataFrame | None
    value_display_map: dict[str, str]
    time_order: list[str]
    time_order_metadata: dict
    normalization_metadata = _base_normalization_metadata(format_type, replicate_strategy)
    if format_type == "long_single":
        normalized_df, value_display_map = _normalize_long_single(base, raw_df, value_cols)
        time_order, time_order_metadata = [], {"method": "not_applicable", "reason": "No time variable in long_single format."}
    elif format_type == "long_time":
        normalized_df, value_display_map = _normalize_long_time(base, raw_df, time_col, value_cols)
        time_order, time_order_metadata = infer_time_order(normalized_df["time"].tolist())
    elif format_type == "wide_time":
        normalized_df, value_display_map = _normalize_wide_time(base, raw_df, wide_value_cols)
        time_order, time_order_metadata = infer_time_order(wide_value_cols)
    elif format_type == "replicate":
        normalized_df, value_display_map, replicate_metadata = _normalize_replicates(base, raw_df, wide_value_cols, replicate_strategy)
        normalization_metadata.update(replicate_metadata)
        time_order, time_order_metadata = [], {"method": "not_applicable", "reason": "Replicate normalization does not create a longitudinal time axis."}
    else:
        normalized_df = None
        value_display_map = {}
        time_order, time_order_metadata = [], {}
        warnings.append("Format type is not supported.")

    analysis_status = "ready" if normalized_df is not None else "blocked"
    if time_order_metadata.get("warning"):
        warnings.append(time_order_metadata["warning"])
    if normalization_metadata.get("replicate_preserved"):
        warnings.append(KEEP_LONG_BLOCKING_MESSAGE)
        suggested_actions.append(KEEP_LONG_SUGGESTED_ACTION)
    return {
        "normalized_df": normalized_df,
        "detected_schema": detected_schema,
        "confidence": schema_result["confidence"],
        "warnings": sorted(set(warnings)),
        "blocking_reasons": blocking_reasons,
        "suggested_actions": sorted(set(suggested_actions)),
        "analysis_status": analysis_status,
        "value_display_map": value_display_map,
        "time_order": time_order,
        "time_order_metadata": time_order_metadata,
        "normalization_metadata": normalization_metadata,
        "replicate_preserved": bool(normalization_metadata.get("replicate_preserved", False)),
    }


def infer_time_order(time_labels: list[object]) -> tuple[list[str], dict]:
    first_seen = _first_seen_strings(time_labels)
    if not first_seen:
        return [], {"method": "empty", "reason": "No non-null time labels were available."}

    parsed = [_parse_time_label(label, index) for index, label in enumerate(first_seen)]
    unparsable = [item["label"] for item in parsed if item["family"] == "unknown"]
    non_baseline_families = {item["family"] for item in parsed if item["family"] not in {"baseline"}}

    if not unparsable and len(non_baseline_families) <= 1:
        ordered = [item["label"] for item in sorted(parsed, key=lambda item: (item["sort_rank"], item["value"], item["first_seen_index"]))]
        family = next(iter(non_baseline_families), "baseline")
        return ordered, {
            "method": "numeric_aware",
            "family": family,
            "reason": f"Used parsed {family} ordering with baseline/pre labels first when present.",
            "ambiguous": False,
        }

    warning = "Time labels were ambiguous, so first-seen order was preserved instead of lexical sorting."
    return first_seen, {
        "method": "first_seen",
        "reason": "Mixed or non-parsable time labels prevented reliable numeric-aware ordering.",
        "ambiguous": True,
        "unparsable_labels": unparsable,
        "families_seen": sorted(non_baseline_families),
        "warning": warning,
    }


def _normalize_long_single(base: pd.DataFrame, raw_df: pd.DataFrame, value_cols: list[str]) -> tuple[pd.DataFrame, dict[str, str]]:
    normalized_df = base.copy()
    normalized_df["time"] = pd.NA
    value_display_map: dict[str, str] = {}
    for idx, value_col in enumerate(value_cols, start=1):
        canonical = f"value_{idx}"
        normalized_df[canonical] = pd.to_numeric(raw_df[value_col], errors="coerce")
        value_display_map[canonical] = str(value_col)
    return normalized_df, value_display_map


def _normalize_long_time(base: pd.DataFrame, raw_df: pd.DataFrame, time_col: str, value_cols: list[str]) -> tuple[pd.DataFrame, dict[str, str]]:
    normalized_df = base.copy()
    normalized_df["time"] = raw_df[time_col].astype(str)
    value_display_map: dict[str, str] = {}
    for idx, value_col in enumerate(value_cols, start=1):
        canonical = f"value_{idx}"
        normalized_df[canonical] = pd.to_numeric(raw_df[value_col], errors="coerce")
        value_display_map[canonical] = str(value_col)
    return normalized_df, value_display_map


def _normalize_wide_time(base: pd.DataFrame, raw_df: pd.DataFrame, wide_value_cols: list[str]) -> tuple[pd.DataFrame, dict[str, str]]:
    wide = base[["group", "subject", "factor2"]].copy()
    for col in wide_value_cols:
        wide[col] = pd.to_numeric(raw_df[col], errors="coerce")
    normalized_df = wide.melt(id_vars=["group", "subject", "factor2"], value_vars=wide_value_cols, var_name="time", value_name="value_1")
    return normalized_df, {"value_1": "Wide-format value"}


def _normalize_replicates(
    base: pd.DataFrame,
    raw_df: pd.DataFrame,
    wide_value_cols: list[str],
    replicate_strategy: str,
) -> tuple[pd.DataFrame, dict[str, str], dict]:
    replicate_frame = raw_df[list(wide_value_cols)].apply(pd.to_numeric, errors="coerce")
    if replicate_strategy == "keep_long":
        long_df = base.copy()
        long_df["time"] = pd.NA
        for replicate_col in wide_value_cols:
            long_df[replicate_col] = pd.to_numeric(raw_df[replicate_col], errors="coerce")
        normalized_df = long_df.melt(
            id_vars=["group", "subject", "factor2", "time"],
            value_vars=wide_value_cols,
            var_name="replicate_id",
            value_name="value_1",
        )
        return normalized_df, {"value_1": "Replicate value"}, {
            "replicate_preserved": True,
            "replicate_strategy": "keep_long",
            "replicate_id_col": "replicate_id",
            "inferential_analysis_supported": False,
            "policy": "blocked_inferential",
        }

    collapsed = replicate_frame.mean(axis=1) if replicate_strategy == "mean" else replicate_frame.median(axis=1)
    normalized_df = base.copy()
    normalized_df["time"] = pd.NA
    normalized_df["value_1"] = collapsed
    return normalized_df, {"value_1": f"Replicates ({replicate_strategy})"}, {
        "replicate_preserved": False,
        "replicate_strategy": replicate_strategy,
        "replicate_id_col": None,
        "inferential_analysis_supported": True,
        "policy": "collapsed_before_analysis",
    }


def _validate_required_mappings(
    raw_df: pd.DataFrame,
    format_type: str,
    group_col: str | None,
    subject_col: str | None,
    time_col: str | None,
    factor2_col: str | None,
    value_cols: list[str],
    wide_value_cols: list[str],
) -> list[str]:
    reasons: list[str] = []
    for label, column in (("group", group_col), ("subject", subject_col), ("time", time_col), ("factor2", factor2_col)):
        if column and column not in raw_df.columns:
            reasons.append(f"Selected {label} column '{column}' is not present in the parsed data.")

    if format_type == "long_single":
        if not value_cols:
            reasons.append("long_single normalization requires at least one value column.")
    elif format_type == "long_time":
        if not subject_col:
            reasons.append("long_time normalization requires a subject column.")
        if not time_col:
            reasons.append("long_time normalization requires a time column.")
        if not value_cols:
            reasons.append("long_time normalization requires at least one value column.")
    elif format_type == "wide_time":
        if not subject_col:
            reasons.append("wide_time normalization requires a subject column.")
        if not wide_value_cols:
            reasons.append("wide_time normalization requires at least one wide time-value column.")
    elif format_type == "replicate":
        if not subject_col:
            reasons.append("replicate normalization requires a subject column.")
        if not wide_value_cols:
            reasons.append("replicate normalization requires at least one replicate column.")
    else:
        reasons.append(f"Unsupported format type '{format_type}'.")
    return reasons


def _base_normalization_metadata(format_type: str, replicate_strategy: str) -> dict:
    return {
        "format_type": format_type,
        "replicate_strategy": replicate_strategy,
        "replicate_preserved": False,
        "replicate_id_col": None,
        "inferential_analysis_supported": True,
        "policy": "standard",
    }


def _as_existing_columns(raw_df: pd.DataFrame, columns: list[str]) -> list[str]:
    return [column for column in columns if column in raw_df.columns]


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
    time_like = [col for col in columns if _is_time_like_header(col)]
    return len(time_like) >= 2


def _is_time_like_header(value: object) -> bool:
    label = str(value).strip()
    if not label:
        return False
    normalized = _normalize_time_token(label)
    return any(pattern.match(normalized) for pattern in TIME_HEADER_PATTERNS)


def _first_seen_strings(values: list[object]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value is None or pd.isna(value):
            continue
        label = str(value).strip()
        if not label or label in seen:
            continue
        seen.add(label)
        ordered.append(label)
    return ordered


def _parse_time_label(label: str, first_seen_index: int) -> dict:
    normalized = _normalize_time_token(label)
    family = "unknown"
    sort_rank = 99
    value = float("inf")

    if BASELINE_PATTERN.match(normalized):
        family = "baseline"
        sort_rank = 0
        value = 0.0
    elif match := HOUR_PATTERN.match(normalized):
        family = "hour"
        sort_rank = 1
        value = float(match.group(1))
    elif match := MINUTE_PATTERN.match(normalized):
        family = "minute"
        sort_rank = 1
        value = float(match.group(1))
    elif match := DAY_PATTERN.match(normalized):
        family = "day"
        sort_rank = 1
        value = float(match.group(1))
    elif match := WEEK_PATTERN.match(normalized):
        family = "week"
        sort_rank = 1
        value = float(match.group(1))
    elif match := MONTH_PATTERN.match(normalized):
        family = "month"
        sort_rank = 1
        value = float(match.group(1))
    elif match := GENERIC_NUMBER_PATTERN.match(normalized):
        family = "numeric"
        sort_rank = 1
        value = float(match.group(1))

    return {
        "label": label,
        "normalized": normalized,
        "family": family,
        "sort_rank": sort_rank,
        "value": value,
        "first_seen_index": first_seen_index,
    }


def _normalize_time_token(label: str) -> str:
    return re.sub(r"\s+", " ", str(label).strip().replace("_", " ").replace("-", " ")).strip().lower()
