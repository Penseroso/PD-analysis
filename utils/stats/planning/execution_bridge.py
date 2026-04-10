from __future__ import annotations

from utils.stats.contracts.diagnostics import CombinedDiagnosticsSummary
from utils.stats.diagnostics.outliers import detect_outliers, filter_flagged_outliers, update_summary_handling
from utils.stats.engines.dispatch import execute_plan
from utils.stats.formatting.result_normalizer import normalize_execution_result
from utils.stats.formatting.star_map import build_star_map


def execute_and_normalize(
    plan,
    *,
    diagnostics: CombinedDiagnosticsSummary,
    outlier_method: str = "modified_zscore",
    outlier_handling: str = "include_all",
    **execution_kwargs,
):
    """Bridge planning outputs to execution and normalization for legacy wrappers."""
    source_df = execution_kwargs["df"]
    subject_col = execution_kwargs.get("subject_col", "subject")
    group_col = execution_kwargs.get("group_col", "group")
    outlier_summary = detect_outliers(
        df=source_df,
        dv_col=execution_kwargs["dv_col"],
        method=outlier_method,
        group_col=group_col,
        subject_col=subject_col if subject_col in source_df.columns else None,
        handling_mode=outlier_handling,
    )
    diagnostics.outliers = outlier_summary

    execution_df = source_df
    sensitivity_metadata = None
    additional_warnings: list[str] = list(outlier_summary.warnings)
    if outlier_handling in {"exclude_flagged_outliers", "compare_both"} and outlier_summary.flags:
        filtered_df = filter_flagged_outliers(source_df, outlier_summary)
        if plan.data_type == "longitudinal":
            additional_warnings.append(
                "Flagged observations were retained in the primary result. Excluding outliers from repeated-measures data should be treated as a sensitivity check because row removal can unbalance subject-time structure."
            )
        if outlier_handling == "exclude_flagged_outliers":
            execution_df = filtered_df
        else:
            sensitivity_payload = execute_plan(plan, **{**execution_kwargs, "df": filtered_df})
            sensitivity_payload.setdefault("metadata", {})
            sensitivity_payload["metadata"]["star_map"] = build_star_map(sensitivity_payload.get("pairwise_table"))
            sensitivity_metadata = {
                "exclude_flagged_outliers": {
                    "analysis_status": sensitivity_payload.get("analysis_status", "blocked"),
                    "omnibus_table": sensitivity_payload.get("omnibus_table"),
                    "pairwise_table": sensitivity_payload.get("pairwise_table"),
                    "model_table": sensitivity_payload.get("model_table"),
                    "warnings": sensitivity_payload.get("warnings", []),
                    "flagged_rows_excluded": outlier_summary.flagged_count,
                }
            }

    execution_payload = execute_plan(plan, **{**execution_kwargs, "df": execution_df})
    execution_payload.setdefault("metadata", {})
    execution_payload["metadata"]["star_map"] = build_star_map(execution_payload.get("pairwise_table"))
    execution_payload["metadata"]["outlier_summary"] = update_summary_handling(outlier_summary, outlier_handling).to_dict()
    execution_payload["metadata"]["outlier_detected"] = bool(outlier_summary.flags)
    execution_payload["metadata"]["flagged_observations_count"] = outlier_summary.flagged_count
    execution_payload["metadata"]["outlier_detection_method"] = outlier_summary.method
    execution_payload["metadata"]["sensitivity_run_available"] = outlier_summary.sensitivity_run_available
    execution_payload["metadata"]["outlier_handling"] = outlier_handling
    if sensitivity_metadata is not None:
        execution_payload["metadata"]["sensitivity_analysis"] = sensitivity_metadata
    execution_payload["warnings"] = sorted(set(execution_payload.get("warnings", []) + additional_warnings))
    return normalize_execution_result(plan=plan, diagnostics=diagnostics, execution_payload=execution_payload)
