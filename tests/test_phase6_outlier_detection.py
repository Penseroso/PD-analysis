from __future__ import annotations

import pandas as pd

from utils.stats.contracts.diagnostics import CombinedDiagnosticsSummary
from utils.stats.diagnostics.outliers import detect_outliers
from utils.stats.planning.execution_bridge import execute_and_normalize
from utils.stats.planning.plan_builder import build_analysis_plan_contract


def test_grubbs_detects_single_extreme_value() -> None:
    df = pd.DataFrame({"group": ["A"] * 6, "value": [10.0, 10.5, 11.0, 11.2, 11.4, 25.0]})

    summary = detect_outliers(df=df, dv_col="value", method="grubbs", group_col="group", subject_col=None)

    assert summary.detected is True
    assert summary.method == "grubbs"
    assert summary.flagged_count == 1
    assert summary.flags[0].value == 25.0
    assert summary.flags[0].pvalue is not None


def test_modified_zscore_flags_mad_based_outlier() -> None:
    df = pd.DataFrame({"group": ["A"] * 6, "value": [10.0, 10.3, 10.4, 10.5, 10.6, 30.0]})

    summary = detect_outliers(df=df, dv_col="value", method="modified_zscore", group_col="group", subject_col=None)

    assert summary.detected is True
    assert summary.flagged_count >= 1
    assert any(flag.value == 30.0 for flag in summary.flags)


def test_iqr_rule_flags_extreme_value() -> None:
    df = pd.DataFrame({"group": ["A"] * 8, "value": [10.0, 10.2, 10.4, 10.5, 10.7, 10.8, 11.0, 25.0]})

    summary = detect_outliers(df=df, dv_col="value", method="iqr", group_col="group", subject_col=None)

    assert summary.detected is True
    assert any(flag.value == 25.0 for flag in summary.flags)


def test_default_analysis_keeps_flagged_values_in_primary_run(monkeypatch) -> None:
    observed_lengths: list[int] = []

    def _execute(plan, **kwargs):
        observed_lengths.append(len(kwargs["df"]))
        return {
            "analysis_status": "ready",
            "omnibus_table": pd.DataFrame([{"term": "group", "test": plan.omnibus_method, "pvalue": 0.04}]),
            "pairwise_table": pd.DataFrame(),
            "model_table": None,
            "warnings": [],
            "blocking_reasons": [],
            "suggested_actions": [],
            "metadata": {},
        }

    monkeypatch.setattr("utils.stats.planning.execution_bridge.execute_plan", _execute)

    result = execute_and_normalize(
        build_analysis_plan_contract(data_type="cross", omnibus_method="one_way_anova"),
        diagnostics=CombinedDiagnosticsSummary(),
        df=pd.DataFrame({"group": ["A"] * 6, "value": [10.0, 10.2, 10.4, 10.5, 10.7, 25.0]}),
        dv_col="value",
        group_col="group",
        outlier_method="iqr",
    )

    assert observed_lengths == [6]
    assert result.metadata["outlier_detected"] is True
    assert result.metadata["flagged_observations_count"] >= 1
    assert result.metadata["outlier_handling"] == "include_all"


def test_compare_both_sensitivity_path_returns_secondary_result(monkeypatch) -> None:
    observed_lengths: list[int] = []

    def _execute(plan, **kwargs):
        observed_lengths.append(len(kwargs["df"]))
        return {
            "analysis_status": "ready",
            "omnibus_table": pd.DataFrame([{"term": "group", "test": plan.omnibus_method, "pvalue": 0.04}]),
            "pairwise_table": pd.DataFrame(),
            "model_table": None,
            "warnings": [],
            "blocking_reasons": [],
            "suggested_actions": [],
            "metadata": {},
        }

    monkeypatch.setattr("utils.stats.planning.execution_bridge.execute_plan", _execute)

    result = execute_and_normalize(
        build_analysis_plan_contract(data_type="cross", omnibus_method="one_way_anova"),
        diagnostics=CombinedDiagnosticsSummary(),
        df=pd.DataFrame({"group": ["A"] * 8, "value": [10.0, 10.2, 10.4, 10.5, 10.7, 10.8, 11.0, 25.0]}),
        dv_col="value",
        group_col="group",
        outlier_method="iqr",
        outlier_handling="compare_both",
    )

    assert sorted(observed_lengths) == [7, 8]
    assert len(observed_lengths) == 2
    assert result.metadata["sensitivity_run_available"] is True
    assert "exclude_flagged_outliers" in result.metadata["sensitivity_analysis"]
