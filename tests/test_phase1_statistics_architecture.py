from __future__ import annotations

import pandas as pd

from utils.stats.contracts.analysis_plan import AnalysisPlan
from utils.stats.contracts.analysis_result import AnalysisResult
from utils.stats.contracts.diagnostics import (
    AssumptionSummary,
    BalanceSummary,
    CombinedDiagnosticsSummary,
    RepeatedStructureSummary,
)
from utils.stats.diagnostics.design_inspector import inspect_repeated_structure, summarize_balance
from utils.stats.registry.methods import get_engine_for_method
from utils.stats.validation.compatibility import validate_resolved_plan_compatibility
from utils.validators import validate_normalized_df


def test_typed_contracts_instantiate_correctly() -> None:
    repeated = RepeatedStructureSummary(detected=True, n_time_levels=3, recommended_data_type="longitudinal")
    balance = BalanceSummary(expected_time_levels=3, n_complete_subjects_per_group={"A": 2})
    assumptions = AssumptionSummary(normality={"A|T1": {"is_normal": True}}, sphericity={"applies": True})
    diagnostics = CombinedDiagnosticsSummary(
        repeated_structure=repeated,
        balance=balance,
        assumptions=assumptions,
        warnings=["preview"],
    )
    plan = AnalysisPlan(
        data_type="longitudinal",
        design_family="longitudinal",
        comparison_mode="all_pairs",
        omnibus_method="rm_anova",
        posthoc_method="pairwise_ttests",
        multiplicity_method="bonferroni",
        engine="pingouin",
        effect_size_policy="partial_eta_squared",
        warnings=["selector"],
    )
    result = AnalysisResult(
        analysis_status="ready",
        plan=plan,
        diagnostics=diagnostics,
        warnings=["result"],
        metadata={"dv_col": "value_a"},
    )

    assert result.plan.omnibus_method == "rm_anova"
    assert result.diagnostics.repeated_structure.detected is True
    assert result.metadata["dv_col"] == "value_a"


def test_design_inspector_detects_repeated_structure_correctly() -> None:
    df = pd.DataFrame(
        {
            "subject": ["s1", "s1", "s2", "s2"],
            "time": ["T1", "T2", "T1", "T2"],
            "group": ["A", "A", "B", "B"],
        }
    )

    summary = inspect_repeated_structure(df)

    assert summary.detected is True
    assert summary.n_time_levels == 2
    assert summary.n_subjects_with_multiple_timepoints == 2
    assert summary.recommended_data_type == "longitudinal"


def test_design_inspector_summarizes_missing_repeated_cells_and_complete_subjects() -> None:
    df = pd.DataFrame(
        {
            "subject": ["s1", "s1", "s2", "s3", "s3", "s4"],
            "time": ["T1", "T2", "T1", "T1", "T2", "T1"],
            "group": ["A", "A", "A", "B", "B", "B"],
        }
    )

    summary = summarize_balance(df=df, subject_col="subject", time_col="time", group_col="group")

    assert summary.expected_time_levels == 2
    assert summary.has_missing_repeated_cells is True
    assert summary.n_complete_subjects_per_group == {"A": 1, "B": 1}
    assert summary.incomplete_subjects_by_group == {"A": ["s2"], "B": ["s4"]}
    assert {"group": "A", "subject": "s2", "time": "T2"} in summary.missing_repeated_cells
    assert {"group": "B", "subject": "s4", "time": "T2"} in summary.missing_repeated_cells


def test_compatibility_validator_blocks_invalid_longitudinal_override_cases() -> None:
    validation_result = {
        "data_type": "longitudinal",
        "between_factors": ["group"],
        "balance_info": {
            "is_balanced": False,
            "has_missing_repeated_cells": True,
            "n_subjects_per_group": {"A": 2, "B": 2},
            "n_complete_subjects_per_group": {"A": 1, "B": 2},
        },
    }

    reasons = validate_resolved_plan_compatibility(validation_result=validation_result, final_method="mixed_anova")

    assert any("requires complete repeated cells" in reason for reason in reasons)
    assert any("at least 2 complete subjects per group" in reason for reason in reasons)


def test_method_registry_returns_expected_engine_per_supported_method() -> None:
    assert get_engine_for_method("one_way_anova") == "pingouin"
    assert get_engine_for_method("kruskal") == "scipy"
    assert get_engine_for_method("mixedlm") == "statsmodels"


def test_legacy_validator_wrapper_still_returns_expected_keys() -> None:
    df = pd.DataFrame(
        {
            "group": ["A", "A", "B", "B"],
            "value_x": [1.0, 2.0, 3.0, 4.0],
        }
    )

    result = validate_normalized_df(
        df=df,
        data_type="cross",
        selected_dv_cols=["value_x"],
        between_factors=["group"],
    )

    for key in (
        "analysis_status",
        "warnings",
        "blocking_reasons",
        "suggested_actions",
        "balance_info",
        "missingness_info",
        "n_per_group",
        "repeated_structure_info",
        "recommended_data_type",
    ):
        assert key in result
    assert result["analysis_status"] == "ready"
