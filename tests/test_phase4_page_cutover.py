from __future__ import annotations

from pathlib import Path

import pandas as pd

from utils.stats.contracts.analysis_plan import AnalysisPlan
from utils.stats.contracts.analysis_result import AnalysisResult
from utils.stats.contracts.diagnostics import CombinedDiagnosticsSummary
from utils.stats.planning.page_bridge import run_page_analysis_for_dv
from utils.stats.validation.compatibility import validate_plan_compatibility


def _validation_result(*, data_type: str, balance_info: dict | None = None, n_per_group: dict | None = None) -> dict:
    return {
        "analysis_status": "ready",
        "warnings": [],
        "blocking_reasons": [],
        "suggested_actions": [],
        "balance_info": balance_info or {},
        "n_per_group": n_per_group or {},
        "data_type": data_type,
        "between_factors": ["group"],
        "factor2_col": None,
        "control_group": "A" if data_type == "cross" else None,
        "reference_group": "A" if data_type == "longitudinal" else None,
        "replicate_preserved": False,
        "repeated_structure_info": {"detected": False},
    }


def test_plan_aware_compatibility_validation_passes_valid_cross_plan() -> None:
    plan = AnalysisPlan(
        data_type="cross",
        design_family="cross",
        comparison_mode="control_based",
        omnibus_method="one_way_anova",
        posthoc_method="dunnett",
        multiplicity_method=None,
        engine="pingouin",
        effect_size_policy="omega_squared",
        control_group="A",
    )

    reasons = validate_plan_compatibility(plan, _validation_result(data_type="cross", n_per_group={"A": 4, "B": 4}))

    assert reasons == []


def test_plan_aware_compatibility_blocks_incomplete_repeated_balanced_only_method() -> None:
    plan = AnalysisPlan(
        data_type="longitudinal",
        design_family="longitudinal",
        comparison_mode="all_pairs",
        omnibus_method="mixed_anova",
        posthoc_method="pairwise_tests",
        multiplicity_method="bonferroni",
        engine="pingouin",
        effect_size_policy="partial_eta_squared",
    )

    reasons = validate_plan_compatibility(
        plan,
        _validation_result(
            data_type="longitudinal",
            balance_info={
                "is_balanced": False,
                "has_missing_repeated_cells": True,
                "n_subjects_per_group": {"A": 3, "B": 3},
                "n_complete_subjects_per_group": {"A": 2, "B": 2},
            },
            n_per_group={"A": 3, "B": 3},
        ),
    )

    assert any("requires complete repeated cells" in reason for reason in reasons)


def test_page_bridge_returns_page_consumable_cross_payload(monkeypatch) -> None:
    plan = AnalysisPlan(
        data_type="cross",
        design_family="cross",
        comparison_mode="control_based",
        omnibus_method="one_way_anova",
        posthoc_method="dunnett",
        multiplicity_method=None,
        engine="pingouin",
        effect_size_policy="omega_squared",
        control_group="A",
    )
    result = AnalysisResult(
        analysis_status="ready",
        plan=plan,
        diagnostics=CombinedDiagnosticsSummary(),
        omnibus_table=pd.DataFrame([{"term": "group", "test": "one_way_anova", "pvalue": 0.01}]),
        pairwise_table=pd.DataFrame([{"group_a": "A", "group_b": "B", "comparison": "A vs B", "pvalue": 0.02}]),
        model_table=None,
        warnings=[],
        blocking_reasons=[],
        suggested_actions=[],
        metadata={"star_map": [{"comparison": "A vs B", "pvalue": 0.02, "label": "*"}], "effect_sizes": {"omnibus": pd.DataFrame(), "pairwise": pd.DataFrame()}},
    )
    monkeypatch.setattr(
        "utils.stats.planning.page_bridge.execute_and_normalize",
        lambda *a, **k: result,
    )

    payload = run_page_analysis_for_dv(
        df=pd.DataFrame({"group": ["A", "A", "B", "B"], "value_x": [1.0, 2.0, 3.0, 4.0]}),
        dv_col="value_x",
        dv_label="Value X",
        validation_result=_validation_result(data_type="cross", n_per_group={"A": 4, "B": 4}),
        data_type="cross",
        between_factors=["group"],
        method_override=None,
        control_group="A",
        reference_group=None,
        factor2_col=None,
        time_order=[],
    )

    for key in ("analysis_status", "omnibus", "posthoc_table", "star_map", "effect_sizes", "used_method", "selector", "dv_label"):
        assert key in payload


def test_page_bridge_returns_page_consumable_mixedlm_payload(monkeypatch) -> None:
    plan = AnalysisPlan(
        data_type="longitudinal",
        design_family="longitudinal",
        comparison_mode="reference_based",
        omnibus_method="mixedlm",
        posthoc_method="reference_contrasts",
        multiplicity_method=None,
        engine="statsmodels",
        effect_size_policy="beta_se_ci",
        reference_group="A",
    )
    result = AnalysisResult(
        analysis_status="ready",
        plan=plan,
        diagnostics=CombinedDiagnosticsSummary(),
        omnibus_table=None,
        pairwise_table=pd.DataFrame([{"comparison": "A vs B", "pvalue": 0.02}]),
        model_table=pd.DataFrame([{"term": "Intercept", "beta": 1.0, "pvalue": 0.01}]),
        warnings=[],
        blocking_reasons=[],
        suggested_actions=[],
        metadata={
            "star_map": [{"comparison": "A vs B", "pvalue": 0.02, "label": "*"}],
            "effect_sizes": pd.DataFrame(),
            "model_summary": "summary",
            "used_formula": "y ~ x",
            "reference_group_used": "A",
            "time_order": ["T1", "T2"],
        },
    )
    monkeypatch.setattr("utils.stats.planning.page_bridge.execute_and_normalize", lambda *a, **k: result)

    payload = run_page_analysis_for_dv(
        df=pd.DataFrame({"group": ["A", "B"], "subject": ["s1", "s2"], "time": ["T1", "T1"], "value_x": [1.0, 2.0]}),
        dv_col="value_x",
        dv_label="Value X",
        validation_result=_validation_result(
            data_type="longitudinal",
            balance_info={"is_balanced": False, "has_missing_repeated_cells": True, "n_subjects_per_group": {"A": 1, "B": 1}},
            n_per_group={"A": 1, "B": 1},
        ),
        data_type="longitudinal",
        between_factors=["group", "factor2"],
        method_override="mixedlm",
        control_group=None,
        reference_group="A",
        factor2_col="factor2",
        time_order=["T1", "T2"],
    )

    for key in ("analysis_status", "model_summary", "fixed_effects", "contrast_table", "star_map", "used_method", "selector", "dv_label"):
        assert key in payload


def test_page_no_longer_imports_legacy_runners() -> None:
    page_text = Path("C:/Projects/PD data analysis/pages/02_analysis.py").read_text(encoding="utf-8")

    assert "from utils.stats_cross import" not in page_text
    assert "from utils.stats_longitudinal import" not in page_text
    assert "from utils.stats_mixedlm import" not in page_text
    assert "run_cross_sectional(" not in page_text
    assert "run_longitudinal(" not in page_text
    assert "run_mixedlm(" not in page_text
