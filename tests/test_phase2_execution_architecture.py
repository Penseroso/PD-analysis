from __future__ import annotations

import pandas as pd

from utils.stats.contracts.analysis_plan import AnalysisPlan
from utils.stats.contracts.analysis_result import AnalysisResult
from utils.stats.contracts.diagnostics import CombinedDiagnosticsSummary
from utils.stats.engines.dispatch import execute_plan
from utils.stats.formatting.result_normalizer import normalize_execution_result
from utils.stats.formatting.star_map import build_star_map
from utils.stats.planning.execution_bridge import execute_and_normalize
from utils.stats.planning.plan_builder import build_analysis_plan_contract
from utils.stats_cross import run_cross_sectional
from utils.stats_longitudinal import run_longitudinal
from utils.stats_mixedlm import run_mixedlm


def _dummy_result(plan: AnalysisPlan) -> AnalysisResult:
    return AnalysisResult(
        analysis_status="ready",
        plan=plan,
        diagnostics=CombinedDiagnosticsSummary(),
        omnibus_table=pd.DataFrame([{"term": "x", "test": plan.omnibus_method, "pvalue": 0.04}]),
        pairwise_table=pd.DataFrame([{"group_a": "A", "group_b": "B", "comparison": "A vs B", "pvalue": 0.04}]),
        model_table=pd.DataFrame([{"term": "Intercept", "beta": 1.0, "pvalue": 0.04}]),
        warnings=[],
        blocking_reasons=[],
        suggested_actions=[],
        metadata={
            "star_map": [{"comparison": "A vs B", "pvalue": 0.04, "label": "*"}],
            "effect_sizes": {"omnibus": pd.DataFrame(), "pairwise": pd.DataFrame()},
            "correction_applied": {"method": "auto"},
            "model_summary": "summary",
            "used_formula": "y ~ x",
            "reference_group_used": "A",
            "time_order": ["T1", "T2"],
        },
    )


def test_cross_dispatch_routes_correct_omnibus_runner_for_each_method(monkeypatch) -> None:
    calls: list[str] = []

    def _omnibus(name: str):
        def runner(*args, **kwargs):
            calls.append(name)
            return {"omnibus_table": pd.DataFrame([{"term": name}]), "warnings": [], "metadata": {"effect_sizes": {"omnibus": pd.DataFrame()}}}

        return runner

    monkeypatch.setattr("utils.stats.engines.cross_omnibus.run_one_way_anova", _omnibus("one_way_anova"))
    monkeypatch.setattr("utils.stats.engines.cross_omnibus.run_welch_anova", _omnibus("welch_anova"))
    monkeypatch.setattr("utils.stats.engines.cross_omnibus.run_kruskal", _omnibus("kruskal"))
    monkeypatch.setattr("utils.stats.engines.cross_posthoc.run_dunnett", lambda *a, **k: {"pairwise_table": pd.DataFrame(), "warnings": [], "metadata": {"effect_sizes": {"pairwise": pd.DataFrame()}}})
    monkeypatch.setattr("utils.stats.engines.cross_posthoc.run_games_howell", lambda *a, **k: {"pairwise_table": pd.DataFrame(), "warnings": [], "metadata": {"effect_sizes": {"pairwise": pd.DataFrame()}}})
    monkeypatch.setattr("utils.stats.engines.cross_posthoc.run_pairwise_mannwhitney", lambda *a, **k: {"pairwise_table": pd.DataFrame(), "warnings": [], "metadata": {"effect_sizes": {"pairwise": pd.DataFrame()}}})

    df = pd.DataFrame({"group": ["A", "B"], "value": [1.0, 2.0]})
    for method in ("one_way_anova", "welch_anova", "kruskal"):
        plan = build_analysis_plan_contract(data_type="cross", omnibus_method=method)
        execute_plan(plan, df=df, dv_col="value", group_col="group")

    assert calls == ["one_way_anova", "welch_anova", "kruskal"]


def test_cross_dispatch_routes_correct_posthoc_runner_for_each_path(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr("utils.stats.engines.cross_omnibus.run_one_way_anova", lambda *a, **k: {"omnibus_table": pd.DataFrame(), "warnings": [], "metadata": {"effect_sizes": {"omnibus": pd.DataFrame()}}})
    monkeypatch.setattr("utils.stats.engines.cross_omnibus.run_welch_anova", lambda *a, **k: {"omnibus_table": pd.DataFrame(), "warnings": [], "metadata": {"effect_sizes": {"omnibus": pd.DataFrame()}}})
    monkeypatch.setattr("utils.stats.engines.cross_omnibus.run_kruskal", lambda *a, **k: {"omnibus_table": pd.DataFrame(), "warnings": [], "metadata": {"effect_sizes": {"omnibus": pd.DataFrame()}}})
    monkeypatch.setattr("utils.stats.engines.cross_posthoc.run_dunnett", lambda *a, **k: calls.append("dunnett") or {"pairwise_table": pd.DataFrame(), "warnings": [], "metadata": {"effect_sizes": {"pairwise": pd.DataFrame()}}})
    monkeypatch.setattr("utils.stats.engines.cross_posthoc.run_games_howell", lambda *a, **k: calls.append("games_howell") or {"pairwise_table": pd.DataFrame(), "warnings": [], "metadata": {"effect_sizes": {"pairwise": pd.DataFrame()}}})
    monkeypatch.setattr("utils.stats.engines.cross_posthoc.run_pairwise_mannwhitney", lambda *a, **k: calls.append("mannwhitney_pairwise") or {"pairwise_table": pd.DataFrame(), "warnings": [], "metadata": {"effect_sizes": {"pairwise": pd.DataFrame()}}})

    df = pd.DataFrame({"group": ["A", "B"], "value": [1.0, 2.0]})
    for method in ("one_way_anova", "welch_anova", "kruskal"):
        plan = build_analysis_plan_contract(data_type="cross", omnibus_method=method)
        execute_plan(plan, df=df, dv_col="value", group_col="group")

    assert calls == ["dunnett", "games_howell", "mannwhitney_pairwise"]


def test_longitudinal_dispatch_routes_correct_omnibus_and_pairwise_runners(monkeypatch) -> None:
    calls: list[str] = []

    monkeypatch.setattr("utils.stats.engines.longitudinal_omnibus.run_rm_anova", lambda *a, **k: calls.append("rm_anova") or {"omnibus_table": pd.DataFrame(), "warnings": [], "metadata": {"effect_sizes": {"omnibus": pd.DataFrame()}}})
    monkeypatch.setattr("utils.stats.engines.longitudinal_omnibus.run_friedman", lambda *a, **k: calls.append("friedman") or {"omnibus_table": pd.DataFrame(), "warnings": [], "metadata": {"effect_sizes": {"omnibus": pd.DataFrame()}}})
    monkeypatch.setattr("utils.stats.engines.longitudinal_omnibus.run_mixed_anova", lambda *a, **k: calls.append("mixed_anova") or {"omnibus_table": pd.DataFrame(), "warnings": [], "metadata": {"effect_sizes": {"omnibus": pd.DataFrame()}}})
    monkeypatch.setattr("utils.stats.engines.longitudinal_posthoc.run_pairwise_time_tests", lambda *a, **k: calls.append("pairwise_time") or {"pairwise_table": pd.DataFrame(), "warnings": [], "metadata": {"effect_sizes": {"pairwise": pd.DataFrame()}}})
    monkeypatch.setattr("utils.stats.engines.longitudinal_posthoc.run_pairwise_wilcoxon", lambda *a, **k: calls.append("pairwise_wilcoxon") or {"pairwise_table": pd.DataFrame(), "warnings": [], "metadata": {"effect_sizes": {"pairwise": pd.DataFrame()}}})
    monkeypatch.setattr("utils.stats.engines.longitudinal_posthoc.run_pairwise_group_at_time_tests", lambda *a, **k: calls.append("pairwise_group_at_time") or {"pairwise_table": pd.DataFrame(), "warnings": [], "metadata": {"effect_sizes": {"pairwise": pd.DataFrame()}}})

    df = pd.DataFrame({"group": ["A", "A"], "subject": ["s1", "s1"], "time": ["T1", "T2"], "value": [1.0, 2.0]})
    for method in ("rm_anova", "friedman", "mixed_anova"):
        plan = build_analysis_plan_contract(data_type="longitudinal", omnibus_method=method)
        execute_plan(plan, df=df, dv_col="value", group_col="group", subject_col="subject", time_col="time")

    assert calls == ["rm_anova", "pairwise_time", "friedman", "pairwise_wilcoxon", "mixed_anova", "pairwise_group_at_time"]


def test_mixedlm_dispatch_returns_normalized_output_with_expected_core_fields(monkeypatch) -> None:
    monkeypatch.setattr(
        "utils.stats.engines.mixedlm.run_mixedlm_engine",
        lambda *a, **k: {
            "analysis_status": "ready",
            "omnibus_table": None,
            "pairwise_table": pd.DataFrame([{"comparison": "A vs B", "pvalue": 0.02}]),
            "model_table": pd.DataFrame([{"term": "Intercept", "beta": 1.0, "pvalue": 0.01}]),
            "warnings": [],
            "blocking_reasons": [],
            "suggested_actions": [],
            "metadata": {"model_summary": "summary", "used_formula": "y ~ x", "reference_group_used": "A", "effect_sizes": pd.DataFrame()},
        },
    )
    plan = build_analysis_plan_contract(data_type="longitudinal", omnibus_method="mixedlm")
    result = execute_and_normalize(
        plan,
        diagnostics=CombinedDiagnosticsSummary(),
        df=pd.DataFrame({"group": ["A"], "subject": ["s1"], "time": ["T1"], "value": [1.0]}),
        dv_col="value",
        group_col="group",
        subject_col="subject",
        time_col="time",
    )

    assert result.analysis_status == "ready"
    assert result.plan.omnibus_method == "mixedlm"
    assert result.pairwise_table is not None
    assert result.model_table is not None
    assert "star_map" in result.metadata


def test_result_normalizer_produces_common_shape_for_cross_longitudinal_and_mixedlm() -> None:
    cross_plan = build_analysis_plan_contract(data_type="cross", omnibus_method="one_way_anova")
    long_plan = build_analysis_plan_contract(data_type="longitudinal", omnibus_method="rm_anova")
    mixed_plan = build_analysis_plan_contract(data_type="longitudinal", omnibus_method="mixedlm")

    cross = normalize_execution_result(
        plan=cross_plan,
        diagnostics=CombinedDiagnosticsSummary(),
        execution_payload={"analysis_status": "ready", "omnibus_table": pd.DataFrame([{"term": "group"}]), "pairwise_table": pd.DataFrame([{"comparison": "A vs B"}]), "model_table": None, "warnings": [], "blocking_reasons": [], "suggested_actions": [], "metadata": {}},
    )
    longitudinal = normalize_execution_result(
        plan=long_plan,
        diagnostics=CombinedDiagnosticsSummary(),
        execution_payload={"analysis_status": "ready", "omnibus_table": pd.DataFrame([{"term": "time"}]), "pairwise_table": pd.DataFrame([{"comparison": "T1 vs T2"}]), "model_table": None, "warnings": [], "blocking_reasons": [], "suggested_actions": [], "metadata": {}},
    )
    mixed = normalize_execution_result(
        plan=mixed_plan,
        diagnostics=CombinedDiagnosticsSummary(),
        execution_payload={"analysis_status": "ready", "omnibus_table": None, "pairwise_table": pd.DataFrame([{"comparison": "A vs B"}]), "model_table": pd.DataFrame([{"term": "Intercept"}]), "warnings": [], "blocking_reasons": [], "suggested_actions": [], "metadata": {}},
    )

    assert cross.omnibus_table is not None and cross.model_table is None
    assert longitudinal.omnibus_table is not None and longitudinal.pairwise_table is not None
    assert mixed.model_table is not None and mixed.pairwise_table is not None


def test_shared_star_map_builder_produces_expected_labels() -> None:
    table = pd.DataFrame(
        [
            {"comparison": "A vs B", "pvalue": 0.04},
            {"comparison": "A vs C", "pvalue": 0.009},
            {"comparison": "A vs D", "pvalue": 0.0009},
            {"comparison": "A vs E", "pvalue": 0.5},
        ]
    )

    labels = [item["label"] for item in build_star_map(table)]

    assert labels == ["*", "**", "***"]


def test_legacy_wrapper_functions_still_expose_expected_top_level_keys(monkeypatch) -> None:
    cross_plan = build_analysis_plan_contract(data_type="cross", omnibus_method="one_way_anova")
    long_plan = build_analysis_plan_contract(data_type="longitudinal", omnibus_method="rm_anova")
    mixed_plan = build_analysis_plan_contract(data_type="longitudinal", omnibus_method="mixedlm")

    monkeypatch.setattr("utils.stats_cross.execute_and_normalize", lambda *a, **k: _dummy_result(cross_plan))
    monkeypatch.setattr("utils.stats_longitudinal.execute_and_normalize", lambda *a, **k: _dummy_result(long_plan))
    monkeypatch.setattr("utils.stats_mixedlm.execute_and_normalize", lambda *a, **k: _dummy_result(mixed_plan))

    cross = run_cross_sectional(
        df=pd.DataFrame({"group": ["A", "A", "B", "B"], "value_x": [1.0, 2.0, 3.0, 4.0]}),
        dv_col="value_x",
        group_col="group",
        control_group="A",
        method="one_way_anova",
    )
    longitudinal = run_longitudinal(
        df=pd.DataFrame({"group": ["A", "A"], "subject": ["s1", "s1"], "time": ["T1", "T2"], "value_x": [1.0, 2.0]}),
        dv_col="value_x",
        group_col="group",
        subject_col="subject",
        time_col="time",
        between_factors=["group"],
        factor2_col=None,
        method="rm_anova",
        time_order=["T1", "T2"],
    )
    mixed = run_mixedlm(
        df=pd.DataFrame({"group": ["A"], "subject": ["s1"], "time": ["T1"], "value_x": [1.0]}),
        dv_col="value_x",
        subject_col="subject",
        time_col="time",
        group_col="group",
        factor2_col=None,
        formula_mode="default",
        reference_group="A",
        time_order=["T1"],
    )

    for key in ("analysis_status", "omnibus", "posthoc_table", "star_map", "effect_sizes", "used_method", "warnings", "blocking_reasons", "suggested_actions", "dv_col"):
        assert key in cross
    for key in ("analysis_status", "omnibus", "posthoc_table", "star_map", "effect_sizes", "used_method", "correction_applied", "engine_used", "warnings", "blocking_reasons", "suggested_actions", "dv_col", "time_order"):
        assert key in longitudinal
    for key in ("analysis_status", "model_summary", "fixed_effects", "contrast_table", "star_map", "effect_sizes", "used_method", "used_formula", "engine_used", "warnings", "blocking_reasons", "suggested_actions", "dv_col", "time_order"):
        assert key in mixed
