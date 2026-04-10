from __future__ import annotations

import pandas as pd

from utils.stats.contracts.analysis_plan import AnalysisPlan
from utils.stats.engines.dispatch import execute_plan
from utils.stats.planning.plan_builder import apply_plan_overrides, build_analysis_plan_contract
from utils.stats.registry.multiplicity import get_multiplicity_metadata
from utils.stats.registry.posthocs import get_posthoc_metadata, resolve_posthoc_id
from utils.stats.validation.compatibility import validate_plan_compatibility


def _validation_result(
    *,
    data_type: str,
    between_factors: list[str] | None = None,
    balance_info: dict | None = None,
    control_group: str | None = None,
    reference_group: str | None = None,
) -> dict:
    return {
        "analysis_status": "ready",
        "warnings": [],
        "blocking_reasons": [],
        "suggested_actions": [],
        "balance_info": balance_info or {},
        "n_per_group": {"A": 4, "B": 4} if data_type == "cross" else {"A": 4},
        "data_type": data_type,
        "between_factors": between_factors or (["group"] if data_type == "cross" else []),
        "factor2_col": None,
        "control_group": control_group,
        "reference_group": reference_group,
        "replicate_preserved": False,
        "repeated_structure_info": {"detected": data_type == "longitudinal"},
    }


def test_analysis_plan_round_trip_includes_comparison_posthoc_and_multiplicity() -> None:
    plan = AnalysisPlan(
        data_type="cross",
        design_family="cross",
        comparison_mode="all_pairs",
        omnibus_method="kruskal",
        posthoc_method="mannwhitney_pairwise",
        multiplicity_method="bonferroni",
        engine="scipy",
        effect_size_policy="rank_biserial",
        control_group=None,
        reference_group=None,
        factor2_col=None,
        warnings=["preview"],
    )

    payload = plan.to_dict()

    assert payload["comparison_mode"] == "all_pairs"
    assert payload["posthoc_method"] == "mannwhitney_pairwise"
    assert payload["multiplicity_method"] == "bonferroni"


def test_registry_lookup_exposes_expected_posthoc_and_multiplicity_contracts() -> None:
    dunnett = get_posthoc_metadata("dunnett")
    bonferroni = get_multiplicity_metadata("bonferroni")
    dunn = get_posthoc_metadata("dunn")
    holm = get_multiplicity_metadata("holm")

    assert dunnett is not None and dunnett.comparison_mode == "control_based"
    assert dunnett.default_multiplicity_method is None
    assert bonferroni is not None and "mannwhitney_pairwise" in bonferroni.compatible_posthocs
    assert dunn is not None and dunn.default_multiplicity_method == "holm"
    assert holm is not None and "dunn" in holm.compatible_posthocs


def test_valid_cross_control_based_plan_passes_compatibility() -> None:
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
        reference_group=None,
        factor2_col=None,
        warnings=[],
    )

    reasons = validate_plan_compatibility(plan, _validation_result(data_type="cross", control_group="A"))

    assert reasons == []


def test_valid_cross_all_pairs_plan_passes_compatibility() -> None:
    plan = AnalysisPlan(
        data_type="cross",
        design_family="cross",
        comparison_mode="all_pairs",
        omnibus_method="kruskal",
        posthoc_method="mannwhitney_pairwise",
        multiplicity_method="bonferroni",
        engine="scipy",
        effect_size_policy="rank_biserial",
        control_group=None,
        reference_group=None,
        factor2_col=None,
        warnings=[],
    )

    reasons = validate_plan_compatibility(plan, _validation_result(data_type="cross"))

    assert reasons == []


def test_dunnett_with_all_pairs_mode_is_blocked() -> None:
    plan = AnalysisPlan(
        data_type="cross",
        design_family="cross",
        comparison_mode="all_pairs",
        omnibus_method="one_way_anova",
        posthoc_method="dunnett",
        multiplicity_method=None,
        engine="pingouin",
        effect_size_policy="omega_squared",
        control_group="A",
        reference_group=None,
        factor2_col=None,
        warnings=[],
    )

    reasons = validate_plan_compatibility(plan, _validation_result(data_type="cross", control_group="A"))

    assert "requires comparison_mode='control_based'" in " ".join(reasons)


def test_control_based_comparisons_require_control_group() -> None:
    plan = AnalysisPlan(
        data_type="cross",
        design_family="cross",
        comparison_mode="control_based",
        omnibus_method="one_way_anova",
        posthoc_method="dunnett",
        multiplicity_method=None,
        engine="pingouin",
        effect_size_policy="omega_squared",
        control_group=None,
        reference_group=None,
        factor2_col=None,
        warnings=[],
    )

    reasons = validate_plan_compatibility(plan, _validation_result(data_type="cross"))

    assert "Control-based comparisons require a selected control group." in reasons


def test_internal_error_control_posthoc_rejects_external_multiplicity() -> None:
    plan = AnalysisPlan(
        data_type="cross",
        design_family="cross",
        comparison_mode="all_pairs",
        omnibus_method="welch_anova",
        posthoc_method="games_howell",
        multiplicity_method="bonferroni",
        engine="pingouin",
        effect_size_policy="omega_squared",
        control_group=None,
        reference_group=None,
        factor2_col=None,
        warnings=[],
    )

    reasons = validate_plan_compatibility(plan, _validation_result(data_type="cross"))

    assert any("already controls pairwise error internally" in reason for reason in reasons)


def test_reference_contrasts_require_reference_group() -> None:
    plan = AnalysisPlan(
        data_type="longitudinal",
        design_family="longitudinal",
        comparison_mode="reference_based",
        omnibus_method="mixedlm",
        posthoc_method="reference_contrasts",
        multiplicity_method=None,
        engine="statsmodels",
        effect_size_policy="beta_se_ci",
        control_group=None,
        reference_group=None,
        factor2_col="factor2",
        warnings=[],
    )

    reasons = validate_plan_compatibility(
        plan,
        _validation_result(
            data_type="longitudinal",
            between_factors=["group", "factor2"],
            balance_info={"is_balanced": False, "has_missing_repeated_cells": True, "n_subjects_per_group": {"A": 3, "B": 3}},
        ),
    )

    assert "Reference contrasts require a selected reference group." in reasons


def test_dispatch_passes_multiplicity_method_into_pairwise_execution(monkeypatch) -> None:
    captured: dict[str, str | None] = {}

    monkeypatch.setattr(
        "utils.stats.engines.cross_omnibus.run_kruskal",
        lambda *a, **k: {"omnibus_table": pd.DataFrame(), "warnings": [], "metadata": {"effect_sizes": {"omnibus": pd.DataFrame()}}},
    )

    def _pairwise(*args, multiplicity_method=None, **kwargs):
        captured["multiplicity_method"] = multiplicity_method
        return {"pairwise_table": pd.DataFrame(), "warnings": [], "metadata": {"effect_sizes": {"pairwise": pd.DataFrame()}}}

    monkeypatch.setattr("utils.stats.engines.cross_posthoc.run_dunn", _pairwise)

    plan = build_analysis_plan_contract(data_type="cross", omnibus_method="kruskal")
    execute_plan(
        plan,
        df=pd.DataFrame({"group": ["A", "B"], "value": [1.0, 2.0]}),
        dv_col="value",
        group_col="group",
    )

    assert captured["multiplicity_method"] == "holm"


def test_one_way_anova_with_tukey_hsd_executes() -> None:
    df = pd.DataFrame(
        {
            "group": ["A"] * 4 + ["B"] * 4 + ["C"] * 4,
            "value": [1.0, 1.1, 1.2, 1.3, 2.0, 2.1, 2.2, 2.3, 3.0, 3.1, 3.2, 3.3],
        }
    )
    plan = AnalysisPlan(
        data_type="cross",
        design_family="cross",
        comparison_mode="all_pairs",
        omnibus_method="one_way_anova",
        posthoc_method="tukey_hsd",
        multiplicity_method=None,
        engine="pingouin",
        effect_size_policy="omega_squared",
        control_group=None,
        reference_group=None,
        factor2_col=None,
        warnings=[],
    )

    result = execute_plan(plan, df=df, dv_col="value", group_col="group")

    assert result["omnibus_table"] is not None
    assert result["pairwise_table"] is not None
    assert set(result["pairwise_table"]["test"]) == {"tukey_hsd"}


def test_kruskal_with_dunn_and_holm_executes() -> None:
    df = pd.DataFrame(
        {
            "group": ["A"] * 4 + ["B"] * 4 + ["C"] * 4,
            "value": [1.0, 1.4, 1.9, 2.2, 3.2, 3.7, 4.1, 4.4, 5.1, 5.5, 5.9, 6.2],
        }
    )
    plan = AnalysisPlan(
        data_type="cross",
        design_family="cross",
        comparison_mode="all_pairs",
        omnibus_method="kruskal",
        posthoc_method="dunn",
        multiplicity_method="holm",
        engine="scipy",
        effect_size_policy="epsilon_squared",
        control_group=None,
        reference_group=None,
        factor2_col=None,
        warnings=[],
    )

    result = execute_plan(plan, df=df, dv_col="value", group_col="group")

    assert set(result["pairwise_table"]["test"]) == {"dunn"}
    assert set(result["pairwise_table"]["p_adjust"]) == {"holm"}


def test_kruskal_with_dunn_and_fdr_executes() -> None:
    df = pd.DataFrame(
        {
            "group": ["A"] * 4 + ["B"] * 4 + ["C"] * 4,
            "value": [1.0, 1.4, 1.9, 2.2, 3.2, 3.7, 4.1, 4.4, 5.1, 5.5, 5.9, 6.2],
        }
    )
    plan = AnalysisPlan(
        data_type="cross",
        design_family="cross",
        comparison_mode="all_pairs",
        omnibus_method="kruskal",
        posthoc_method="dunn",
        multiplicity_method="fdr_bh",
        engine="scipy",
        effect_size_policy="epsilon_squared",
        control_group=None,
        reference_group=None,
        factor2_col=None,
        warnings=[],
    )

    result = execute_plan(plan, df=df, dv_col="value", group_col="group")

    assert set(result["pairwise_table"]["p_adjust"]) == {"fdr_bh"}


def test_mannwhitney_pairwise_with_holm_executes() -> None:
    df = pd.DataFrame(
        {
            "group": ["A"] * 4 + ["B"] * 4 + ["C"] * 4,
            "value": [1.0, 1.4, 1.9, 2.2, 3.2, 3.7, 4.1, 4.4, 5.1, 5.5, 5.9, 6.2],
        }
    )
    plan = AnalysisPlan(
        data_type="cross",
        design_family="cross",
        comparison_mode="all_pairs",
        omnibus_method="kruskal",
        posthoc_method="mannwhitney_pairwise",
        multiplicity_method="holm",
        engine="scipy",
        effect_size_policy="epsilon_squared",
        control_group=None,
        reference_group=None,
        factor2_col=None,
        warnings=[],
    )

    result = execute_plan(plan, df=df, dv_col="value", group_col="group")

    assert set(result["pairwise_table"]["p_adjust"]) == {"holm"}


def test_games_howell_with_external_correction_is_blocked() -> None:
    plan = AnalysisPlan(
        data_type="cross",
        design_family="cross",
        comparison_mode="all_pairs",
        omnibus_method="welch_anova",
        posthoc_method="games_howell",
        multiplicity_method="holm",
        engine="pingouin",
        effect_size_policy="omega_squared",
        control_group=None,
        reference_group=None,
        factor2_col=None,
        warnings=[],
    )

    reasons = validate_plan_compatibility(plan, _validation_result(data_type="cross"))

    assert any("already controls pairwise error internally" in reason for reason in reasons)


def test_two_way_anova_with_factor2_executes() -> None:
    df = pd.DataFrame(
        {
            "group": ["A"] * 6 + ["B"] * 6,
            "factor2": ["X"] * 3 + ["Y"] * 3 + ["X"] * 3 + ["Y"] * 3,
            "value": [1.0, 1.1, 1.2, 1.5, 1.6, 1.7, 2.0, 2.1, 2.2, 2.7, 2.8, 2.9],
        }
    )
    plan = AnalysisPlan(
        data_type="cross",
        design_family="cross",
        comparison_mode="all_pairs",
        omnibus_method="two_way_anova",
        posthoc_method="group_pairwise_by_factor",
        multiplicity_method="bonferroni",
        engine="pingouin",
        effect_size_policy="omega_squared",
        control_group=None,
        reference_group=None,
        factor2_col="factor2",
        warnings=[],
    )

    result = execute_plan(plan, df=df, dv_col="value", group_col="group", factor2_col="factor2")

    assert set(result["omnibus_table"]["term"]) == {"group", "factor2", "group * factor2"}
    assert result["pairwise_table"] is not None


def test_two_way_anova_without_factor2_is_blocked() -> None:
    plan = AnalysisPlan(
        data_type="cross",
        design_family="cross",
        comparison_mode="all_pairs",
        omnibus_method="two_way_anova",
        posthoc_method="group_pairwise_by_factor",
        multiplicity_method="bonferroni",
        engine="pingouin",
        effect_size_policy="omega_squared",
        control_group=None,
        reference_group=None,
        factor2_col=None,
        warnings=[],
    )

    reasons = validate_plan_compatibility(
        plan,
        _validation_result(data_type="cross", between_factors=["group", "factor2"]),
    )

    assert "requires a selected factor2 column" in " ".join(reasons)


def test_legacy_posthoc_alias_resolves_to_new_identifier() -> None:
    assert resolve_posthoc_id("mannwhitney_bonferroni") == "mannwhitney_pairwise"
    assert resolve_posthoc_id("pairwise_ttests_bonferroni") == "pairwise_ttests"
    assert resolve_posthoc_id("pairwise_tests_bonferroni") == "pairwise_tests"
    assert resolve_posthoc_id("wilcoxon_bonferroni") == "pairwise_wilcoxon"


def test_apply_plan_overrides_re_normalizes_dependent_defaults() -> None:
    base_plan = build_analysis_plan_contract(data_type="cross", omnibus_method="one_way_anova", control_group="A")

    updated = apply_plan_overrides(base_plan, omnibus_method="kruskal")

    assert updated.omnibus_method == "kruskal"
    assert updated.posthoc_method == "dunn"
    assert updated.multiplicity_method == "holm"
    assert updated.comparison_mode == "all_pairs"
