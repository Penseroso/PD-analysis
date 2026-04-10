from __future__ import annotations

from utils.stats.planning.selector import recommend_analysis_plan
from utils.stats_selector import build_analysis_plan, select_method


def _validation_result(
    *,
    data_type: str,
    between_factors: list[str] | None = None,
    balance_info: dict | None = None,
    n_per_group: dict | None = None,
    warnings: list[str] | None = None,
    blocking_reasons: list[str] | None = None,
    suggested_actions: list[str] | None = None,
    factor2_col: str | None = None,
    control_group: str | None = None,
    replicate_preserved: bool = False,
    repeated_detected: bool = False,
) -> dict:
    return {
        "analysis_status": "ready",
        "warnings": warnings or [],
        "blocking_reasons": blocking_reasons or [],
        "suggested_actions": suggested_actions or [],
        "balance_info": balance_info or {},
        "n_per_group": n_per_group or {},
        "data_type": data_type,
        "between_factors": between_factors or ["group"],
        "factor2_col": factor2_col,
        "control_group": control_group,
        "replicate_preserved": replicate_preserved,
        "repeated_structure_info": {"detected": repeated_detected},
    }


def _diagnostics(*, normality: dict | None = None, levene: dict | None = None, sphericity: dict | None = None) -> dict:
    return {
        "normality": normality or {},
        "levene": levene or {},
        "sphericity": sphericity,
    }


def test_cross_normal_equal_variance_plan() -> None:
    result = recommend_analysis_plan(
        validation_result=_validation_result(data_type="cross", n_per_group={"A": 4, "B": 4}),
        diagnostics_context=_diagnostics(
            normality={"A": {"is_normal": True}, "B": {"is_normal": True}},
            levene={"equal_variance": True},
        ),
    )

    plan = result["recommended_plan"]
    assert plan.design_family == "cross"
    assert plan.omnibus_method == "one_way_anova"
    assert plan.posthoc_method == "tukey_hsd"
    assert plan.comparison_mode == "all_pairs"
    assert plan.multiplicity_method is None
    assert plan.engine == "pingouin"


def test_cross_unequal_variance_plan() -> None:
    result = recommend_analysis_plan(
        validation_result=_validation_result(data_type="cross", n_per_group={"A": 4, "B": 4}),
        diagnostics_context=_diagnostics(
            normality={"A": {"is_normal": True}, "B": {"is_normal": True}},
            levene={"equal_variance": False},
        ),
    )

    plan = result["recommended_plan"]
    assert plan.omnibus_method == "welch_anova"
    assert plan.posthoc_method == "games_howell"
    assert plan.comparison_mode == "all_pairs"
    assert plan.multiplicity_method is None
    assert plan.engine == "pingouin"


def test_cross_non_normal_plan() -> None:
    result = recommend_analysis_plan(
        validation_result=_validation_result(data_type="cross", n_per_group={"A": 4, "B": 4}),
        diagnostics_context=_diagnostics(
            normality={"A": {"is_normal": False}, "B": {"is_normal": True}},
            levene={"equal_variance": True},
        ),
    )

    plan = result["recommended_plan"]
    assert plan.omnibus_method == "kruskal"
    assert plan.posthoc_method == "dunn"
    assert plan.comparison_mode == "all_pairs"
    assert plan.multiplicity_method == "holm"
    assert plan.engine == "scipy"


def test_single_group_repeated_normal_plan() -> None:
    result = recommend_analysis_plan(
        validation_result=_validation_result(
            data_type="longitudinal",
            balance_info={
                "is_balanced": True,
                "has_missing_repeated_cells": False,
                "n_subjects_per_group": {"A": 4},
                "n_complete_subjects_per_group": {"A": 4},
            },
            n_per_group={"A": 4},
        ),
        diagnostics_context=_diagnostics(
            normality={"T1": {"is_normal": True}, "T2": {"is_normal": True}},
            sphericity={"applies": True},
        ),
    )

    plan = result["recommended_plan"]
    assert plan.omnibus_method == "rm_anova"
    assert plan.posthoc_method == "pairwise_ttests"
    assert plan.comparison_mode == "all_pairs"
    assert plan.multiplicity_method == "bonferroni"
    assert plan.engine == "pingouin"


def test_single_group_repeated_non_normal_plan() -> None:
    result = recommend_analysis_plan(
        validation_result=_validation_result(
            data_type="longitudinal",
            balance_info={
                "is_balanced": True,
                "has_missing_repeated_cells": False,
                "n_subjects_per_group": {"A": 4},
                "n_complete_subjects_per_group": {"A": 4},
            },
            n_per_group={"A": 4},
        ),
        diagnostics_context=_diagnostics(
            normality={"T1": {"is_normal": False}, "T2": {"is_normal": True}},
            sphericity={"applies": True},
        ),
    )

    plan = result["recommended_plan"]
    assert plan.omnibus_method == "friedman"
    assert plan.posthoc_method == "pairwise_wilcoxon"
    assert plan.comparison_mode == "all_pairs"
    assert plan.multiplicity_method == "bonferroni"
    assert plan.engine == "pingouin"


def test_balanced_multi_group_repeated_plan() -> None:
    result = recommend_analysis_plan(
        validation_result=_validation_result(
            data_type="longitudinal",
            balance_info={
                "is_balanced": True,
                "has_missing_repeated_cells": False,
                "n_subjects_per_group": {"A": 4, "B": 4},
                "n_complete_subjects_per_group": {"A": 4, "B": 4},
            },
            n_per_group={"A": 4, "B": 4},
        ),
        diagnostics_context=_diagnostics(
            normality={"A|T1": {"is_normal": True}, "B|T1": {"is_normal": True}},
            sphericity={"applies": True},
        ),
    )

    plan = result["recommended_plan"]
    assert plan.omnibus_method == "mixed_anova"
    assert plan.posthoc_method == "pairwise_tests"
    assert plan.comparison_mode == "all_pairs"
    assert plan.multiplicity_method == "bonferroni"
    assert plan.engine == "pingouin"


def test_cross_two_factor_design_prefers_two_way_anova() -> None:
    result = recommend_analysis_plan(
        validation_result=_validation_result(
            data_type="cross",
            between_factors=["group", "factor2"],
            factor2_col="factor2",
            n_per_group={"A": 6, "B": 6},
        ),
        diagnostics_context=_diagnostics(
            normality={"A|X": {"is_normal": True}, "B|X": {"is_normal": True}},
            levene={"equal_variance": True},
        ),
        factor2_col="factor2",
    )

    plan = result["recommended_plan"]
    assert plan.omnibus_method == "two_way_anova"
    assert plan.posthoc_method == "group_pairwise_by_factor"
    assert plan.factor2_col == "factor2"


def test_multifactor_or_incomplete_repeated_plan_uses_mixedlm_with_rationale() -> None:
    multifactor = recommend_analysis_plan(
        validation_result=_validation_result(
            data_type="longitudinal",
            between_factors=["group", "factor2"],
            factor2_col="factor2",
            balance_info={"is_balanced": True, "has_missing_repeated_cells": False, "n_subjects_per_group": {"A": 4, "B": 4}},
            n_per_group={"A": 4, "B": 4},
        ),
        diagnostics_context=_diagnostics(normality={"A|T1": {"is_normal": True}}),
    )
    incomplete = recommend_analysis_plan(
        validation_result=_validation_result(
            data_type="longitudinal",
            balance_info={"is_balanced": False, "has_missing_repeated_cells": True, "n_subjects_per_group": {"A": 4, "B": 4}},
            n_per_group={"A": 4, "B": 4},
        ),
        diagnostics_context=_diagnostics(normality={"A|T1": {"is_normal": True}}),
    )

    assert multifactor["recommended_plan"].engine == "statsmodels"
    assert multifactor["recommended_plan"].omnibus_method == "mixedlm"
    assert any("between-subject factors require the MixedLM engine" in item for item in multifactor["rationale"])
    assert incomplete["recommended_plan"].omnibus_method == "mixedlm"
    assert any("incomplete or unbalanced" in item for item in incomplete["rationale"])


def test_legacy_method_override_translates_into_new_plan_structure_correctly() -> None:
    result = recommend_analysis_plan(
        validation_result=_validation_result(data_type="cross", n_per_group={"A": 4, "B": 4}),
        diagnostics_context=_diagnostics(
            normality={"A": {"is_normal": True}, "B": {"is_normal": True}},
            levene={"equal_variance": True},
        ),
        method_override="kruskal",
        control_group="A",
    )

    plan = result["resolved_plan"]
    assert plan.omnibus_method == "kruskal"
    assert plan.posthoc_method == "dunn"
    assert plan.comparison_mode == "all_pairs"
    assert plan.multiplicity_method == "holm"
    assert plan.engine == "scipy"
    assert plan.control_group == "A"


def test_invalid_override_design_combination_is_blocked_in_planning() -> None:
    validation_result = _validation_result(
        data_type="longitudinal",
        balance_info={
            "is_balanced": True,
            "has_missing_repeated_cells": False,
            "n_subjects_per_group": {"A": 4, "B": 4},
            "n_complete_subjects_per_group": {"A": 4, "B": 4},
        },
        n_per_group={"A": 4, "B": 4},
    )
    selector_result = select_method(
        data_type="longitudinal",
        normality={"A|T1": {"is_normal": True}},
        sphericity={"applies": True},
        levene={},
        balance_info=validation_result["balance_info"],
        between_factors=["group"],
        n_per_group=validation_result["n_per_group"],
    )

    plan_result = build_analysis_plan(validation_result, selector_result, method_override="friedman")

    assert plan_result["analysis_status"] == "blocked"
    assert any("requires a single-group repeated-measures design" in item for item in plan_result["blocking_reasons"])


def test_legacy_selector_wrapper_still_returns_page_expected_keys() -> None:
    result = select_method(
        data_type="cross",
        normality={"A": {"is_normal": True}, "B": {"is_normal": True}},
        sphericity=None,
        levene={"equal_variance": True},
        balance_info={},
        between_factors=["group"],
        n_per_group={"A": 4, "B": 4},
    )

    for key in ("recommended_method", "recommended_engine", "recommended_plan", "resolved_plan", "rationale", "can_override", "fallback_reason", "selector_metadata"):
        assert key in result
    assert result["recommended_method"] == "one_way_anova"
    assert result["recommended_plan"].posthoc_method == "tukey_hsd"
    assert result["selector_metadata"]["effective_comparison_mode"] == "all_pairs"
