from __future__ import annotations

import pandas as pd

from utils.stats.engines import cross_omnibus, cross_posthoc, longitudinal_omnibus, longitudinal_posthoc, mixedlm
from utils.stats.registry.posthocs import resolve_posthoc_id


def execute_plan(
    plan,
    *,
    df: pd.DataFrame,
    dv_col: str,
    group_col: str = "group",
    subject_col: str = "subject",
    time_col: str = "time",
    factor2_col: str | None = None,
    control_group: str | None = None,
    reference_group: str | None = None,
    formula_mode: str = "default",
) -> dict:
    """Execute a fully resolved plan without making new planning decisions."""
    if plan.omnibus_method == "mixedlm":
        return mixedlm.run_mixedlm_engine(
            df=df,
            dv_col=dv_col,
            subject_col=subject_col,
            time_col=time_col,
            group_col=group_col,
            factor2_col=factor2_col,
            formula_mode=formula_mode,
            reference_group=reference_group or plan.reference_group,
        )
    if plan.data_type == "cross":
        return _execute_cross_plan(
            plan,
            df=df,
            dv_col=dv_col,
            group_col=group_col,
            factor2_col=factor2_col or plan.factor2_col,
            control_group=control_group or plan.control_group,
        )
    return _execute_longitudinal_plan(plan, df=df, dv_col=dv_col, group_col=group_col, subject_col=subject_col, time_col=time_col)


def _execute_cross_plan(plan, *, df: pd.DataFrame, dv_col: str, group_col: str, factor2_col: str | None, control_group: str | None) -> dict:
    omnibus_runner = {
        "one_way_anova": cross_omnibus.run_one_way_anova,
        "welch_anova": cross_omnibus.run_welch_anova,
        "kruskal": cross_omnibus.run_kruskal,
        "two_way_anova": lambda frame, dv_name, group_name: cross_omnibus.run_two_way_anova(frame, dv_name, group_name, factor2_col),
    }[plan.omnibus_method]
    posthoc_runner = {
        "dunnett": lambda: cross_posthoc.run_dunnett(df, dv_col, group_col, control_group),
        "tukey_hsd": lambda: cross_posthoc.run_tukey_hsd(df, dv_col, group_col),
        "games_howell": lambda: cross_posthoc.run_games_howell(df, dv_col, group_col),
        "dunn": lambda: cross_posthoc.run_dunn(df, dv_col, group_col, multiplicity_method=plan.multiplicity_method),
        "mannwhitney_pairwise": lambda: cross_posthoc.run_pairwise_mannwhitney(
            df,
            dv_col,
            group_col,
            multiplicity_method=plan.multiplicity_method,
        ),
        "group_pairwise_by_factor": lambda: cross_posthoc.run_group_pairwise_by_factor(
            df,
            dv_col,
            group_col,
            factor2_col,
            multiplicity_method=plan.multiplicity_method,
        ),
        None: lambda: {"pairwise_table": None, "warnings": [], "metadata": {"effect_sizes": {"pairwise": None}}},
    }[resolve_posthoc_id(plan.posthoc_method)]
    return _merge_payloads(omnibus_runner(df, dv_col, group_col), posthoc_runner())


def _execute_longitudinal_plan(plan, *, df: pd.DataFrame, dv_col: str, group_col: str, subject_col: str, time_col: str) -> dict:
    omnibus_runner = {
        "rm_anova": lambda: longitudinal_omnibus.run_rm_anova(df, dv_col, subject_col, time_col),
        "friedman": lambda: longitudinal_omnibus.run_friedman(df, dv_col, subject_col, time_col),
        "mixed_anova": lambda: longitudinal_omnibus.run_mixed_anova(df, dv_col, subject_col, time_col, group_col),
    }[plan.omnibus_method]
    posthoc_runner = {
        "pairwise_ttests": lambda: longitudinal_posthoc.run_pairwise_time_tests(
            df,
            dv_col,
            subject_col,
            time_col,
            multiplicity_method=plan.multiplicity_method,
        ),
        "pairwise_wilcoxon": lambda: longitudinal_posthoc.run_pairwise_wilcoxon(
            df,
            dv_col,
            subject_col,
            time_col,
            multiplicity_method=plan.multiplicity_method,
        ),
        "pairwise_tests": lambda: longitudinal_posthoc.run_pairwise_group_at_time_tests(
            df,
            dv_col,
            subject_col,
            time_col,
            group_col,
            multiplicity_method=plan.multiplicity_method,
        ),
        None: lambda: {"pairwise_table": None, "warnings": [], "metadata": {"effect_sizes": {"pairwise": None}}},
    }[resolve_posthoc_id(plan.posthoc_method)]
    return _merge_payloads(omnibus_runner(), posthoc_runner())


def _merge_payloads(omnibus_payload: dict, pairwise_payload: dict) -> dict:
    omnibus_effect_sizes = (omnibus_payload.get("metadata", {}).get("effect_sizes") or {}).get("omnibus")
    pairwise_effect_sizes = (pairwise_payload.get("metadata", {}).get("effect_sizes") or {}).get("pairwise")
    metadata = dict(omnibus_payload.get("metadata", {}))
    metadata.update(pairwise_payload.get("metadata", {}))
    metadata["effect_sizes"] = {"omnibus": omnibus_effect_sizes, "pairwise": pairwise_effect_sizes}
    return {
        "analysis_status": "ready",
        "omnibus_table": omnibus_payload.get("omnibus_table"),
        "pairwise_table": pairwise_payload.get("pairwise_table"),
        "model_table": None,
        "warnings": sorted(set(omnibus_payload.get("warnings", []) + pairwise_payload.get("warnings", []))),
        "blocking_reasons": [],
        "suggested_actions": [],
        "metadata": metadata,
    }
