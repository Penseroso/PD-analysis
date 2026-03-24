from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import ast
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.stats_cross import compute_cross_assumptions, run_cross_sectional
from utils.stats_longitudinal import compute_longitudinal_assumptions, run_longitudinal
from utils.stats_mixedlm import run_mixedlm
from utils.stats_selector import build_analysis_plan, select_method
from utils.validators import validate_normalized_df


EXPECTATIONS_PATH = ROOT / "tests" / "smoke_expectations.yaml"
DATA_DIR = ROOT / "tests" / "smoke"


@dataclass
class SmokeResult:
    name: str
    passed: bool
    details: list[str]



def main() -> int:
    expectations = load_expectations(EXPECTATIONS_PATH)
    results = [run_case(name, spec) for name, spec in expectations.items()]

    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(f"[{status}] {result.name}")
        for detail in result.details:
            print(f"  - {detail}")

    passed_count = sum(result.passed for result in results)
    print(f"\nSummary: {passed_count}/{len(results)} passed")
    return 0 if passed_count == len(results) else 1



def run_case(name: str, spec: dict) -> SmokeResult:
    details: list[str] = []
    failures: list[str] = []

    df = pd.read_csv(DATA_DIR / f"{name}.tsv", sep="\t")
    selected_dv_cols = [column for column in df.columns if column.startswith("value_")]
    validation = validate_normalized_df(
        df=df,
        data_type=spec["data_type"],
        selected_dv_cols=selected_dv_cols,
        between_factors=spec.get("between_factors", ["group"]),
        factor2_col=spec.get("factor2_col"),
        control_group=spec.get("control_group"),
    )

    assumptions = build_assumptions(df, spec, selected_dv_cols[0] if selected_dv_cols else "value_1")
    selector_result = select_method(
        data_type=spec["data_type"],
        normality=assumptions.get("normality", {}),
        sphericity=assumptions.get("sphericity"),
        levene=assumptions.get("levene", {}),
        balance_info=validation["balance_info"],
        between_factors=spec.get("between_factors", ["group"]),
        n_per_group=validation["n_per_group"],
    )
    plan = build_analysis_plan(validation, selector_result, None)
    result = execute_analysis(df, spec, plan, selected_dv_cols[0] if selected_dv_cols else "value_1", validation)

    details.append(f"analysis_status={result.get('analysis_status')}")
    details.append(
        f"recommended={selector_result.get('recommended_method')} via {selector_result.get('recommended_engine')}"
    )

    check_expected_status(spec, result, failures)
    check_selector(spec, selector_result, failures)
    check_result_shape(spec, result, failures)
    check_messages(spec, selector_result, result, failures)

    if "expected_used_method" in spec:
        actual_method = result.get("used_method")
        if actual_method != spec["expected_used_method"]:
            failures.append(
                f"used_method mismatch: expected {spec['expected_used_method']}, got {actual_method}"
            )

    if "expected_used_formula_contains" in spec:
        used_formula = result.get("used_formula", "")
        for token in spec["expected_used_formula_contains"]:
            if token not in used_formula:
                failures.append(f"used_formula missing token: {token}")

    if failures:
        details.extend(failures)
    return SmokeResult(name=name, passed=not failures, details=details)



def load_expectations(path: Path) -> dict:
    lines = path.read_text(encoding="utf-8").splitlines()
    output: dict[str, dict] = {}
    current_key: str | None = None
    block_key: str | None = None
    block_buffer: list[str] = []

    for raw_line in lines:
        if not raw_line.strip():
            continue
        if not raw_line.startswith(" "):
            if current_key is not None and block_key is not None:
                output[current_key][block_key] = " ".join(part.strip() for part in block_buffer).strip()
                block_key = None
                block_buffer = []
            current_key = raw_line.rstrip(":").strip()
            output[current_key] = {}
            continue

        if current_key is None:
            continue

        stripped = raw_line.strip()
        if block_key is not None:
            if raw_line.startswith("    "):
                block_buffer.append(stripped)
                continue
            output[current_key][block_key] = " ".join(part.strip() for part in block_buffer).strip()
            block_key = None
            block_buffer = []

        if ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        key = key.strip()
        value = value.strip()
        if value == ">":
            block_key = key
            block_buffer = []
            continue
        output[current_key][key] = _parse_scalar(value)

    if current_key is not None and block_key is not None:
        output[current_key][block_key] = " ".join(part.strip() for part in block_buffer).strip()
    return output



def _parse_scalar(value: str):
    if value == "null":
        return None
    if value == "true":
        return True
    if value == "false":
        return False
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(part.strip()) for part in inner.split(",")]
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    try:
        return ast.literal_eval(value)
    except Exception:
        return value



def build_assumptions(df: pd.DataFrame, spec: dict, dv_col: str) -> dict:
    if spec["data_type"] == "cross":
        return compute_cross_assumptions(df=df, dv_col=dv_col, group_col="group")
    required = {"group", "subject", "time", dv_col}
    if not required.issubset(df.columns):
        return {"normality": {}, "sphericity": None}
    return compute_longitudinal_assumptions(
        df=df,
        dv_col=dv_col,
        group_col="group",
        subject_col="subject",
        time_col="time",
        between_factors=spec.get("between_factors", ["group"]),
    )



def execute_analysis(df: pd.DataFrame, spec: dict, plan: dict, dv_col: str, validation: dict | None = None) -> dict:
    if plan["analysis_status"] == "blocked":
        return {
            "analysis_status": "blocked",
            "used_method": plan.get("final_method"),
            "omnibus": None,
            "posthoc_table": None,
            "star_map": [],
            "warnings": plan.get("warnings", validation.get("warnings", []) if validation else []),
            "blocking_reasons": plan.get("blocking_reasons", validation.get("blocking_reasons", []) if validation else []),
            "suggested_actions": validation.get("suggested_actions", []) if validation else [],
        }

    if spec["data_type"] == "cross":
        return run_cross_sectional(
            df=df,
            dv_col=dv_col,
            group_col="group",
            control_group=spec.get("control_group"),
            method=plan["final_method"],
        )

    if plan["engine"] == "statsmodels":
        return run_mixedlm(
            df=df,
            dv_col=dv_col,
            subject_col="subject",
            time_col="time",
            group_col="group",
            factor2_col=spec.get("factor2_col"),
            formula_mode="default",
            reference_group=spec.get("reference_group", spec.get("control_group")),
        )

    return run_longitudinal(
        df=df,
        dv_col=dv_col,
        group_col="group",
        subject_col="subject",
        time_col="time",
        control_group=None,
        between_factors=spec.get("between_factors", ["group"]),
        factor2_col=spec.get("factor2_col"),
        method=plan["final_method"],
    )



def check_expected_status(spec: dict, result: dict, failures: list[str]) -> None:
    actual = result.get("analysis_status")
    if "expected_analysis_status" in spec and actual != spec["expected_analysis_status"]:
        failures.append(
            f"analysis_status mismatch: expected {spec['expected_analysis_status']}, got {actual}"
        )
    if "expected_analysis_status_any_of" in spec and actual not in spec["expected_analysis_status_any_of"]:
        failures.append(
            f"analysis_status mismatch: expected one of {spec['expected_analysis_status_any_of']}, got {actual}"
        )



def check_selector(spec: dict, selector_result: dict, failures: list[str]) -> None:
    if "expected_recommended_method" in spec:
        actual_method = selector_result.get("recommended_method")
        if actual_method != spec["expected_recommended_method"]:
            failures.append(
                f"recommended_method mismatch: expected {spec['expected_recommended_method']}, got {actual_method}"
            )
    if "expected_recommended_engine" in spec:
        actual_engine = selector_result.get("recommended_engine")
        if actual_engine != spec["expected_recommended_engine"]:
            failures.append(
                f"recommended_engine mismatch: expected {spec['expected_recommended_engine']}, got {actual_engine}"
            )



def check_result_shape(spec: dict, result: dict, failures: list[str]) -> None:
    omnibus = result.get("omnibus")
    posthoc = result.get("posthoc_table")
    star_map = result.get("star_map", [])

    omnibus_should_exist = spec.get("omnibus_should_exist")
    if omnibus_should_exist is True and not isinstance(omnibus, pd.DataFrame):
        failures.append("expected omnibus DataFrame to exist")
    if omnibus_should_exist is False and omnibus is not None:
        failures.append("expected omnibus to be absent")

    posthoc_should_exist = spec.get("posthoc_should_exist")
    if posthoc_should_exist is True and not isinstance(posthoc, pd.DataFrame):
        failures.append("expected posthoc_table DataFrame to exist")
    if posthoc_should_exist is False and posthoc is not None:
        failures.append("expected posthoc_table to be absent")

    if "star_map_min_count" in spec and len(star_map) < spec["star_map_min_count"]:
        failures.append(
            f"expected at least {spec['star_map_min_count']} star_map entries, got {len(star_map)}"
        )

    if isinstance(omnibus, pd.DataFrame) and "expected_omnibus_terms" in spec:
        actual_terms = {str(term).lower() for term in omnibus["term"].tolist()}
        expected_terms = {str(term).lower() for term in spec["expected_omnibus_terms"]}
        if not expected_terms.issubset(actual_terms):
            failures.append(f"omnibus terms mismatch: expected at least {sorted(expected_terms)}, got {sorted(actual_terms)}")

    if isinstance(omnibus, pd.DataFrame) and "expected_omnibus_effect_metric" in spec:
        metrics = {str(metric).lower() for metric in omnibus["effect_metric"].dropna().tolist()}
        if spec["expected_omnibus_effect_metric"].lower() not in metrics:
            failures.append(
                f"omnibus effect metric mismatch: expected {spec['expected_omnibus_effect_metric']}, got {sorted(metrics)}"
            )

    effect_sizes = result.get("effect_sizes")
    if isinstance(effect_sizes, dict):
        pairwise_effects = effect_sizes.get("pairwise")
        omnibus_effects = effect_sizes.get("omnibus")
        if "expected_pairwise_effect_metric" in spec and isinstance(pairwise_effects, pd.DataFrame):
            metrics = {str(metric).lower() for metric in pairwise_effects.get("effect_metric", pd.Series(dtype=object)).dropna().tolist()}
            if spec["expected_pairwise_effect_metric"].lower() not in metrics:
                failures.append(
                    f"pairwise effect metric mismatch: expected {spec['expected_pairwise_effect_metric']}, got {sorted(metrics)}"
                )
        if "expected_effect_metric" in spec and isinstance(omnibus_effects, pd.DataFrame):
            metrics = {str(metric).lower() for metric in omnibus_effects.get("effect_metric", pd.Series(dtype=object)).dropna().tolist()}
            if spec["expected_effect_metric"].lower() not in metrics:
                failures.append(
                    f"effect metric mismatch: expected {spec['expected_effect_metric']}, got {sorted(metrics)}"
                )
    elif isinstance(effect_sizes, pd.DataFrame) and "expected_effect_metric" in spec:
        metrics = {str(metric).lower() for metric in effect_sizes.get("effect_metric", pd.Series(dtype=object)).dropna().tolist()}
        if spec["expected_effect_metric"].lower() not in metrics:
            failures.append(
                f"effect metric mismatch: expected {spec['expected_effect_metric']}, got {sorted(metrics)}"
            )

    if spec.get("fixed_effects_should_exist") and not isinstance(result.get("fixed_effects"), pd.DataFrame):
        failures.append("expected fixed_effects DataFrame to exist")
    if spec.get("contrast_table_should_exist") and not isinstance(result.get("contrast_table"), pd.DataFrame):
        failures.append("expected contrast_table DataFrame to exist")



def check_messages(spec: dict, selector_result: dict, result: dict, failures: list[str]) -> None:
    warnings = " | ".join(result.get("warnings", []))
    blocking = " | ".join(result.get("blocking_reasons", []))
    fallback_reason = selector_result.get("fallback_reason", "") or ""

    for token in spec.get("expected_warning_contains", []):
        if token.lower() not in warnings.lower():
            failures.append(f"missing warning token: {token}")
    for token in spec.get("expected_blocking_reason_contains", []):
        if token.lower() not in blocking.lower():
            failures.append(f"missing blocking reason token: {token}")
    for token in spec.get("expected_fallback_reason_contains", []):
        if token.lower() not in fallback_reason.lower():
            failures.append(f"missing fallback reason token: {token}")


if __name__ == "__main__":
    raise SystemExit(main())
