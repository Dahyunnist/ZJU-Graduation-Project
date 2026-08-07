"""Structural hard gates and diagnostic warnings for generated pools."""

from __future__ import annotations

from typing import Any


def evaluate_quality_gate(
    *,
    access_audit: dict[str, Any],
    pool_validation: dict[str, Any],
    quality: dict[str, Any],
    real_target_values: set[str],
    reload_validation: dict[str, Any],
    required_artifacts_present: bool,
    cross_run_content_distinct: bool = True,
) -> dict[str, Any]:
    failures: list[str] = []
    if not access_audit.get("passed") or access_audit.get("actual_partitions") != ["R_source_train"]:
        failures.append("generator_access")
    fmt = quality.get("format", {})
    if not fmt.get("schema_valid") or fmt.get("numeric_parse_failures") != 0:
        failures.append("schema_or_numeric_format")
    generated_targets = set(str(value) for value in fmt.get("target_values", []))
    if not generated_targets.issubset(real_target_values):
        failures.append("illegal_target_value")
    if generated_targets != real_target_values:
        failures.append("target_class_collapse")
    if not pool_validation.get("all_synth_ids_unique"):
        failures.append("pool_id_isolation")
    if not reload_validation.get("model_reload_ok") or not reload_validation.get("schema_order_ok"):
        failures.append("model_reload")
    if quality.get("TSTR", {}).get("status") != "ok":
        failures.append("tstr_unavailable")
    if not required_artifacts_present:
        failures.append("required_artifacts")
    if not cross_run_content_distinct:
        failures.append("cross_run_sampling_seed_ineffective")
    warnings = {
        "sdmetrics": quality.get("sdmetrics"),
        "overlap": quality.get("overlap"),
        "target_values": sorted(generated_targets),
        "tstr_trtr_auroc_gap": (
            quality.get("TRTR", {}).get("auroc") - quality.get("TSTR", {}).get("auroc")
            if quality.get("TRTR", {}).get("auroc") is not None
            and quality.get("TSTR", {}).get("auroc") is not None
            else None
        ),
    }
    return {"passed": not failures, "hard_failures": failures, "diagnostic_warnings": warnings}
