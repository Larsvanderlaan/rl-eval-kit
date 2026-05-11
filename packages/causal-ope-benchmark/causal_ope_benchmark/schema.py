from __future__ import annotations

from typing import Any

from causal_ope_benchmark.constants import (
    BENCHMARK_SCHEMA_VERSION,
    CALIBRATION_OUTPUT_FILES,
    CALIBRATION_SCHEMA_VERSION,
    DEFAULT_OUTPUT_FILES,
    FAMILY_REGISTRY_VERSION,
    OUTPUT_SCHEMA_VERSION,
    PACKAGE_VERSION,
    RESULT_SCHEMA_VERSION,
    VALID_STATUSES,
)


def output_schema() -> dict[str, Any]:
    """Return the machine-readable schema payload written with each run."""

    return {
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
        "benchmark_schema_version": BENCHMARK_SCHEMA_VERSION,
        "calibration_schema_version": CALIBRATION_SCHEMA_VERSION,
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "family_registry_version": FAMILY_REGISTRY_VERSION,
        "package_version": PACKAGE_VERSION,
        "files": DEFAULT_OUTPUT_FILES,
        "calibration_files": CALIBRATION_OUTPUT_FILES,
        "statuses": list(VALID_STATUSES),
        "result_required_columns": [
            "benchmark_schema_version",
            "result_schema_version",
            "family_registry_version",
            "package_version",
            "profile",
            "family",
            "dataset",
            "scenario",
            "estimator",
            "status",
            "gamma",
            "seed",
            "sample_size",
            "row_count",
            "target_policy",
            "runtime_sec",
            "diagnostic_only",
            "leaderboard_result_eligible",
        ],
        "summary_required_columns": [
            "profile",
            "family",
            "estimator",
            "n_rows",
            "ok_rows",
            "deployable_rows",
            "leaderboard_eligible_rows",
        ],
        "calibration_result_required_columns": [
            "calibration_schema_version",
            "package_version",
            "family",
            "scenario",
            "estimator",
            "tuning_track",
            "status",
            "diagnostic_only",
            "leaderboard_result_eligible",
        ],
        "calibration_summary_required_columns": [
            "family",
            "scenario",
            "estimator",
            "tuning_track",
        ],
        "sealed_scorer_data_policy": "Public datasets, adapters, manifests, and output schemas contain no scorer-only quantities or hidden scenario labels.",
    }
