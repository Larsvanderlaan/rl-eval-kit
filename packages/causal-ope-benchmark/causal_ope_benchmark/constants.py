from __future__ import annotations

BENCHMARK_SCHEMA_VERSION = "benchmark_v1"
RESULT_SCHEMA_VERSION = "results_v1"
CALIBRATION_SCHEMA_VERSION = "calibration_v1"
DIFFICULTY_SCHEMA_VERSION = "difficulty_v1"
OUTPUT_SCHEMA_VERSION = "output_bundle_v1"
FAMILY_REGISTRY_VERSION = "family_registry_v1"
PACKAGE_VERSION = "0.2.0"

STATUS_OK = "ok"
STATUS_SKIPPED = "skipped"
STATUS_MISSING_DEPENDENCY = "missing_dependency"
STATUS_INCOMPLETE = "incomplete"
STATUS_ERROR = "error"
VALID_STATUSES = (
    STATUS_OK,
    STATUS_SKIPPED,
    STATUS_MISSING_DEPENDENCY,
    STATUS_INCOMPLETE,
    STATUS_ERROR,
)

DEFAULT_FAMILIES = ("streamlift", "streamretain", "clinic_dtr")
ALL_FAMILIES = (*DEFAULT_FAMILIES, "epicare")

DEFAULT_OUTPUT_FILES = {
    "results": "results.csv",
    "summary": "summary.csv",
    "tuning_results": "tuning_results.csv",
    "manifest": "manifest.json",
    "diagnostics": "diagnostics.json",
    "readout": "benchmark_readout.md",
    "output_schema": "output_schema.json",
}

CALIBRATION_OUTPUT_FILES = {
    "results": "calibration_results.csv",
    "summary": "calibration_summary.csv",
    "candidates": "calibration_candidates.csv",
    "manifest": "calibration_manifest.json",
    "readout": "calibration_readout.md",
}

DIFFICULTY_OUTPUT_FILES = {
    "results": "difficulty_results.csv",
    "summary": "difficulty_summary.csv",
    "candidates": "difficulty_candidates.csv",
    "manifest": "difficulty_manifest.json",
    "readout": "difficulty_readout.md",
}
