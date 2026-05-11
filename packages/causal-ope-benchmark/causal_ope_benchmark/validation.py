from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from causal_ope_benchmark.constants import (
    BENCHMARK_SCHEMA_VERSION,
    CALIBRATION_OUTPUT_FILES,
    CALIBRATION_SCHEMA_VERSION,
    DEFAULT_OUTPUT_FILES,
    DIFFICULTY_OUTPUT_FILES,
    DIFFICULTY_SCHEMA_VERSION,
    RESULT_SCHEMA_VERSION,
    VALID_STATUSES,
)
from causal_ope_benchmark.exceptions import AdapterValidationError
from causal_ope_benchmark.types import BenchmarkProblem, LongitudinalDataset, assert_no_forbidden_public_keys


def validate_status(status: str, *, context: str = "status") -> str:
    """Validate a standardized benchmark status string."""

    value = str(status)
    if value not in VALID_STATUSES:
        raise AdapterValidationError(f"{context} must be one of {', '.join(VALID_STATUSES)}; got {value!r}.")
    return value


def validate_longitudinal_dataset(dataset: LongitudinalDataset) -> None:
    """Run public shape and finiteness checks on a longitudinal dataset."""

    n = int(dataset.n)
    _require_2d("states", dataset.states, n_rows=n)
    _require_2d("next_states", dataset.next_states, n_rows=n)
    _require_2d("actions", dataset.actions, n_rows=n)
    _require_2d("target_actions", dataset.target_actions, n_rows=n)
    _require_2d("next_target_actions", dataset.next_target_actions, n_rows=n)
    if np.asarray(dataset.states).shape != np.asarray(dataset.next_states).shape:
        raise AdapterValidationError("states and next_states must have the same shape.")
    if np.asarray(dataset.actions).shape != np.asarray(dataset.target_actions).shape:
        raise AdapterValidationError("actions and target_actions must have the same shape.")
    for name in ("rewards", "terminals", "censoring", "behavior_propensity", "target_propensity_observed_action"):
        _require_vector(name, getattr(dataset, name), n_rows=n)
    if np.any(np.asarray(dataset.behavior_propensity, dtype=np.float64) <= 0.0):
        raise AdapterValidationError("behavior_propensity must be strictly positive.")
    if np.any(np.asarray(dataset.target_propensity_observed_action, dtype=np.float64) <= 0.0):
        raise AdapterValidationError("target_propensity_observed_action must be strictly positive.")
    if dataset.target_action_probabilities is not None:
        _require_probability_matrix("target_action_probabilities", dataset.target_action_probabilities, n, dataset.action_dim)
    if dataset.next_target_action_probabilities is not None:
        _require_probability_matrix("next_target_action_probabilities", dataset.next_target_action_probabilities, n, dataset.action_dim)
    if dataset.initial_action_probabilities is not None:
        _require_probability_matrix(
            "initial_action_probabilities",
            dataset.initial_action_probabilities,
            np.asarray(dataset.initial_actions).shape[0],
            dataset.action_dim,
        )


def validate_problem(problem: BenchmarkProblem) -> None:
    """Validate public dataset shape and sealed-truth alignment for a problem.

    The helper checks consistency between ``dataset`` and ``truth`` without
    exposing truth values through public adapters or metadata.
    """

    validate_longitudinal_dataset(problem.dataset)
    if problem.dataset.name != problem.truth.dataset_name:
        raise AdapterValidationError("problem.dataset.name must match problem.truth.dataset_name.")
    if problem.dataset.family != problem.truth.family:
        raise AdapterValidationError("problem.dataset.family must match problem.truth.family.")
    assert_no_forbidden_public_keys(problem.dataset.metadata_public)


def validate_output_bundle(output_dir: str | Path) -> None:
    """Validate the files and core schemas in a benchmark output directory."""

    root = Path(output_dir)
    _require_files(root, DEFAULT_OUTPUT_FILES.values())
    results = _read_csv(root / DEFAULT_OUTPUT_FILES["results"])
    summary = _read_csv(root / DEFAULT_OUTPUT_FILES["summary"])
    manifest = _read_json(root / DEFAULT_OUTPUT_FILES["manifest"])
    diagnostics = _read_json(root / DEFAULT_OUTPUT_FILES["diagnostics"])
    schema = _read_json(root / DEFAULT_OUTPUT_FILES["output_schema"])
    if not isinstance(diagnostics, dict):
        raise AdapterValidationError("diagnostics.json must contain a JSON object.")
    _require_json_value(manifest, "benchmark_schema_version", BENCHMARK_SCHEMA_VERSION, "manifest.json")
    _require_json_value(manifest, "result_schema_version", RESULT_SCHEMA_VERSION, "manifest.json")
    _require_json_value(schema, "benchmark_schema_version", BENCHMARK_SCHEMA_VERSION, "output_schema.json")
    _require_json_value(schema, "result_schema_version", RESULT_SCHEMA_VERSION, "output_schema.json")
    _validate_rows(
        results,
        required=set(schema.get("result_required_columns", ())),
        schema_version_column="benchmark_schema_version",
        schema_version=BENCHMARK_SCHEMA_VERSION,
        label="results.csv",
    )
    _validate_rows(
        summary,
        required=set(schema.get("summary_required_columns", ())),
        schema_version_column=None,
        schema_version=None,
        label="summary.csv",
    )


def validate_calibration_output_bundle(output_dir: str | Path) -> None:
    """Validate the files and core schemas in a calibration output directory."""

    root = Path(output_dir)
    _require_files(root, CALIBRATION_OUTPUT_FILES.values())
    results = _read_csv(root / CALIBRATION_OUTPUT_FILES["results"])
    summary = _read_csv(root / CALIBRATION_OUTPUT_FILES["summary"])
    candidates = _read_csv(root / CALIBRATION_OUTPUT_FILES["candidates"])
    manifest = _read_json(root / CALIBRATION_OUTPUT_FILES["manifest"])
    _require_json_value(manifest, "calibration_schema_version", CALIBRATION_SCHEMA_VERSION, "calibration_manifest.json")
    required_results = {
        "calibration_schema_version",
        "package_version",
        "family",
        "scenario",
        "estimator",
        "tuning_track",
        "status",
        "diagnostic_only",
        "leaderboard_result_eligible",
    }
    _validate_rows(
        results,
        required=required_results,
        schema_version_column="calibration_schema_version",
        schema_version=CALIBRATION_SCHEMA_VERSION,
        label="calibration_results.csv",
    )
    _validate_rows(summary, required={"family", "scenario", "estimator", "tuning_track"}, schema_version_column=None, schema_version=None, label="calibration_summary.csv")
    if candidates:
        _validate_rows(candidates, required={"family", "scenario", "estimator", "tuning_track"}, schema_version_column=None, schema_version=None, label="calibration_candidates.csv")


def validate_difficulty_output_bundle(output_dir: str | Path) -> None:
    """Validate the files and core schemas in a difficulty stress output directory."""

    root = Path(output_dir)
    _require_files(root, DIFFICULTY_OUTPUT_FILES.values())
    results = _read_csv(root / DIFFICULTY_OUTPUT_FILES["results"])
    summary = _read_csv(root / DIFFICULTY_OUTPUT_FILES["summary"])
    candidates = _read_csv(root / DIFFICULTY_OUTPUT_FILES["candidates"])
    manifest = _read_json(root / DIFFICULTY_OUTPUT_FILES["manifest"])
    _require_json_value(manifest, "difficulty_schema_version", DIFFICULTY_SCHEMA_VERSION, "difficulty_manifest.json")
    required_results = {
        "difficulty_schema_version",
        "package_version",
        "scale",
        "difficulty",
        "family",
        "stress_dimension",
        "method",
        "tuning_track",
        "status",
        "diagnostic_only",
        "target_estimand",
    }
    _validate_rows(
        results,
        required=required_results,
        schema_version_column="difficulty_schema_version",
        schema_version=DIFFICULTY_SCHEMA_VERSION,
        label="difficulty_results.csv",
    )
    _validate_rows(
        summary,
        required={"difficulty", "family", "stress_dimension", "method", "tuning_track", "hardness_verdict"},
        schema_version_column=None,
        schema_version=None,
        label="difficulty_summary.csv",
    )
    if candidates:
        _validate_rows(candidates, required={"difficulty", "family", "scenario", "estimator", "tuning_track"}, schema_version_column=None, schema_version=None, label="difficulty_candidates.csv")


def validate_scope_rl_logged_dataset(payload: dict[str, Any]) -> None:
    """Validate the public SCOPE-RL logged-dataset export shape."""

    required = {
        "size",
        "n_trajectories",
        "step_per_trajectory",
        "action_type",
        "n_actions",
        "state_dim",
        "state",
        "action",
        "reward",
        "done",
        "terminal",
        "pscore",
        "info",
        "behavior_policy",
        "dataset_id",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise AdapterValidationError(f"SCOPE-RL export is missing required keys: {missing}.")
    states = np.asarray(payload["state"])
    rewards = np.asarray(payload["reward"])
    actions = np.asarray(payload["action"])
    done = np.asarray(payload["done"])
    terminal = np.asarray(payload["terminal"])
    pscore = np.asarray(payload["pscore"], dtype=np.float64)
    expected = (int(payload["n_trajectories"]), int(payload["step_per_trajectory"]))
    if states.shape != (*expected, int(payload["state_dim"])):
        raise AdapterValidationError("state must have shape (n_trajectories, step_per_trajectory, state_dim).")
    for name, arr in (("reward", rewards), ("action", actions), ("done", done), ("terminal", terminal), ("pscore", pscore)):
        if arr.shape != expected:
            raise AdapterValidationError(f"{name} must have shape (n_trajectories, step_per_trajectory).")
    if not np.all(np.isfinite(pscore)) or np.any(pscore <= 0.0):
        raise AdapterValidationError("pscore must contain finite positive values.")
    info = payload["info"]
    if not isinstance(info, dict):
        raise AdapterValidationError("info must be a dictionary.")
    availability = np.asarray(info.get("action_available"))
    if availability.shape != (*expected, int(payload["n_actions"])):
        raise AdapterValidationError("info['action_available'] must align with padded trajectories and n_actions.")


def _require_2d(name: str, value: Any, *, n_rows: int) -> None:
    arr = np.asarray(value)
    if arr.ndim != 2 or arr.shape[0] != int(n_rows):
        raise AdapterValidationError(f"{name} must be a 2D array with {int(n_rows)} rows.")
    if not np.all(np.isfinite(arr)):
        raise AdapterValidationError(f"{name} must contain finite values.")


def _require_vector(name: str, value: Any, *, n_rows: int) -> None:
    arr = np.asarray(value)
    if arr.reshape(-1).shape[0] != int(n_rows):
        raise AdapterValidationError(f"{name} must have {int(n_rows)} rows.")
    if not np.all(np.isfinite(arr)):
        raise AdapterValidationError(f"{name} must contain finite values.")


def _require_probability_matrix(name: str, value: Any, n_rows: int, n_actions: int) -> None:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (int(n_rows), int(n_actions)):
        raise AdapterValidationError(f"{name} must have shape ({int(n_rows)}, {int(n_actions)}).")
    if not np.all(np.isfinite(arr)) or np.any(arr < 0.0):
        raise AdapterValidationError(f"{name} must contain finite nonnegative values.")
    if arr.shape[0] and not np.allclose(arr.sum(axis=1), 1.0):
        raise AdapterValidationError(f"{name} rows must sum to 1.")


def _require_files(root: Path, filenames: Any) -> None:
    missing = [name for name in filenames if not (root / str(name)).exists()]
    if missing:
        raise AdapterValidationError(f"{root} is missing required output files: {missing}.")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise AdapterValidationError(f"{path.name} must contain a JSON object.")
    return payload


def _require_json_value(payload: dict[str, Any], key: str, expected: str, label: str) -> None:
    if payload.get(key) != expected:
        raise AdapterValidationError(f"{label}.{key} must be {expected!r}; got {payload.get(key)!r}.")


def _validate_rows(
    rows: list[dict[str, str]],
    *,
    required: set[str],
    schema_version_column: str | None,
    schema_version: str | None,
    label: str,
) -> None:
    if not rows:
        raise AdapterValidationError(f"{label} must contain at least one data row.")
    missing = sorted(required - set(rows[0]))
    if missing:
        raise AdapterValidationError(f"{label} is missing required columns: {missing}.")
    for index, row in enumerate(rows):
        if "status" in row:
            validate_status(row["status"], context=f"{label}[{index}].status")
        if schema_version_column and row.get(schema_version_column) != schema_version:
            raise AdapterValidationError(
                f"{label}[{index}].{schema_version_column} must be {schema_version!r}; got {row.get(schema_version_column)!r}."
            )
