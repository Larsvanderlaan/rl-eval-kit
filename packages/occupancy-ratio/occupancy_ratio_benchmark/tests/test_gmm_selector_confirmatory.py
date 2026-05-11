from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Sequence

import pytest

from occupancy_ratio_benchmark.gmm_selector_confirmatory import (
    ARMS,
    aggregate_gmm_selector_outputs,
    cell_regret_rows,
    run_gmm_selector_confirmatory,
    selection_delta_rows,
    selected_tuning_by_arm_cell,
    smoke_matrix_specs,
)
from occupancy_ratio_benchmark.io import write_csv


def test_selected_tuning_prefers_selected_full_stage_candidate() -> None:
    base = {
        "arm": "gmm_ratio",
        "matrix_id": "m",
        "setting": "discrete_chain",
        "dataset_variant": "",
        "policy_shift": "0.65",
        "gamma": "0.9",
        "sample_size": "100",
        "seed": "0",
        "tuning_stage": "automl_candidate",
        "selected": "1.0",
    }
    selected = selected_tuning_by_arm_cell(
        [
            {**base, "budget_stage": "screen", "candidate_id": "screen_pick", "score": "1.0"},
            {**base, "budget_stage": "full", "candidate_id": "full_pick", "score": "2.0"},
        ]
    )
    row = next(iter(selected.values()))
    assert row["candidate_id"] == "full_pick"


def test_selection_delta_and_regret_use_completed_arm_selected_models() -> None:
    rows = [
        _summary_row("legacy_current", "ok", ratio_tv=0.30, ope=0.50, candidate="legacy"),
        _summary_row("gmm_ratio", "ok", ratio_tv=0.20, ope=0.60, candidate="ratio"),
        _summary_row("gmm_ope", "timeout", ratio_tv=None, ope=None, candidate=""),
        _summary_row("gmm_ope", "ok", ratio_tv=0.25, ope=0.40, candidate="ope", seed="1"),
        _summary_row("legacy_current", "ok", ratio_tv=0.28, ope=0.45, candidate="legacy", seed="1"),
    ]
    regrets = cell_regret_rows(rows)
    ratio_regret = next(row for row in regrets if row["arm"] == "gmm_ratio" and row["seed"] == "0")
    assert ratio_regret["best_completed_ratio_arm"] == "gmm_ratio"
    assert ratio_regret["ratio_tv_regret_vs_best_completed_arm"] == pytest.approx(0.0)
    timeout = next(row for row in regrets if row["arm"] == "gmm_ope" and row["status"] == "timeout")
    assert timeout["status"] == "timeout"

    merged = []
    by_arm_cell = {(row["arm"], tuple(row[field] for field in ("matrix_id", "setting", "dataset_variant", "policy_shift", "gamma", "sample_size", "seed"))): row for row in regrets}
    for row in rows:
        out = dict(row)
        regret = by_arm_cell[(row["arm"], tuple(row[field] for field in ("matrix_id", "setting", "dataset_variant", "policy_shift", "gamma", "sample_size", "seed")))]
        out.update(regret)
        merged.append(out)
    deltas = selection_delta_rows(merged)
    ratio_delta = next(row for row in deltas if row["arm"] == "gmm_ratio" and row["seed"] == "0")
    assert ratio_delta["ratio_outcome"] == "helped"
    assert ratio_delta["ope_outcome"] == "hurt"
    ope_delta = next(row for row in deltas if row["arm"] == "gmm_ope" and row["seed"] == "1")
    assert ope_delta["ratio_outcome"] == "helped"
    assert ope_delta["ope_outcome"] == "helped"


def test_aggregate_outputs_retains_timeout_rows(tmp_path: Path) -> None:
    for arm, status, ratio_tv, ope in (
        ("legacy_current", "ok", 0.30, 0.50),
        ("gmm_ratio", "timeout", None, None),
        ("gmm_ope", "ok", 0.25, 0.40),
    ):
        out = tmp_path / arm / "matrix" / "smoke"
        write_csv(out / "results.csv", [_result_row(status=status, ratio_tv=ratio_tv, ope=ope)])
        tuning_rows = [] if status == "timeout" else [_tuning_row(arm=arm, candidate=f"{arm}_candidate")]
        write_csv(out / "tuning_results.csv", tuning_rows)

    result = aggregate_gmm_selector_outputs(tmp_path)
    assert result.selector_summary_path.exists()
    assert result.selection_delta_path.exists()
    assert result.cell_regret_path.exists()
    assert result.selector_report_path.exists()
    summary = _read_csv(result.selector_summary_path)
    assert any(row["arm"] == "gmm_ratio" and row["status"] == "timeout" for row in summary)
    deltas = _read_csv(result.selection_delta_path)
    assert any(row["arm"] == "gmm_ope" and row["ope_outcome"] == "helped" for row in deltas)


def test_smoke_driver_writes_all_outputs_with_three_arms(tmp_path: Path) -> None:
    calls: list[tuple[str, ...]] = []

    def fake_run(cmd: Sequence[str], env: dict[str, str]) -> tuple[int, str]:
        del env
        command = tuple(cmd)
        calls.append(command)
        output_root = Path(command[command.index("--output-root") + 1])
        arm = output_root.parent.name
        matrix_id = output_root.name
        stage_dir = output_root / "smoke"
        setting, shift = {
            "smoke_discrete": ("discrete_chain", "0.65"),
            "smoke_gaussian": ("linear_gaussian", "1.0"),
            "smoke_gym": ("gym_pendulum", ""),
        }[matrix_id]
        write_csv(
            stage_dir / "results.csv",
            [
                _result_row(
                    status="ok",
                    ratio_tv=None if setting.startswith("gym_") else 0.3,
                    ope=0.5,
                    setting=setting,
                    policy_shift=shift,
                    sample_size="100",
                )
            ],
        )
        write_csv(
            stage_dir / "tuning_results.csv",
            [
                _tuning_row(
                    arm=arm,
                    matrix_id=matrix_id,
                    setting=setting,
                    policy_shift=shift,
                    sample_size="100",
                    candidate=f"{arm}_{matrix_id}",
                )
            ],
        )
        return 0, "ok"

    result = run_gmm_selector_confirmatory(mode="smoke", output_root=tmp_path, run_command=fake_run, resume=False)
    assert result.selector_summary_path.exists()
    assert result.selection_delta_path.exists()
    assert result.cell_regret_path.exists()
    assert result.selector_report_path.exists()
    assert len(calls) == len(ARMS) * len(smoke_matrix_specs())
    assert {call[call.index("--cv-score-method") + 1] for call in calls} == {"legacy_rank", "bellman_gmm"}
    summary = _read_csv(result.selector_summary_path)
    assert {row["arm"] for row in summary} == {arm.arm_id for arm in ARMS}
    assert len(summary) == len(ARMS) * len(smoke_matrix_specs())


def _summary_row(
    arm: str,
    status: str,
    *,
    ratio_tv: float | None,
    ope: float | None,
    candidate: str,
    seed: str = "0",
) -> dict[str, Any]:
    return {
        "arm": arm,
        "matrix_id": "m",
        "setting": "discrete_chain",
        "dataset_variant": "",
        "policy_shift": "0.65",
        "gamma": "0.9",
        "sample_size": "100",
        "seed": seed,
        "status": status,
        "selected_candidate_id": candidate,
        "selected_candidate_label": candidate,
        "ratio_tv_behavior": "" if ratio_tv is None else ratio_tv,
        "ratio_l1_behavior": "" if ratio_tv is None else 2.0 * ratio_tv,
        "ope_value_abs_error": "" if ope is None else ope,
        "runtime_sec": "1.0",
    }


def _result_row(
    *,
    status: str,
    ratio_tv: float | None,
    ope: float | None,
    setting: str = "discrete_chain",
    policy_shift: str = "0.65",
    sample_size: str = "100",
) -> dict[str, Any]:
    return {
        "estimator": "neural_network_stable",
        "setting": setting,
        "dataset_variant": "",
        "policy_shift": policy_shift,
        "gamma": "0.9",
        "sample_size": sample_size,
        "seed": "0",
        "status": status,
        "ratio_tv": "" if ratio_tv is None else ratio_tv,
        "ratio_l1": "" if ratio_tv is None else 2.0 * ratio_tv,
        "ratio_normalized_l1": "" if ratio_tv is None else ratio_tv,
        "ope_value_abs_error": "" if ope is None else ope,
        "ope_value_abs_error_se_units": "" if ope is None else 1.5,
        "effective_sample_size_fraction": "0.8",
        "weight_cv": "0.4",
        "weight_q99_to_median": "3.0",
        "clipping_fraction": "0.0",
        "normalization_error": "0.0",
        "runtime_sec": "1.0",
    }


def _tuning_row(
    *,
    arm: str,
    candidate: str,
    matrix_id: str = "matrix",
    setting: str = "discrete_chain",
    policy_shift: str = "0.65",
    sample_size: str = "100",
) -> dict[str, Any]:
    return {
        "arm": arm,
        "matrix_id": matrix_id,
        "setting": setting,
        "dataset_variant": "",
        "policy_shift": policy_shift,
        "gamma": "0.9",
        "sample_size": sample_size,
        "seed": "0",
        "tuning_stage": "automl_candidate",
        "budget_stage": "full",
        "selected": "1.0",
        "candidate_id": candidate,
        "candidate_label": candidate,
        "score": "0.1",
        "metric_selection_risk": "0.1",
        "metric_constraint_violated": "0.0",
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))
