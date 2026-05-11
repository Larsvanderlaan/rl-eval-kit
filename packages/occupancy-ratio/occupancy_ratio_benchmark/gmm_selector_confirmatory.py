from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from occupancy_ratio_benchmark.gym_control import GYM_CONTROL_SETTINGS
from occupancy_ratio_benchmark.io import write_csv, write_json


CONTROLLED_SETTINGS = {"discrete_chain", "discrete_grid", "linear_gaussian"}
GYM_SETTINGS = {"gym_pendulum", "gym_mountain_car_continuous", "gym_halfcheetah", "gym_hopper"}
CELL_FIELDS = ("matrix_id", "setting", "dataset_variant", "policy_shift", "gamma", "sample_size", "seed")
LEGACY_ARM_ID = "legacy_current"
DELTA_TOL = 1e-9


@dataclass(frozen=True)
class SelectorArm:
    arm_id: str
    description: str
    flags: tuple[str, ...]


@dataclass(frozen=True)
class MatrixSpec:
    matrix_id: str
    settings: tuple[str, ...]
    sample_sizes: tuple[int, ...]
    gammas: tuple[float, ...]
    seeds: tuple[int, ...]
    discrete_policy_shifts: tuple[float, ...] = ()
    linear_gaussian_policy_shifts: tuple[float, ...] = ()
    gym_target_value_rollouts: int = 128


@dataclass(frozen=True)
class GMMSelectorConfirmatoryResult:
    output_root: Path
    run_log_path: Path
    selector_summary_path: Path
    selection_delta_path: Path
    cell_regret_path: Path
    selector_report_path: Path


RunCommand = Callable[[Sequence[str], dict[str, str]], tuple[int, str]]


ARMS: tuple[SelectorArm, ...] = (
    SelectorArm(
        arm_id=LEGACY_ARM_ID,
        description="Current legacy AutoML ranking selector.",
        flags=("--cv-score-method", "legacy_rank"),
    ),
    SelectorArm(
        arm_id="gmm_ratio",
        description="Cross-fitted covariance-whitened broad Bellman GMM ratio selector.",
        flags=(
            "--cv-score-method",
            "bellman_gmm",
            "--cv-gmm-objective",
            "ratio",
            "--cv-gmm-cov-ridge",
            "0.10",
            "--cv-gmm-complexity-weight",
            "0.05",
        ),
    ),
    SelectorArm(
        arm_id="gmm_ope",
        description="Reward/value-targeted Bellman GMM OPE selector.",
        flags=(
            "--cv-score-method",
            "bellman_gmm",
            "--cv-gmm-objective",
            "ope",
            "--cv-gmm-ope-broad-weight",
            "0.0",
            "--cv-gmm-cov-ridge",
            "0.10",
            "--cv-gmm-complexity-weight",
            "0.05",
        ),
    ),
)


def smoke_matrix_specs() -> tuple[MatrixSpec, ...]:
    return (
        MatrixSpec(
            matrix_id="smoke_discrete",
            settings=("discrete_chain",),
            sample_sizes=(100,),
            gammas=(0.9,),
            seeds=(0,),
            discrete_policy_shifts=(0.65,),
            gym_target_value_rollouts=4,
        ),
        MatrixSpec(
            matrix_id="smoke_gaussian",
            settings=("linear_gaussian",),
            sample_sizes=(100,),
            gammas=(0.9,),
            seeds=(0,),
            linear_gaussian_policy_shifts=(1.0,),
            gym_target_value_rollouts=4,
        ),
        MatrixSpec(
            matrix_id="smoke_gym",
            settings=("gym_pendulum",),
            sample_sizes=(100,),
            gammas=(0.9,),
            seeds=(0,),
            gym_target_value_rollouts=4,
        ),
    )


def confirmatory_matrix_specs(*, include_heavy_gym: bool = False) -> tuple[MatrixSpec, ...]:
    specs = [
        MatrixSpec(
            matrix_id="controlled_tabular",
            settings=("discrete_chain", "discrete_grid"),
            sample_sizes=(1_000, 5_000),
            gammas=(0.9, 0.99),
            seeds=tuple(range(5)),
            discrete_policy_shifts=(0.0, 0.65, 1.5),
        ),
        MatrixSpec(
            matrix_id="controlled_gaussian",
            settings=("linear_gaussian",),
            sample_sizes=(1_000, 5_000),
            gammas=(0.9, 0.99),
            seeds=tuple(range(5)),
            linear_gaussian_policy_shifts=(0.0, 1.0, 3.0),
        ),
        MatrixSpec(
            matrix_id="gym",
            settings=("gym_pendulum", "gym_mountain_car_continuous"),
            sample_sizes=(1_000, 5_000),
            gammas=(0.9, 0.95),
            seeds=tuple(range(5)),
            gym_target_value_rollouts=128,
        ),
    ]
    if include_heavy_gym:
        heavy = tuple(setting for setting in ("gym_halfcheetah", "gym_hopper") if _gym_setting_available(setting))
        if heavy:
            specs.append(
                MatrixSpec(
                    matrix_id="gym_heavy",
                    settings=heavy,
                    sample_sizes=(1_000, 5_000),
                    gammas=(0.9, 0.95),
                    seeds=tuple(range(5)),
                    gym_target_value_rollouts=128,
                )
            )
    return tuple(specs)


def run_gmm_selector_confirmatory(
    *,
    mode: str,
    output_root: str | Path | None = None,
    include_heavy_gym: bool = False,
    automl_tuning: str = "fast",
    cv_folds: int = 3,
    benchmark_stage: str = "smoke",
    estimator_timeout_sec: float | None = None,
    external_repo_path: str | Path | None = None,
    resume: bool = True,
    write_plots: bool = False,
    python_executable: str | None = None,
    run_command: RunCommand | None = None,
) -> GMMSelectorConfirmatoryResult:
    if mode not in {"smoke", "confirmatory"}:
        raise ValueError("mode must be 'smoke' or 'confirmatory'.")
    root = Path(
        output_root
        if output_root is not None
        else ("outputs/gmm_selector_confirmatory_smoke" if mode == "smoke" else "outputs/gmm_selector_confirmatory")
    )
    root.mkdir(parents=True, exist_ok=True)
    specs = smoke_matrix_specs() if mode == "smoke" else confirmatory_matrix_specs(include_heavy_gym=include_heavy_gym)
    timeout = estimator_timeout_sec
    if timeout is None:
        timeout = 180.0 if mode == "smoke" else 900.0
    runner = run_command or _run_subprocess
    env = _subprocess_env()
    run_log: list[dict[str, Any]] = []
    for arm in ARMS:
        for spec in specs:
            cmd = benchmark_command(
                arm=arm,
                spec=spec,
                output_root=root / arm.arm_id / spec.matrix_id,
                automl_tuning=automl_tuning,
                cv_folds=cv_folds,
                benchmark_stage=benchmark_stage,
                estimator_timeout_sec=timeout,
                external_repo_path=external_repo_path,
                resume=resume,
                write_plots=write_plots,
                python_executable=python_executable or sys.executable,
            )
            start = time.perf_counter()
            returncode, output = runner(cmd, env)
            runtime = float(time.perf_counter() - start)
            run_log.append(
                {
                    "arm": arm.arm_id,
                    "matrix_id": spec.matrix_id,
                    "status": "ok" if int(returncode) == 0 else f"rc_{int(returncode)}",
                    "returncode": int(returncode),
                    "runtime_sec": runtime,
                    "command": shlex.join(tuple(cmd)),
                    "output_tail": str(output)[-2_000:],
                }
            )
    run_log_path = root / "run_log.csv"
    write_csv(run_log_path, run_log)
    result = aggregate_gmm_selector_outputs(root)
    write_json(
        root / "gmm_selector_confirmatory_manifest.json",
        {
            "mode": mode,
            "arms": [arm.__dict__ for arm in ARMS],
            "matrix_specs": [spec.__dict__ for spec in specs],
            "automl_tuning": automl_tuning,
            "cv_folds": int(cv_folds),
            "benchmark_stage": benchmark_stage,
            "estimator_timeout_sec": timeout,
            "include_heavy_gym": bool(include_heavy_gym),
        },
    )
    return result


def benchmark_command(
    *,
    arm: SelectorArm,
    spec: MatrixSpec,
    output_root: Path,
    automl_tuning: str,
    cv_folds: int,
    benchmark_stage: str,
    estimator_timeout_sec: float | None,
    external_repo_path: str | Path | None,
    resume: bool,
    write_plots: bool,
    python_executable: str,
) -> tuple[str, ...]:
    cmd: list[str] = [
        str(python_executable),
        "-m",
        "occupancy_ratio_benchmark.run",
        "--stage",
        str(benchmark_stage),
        "--settings",
        *spec.settings,
        "--estimators",
        "neural_network",
        "--neural-estimator-presets",
        "stable",
        "--seeds",
        *(str(int(seed)) for seed in spec.seeds),
        "--sample-sizes",
        *(str(int(size)) for size in spec.sample_sizes),
        "--gammas",
        *(str(float(gamma)) for gamma in spec.gammas),
        "--tune-cv",
        "--automl-tuning",
        str(automl_tuning),
        "--cv-folds",
        str(int(cv_folds)),
        "--no-google-dualdice",
        "--output-root",
        str(output_root),
    ]
    if spec.discrete_policy_shifts:
        cmd.extend(["--discrete-policy-shifts", *(str(float(shift)) for shift in spec.discrete_policy_shifts)])
    if spec.linear_gaussian_policy_shifts:
        cmd.extend(["--linear-gaussian-policy-shifts", *(str(float(shift)) for shift in spec.linear_gaussian_policy_shifts)])
    if any(str(setting) in GYM_SETTINGS for setting in spec.settings):
        cmd.extend(["--gym-target-value-rollouts", str(int(spec.gym_target_value_rollouts))])
    if estimator_timeout_sec is not None:
        cmd.extend(["--estimator-timeout-sec", str(float(estimator_timeout_sec))])
    if external_repo_path is not None:
        cmd.extend(["--external-repo-path", str(external_repo_path)])
    if not resume:
        cmd.append("--no-resume")
    if not write_plots:
        cmd.append("--no-plots")
    cmd.extend(arm.flags)
    return tuple(cmd)


def aggregate_gmm_selector_outputs(output_root: str | Path) -> GMMSelectorConfirmatoryResult:
    root = Path(output_root)
    results_rows, tuning_rows = load_gmm_selector_rows(root)
    selected = selected_tuning_by_arm_cell(tuning_rows)
    summary_rows = selector_summary_rows(results_rows, selected)
    regret_rows = cell_regret_rows(summary_rows)
    summary_with_regret = _merge_regrets(summary_rows, regret_rows)
    delta_rows = selection_delta_rows(summary_with_regret)
    summary_path = root / "selector_summary.csv"
    delta_path = root / "selection_delta.csv"
    regret_path = root / "cell_regret.csv"
    report_path = root / "selector_report.md"
    write_csv(summary_path, summary_with_regret)
    write_csv(delta_path, delta_rows)
    write_csv(regret_path, regret_rows)
    report_path.write_text(render_selector_report(summary_with_regret, delta_rows, regret_rows, root=root), encoding="utf-8")
    return GMMSelectorConfirmatoryResult(
        output_root=root,
        run_log_path=root / "run_log.csv",
        selector_summary_path=summary_path,
        selection_delta_path=delta_path,
        cell_regret_path=regret_path,
        selector_report_path=report_path,
    )


def load_gmm_selector_rows(output_root: str | Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    root = Path(output_root)
    results: list[dict[str, Any]] = []
    tuning: list[dict[str, Any]] = []
    for arm in ARMS:
        for path in sorted((root / arm.arm_id).glob("*/*/results.csv")):
            matrix_id = path.parts[-3]
            stage = path.parts[-2]
            for row in _read_csv(path):
                results.append({"arm": arm.arm_id, "matrix_id": matrix_id, "benchmark_stage": stage, **row})
        for path in sorted((root / arm.arm_id).glob("*/*/tuning_results.csv")):
            matrix_id = path.parts[-3]
            stage = path.parts[-2]
            for row in _read_csv(path):
                tuning.append({"arm": arm.arm_id, "matrix_id": matrix_id, "benchmark_stage": stage, **row})
    return results, tuning


def selected_tuning_by_arm_cell(tuning_rows: Iterable[dict[str, Any]]) -> dict[tuple[str, tuple[str, ...]], dict[str, Any]]:
    out: dict[tuple[str, tuple[str, ...]], dict[str, Any]] = {}
    ranks: dict[tuple[str, tuple[str, ...]], tuple[int, int]] = {}
    for order, row in enumerate(tuning_rows):
        if str(row.get("tuning_stage", "")) != "automl_candidate":
            continue
        if _to_float(row.get("selected"), 0.0) < 0.5:
            continue
        key = (str(row.get("arm", "")), _cell_key(row))
        stage_rank = 0 if str(row.get("budget_stage", "")) == "full" else 1
        rank = (stage_rank, order)
        if key not in ranks or rank < ranks[key]:
            ranks[key] = rank
            out[key] = dict(row)
    return out


def selector_summary_rows(
    result_rows: Iterable[dict[str, Any]],
    selected_rows: dict[tuple[str, tuple[str, ...]], dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    selected = selected_rows or {}
    out: list[dict[str, Any]] = []
    for row in result_rows:
        if str(row.get("estimator", "")) != "neural_network_stable":
            continue
        arm = str(row.get("arm", ""))
        cell = _cell_key(row)
        chosen = selected.get((arm, cell), {})
        ess = _first_finite(row, "effective_sample_size_fraction", "ess_fraction_final")
        clipping = _first_finite(row, "clipping_fraction", "projection_clipped_fraction_final")
        out.append(
            {
                "arm": arm,
                "matrix_id": cell[0],
                "setting": cell[1],
                "dataset_variant": cell[2],
                "policy_shift": cell[3],
                "gamma": cell[4],
                "sample_size": cell[5],
                "seed": cell[6],
                "status": row.get("status", ""),
                "selected_candidate_id": chosen.get("candidate_id", ""),
                "selected_candidate_label": chosen.get("candidate_label", ""),
                "selected_budget_stage": chosen.get("budget_stage", ""),
                "selected_score": _to_float(chosen.get("score")),
                "selection_risk": _to_float(chosen.get("metric_selection_risk")),
                "selection_risk_raw": _to_float(chosen.get("metric_selection_risk_raw")),
                "selection_effective_dim": _to_float(chosen.get("metric_selection_effective_dim")),
                "constraint_violated": _to_float(chosen.get("metric_constraint_violated"), 0.0),
                "constraint_all_violated": _to_float(chosen.get("metric_constraint_all_violated"), 0.0),
                "constraint_catastrophic_ess": _to_float(chosen.get("metric_constraint_catastrophic_ess"), 0.0),
                "constraint_clipping": _to_float(chosen.get("metric_constraint_clipping"), 0.0),
                "constraint_normalization": _to_float(chosen.get("metric_constraint_normalization"), 0.0),
                "constraint_near_uniform_collapse": _to_float(chosen.get("metric_constraint_near_uniform_collapse"), 0.0),
                "final_constraint_violated": _to_float(chosen.get("metric_final_constraint_violated"), 0.0),
                "final_constraint_near_uniform_collapse": _to_float(
                    chosen.get("metric_final_constraint_near_uniform_collapse"), 0.0
                ),
                "ratio_tv_behavior": _to_float(row.get("ratio_tv")),
                "ratio_l1_behavior": _to_float(row.get("ratio_l1")),
                "ratio_normalized_l1": _to_float(row.get("ratio_normalized_l1")),
                "ratio_corr": _to_float(row.get("ratio_corr")),
                "log_ratio_rmse_diagnostic": _to_float(row.get("log_ratio_rmse")),
                "ope_value_abs_error": _to_float(row.get("ope_value_abs_error")),
                "ope_value_abs_error_se_units": _to_float(row.get("ope_value_abs_error_se_units")),
                "ope_value_estimate": _to_float(row.get("ope_value_estimate")),
                "ope_value_target": _first_finite(row, "ope_value_target", "target_policy_value"),
                "ess_fraction": ess,
                "weight_cv": _to_float(row.get("weight_cv")),
                "weight_q99_to_median": _to_float(row.get("weight_q99_to_median")),
                "weight_max": _to_float(row.get("weight_max")),
                "clipping_fraction": clipping,
                "normalization_error": _to_float(row.get("normalization_error")),
                "runtime_sec": _to_float(row.get("runtime_sec")),
                "timeout_sec": _to_float(row.get("timeout_sec")),
                "error_type": row.get("error_type", ""),
                "skip_reason": row.get("skip_reason", ""),
            }
        )
    return out


def cell_regret_rows(summary_rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [dict(row) for row in summary_rows]
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(_cell_key(row), []).append(row)
    out: list[dict[str, Any]] = []
    for cell, group in sorted(grouped.items()):
        ok = [row for row in group if row.get("status") == "ok"]
        best_ratio = _best_row(ok, "ratio_tv_behavior")
        best_ope = _best_row(ok, "ope_value_abs_error")
        for row in group:
            ratio_value = _to_float(row.get("ratio_tv_behavior"))
            ope_value = _to_float(row.get("ope_value_abs_error"))
            best_ratio_value = _to_float(best_ratio.get("ratio_tv_behavior") if best_ratio else None)
            best_ope_value = _to_float(best_ope.get("ope_value_abs_error") if best_ope else None)
            out.append(
                {
                    "arm": row.get("arm", ""),
                    "matrix_id": cell[0],
                    "setting": cell[1],
                    "dataset_variant": cell[2],
                    "policy_shift": cell[3],
                    "gamma": cell[4],
                    "sample_size": cell[5],
                    "seed": cell[6],
                    "status": row.get("status", ""),
                    "best_completed_ratio_arm": "" if best_ratio is None else best_ratio.get("arm", ""),
                    "best_completed_ratio_tv": best_ratio_value,
                    "ratio_tv_regret_vs_best_completed_arm": (
                        ratio_value - best_ratio_value if np.isfinite(ratio_value) and np.isfinite(best_ratio_value) else np.nan
                    ),
                    "best_completed_ope_arm": "" if best_ope is None else best_ope.get("arm", ""),
                    "best_completed_ope_abs_error": best_ope_value,
                    "ope_regret_vs_best_completed_arm": (
                        ope_value - best_ope_value if np.isfinite(ope_value) and np.isfinite(best_ope_value) else np.nan
                    ),
                }
            )
    return out


def selection_delta_rows(summary_rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [dict(row) for row in summary_rows]
    legacy_by_cell = {_cell_key(row): row for row in rows if row.get("arm") == LEGACY_ARM_ID}
    out: list[dict[str, Any]] = []
    for row in rows:
        arm = str(row.get("arm", ""))
        if arm == LEGACY_ARM_ID:
            continue
        cell = _cell_key(row)
        legacy = legacy_by_cell.get(cell)
        if legacy is None:
            continue
        ratio_delta, ratio_outcome = _delta_outcome(row.get("ratio_tv_behavior"), legacy.get("ratio_tv_behavior"))
        ope_delta, ope_outcome = _delta_outcome(row.get("ope_value_abs_error"), legacy.get("ope_value_abs_error"))
        out.append(
            {
                "arm": arm,
                "matrix_id": cell[0],
                "setting": cell[1],
                "dataset_variant": cell[2],
                "policy_shift": cell[3],
                "gamma": cell[4],
                "sample_size": cell[5],
                "seed": cell[6],
                "legacy_candidate_id": legacy.get("selected_candidate_id", ""),
                "legacy_candidate_label": legacy.get("selected_candidate_label", ""),
                "arm_candidate_id": row.get("selected_candidate_id", ""),
                "arm_candidate_label": row.get("selected_candidate_label", ""),
                "changed_winner": str(row.get("selected_candidate_id", "")) != str(legacy.get("selected_candidate_id", "")),
                "legacy_ratio_tv_behavior": _to_float(legacy.get("ratio_tv_behavior")),
                "arm_ratio_tv_behavior": _to_float(row.get("ratio_tv_behavior")),
                "ratio_tv_delta_vs_legacy": ratio_delta,
                "ratio_outcome": ratio_outcome,
                "legacy_ratio_l1_behavior": _to_float(legacy.get("ratio_l1_behavior")),
                "arm_ratio_l1_behavior": _to_float(row.get("ratio_l1_behavior")),
                "legacy_ope_value_abs_error": _to_float(legacy.get("ope_value_abs_error")),
                "arm_ope_value_abs_error": _to_float(row.get("ope_value_abs_error")),
                "ope_delta_vs_legacy": ope_delta,
                "ope_outcome": ope_outcome,
                "legacy_ess_fraction": _to_float(legacy.get("ess_fraction")),
                "arm_ess_fraction": _to_float(row.get("ess_fraction")),
                "legacy_weight_cv": _to_float(legacy.get("weight_cv")),
                "arm_weight_cv": _to_float(row.get("weight_cv")),
                "legacy_runtime_sec": _to_float(legacy.get("runtime_sec")),
                "arm_runtime_sec": _to_float(row.get("runtime_sec")),
                "runtime_delta_vs_legacy": _to_float(row.get("runtime_sec")) - _to_float(legacy.get("runtime_sec")),
                "ratio_regret_vs_best_completed_arm": _to_float(row.get("ratio_tv_regret_vs_best_completed_arm")),
                "ope_regret_vs_best_completed_arm": _to_float(row.get("ope_regret_vs_best_completed_arm")),
            }
        )
    return out


def render_selector_report(
    summary_rows: Sequence[dict[str, Any]],
    delta_rows: Sequence[dict[str, Any]],
    regret_rows: Sequence[dict[str, Any]],
    *,
    root: Path,
) -> str:
    del regret_rows
    lines = [
        "# GMM Selector Confirmatory Benchmark",
        "",
        "Product defaults are unchanged: `legacy_rank` remains the default selector.",
        "Headline ratio metrics are behavior-distribution TV/L1; `log_ratio_rmse` is diagnostic only.",
        "",
        f"- Selector summary: `{root / 'selector_summary.csv'}`",
        f"- Selection deltas: `{root / 'selection_delta.csv'}`",
        f"- Cell regret: `{root / 'cell_regret.csv'}`",
        f"- Run log: `{root / 'run_log.csv'}`",
        "",
        "## Completion",
        "",
        _markdown_table(_completion_rows(summary_rows), ("arm", "ok", "timeout", "error", "skipped")),
        "",
        "## Arm Summary",
        "",
        _markdown_table(
            _arm_summary_rows(summary_rows),
            (
                "arm",
                "cells",
                "controlled_cells",
                "controlled_median_tv",
                "controlled_worst_tv_regret",
                "gym_cells",
                "gym_median_ope",
                "gym_median_se_units",
                "mean_runtime_sec",
            ),
        ),
        "",
        "## Wins Vs Legacy",
        "",
        _markdown_table(
            _delta_summary_rows(delta_rows),
            (
                "arm",
                "ratio_helped",
                "ratio_hurt",
                "ratio_tied",
                "ope_helped",
                "ope_hurt",
                "ope_tied",
                "changed_winner_cells",
                "mean_runtime_delta_sec",
            ),
        ),
        "",
        "## Safety Diagnostics",
        "",
        _markdown_table(
            _safety_summary_rows(summary_rows),
            (
                "arm",
                "timeout_count",
                "constraint_violated",
                "near_uniform_collapse",
                "clipping_constraint",
                "max_clipping_fraction",
                "max_weight_q99_to_median",
            ),
        ),
        "",
        "## Decision Checklist",
        "",
        *[f"- {line}" for line in _decision_checklist(summary_rows, delta_rows)],
        "",
        "## Changed-Winner Cells",
        "",
        _markdown_table(
            [row for row in delta_rows if bool(row.get("changed_winner"))],
            (
                "arm",
                "setting",
                "policy_shift",
                "gamma",
                "sample_size",
                "seed",
                "ratio_outcome",
                "ratio_tv_delta_vs_legacy",
                "ope_outcome",
                "ope_delta_vs_legacy",
                "legacy_candidate_label",
                "arm_candidate_label",
            ),
        ),
        "",
    ]
    return "\n".join(lines)


def _merge_regrets(summary_rows: Sequence[dict[str, Any]], regret_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    regret_by_arm_cell = {(str(row.get("arm", "")), _cell_key(row)): row for row in regret_rows}
    out = []
    for row in summary_rows:
        merged = dict(row)
        regret = regret_by_arm_cell.get((str(row.get("arm", "")), _cell_key(row)), {})
        for key in (
            "best_completed_ratio_arm",
            "best_completed_ratio_tv",
            "ratio_tv_regret_vs_best_completed_arm",
            "best_completed_ope_arm",
            "best_completed_ope_abs_error",
            "ope_regret_vs_best_completed_arm",
        ):
            merged[key] = regret.get(key, "")
        out.append(merged)
    return out


def _completion_rows(summary_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for arm in (arm.arm_id for arm in ARMS):
        group = [row for row in summary_rows if row.get("arm") == arm]
        out.append(
            {
                "arm": arm,
                "ok": sum(row.get("status") == "ok" for row in group),
                "timeout": sum(row.get("status") == "timeout" for row in group),
                "error": sum(row.get("status") == "error" for row in group),
                "skipped": sum(row.get("status") == "skipped" for row in group),
            }
        )
    return out


def _arm_summary_rows(summary_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for arm in (arm.arm_id for arm in ARMS):
        group = [row for row in summary_rows if row.get("arm") == arm and row.get("status") == "ok"]
        controlled = [row for row in group if str(row.get("setting")) in CONTROLLED_SETTINGS]
        gym = [row for row in group if str(row.get("setting")) in GYM_SETTINGS]
        out.append(
            {
                "arm": arm,
                "cells": len(group),
                "controlled_cells": len(controlled),
                "controlled_median_tv": _median(_to_float(row.get("ratio_tv_behavior")) for row in controlled),
                "controlled_worst_tv_regret": _max_finite(
                    _to_float(row.get("ratio_tv_regret_vs_best_completed_arm")) for row in controlled
                ),
                "gym_cells": len(gym),
                "gym_median_ope": _median(_to_float(row.get("ope_value_abs_error")) for row in gym),
                "gym_median_se_units": _median(_to_float(row.get("ope_value_abs_error_se_units")) for row in gym),
                "mean_runtime_sec": _mean(_to_float(row.get("runtime_sec")) for row in group),
            }
        )
    return out


def _delta_summary_rows(delta_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for arm in (arm.arm_id for arm in ARMS if arm.arm_id != LEGACY_ARM_ID):
        group = [row for row in delta_rows if row.get("arm") == arm]
        out.append(
            {
                "arm": arm,
                "ratio_helped": sum(row.get("ratio_outcome") == "helped" for row in group),
                "ratio_hurt": sum(row.get("ratio_outcome") == "hurt" for row in group),
                "ratio_tied": sum(row.get("ratio_outcome") == "tied" for row in group),
                "ope_helped": sum(row.get("ope_outcome") == "helped" for row in group),
                "ope_hurt": sum(row.get("ope_outcome") == "hurt" for row in group),
                "ope_tied": sum(row.get("ope_outcome") == "tied" for row in group),
                "changed_winner_cells": sum(bool(row.get("changed_winner")) for row in group),
                "mean_runtime_delta_sec": _mean(_to_float(row.get("runtime_delta_vs_legacy")) for row in group),
            }
        )
    return out


def _safety_summary_rows(summary_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for arm in (arm.arm_id for arm in ARMS):
        group = [row for row in summary_rows if row.get("arm") == arm]
        out.append(
            {
                "arm": arm,
                "timeout_count": sum(row.get("status") == "timeout" for row in group),
                "constraint_violated": sum(_to_float(row.get("constraint_violated"), 0.0) >= 0.5 for row in group),
                "near_uniform_collapse": sum(
                    max(
                        _to_float(row.get("constraint_near_uniform_collapse"), 0.0),
                        _to_float(row.get("final_constraint_near_uniform_collapse"), 0.0),
                    )
                    >= 0.5
                    for row in group
                ),
                "clipping_constraint": sum(_to_float(row.get("constraint_clipping"), 0.0) >= 0.5 for row in group),
                "max_clipping_fraction": _max_finite(_to_float(row.get("clipping_fraction")) for row in group),
                "max_weight_q99_to_median": _max_finite(_to_float(row.get("weight_q99_to_median")) for row in group),
            }
        )
    return out


def _decision_checklist(summary_rows: Sequence[dict[str, Any]], delta_rows: Sequence[dict[str, Any]]) -> list[str]:
    out = ["This report is evidence for a later promotion decision; it does not change product defaults."]
    for arm_id in ("gmm_ratio", "gmm_ope"):
        controlled = [
            row
            for row in summary_rows
            if row.get("arm") == arm_id and str(row.get("setting")) in CONTROLLED_SETTINGS and row.get("status") == "ok"
        ]
        gym = [
            row
            for row in summary_rows
            if row.get("arm") == arm_id and str(row.get("setting")) in GYM_SETTINGS and row.get("status") == "ok"
        ]
        deltas = [row for row in delta_rows if row.get("arm") == arm_id]
        ratio_helped = sum(row.get("ratio_outcome") == "helped" for row in deltas)
        ratio_hurt = sum(row.get("ratio_outcome") == "hurt" for row in deltas)
        ope_helped = sum(row.get("ope_outcome") == "helped" for row in deltas)
        ope_hurt = sum(row.get("ope_outcome") == "hurt" for row in deltas)
        out.append(
            f"`{arm_id}` controlled cells={len(controlled)}, gym cells={len(gym)}, "
            f"ratio helped/hurt={ratio_helped}/{ratio_hurt}, OPE helped/hurt={ope_helped}/{ope_hurt}."
        )
    out.append("Promotion requires GMM wins to exceed losses without worse timeout, clipping, collapse, or material runtime.")
    return out


def _delta_outcome(new_value: Any, old_value: Any) -> tuple[float, str]:
    new = _to_float(new_value)
    old = _to_float(old_value)
    if not (np.isfinite(new) and np.isfinite(old)):
        return np.nan, "na"
    delta = float(new - old)
    if abs(delta) <= DELTA_TOL:
        return delta, "tied"
    return delta, "helped" if delta < 0.0 else "hurt"


def _best_row(rows: Sequence[dict[str, Any]], metric: str) -> dict[str, Any] | None:
    finite = [row for row in rows if np.isfinite(_to_float(row.get(metric)))]
    if not finite:
        return None
    return min(finite, key=lambda row: _to_float(row.get(metric)))


def _cell_key(row: dict[str, Any]) -> tuple[str, ...]:
    return tuple(_clean_cell_value(row.get(field, "")) for field in CELL_FIELDS)


def _clean_cell_value(value: Any) -> str:
    if value is None:
        return ""
    try:
        if isinstance(value, float) and np.isnan(value):
            return ""
    except TypeError:
        pass
    text = str(value)
    return "" if text.lower() == "nan" else text


def _first_finite(row: dict[str, Any], *names: str) -> float:
    for name in names:
        value = _to_float(row.get(name))
        if np.isfinite(value):
            return value
    return float("nan")


def _to_float(value: Any, default: float = float("nan")) -> float:
    if value is None or value == "":
        return float(default)
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if np.isfinite(out) else float(default)


def _finite(values: Iterable[float]) -> list[float]:
    return [float(value) for value in values if np.isfinite(float(value))]


def _mean(values: Iterable[float]) -> float:
    finite = _finite(values)
    return float(np.mean(finite)) if finite else float("nan")


def _median(values: Iterable[float]) -> float:
    finite = _finite(values)
    return float(np.median(finite)) if finite else float("nan")


def _max_finite(values: Iterable[float]) -> float:
    finite = _finite(values)
    return float(np.max(finite)) if finite else float("nan")


def _fmt(value: Any) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    numeric = _to_float(value)
    if np.isfinite(numeric):
        if abs(numeric) >= 100.0:
            return f"{numeric:.1f}"
        if abs(numeric) >= 1.0:
            return f"{numeric:.4g}"
        return f"{numeric:.4f}"
    if isinstance(value, (float, np.floating)):
        return ""
    if value is None:
        return ""
    return str(value)


def _markdown_table(rows: Sequence[dict[str, Any]], columns: Sequence[str]) -> str:
    if not rows:
        return "_No rows._"
    widths = [len(column) for column in columns]
    body = []
    for row in rows:
        values = [_fmt(row.get(column, "")) for column in columns]
        body.append(values)
        for idx, value in enumerate(values):
            widths[idx] = max(widths[idx], len(value))
    header = "| " + " | ".join(column.ljust(widths[idx]) for idx, column in enumerate(columns)) + " |"
    sep = "| " + " | ".join("-" * width for width in widths) + " |"
    lines = [header, sep]
    for values in body:
        lines.append("| " + " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(values)) + " |")
    return "\n".join(lines)


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _run_subprocess(cmd: Sequence[str], env: dict[str, str]) -> tuple[int, str]:
    result = subprocess.run(tuple(cmd), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
    return int(result.returncode), str(result.stdout)


def _subprocess_env() -> dict[str, str]:
    env = dict(os.environ)
    if "MPLCONFIGDIR" not in env:
        cache_dir = Path(os.environ.get("TMPDIR", "/tmp")) / "rltools-matplotlib-cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        env["MPLCONFIGDIR"] = str(cache_dir)
    return env


def _gym_setting_available(setting: str) -> bool:
    try:
        import gymnasium as gym

        env = gym.make(GYM_CONTROL_SETTINGS[setting])
        env.close()
        return True
    except Exception:
        return False


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the frozen GMM selector confirmatory benchmark.")
    subparsers = parser.add_subparsers(dest="mode", required=True)
    for mode in ("smoke", "confirmatory"):
        sub = subparsers.add_parser(mode)
        default_root = "outputs/gmm_selector_confirmatory_smoke" if mode == "smoke" else "outputs/gmm_selector_confirmatory"
        sub.add_argument("--output-root", default=default_root)
        sub.add_argument("--automl-tuning", choices=("fast", "balanced"), default="fast")
        sub.add_argument("--cv-folds", type=int, default=3)
        sub.add_argument("--benchmark-stage", default="smoke")
        sub.add_argument("--estimator-timeout-sec", type=float, default=None)
        sub.add_argument("--external-repo-path", default=None)
        sub.add_argument("--include-heavy-gym", action="store_true")
        sub.add_argument("--no-resume", action="store_true")
        sub.add_argument("--write-plots", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = run_gmm_selector_confirmatory(
        mode=str(args.mode),
        output_root=args.output_root,
        include_heavy_gym=bool(args.include_heavy_gym),
        automl_tuning=str(args.automl_tuning),
        cv_folds=int(args.cv_folds),
        benchmark_stage=str(args.benchmark_stage),
        estimator_timeout_sec=args.estimator_timeout_sec,
        external_repo_path=args.external_repo_path,
        resume=not bool(args.no_resume),
        write_plots=bool(args.write_plots),
    )
    print(f"Wrote selector summary: {result.selector_summary_path}")
    print(f"Wrote selection delta: {result.selection_delta_path}")
    print(f"Wrote cell regret: {result.cell_regret_path}")
    print(f"Wrote selector report: {result.selector_report_path}")


if __name__ == "__main__":
    main()
