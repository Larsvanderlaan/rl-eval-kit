from __future__ import annotations

import argparse
import csv
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter
from typing import Any


for _thread_var in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_thread_var, "1")


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


from FQE_neurips.controlled_discounted_benchmark.configs import (  # noqa: E402
    EvaluationConfig,
    FQESolverConfig,
    NeuralFQEConfig,
    NeuralRatioConfig,
    RatioFeatureConfig,
    WeightEstimatorConfig,
    stationary_shift_paper_stage_grid,
)
from FQE_neurips.controlled_discounted_benchmark.run_experiment import (  # noqa: E402
    GAMMA_RESULT_COLUMNS,
    SUMMARY_EXTRA_COLUMNS,
    _aggregate_rows,
    _run_single_gamma_configuration,
    _write_csv,
    _write_gamma_sweep_report,
)


STAGE = "stationary_shift_paper"
PREFIX = "gamma_sweep"


def _optional_float(value: Any, default: float = -1.0) -> float:
    return default if value is None else float(value)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as handle:
        return list(csv.DictReader(handle))


def _task_key(task: dict[str, Any]) -> tuple[float, float, float, int, float, float, float, str, int]:
    return (
        float(task["value_gamma"]),
        float(task["ratio_gamma"]),
        float(task["shift"]),
        int(task["sample_size"]),
        float(task["process_noise_sd"]),
        _optional_float(task.get("target_action_sd")),
        float(task["behavior_action_sd"]),
        str(task["feature_regime"]),
        int(task["seed"]),
    )


def _row_key(row: dict[str, Any]) -> tuple[float, float, float, int, float, float, float, str, int]:
    return (
        float(row["value_gamma"]),
        float(row["ratio_gamma"]),
        float(row["shift"]),
        int(float(row["sample_size"])),
        float(row["process_noise_sd"]),
        _optional_float(row.get("target_action_sd")),
        float(row["behavior_action_sd"]),
        str(row["feature_regime"]),
        int(float(row["seed"])),
    )


def _build_tasks(
    max_configs: int | None,
    seed_count: int | None = None,
    sample_size_override: int | None = None,
    shifts_override: list[float] | None = None,
    feature_regimes_override: list[str] | None = None,
) -> list[dict[str, Any]]:
    grid = stationary_shift_paper_stage_grid()
    seeds = list(grid["seeds"])
    if seed_count is not None:
        seeds = seeds[: max(0, int(seed_count))]
    sample_sizes = [int(sample_size_override)] if sample_size_override is not None else list(grid["sample_sizes"])
    shifts = shifts_override if shifts_override is not None else list(grid["shifts"])
    feature_regimes = feature_regimes_override if feature_regimes_override is not None else list(grid["feature_regimes"])
    behavior_shift_direction_scale = float(grid.get("behavior_shift_direction_scale", 1.0))
    behavior_shift_direction = grid.get("behavior_shift_direction")
    tasks: list[dict[str, Any]] = []
    for value_gamma in grid["value_gammas"]:
        for ratio_gamma in grid["ratio_gammas"]:
            for shift in shifts:
                for sample_size in sample_sizes:
                    for process_noise_sd in grid["process_noise_sds"]:
                        for target_action_sd in grid.get("target_action_sds", [None]):
                            for behavior_action_sd in grid["behavior_action_sds"]:
                                for feature_regime in feature_regimes:
                                    for seed in seeds:
                                        tasks.append(
                                            {
                                                "value_gamma": float(value_gamma),
                                                "ratio_gamma": float(ratio_gamma),
                                                "shift": float(shift),
                                                "sample_size": int(sample_size),
                                                "process_noise_sd": float(process_noise_sd),
                                                "target_action_sd": None
                                                if target_action_sd is None
                                                else float(target_action_sd),
                                                "behavior_action_sd": float(behavior_action_sd),
                                                "feature_regime": str(feature_regime),
                                                "seed": int(seed),
                                                "behavior_shift_direction_scale": behavior_shift_direction_scale,
                                                "behavior_shift_direction": behavior_shift_direction,
                                            }
                                        )
                                        if max_configs is not None and len(tasks) >= max_configs:
                                            return tasks
    return tasks


def _run_task(task: dict[str, Any]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    _run_single_gamma_configuration(
        stage=STAGE,
        output_stage_rows=rows,
        value_gamma=float(task["value_gamma"]),
        ratio_gamma=float(task["ratio_gamma"]),
        shift=float(task["shift"]),
        sample_size=int(task["sample_size"]),
        process_noise_sd=float(task["process_noise_sd"]),
        target_action_sd=task.get("target_action_sd"),
        behavior_action_sd=float(task["behavior_action_sd"]),
        feature_regime=str(task["feature_regime"]),
        seed=int(task["seed"]),
        include_neural_ratio=False,
        include_neural_fqe=False,
        ratio_feature_config=RatioFeatureConfig(
            n_rbf_centers=int(task.get("ratio_rbf_centers", RatioFeatureConfig().n_rbf_centers)),
            standardize_features=bool(task.get("standardize_ratio_features", False)),
        ),
        fqe_solver_config=FQESolverConfig(),
        weight_config=WeightEstimatorConfig(),
        neural_ratio_config=NeuralRatioConfig(max_steps=1),
        neural_fqe_config=NeuralFQEConfig(n_outer_iters=1, epochs_per_iter=1),
        evaluation_config=EvaluationConfig(),
        behavior_shift_direction_scale=float(task.get("behavior_shift_direction_scale", 1.0)),
        behavior_shift_direction=task.get("behavior_shift_direction"),
        estimator_mode=str(task.get("estimator_mode", "full")),
    )
    return rows


def _write_outputs(output_root: Path, rows: list[dict[str, Any]]) -> tuple[Path, Path]:
    results_path = output_root / f"{PREFIX}_results.csv"
    summary_path = output_root / f"{PREFIX}_summary.csv"
    summary_rows = _aggregate_rows(rows, GAMMA_RESULT_COLUMNS, gamma_mode=True)
    _write_csv(results_path, rows, fieldnames=GAMMA_RESULT_COLUMNS)
    _write_csv(summary_path, summary_rows, fieldnames=GAMMA_RESULT_COLUMNS + SUMMARY_EXTRA_COLUMNS)
    _write_gamma_sweep_report(output_root / "gamma_sweep_report.md", summary_rows, STAGE)
    return results_path, summary_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Linear-only stationary-weight shift sweep for the paper composite figure."
    )
    parser.add_argument("--output-root", type=Path, default=Path("FQE_neurips/results/stationary_shift_paper_100"))
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--max-configs", type=int, default=None)
    parser.add_argument("--seed-count", type=int, default=None)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--shifts", type=str, default=None)
    parser.add_argument("--feature-regimes", type=str, default=None)
    parser.add_argument(
        "--estimator-mode",
        type=str,
        default="full",
        choices=["full", "outer_control", "minimax_sensitivity", "oracle_tuned_comparison"],
    )
    parser.add_argument("--ratio-rbf-centers", type=int, default=RatioFeatureConfig().n_rbf_centers)
    parser.add_argument("--standardize-ratio-features", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    results_path = args.output_root / f"{PREFIX}_results.csv"
    rows: list[dict[str, Any]] = []
    completed_keys: set[tuple[float, float, float, int, float, float, str, int]] = set()
    if args.resume and results_path.exists():
        rows.extend(_read_csv(results_path))
        completed_keys = {_row_key(row) for row in rows}

    shifts_override = None
    if args.shifts:
        shifts_override = [float(item) for item in args.shifts.split(",") if item.strip()]
    feature_regimes_override = None
    if args.feature_regimes:
        feature_regimes_override = [item.strip() for item in args.feature_regimes.split(",") if item.strip()]
    tasks = [
        {
            **task,
            "ratio_rbf_centers": int(args.ratio_rbf_centers),
            "standardize_ratio_features": bool(args.standardize_ratio_features),
            "estimator_mode": str(args.estimator_mode),
        }
        for task in _build_tasks(
            args.max_configs,
            seed_count=args.seed_count,
            sample_size_override=args.sample_size,
            shifts_override=shifts_override,
            feature_regimes_override=feature_regimes_override,
        )
        if _task_key(task) not in completed_keys
    ]
    total = len(tasks)
    if total == 0:
        results_path, summary_path = _write_outputs(args.output_root, rows)
        print(f"[{STAGE}] no new configurations; wrote {results_path} and {summary_path}", flush=True)
        return

    start = perf_counter()
    completed = 0
    print(
        f"[{STAGE}] running {total} linear-only configurations with n_jobs={max(1, args.n_jobs)}",
        flush=True,
    )
    if args.n_jobs <= 1:
        for task in tasks:
            rows.extend(_run_task(task))
            completed += 1
            if completed % args.checkpoint_every == 0 or completed == total:
                _write_outputs(args.output_root, rows)
                elapsed = perf_counter() - start
                print(f"[{STAGE}] completed {completed}/{total} configs in {elapsed:.1f}s", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.n_jobs) as executor:
            future_to_task = {executor.submit(_run_task, task): task for task in tasks}
            for future in as_completed(future_to_task):
                rows.extend(future.result())
                completed += 1
                if completed % args.checkpoint_every == 0 or completed == total:
                    _write_outputs(args.output_root, rows)
                    elapsed = perf_counter() - start
                    print(f"[{STAGE}] completed {completed}/{total} configs in {elapsed:.1f}s", flush=True)

    results_path, summary_path = _write_outputs(args.output_root, rows)
    print(f"[{STAGE}] wrote {results_path} and {summary_path}", flush=True)


if __name__ == "__main__":
    main()
