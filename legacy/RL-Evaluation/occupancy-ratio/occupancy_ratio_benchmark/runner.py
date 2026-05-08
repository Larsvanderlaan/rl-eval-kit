from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
from importlib import metadata
import multiprocessing as mp
import os
from pathlib import Path
import platform
import queue
import traceback
import time
from typing import Any

import numpy as np

from occupancy_ratio_benchmark.config import OccupancyRatioBenchmarkConfig
from occupancy_ratio_benchmark.diagnostics import summarize_rows
from occupancy_ratio_benchmark.discrete import make_discrete_dataset
from occupancy_ratio_benchmark.estimators import (
    EstimatorResult,
    GoogleDualDICEPreflight,
    run_estimator,
)
from occupancy_ratio_benchmark.external_baselines import preflight_google_dualdice, preflight_google_gridwalk
from occupancy_ratio_benchmark.gaussian import make_linear_gaussian_dataset
from occupancy_ratio_benchmark.gym_control import GYM_CONTROL_SETTINGS, make_gym_control_dataset
from occupancy_ratio_benchmark.io import write_csv, write_json
from occupancy_ratio_benchmark.nonlinear import make_nonlinear_dataset
from occupancy_ratio_benchmark.plots import write_plots
from occupancy_ratio_benchmark.tabular import (
    OptionalDatasetUnavailable,
    make_minari_dataset,
    make_obp_logged_bandit_dataset,
    make_openml_contextual_bandit_dataset,
    make_openml_finite_mdp_dataset,
)


@dataclass
class BenchmarkRunResult:
    output_dir: Path
    results_path: Path
    summary_path: Path
    diagnostics_path: Path
    manifest_path: Path
    winner_path: Path
    tuning_path: Path
    plot_status: str
    rows: list[dict[str, Any]]
    summary_rows: list[dict[str, Any]]
    winner_rows: list[dict[str, Any]]
    tuning_rows: list[dict[str, Any]]


def run_benchmark(config: OccupancyRatioBenchmarkConfig) -> BenchmarkRunResult:
    """Run the configured benchmark and write CSV/JSON diagnostics."""
    output_dir = config.output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    _ensure_runtime_cache_dir(output_dir)
    partial_results_path = output_dir / "results.partial.csv"
    partial_tuning_path = output_dir / "tuning_results.partial.csv"
    google_preflight = (
        preflight_google_dualdice(config.external_repo_path)
        if config.include_google_dual_dice
        else GoogleDualDICEPreflight(False, "Google DualDICE disabled by config.", config.external_repo_path)
    )
    if config.stage in {"full", "overnight"} and config.include_google_dual_dice and not google_preflight.available:
        raise RuntimeError(google_preflight.reason)

    rows: list[dict[str, Any]] = []
    tuning_rows: list[dict[str, Any]] = []
    if bool(config.resume):
        rows = _read_csv_rows(partial_results_path)
        tuning_rows = _read_csv_rows(partial_tuning_path)
    completed_keys = {_completion_key(row) for row in rows if row.get("status") not in {"", None}}
    needs_gridwalk = bool(config.include_dualdice_gridwalk) or (
        "google_tabular_dualdice_gridwalk" in config.resolved_estimators()
    )
    gridwalk_preflight = (
        preflight_google_gridwalk(config.external_repo_path)
        if needs_gridwalk
        else GoogleDualDICEPreflight(False, "Google GridWalk disabled by config.", config.external_repo_path)
    )
    diagnostics: dict[str, Any] = {
        "google_dualdice_preflight": asdict(google_preflight),
        "google_gridwalk_preflight": asdict(gridwalk_preflight),
        "failures": [],
    }
    torch_guard = _apply_torch_thread_guard()
    diagnostics["torch_thread_guard"] = torch_guard

    if config.include_dualdice_gridwalk:
        if not gridwalk_preflight.available:
            rows.append(
                {
                    "stage": config.stage,
                    "profile": str(config.profile),
                    "setting": "gridwalk_tabular",
                    "estimator": "google_tabular_dualdice_gridwalk",
                    "status": "skipped",
                    "skip_reason": gridwalk_preflight.reason,
                    "gamma": "",
                    "seed": "",
                    "sample_size": "",
                    "runtime_sec": 0.0,
                }
            )
        else:
            try:
                from occupancy_ratio_benchmark.dualdice_grid import GridBenchmarkConfig, run_gridwalk_benchmark

                grid_rows = run_gridwalk_benchmark(
                    GridBenchmarkConfig(
                        output_root=output_dir,
                        google_research_root=config.external_repo_path,
                        seeds=tuple(int(seed) for seed in config.seeds),
                        alphas=(0.0, 0.5),
                        gammas=tuple(float(gamma) for gamma in config.gammas),
                        num_trajectories=20 if config.stage == "smoke" else 50,
                        max_trajectory_length=50 if config.stage == "smoke" else 100,
                        boosted_losses=("huber",),
                        boosted_num_iterations=int(config.boosted_num_iterations),
                        boosted_mcmc_samples=int(config.boosted_mcmc_samples),
                        include_neural=any(str(name).startswith("neural_network") for name in config.resolved_estimators()),
                        neural_num_iterations=int(config.neural_num_iterations),
                        neural_mcmc_samples=int(config.neural_mcmc_samples),
                        neural_gradient_steps_per_iteration=int(config.neural_gradient_steps_per_iteration),
                        neural_action_steps=int(config.neural_action_steps),
                        neural_transition_steps=int(config.neural_transition_steps),
                        neural_batch_size=int(config.neural_batch_size),
                        neural_hidden_dim=int(tuple(config.neural_hidden_dims)[0]),
                        huber_delta_scale=float(config.huber_delta_scale),
                        estimator_timeout_sec=60.0 if config.stage == "smoke" else float(config.estimator_timeout_sec or 300.0),
                    )
                )
                for row in grid_rows:
                    rows.append({"stage": config.stage, "profile": str(config.profile), **row})
            except Exception as exc:
                rows.append(_error_row(config, "gridwalk_tabular", 0, 0.0, 0, "google_tabular_dualdice_gridwalk", exc))
                diagnostics["failures"].append(traceback.format_exc())

    for setting in config.settings:
        for sample_size in config.sample_sizes:
            for gamma in config.gammas:
                for seed in config.seeds:
                    for policy_shift in _policy_shifts_for_setting(config, setting):
                        for dataset_variant in _dataset_variants_for_setting(config, setting):
                            print(
                                "dataset "
                                f"profile={config.profile} setting={setting} variant={dataset_variant or ''} "
                                f"n={sample_size} gamma={gamma} seed={seed} policy_shift={policy_shift}",
                                flush=True,
                            )
                            try:
                                dataset = make_dataset(
                                    setting=setting,
                                    gamma=float(gamma),
                                    sample_size=int(sample_size),
                                    seed=int(seed),
                                    config=config,
                                    policy_shift=policy_shift,
                                    dataset_variant=dataset_variant,
                                )
                            except OptionalDatasetUnavailable as exc:
                                for estimator in _expanded_estimators(config):
                                    planned_key = _planned_key(
                                        config,
                                        setting,
                                        sample_size,
                                        gamma,
                                        seed,
                                        policy_shift,
                                        dataset_variant,
                                        estimator,
                                    )
                                    if planned_key in completed_keys:
                                        continue
                                    rows.append(
                                        _skip_row(
                                            config,
                                            setting,
                                            sample_size,
                                            gamma,
                                            seed,
                                            estimator,
                                            str(exc),
                                            dataset_variant=dataset_variant,
                                            policy_shift=policy_shift,
                                        )
                                    )
                                    completed_keys.add(planned_key)
                                write_csv(partial_results_path, rows)
                                continue
                            except Exception as exc:
                                rows.append(
                                    _error_row(
                                        config,
                                        setting,
                                        sample_size,
                                        gamma,
                                        seed,
                                        "dataset",
                                        exc,
                                        dataset_variant=dataset_variant,
                                        policy_shift=policy_shift,
                                    )
                                )
                                diagnostics["failures"].append(traceback.format_exc())
                                write_csv(partial_results_path, rows)
                                continue

                            for estimator in _expanded_estimators(config):
                                planned_key = _planned_key(
                                    config,
                                    setting,
                                    sample_size,
                                    gamma,
                                    seed,
                                    policy_shift,
                                    dataset_variant,
                                    estimator,
                                )
                                if planned_key in completed_keys:
                                    print(f"  estimator {estimator} (resume skip)", flush=True)
                                    continue
                                print(f"  estimator {estimator}", flush=True)
                                try:
                                    result = _run_estimator_maybe_timeout(estimator, dataset, config, google_preflight)
                                    rows.append(_row_from_result(config, dataset, result))
                                    tuning_rows.extend(result.tuning_rows)
                                    completed_keys.add(planned_key)
                                    write_csv(partial_results_path, rows)
                                    write_csv(partial_tuning_path, tuning_rows)
                                except Exception as exc:
                                    rows.append(
                                        _error_row(
                                            config,
                                            setting,
                                            sample_size,
                                            gamma,
                                            seed,
                                            estimator,
                                            exc,
                                            dataset_variant=dataset_variant,
                                            policy_shift=policy_shift,
                                        )
                                    )
                                    diagnostics["failures"].append(traceback.format_exc())
                                    completed_keys.add(planned_key)
                                    write_csv(partial_results_path, rows)

    summary = summarize_rows(rows)
    winners = make_winner_table(rows)
    results_path = output_dir / "results.csv"
    summary_path = output_dir / "summary.csv"
    winner_path = output_dir / "winner_table.csv"
    tuning_path = output_dir / "tuning_results.csv"
    diagnostics_path = output_dir / "diagnostics.json"
    manifest_path = output_dir / "manifest.json"
    write_csv(results_path, rows)
    write_csv(summary_path, summary)
    write_csv(winner_path, winners)
    write_csv(tuning_path, tuning_rows)
    write_json(diagnostics_path, diagnostics)
    write_json(manifest_path, _manifest(config))
    plot_status = write_plots(output_dir, rows) if config.write_plots else "plotting disabled"
    return BenchmarkRunResult(
        output_dir=output_dir,
        results_path=results_path,
        summary_path=summary_path,
        diagnostics_path=diagnostics_path,
        manifest_path=manifest_path,
        winner_path=winner_path,
        tuning_path=tuning_path,
        plot_status=plot_status,
        rows=rows,
        summary_rows=summary,
        winner_rows=winners,
        tuning_rows=tuning_rows,
    )


def make_dataset(
    *,
    setting: str,
    gamma: float,
    sample_size: int,
    seed: int,
    config: OccupancyRatioBenchmarkConfig,
    policy_shift: float | None = None,
    dataset_variant: str | None = None,
):
    if setting in {"discrete_chain", "discrete_grid"}:
        return make_discrete_dataset(setting=setting, gamma=gamma, sample_size=sample_size, seed=seed)
    if setting == "linear_gaussian":
        shift = 1.0 if policy_shift is None else float(policy_shift)
        return make_linear_gaussian_dataset(gamma=gamma, sample_size=sample_size, seed=seed, policy_shift=shift)
    if setting == "nonlinear_monte_carlo":
        return make_nonlinear_dataset(
            gamma=gamma,
            sample_size=sample_size,
            seed=seed,
            mc_truth_samples=int(config.mc_truth_samples),
        )
    if setting in GYM_CONTROL_SETTINGS:
        return make_gym_control_dataset(
            setting=setting,
            gamma=gamma,
            sample_size=sample_size,
            seed=seed,
            target_value_rollouts=int(config.gym_target_value_rollouts),
        )
    if setting == "openml_contextual_bandit":
        return make_openml_contextual_bandit_dataset(
            task_id=int(dataset_variant if dataset_variant is not None else tuple(config.openml_task_ids)[0]),
            gamma=gamma,
            sample_size=sample_size,
            seed=seed,
        )
    if setting == "openml_finite_mdp":
        cap = int(config.tabular_state_cap or (256 if config.stage == "medium" else 512))
        return make_openml_finite_mdp_dataset(
            task_id=int(dataset_variant if dataset_variant is not None else tuple(config.openml_task_ids)[0]),
            gamma=gamma,
            sample_size=sample_size,
            seed=seed,
            state_cap=cap,
        )
    if setting == "obp_logged_bandit":
        return make_obp_logged_bandit_dataset(
            campaign=str(dataset_variant if dataset_variant is not None else tuple(config.obp_campaigns)[0]),
            gamma=gamma,
            sample_size=sample_size,
            seed=seed,
        )
    if setting in {"minari_pointmaze", "minari_minigrid"}:
        candidates = _dataset_variants_for_setting(config, setting)
        return make_minari_dataset(
            setting=setting,
            dataset_id=str(dataset_variant if dataset_variant is not None else candidates[0]),
            gamma=gamma,
            sample_size=sample_size,
            seed=seed,
        )
    raise ValueError(f"Unknown setting '{setting}'.")


def _policy_shifts_for_setting(config: OccupancyRatioBenchmarkConfig, setting: str) -> tuple[float | None, ...]:
    if setting == "linear_gaussian":
        return tuple(float(shift) for shift in config.linear_gaussian_policy_shifts)
    return (None,)


def _dataset_variants_for_setting(config: OccupancyRatioBenchmarkConfig, setting: str) -> tuple[str | None, ...]:
    if setting in {"openml_contextual_bandit", "openml_finite_mdp"}:
        task_ids = tuple(str(int(task_id)) for task_id in config.openml_task_ids)
        if config.openml_max_tasks is not None:
            task_ids = task_ids[: int(config.openml_max_tasks)]
        return task_ids or (None,)
    if setting == "obp_logged_bandit":
        return tuple(str(campaign) for campaign in config.obp_campaigns) or (None,)
    if setting == "minari_pointmaze":
        out = tuple(str(dataset_id) for dataset_id in config.minari_dataset_ids if "pointmaze" in str(dataset_id).lower())
        return out or (None,)
    if setting == "minari_minigrid":
        out = tuple(str(dataset_id) for dataset_id in config.minari_dataset_ids if "minigrid" in str(dataset_id).lower())
        return out or (None,)
    return (None,)


def _expanded_estimators(config: OccupancyRatioBenchmarkConfig) -> tuple[str, ...]:
    expanded = []
    for estimator in config.resolved_estimators():
        if estimator == "boosted_tree":
            variants = tuple(config.boosted_stabilization_presets) or tuple(config.boosted_estimator_presets)
            expanded.extend(f"boosted_tree_{_canonical_boosted_variant(variant)}" for variant in variants)
        elif estimator == "neural_network":
            variants = tuple(config.neural_stabilization_presets) or tuple(config.neural_estimator_presets)
            expanded.extend(f"neural_network_{_canonical_neural_variant(variant)}" for variant in variants)
        elif estimator == "google_dualdice":
            expanded.append("google_dualdice_neural")
        else:
            expanded.append(estimator)
    return tuple(dict.fromkeys(expanded))


def _canonical_boosted_variant(variant: str) -> str:
    aliases = {
        "huber_projection_damping": "stable",
        "huber_projection_damping_transition_norm": "transition_norm",
    }
    return aliases.get(str(variant), str(variant))


def _canonical_neural_variant(variant: str) -> str:
    aliases = {
        "huber_projection_damping": "stable",
        "huber_projection_damping_transition_norm": "transition_norm",
    }
    return aliases.get(str(variant), str(variant))


def _row_from_result(
    config: OccupancyRatioBenchmarkConfig,
    dataset,
    result: EstimatorResult,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "stage": config.stage,
        "profile": str(config.profile),
        "setting": dataset.setting,
        "estimator": result.estimator,
        "status": result.status,
        "skip_reason": result.skip_reason,
        "gamma": float(dataset.gamma),
        "seed": int(dataset.seed),
        "sample_size": int(dataset.sample_size),
        "runtime_sec": float(result.runtime_sec),
    }
    row.update(dataset.metadata)
    row.update(result.diagnostics)
    return row


def make_winner_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select the lowest-error successful non-oracle estimator per benchmark cell."""
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("status") != "ok" or row.get("estimator") in {"oracle", "behavior"}:
            continue
        key = (
            row.get("profile", row.get("stage")),
            row.get("setting"),
            row.get("dataset_variant", ""),
            row.get("policy_shift", ""),
            row.get("gamma", ""),
            row.get("sample_size", ""),
        )
        groups.setdefault(key, []).append(row)

    winners: list[dict[str, Any]] = []
    for key, group in groups.items():
        scored = []
        for row in group:
            metric_name = _primary_metric_name(row)
            if metric_name is None:
                continue
            try:
                metric_value = float(row[metric_name])
            except (TypeError, ValueError):
                continue
            if not np.isfinite(metric_value):
                continue
            scored.append((metric_value, metric_name, row))
        if not scored:
            continue
        metric_value, metric_name, row = min(scored, key=lambda item: item[0])
        winners.append(
            {
                "profile": key[0],
                "setting": key[1],
                "dataset_variant": key[2],
                "policy_shift": key[3],
                "gamma": key[4],
                "sample_size": key[5],
                "winning_estimator": row.get("estimator"),
                "primary_metric": metric_name,
                "primary_metric_value": float(metric_value),
                "runtime_sec": row.get("runtime_sec", ""),
                "effective_sample_size_fraction": row.get("effective_sample_size_fraction", ""),
            }
        )
    return winners


def _primary_metric_name(row: dict[str, Any]) -> str | None:
    for name in ("ope_value_abs_error", "ratio_rel_mse", "absolute_error", "log_ratio_rmse"):
        if name in row and row[name] != "":
            return name
    return None


def _run_estimator_maybe_timeout(
    estimator: str,
    dataset,
    config: OccupancyRatioBenchmarkConfig,
    google_preflight: GoogleDualDICEPreflight,
) -> EstimatorResult:
    timeout = config.estimator_timeout_sec
    if timeout is None:
        return run_estimator(estimator, dataset, config, google_preflight)
    timeout = float(timeout)
    # Spawn is slower than fork, but avoids macOS ObjC/MPS crashes when a
    # timed estimator imports or trains with PyTorch/TensorFlow in the child.
    ctx_name = "spawn" if "spawn" in mp.get_all_start_methods() else mp.get_all_start_methods()[0]
    ctx = mp.get_context(ctx_name)
    result_queue = ctx.Queue(maxsize=1)
    process = ctx.Process(
        target=_estimator_worker,
        args=(result_queue, estimator, dataset, config, google_preflight),
    )
    start = time.perf_counter()
    process.start()
    process.join(timeout)
    runtime = float(time.perf_counter() - start)
    if process.is_alive():
        process.terminate()
        process.join(5.0)
        return EstimatorResult(
            estimator=estimator,
            status="timeout",
            weights=None,
            raw_weights=None,
            runtime_sec=runtime,
            diagnostics={},
            skip_reason=f"Estimator exceeded {timeout:g} seconds.",
        )
    try:
        status, payload = result_queue.get(timeout=1.0)
    except queue.Empty:
        if process.exitcode == 0:
            raise RuntimeError("Estimator process exited without returning a result.")
        raise RuntimeError(f"Estimator process exited with code {process.exitcode}.")
    if status == "ok":
        return payload
    raise RuntimeError(str(payload))


def _estimator_worker(
    result_queue,
    estimator: str,
    dataset,
    config: OccupancyRatioBenchmarkConfig,
    google_preflight: GoogleDualDICEPreflight,
) -> None:
    try:
        result_queue.put(("ok", run_estimator(estimator, dataset, config, google_preflight)))
    except Exception:
        result_queue.put(("error", traceback.format_exc()))


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _completion_key(row: dict[str, Any]) -> tuple[str, str, str, str, str, str, str, str]:
    return (
        str(row.get("profile", row.get("stage", ""))),
        str(row.get("setting", "")),
        str(row.get("sample_size", "")),
        str(row.get("gamma", "")),
        str(row.get("seed", "")),
        str(row.get("policy_shift", "")),
        str(row.get("dataset_variant", "")),
        str(row.get("estimator", "")),
    )


def _planned_key(
    config: OccupancyRatioBenchmarkConfig,
    setting: str,
    sample_size: int,
    gamma: float,
    seed: int,
    policy_shift: float | None,
    dataset_variant: str | None,
    estimator: str,
) -> tuple[str, str, str, str, str, str, str, str]:
    return (
        str(config.profile),
        str(setting),
        str(int(sample_size)),
        str(float(gamma)),
        str(int(seed)),
        "" if policy_shift is None else str(float(policy_shift)),
        "" if dataset_variant is None else str(dataset_variant),
        str(estimator),
    )


def _apply_torch_thread_guard() -> dict[str, Any]:
    out: dict[str, Any] = {"attempted": True, "applied": False}
    try:
        import torch

        torch.set_num_threads(1)
        interop_applied = True
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            interop_applied = False
        out.update(
            {
                "applied": True,
                "num_threads": int(torch.get_num_threads()),
                "num_interop_threads": int(torch.get_num_interop_threads()),
                "interop_applied": bool(interop_applied),
            }
        )
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def _error_row(
    config: OccupancyRatioBenchmarkConfig,
    setting: str,
    sample_size: int,
    gamma: float,
    seed: int,
    estimator: str,
    exc: Exception,
    dataset_variant: str | None = None,
    policy_shift: float | None = None,
) -> dict[str, Any]:
    row = {
        "stage": config.stage,
        "profile": str(config.profile),
        "setting": setting,
        "estimator": estimator,
        "status": "error",
        "skip_reason": f"{type(exc).__name__}: {exc}",
        "gamma": float(gamma),
        "seed": int(seed),
        "sample_size": int(sample_size),
        "runtime_sec": 0.0,
    }
    if dataset_variant is not None:
        row["dataset_variant"] = str(dataset_variant)
    if policy_shift is not None:
        row["policy_shift"] = float(policy_shift)
    return row


def _skip_row(
    config: OccupancyRatioBenchmarkConfig,
    setting: str,
    sample_size: int,
    gamma: float,
    seed: int,
    estimator: str,
    reason: str,
    *,
    dataset_variant: str | None = None,
    policy_shift: float | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "stage": config.stage,
        "profile": str(config.profile),
        "setting": setting,
        "estimator": estimator,
        "status": "skipped",
        "skip_reason": reason,
        "gamma": float(gamma),
        "seed": int(seed),
        "sample_size": int(sample_size),
        "runtime_sec": 0.0,
    }
    if dataset_variant is not None:
        row["dataset_variant"] = str(dataset_variant)
    if policy_shift is not None:
        row["policy_shift"] = float(policy_shift)
    return row


def _manifest(config: OccupancyRatioBenchmarkConfig) -> dict[str, Any]:
    packages = {}
    for name in (
        "numpy",
        "lightgbm",
        "torch",
        "tensorflow",
        "tensorflow-addons",
        "matplotlib",
        "openml",
        "sklearn",
        "obp",
        "minari",
    ):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "config": asdict(config),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "torch_thread_guard": _apply_torch_thread_guard(),
        "randomness": {
            "seeds": list(map(int, config.seeds)),
            "note": "Each setting/estimator receives deterministic NumPy seeds derived from config seed.",
        },
    }


def _ensure_runtime_cache_dir(output_dir: Path) -> None:
    if "MPLCONFIGDIR" in os.environ:
        return
    cache_dir = output_dir / ".matplotlib-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache_dir)
