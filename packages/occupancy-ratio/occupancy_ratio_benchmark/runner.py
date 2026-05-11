from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
from importlib import metadata
import multiprocessing as mp
import os
from pathlib import Path
import platform
import queue
import subprocess
import traceback
import time
from typing import Any

import numpy as np

from occupancy_ratio_benchmark.application_simulators import (
    APPLICATION_SIMULATOR_SETTINGS,
    make_application_simulator_dataset,
)
from occupancy_ratio_benchmark.config import OccupancyRatioBenchmarkConfig
from occupancy_ratio_benchmark.diagnostics import summarize_rows
from occupancy_ratio_benchmark.discrete import make_discrete_dataset
from occupancy_ratio_benchmark.estimators import (
    EstimatorResult,
    GoogleDICERLPreflight,
    GoogleDualDICEPreflight,
    run_estimator,
)
from occupancy_ratio_benchmark.external_baselines import (
    preflight_google_dice_rl,
    preflight_google_dualdice,
    preflight_google_gridwalk,
)
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
    high_stakes_recommendation_path: Path
    defaults_report_path: Path
    neural_vs_dice_path: Path
    conservatism_audit_path: Path
    conservatism_report_path: Path
    benchmark_readout_path: Path
    plot_status: str
    rows: list[dict[str, Any]]
    summary_rows: list[dict[str, Any]]
    winner_rows: list[dict[str, Any]]
    tuning_rows: list[dict[str, Any]]
    high_stakes_recommendation_rows: list[dict[str, Any]]


def run_benchmark(config: OccupancyRatioBenchmarkConfig) -> BenchmarkRunResult:
    """Run the configured benchmark and write CSV/JSON diagnostics."""
    output_dir = config.output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    _ensure_runtime_cache_dir(output_dir)
    run_metadata = _run_metadata(config)
    partial_results_path = output_dir / "results.partial.csv"
    partial_tuning_path = output_dir / "tuning_results.partial.csv"
    google_preflight = (
        preflight_google_dualdice(config.external_repo_path)
        if config.include_google_dual_dice
        else GoogleDualDICEPreflight(False, "Google DualDICE disabled by config.", config.external_repo_path)
    )
    if config.stage in {"full", "overnight", "high_stakes"} and config.include_google_dual_dice and not google_preflight.available:
        raise RuntimeError(google_preflight.reason)
    needs_dice_rl = _needs_dice_rl(config)
    dice_rl_preflight = (
        preflight_google_dice_rl(config.dice_rl_repo_path)
        if needs_dice_rl and config.include_dice_rl
        else GoogleDICERLPreflight(False, "Google DICE-RL disabled by config.", config.dice_rl_repo_path)
    )
    if config.stage in {"full", "overnight", "high_stakes"} and needs_dice_rl and not dice_rl_preflight.available:
        raise RuntimeError(dice_rl_preflight.reason)

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
        "run_metadata": run_metadata,
        "google_dualdice_preflight": asdict(google_preflight),
        "google_dice_rl_preflight": asdict(dice_rl_preflight),
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
                    "error_type": "",
                    "error_message": "",
                    "timeout_sec": "",
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
                        alphas=tuple(float(alpha) for alpha in config.gridwalk_alphas),
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
                        include_bellman_moment_calibration=any(
                            "bellman_moment_calibrated" in str(name)
                            for name in _expanded_estimators(config)
                        ),
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
                                    result = _run_estimator_maybe_timeout(estimator, dataset, config, google_preflight, dice_rl_preflight)
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
    high_stakes_recommendations = make_high_stakes_recommendations(rows) if config.profile == "high_stakes" else []
    results_path = output_dir / "results.csv"
    summary_path = output_dir / "summary.csv"
    winner_path = output_dir / "winner_table.csv"
    tuning_path = output_dir / "tuning_results.csv"
    high_stakes_recommendation_path = output_dir / "high_stakes_recommendations.csv"
    diagnostics_path = output_dir / "diagnostics.json"
    manifest_path = output_dir / "manifest.json"
    write_csv(results_path, rows)
    write_csv(summary_path, summary)
    write_csv(winner_path, winners)
    write_csv(tuning_path, tuning_rows)
    write_csv(high_stakes_recommendation_path, high_stakes_recommendations)
    write_json(diagnostics_path, diagnostics)
    write_json(manifest_path, _manifest(config, run_metadata=run_metadata))
    defaults_paths = _write_defaults_report(results_path, output_dir)
    conservatism_paths = _write_conservatism_audit(rows, output_dir)
    benchmark_readout_path = output_dir / "benchmark_readout.md"
    benchmark_readout_path.write_text(
        _render_benchmark_readout(
            config=config,
            rows=rows,
            summary=summary,
            winners=winners,
            diagnostics=diagnostics,
            defaults_paths=defaults_paths,
            conservatism_paths=conservatism_paths,
        ),
        encoding="utf-8",
    )
    plot_status = write_plots(output_dir, rows) if config.write_plots else "plotting disabled"
    return BenchmarkRunResult(
        output_dir=output_dir,
        results_path=results_path,
        summary_path=summary_path,
        diagnostics_path=diagnostics_path,
        manifest_path=manifest_path,
        winner_path=winner_path,
        tuning_path=tuning_path,
        high_stakes_recommendation_path=high_stakes_recommendation_path,
        defaults_report_path=defaults_paths.get("report", output_dir / "defaults_report.md"),
        neural_vs_dice_path=defaults_paths.get("neural_vs_dice", output_dir / "defaults_neural_vs_dice.csv"),
        conservatism_audit_path=conservatism_paths.get("audit", output_dir / "conservatism_audit.csv"),
        conservatism_report_path=conservatism_paths.get("report", output_dir / "conservatism_audit.md"),
        benchmark_readout_path=benchmark_readout_path,
        plot_status=plot_status,
        rows=rows,
        summary_rows=summary,
        winner_rows=winners,
        tuning_rows=tuning_rows,
        high_stakes_recommendation_rows=high_stakes_recommendations,
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
    if setting in {"discrete_chain", "discrete_grid", "random_tabular_mdp"}:
        n_states = None
        n_actions = None
        if setting == "random_tabular_mdp" and dataset_variant:
            parts = str(dataset_variant).lower().split("x", maxsplit=1)
            if len(parts) == 2:
                n_states, n_actions = int(parts[0]), int(parts[1])
        return make_discrete_dataset(
            setting=setting,
            gamma=gamma,
            sample_size=sample_size,
            seed=seed,
            policy_shift=policy_shift,
            n_states=n_states,
            n_actions=n_actions,
        )
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
    if setting in APPLICATION_SIMULATOR_SETTINGS:
        return make_application_simulator_dataset(
            setting=setting,
            gamma=gamma,
            sample_size=sample_size,
            seed=seed,
            target_value_rollouts=int(config.application_target_value_rollouts),
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
    if setting in {"discrete_chain", "discrete_grid", "random_tabular_mdp"}:
        shifts = tuple(float(shift) for shift in config.discrete_policy_shifts)
        return shifts if shifts else (None,)
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
    if setting == "random_tabular_mdp":
        return tuple(
            f"{int(n_states)}x{int(n_actions)}"
            for n_states in config.random_tabular_state_counts
            for n_actions in config.random_tabular_action_counts
        )
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
        "error_type": "",
        "error_message": "",
        "timeout_sec": "",
        "gamma": float(dataset.gamma),
        "seed": int(dataset.seed),
        "sample_size": int(dataset.sample_size),
        "runtime_sec": float(result.runtime_sec),
    }
    row.update(dataset.metadata)
    row.update(result.diagnostics)
    if result.status == "timeout":
        row["error_type"] = "TimeoutError"
        row["error_message"] = result.skip_reason
        row["timeout_sec"] = "" if config.estimator_timeout_sec is None else float(config.estimator_timeout_sec)
    elif result.status not in {"ok", "skipped"}:
        row["error_type"] = str(result.status)
        row["error_message"] = result.skip_reason
    if config.profile == "high_stakes":
        row.update(_high_stakes_diagnostic_status(row))
    return row


def make_winner_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select the lowest-error successful non-oracle estimator per benchmark cell."""
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("status") != "ok" or not _is_deployable_estimator(str(row.get("estimator", ""))):
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


def make_high_stakes_recommendations(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidate_estimators = {
        "boosted_tree_stable",
        "boosted_tree_relaxed_tail",
        "boosted_tree_auto",
        "neural_network_stable",
        "neural_network_relaxed_tail",
        "neural_network_google_parity",
        "neural_network_auto",
        "google_dualdice_neural",
    }
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("estimator") not in candidate_estimators:
            continue
        key = (
            row.get("profile", row.get("stage")),
            row.get("setting"),
            row.get("dataset_variant", ""),
            row.get("policy_shift", ""),
            row.get("gamma", ""),
            row.get("sample_size", ""),
            row.get("seed", ""),
        )
        groups.setdefault(key, []).append(row)

    out: list[dict[str, Any]] = []
    for key, group in groups.items():
        passing = [
            row
            for row in group
            if row.get("diagnostic_status") == "pass" and np.isfinite(_to_float(row.get("ope_value_estimate")))
        ]
        estimates = [_to_float(row.get("ope_value_estimate")) for row in passing]
        if len(estimates) >= 2:
            decision_status = "pass"
            recommended = float(np.median(estimates))
            reason = "at least two stabilized estimators passed diagnostics"
        elif len(estimates) == 1:
            decision_status = "single_estimator_warning"
            recommended = float(estimates[0])
            reason = "only one stabilized estimator passed diagnostics"
        else:
            decision_status = "needs_review"
            recommended = ""
            reason = "no stabilized estimator passed diagnostics"
        out.append(
            {
                "profile": key[0],
                "setting": key[1],
                "dataset_variant": key[2],
                "policy_shift": key[3],
                "gamma": key[4],
                "sample_size": key[5],
                "seed": key[6],
                "decision_status": decision_status,
                "recommended_ope_value": recommended,
                "passing_estimator_count": int(len(estimates)),
                "candidate_estimator_count": int(len(group)),
                "passing_estimators": "|".join(str(row.get("estimator")) for row in passing),
                "reason": reason,
            }
        )
    return out


def _high_stakes_diagnostic_status(row: dict[str, Any]) -> dict[str, Any]:
    if row.get("status") != "ok":
        return {"diagnostic_status": "fail", "diagnostic_reason": "estimator did not complete successfully"}
    if row.get("estimator") == "oracle":
        return {"diagnostic_status": "pass", "diagnostic_reason": ""}

    fail_reasons: list[str] = []
    warn_reasons: list[str] = []
    ope_value = _to_float(row.get("ope_value_estimate"))
    if not np.isfinite(ope_value):
        fail_reasons.append("non-finite or missing OPE estimate")
    ess = _to_float(row.get("effective_sample_size_fraction"))
    if np.isfinite(ess):
        true_ess = _to_float(row.get("true_effective_sample_size_fraction"))
        if np.isfinite(true_ess):
            if true_ess > 0.0 and ess < 0.25 * true_ess and abs(ess - true_ess) > 0.05:
                warn_reasons.append("ESS fraction far below oracle ESS")
        elif ess < 0.01:
            fail_reasons.append("ESS fraction below 0.01")
        elif ess < 0.03:
            warn_reasons.append("ESS fraction below 0.03")
    clipping = _high_stakes_final_clipping_fraction(row)
    if np.isfinite(clipping):
        if clipping > 0.10:
            fail_reasons.append("clipping fraction above 0.10")
        elif clipping > 0.02:
            warn_reasons.append("clipping fraction above 0.02")
    source_enabled = _to_float(row.get("source_state_ratio_enabled")) > 0.5
    if source_enabled:
        source_ess = _to_float(row.get("source_state_ratio_ess_fraction"))
        source_max = _to_float(row.get("source_state_ratio_max"))
        if np.isfinite(source_ess) and source_ess < 0.05:
            fail_reasons.append("source-state ESS fraction below 0.05")
        if np.isfinite(source_max) and source_max > 25.0:
            warn_reasons.append("source-state ratio max above 25")
    rel_change = _to_float(row.get("fixed_point_rel_change_final"))
    if np.isfinite(rel_change) and rel_change > 0.05:
        fail_reasons.append("fixed-point relative change above 0.05")
    weight_q99_to_median = _to_float(row.get("weight_q99_to_median"))
    weight_max = _to_float(row.get("weight_max"))
    if (np.isfinite(weight_q99_to_median) and weight_q99_to_median > 100.0) or (
        np.isfinite(weight_max) and weight_max >= 49.5
    ):
        warn_reasons.append("tail weights indicate cap pressure")

    if fail_reasons:
        return {"diagnostic_status": "fail", "diagnostic_reason": "; ".join(fail_reasons)}
    if warn_reasons:
        return {"diagnostic_status": "warn", "diagnostic_reason": "; ".join(warn_reasons)}
    return {"diagnostic_status": "pass", "diagnostic_reason": ""}


def _high_stakes_final_clipping_fraction(row: dict[str, Any]) -> float:
    values = []
    for key in ("clipping_fraction", "projection_clipped_fraction_final"):
        value = _to_float(row.get(key))
        if np.isfinite(value):
            values.append(value)
    return float(max(values)) if values else float("nan")


def _to_float(value: Any) -> float:
    try:
        if value in ("", None):
            return float("nan")
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def _primary_metric_name(row: dict[str, Any]) -> str | None:
    for name in ("ratio_normalized_l1", "ratio_l1", "ratio_rel_mse", "log_ratio_rmse", "ope_value_abs_error", "absolute_error"):
        if name in row and row[name] != "":
            return name
    return None


def _is_deployable_estimator(estimator: str) -> bool:
    if estimator in {"", "oracle", "behavior"}:
        return False
    if estimator == "dice_rl_best_regularized":
        return False
    return "oracle" not in estimator


def _needs_dice_rl(config: OccupancyRatioBenchmarkConfig) -> bool:
    return any(str(estimator).startswith("dice_rl_") for estimator in config.resolved_estimators())


def _run_estimator_maybe_timeout(
    estimator: str,
    dataset,
    config: OccupancyRatioBenchmarkConfig,
    google_preflight: GoogleDualDICEPreflight,
    dice_rl_preflight: GoogleDICERLPreflight | None = None,
) -> EstimatorResult:
    timeout = config.estimator_timeout_sec
    if timeout is None:
        return run_estimator(estimator, dataset, config, google_preflight, dice_rl_preflight)
    timeout = float(timeout)
    # Spawn is slower than fork, but avoids macOS ObjC/MPS crashes when a
    # timed estimator imports or trains with PyTorch/TensorFlow in the child.
    ctx_name = "spawn" if "spawn" in mp.get_all_start_methods() else mp.get_all_start_methods()[0]
    ctx = mp.get_context(ctx_name)
    result_queue = ctx.Queue(maxsize=1)
    process = ctx.Process(
        target=_estimator_worker,
        args=(result_queue, estimator, dataset, config, google_preflight, dice_rl_preflight),
    )
    process.daemon = True
    start = time.perf_counter()
    try:
        process.start()
        status_payload = None
        deadline = start + timeout
        while True:
            remaining = deadline - time.perf_counter()
            if remaining <= 0.0:
                break
            try:
                status_payload = result_queue.get(timeout=min(0.25, remaining))
                break
            except queue.Empty:
                if not process.is_alive():
                    break
        if status_payload is not None:
            process.join(5.0)
            if process.is_alive():
                process.terminate()
                process.join(5.0)
                if process.is_alive():
                    process.kill()
                    process.join(5.0)
            status, payload = status_payload
            if status == "ok":
                return payload
            raise RuntimeError(str(payload))
        process.join(0.0)
        runtime = float(time.perf_counter() - start)
        if process.is_alive():
            process.terminate()
            process.join(5.0)
            if process.is_alive():
                process.kill()
                process.join(5.0)
            return EstimatorResult(
                estimator=estimator,
                status="timeout",
                weights=None,
                raw_weights=None,
                runtime_sec=runtime,
                diagnostics={"timeout_sec": timeout},
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
    finally:
        if process.pid is not None and process.is_alive():
            process.kill()
            process.join(5.0)
        result_queue.close()
        result_queue.cancel_join_thread()
        try:
            process.close()
        except ValueError:
            pass


def _estimator_worker(
    result_queue,
    estimator: str,
    dataset,
    config: OccupancyRatioBenchmarkConfig,
    google_preflight: GoogleDualDICEPreflight,
    dice_rl_preflight: GoogleDICERLPreflight | None = None,
) -> None:
    try:
        result_queue.put(("ok", run_estimator(estimator, dataset, config, google_preflight, dice_rl_preflight)))
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
    row: dict[str, Any] = {
        "stage": config.stage,
        "profile": str(config.profile),
        "setting": setting,
        "estimator": estimator,
        "status": "error",
        "skip_reason": f"{type(exc).__name__}: {exc}",
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "timeout_sec": "",
        "gamma": float(gamma),
        "seed": int(seed),
        "sample_size": int(sample_size),
        "runtime_sec": 0.0,
    }
    if dataset_variant is not None:
        row["dataset_variant"] = str(dataset_variant)
    if policy_shift is not None:
        row["policy_shift"] = float(policy_shift)
    if config.profile == "high_stakes":
        row.update(_high_stakes_diagnostic_status(row))
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
        "error_type": "",
        "error_message": "",
        "timeout_sec": "",
        "gamma": float(gamma),
        "seed": int(seed),
        "sample_size": int(sample_size),
        "runtime_sec": 0.0,
    }
    if dataset_variant is not None:
        row["dataset_variant"] = str(dataset_variant)
    if policy_shift is not None:
        row["policy_shift"] = float(policy_shift)
    if config.profile == "high_stakes":
        row.update({"diagnostic_status": "warn", "diagnostic_reason": "dataset or estimator skipped"})
    return row


def _manifest(config: OccupancyRatioBenchmarkConfig, *, run_metadata: dict[str, Any] | None = None) -> dict[str, Any]:
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
        "scope-rl",
        "rtbgym",
        "recgym",
        "minari",
    ):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "config": asdict(config),
        "run_metadata": run_metadata or _run_metadata(config),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "torch_thread_guard": _apply_torch_thread_guard(),
        "randomness": {
            "seeds": list(map(int, config.seeds)),
            "note": "Each setting/estimator receives deterministic NumPy seeds derived from config seed.",
        },
    }


def _write_defaults_report(results_path: Path, output_dir: Path) -> dict[str, Path]:
    try:
        from occupancy_ratio_benchmark.defaults_report import generate_defaults_report

        return generate_defaults_report(results_path, output_dir=output_dir)
    except Exception as exc:  # pragma: no cover - report generation is best-effort
        failure_path = output_dir / "defaults_report_error.txt"
        failure_path.write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
        return {"report": failure_path, "neural_vs_dice": output_dir / "defaults_neural_vs_dice.csv"}


def _write_conservatism_audit(rows: list[dict[str, Any]], output_dir: Path) -> dict[str, Path]:
    try:
        from occupancy_ratio_benchmark.conservatism_audit import write_conservatism_audit

        return write_conservatism_audit(rows, output_dir)
    except Exception as exc:  # pragma: no cover - report generation is best-effort
        failure_path = output_dir / "conservatism_audit_error.txt"
        failure_path.write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
        return {"audit": output_dir / "conservatism_audit.csv", "report": failure_path}


def _run_metadata(config: OccupancyRatioBenchmarkConfig) -> dict[str, Any]:
    return {
        "git": _git_metadata(),
        "config_path": "" if config.config_path is None else str(config.config_path),
        "config_sha256": str(config.config_sha256),
        "output_root": str(config.output_root),
        "external_repo_path": str(config.external_repo_path),
    }


def _git_metadata() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[3]
    return {
        "branch": _git_output(root, "branch", "--show-current"),
        "sha": _git_output(root, "rev-parse", "HEAD"),
        "dirty": bool(_git_output(root, "status", "--porcelain")),
    }


def _git_output(cwd: Path, *args: str) -> str:
    try:
        completed = subprocess.run(
            ("git", *args),
            cwd=str(cwd),
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
    except Exception:
        return ""
    if completed.returncode != 0:
        return ""
    return completed.stdout.strip()


def _render_benchmark_readout(
    *,
    config: OccupancyRatioBenchmarkConfig,
    rows: list[dict[str, Any]],
    summary: list[dict[str, Any]],
    winners: list[dict[str, Any]],
    diagnostics: dict[str, Any],
    defaults_paths: dict[str, Path],
    conservatism_paths: dict[str, Path],
) -> str:
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    failed_rows = [row for row in rows if row.get("status") in {"error", "timeout"}]
    skipped_rows = [row for row in rows if row.get("status") == "skipped"]
    google = diagnostics.get("google_dualdice_preflight", {})
    lines = [
        "# DualDICE OPE Benchmark Readout",
        "",
        f"Profile: `{config.profile}`",
        f"Rows: {len(rows)} total, {len(ok_rows)} ok, {len(failed_rows)} failed/timeout, {len(skipped_rows)} skipped.",
        f"Google DualDICE: {'available' if google.get('available') else 'unavailable'}",
        "",
        "## Main Files",
        "",
        "- `results.csv`: row-level estimator metrics",
        "- `summary.csv`: grouped estimator summaries",
        "- `winner_table.csv`: best non-oracle estimator by benchmark cell",
        f"- `{Path(defaults_paths.get('report', Path('defaults_report.md'))).name}`: default-selection report",
        f"- `{Path(defaults_paths.get('neural_vs_dice', Path('defaults_neural_vs_dice.csv'))).name}`: neural FORI vs Google DualDICE table",
        f"- `{Path(conservatism_paths.get('report', Path('conservatism_audit.md'))).name}`: conservatism audit report",
        f"- `{Path(conservatism_paths.get('audit', Path('conservatism_audit.csv'))).name}`: conservatism audit table",
        "",
        "## Estimator Snapshot",
        "",
        "| estimator | status | n | mean OPE abs error | mean log-ratio RMSE | mean ESS |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in _compact_summary_rows(summary):
        lines.append(
            "| {estimator} | {status} | {n_runs} | {ope} | {log_rmse} | {ess} |".format(
                estimator=row.get("estimator", ""),
                status=row.get("status", ""),
                n_runs=row.get("n_runs", ""),
                ope=_fmt_md(row.get("ope_value_abs_error_mean")),
                log_rmse=_fmt_md(row.get("log_ratio_rmse_mean")),
                ess=_fmt_md(row.get("effective_sample_size_fraction_mean")),
            )
        )
    if winners:
        lines.extend(["", "## Winner Counts", "", "| estimator | wins |", "|---|---:|"])
        counts: dict[str, int] = {}
        for row in winners:
            estimator = str(row.get("winning_estimator", ""))
            counts[estimator] = counts.get(estimator, 0) + 1
        for estimator, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
            lines.append(f"| {estimator} | {count} |")
    if failed_rows:
        lines.extend(["", "## Failures", ""])
        for row in failed_rows[:20]:
            lines.append(
                "- {setting} / {estimator} / seed {seed}: {reason}".format(
                    setting=row.get("setting", ""),
                    estimator=row.get("estimator", ""),
                    seed=row.get("seed", ""),
                    reason=row.get("error_message") or row.get("skip_reason", ""),
                )
            )
    return "\n".join(lines) + "\n"


def _compact_summary_rows(summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ok_summary = [row for row in summary if row.get("status") == "ok"]
    return sorted(ok_summary, key=lambda row: (str(row.get("estimator", "")), str(row.get("setting", ""))))[:30]


def _fmt_md(value: Any) -> str:
    try:
        if value in ("", None):
            return ""
        out = float(value)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(out):
        return ""
    return f"{out:.4g}"


def _ensure_runtime_cache_dir(output_dir: Path) -> None:
    if "MPLCONFIGDIR" in os.environ:
        return
    cache_dir = output_dir / ".matplotlib-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache_dir)
