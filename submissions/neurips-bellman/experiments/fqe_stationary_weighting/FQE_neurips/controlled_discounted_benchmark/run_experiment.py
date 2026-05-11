from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .configs import (
    EvaluationConfig,
    FQESolverConfig,
    NeuralFQEConfig,
    NeuralRatioConfig,
    RatioFeatureConfig,
    WeightEstimatorConfig,
    design_search_grid,
    final_stage_defaults,
    gamma_design_stage_grid,
    gamma_final_stage_grid,
    gamma_paper_stage_grid,
    gamma_smoke_stage_grid,
    smoke_stage_grid,
    stationary_shift_paper_stage_grid,
)
from .data import sample_behavior_trajectory_batch, sample_discounted_behavior_batch, sample_reference_behavior_batch
from .envs import LinearGaussianEnv, LinearGaussianEnvConfig
from .features import RatioFeatureMap, StateActionFeatureMap
from .fqe import fit_linear_fqe
from .metrics import EstimatorMetrics, draw_evaluation_samples, evaluate_estimator
from .minimax_q import _moment_risk, fit_minimax_linear_q
from .neural_fqe import fit_neural_fqe
from .truth import build_discounted_occupancy_mixture, build_reference_occupancy_mixture, solve_policy_truth
from .weights import (
    estimate_discounted_occupancy_ratio,
    estimate_exponential_quadratic_moment_ratio,
    estimate_neural_discounted_occupancy_ratio,
    exponential_quadratic_moment_raw_weights,
    feature_calibration_l2,
    oracle_density_ratio,
    process_ess_adaptive_winsor_weights,
    process_raw_weights,
    ratio_quality,
    summarize_weights,
    weighted_design_condition_number,
)
from FQE_neurips.utils import TransitionBatch


RESULT_COLUMNS = [
    "study_stage",
    "gamma",
    "shift",
    "sample_size",
    "process_noise_sd",
    "behavior_action_sd",
    "feature_regime",
    "seed",
    "estimator",
    "target_q_mse",
    "target_q_mse_se",
    "behavior_q_mse",
    "behavior_q_mse_se",
    "behavior_target_action_q_mse",
    "behavior_target_action_q_mse_se",
    "policy_value_estimate",
    "policy_value_true",
    "policy_value_error",
    "policy_value_absolute_error",
    "policy_value_squared_error",
    "initial_v_mse",
    "initial_v_mse_se",
    "behavior_v_mse",
    "behavior_v_mse_se",
    "target_v_mse",
    "target_v_mse_se",
    "behavior_bellman_residual_mse",
    "target_bellman_residual_mse",
    "weight_mean",
    "weight_std",
    "weight_max",
    "weight_q90",
    "weight_q95",
    "weight_q99",
    "effective_sample_size",
    "effective_sample_size_fraction",
    "fraction_clipped",
    "chosen_uniform_mix",
    "clip_level",
    "weight_stabilization",
    "cap_selection_rule",
    "adaptive_ess_target",
    "fixed_cap",
    "cv_selected_cap",
    "cv_selected_ridge",
    "cv_selected_ridge_min",
    "cv_selected_ridge_one_se",
    "cv_validation_ratio_moment_risk_se",
    "cv_one_se_threshold",
    "cv_n_folds",
    "cv_validation_bellman_mse",
    "cv_validation_ratio_moment_risk",
    "tikhonov_ridge_primal",
    "tikhonov_ridge_dual",
    "normalized_tikhonov_eta",
    "normalized_tikhonov_primal_scale",
    "normalized_tikhonov_dual_scale",
    "ess_fraction_before_stabilization",
    "ess_fraction_after_stabilization",
    "oracle_log_ratio_rmse",
    "oracle_estimated_weight_corr",
    "oracle_estimated_weight_mae",
    "oracle_estimated_weight_rel_mse",
    "weighted_design_condition_number",
    "unstable_run_flag",
    "unstable_reason",
]


WEIGHT_META_COLUMNS = {
    "weight_stabilization",
    "cap_selection_rule",
    "adaptive_ess_target",
    "fixed_cap",
    "cv_selected_cap",
    "cv_selected_ridge",
    "cv_selected_ridge_min",
    "cv_selected_ridge_one_se",
    "cv_validation_ratio_moment_risk_se",
    "cv_one_se_threshold",
    "cv_n_folds",
    "cv_validation_bellman_mse",
    "cv_validation_ratio_moment_risk",
    "tikhonov_ridge_primal",
    "tikhonov_ridge_dual",
    "normalized_tikhonov_eta",
    "normalized_tikhonov_primal_scale",
    "normalized_tikhonov_dual_scale",
    "ess_fraction_before_stabilization",
    "ess_fraction_after_stabilization",
}


SUMMARY_EXTRA_COLUMNS = [
    "n_seeds",
    "policy_value_bias",
    "policy_value_variance",
    "policy_value_mse",
    "unstable_run_count",
    "unstable_run_fraction",
]


GAMMA_EXTRA_COLUMNS = [
    "value_gamma",
    "ratio_gamma",
    "reference_distribution",
    "data_mode",
    "fqe_family",
    "ratio_estimator",
    "ratio_moment_violation_l2",
    "ratio_normalization_error",
    "target_ratio_feature_calibration_l2",
    "target_fqe_feature_calibration_l2",
    "ratio_solver",
]


GAMMA_RESULT_COLUMNS = RESULT_COLUMNS + GAMMA_EXTRA_COLUMNS


@dataclass(frozen=True)
class SelectedFamily:
    gamma: float
    process_noise_sd: float
    behavior_action_sd: float
    low_shift: float
    moderate_shift: float
    severe_shift: float
    status: str
    note: str


@dataclass(frozen=True)
class CVTikhonovSelection:
    ratio: object
    config: WeightEstimatorConfig
    validation_loss: float
    validation_loss_se: float
    oof_raw_weights: np.ndarray
    diagnostics: dict[str, object]


@dataclass(frozen=True)
class CVQuadraticMomentSelection:
    ratio: object
    ridge: float
    validation_loss: float
    validation_loss_se: float
    diagnostics: dict[str, object]


RICH_TIKHONOV_RIDGE_GRID = (
    1e-5,
    3e-5,
    1e-4,
    3e-4,
    1e-3,
    3e-3,
    1e-2,
    3e-2,
    1e-1,
    3e-1,
    1.0,
    3.0,
    10.0,
)

QUADRATIC_MOMENT_RIDGE_GRID = (
    1e-4,
    3e-4,
    1e-3,
    3e-3,
    1e-2,
    3e-2,
    1e-1,
    3e-1,
    1.0,
)

FIXED_TIKHONOV_ETA_GRID = (1e-3, 1e-2, 1e-1, 1.0, 10.0)
FIXED_TIKHONOV_MAIN_ETA = 1e-1


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as handle:
        return list(csv.DictReader(handle))


def _metric_row(
    *,
    stage: str,
    gamma: float,
    shift: float,
    sample_size: int,
    process_noise_sd: float,
    behavior_action_sd: float,
    feature_regime: str,
    seed: int,
    metrics: EstimatorMetrics,
) -> dict[str, object]:
    row = {
        "study_stage": stage,
        "gamma": gamma,
        "shift": shift,
        "sample_size": sample_size,
        "process_noise_sd": process_noise_sd,
        "behavior_action_sd": behavior_action_sd,
        "feature_regime": feature_regime,
        "seed": seed,
        "estimator": metrics.estimator,
        "target_q_mse": metrics.target_q_mse,
        "target_q_mse_se": metrics.target_q_mse_se,
        "behavior_q_mse": metrics.behavior_q_mse,
        "behavior_q_mse_se": metrics.behavior_q_mse_se,
        "behavior_target_action_q_mse": metrics.behavior_target_action_q_mse,
        "behavior_target_action_q_mse_se": metrics.behavior_target_action_q_mse_se,
        "policy_value_estimate": metrics.policy_value_estimate,
        "policy_value_true": metrics.policy_value_true,
        "policy_value_error": metrics.policy_value_error,
        "policy_value_absolute_error": metrics.policy_value_absolute_error,
        "policy_value_squared_error": metrics.policy_value_squared_error,
        "initial_v_mse": metrics.initial_v_mse,
        "initial_v_mse_se": metrics.initial_v_mse_se,
        "behavior_v_mse": metrics.behavior_v_mse,
        "behavior_v_mse_se": metrics.behavior_v_mse_se,
        "target_v_mse": metrics.target_v_mse,
        "target_v_mse_se": metrics.target_v_mse_se,
        "behavior_bellman_residual_mse": metrics.behavior_bellman_residual_mse,
        "target_bellman_residual_mse": metrics.target_bellman_residual_mse,
    }
    row.update(metrics.weight_stats)
    return row


def _gamma_metric_row(
    *,
    stage: str,
    value_gamma: float,
    ratio_gamma: float,
    reference_distribution: str,
    data_mode: str,
    shift: float,
    sample_size: int,
    process_noise_sd: float,
    behavior_action_sd: float,
    feature_regime: str,
    seed: int,
    fqe_family: str,
    ratio_estimator: str,
    metrics: EstimatorMetrics,
) -> dict[str, object]:
    row = _metric_row(
        stage=stage,
        gamma=value_gamma,
        shift=shift,
        sample_size=sample_size,
        process_noise_sd=process_noise_sd,
        behavior_action_sd=behavior_action_sd,
        feature_regime=feature_regime,
        seed=seed,
        metrics=metrics,
    )
    row.update(
        {
            "value_gamma": value_gamma,
            "ratio_gamma": ratio_gamma,
            "reference_distribution": reference_distribution,
            "data_mode": data_mode,
            "fqe_family": fqe_family,
            "ratio_estimator": ratio_estimator,
        }
    )
    for column in GAMMA_EXTRA_COLUMNS:
        row.setdefault(column, metrics.weight_stats.get(column, float("nan")))
    return row


def _gamma_group_key(row: dict[str, str | float | int]) -> tuple[object, ...]:
    return (
        row["study_stage"],
        float(row["value_gamma"]),
        float(row["ratio_gamma"]),
        str(row["reference_distribution"]),
        str(row["data_mode"]),
        float(row["shift"]),
        int(row["sample_size"]),
        float(row["process_noise_sd"]),
        float(row["behavior_action_sd"]),
        row["feature_regime"],
        row["fqe_family"],
        row["ratio_estimator"],
        row["estimator"],
    )


def _aggregate_rows(rows: list[dict[str, object]], fieldnames: list[str], gamma_mode: bool = False) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = {}
    key_fn = _gamma_group_key if gamma_mode else _group_key
    for row in rows:
        grouped.setdefault(key_fn(row), []).append(row)
    summary_rows = []
    id_columns = {
        "study_stage",
        "gamma",
        "value_gamma",
        "ratio_gamma",
        "reference_distribution",
        "data_mode",
        "shift",
        "sample_size",
        "process_noise_sd",
        "behavior_action_sd",
        "feature_regime",
        "seed",
        "fqe_family",
        "ratio_estimator",
        "estimator",
    }
    for _key, group in grouped.items():
        base = dict(group[0])
        for column in fieldnames:
            if column in id_columns:
                continue
            if column in {"unstable_reason", "ratio_solver", "weight_stabilization", "cap_selection_rule"}:
                reasons = sorted(
                    {
                        str(item.get(column, "stable"))
                        for item in group
                        if str(item.get(column, "stable")) not in {"", "stable", "nan"}
                    }
                )
                base[column] = "stable" if not reasons else ";".join(reasons)
                continue
            values = []
            for item in group:
                try:
                    values.append(float(item[column]))
                except (KeyError, TypeError, ValueError):
                    pass
            base[column] = float(np.mean(values)) if values else float("nan")
        estimates = np.asarray([float(item["policy_value_estimate"]) for item in group], dtype=np.float64)
        truth = float(group[0]["policy_value_true"])
        errors = estimates - truth
        unstable_flags = np.asarray([float(item.get("unstable_run_flag", 0.0)) for item in group], dtype=np.float64)
        base["n_seeds"] = len(group)
        base["policy_value_bias"] = float(np.mean(errors))
        base["policy_value_variance"] = float(np.var(estimates, ddof=0))
        base["policy_value_mse"] = float(np.mean(errors**2))
        base["unstable_run_count"] = int(np.sum(unstable_flags > 0.5))
        base["unstable_run_fraction"] = float(np.mean(unstable_flags > 0.5))
        summary_rows.append(base)
    summary_rows.sort(
        key=lambda row: (
            row["study_stage"],
            float(row.get("value_gamma", row.get("gamma", 0.0))),
            float(row.get("ratio_gamma", row.get("gamma", 0.0))),
            str(row.get("reference_distribution", "")),
            str(row.get("data_mode", "")),
            float(row["process_noise_sd"]),
            float(row["behavior_action_sd"]),
            float(row["shift"]),
            int(row["sample_size"]),
            row["feature_regime"],
            str(row.get("fqe_family", "")),
            str(row.get("ratio_estimator", "")),
            row["estimator"],
        )
    )
    return summary_rows


def _group_key(row: dict[str, str | float | int]) -> tuple[object, ...]:
    return (
        row["study_stage"],
        float(row["gamma"]),
        float(row["shift"]),
        int(row["sample_size"]),
        float(row["process_noise_sd"]),
        float(row["behavior_action_sd"]),
        row["feature_regime"],
        row["estimator"],
    )


def _aggregate_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(_group_key(row), []).append(row)
    summary_rows = []
    for key, group in grouped.items():
        base = dict(group[0])
        for column in RESULT_COLUMNS:
            if column in {
                "study_stage",
                "gamma",
                "shift",
                "sample_size",
                "process_noise_sd",
                "behavior_action_sd",
                "feature_regime",
                "estimator",
            }:
                continue
            if column in {"unstable_reason", "weight_stabilization", "cap_selection_rule"}:
                reasons = sorted(
                    {
                        str(item.get(column, "stable"))
                        for item in group
                        if str(item.get(column, "stable")) not in {"", "stable", "nan"}
                    }
                )
                base[column] = "stable" if not reasons else ";".join(reasons)
                continue
            values = []
            for item in group:
                try:
                    values.append(float(item[column]))
                except (KeyError, TypeError, ValueError):
                    pass
            base[column] = float(np.mean(values)) if values else float("nan")
        estimates = np.asarray([float(item["policy_value_estimate"]) for item in group], dtype=np.float64)
        truth = float(group[0]["policy_value_true"])
        errors = estimates - truth
        unstable_flags = np.asarray([float(item.get("unstable_run_flag", 0.0)) for item in group], dtype=np.float64)
        base["n_seeds"] = len(group)
        base["policy_value_bias"] = float(np.mean(errors))
        base["policy_value_variance"] = float(np.var(estimates, ddof=0))
        base["policy_value_mse"] = float(np.mean(errors**2))
        base["unstable_run_count"] = int(np.sum(unstable_flags > 0.5))
        base["unstable_run_fraction"] = float(np.mean(unstable_flags > 0.5))
        summary_rows.append(base)
    summary_rows.sort(
        key=lambda row: (
            row["study_stage"],
            float(row["gamma"]),
            float(row["process_noise_sd"]),
            float(row["behavior_action_sd"]),
            float(row["shift"]),
            int(row["sample_size"]),
            row["feature_regime"],
            row["estimator"],
        )
    )
    return summary_rows


def _mean_metric(
    summary_rows: list[dict[str, object]],
    *,
    gamma: float,
    process_noise_sd: float,
    behavior_action_sd: float,
    shift: float,
    sample_size: int,
    feature_regime: str,
    estimator: str,
    metric: str,
) -> float:
    for row in summary_rows:
        if (
            float(row["gamma"]) == gamma
            and float(row["process_noise_sd"]) == process_noise_sd
            and float(row["behavior_action_sd"]) == behavior_action_sd
            and float(row["shift"]) == shift
            and int(row["sample_size"]) == sample_size
            and row["feature_regime"] == feature_regime
            and row["estimator"] == estimator
        ):
            return float(row[metric])
    return float("nan")


def _stabilization_meta(
    meta: dict[str, float],
    *,
    weight_stabilization: str,
    cap_selection_rule: str,
    adaptive_ess_target: float | None = None,
    fixed_cap: float | None = None,
    cv_selected_cap: float | None = None,
    cv_selected_ridge: float | None = None,
    cv_selected_ridge_min: float | None = None,
    cv_selected_ridge_one_se: float | None = None,
    cv_validation_ratio_moment_risk_se: float | None = None,
    cv_one_se_threshold: float | None = None,
    cv_n_folds: int | None = None,
    cv_validation_bellman_mse: float | None = None,
    cv_validation_ratio_moment_risk: float | None = None,
    tikhonov_ridge_primal: float | None = None,
    tikhonov_ridge_dual: float | None = None,
    normalized_tikhonov_eta: float | None = None,
    normalized_tikhonov_primal_scale: float | None = None,
    normalized_tikhonov_dual_scale: float | None = None,
) -> dict[str, object]:
    out: dict[str, object] = dict(meta)
    before = out.get("ess_fraction_before_winsor", out.get("ess_fraction_before_mix", np.nan))
    after = out.get("ess_fraction_after_winsor", out.get("ess_fraction_after_mix", np.nan))
    out.update(
        {
            "weight_stabilization": weight_stabilization,
            "cap_selection_rule": cap_selection_rule,
            "adaptive_ess_target": np.nan if adaptive_ess_target is None else float(adaptive_ess_target),
            "fixed_cap": np.nan if fixed_cap is None else float(fixed_cap),
            "cv_selected_cap": np.nan if cv_selected_cap is None else float(cv_selected_cap),
            "cv_selected_ridge": np.nan if cv_selected_ridge is None else float(cv_selected_ridge),
            "cv_selected_ridge_min": (
                np.nan if cv_selected_ridge_min is None else float(cv_selected_ridge_min)
            ),
            "cv_selected_ridge_one_se": (
                np.nan if cv_selected_ridge_one_se is None else float(cv_selected_ridge_one_se)
            ),
            "cv_validation_ratio_moment_risk_se": (
                np.nan
                if cv_validation_ratio_moment_risk_se is None
                else float(cv_validation_ratio_moment_risk_se)
            ),
            "cv_one_se_threshold": np.nan if cv_one_se_threshold is None else float(cv_one_se_threshold),
            "cv_n_folds": np.nan if cv_n_folds is None else int(cv_n_folds),
            "cv_validation_bellman_mse": (
                np.nan if cv_validation_bellman_mse is None else float(cv_validation_bellman_mse)
            ),
            "cv_validation_ratio_moment_risk": (
                np.nan if cv_validation_ratio_moment_risk is None else float(cv_validation_ratio_moment_risk)
            ),
            "tikhonov_ridge_primal": (
                np.nan if tikhonov_ridge_primal is None else float(tikhonov_ridge_primal)
            ),
            "tikhonov_ridge_dual": (
                np.nan if tikhonov_ridge_dual is None else float(tikhonov_ridge_dual)
            ),
            "normalized_tikhonov_eta": (
                np.nan if normalized_tikhonov_eta is None else float(normalized_tikhonov_eta)
            ),
            "normalized_tikhonov_primal_scale": (
                np.nan
                if normalized_tikhonov_primal_scale is None
                else float(normalized_tikhonov_primal_scale)
            ),
            "normalized_tikhonov_dual_scale": (
                np.nan
                if normalized_tikhonov_dual_scale is None
                else float(normalized_tikhonov_dual_scale)
            ),
            "ess_fraction_before_stabilization": float(before),
            "ess_fraction_after_stabilization": float(after),
        }
    )
    return out


def _subset_batch(batch: TransitionBatch, indices: np.ndarray) -> TransitionBatch:
    idx = np.asarray(indices, dtype=np.int64)
    return TransitionBatch(
        states=batch.states[idx],
        actions=batch.actions[idx],
        rewards=batch.rewards[idx],
        next_states=batch.next_states[idx],
        next_actions=batch.next_actions[idx],
    )


def _solve_stable(matrix: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(matrix, rhs, rcond=None)[0]


def _trace_scale(matrix: np.ndarray) -> float:
    arr = np.asarray(matrix, dtype=np.float64)
    if arr.size == 0:
        return 1.0
    scale = float(np.trace(arr) / max(arr.shape[0], 1))
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = float(np.linalg.norm(arr, ord="fro") / max(arr.shape[0], 1))
    return float(max(scale, 1e-12))


def _normalized_ratio_tikhonov_config(
    *,
    batch: TransitionBatch,
    ratio_feature_map: RatioFeatureMap,
    target_policy,
    gamma: float,
    base_config: WeightEstimatorConfig,
    eta: float,
) -> tuple[WeightEstimatorConfig, dict[str, float]]:
    current_features = ratio_feature_map.transform(batch.states, batch.actions)
    next_expected_features = ratio_feature_map.expected_given_state(batch.next_states, target_policy)
    delta = current_features - float(gamma) * next_expected_features
    n = max(current_features.shape[0], 1)
    a_mat = (current_features.T @ delta) / n
    b_mat = (current_features.T @ current_features) / n
    m_vec = current_features.mean(axis=0)
    dual_scale = _trace_scale(b_mat)
    h0 = b_mat + 1e-10 * dual_scale * np.eye(b_mat.shape[0], dtype=np.float64)
    h_inv_a = _solve_stable(h0, a_mat.T)
    reduced = (
        a_mat @ h_inv_a
        + 2.0 * float(base_config.normalization_penalty) * np.outer(m_vec, m_vec)
    )
    primal_scale = _trace_scale(reduced)
    ridge_primal = float(eta) * primal_scale
    ridge_dual = float(eta) * dual_scale
    return (
        replace(base_config, ridge_primal=ridge_primal, ridge_dual=ridge_dual),
        {
            "normalized_tikhonov_eta": float(eta),
            "normalized_tikhonov_primal_scale": float(primal_scale),
            "normalized_tikhonov_dual_scale": float(dual_scale),
            "tikhonov_ridge_primal": ridge_primal,
            "tikhonov_ridge_dual": ridge_dual,
        },
    )


def _normalized_minimax_ridges(
    *,
    batch: TransitionBatch,
    feature_map: StateActionFeatureMap,
    critic_feature_map: RatioFeatureMap,
    target_policy,
    gamma: float,
    eta: float,
    sample_weights: np.ndarray | None = None,
) -> tuple[float, float, dict[str, float]]:
    phi = feature_map.transform(batch.states, batch.actions)
    phi_next = feature_map.expected_features_given_state(batch.next_states, target_policy)
    critic = critic_feature_map.transform(batch.states, batch.actions)
    delta_phi = phi - float(gamma) * phi_next
    n = max(phi.shape[0], 1)
    if sample_weights is None:
        weights = np.ones(n, dtype=np.float64)
    else:
        weights = np.maximum(np.asarray(sample_weights, dtype=np.float64).reshape(-1), 1e-12)
        weights = weights / np.maximum(float(np.mean(weights)), 1e-12)
    moment_matrix = (critic.T @ (weights[:, None] * delta_phi)) / n
    critic_gram = (critic.T @ (weights[:, None] * critic)) / n
    dual_scale = _trace_scale(critic_gram)
    h0 = critic_gram + 1e-10 * dual_scale * np.eye(critic_gram.shape[0], dtype=np.float64)
    h_inv_g = _solve_stable(h0, moment_matrix)
    reduced = moment_matrix.T @ h_inv_g
    primal_scale = _trace_scale(reduced)
    q_ridge = float(eta) * primal_scale
    critic_ridge = float(eta) * dual_scale
    return (
        q_ridge,
        critic_ridge,
        {
            "normalized_tikhonov_eta": float(eta),
            "normalized_tikhonov_primal_scale": float(primal_scale),
            "normalized_tikhonov_dual_scale": float(dual_scale),
            "tikhonov_ridge_primal": float(q_ridge),
            "tikhonov_ridge_dual": float(critic_ridge),
        },
    )


def _make_q_feature_map(
    feature_regime: str,
    *,
    states: np.ndarray,
    actions: np.ndarray,
    ratio_feature_config: RatioFeatureConfig,
):
    if feature_regime == "flexible_rbf":
        return RatioFeatureMap.from_behavior_samples(
            states,
            actions,
            n_centers=ratio_feature_config.n_rbf_centers,
            bandwidth=ratio_feature_config.bandwidth,
            bandwidth_scale=ratio_feature_config.bandwidth_scale,
            standardize_features=ratio_feature_config.standardize_features,
        )
    return StateActionFeatureMap(feature_regime)


def _behavior_bellman_validation_mse(
    *,
    fit,
    batch: TransitionBatch,
    feature_map: StateActionFeatureMap,
    target_policy,
    gamma: float,
) -> float:
    phi = feature_map.transform(batch.states, batch.actions)
    phi_next = feature_map.expected_features_given_state(batch.next_states, target_policy)
    residual = phi @ fit.theta - (np.asarray(batch.rewards, dtype=np.float64).reshape(-1) + gamma * (phi_next @ fit.theta))
    return float(np.mean(residual**2))


def _target_q_mse_on_samples(
    *,
    q_function,
    truth,
    evaluation_samples,
) -> float:
    target_sa = evaluation_samples.target_sa
    q_true = truth.q_function.evaluate(target_sa[:, :2], target_sa[:, 2:])
    q_hat = q_function.evaluate(target_sa[:, :2], target_sa[:, 2:])
    return float(np.mean((q_hat - q_true) ** 2))


def _ratio_moment_validation_risk(
    weights: np.ndarray,
    *,
    batch: TransitionBatch,
    env: LinearGaussianEnv,
    target_policy,
    gamma: float,
    ratio_feature_map: RatioFeatureMap,
    weight_config: WeightEstimatorConfig,
) -> float:
    current_features = ratio_feature_map.transform(batch.states, batch.actions)
    next_expected_features = ratio_feature_map.expected_given_state(batch.next_states, target_policy)
    delta = current_features - gamma * next_expected_features
    normalized_weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    weighted_moment = np.mean(normalized_weights[:, None] * delta, axis=0)
    initial_rhs = ratio_feature_map.expectation_under_initial_distribution(
        env.config.initial_mean,
        env.config.initial_cov,
        target_policy,
    )
    rhs = (1.0 - gamma) * initial_rhs
    B = (current_features.T @ current_features) / max(current_features.shape[0], 1)
    H = B + weight_config.ridge_dual * np.eye(B.shape[0], dtype=np.float64)
    moment = weighted_moment - rhs
    try:
        dual_norm = float(moment @ np.linalg.solve(H, moment))
    except np.linalg.LinAlgError:
        dual_norm = float(moment @ np.linalg.lstsq(H, moment, rcond=None)[0])
    normalization = float(np.mean(normalized_weights) - 1.0)
    return dual_norm + float(weight_config.normalization_penalty) * normalization**2


def _select_cv_cap_weights(
    raw_weights: np.ndarray,
    *,
    batch: TransitionBatch,
    env: LinearGaussianEnv,
    ratio_feature_map: RatioFeatureMap,
    target_policy,
    gamma: float,
    weight_config: WeightEstimatorConfig,
    min_weight: float,
    seed: int,
    candidate_caps: Sequence[float | None] = (1.5, 2.0, 3.0, 5.0, 10.0, 25.0, None),
    n_folds: int = 3,
) -> tuple[np.ndarray, dict[str, object]]:
    """Choose a truncation cap by held-out stationary-flow moment risk."""

    raw = np.asarray(raw_weights, dtype=np.float64).reshape(-1)
    n = raw.shape[0]
    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(n), max(2, min(n_folds, n)))
    cap_losses: list[tuple[float, float | None]] = []
    all_idx = np.arange(n)
    for cap in candidate_caps:
        fold_losses = []
        for val_idx in folds:
            train_mask = np.ones(n, dtype=bool)
            train_mask[val_idx] = False
            train_idx = all_idx[train_mask]
            train_ratio = estimate_discounted_occupancy_ratio(
                batch.states[train_idx],
                batch.actions[train_idx],
                batch.next_states[train_idx],
                env=env,
                target_policy=target_policy,
                gamma=gamma,
                ratio_feature_map=ratio_feature_map,
                config=weight_config,
            )
            val_raw = ratio_feature_map.transform(batch.states[val_idx], batch.actions[val_idx]) @ train_ratio.alpha
            val_weights, _ = process_raw_weights(
                val_raw,
                min_weight=min_weight,
                max_weight=cap,
            )
            fold_losses.append(
                _ratio_moment_validation_risk(
                    val_weights,
                    batch=_subset_batch(batch, val_idx),
                    env=env,
                    target_policy=target_policy,
                    gamma=gamma,
                    ratio_feature_map=ratio_feature_map,
                    weight_config=weight_config,
                )
            )
        cap_losses.append((float(np.mean(fold_losses)), cap))
    cap_losses.sort(key=lambda item: (item[0], float("inf") if item[1] is None else item[1]))
    best_loss, best_cap = cap_losses[0]
    weights, meta = process_raw_weights(
        raw,
        min_weight=min_weight,
        max_weight=best_cap,
    )
    return weights, _stabilization_meta(
        meta,
        weight_stabilization="cv_cap",
        cap_selection_rule="ratio_moment_cv",
        cv_selected_cap=best_cap,
        cv_validation_ratio_moment_risk=best_loss,
    )


def _make_cv_folds(n: int, *, seed: int, n_folds: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [
        np.asarray(fold, dtype=np.int64)
        for fold in np.array_split(rng.permutation(n), max(2, min(n_folds, n)))
        if len(fold) > 0
    ]


def _crossfit_ratio_raw_weights(
    *,
    batch: TransitionBatch,
    env: LinearGaussianEnv,
    ratio_feature_map: RatioFeatureMap,
    target_policy,
    gamma: float,
    config: WeightEstimatorConfig,
    folds: Sequence[np.ndarray],
) -> np.ndarray:
    n = batch.states.shape[0]
    all_idx = np.arange(n)
    oof_raw = np.empty(n, dtype=np.float64)
    for val_idx in folds:
        train_mask = np.ones(n, dtype=bool)
        train_mask[val_idx] = False
        train_idx = all_idx[train_mask]
        ratio = estimate_discounted_occupancy_ratio(
            batch.states[train_idx],
            batch.actions[train_idx],
            batch.next_states[train_idx],
            env=env,
            target_policy=target_policy,
            gamma=gamma,
            ratio_feature_map=ratio_feature_map,
            config=config,
        )
        oof_raw[val_idx] = ratio_feature_map.transform(batch.states[val_idx], batch.actions[val_idx]) @ ratio.alpha
    return oof_raw


def _select_cv_tikhonov_ratios(
    *,
    batch: TransitionBatch,
    env: LinearGaussianEnv,
    ratio_feature_map: RatioFeatureMap,
    target_policy,
    gamma: float,
    base_config: WeightEstimatorConfig,
    seed: int,
    candidate_ridges: Sequence[float] = RICH_TIKHONOV_RIDGE_GRID,
    n_folds: int = 5,
) -> dict[str, CVTikhonovSelection]:
    """Select Tikhonov ridges by 5-fold held-out stationary-flow moment risk.

    Returns both the minimum-risk ridge and a conservative one-standard-error
    ridge. Cross-fitted raw weights are computed for both selections.
    """

    n = batch.states.shape[0]
    folds = _make_cv_folds(n, seed=seed, n_folds=n_folds)
    all_idx = np.arange(n)
    ridge_stats: list[dict[str, float]] = []
    for ridge in candidate_ridges:
        config = replace(
            base_config,
            ridge_primal=float(ridge),
            ridge_dual=float(ridge),
        )
        fold_losses = []
        for val_idx in folds:
            train_mask = np.ones(n, dtype=bool)
            train_mask[val_idx] = False
            train_idx = all_idx[train_mask]
            ratio = estimate_discounted_occupancy_ratio(
                batch.states[train_idx],
                batch.actions[train_idx],
                batch.next_states[train_idx],
                env=env,
                target_policy=target_policy,
                gamma=gamma,
                ratio_feature_map=ratio_feature_map,
                config=config,
            )
            val_raw = ratio_feature_map.transform(batch.states[val_idx], batch.actions[val_idx]) @ ratio.alpha
            val_weights, _ = process_raw_weights(
                val_raw,
                min_weight=config.min_weight,
            )
            fold_losses.append(
                _ratio_moment_validation_risk(
                    val_weights,
                    batch=_subset_batch(batch, val_idx),
                    env=env,
                    target_policy=target_policy,
                    gamma=gamma,
                    ratio_feature_map=ratio_feature_map,
                    weight_config=config,
                )
            )
        losses = np.asarray(fold_losses, dtype=np.float64)
        ridge_stats.append(
            {
                "mean": float(np.mean(losses)),
                "se": float(np.std(losses, ddof=1) / np.sqrt(max(len(losses), 1)))
                if len(losses) > 1
                else 0.0,
                "ridge": float(ridge),
            }
        )
    ridge_stats.sort(key=lambda item: (item["mean"], item["ridge"]))
    best = ridge_stats[0]
    one_se_threshold = best["mean"] + best["se"]
    eligible = [item for item in ridge_stats if item["mean"] <= one_se_threshold]
    one_se = max(eligible, key=lambda item: item["ridge"]) if eligible else best

    def make_selection(stat: dict[str, float], rule: str) -> CVTikhonovSelection:
        selected_config = replace(
            base_config,
            ridge_primal=float(stat["ridge"]),
            ridge_dual=float(stat["ridge"]),
        )
        final_ratio = estimate_discounted_occupancy_ratio(
            batch.states,
            batch.actions,
            batch.next_states,
            env=env,
            target_policy=target_policy,
            gamma=gamma,
            ratio_feature_map=ratio_feature_map,
            config=selected_config,
        )
        oof_raw = _crossfit_ratio_raw_weights(
            batch=batch,
            env=env,
            ratio_feature_map=ratio_feature_map,
            target_policy=target_policy,
            gamma=gamma,
            config=selected_config,
            folds=folds,
        )
        diagnostics = {
            "selection_rule": rule,
            "cv_selected_ridge": float(stat["ridge"]),
            "cv_selected_ridge_min": float(best["ridge"]),
            "cv_selected_ridge_one_se": float(one_se["ridge"]),
            "cv_validation_ratio_moment_risk": float(stat["mean"]),
            "cv_validation_ratio_moment_risk_se": float(stat["se"]),
            "cv_min_validation_ratio_moment_risk": float(best["mean"]),
            "cv_min_validation_ratio_moment_risk_se": float(best["se"]),
            "cv_one_se_threshold": float(one_se_threshold),
            "cv_n_folds": len(folds),
        }
        return CVTikhonovSelection(
            ratio=final_ratio,
            config=selected_config,
            validation_loss=float(stat["mean"]),
            validation_loss_se=float(stat["se"]),
            oof_raw_weights=oof_raw,
            diagnostics=diagnostics,
        )

    return {
        "min": make_selection(best, "min"),
        "one_se": make_selection(one_se, "one_standard_error"),
    }


def _select_cv_quadratic_moment_ratio(
    *,
    batch: TransitionBatch,
    env: LinearGaussianEnv,
    ratio_feature_map: RatioFeatureMap,
    target_policy,
    gamma: float,
    seed: int,
    candidate_ridges: Sequence[float] = QUADRATIC_MOMENT_RIDGE_GRID,
) -> CVQuadraticMomentSelection:
    """Select the exponential-quadratic log-ratio ridge by held-out moment risk."""

    ridge_stats: list[dict[str, float]] = []
    for idx, ridge in enumerate(candidate_ridges):
        ratio = estimate_exponential_quadratic_moment_ratio(
            batch.states,
            batch.actions,
            batch.next_states,
            env=env,
            target_policy=target_policy,
            gamma=gamma,
            ratio_feature_map=ratio_feature_map,
            seed=seed + 997 * (idx + 1),
            ridge=float(ridge),
            max_steps=350,
            batch_size=2048,
            valid_fraction=0.25,
            patience=45,
        )
        ridge_stats.append(
            {
                "ridge": float(ridge),
                "mean": float(ratio.diagnostics.get("best_valid_score", np.inf)),
                "train_last": float(ratio.diagnostics.get("train_objective_last", np.nan)),
                "valid_last": float(ratio.diagnostics.get("valid_objective_last", np.nan)),
            }
        )
    ridge_stats.sort(key=lambda item: (item["mean"], item["ridge"]))
    best = ridge_stats[0]
    final_ratio = estimate_exponential_quadratic_moment_ratio(
        batch.states,
        batch.actions,
        batch.next_states,
        env=env,
        target_policy=target_policy,
        gamma=gamma,
        ratio_feature_map=ratio_feature_map,
        seed=seed + 99_991,
        ridge=float(best["ridge"]),
        max_steps=900,
        batch_size=2048,
        valid_fraction=0.0,
        patience=90,
    )
    diagnostics = {
        "selection_rule": "heldout_min",
        "cv_selected_ridge": float(best["ridge"]),
        "cv_selected_ridge_min": float(best["ridge"]),
        "cv_selected_ridge_one_se": float("nan"),
        "cv_validation_ratio_moment_risk": float(best["mean"]),
        "cv_validation_ratio_moment_risk_se": 0.0,
        "cv_min_validation_ratio_moment_risk": float(best["mean"]),
        "cv_min_validation_ratio_moment_risk_se": 0.0,
        "cv_one_se_threshold": float("nan"),
        "cv_n_folds": 1,
        "cv_candidate_count": len(ridge_stats),
        "cv_train_objective_last": float(best["train_last"]),
        "cv_valid_objective_last": float(best["valid_last"]),
    }
    return CVQuadraticMomentSelection(
        ratio=final_ratio,
        ridge=float(best["ridge"]),
        validation_loss=float(best["mean"]),
        validation_loss_se=0.0,
        diagnostics=diagnostics,
    )


def _select_oracle_tikhonov_ratio_weights(
    *,
    batch: TransitionBatch,
    env: LinearGaussianEnv,
    ratio_feature_map: RatioFeatureMap,
    feature_map: StateActionFeatureMap,
    target_policy,
    ratio_gamma: float,
    value_gamma: float,
    base_weight_config: WeightEstimatorConfig,
    fqe_solver_config: FQESolverConfig,
    truth,
    evaluation_samples,
    candidate_ridges: Sequence[float] = RICH_TIKHONOV_RIDGE_GRID,
) -> tuple[np.ndarray, dict[str, object], dict[str, object]]:
    """Oracle-select the local RBF ratio ridge by target-stationary Q MSE.

    This is diagnostic only: it uses analytic truth on the independent
    evaluation draw to ask whether the local RBF moment class could work with
    ideal stabilization.
    """

    candidates: list[dict[str, object]] = []
    for ridge in candidate_ridges:
        config = replace(
            base_weight_config,
            ridge_primal=float(ridge),
            ridge_dual=float(ridge),
        )
        ratio = estimate_discounted_occupancy_ratio(
            batch.states,
            batch.actions,
            batch.next_states,
            env=env,
            target_policy=target_policy,
            gamma=ratio_gamma,
            ratio_feature_map=ratio_feature_map,
            config=config,
        )
        weights, meta = process_raw_weights(
            ratio.raw_weights,
            min_weight=config.min_weight,
        )
        fit = fit_linear_fqe(
            batch=batch,
            feature_map=feature_map,
            target_policy=target_policy,
            gamma=value_gamma,
            solver_config=fqe_solver_config,
            sample_weights=weights,
            seed=0,
        )
        target_q_mse = _target_q_mse_on_samples(
            q_function=fit.q_function,
            truth=truth,
            evaluation_samples=evaluation_samples,
        )
        candidates.append(
            {
                "ridge": float(ridge),
                "target_q_mse": float(target_q_mse),
                "ratio": ratio,
                "weights": weights,
                "meta": meta,
            }
        )
    candidates.sort(key=lambda item: (float(item["target_q_mse"]), float(item["ridge"])))
    best = candidates[0]
    ratio = best["ratio"]
    meta = _stabilization_meta(
        dict(best["meta"]),
        weight_stabilization="tikhonov_ratio",
        cap_selection_rule="oracle_target_q_mse",
        cv_selected_ridge=float(best["ridge"]),
        cv_selected_ridge_min=float(best["ridge"]),
        cv_validation_bellman_mse=float(best["target_q_mse"]),
        tikhonov_ridge_primal=float(best["ridge"]),
        tikhonov_ridge_dual=float(best["ridge"]),
    )
    diagnostics = {
        **ratio.diagnostics,
        "solver": "linear_oracle_tuned_tikhonov_reduced_moment",
        "oracle_selected_target_q_mse": float(best["target_q_mse"]),
    }
    return np.asarray(best["weights"], dtype=np.float64), meta, diagnostics


def _select_oracle_quadratic_moment_weights(
    *,
    batch: TransitionBatch,
    env: LinearGaussianEnv,
    ratio_feature_map: RatioFeatureMap,
    feature_map: StateActionFeatureMap,
    target_policy,
    ratio_gamma: float,
    value_gamma: float,
    fqe_solver_config: FQESolverConfig,
    truth,
    evaluation_samples,
    seed: int,
    candidate_ridges: Sequence[float] = QUADRATIC_MOMENT_RIDGE_GRID,
) -> tuple[np.ndarray, dict[str, object], dict[str, object]]:
    """Oracle-select the exponential-quadratic log-ratio ridge by target Q MSE."""

    candidates: list[dict[str, object]] = []
    for idx, ridge in enumerate(candidate_ridges):
        ratio = estimate_exponential_quadratic_moment_ratio(
            batch.states,
            batch.actions,
            batch.next_states,
            env=env,
            target_policy=target_policy,
            gamma=ratio_gamma,
            ratio_feature_map=ratio_feature_map,
            seed=seed + 113 * (idx + 1),
            ridge=float(ridge),
            max_steps=700,
            batch_size=2048,
            valid_fraction=0.0,
            patience=70,
        )
        weights, meta = process_raw_weights(ratio.raw_weights, min_weight=1e-8)
        fit = fit_linear_fqe(
            batch=batch,
            feature_map=feature_map,
            target_policy=target_policy,
            gamma=value_gamma,
            solver_config=fqe_solver_config,
            sample_weights=weights,
            seed=0,
        )
        target_q_mse = _target_q_mse_on_samples(
            q_function=fit.q_function,
            truth=truth,
            evaluation_samples=evaluation_samples,
        )
        candidates.append(
            {
                "ridge": float(ridge),
                "target_q_mse": float(target_q_mse),
                "ratio": ratio,
                "weights": weights,
                "meta": meta,
            }
        )
    candidates.sort(key=lambda item: (float(item["target_q_mse"]), float(item["ridge"])))
    best = candidates[0]
    ratio = best["ratio"]
    meta = _stabilization_meta(
        dict(best["meta"]),
        weight_stabilization="exponential_quadratic_moment_oracle_reg",
        cap_selection_rule="oracle_target_q_mse",
        cv_selected_ridge=float(best["ridge"]),
        cv_selected_ridge_min=float(best["ridge"]),
        cv_validation_bellman_mse=float(best["target_q_mse"]),
        tikhonov_ridge_primal=float(best["ridge"]),
    )
    diagnostics = {
        **ratio.diagnostics,
        "solver": "exponential_quadratic_moment_oracle_reg",
        "oracle_selected_target_q_mse": float(best["target_q_mse"]),
    }
    return np.asarray(best["weights"], dtype=np.float64), meta, diagnostics


def _select_oracle_minimax_q(
    *,
    batch: TransitionBatch,
    feature_map: StateActionFeatureMap,
    critic_feature_map: RatioFeatureMap,
    target_policy,
    gamma: float,
    truth,
    evaluation_samples,
    candidate_ridges: Sequence[float] = RICH_TIKHONOV_RIDGE_GRID,
):
    """Oracle-select minimax Q Tikhonov ridge by target-stationary Q MSE."""

    candidates = []
    for ridge in candidate_ridges:
        fit = fit_minimax_linear_q(
            batch,
            feature_map=feature_map,
            critic_feature_map=critic_feature_map,
            target_policy=target_policy,
            gamma=gamma,
            q_ridge=float(ridge),
            critic_ridge=float(ridge),
        )
        target_q_mse = _target_q_mse_on_samples(
            q_function=fit.q_function,
            truth=truth,
            evaluation_samples=evaluation_samples,
        )
        candidates.append({"ridge": float(ridge), "target_q_mse": float(target_q_mse), "fit": fit})
    candidates.sort(key=lambda item: (float(item["target_q_mse"]), float(item["ridge"])))
    best = candidates[0]
    fit = best["fit"]
    fit.diagnostics.update(
        {
            "solver": "minimax_q_rbf_critic_oracle_tikhonov",
            "cv_selected_ridge": float(best["ridge"]),
            "cv_selected_ridge_min": float(best["ridge"]),
            "cv_validation_bellman_mse": float(best["target_q_mse"]),
            "oracle_selected_target_q_mse": float(best["target_q_mse"]),
        }
    )
    return fit


def _select_cv_normalized_minimax_q(
    *,
    batch: TransitionBatch,
    feature_map,
    critic_feature_map: RatioFeatureMap,
    target_policy,
    gamma: float,
    candidate_etas: Sequence[float] = FIXED_TIKHONOV_ETA_GRID,
    seed: int,
    n_folds: int = 5,
):
    """Select normalized minimax Tikhonov level by held-out Bellman moment risk."""

    n = batch.states.shape[0]
    rng = np.random.default_rng(seed)
    folds = [
        np.asarray(fold, dtype=np.int64)
        for fold in np.array_split(rng.permutation(n), max(2, min(int(n_folds), n)))
        if len(fold) > 0
    ]
    all_idx = np.arange(n)
    eta_stats: list[dict[str, float]] = []
    for eta in candidate_etas:
        fold_losses: list[float] = []
        q_ridges: list[float] = []
        critic_ridges: list[float] = []
        for val_idx in folds:
            train_mask = np.ones(n, dtype=bool)
            train_mask[val_idx] = False
            train_idx = all_idx[train_mask]
            train_batch = TransitionBatch(
                states=batch.states[train_idx],
                actions=batch.actions[train_idx],
                rewards=batch.rewards[train_idx],
                next_states=batch.next_states[train_idx],
                next_actions=batch.next_actions[train_idx],
            )
            val_batch = TransitionBatch(
                states=batch.states[val_idx],
                actions=batch.actions[val_idx],
                rewards=batch.rewards[val_idx],
                next_states=batch.next_states[val_idx],
                next_actions=batch.next_actions[val_idx],
            )
            val_critic = critic_feature_map.transform(val_batch.states, val_batch.actions)
            val_critic_gram = (val_critic.T @ val_critic) / max(val_critic.shape[0], 1)
            validation_critic_ridge = 1e-3 * _trace_scale(val_critic_gram)
            q_ridge, critic_ridge, _diag = _normalized_minimax_ridges(
                batch=train_batch,
                feature_map=feature_map,
                critic_feature_map=critic_feature_map,
                target_policy=target_policy,
                gamma=gamma,
                eta=float(eta),
            )
            fit = fit_minimax_linear_q(
                train_batch,
                feature_map=feature_map,
                critic_feature_map=critic_feature_map,
                target_policy=target_policy,
                gamma=gamma,
                q_ridge=q_ridge,
                critic_ridge=critic_ridge,
            )
            fold_losses.append(
                _moment_risk(
                    fit.theta,
                    batch=val_batch,
                    feature_map=feature_map,
                    critic_feature_map=critic_feature_map,
                    target_policy=target_policy,
                    gamma=gamma,
                    critic_ridge=validation_critic_ridge,
                )
            )
            q_ridges.append(float(q_ridge))
            critic_ridges.append(float(critic_ridge))
        losses = np.asarray(fold_losses, dtype=np.float64)
        eta_stats.append(
            {
                "eta": float(eta),
                "mean": float(np.mean(losses)),
                "se": float(np.std(losses, ddof=1) / np.sqrt(max(len(losses), 1)))
                if len(losses) > 1
                else 0.0,
                "mean_q_ridge": float(np.mean(q_ridges)),
                "mean_critic_ridge": float(np.mean(critic_ridges)),
            }
        )
    eta_stats.sort(key=lambda item: (item["mean"], item["eta"]))
    best = eta_stats[0]
    q_ridge, critic_ridge, normalized_diag = _normalized_minimax_ridges(
        batch=batch,
        feature_map=feature_map,
        critic_feature_map=critic_feature_map,
        target_policy=target_policy,
        gamma=gamma,
        eta=float(best["eta"]),
    )
    fit = fit_minimax_linear_q(
        batch,
        feature_map=feature_map,
        critic_feature_map=critic_feature_map,
        target_policy=target_policy,
        gamma=gamma,
        q_ridge=q_ridge,
        critic_ridge=critic_ridge,
    )
    fit.diagnostics.update(
        {
            "solver": "minimax_q_rbf_critic_cv_normalized_tikhonov",
            "cv_selected_ridge": float(q_ridge),
            "cv_selected_ridge_min": float(q_ridge),
            "cv_selected_ridge_one_se": np.nan,
            "cv_validation_ratio_moment_risk": float(best["mean"]),
            "cv_validation_ratio_moment_risk_se": float(best["se"]),
            "cv_n_folds": len(folds),
            "normalized_tikhonov_eta": float(best["eta"]),
            "normalized_tikhonov_primal_scale": float(
                normalized_diag["normalized_tikhonov_primal_scale"]
            ),
            "normalized_tikhonov_dual_scale": float(
                normalized_diag["normalized_tikhonov_dual_scale"]
            ),
        }
    )
    return fit


def _weight_stats_with_diagnostics(
    weights: np.ndarray,
    *,
    oracle_reference_weights: np.ndarray,
    design_features: np.ndarray,
    fqe_solver_config: FQESolverConfig,
    fraction_clipped: float = 0.0,
    chosen_uniform_mix: float = 0.0,
    clip_level: float | None = None,
    extra_diagnostics: dict[str, object] | None = None,
) -> dict[str, object]:
    stats: dict[str, object] = summarize_weights(
        weights,
        fraction_clipped=fraction_clipped,
        chosen_uniform_mix=chosen_uniform_mix,
        clip_level=clip_level,
    )
    stats.update(ratio_quality(oracle_reference_weights, weights))
    stats["weighted_design_condition_number"] = weighted_design_condition_number(
        design_features,
        weights,
        ridge=fqe_solver_config.ridge,
    )
    stats.update(
        {
            "weight_stabilization": "none",
            "cap_selection_rule": "none",
            "adaptive_ess_target": np.nan,
            "fixed_cap": np.nan,
            "cv_selected_cap": np.nan,
            "cv_selected_ridge": np.nan,
            "cv_selected_ridge_min": np.nan,
            "cv_selected_ridge_one_se": np.nan,
            "cv_validation_ratio_moment_risk_se": np.nan,
            "cv_one_se_threshold": np.nan,
            "cv_n_folds": np.nan,
            "cv_validation_bellman_mse": np.nan,
            "cv_validation_ratio_moment_risk": np.nan,
            "tikhonov_ridge_primal": np.nan,
            "tikhonov_ridge_dual": np.nan,
            "normalized_tikhonov_eta": np.nan,
            "normalized_tikhonov_primal_scale": np.nan,
            "normalized_tikhonov_dual_scale": np.nan,
            "ess_fraction_before_stabilization": np.nan,
            "ess_fraction_after_stabilization": stats["effective_sample_size_fraction"],
        }
    )
    if extra_diagnostics:
        stats.update(
            {
                key: value
                for key, value in extra_diagnostics.items()
                if key in WEIGHT_META_COLUMNS
            }
        )
    stats["unstable_run_flag"] = 0.0
    stats["unstable_reason"] = "stable"
    return stats


def _classify_weighted_run(
    *,
    estimator: str,
    metrics: EstimatorMetrics,
    all_metrics: dict[str, EstimatorMetrics],
    weight_config: WeightEstimatorConfig,
) -> tuple[float, str]:
    if estimator == "standard_fqe":
        return 0.0, "stable"

    reasons: list[str] = []
    stats = metrics.weight_stats
    ess_fraction = float(stats.get("effective_sample_size_fraction", np.nan))
    q99 = float(stats.get("weight_q99", np.nan))
    weight_max = float(stats.get("weight_max", np.nan))
    condition_number = float(stats.get("weighted_design_condition_number", np.nan))
    if (
        np.isfinite(ess_fraction)
        and ess_fraction < 0.15
        or np.isfinite(q99)
        and q99 > weight_config.severe_q99_threshold
        or np.isfinite(weight_max)
        and weight_max > 5.0 * weight_config.severe_q99_threshold
    ):
        reasons.append("overlap_collapse")
    if np.isfinite(condition_number) and condition_number > 1.0e8 and not reasons:
        reasons.append("weighted_ls_ill_conditioned")

    paired_clipped = {
        "oracle_weighted_fqe": "oracle_weighted_fqe_clipped",
        "estimated_weighted_fqe": "estimated_weighted_fqe_clipped",
    }.get(estimator)
    standard = all_metrics.get("standard_fqe")
    if paired_clipped is not None and paired_clipped in all_metrics:
        clipped_metric = all_metrics[paired_clipped]
        target_q_fixed = clipped_metric.target_q_mse < 0.75 * metrics.target_q_mse
        value_fixed = clipped_metric.policy_value_squared_error < 0.75 * metrics.policy_value_squared_error
        if standard is None or (
            metrics.target_q_mse > 1.10 * standard.target_q_mse
            or metrics.policy_value_squared_error > 1.10 * standard.policy_value_squared_error
        ):
            if target_q_fixed or value_fixed:
                reasons.append("finite_sample_variance_or_clipping")

    if standard is not None and not reasons:
        q_unstable = metrics.target_q_mse > 2.0 * standard.target_q_mse and metrics.target_q_mse > 10.0
        value_unstable = (
            metrics.policy_value_squared_error > 2.0 * standard.policy_value_squared_error
            and metrics.policy_value_absolute_error > 2.0
        )
        if q_unstable or value_unstable:
            reasons.append("metric_instability_unclassified")

    if not reasons:
        return 0.0, "stable"
    return 1.0, ";".join(dict.fromkeys(reasons))


def _run_single_configuration(
    *,
    stage: str,
    output_stage_rows: list[dict[str, object]],
    gamma: float,
    shift: float,
    sample_size: int,
    process_noise_sd: float,
    behavior_action_sd: float,
    feature_regime: str,
    seed: int,
    ratio_feature_config: RatioFeatureConfig,
    fqe_solver_config: FQESolverConfig,
    weight_config: WeightEstimatorConfig,
    evaluation_config: EvaluationConfig,
    data_mode: str = "discounted_iid",
) -> None:
    env = LinearGaussianEnv(
        LinearGaussianEnvConfig(
            process_noise_sd=process_noise_sd,
            behavior_action_sd=behavior_action_sd,
        )
    )
    target_policy = env.target_policy()
    behavior_policy = env.behavior_policy(shift=shift, action_sd=behavior_action_sd)
    truth = solve_policy_truth(env=env, policy=target_policy, gamma=gamma)
    target_joint, target_state, _ = build_discounted_occupancy_mixture(env=env, policy=target_policy, gamma=gamma)
    behavior_joint, behavior_state, behavior_sequence = build_discounted_occupancy_mixture(
        env=env,
        policy=behavior_policy,
        gamma=gamma,
    )

    if data_mode == "trajectory":
        transition_data = sample_behavior_trajectory_batch(
            env=env,
            behavior_policy=behavior_policy,
            target_policy=target_policy,
            n_samples=sample_size,
            seed=seed,
        )
    else:
        transition_data = sample_discounted_behavior_batch(
            env=env,
            behavior_policy=behavior_policy,
            target_policy=target_policy,
            gamma=gamma,
            n_samples=sample_size,
            seed=seed,
            sequence=behavior_sequence,
        )
    batch = transition_data.batch

    ratio_feature_map = RatioFeatureMap.from_behavior_samples(
        batch.states,
        batch.actions,
        n_centers=ratio_feature_config.n_rbf_centers,
        bandwidth=ratio_feature_config.bandwidth,
        bandwidth_scale=ratio_feature_config.bandwidth_scale,
        standardize_features=ratio_feature_config.standardize_features,
    )
    ratio_estimate = estimate_discounted_occupancy_ratio(
        batch.states,
        batch.actions,
        batch.next_states,
        env=env,
        target_policy=target_policy,
        gamma=gamma,
        ratio_feature_map=ratio_feature_map,
        config=weight_config,
    )
    oracle_raw = oracle_density_ratio(
        batch.states,
        batch.actions,
        target_mixture=target_joint,
        behavior_mixture=behavior_joint,
    )
    feature_map = _make_q_feature_map(
        feature_regime,
        states=batch.states,
        actions=batch.actions,
        ratio_feature_config=ratio_feature_config,
    )
    oracle_weights, oracle_meta = process_raw_weights(oracle_raw, min_weight=weight_config.min_weight)
    oracle_clipped_weights, oracle_clipped_meta = process_raw_weights(
        oracle_raw,
        min_weight=weight_config.min_weight,
        clip_quantile=weight_config.clipped_clip_quantile,
        max_weight=weight_config.clipped_max_weight,
        target_ess_fraction=weight_config.clipped_target_ess_fraction,
        uniform_mix=weight_config.clipped_uniform_mix,
        max_uniform_mix=weight_config.clipped_max_uniform_mix,
    )
    estimated_clip95_weights, estimated_clip95_meta = process_raw_weights(
        ratio_estimate.raw_weights,
        min_weight=weight_config.min_weight,
        clip_quantile=0.95,
        max_weight=15.0,
    )
    estimated_clip99_ess40_weights, estimated_clip99_ess40_meta = process_raw_weights(
        ratio_estimate.raw_weights,
        min_weight=weight_config.min_weight,
        clip_quantile=0.99,
        max_weight=weight_config.clipped_max_weight,
        target_ess_fraction=0.40,
        uniform_mix=0.0,
        max_uniform_mix=0.65,
    )
    standard_weights = np.ones(sample_size, dtype=np.float64)
    feature_map = _make_q_feature_map(
        feature_regime,
        states=batch.states,
        actions=batch.actions,
        ratio_feature_config=ratio_feature_config,
    )
    design_features = feature_map.transform(batch.states, batch.actions)

    clipped_meta = {
        "fraction_clipped": ratio_estimate.diagnostics["clipped_fraction_clipped"],
        "chosen_uniform_mix": 0.0,
        "clip_level": ratio_estimate.diagnostics["clipped_clip_level"],
    }

    weight_methods: dict[str, tuple[np.ndarray, dict[str, float]]] = {
        "standard_fqe": (
            standard_weights,
            {"fraction_clipped": 0.0, "chosen_uniform_mix": 0.0, "clip_level": np.nan},
        ),
        "oracle_weighted_fqe": (oracle_weights, oracle_meta),
        "oracle_weighted_fqe_clipped": (oracle_clipped_weights, oracle_clipped_meta),
        "estimated_weighted_fqe": (
            ratio_estimate.processed_weights,
            {"fraction_clipped": 0.0, "chosen_uniform_mix": 0.0, "clip_level": np.nan},
        ),
        "estimated_weighted_fqe_clipped": (ratio_estimate.clipped_weights, clipped_meta),
        "estimated_weighted_fqe_clip95": (estimated_clip95_weights, estimated_clip95_meta),
        "estimated_weighted_fqe_clip99_ess40": (
            estimated_clip99_ess40_weights,
            estimated_clip99_ess40_meta,
        ),
    }

    evaluation_samples = draw_evaluation_samples(
        env=env,
        target_policy=target_policy,
        evaluation_config=evaluation_config,
        target_joint_mixture=target_joint,
        behavior_joint_mixture=behavior_joint,
        target_state_mixture=target_state,
        behavior_state_mixture=behavior_state,
        rng=np.random.default_rng(10_000 + seed),
    )

    metrics_by_estimator: dict[str, EstimatorMetrics] = {}
    for estimator_name, (sample_weights, weight_stats) in weight_methods.items():
        clip_level = float(weight_stats.get("clip_level", np.nan))
        diagnostics = _weight_stats_with_diagnostics(
            sample_weights,
            oracle_reference_weights=oracle_weights,
            design_features=design_features,
            fqe_solver_config=fqe_solver_config,
            fraction_clipped=float(weight_stats.get("fraction_clipped", 0.0)),
            chosen_uniform_mix=float(weight_stats.get("chosen_uniform_mix", 0.0)),
            clip_level=None if np.isnan(clip_level) else clip_level,
            extra_diagnostics=weight_stats,
        )
        fit = fit_linear_fqe(
            batch=batch,
            feature_map=feature_map,
            target_policy=target_policy,
            gamma=gamma,
            solver_config=fqe_solver_config,
            sample_weights=sample_weights,
            seed=seed,
        )
        metrics = evaluate_estimator(
            estimator=estimator_name,
            q_function=fit.q_function,
            truth=truth,
            env=env,
            target_policy=target_policy,
            gamma=gamma,
            evaluation_config=evaluation_config,
            target_joint_mixture=target_joint,
            behavior_joint_mixture=behavior_joint,
            target_state_mixture=target_state,
            behavior_state_mixture=behavior_state,
            rng=np.random.default_rng(10_000 + seed),
            weight_stats=diagnostics,
            evaluation_samples=evaluation_samples,
        )
        metrics_by_estimator[estimator_name] = metrics

    for estimator_name, metrics in metrics_by_estimator.items():
        unstable_flag, unstable_reason = _classify_weighted_run(
            estimator=estimator_name,
            metrics=metrics,
            all_metrics=metrics_by_estimator,
            weight_config=weight_config,
        )
        metrics.weight_stats["unstable_run_flag"] = unstable_flag
        metrics.weight_stats["unstable_reason"] = unstable_reason
        output_stage_rows.append(
            _metric_row(
                stage=stage,
                gamma=gamma,
                shift=shift,
                sample_size=sample_size,
                process_noise_sd=process_noise_sd,
                behavior_action_sd=behavior_action_sd,
                feature_regime=feature_regime,
                seed=seed,
                metrics=metrics,
            )
        )


def _stage_neural_ratio_config(stage: str) -> NeuralRatioConfig:
    if stage == "gamma_smoke":
        return NeuralRatioConfig(max_steps=80, early_stopping_patience=6, hidden_dims=(48, 48))
    if stage in {"gamma_design", "gamma_final", "gamma_paper"}:
        return NeuralRatioConfig(max_steps=140, early_stopping_patience=8, hidden_dims=(64, 64))
    return NeuralRatioConfig()


def _stage_neural_fqe_config(stage: str) -> NeuralFQEConfig:
    if stage == "gamma_smoke":
        return NeuralFQEConfig(n_outer_iters=8, epochs_per_iter=4, hidden_dims=(48, 48), batch_size=512)
    if stage in {"gamma_design", "gamma_final", "gamma_paper"}:
        return NeuralFQEConfig(n_outer_iters=10, epochs_per_iter=5, hidden_dims=(64, 64), batch_size=512)
    return NeuralFQEConfig()


def _gamma_weight_stats_with_diagnostics(
    weights: np.ndarray,
    *,
    oracle_reference_weights: np.ndarray,
    design_features: np.ndarray,
    fqe_solver_config: FQESolverConfig,
    source_diagnostics: dict[str, object],
    ratio_feature_map: RatioFeatureMap,
    feature_map: StateActionFeatureMap,
    batch_states: np.ndarray,
    batch_actions: np.ndarray,
    target_sa: np.ndarray,
    fraction_clipped: float = 0.0,
    chosen_uniform_mix: float = 0.0,
    clip_level: float | None = None,
    extra_diagnostics: dict[str, object] | None = None,
) -> dict[str, object]:
    stats = _weight_stats_with_diagnostics(
        weights,
        oracle_reference_weights=oracle_reference_weights,
        design_features=design_features,
        fqe_solver_config=fqe_solver_config,
        fraction_clipped=fraction_clipped,
        chosen_uniform_mix=chosen_uniform_mix,
        clip_level=clip_level,
        extra_diagnostics=extra_diagnostics,
    )
    sample_ratio_features = ratio_feature_map.transform(batch_states, batch_actions)
    target_ratio_features = ratio_feature_map.transform(target_sa[:, :2], target_sa[:, 2:])
    sample_fqe_features = feature_map.transform(batch_states, batch_actions)
    target_fqe_features = feature_map.transform(target_sa[:, :2], target_sa[:, 2:])
    stats.update(
        {
            "ratio_moment_violation_l2": float(source_diagnostics.get("moment_violation_l2", np.nan)),
            "ratio_normalization_error": float(source_diagnostics.get("normalization_error", np.nan)),
            "target_ratio_feature_calibration_l2": feature_calibration_l2(
                weights,
                sample_ratio_features,
                target_ratio_features,
            ),
            "target_fqe_feature_calibration_l2": feature_calibration_l2(
                weights,
                sample_fqe_features,
                target_fqe_features,
            ),
            "ratio_solver": str(source_diagnostics.get("solver", "analytic_or_uniform")),
        }
    )
    return stats


def _run_single_gamma_configuration(
    *,
    stage: str,
    output_stage_rows: list[dict[str, object]],
    value_gamma: float,
    ratio_gamma: float,
    shift: float,
    sample_size: int,
    process_noise_sd: float,
    target_action_sd: float | None = None,
    behavior_action_sd: float,
    feature_regime: str,
    seed: int,
    include_neural_ratio: bool,
    include_neural_fqe: bool,
    ratio_feature_config: RatioFeatureConfig,
    fqe_solver_config: FQESolverConfig,
    weight_config: WeightEstimatorConfig,
    neural_ratio_config: NeuralRatioConfig,
    neural_fqe_config: NeuralFQEConfig,
    evaluation_config: EvaluationConfig,
    data_mode: str = "matched_iid",
    behavior_shift_direction_scale: float = 1.0,
    behavior_shift_direction: Sequence[float] | None = None,
    estimator_mode: str = "full",
) -> None:
    default_shift_direction = LinearGaussianEnvConfig().behavior_shift_direction
    if behavior_shift_direction is None:
        shift_direction = default_shift_direction * float(behavior_shift_direction_scale)
    else:
        shift_direction = np.asarray(behavior_shift_direction, dtype=np.float64).reshape(1, 2)
    env_config = LinearGaussianEnvConfig(
        process_noise_sd=process_noise_sd,
        target_action_sd=LinearGaussianEnvConfig().target_action_sd if target_action_sd is None else float(target_action_sd),
        behavior_action_sd=behavior_action_sd,
        behavior_shift_direction=shift_direction,
    )
    env = LinearGaussianEnv(
        env_config
    )
    target_policy = env.target_policy()
    behavior_policy = env.behavior_policy(shift=shift, action_sd=behavior_action_sd)
    truth = solve_policy_truth(env=env, policy=target_policy, gamma=value_gamma)
    target_joint, target_state, _, reference_kind = build_reference_occupancy_mixture(
        env=env,
        policy=target_policy,
        ratio_gamma=ratio_gamma,
    )
    behavior_joint, behavior_state, behavior_sequence, _ = build_reference_occupancy_mixture(
        env=env,
        policy=behavior_policy,
        ratio_gamma=ratio_gamma,
    )
    if data_mode == "trajectory":
        transition_data = sample_behavior_trajectory_batch(
            env=env,
            behavior_policy=behavior_policy,
            target_policy=target_policy,
            n_samples=sample_size,
            seed=seed,
        )
    else:
        transition_data = sample_reference_behavior_batch(
            env=env,
            behavior_policy=behavior_policy,
            target_policy=target_policy,
            ratio_gamma=ratio_gamma,
            n_samples=sample_size,
            seed=seed,
            sequence=behavior_sequence,
        )
    batch = transition_data.batch
    ratio_feature_map = RatioFeatureMap.from_behavior_samples(
        batch.states,
        batch.actions,
        n_centers=ratio_feature_config.n_rbf_centers,
        bandwidth=ratio_feature_config.bandwidth,
        bandwidth_scale=ratio_feature_config.bandwidth_scale,
        standardize_features=ratio_feature_config.standardize_features,
    )
    feature_map = _make_q_feature_map(
        feature_regime,
        states=batch.states,
        actions=batch.actions,
        ratio_feature_config=ratio_feature_config,
    )
    oracle_raw = oracle_density_ratio(
        batch.states,
        batch.actions,
        target_mixture=target_joint,
        behavior_mixture=behavior_joint,
    )
    oracle_weights, oracle_meta = process_raw_weights(oracle_raw, min_weight=weight_config.min_weight)
    standard_weights = np.ones(sample_size, dtype=np.float64)
    design_features = feature_map.transform(batch.states, batch.actions)
    evaluation_samples = draw_evaluation_samples(
        env=env,
        target_policy=target_policy,
        evaluation_config=evaluation_config,
        target_joint_mixture=target_joint,
        behavior_joint_mixture=behavior_joint,
        target_state_mixture=target_state,
        behavior_state_mixture=behavior_state,
        rng=np.random.default_rng(20_000 + seed),
    )
    if estimator_mode in {"outer_control", "minimax_sensitivity"}:
        weight_methods: dict[str, tuple[np.ndarray, dict[str, float], dict[str, object]]] = {
            "standard": (
                standard_weights,
                {"fraction_clipped": 0.0, "chosen_uniform_mix": 0.0, "clip_level": np.nan},
                {"solver": "uniform", "moment_violation_l2": np.nan, "normalization_error": 0.0},
            ),
            "oracle_raw": (
                oracle_weights,
                oracle_meta,
                {"solver": "oracle_exact", "moment_violation_l2": 0.0, "normalization_error": 0.0},
            ),
        }
        estimator_specs = [
            ("linear_standard_fqe", "linear", "standard"),
            ("linear_oracle_raw_fqe", "linear", "oracle_raw"),
        ]
        minimax_eta_by_estimator: dict[str, float] = {
            "linear_minimax_q_rbf": FIXED_TIKHONOV_MAIN_ETA,
        }
        if estimator_mode == "outer_control":
            estimator_specs.append(("linear_minimax_q_rbf", "linear", "standard"))
        else:
            estimator_specs.extend(
                [
                    ("linear_minimax_q_unregularized", "linear", "standard"),
                    ("linear_minimax_q_cv_tikhonov", "linear", "standard"),
                    ("linear_minimax_q_rbf", "linear", "standard"),
                ]
            )
            for eta in FIXED_TIKHONOV_ETA_GRID:
                if abs(float(eta) - FIXED_TIKHONOV_MAIN_ETA) >= 1e-15:
                    estimator_name = f"linear_minimax_q_eta_{eta:g}"
                    minimax_eta_by_estimator[estimator_name] = float(eta)
                    estimator_specs.append((estimator_name, "linear", "standard"))
        metrics_by_estimator: dict[str, EstimatorMetrics] = {}
        for estimator_name, fqe_family, weight_key in estimator_specs:
            sample_weights, weight_meta, source_diag = weight_methods[weight_key]
            clip_level = float(weight_meta.get("clip_level", np.nan))
            diagnostics = _gamma_weight_stats_with_diagnostics(
                sample_weights,
                oracle_reference_weights=oracle_weights,
                design_features=design_features,
                fqe_solver_config=fqe_solver_config,
                source_diagnostics=source_diag,
                ratio_feature_map=ratio_feature_map,
                feature_map=feature_map,
                batch_states=batch.states,
                batch_actions=batch.actions,
                target_sa=evaluation_samples.target_sa,
                fraction_clipped=float(weight_meta.get("fraction_clipped", 0.0)),
                chosen_uniform_mix=float(weight_meta.get("chosen_uniform_mix", 0.0)),
                clip_level=None if np.isnan(clip_level) else clip_level,
                extra_diagnostics=weight_meta,
            )
            if (
                estimator_name in minimax_eta_by_estimator
                or estimator_name
                in {"linear_minimax_q_unregularized", "linear_minimax_q_cv_tikhonov"}
            ):
                normalized_minimax_diag: dict[str, float] = {}
                if estimator_name == "linear_minimax_q_cv_tikhonov":
                    fit = _select_cv_normalized_minimax_q(
                        batch=batch,
                        feature_map=feature_map,
                        critic_feature_map=ratio_feature_map,
                        target_policy=target_policy,
                        gamma=value_gamma,
                        seed=90_000 + seed,
                    )
                    normalized_minimax_diag = {
                        "normalized_tikhonov_eta": float(
                            fit.diagnostics.get("normalized_tikhonov_eta", np.nan)
                        ),
                        "normalized_tikhonov_primal_scale": float(
                            fit.diagnostics.get("normalized_tikhonov_primal_scale", np.nan)
                        ),
                        "normalized_tikhonov_dual_scale": float(
                            fit.diagnostics.get("normalized_tikhonov_dual_scale", np.nan)
                        ),
                    }
                elif estimator_name == "linear_minimax_q_unregularized":
                    fit = fit_minimax_linear_q(
                        batch,
                        feature_map=feature_map,
                        critic_feature_map=ratio_feature_map,
                        target_policy=target_policy,
                        gamma=value_gamma,
                        q_ridge=0.0,
                        critic_ridge=0.0,
                    )
                else:
                    q_ridge, critic_ridge, normalized_minimax_diag = _normalized_minimax_ridges(
                        batch=batch,
                        feature_map=feature_map,
                        critic_feature_map=ratio_feature_map,
                        target_policy=target_policy,
                        gamma=value_gamma,
                        eta=minimax_eta_by_estimator[estimator_name],
                    )
                    fit = fit_minimax_linear_q(
                        batch,
                        feature_map=feature_map,
                        critic_feature_map=ratio_feature_map,
                        target_policy=target_policy,
                        gamma=value_gamma,
                        q_ridge=q_ridge,
                        critic_ridge=critic_ridge,
                    )
                q_function = fit.q_function
                diagnostics.update(
                    {
                        "ratio_solver": str(fit.diagnostics.get("solver", "minimax_q_rbf_critic")),
                        "ratio_moment_violation_l2": float(
                            fit.diagnostics.get("moment_violation_l2", np.nan)
                        ),
                        "ratio_normalization_error": 0.0,
                        "weight_stabilization": "not_weighted",
                        "cap_selection_rule": "none"
                        if estimator_name == "linear_minimax_q_unregularized"
                        else (
                            "bellman_moment_cv_normalized_eta"
                            if estimator_name == "linear_minimax_q_cv_tikhonov"
                            else "pre_specified_normalized_eta"
                        ),
                        "cv_selected_ridge": float(fit.diagnostics.get("cv_selected_ridge", np.nan)),
                        "cv_selected_ridge_min": float(fit.diagnostics.get("cv_selected_ridge_min", np.nan)),
                        "cv_selected_ridge_one_se": np.nan,
                        "cv_validation_bellman_mse": float(
                            fit.diagnostics.get("cv_validation_bellman_mse", np.nan)
                        ),
                        "cv_validation_ratio_moment_risk": float(
                            fit.diagnostics.get("critic_norm_moment_risk", np.nan)
                        ),
                        "cv_validation_ratio_moment_risk_se": np.nan,
                        "cv_n_folds": np.nan,
                        "tikhonov_ridge_primal": float(fit.diagnostics.get("q_ridge", np.nan)),
                        "tikhonov_ridge_dual": float(fit.diagnostics.get("critic_ridge", np.nan)),
                        "normalized_tikhonov_eta": float(
                            normalized_minimax_diag.get("normalized_tikhonov_eta", np.nan)
                        ),
                        "normalized_tikhonov_primal_scale": float(
                            normalized_minimax_diag.get("normalized_tikhonov_primal_scale", np.nan)
                        ),
                        "normalized_tikhonov_dual_scale": float(
                            normalized_minimax_diag.get("normalized_tikhonov_dual_scale", np.nan)
                        ),
                        "weighted_design_condition_number": float(
                            fit.diagnostics.get("primal_condition_number", np.nan)
                        ),
                    }
                )
            else:
                fit = fit_linear_fqe(
                    batch=batch,
                    feature_map=feature_map,
                    target_policy=target_policy,
                    gamma=value_gamma,
                    solver_config=fqe_solver_config,
                    sample_weights=sample_weights,
                    seed=seed,
                )
                q_function = fit.q_function
            metrics = evaluate_estimator(
                estimator=estimator_name,
                q_function=q_function,
                truth=truth,
                env=env,
                target_policy=target_policy,
                gamma=value_gamma,
                evaluation_config=evaluation_config,
                target_joint_mixture=target_joint,
                behavior_joint_mixture=behavior_joint,
                target_state_mixture=target_state,
                behavior_state_mixture=behavior_state,
                rng=np.random.default_rng(20_000 + seed),
                weight_stats=diagnostics,
                evaluation_samples=evaluation_samples,
            )
            metrics_by_estimator[estimator_name] = metrics
            unstable_flag, unstable_reason = _classify_weighted_run(
                estimator=estimator_name,
                metrics=metrics,
                all_metrics=metrics_by_estimator,
                weight_config=weight_config,
            )
            metrics.weight_stats["unstable_run_flag"] = unstable_flag
            metrics.weight_stats["unstable_reason"] = unstable_reason
            output_stage_rows.append(
                _gamma_metric_row(
                    stage=stage,
                    value_gamma=value_gamma,
                    ratio_gamma=ratio_gamma,
                    reference_distribution=reference_kind,
                    data_mode=data_mode,
                    shift=shift,
                    sample_size=sample_size,
                    process_noise_sd=process_noise_sd,
                    behavior_action_sd=behavior_action_sd,
                    feature_regime=feature_regime,
                    seed=seed,
                    fqe_family=fqe_family,
                    ratio_estimator=weight_key,
                    metrics=metrics,
                )
            )
        return
    if estimator_mode == "oracle_tuned_comparison":
        tikhonov_selections = _select_cv_tikhonov_ratios(
            batch=batch,
            env=env,
            ratio_feature_map=ratio_feature_map,
            target_policy=target_policy,
            gamma=ratio_gamma,
            base_config=weight_config,
            seed=50_000 + seed,
        )
        tikhonov_min = tikhonov_selections["min"]
        linear_tikhonov_weights, linear_tikhonov_meta = process_raw_weights(
            tikhonov_min.ratio.raw_weights,
            min_weight=weight_config.min_weight,
        )
        quadratic_moment_cv = _select_cv_quadratic_moment_ratio(
            batch=batch,
            env=env,
            ratio_feature_map=ratio_feature_map,
            target_policy=target_policy,
            gamma=ratio_gamma,
            seed=75_000 + seed,
        )
        quadratic_moment_cv_weights, quadratic_moment_cv_meta = process_raw_weights(
            quadratic_moment_cv.ratio.raw_weights,
            min_weight=weight_config.min_weight,
        )
        linear_oracle_tikhonov_weights, linear_oracle_tikhonov_meta, linear_oracle_tikhonov_diag = (
            _select_oracle_tikhonov_ratio_weights(
                batch=batch,
                env=env,
                ratio_feature_map=ratio_feature_map,
                feature_map=feature_map,
                target_policy=target_policy,
                ratio_gamma=ratio_gamma,
                value_gamma=value_gamma,
                base_weight_config=weight_config,
                fqe_solver_config=fqe_solver_config,
                truth=truth,
                evaluation_samples=evaluation_samples,
            )
        )
        quadratic_moment_oracle_weights, quadratic_moment_oracle_meta, quadratic_moment_oracle_diag = (
            _select_oracle_quadratic_moment_weights(
                batch=batch,
                env=env,
                ratio_feature_map=ratio_feature_map,
                feature_map=feature_map,
                target_policy=target_policy,
                ratio_gamma=ratio_gamma,
                value_gamma=value_gamma,
                fqe_solver_config=fqe_solver_config,
                truth=truth,
                evaluation_samples=evaluation_samples,
                seed=85_000 + seed,
            )
        )
        weight_methods: dict[str, tuple[np.ndarray, dict[str, float], dict[str, object]]] = {
            "standard": (
                standard_weights,
                {"fraction_clipped": 0.0, "chosen_uniform_mix": 0.0, "clip_level": np.nan},
                {"solver": "uniform", "moment_violation_l2": np.nan, "normalization_error": 0.0},
            ),
            "oracle_raw": (
                oracle_weights,
                oracle_meta,
                {"solver": "oracle_exact", "moment_violation_l2": 0.0, "normalization_error": 0.0},
            ),
            "linear_tikhonov": (
                linear_tikhonov_weights,
                _stabilization_meta(
                    linear_tikhonov_meta,
                    weight_stabilization="tikhonov_ratio",
                    cap_selection_rule="ratio_moment_cv_min",
                    cv_selected_ridge=tikhonov_min.config.ridge_primal,
                    cv_selected_ridge_min=tikhonov_min.diagnostics["cv_selected_ridge_min"],
                    cv_selected_ridge_one_se=tikhonov_min.diagnostics["cv_selected_ridge_one_se"],
                    cv_validation_ratio_moment_risk=tikhonov_min.validation_loss,
                    cv_validation_ratio_moment_risk_se=tikhonov_min.validation_loss_se,
                    cv_one_se_threshold=tikhonov_min.diagnostics["cv_one_se_threshold"],
                    cv_n_folds=int(tikhonov_min.diagnostics["cv_n_folds"]),
                    tikhonov_ridge_primal=tikhonov_min.config.ridge_primal,
                    tikhonov_ridge_dual=tikhonov_min.config.ridge_dual,
                ),
                {
                    **tikhonov_min.ratio.diagnostics,
                    **tikhonov_min.diagnostics,
                    "solver": "linear_tikhonov_reduced_moment",
                },
            ),
            "linear_oracle_tikhonov": (
                linear_oracle_tikhonov_weights,
                linear_oracle_tikhonov_meta,
                linear_oracle_tikhonov_diag,
            ),
            "quadratic_moment_cv": (
                quadratic_moment_cv_weights,
                _stabilization_meta(
                    quadratic_moment_cv_meta,
                    weight_stabilization="exponential_quadratic_moment_cv_reg",
                    cap_selection_rule="held_out_moment_risk",
                    cv_selected_ridge=quadratic_moment_cv.ridge,
                    cv_selected_ridge_min=quadratic_moment_cv.diagnostics["cv_selected_ridge_min"],
                    cv_selected_ridge_one_se=quadratic_moment_cv.diagnostics["cv_selected_ridge_one_se"],
                    cv_validation_ratio_moment_risk=quadratic_moment_cv.validation_loss,
                    cv_validation_ratio_moment_risk_se=quadratic_moment_cv.validation_loss_se,
                    cv_one_se_threshold=quadratic_moment_cv.diagnostics["cv_one_se_threshold"],
                    cv_n_folds=int(quadratic_moment_cv.diagnostics["cv_n_folds"]),
                    tikhonov_ridge_primal=quadratic_moment_cv.ridge,
                    tikhonov_ridge_dual=np.nan,
                ),
                {
                    **quadratic_moment_cv.ratio.diagnostics,
                    **quadratic_moment_cv.diagnostics,
                    "solver": "exponential_quadratic_moment_cv_reg",
                },
            ),
            "quadratic_moment_oracle": (
                quadratic_moment_oracle_weights,
                quadratic_moment_oracle_meta,
                quadratic_moment_oracle_diag,
            ),
        }
        estimator_specs = [
            ("linear_standard_fqe", "linear", "standard"),
            ("linear_oracle_raw_fqe", "linear", "oracle_raw"),
            ("linear_estimated_tikhonov_fqe", "linear", "linear_tikhonov"),
            ("linear_estimated_rbf_oracle_tikhonov_fqe", "linear", "linear_oracle_tikhonov"),
            ("linear_estimated_quadratic_moment_cv_fqe", "linear", "quadratic_moment_cv"),
            ("linear_estimated_quadratic_moment_oracle_fqe", "linear", "quadratic_moment_oracle"),
            ("linear_minimax_q_cv_tikhonov", "linear", "standard"),
            ("linear_minimax_q_oracle_tikhonov", "linear", "standard"),
        ]
        metrics_by_estimator: dict[str, EstimatorMetrics] = {}
        for estimator_name, fqe_family, weight_key in estimator_specs:
            sample_weights, weight_meta, source_diag = weight_methods[weight_key]
            clip_level = float(weight_meta.get("clip_level", np.nan))
            diagnostics = _gamma_weight_stats_with_diagnostics(
                sample_weights,
                oracle_reference_weights=oracle_weights,
                design_features=design_features,
                fqe_solver_config=fqe_solver_config,
                source_diagnostics=source_diag,
                ratio_feature_map=ratio_feature_map,
                feature_map=feature_map,
                batch_states=batch.states,
                batch_actions=batch.actions,
                target_sa=evaluation_samples.target_sa,
                fraction_clipped=float(weight_meta.get("fraction_clipped", 0.0)),
                chosen_uniform_mix=float(weight_meta.get("chosen_uniform_mix", 0.0)),
                clip_level=None if np.isnan(clip_level) else clip_level,
                extra_diagnostics=weight_meta,
            )
            if estimator_name in {
                "linear_minimax_q_cv_tikhonov",
                "linear_minimax_q_oracle_tikhonov",
            }:
                if estimator_name == "linear_minimax_q_cv_tikhonov":
                    fit = _select_cv_normalized_minimax_q(
                        batch=batch,
                        feature_map=feature_map,
                        critic_feature_map=ratio_feature_map,
                        target_policy=target_policy,
                        gamma=value_gamma,
                        seed=90_000 + seed,
                    )
                    cap_selection_rule = "bellman_moment_cv_normalized_eta"
                else:
                    fit = _select_oracle_minimax_q(
                        batch=batch,
                        feature_map=feature_map,
                        critic_feature_map=ratio_feature_map,
                        target_policy=target_policy,
                        gamma=value_gamma,
                        truth=truth,
                        evaluation_samples=evaluation_samples,
                    )
                    cap_selection_rule = "oracle_target_q_mse"
                q_function = fit.q_function
                diagnostics.update(
                    {
                        "ratio_solver": str(fit.diagnostics.get("solver", "minimax_q_rbf_critic")),
                        "ratio_moment_violation_l2": float(
                            fit.diagnostics.get("moment_violation_l2", np.nan)
                        ),
                        "ratio_normalization_error": 0.0,
                        "weight_stabilization": "not_weighted",
                        "cap_selection_rule": cap_selection_rule,
                        "cv_selected_ridge": float(fit.diagnostics.get("cv_selected_ridge", np.nan)),
                        "cv_selected_ridge_min": float(fit.diagnostics.get("cv_selected_ridge_min", np.nan)),
                        "cv_selected_ridge_one_se": np.nan,
                        "cv_validation_bellman_mse": float(
                            fit.diagnostics.get("cv_validation_bellman_mse", np.nan)
                        ),
                        "cv_validation_ratio_moment_risk": float(
                            fit.diagnostics.get("critic_norm_moment_risk", np.nan)
                        ),
                        "cv_validation_ratio_moment_risk_se": np.nan,
                        "cv_n_folds": np.nan,
                        "tikhonov_ridge_primal": float(fit.diagnostics.get("q_ridge", np.nan)),
                        "tikhonov_ridge_dual": float(fit.diagnostics.get("critic_ridge", np.nan)),
                        "normalized_tikhonov_eta": float(
                            fit.diagnostics.get("normalized_tikhonov_eta", np.nan)
                        ),
                        "normalized_tikhonov_primal_scale": float(
                            fit.diagnostics.get("normalized_tikhonov_primal_scale", np.nan)
                        ),
                        "normalized_tikhonov_dual_scale": float(
                            fit.diagnostics.get("normalized_tikhonov_dual_scale", np.nan)
                        ),
                        "weighted_design_condition_number": float(
                            fit.diagnostics.get("primal_condition_number", np.nan)
                        ),
                    }
                )
            else:
                fit = fit_linear_fqe(
                    batch=batch,
                    feature_map=feature_map,
                    target_policy=target_policy,
                    gamma=value_gamma,
                    solver_config=fqe_solver_config,
                    sample_weights=sample_weights,
                    seed=seed,
                )
                q_function = fit.q_function
            metrics = evaluate_estimator(
                estimator=estimator_name,
                q_function=q_function,
                truth=truth,
                env=env,
                target_policy=target_policy,
                gamma=value_gamma,
                evaluation_config=evaluation_config,
                target_joint_mixture=target_joint,
                behavior_joint_mixture=behavior_joint,
                target_state_mixture=target_state,
                behavior_state_mixture=behavior_state,
                rng=np.random.default_rng(20_000 + seed),
                weight_stats=diagnostics,
                evaluation_samples=evaluation_samples,
            )
            metrics_by_estimator[estimator_name] = metrics
            unstable_flag, unstable_reason = _classify_weighted_run(
                estimator=estimator_name,
                metrics=metrics,
                all_metrics=metrics_by_estimator,
                weight_config=weight_config,
            )
            metrics.weight_stats["unstable_run_flag"] = unstable_flag
            metrics.weight_stats["unstable_reason"] = unstable_reason
            output_stage_rows.append(
                _gamma_metric_row(
                    stage=stage,
                    value_gamma=value_gamma,
                    ratio_gamma=ratio_gamma,
                    reference_distribution=reference_kind,
                    data_mode=data_mode,
                    shift=shift,
                    sample_size=sample_size,
                    process_noise_sd=process_noise_sd,
                    behavior_action_sd=behavior_action_sd,
                    feature_regime=feature_regime,
                    seed=seed,
                    fqe_family=fqe_family,
                    ratio_estimator=weight_key,
                    metrics=metrics,
                )
            )
        return
    linear_ratio = estimate_discounted_occupancy_ratio(
        batch.states,
        batch.actions,
        batch.next_states,
        env=env,
        target_policy=target_policy,
        gamma=ratio_gamma,
        ratio_feature_map=ratio_feature_map,
        config=weight_config,
    )
    unregularized_weight_config = replace(
        weight_config,
        ridge_primal=0.0,
        ridge_dual=0.0,
    )
    linear_unregularized_ratio = estimate_discounted_occupancy_ratio(
        batch.states,
        batch.actions,
        batch.next_states,
        env=env,
        target_policy=target_policy,
        gamma=ratio_gamma,
        ratio_feature_map=ratio_feature_map,
        config=unregularized_weight_config,
    )
    normalized_ratio_tikhonov: dict[float, tuple[object, WeightEstimatorConfig, dict[str, float]]] = {}
    for eta in FIXED_TIKHONOV_ETA_GRID:
        eta_config, eta_diag = _normalized_ratio_tikhonov_config(
            batch=batch,
            ratio_feature_map=ratio_feature_map,
            target_policy=target_policy,
            gamma=ratio_gamma,
            base_config=weight_config,
            eta=float(eta),
        )
        eta_ratio = estimate_discounted_occupancy_ratio(
            batch.states,
            batch.actions,
            batch.next_states,
            env=env,
            target_policy=target_policy,
            gamma=ratio_gamma,
            ratio_feature_map=ratio_feature_map,
            config=eta_config,
        )
        normalized_ratio_tikhonov[float(eta)] = (eta_ratio, eta_config, eta_diag)
    quadratic_moment_ratio = estimate_exponential_quadratic_moment_ratio(
        batch.states,
        batch.actions,
        batch.next_states,
        env=env,
        target_policy=target_policy,
        gamma=ratio_gamma,
        ratio_feature_map=ratio_feature_map,
        seed=70_000 + seed,
    )
    quadratic_moment_cv = _select_cv_quadratic_moment_ratio(
        batch=batch,
        env=env,
        ratio_feature_map=ratio_feature_map,
        target_policy=target_policy,
        gamma=ratio_gamma,
        seed=75_000 + seed,
    )
    tikhonov_selections = _select_cv_tikhonov_ratios(
        batch=batch,
        env=env,
        ratio_feature_map=ratio_feature_map,
        target_policy=target_policy,
        gamma=ratio_gamma,
        base_config=weight_config,
        seed=50_000 + seed,
    )
    tikhonov_min = tikhonov_selections["min"]
    tikhonov_one_se = tikhonov_selections["one_se"]
    neural_ratio = None
    if include_neural_ratio:
        neural_ratio = estimate_neural_discounted_occupancy_ratio(
            batch.states,
            batch.actions,
            batch.next_states,
            env=env,
            target_policy=target_policy,
            ratio_gamma=ratio_gamma,
            ratio_feature_map=ratio_feature_map,
            config=neural_ratio_config,
            seed=10_000 + seed,
        )
    oracle_clipped_weights, oracle_clipped_meta = process_raw_weights(
        oracle_raw,
        min_weight=weight_config.min_weight,
        clip_quantile=weight_config.clipped_clip_quantile,
        max_weight=weight_config.clipped_max_weight,
        target_ess_fraction=0.40,
        uniform_mix=weight_config.clipped_uniform_mix,
        max_uniform_mix=0.65,
    )
    fixed_cap_level = 2.0
    adaptive_ess_target = 0.60
    linear_clipped_weights, linear_clipped_meta = process_raw_weights(
        linear_ratio.raw_weights,
        min_weight=weight_config.min_weight,
        clip_quantile=0.99,
        max_weight=weight_config.clipped_max_weight,
        target_ess_fraction=0.40,
        max_uniform_mix=0.65,
    )
    linear_fixed_cap_weights, linear_fixed_cap_meta = process_raw_weights(
        linear_ratio.raw_weights,
        min_weight=weight_config.min_weight,
        max_weight=fixed_cap_level,
    )
    linear_ess_winsor_weights, linear_ess_winsor_meta = process_ess_adaptive_winsor_weights(
        linear_ratio.raw_weights,
        min_weight=weight_config.min_weight,
        target_ess_fraction=adaptive_ess_target,
    )
    linear_tikhonov_weights, linear_tikhonov_meta = process_raw_weights(
        tikhonov_min.ratio.raw_weights,
        min_weight=weight_config.min_weight,
    )
    linear_tikhonov_xfit_weights, linear_tikhonov_xfit_meta = process_raw_weights(
        tikhonov_min.oof_raw_weights,
        min_weight=weight_config.min_weight,
    )
    linear_tikhonov_1se_weights, linear_tikhonov_1se_meta = process_raw_weights(
        tikhonov_one_se.ratio.raw_weights,
        min_weight=weight_config.min_weight,
    )
    linear_tikhonov_1se_xfit_weights, linear_tikhonov_1se_xfit_meta = process_raw_weights(
        tikhonov_one_se.oof_raw_weights,
        min_weight=weight_config.min_weight,
    )
    linear_unregularized_weights, linear_unregularized_meta = process_raw_weights(
        linear_unregularized_ratio.raw_weights,
        min_weight=weight_config.min_weight,
    )
    normalized_ratio_tikhonov_weights: dict[float, tuple[np.ndarray, dict[str, float]]] = {}
    for eta, (eta_ratio, _eta_config, _eta_diag) in normalized_ratio_tikhonov.items():
        normalized_ratio_tikhonov_weights[eta] = process_raw_weights(
            eta_ratio.raw_weights,
            min_weight=weight_config.min_weight,
        )
    quadratic_moment_weights, quadratic_moment_meta = process_raw_weights(
        quadratic_moment_ratio.raw_weights,
        min_weight=weight_config.min_weight,
    )
    quadratic_moment_cv_weights, quadratic_moment_cv_meta = process_raw_weights(
        quadratic_moment_cv.ratio.raw_weights,
        min_weight=weight_config.min_weight,
    )
    linear_cv_cap_weights, linear_cv_cap_meta = _select_cv_cap_weights(
        linear_ratio.raw_weights,
        batch=batch,
        env=env,
        ratio_feature_map=ratio_feature_map,
        target_policy=target_policy,
        gamma=ratio_gamma,
        weight_config=weight_config,
        min_weight=weight_config.min_weight,
        seed=40_000 + seed,
    )
    if neural_ratio is not None:
        neural_clipped_weights, neural_clipped_meta = process_raw_weights(
            neural_ratio.raw_weights,
            min_weight=neural_ratio_config.min_weight,
            clip_quantile=neural_ratio_config.clip_quantile,
            max_weight=neural_ratio_config.max_weight,
            target_ess_fraction=neural_ratio_config.target_ess_fraction,
            max_uniform_mix=neural_ratio_config.max_uniform_mix,
        )
    linear_oracle_tikhonov_weights, linear_oracle_tikhonov_meta, linear_oracle_tikhonov_diag = (
        _select_oracle_tikhonov_ratio_weights(
            batch=batch,
            env=env,
            ratio_feature_map=ratio_feature_map,
            feature_map=feature_map,
            target_policy=target_policy,
            ratio_gamma=ratio_gamma,
            value_gamma=value_gamma,
            base_weight_config=weight_config,
            fqe_solver_config=fqe_solver_config,
            truth=truth,
            evaluation_samples=evaluation_samples,
        )
    )
    quadratic_moment_oracle_weights, quadratic_moment_oracle_meta, quadratic_moment_oracle_diag = (
        _select_oracle_quadratic_moment_weights(
            batch=batch,
            env=env,
            ratio_feature_map=ratio_feature_map,
            feature_map=feature_map,
            target_policy=target_policy,
            ratio_gamma=ratio_gamma,
            value_gamma=value_gamma,
            fqe_solver_config=fqe_solver_config,
            truth=truth,
            evaluation_samples=evaluation_samples,
            seed=85_000 + seed,
        )
    )

    weight_methods: dict[str, tuple[np.ndarray, dict[str, float], dict[str, object]]] = {
        "standard": (
            standard_weights,
            {"fraction_clipped": 0.0, "chosen_uniform_mix": 0.0, "clip_level": np.nan},
            {"solver": "uniform", "moment_violation_l2": np.nan, "normalization_error": 0.0},
        ),
        "oracle_raw": (
            oracle_weights,
            oracle_meta,
            {"solver": "oracle_exact", "moment_violation_l2": 0.0, "normalization_error": 0.0},
        ),
        "oracle_clipped": (
            oracle_clipped_weights,
            oracle_clipped_meta,
            {"solver": "oracle_exact_clipped", "moment_violation_l2": 0.0, "normalization_error": 0.0},
        ),
        "linear_clipped": (
            linear_clipped_weights,
            _stabilization_meta(
                linear_clipped_meta,
                weight_stabilization="quantile_cap_ess_mix",
                cap_selection_rule="q99_cap_plus_ess_mix",
                adaptive_ess_target=0.40,
                fixed_cap=weight_config.clipped_max_weight,
            ),
            linear_ratio.diagnostics,
        ),
        "linear_fixed_cap": (
            linear_fixed_cap_weights,
            _stabilization_meta(
                linear_fixed_cap_meta,
                weight_stabilization="fixed_cap",
                cap_selection_rule="pre_specified",
                fixed_cap=fixed_cap_level,
            ),
            linear_ratio.diagnostics,
        ),
        "linear_ess_winsor": (
            linear_ess_winsor_weights,
            _stabilization_meta(
                linear_ess_winsor_meta,
                weight_stabilization="ess_adaptive_winsor",
                cap_selection_rule="ess_adaptive",
                adaptive_ess_target=adaptive_ess_target,
            ),
            linear_ratio.diagnostics,
        ),
        "linear_tikhonov": (
            linear_tikhonov_weights,
            _stabilization_meta(
                linear_tikhonov_meta,
                weight_stabilization="tikhonov_ratio",
                cap_selection_rule="ratio_moment_cv_min",
                cv_selected_ridge=tikhonov_min.config.ridge_primal,
                cv_selected_ridge_min=tikhonov_min.diagnostics["cv_selected_ridge_min"],
                cv_selected_ridge_one_se=tikhonov_min.diagnostics["cv_selected_ridge_one_se"],
                cv_validation_ratio_moment_risk=tikhonov_min.validation_loss,
                cv_validation_ratio_moment_risk_se=tikhonov_min.validation_loss_se,
                cv_one_se_threshold=tikhonov_min.diagnostics["cv_one_se_threshold"],
                cv_n_folds=int(tikhonov_min.diagnostics["cv_n_folds"]),
                tikhonov_ridge_primal=tikhonov_min.config.ridge_primal,
                tikhonov_ridge_dual=tikhonov_min.config.ridge_dual,
            ),
            {
                **tikhonov_min.ratio.diagnostics,
                **tikhonov_min.diagnostics,
                "solver": "linear_tikhonov_reduced_moment",
            },
        ),
        "linear_tikhonov_xfit": (
            linear_tikhonov_xfit_weights,
            _stabilization_meta(
                linear_tikhonov_xfit_meta,
                weight_stabilization="tikhonov_ratio_crossfit",
                cap_selection_rule="ratio_moment_cv_min",
                cv_selected_ridge=tikhonov_min.config.ridge_primal,
                cv_selected_ridge_min=tikhonov_min.diagnostics["cv_selected_ridge_min"],
                cv_selected_ridge_one_se=tikhonov_min.diagnostics["cv_selected_ridge_one_se"],
                cv_validation_ratio_moment_risk=tikhonov_min.validation_loss,
                cv_validation_ratio_moment_risk_se=tikhonov_min.validation_loss_se,
                cv_one_se_threshold=tikhonov_min.diagnostics["cv_one_se_threshold"],
                cv_n_folds=int(tikhonov_min.diagnostics["cv_n_folds"]),
                tikhonov_ridge_primal=tikhonov_min.config.ridge_primal,
                tikhonov_ridge_dual=tikhonov_min.config.ridge_dual,
            ),
            {
                **tikhonov_min.diagnostics,
                "solver": "linear_tikhonov_crossfit_reduced_moment",
                "moment_violation_l2": tikhonov_min.ratio.diagnostics.get("moment_violation_l2", np.nan),
                "normalization_error": tikhonov_min.ratio.diagnostics.get("normalization_error", np.nan),
            },
        ),
        "linear_tikhonov_1se": (
            linear_tikhonov_1se_weights,
            _stabilization_meta(
                linear_tikhonov_1se_meta,
                weight_stabilization="tikhonov_ratio",
                cap_selection_rule="ratio_moment_cv_one_se",
                cv_selected_ridge=tikhonov_one_se.config.ridge_primal,
                cv_selected_ridge_min=tikhonov_one_se.diagnostics["cv_selected_ridge_min"],
                cv_selected_ridge_one_se=tikhonov_one_se.diagnostics["cv_selected_ridge_one_se"],
                cv_validation_ratio_moment_risk=tikhonov_one_se.validation_loss,
                cv_validation_ratio_moment_risk_se=tikhonov_one_se.validation_loss_se,
                cv_one_se_threshold=tikhonov_one_se.diagnostics["cv_one_se_threshold"],
                cv_n_folds=int(tikhonov_one_se.diagnostics["cv_n_folds"]),
                tikhonov_ridge_primal=tikhonov_one_se.config.ridge_primal,
                tikhonov_ridge_dual=tikhonov_one_se.config.ridge_dual,
            ),
            {
                **tikhonov_one_se.ratio.diagnostics,
                **tikhonov_one_se.diagnostics,
                "solver": "linear_tikhonov_one_se_reduced_moment",
            },
        ),
        "linear_tikhonov_1se_xfit": (
            linear_tikhonov_1se_xfit_weights,
            _stabilization_meta(
                linear_tikhonov_1se_xfit_meta,
                weight_stabilization="tikhonov_ratio_crossfit",
                cap_selection_rule="ratio_moment_cv_one_se",
                cv_selected_ridge=tikhonov_one_se.config.ridge_primal,
                cv_selected_ridge_min=tikhonov_one_se.diagnostics["cv_selected_ridge_min"],
                cv_selected_ridge_one_se=tikhonov_one_se.diagnostics["cv_selected_ridge_one_se"],
                cv_validation_ratio_moment_risk=tikhonov_one_se.validation_loss,
                cv_validation_ratio_moment_risk_se=tikhonov_one_se.validation_loss_se,
                cv_one_se_threshold=tikhonov_one_se.diagnostics["cv_one_se_threshold"],
                cv_n_folds=int(tikhonov_one_se.diagnostics["cv_n_folds"]),
                tikhonov_ridge_primal=tikhonov_one_se.config.ridge_primal,
                tikhonov_ridge_dual=tikhonov_one_se.config.ridge_dual,
            ),
            {
                **tikhonov_one_se.diagnostics,
                "solver": "linear_tikhonov_one_se_crossfit_reduced_moment",
                "moment_violation_l2": tikhonov_one_se.ratio.diagnostics.get("moment_violation_l2", np.nan),
                "normalization_error": tikhonov_one_se.ratio.diagnostics.get("normalization_error", np.nan),
            },
        ),
        "linear_unregularized": (
            linear_unregularized_weights,
            _stabilization_meta(
                linear_unregularized_meta,
                weight_stabilization="none",
                cap_selection_rule="none",
                tikhonov_ridge_primal=0.0,
                tikhonov_ridge_dual=0.0,
            ),
            {
                **linear_unregularized_ratio.diagnostics,
                "solver": "linear_unregularized_reduced_moment",
            },
        ),
        "linear_oracle_tikhonov": (
            linear_oracle_tikhonov_weights,
            linear_oracle_tikhonov_meta,
            linear_oracle_tikhonov_diag,
        ),
        "quadratic_moment": (
            quadratic_moment_weights,
            _stabilization_meta(
                quadratic_moment_meta,
                weight_stabilization="exponential_quadratic_moment",
                cap_selection_rule="held_out_moment_early_stopping",
            ),
            {
                **quadratic_moment_ratio.diagnostics,
                "solver": "exponential_quadratic_moment",
            },
        ),
        "quadratic_moment_cv": (
            quadratic_moment_cv_weights,
            _stabilization_meta(
                quadratic_moment_cv_meta,
                weight_stabilization="exponential_quadratic_moment_cv_reg",
                cap_selection_rule="held_out_moment_risk",
                cv_selected_ridge=quadratic_moment_cv.ridge,
                cv_selected_ridge_min=quadratic_moment_cv.diagnostics["cv_selected_ridge_min"],
                cv_selected_ridge_one_se=quadratic_moment_cv.diagnostics["cv_selected_ridge_one_se"],
                cv_validation_ratio_moment_risk=quadratic_moment_cv.validation_loss,
                cv_validation_ratio_moment_risk_se=quadratic_moment_cv.validation_loss_se,
                cv_one_se_threshold=quadratic_moment_cv.diagnostics["cv_one_se_threshold"],
                cv_n_folds=int(quadratic_moment_cv.diagnostics["cv_n_folds"]),
                tikhonov_ridge_primal=quadratic_moment_cv.ridge,
                tikhonov_ridge_dual=np.nan,
            ),
            {
                **quadratic_moment_cv.ratio.diagnostics,
                **quadratic_moment_cv.diagnostics,
                "solver": "exponential_quadratic_moment_cv_reg",
            },
        ),
        "quadratic_moment_oracle": (
            quadratic_moment_oracle_weights,
            quadratic_moment_oracle_meta,
            quadratic_moment_oracle_diag,
        ),
        "linear_cv_cap": (
            linear_cv_cap_weights,
            linear_cv_cap_meta,
            linear_ratio.diagnostics,
        ),
    }
    for eta, (eta_ratio, eta_config, eta_diag) in normalized_ratio_tikhonov.items():
        eta_weights, eta_meta = normalized_ratio_tikhonov_weights[eta]
        weight_methods[f"linear_fixed_tikhonov_eta_{eta:g}"] = (
            eta_weights,
            _stabilization_meta(
                eta_meta,
                weight_stabilization="tikhonov_ratio",
                cap_selection_rule="pre_specified_normalized_eta",
                cv_selected_ridge=eta_config.ridge_primal,
                tikhonov_ridge_primal=eta_config.ridge_primal,
                tikhonov_ridge_dual=eta_config.ridge_dual,
                normalized_tikhonov_eta=eta,
                normalized_tikhonov_primal_scale=eta_diag["normalized_tikhonov_primal_scale"],
                normalized_tikhonov_dual_scale=eta_diag["normalized_tikhonov_dual_scale"],
            ),
            {
                **eta_ratio.diagnostics,
                "solver": "linear_normalized_tikhonov_reduced_moment",
            },
        )
    if neural_ratio is not None:
        weight_methods["neural_clipped"] = (
            neural_clipped_weights,
            neural_clipped_meta,
            neural_ratio.diagnostics,
        )
    estimator_specs = [
        ("linear_standard_fqe", "linear", "standard"),
        ("linear_oracle_raw_fqe", "linear", "oracle_raw"),
        ("linear_oracle_clipped_fqe", "linear", "oracle_clipped"),
        ("linear_estimated_clipped_fqe", "linear", "linear_clipped"),
        ("linear_estimated_fixed_cap_fqe", "linear", "linear_fixed_cap"),
        ("linear_estimated_ess_winsor_fqe", "linear", "linear_ess_winsor"),
        ("linear_estimated_unregularized_fqe", "linear", "linear_unregularized"),
        ("linear_estimated_rbf_oracle_tikhonov_fqe", "linear", "linear_oracle_tikhonov"),
        ("linear_estimated_quadratic_moment_fqe", "linear", "quadratic_moment"),
        ("linear_estimated_quadratic_moment_cv_fqe", "linear", "quadratic_moment_cv"),
        ("linear_estimated_quadratic_moment_oracle_fqe", "linear", "quadratic_moment_oracle"),
        ("linear_estimated_tikhonov_fqe", "linear", "linear_tikhonov"),
        ("linear_minimax_q_unregularized", "linear", "standard"),
        ("linear_minimax_q_cv_tikhonov", "linear", "standard"),
        ("linear_minimax_q_rbf", "linear", "standard"),
        ("linear_minimax_q_oracle_tikhonov", "linear", "standard"),
        ("linear_estimated_tikhonov_xfit_fqe", "linear", "linear_tikhonov_xfit"),
        ("linear_estimated_tikhonov_1se_fqe", "linear", "linear_tikhonov_1se"),
        ("linear_estimated_tikhonov_1se_xfit_fqe", "linear", "linear_tikhonov_1se_xfit"),
        ("linear_estimated_cv_cap_fqe", "linear", "linear_cv_cap"),
    ]
    for eta in FIXED_TIKHONOV_ETA_GRID:
        if abs(float(eta) - FIXED_TIKHONOV_MAIN_ETA) < 1e-15:
            estimator_specs.append(
                (
                    "linear_estimated_rbf_fixed_tikhonov_fqe",
                    "linear",
                    f"linear_fixed_tikhonov_eta_{eta:g}",
                )
            )
        else:
            estimator_specs.append(
                (
                    f"linear_estimated_rbf_fixed_tikhonov_eta_{eta:g}_fqe",
                    "linear",
                    f"linear_fixed_tikhonov_eta_{eta:g}",
                )
            )
    minimax_eta_by_estimator: dict[str, float] = {
        "linear_minimax_q_rbf": FIXED_TIKHONOV_MAIN_ETA,
    }
    for eta in FIXED_TIKHONOV_ETA_GRID:
        if abs(float(eta) - FIXED_TIKHONOV_MAIN_ETA) >= 1e-15:
            estimator_name = f"linear_minimax_q_eta_{eta:g}"
            minimax_eta_by_estimator[estimator_name] = float(eta)
            estimator_specs.append((estimator_name, "linear", "standard"))
    if neural_ratio is not None:
        estimator_specs.append(("linear_neural_weighted_clipped_fqe", "linear", "neural_clipped"))
    if include_neural_fqe:
        estimator_specs.extend(
            [
                ("neural_standard_fqe", "neural", "standard"),
                ("neural_oracle_clipped_fqe", "neural", "oracle_clipped"),
                ("neural_estimated_clipped_fqe", "neural", "linear_clipped"),
            ]
        )
        if neural_ratio is not None:
            estimator_specs.append(("neural_neural_weighted_clipped_fqe", "neural", "neural_clipped"))

    metrics_by_estimator: dict[str, EstimatorMetrics] = {}
    for estimator_name, fqe_family, weight_key in estimator_specs:
        sample_weights, weight_meta, source_diag = weight_methods[weight_key]
        clip_level = float(weight_meta.get("clip_level", np.nan))
        diagnostics = _gamma_weight_stats_with_diagnostics(
            sample_weights,
            oracle_reference_weights=oracle_weights,
            design_features=design_features,
            fqe_solver_config=fqe_solver_config,
            source_diagnostics=source_diag,
            ratio_feature_map=ratio_feature_map,
            feature_map=feature_map,
            batch_states=batch.states,
            batch_actions=batch.actions,
            target_sa=evaluation_samples.target_sa,
            fraction_clipped=float(weight_meta.get("fraction_clipped", 0.0)),
            chosen_uniform_mix=float(weight_meta.get("chosen_uniform_mix", 0.0)),
            clip_level=None if np.isnan(clip_level) else clip_level,
            extra_diagnostics=weight_meta,
        )
        if estimator_name in {
            "linear_minimax_q_rbf",
            "linear_minimax_q_unregularized",
            "linear_minimax_q_cv_tikhonov",
            "linear_minimax_q_oracle_tikhonov",
        } or estimator_name in minimax_eta_by_estimator:
            normalized_minimax_diag: dict[str, float] = {}
            if estimator_name == "linear_minimax_q_oracle_tikhonov":
                fit = _select_oracle_minimax_q(
                    batch=batch,
                    feature_map=feature_map,
                    critic_feature_map=ratio_feature_map,
                    target_policy=target_policy,
                    gamma=value_gamma,
                    truth=truth,
                    evaluation_samples=evaluation_samples,
                )
            elif estimator_name == "linear_minimax_q_cv_tikhonov":
                fit = _select_cv_normalized_minimax_q(
                    batch=batch,
                    feature_map=feature_map,
                    critic_feature_map=ratio_feature_map,
                    target_policy=target_policy,
                    gamma=value_gamma,
                    seed=90_000 + seed,
                )
                normalized_minimax_diag = {
                    "normalized_tikhonov_eta": float(
                        fit.diagnostics.get("normalized_tikhonov_eta", np.nan)
                    ),
                    "normalized_tikhonov_primal_scale": float(
                        fit.diagnostics.get("normalized_tikhonov_primal_scale", np.nan)
                    ),
                    "normalized_tikhonov_dual_scale": float(
                        fit.diagnostics.get("normalized_tikhonov_dual_scale", np.nan)
                    ),
                }
            else:
                if estimator_name == "linear_minimax_q_unregularized":
                    q_ridge = 0.0
                    critic_ridge = 0.0
                else:
                    q_ridge, critic_ridge, normalized_minimax_diag = _normalized_minimax_ridges(
                        batch=batch,
                        feature_map=feature_map,
                        critic_feature_map=ratio_feature_map,
                        target_policy=target_policy,
                        gamma=value_gamma,
                        eta=minimax_eta_by_estimator.get(estimator_name, FIXED_TIKHONOV_MAIN_ETA),
                    )
                fit = fit_minimax_linear_q(
                    batch,
                    feature_map=feature_map,
                    critic_feature_map=ratio_feature_map,
                    target_policy=target_policy,
                    gamma=value_gamma,
                    q_ridge=q_ridge,
                    critic_ridge=critic_ridge,
                )
            q_function = fit.q_function
            diagnostics.update(
                {
                    "ratio_solver": str(fit.diagnostics.get("solver", "minimax_q_rbf_critic")),
                    "ratio_moment_violation_l2": float(
                        fit.diagnostics.get("moment_violation_l2", np.nan)
                    ),
                    "ratio_normalization_error": 0.0,
                    "weight_stabilization": "not_weighted",
                    "cap_selection_rule": "none"
                    if estimator_name == "linear_minimax_q_unregularized"
                    else (
                        "oracle_target_q_mse"
                        if estimator_name == "linear_minimax_q_oracle_tikhonov"
                        else (
                            "bellman_moment_cv_normalized_eta"
                            if estimator_name == "linear_minimax_q_cv_tikhonov"
                            else "pre_specified_normalized_eta"
                        )
                    ),
                    "cv_selected_ridge": float(fit.diagnostics.get("cv_selected_ridge", np.nan)),
                    "cv_selected_ridge_min": float(fit.diagnostics.get("cv_selected_ridge_min", np.nan)),
                    "cv_selected_ridge_one_se": np.nan,
                    "cv_validation_bellman_mse": float(
                        fit.diagnostics.get("cv_validation_bellman_mse", np.nan)
                    ),
                    "cv_validation_ratio_moment_risk": float(
                        fit.diagnostics.get("critic_norm_moment_risk", np.nan)
                    ),
                    "cv_validation_ratio_moment_risk_se": np.nan,
                    "cv_n_folds": np.nan,
                    "tikhonov_ridge_primal": float(fit.diagnostics.get("q_ridge", np.nan)),
                    "tikhonov_ridge_dual": float(fit.diagnostics.get("critic_ridge", np.nan)),
                    "normalized_tikhonov_eta": float(
                        normalized_minimax_diag.get("normalized_tikhonov_eta", np.nan)
                    ),
                    "normalized_tikhonov_primal_scale": float(
                        normalized_minimax_diag.get("normalized_tikhonov_primal_scale", np.nan)
                    ),
                    "normalized_tikhonov_dual_scale": float(
                        normalized_minimax_diag.get("normalized_tikhonov_dual_scale", np.nan)
                    ),
                    "weighted_design_condition_number": float(
                        fit.diagnostics.get("primal_condition_number", np.nan)
                    ),
                }
            )
        elif fqe_family == "linear":
            fit = fit_linear_fqe(
                batch=batch,
                feature_map=feature_map,
                target_policy=target_policy,
                gamma=value_gamma,
                solver_config=fqe_solver_config,
                sample_weights=sample_weights,
                seed=seed,
            )
            q_function = fit.q_function
        else:
            fit = fit_neural_fqe(
                batch=batch,
                env=env,
                target_policy=target_policy,
                value_gamma=value_gamma,
                sample_weights=sample_weights,
                config=neural_fqe_config,
                seed=30_000 + seed,
            )
            q_function = fit.q_function
        metrics = evaluate_estimator(
            estimator=estimator_name,
            q_function=q_function,
            truth=truth,
            env=env,
            target_policy=target_policy,
            gamma=value_gamma,
            evaluation_config=evaluation_config,
            target_joint_mixture=target_joint,
            behavior_joint_mixture=behavior_joint,
            target_state_mixture=target_state,
            behavior_state_mixture=behavior_state,
            rng=np.random.default_rng(20_000 + seed),
            weight_stats=diagnostics,
            evaluation_samples=evaluation_samples,
        )
        metrics_by_estimator[estimator_name] = metrics
        unstable_flag, unstable_reason = _classify_weighted_run(
            estimator=estimator_name,
            metrics=metrics,
            all_metrics=metrics_by_estimator,
            weight_config=weight_config,
        )
        metrics.weight_stats["unstable_run_flag"] = unstable_flag
        metrics.weight_stats["unstable_reason"] = unstable_reason
        output_stage_rows.append(
            _gamma_metric_row(
                stage=stage,
                value_gamma=value_gamma,
                ratio_gamma=ratio_gamma,
                reference_distribution=reference_kind,
                data_mode=data_mode,
                shift=shift,
                sample_size=sample_size,
                process_noise_sd=process_noise_sd,
                behavior_action_sd=behavior_action_sd,
                feature_regime=feature_regime,
                seed=seed,
                fqe_family=fqe_family,
                ratio_estimator=weight_key,
                metrics=metrics,
            )
        )


def _select_family(summary_rows: list[dict[str, object]], weight_config: WeightEstimatorConfig) -> SelectedFamily:
    families = sorted(
        {
            (
                float(row["gamma"]),
                float(row["process_noise_sd"]),
                float(row["behavior_action_sd"]),
            )
            for row in summary_rows
            if row["feature_regime"] == "misspecified_affine" and int(row["sample_size"]) == 4000
        }
    )
    best_selection: SelectedFamily | None = None
    best_score = -np.inf
    for gamma, process_noise_sd, behavior_action_sd in families:
        shifts = [0.0, 0.5, 1.0, 1.1, 1.35]
        low_candidates = []
        moderate_candidates = []
        severe_candidates = []
        for shift in shifts:
            uq = _mean_metric(
                summary_rows,
                gamma=gamma,
                process_noise_sd=process_noise_sd,
                behavior_action_sd=behavior_action_sd,
                shift=shift,
                sample_size=4000,
                feature_regime="misspecified_affine",
                estimator="standard_fqe",
                metric="target_q_mse",
            )
            oq = _mean_metric(
                summary_rows,
                gamma=gamma,
                process_noise_sd=process_noise_sd,
                behavior_action_sd=behavior_action_sd,
                shift=shift,
                sample_size=4000,
                feature_regime="misspecified_affine",
                estimator="oracle_weighted_fqe",
                metric="target_q_mse",
            )
            uv = _mean_metric(
                summary_rows,
                gamma=gamma,
                process_noise_sd=process_noise_sd,
                behavior_action_sd=behavior_action_sd,
                shift=shift,
                sample_size=4000,
                feature_regime="misspecified_affine",
                estimator="standard_fqe",
                metric="policy_value_mse",
            )
            ov = _mean_metric(
                summary_rows,
                gamma=gamma,
                process_noise_sd=process_noise_sd,
                behavior_action_sd=behavior_action_sd,
                shift=shift,
                sample_size=4000,
                feature_regime="misspecified_affine",
                estimator="oracle_weighted_fqe",
                metric="policy_value_mse",
            )
            ocv = _mean_metric(
                summary_rows,
                gamma=gamma,
                process_noise_sd=process_noise_sd,
                behavior_action_sd=behavior_action_sd,
                shift=shift,
                sample_size=4000,
                feature_regime="misspecified_affine",
                estimator="oracle_weighted_fqe_clipped",
                metric="policy_value_mse",
            )
            cq = _mean_metric(
                summary_rows,
                gamma=gamma,
                process_noise_sd=process_noise_sd,
                behavior_action_sd=behavior_action_sd,
                shift=shift,
                sample_size=4000,
                feature_regime="misspecified_affine",
                estimator="estimated_weighted_fqe_clipped",
                metric="target_q_mse",
            )
            cv = _mean_metric(
                summary_rows,
                gamma=gamma,
                process_noise_sd=process_noise_sd,
                behavior_action_sd=behavior_action_sd,
                shift=shift,
                sample_size=4000,
                feature_regime="misspecified_affine",
                estimator="estimated_weighted_fqe_clipped",
                metric="policy_value_mse",
            )
            oracle_ess = _mean_metric(
                summary_rows,
                gamma=gamma,
                process_noise_sd=process_noise_sd,
                behavior_action_sd=behavior_action_sd,
                shift=shift,
                sample_size=4000,
                feature_regime="misspecified_affine",
                estimator="oracle_weighted_fqe",
                metric="effective_sample_size_fraction",
            )
            oracle_q99 = _mean_metric(
                summary_rows,
                gamma=gamma,
                process_noise_sd=process_noise_sd,
                behavior_action_sd=behavior_action_sd,
                shift=shift,
                sample_size=4000,
                feature_regime="misspecified_affine",
                estimator="oracle_weighted_fqe",
                metric="weight_q99",
            )
            oracle_max = _mean_metric(
                summary_rows,
                gamma=gamma,
                process_noise_sd=process_noise_sd,
                behavior_action_sd=behavior_action_sd,
                shift=shift,
                sample_size=4000,
                feature_regime="misspecified_affine",
                estimator="oracle_weighted_fqe",
                metric="weight_max",
            )
            raw_ess = _mean_metric(
                summary_rows,
                gamma=gamma,
                process_noise_sd=process_noise_sd,
                behavior_action_sd=behavior_action_sd,
                shift=shift,
                sample_size=4000,
                feature_regime="misspecified_affine",
                estimator="estimated_weighted_fqe",
                metric="effective_sample_size_fraction",
            )
            raw_q99 = _mean_metric(
                summary_rows,
                gamma=gamma,
                process_noise_sd=process_noise_sd,
                behavior_action_sd=behavior_action_sd,
                shift=shift,
                sample_size=4000,
                feature_regime="misspecified_affine",
                estimator="estimated_weighted_fqe",
                metric="weight_q99",
            )
            clipped_ess = _mean_metric(
                summary_rows,
                gamma=gamma,
                process_noise_sd=process_noise_sd,
                behavior_action_sd=behavior_action_sd,
                shift=shift,
                sample_size=4000,
                feature_regime="misspecified_affine",
                estimator="estimated_weighted_fqe_clipped",
                metric="effective_sample_size_fraction",
            )
            clipped_q99 = _mean_metric(
                summary_rows,
                gamma=gamma,
                process_noise_sd=process_noise_sd,
                behavior_action_sd=behavior_action_sd,
                shift=shift,
                sample_size=4000,
                feature_regime="misspecified_affine",
                estimator="estimated_weighted_fqe_clipped",
                metric="weight_q99",
            )
            oracle_improvement_q = max(uq - oq, 0.0) / max(uq, 1e-12)
            oracle_improvement_v = max(uv - ov, 0.0) / max(uv, 1e-12)
            robust_oracle_improvement_v = max(uv - min(ov, ocv), 0.0) / max(uv, 1e-12)
            estimated_improvement_v = max(uv - cv, 0.0) / max(uv, 1e-12)
            clipped_fraction = max(uq - cq, 0.0) / max(uq - oq, 1e-12)
            oracle_overlap_stress = (
                oracle_ess < 0.15
                or oracle_q99 > weight_config.severe_q99_threshold
                or oracle_max > 5.0 * weight_config.severe_q99_threshold
            )
            if abs(oq - uq) / max(uq, 1e-12) <= 0.05 and abs(ov - uv) / max(uv, 1e-12) <= 0.05:
                low_candidates.append(shift)
            if (
                oracle_improvement_q >= 0.20
                and robust_oracle_improvement_v >= 0.15
                and clipped_fraction >= 0.30
                and estimated_improvement_v >= 0.10
                and clipped_ess >= 0.15
            ):
                score = (
                    oracle_improvement_q
                    + 0.5 * robust_oracle_improvement_v
                    + 0.25 * estimated_improvement_v
                    - 0.25 * float(oracle_overlap_stress)
                )
                moderate_candidates.append((score, shift))
            if (
                raw_ess < 0.20
                or raw_q99 > weight_config.severe_q99_threshold
                or oracle_overlap_stress
            ) and (clipped_ess > raw_ess or clipped_q99 < raw_q99):
                severe_candidates.append((raw_q99 - clipped_q99 + clipped_ess - raw_ess, shift))
        if low_candidates and moderate_candidates and severe_candidates:
            low_shift = min(low_candidates)
            for moderate_score, moderate_shift in moderate_candidates:
                valid_severe = [
                    (severe_score, severe_shift)
                    for severe_score, severe_shift in severe_candidates
                    if severe_shift > moderate_shift
                ]
                if not valid_severe:
                    continue
                severe_score, severe_shift = max(valid_severe)
                score = moderate_score + severe_score
                if score > best_score:
                    best_score = score
                    best_selection = SelectedFamily(
                        gamma=gamma,
                        process_noise_sd=process_noise_sd,
                        behavior_action_sd=behavior_action_sd,
                        low_shift=low_shift,
                        moderate_shift=moderate_shift,
                        severe_shift=severe_shift,
                        status="qualified",
                        note=(
                            "Satisfied ordered low/moderate/severe criteria on the misspecified-affine n=4000 slice; "
                            "moderate value gains may use clipped oracle diagnostics when raw oracle WLS is overlap-stressed."
                        ),
                    )
    if best_selection is not None:
        return best_selection

    fallback_family = families[0]
    return SelectedFamily(
        gamma=fallback_family[0],
        process_noise_sd=fallback_family[1],
        behavior_action_sd=fallback_family[2],
        low_shift=0.0,
        moderate_shift=1.0,
        severe_shift=1.35,
        status="fallback",
        note="No exploratory family satisfied every criterion; defaulted to the first family for transparent reruns.",
    )


def _write_selected_family(path: Path, selection: SelectedFamily) -> None:
    _write_csv(
        path,
        [
            {
                "gamma": selection.gamma,
                "process_noise_sd": selection.process_noise_sd,
                "behavior_action_sd": selection.behavior_action_sd,
                "low_shift": selection.low_shift,
                "moderate_shift": selection.moderate_shift,
                "severe_shift": selection.severe_shift,
                "status": selection.status,
                "note": selection.note,
            }
        ],
        fieldnames=[
            "gamma",
            "process_noise_sd",
            "behavior_action_sd",
            "low_shift",
            "moderate_shift",
            "severe_shift",
            "status",
            "note",
        ],
    )


def _load_selected_family(path: Path) -> SelectedFamily:
    row = _read_csv(path)[0]
    return SelectedFamily(
        gamma=float(row["gamma"]),
        process_noise_sd=float(row["process_noise_sd"]),
        behavior_action_sd=float(row["behavior_action_sd"]),
        low_shift=float(row["low_shift"]),
        moderate_shift=float(row["moderate_shift"]),
        severe_shift=float(row["severe_shift"]),
        status=row["status"],
        note=row["note"],
    )


def _fmt(value: object, digits: int = 3) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(numeric):
        return "nan"
    if abs(numeric) >= 1_000 or (abs(numeric) < 0.01 and numeric != 0.0):
        return f"{numeric:.{digits}e}"
    return f"{numeric:.{digits}f}"


def _selected_summary_row(
    summary_rows: list[dict[str, object]],
    *,
    selection: SelectedFamily,
    shift: float,
    estimator: str,
) -> dict[str, object] | None:
    for row in summary_rows:
        if (
            float(row["gamma"]) == selection.gamma
            and float(row["process_noise_sd"]) == selection.process_noise_sd
            and float(row["behavior_action_sd"]) == selection.behavior_action_sd
            and float(row["shift"]) == shift
            and int(row["sample_size"]) == 4000
            and row["feature_regime"] == "misspecified_affine"
            and row["estimator"] == estimator
        ):
            return row
    return None


def _write_design_search_report(
    path: Path,
    selection: SelectedFamily,
    summary_rows: list[dict[str, object]],
    weight_config: WeightEstimatorConfig,
) -> None:
    lines = [
        "# Design Search Report",
        "",
        "## Benchmark Variants Tried",
        "",
        "- Shift grid: `{0.0, 0.5, 1.0, 1.1, 1.35}`",
        "- Sample sizes: `{1000, 4000}`",
        "- Discount factors: `{0.95, 0.99}`",
        "- Process noise SDs: `{0.05, 0.12}`",
        "- Behavior action SDs: `{0.10, 0.20}`",
        "- Feature regimes: `{well_specified, misspecified_affine}`",
        "- Seeds per exploratory cell: `3`",
        "",
        "## Frozen Final Configuration",
        "",
        f"- gamma: `{selection.gamma}`",
        f"- process_noise_sd: `{selection.process_noise_sd}`",
        f"- behavior_action_sd: `{selection.behavior_action_sd}`",
        f"- low shift: `{selection.low_shift}`",
        f"- moderate shift: `{selection.moderate_shift}`",
        f"- severe shift: `{selection.severe_shift}`",
        f"- selection status: `{selection.status}`",
        f"- note: {selection.note}",
        "",
        "## Selected Regime Diagnostics",
        "",
    ]
    lines.extend(
        [
            "| regime | shift | estimator | target-Q MSE | behavior-Q MSE | value MSE | value bias | ESS frac | q99 | max w | log-ratio RMSE | unstable |",
            "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for regime, shift in [
        ("low", selection.low_shift),
        ("moderate", selection.moderate_shift),
        ("severe", selection.severe_shift),
    ]:
        for estimator in [
            "standard_fqe",
            "oracle_weighted_fqe",
            "oracle_weighted_fqe_clipped",
            "estimated_weighted_fqe",
            "estimated_weighted_fqe_clipped",
            "estimated_weighted_fqe_clip95",
            "estimated_weighted_fqe_clip99_ess40",
        ]:
            selected_row = _selected_summary_row(
                summary_rows,
                selection=selection,
                shift=shift,
                estimator=estimator,
            )
            if selected_row is None:
                continue
            lines.append(
                "| "
                + " | ".join(
                    [
                        regime,
                        _fmt(shift),
                        estimator,
                        _fmt(selected_row["target_q_mse"]),
                        _fmt(selected_row["behavior_q_mse"]),
                        _fmt(selected_row["policy_value_mse"]),
                        _fmt(selected_row["policy_value_bias"]),
                        _fmt(selected_row["effective_sample_size_fraction"]),
                        _fmt(selected_row["weight_q99"]),
                        _fmt(selected_row["weight_max"]),
                        _fmt(selected_row["oracle_log_ratio_rmse"]),
                        str(selected_row["unstable_reason"]),
                    ]
                )
                + " |"
            )
    lines.extend(["", "## Oracle Weighting Helped Where", ""])
    oracle_wins = []
    estimated_failures = []
    for row in summary_rows:
        if row["feature_regime"] != "misspecified_affine" or int(row["sample_size"]) != 4000:
            continue
        if row["estimator"] != "oracle_weighted_fqe":
            continue
        uq = _mean_metric(
            summary_rows,
            gamma=float(row["gamma"]),
            process_noise_sd=float(row["process_noise_sd"]),
            behavior_action_sd=float(row["behavior_action_sd"]),
            shift=float(row["shift"]),
            sample_size=4000,
            feature_regime="misspecified_affine",
            estimator="standard_fqe",
            metric="target_q_mse",
        )
        cq = _mean_metric(
            summary_rows,
            gamma=float(row["gamma"]),
            process_noise_sd=float(row["process_noise_sd"]),
            behavior_action_sd=float(row["behavior_action_sd"]),
            shift=float(row["shift"]),
            sample_size=4000,
            feature_regime="misspecified_affine",
            estimator="estimated_weighted_fqe_clipped",
            metric="target_q_mse",
        )
        raw_ess = _mean_metric(
            summary_rows,
            gamma=float(row["gamma"]),
            process_noise_sd=float(row["process_noise_sd"]),
            behavior_action_sd=float(row["behavior_action_sd"]),
            shift=float(row["shift"]),
            sample_size=4000,
            feature_regime="misspecified_affine",
            estimator="estimated_weighted_fqe",
            metric="effective_sample_size_fraction",
        )
        oracle_gain = max(uq - float(row["target_q_mse"]), 0.0) / max(uq, 1e-12)
        clipped_fraction = max(uq - cq, 0.0) / max(uq - float(row["target_q_mse"]), 1e-12)
        text = (
            f"- gamma={row['gamma']}, process_noise_sd={row['process_noise_sd']}, "
            f"behavior_action_sd={row['behavior_action_sd']}, shift={row['shift']}: "
            f"oracle target-Q gain={oracle_gain:.3f}, clipped recovery fraction={clipped_fraction:.3f}, "
            f"estimated raw ESS fraction={raw_ess:.3f}"
        )
        if oracle_gain >= 0.20:
            oracle_wins.append(text)
        if oracle_gain >= 0.20 and (clipped_fraction < 0.30 or raw_ess < 0.15):
            estimated_failures.append(text)
    if oracle_wins:
        lines.extend(oracle_wins[:8])
    else:
        lines.append("- No exploratory cell reached the planned oracle-improvement threshold.")
    lines.extend(
        [
            "",
            "## Estimated Weighting Failed Where",
            "",
        ]
    )
    if estimated_failures:
        lines.extend(estimated_failures[:8])
    else:
        lines.append("- No major estimated-weight failure regime was detected beyond the planned severe-overlap checks.")
    lines.extend(
        [
            "",
            "## Oracle Instability Diagnosis",
            "",
            "- Oracle ratios are exact mixture-density ratios for `d_{pi,gamma} / d_{b,gamma}`; when raw oracle FQE is unstable, the diagnostics point to overlap or weighted least-squares geometry rather than ratio-estimation error.",
            "- The moderate selected regime intentionally reports both raw and clipped oracle rows. In the frozen family, raw oracle has excellent target-Q error but its value estimate is finite-sample/overlap sensitive; clipped oracle stabilizes the value estimate.",
            "- The severe selected regime keeps the overlap-collapse case visible: ESS fractions and upper weight quantiles show why raw weighting is unsafe and why clipped/ESS-regularized variants become bias-variance compromises.",
            "- Estimated weighting is not tuned per shift. When it fails to match oracle, the ratio-quality columns (`oracle_log_ratio_rmse`, correlation, and relative errors in the CSVs) expose the practical bottleneck.",
            "",
            "## Why This Is Scientifically Fair",
            "",
            "- The exploratory grid varies environment overlap, horizon, sample size, and feature misspecification rather than tuning one cell to make weighting win.",
            "- FQE and ratio-estimation hyperparameters are fixed across all cells; only the environment and sample size vary.",
            "- The final study keeps low-shift, moderate-shift, and severe-shift regimes, including the overlap-failure case instead of hiding it.",
            f"- Severe-overlap classification uses a fixed q99 threshold of `{weight_config.severe_q99_threshold}` together with ESS diagnostics.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_gamma_sweep_report(path: Path, summary_rows: list[dict[str, object]], stage: str) -> None:
    main_rows = [
        row
        for row in summary_rows
        if row["feature_regime"] == "misspecified_affine"
        and int(row["sample_size"]) == 4000
        and row["data_mode"] == "matched_iid"
        and row["fqe_family"] == "linear"
        and row["estimator"]
        in {
            "linear_standard_fqe",
            "linear_oracle_clipped_fqe",
            "linear_estimated_clipped_fqe",
            "linear_neural_weighted_clipped_fqe",
        }
    ]
    if not main_rows:
        main_rows = [
            row
            for row in summary_rows
            if row["feature_regime"] == "misspecified_affine"
            and row["data_mode"] == "matched_iid"
            and row["fqe_family"] == "linear"
        ]
    lines = [
        "# Gamma-Sweep Weighting Report",
        "",
        f"- source stage: `{stage}`",
        "- `value_gamma` is the FQE Bellman/value discount.",
        "- `ratio_gamma` is the weighting target; `ratio_gamma=1.0` is stationary weighting, not undiscounted FQE.",
        "- The clean mechanism benchmark samples from the behavior reference distribution matched to `ratio_gamma`.",
        "",
        "## Main Linear-FQE Slice",
        "",
        "| ratio_gamma | shift | estimator | target-ref Q MSE | value MSE | ESS frac | q99 | max w | log-ratio RMSE | calibration | unstable |",
        "|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in sorted(main_rows, key=lambda r: (float(r["ratio_gamma"]), float(r["shift"]), str(r["estimator"]))):
        lines.append(
            "| "
            + " | ".join(
                [
                    _fmt(row["ratio_gamma"]),
                    _fmt(row["shift"]),
                    str(row["estimator"]),
                    _fmt(row["target_q_mse"]),
                    _fmt(row["policy_value_mse"]),
                    _fmt(row["effective_sample_size_fraction"]),
                    _fmt(row["weight_q99"]),
                    _fmt(row["weight_max"]),
                    _fmt(row["oracle_log_ratio_rmse"]),
                    _fmt(row["target_ratio_feature_calibration_l2"]),
                    str(row["unstable_reason"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Guardrails",
            "",
            "- If stationary weighting improves less than discounted weighting, report that as a regularization tradeoff rather than a failure of the method.",
            "- If neural weights have better calibration but worse value error, diagnose WLS/FQE instability and ESS rather than claiming ratio accuracy alone is sufficient.",
            "- If `ratio_gamma=1.0` has very low ESS or high q99/max weights, frame it as stationary-overlap stress.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _run_gamma_stage(output_root: Path, *, stage: str) -> tuple[Path, Path]:
    if stage == "gamma_smoke":
        grid = gamma_smoke_stage_grid()
    elif stage == "gamma_design":
        grid = gamma_design_stage_grid()
    elif stage == "gamma_final":
        grid = gamma_final_stage_grid()
    elif stage == "stationary_shift_paper":
        grid = stationary_shift_paper_stage_grid()
    else:
        grid = gamma_paper_stage_grid()
    rows: list[dict[str, object]] = []
    ratio_feature_config = RatioFeatureConfig()
    fqe_solver_config = FQESolverConfig()
    weight_config = WeightEstimatorConfig()
    neural_ratio_config = _stage_neural_ratio_config(stage)
    neural_fqe_config = _stage_neural_fqe_config(stage)
    evaluation_config = EvaluationConfig()
    completed = 0
    total = (
        len(grid["value_gammas"])
        * len(grid["ratio_gammas"])
        * len(grid["shifts"])
        * len(grid["sample_sizes"])
        * len(grid["process_noise_sds"])
        * len(grid.get("target_action_sds", [None]))
        * len(grid["behavior_action_sds"])
        * len(grid["feature_regimes"])
        * len(grid["seeds"])
    )
    neural_seeds = {int(seed) for seed in grid["neural_seeds"]}
    for value_gamma in grid["value_gammas"]:
        for ratio_gamma in grid["ratio_gammas"]:
            for shift in grid["shifts"]:
                for sample_size in grid["sample_sizes"]:
                    for process_noise_sd in grid["process_noise_sds"]:
                        for target_action_sd in grid.get("target_action_sds", [None]):
                            for behavior_action_sd in grid["behavior_action_sds"]:
                                for feature_regime in grid["feature_regimes"]:
                                    for seed in grid["seeds"]:
                                        _run_single_gamma_configuration(
                                            stage=stage,
                                            output_stage_rows=rows,
                                            value_gamma=float(value_gamma),
                                            ratio_gamma=float(ratio_gamma),
                                            shift=float(shift),
                                            sample_size=int(sample_size),
                                            process_noise_sd=float(process_noise_sd),
                                            target_action_sd=None
                                            if target_action_sd is None
                                            else float(target_action_sd),
                                            behavior_action_sd=float(behavior_action_sd),
                                            feature_regime=str(feature_regime),
                                            seed=int(seed),
                                            include_neural_ratio=int(seed) in neural_seeds,
                                            include_neural_fqe=int(seed) in neural_seeds,
                                            ratio_feature_config=ratio_feature_config,
                                            fqe_solver_config=fqe_solver_config,
                                            weight_config=weight_config,
                                            neural_ratio_config=neural_ratio_config,
                                            neural_fqe_config=neural_fqe_config,
                                            evaluation_config=evaluation_config,
                                            behavior_shift_direction_scale=float(
                                                grid.get("behavior_shift_direction_scale", 1.0)
                                            ),
                                            behavior_shift_direction=grid.get("behavior_shift_direction"),
                                        )
                                        completed += 1
                                        if completed % 10 == 0 or completed == total:
                                            print(
                                                f"[{stage}] completed {completed}/{total} configurations",
                                                flush=True,
                                            )
    prefix = "gamma_sweep" if stage in {"gamma_final", "gamma_paper", "stationary_shift_paper"} else stage
    results_path = output_root / f"{prefix}_results.csv"
    summary_path = output_root / f"{prefix}_summary.csv"
    summary_rows = _aggregate_rows(rows, GAMMA_RESULT_COLUMNS, gamma_mode=True)
    _write_csv(results_path, rows, fieldnames=GAMMA_RESULT_COLUMNS)
    _write_csv(summary_path, summary_rows, fieldnames=GAMMA_RESULT_COLUMNS + SUMMARY_EXTRA_COLUMNS)
    _write_gamma_sweep_report(output_root / "gamma_sweep_report.md", summary_rows, stage)
    if stage in {"gamma_design", "gamma_final", "gamma_paper"}:
        _write_gamma_sweep_report(output_root / "design_search_report.md", summary_rows, stage)
    return results_path, summary_path


def run_smoke(output_root: Path) -> tuple[Path, Path]:
    rows: list[dict[str, object]] = []
    grid = smoke_stage_grid()
    ratio_feature_config = RatioFeatureConfig()
    fqe_solver_config = FQESolverConfig()
    weight_config = WeightEstimatorConfig()
    evaluation_config = EvaluationConfig()
    for gamma in grid["gammas"]:
        for shift in grid["shifts"]:
            for sample_size in grid["sample_sizes"]:
                for process_noise_sd in grid["process_noise_sds"]:
                    for behavior_action_sd in grid["behavior_action_sds"]:
                        for feature_regime in grid["feature_regimes"]:
                            for seed in grid["seeds"]:
                                _run_single_configuration(
                                    stage="smoke",
                                    output_stage_rows=rows,
                                    gamma=float(gamma),
                                    shift=float(shift),
                                    sample_size=int(sample_size),
                                    process_noise_sd=float(process_noise_sd),
                                    behavior_action_sd=float(behavior_action_sd),
                                    feature_regime=str(feature_regime),
                                    seed=int(seed),
                                    ratio_feature_config=ratio_feature_config,
                                    fqe_solver_config=fqe_solver_config,
                                    weight_config=weight_config,
                                    evaluation_config=evaluation_config,
                                )
    results_path = output_root / "smoke_results.csv"
    summary_path = output_root / "smoke_summary.csv"
    _write_csv(results_path, rows, fieldnames=RESULT_COLUMNS)
    _write_csv(summary_path, _aggregate_summary(rows), fieldnames=RESULT_COLUMNS + SUMMARY_EXTRA_COLUMNS)
    return results_path, summary_path


def run_design_search(output_root: Path) -> tuple[Path, Path, Path]:
    rows: list[dict[str, object]] = []
    grid = design_search_grid()
    ratio_feature_config = RatioFeatureConfig()
    fqe_solver_config = FQESolverConfig()
    weight_config = WeightEstimatorConfig()
    evaluation_config = EvaluationConfig()
    completed = 0
    total = (
        len(grid["gammas"])
        * len(grid["shifts"])
        * len(grid["sample_sizes"])
        * len(grid["process_noise_sds"])
        * len(grid["behavior_action_sds"])
        * len(grid["feature_regimes"])
        * len(grid["seeds"])
    )
    for gamma in grid["gammas"]:
        for shift in grid["shifts"]:
            for sample_size in grid["sample_sizes"]:
                for process_noise_sd in grid["process_noise_sds"]:
                    for behavior_action_sd in grid["behavior_action_sds"]:
                        for feature_regime in grid["feature_regimes"]:
                            for seed in grid["seeds"]:
                                _run_single_configuration(
                                    stage="design_search",
                                    output_stage_rows=rows,
                                    gamma=float(gamma),
                                    shift=float(shift),
                                    sample_size=int(sample_size),
                                    process_noise_sd=float(process_noise_sd),
                                    behavior_action_sd=float(behavior_action_sd),
                                    feature_regime=str(feature_regime),
                                    seed=int(seed),
                                    ratio_feature_config=ratio_feature_config,
                                    fqe_solver_config=fqe_solver_config,
                                    weight_config=weight_config,
                                    evaluation_config=evaluation_config,
                                )
                                completed += 1
                                if completed % 25 == 0 or completed == total:
                                    print(
                                        f"[design_search] completed {completed}/{total} configurations",
                                        flush=True,
                                    )
    results_path = output_root / "design_search_results.csv"
    summary_path = output_root / "design_search_summary.csv"
    summary_rows = _aggregate_summary(rows)
    selection = _select_family(summary_rows, weight_config)
    selection_path = output_root / "selected_final_family.csv"
    _write_csv(results_path, rows, fieldnames=RESULT_COLUMNS)
    _write_csv(summary_path, summary_rows, fieldnames=RESULT_COLUMNS + SUMMARY_EXTRA_COLUMNS)
    _write_selected_family(selection_path, selection)
    _write_design_search_report(output_root / "design_search_report.md", selection, summary_rows, weight_config)
    return results_path, summary_path, selection_path


def run_final(output_root: Path) -> tuple[Path, Path]:
    selection_path = output_root / "selected_final_family.csv"
    if not selection_path.exists():
        run_design_search(output_root)
    selection = _load_selected_family(selection_path)
    defaults = final_stage_defaults()
    rows: list[dict[str, object]] = []
    ratio_feature_config = RatioFeatureConfig()
    fqe_solver_config = FQESolverConfig()
    weight_config = WeightEstimatorConfig()
    evaluation_config = EvaluationConfig()
    shifts = [selection.low_shift, selection.moderate_shift, selection.severe_shift]
    completed = 0
    total = len(shifts) * len(defaults["sample_sizes"]) * len(defaults["feature_regimes"]) * len(defaults["seeds"])
    for shift in shifts:
        for sample_size in defaults["sample_sizes"]:
            for feature_regime in defaults["feature_regimes"]:
                for seed in defaults["seeds"]:
                    _run_single_configuration(
                        stage="final",
                        output_stage_rows=rows,
                        gamma=selection.gamma,
                        shift=float(shift),
                        sample_size=int(sample_size),
                        process_noise_sd=selection.process_noise_sd,
                        behavior_action_sd=selection.behavior_action_sd,
                        feature_regime=str(feature_regime),
                        seed=int(seed),
                        ratio_feature_config=ratio_feature_config,
                        fqe_solver_config=fqe_solver_config,
                        weight_config=weight_config,
                        evaluation_config=evaluation_config,
                    )
                    completed += 1
                    if completed % 25 == 0 or completed == total:
                        print(f"[final] completed {completed}/{total} configurations", flush=True)
    results_path = output_root / "final_results.csv"
    summary_path = output_root / "final_summary.csv"
    _write_csv(results_path, rows, fieldnames=RESULT_COLUMNS)
    _write_csv(summary_path, _aggregate_summary(rows), fieldnames=RESULT_COLUMNS + SUMMARY_EXTRA_COLUMNS)
    return results_path, summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Controlled discounted-occupancy FQE benchmark.")
    parser.add_argument(
        "--stage",
        choices=[
            "smoke",
            "design_search",
            "final",
            "gamma_smoke",
            "gamma_design",
            "gamma_final",
            "gamma_paper",
            "stationary_shift_paper",
        ],
        required=True,
    )
    parser.add_argument("--output-root", type=Path, default=Path("FQE_neurips/results"))
    args = parser.parse_args()

    output_root = args.output_root
    if args.stage == "smoke":
        run_smoke(output_root)
    elif args.stage == "design_search":
        run_design_search(output_root)
    elif args.stage == "final":
        run_final(output_root)
    else:
        _run_gamma_stage(output_root, stage=args.stage)


if __name__ == "__main__":
    main()
