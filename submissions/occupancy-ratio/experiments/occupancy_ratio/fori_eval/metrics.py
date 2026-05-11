from __future__ import annotations

from typing import Any

import numpy as np

from experiments.occupancy_ratio.fori_eval.finite_mdp import FiniteDataset, TabularTruth


Array = np.ndarray


def evaluate_grid_weights(
    *,
    dataset: FiniteDataset,
    weights: Array,
    raw_weights: Array | None = None,
    history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Compute exact tabular diagnostics for weights on the full state-action grid."""

    truth = dataset.truth
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    raw = w if raw_weights is None else np.asarray(raw_weights, dtype=np.float64).reshape(-1)
    if w.shape[0] != truth.nu.shape[0]:
        raise ValueError("weights must be defined on the full state-action grid.")

    out: dict[str, Any] = {}
    out.update(ratio_error_metrics(truth, w))
    out.update(weight_tail_metrics(truth.nu, w, raw))
    out.update(calibration_by_true_ratio(truth.nu, truth.omega_star, w))
    out.update(reward_sweep_metrics(dataset, w))
    out["bellman_flow_residual_l1"] = bellman_flow_residual_l1(truth, w)
    out["mass"] = float(np.sum(truth.nu * w))
    out["normalization_error"] = float(out["mass"] - 1.0)
    if history:
        out.update(history_metrics(history))
    return out


def evaluate_sample_weights(
    *,
    dataset: FiniteDataset,
    weights: Array,
    raw_weights: Array | None = None,
) -> dict[str, Any]:
    """Compute empirical diagnostics for estimators that only expose sample weights."""

    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    raw = w if raw_weights is None else np.asarray(raw_weights, dtype=np.float64).reshape(-1)
    truth = dataset.true_ratio_sample.reshape(-1)
    if w.shape != truth.shape:
        raise ValueError("sample weights must match the logged sample size.")
    diff = w - truth
    out: dict[str, Any] = {
        "ratio_l1_empirical": float(np.mean(np.abs(diff))),
        "ratio_rmse_empirical": float(np.sqrt(np.mean(diff**2))),
        "ratio_rel_mse_empirical": float(np.mean(diff**2) / max(float(np.mean(truth**2)), 1e-12)),
        "ratio_bias_empirical": float(np.mean(diff)),
        "sample_only_weights": 1.0,
    }
    sample_prob = np.ones_like(w) / max(w.size, 1)
    out.update(weight_tail_metrics(sample_prob, w, raw))
    rewards = np.asarray(dataset.rewards, dtype=np.float64).reshape(-1)
    out["env_reward_value_error_empirical"] = float(np.mean(w * rewards) - np.mean(truth * rewards))
    out["env_reward_value_abs_error_empirical"] = abs(float(out["env_reward_value_error_empirical"]))
    return out


def ratio_error_metrics(truth: TabularTruth, weights: Array) -> dict[str, float]:
    diff = np.asarray(weights, dtype=np.float64).reshape(-1) - truth.omega_star
    l1 = float(np.sum(truth.nu * np.abs(diff)))
    l2 = float(np.sqrt(np.sum(truth.nu * diff**2)))
    return {
        "ratio_l1_nu": l1,
        "ratio_tv": 0.5 * l1,
        "ratio_rmse_nu": l2,
        "ratio_bias_nu": float(np.sum(truth.nu * diff)),
        "ratio_rel_mse_nu": float(np.sum(truth.nu * diff**2) / max(np.sum(truth.nu * truth.omega_star**2), 1e-12)),
        "log_ratio_rmse_nu": float(
            np.sqrt(
                np.sum(
                    truth.nu
                    * (
                        np.log(np.maximum(weights, 1e-12))
                        - np.log(np.maximum(truth.omega_star, 1e-12))
                    )
                    ** 2
                )
            )
        ),
    }


def bellman_flow_residual_l1(truth: TabularTruth, weights: Array) -> float:
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    return float(np.sum(truth.nu * np.abs(w - truth.bellman_update(w))))


def weight_tail_metrics(nu: Array, weights: Array, raw_weights: Array | None = None) -> dict[str, float]:
    prob = np.asarray(nu, dtype=np.float64).reshape(-1)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    raw = w if raw_weights is None else np.asarray(raw_weights, dtype=np.float64).reshape(-1)
    mass = float(np.sum(prob * w))
    second = float(np.sum(prob * w**2))
    positive_median = float(weighted_quantile(w, prob, 0.50))
    order = np.argsort(w)[::-1]
    cutoff_count = max(1, int(np.ceil(0.01 * w.size)))
    top = order[:cutoff_count]
    return {
        "weight_min": float(np.min(w)),
        "weight_max": float(np.max(w)),
        "weight_mean_nu": mass,
        "weight_q50": positive_median,
        "weight_q90": float(weighted_quantile(w, prob, 0.90)),
        "weight_q95": float(weighted_quantile(w, prob, 0.95)),
        "weight_q99": float(weighted_quantile(w, prob, 0.99)),
        "max_to_median_ratio": float(np.max(w) / max(positive_median, 1e-12)),
        "effective_sample_size_fraction": float((mass**2) / max(second, 1e-12)),
        "top_1pct_mass_share": float(np.sum(prob[top] * w[top]) / max(mass, 1e-12)),
        "negative_raw_fraction": float(np.sum(prob * (raw < 0.0))),
        "clipping_fraction": float(np.sum(prob * (np.abs(raw - w) > 1e-12))),
    }


def calibration_by_true_ratio(nu: Array, truth: Array, weights: Array, n_bins: int = 10) -> dict[str, float]:
    prob = np.asarray(nu, dtype=np.float64).reshape(-1)
    t = np.asarray(truth, dtype=np.float64).reshape(-1)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    bins = np.array_split(np.argsort(t), int(n_bins))
    gaps = []
    rel_gaps = []
    for idx in bins:
        if idx.size == 0:
            continue
        mass = max(float(np.sum(prob[idx])), 1e-12)
        pred_mean = float(np.sum(prob[idx] * w[idx]) / mass)
        true_mean = float(np.sum(prob[idx] * t[idx]) / mass)
        gaps.append(abs(pred_mean - true_mean))
        rel_gaps.append(abs(pred_mean - true_mean) / max(abs(true_mean), 1e-12))
    return {
        "calibration_true_bin_abs_gap_mean": float(np.mean(gaps)) if gaps else np.nan,
        "calibration_true_bin_abs_gap_max": float(np.max(gaps)) if gaps else np.nan,
        "calibration_true_bin_rel_gap_mean": float(np.mean(rel_gaps)) if rel_gaps else np.nan,
    }


def reward_sweep_metrics(dataset: FiniteDataset, weights: Array) -> dict[str, float]:
    prob = dataset.truth.nu
    truth = dataset.truth.omega_star
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    true_values = dataset.reward_panel @ (prob * truth)
    pred_values = dataset.reward_panel @ (prob * w)
    diff = pred_values - true_values
    env_reward = dataset.mdp.rewards.reshape(-1)
    env_error = float(np.sum(prob * w * env_reward) - np.sum(prob * truth * env_reward))
    return {
        "reward_sweep_value_rmse": float(np.sqrt(np.mean(diff**2))),
        "reward_sweep_value_mae": float(np.mean(np.abs(diff))),
        "reward_sweep_value_max_abs": float(np.max(np.abs(diff))),
        "env_reward_value_error": env_error,
        "env_reward_value_abs_error": float(abs(env_error)),
    }


def history_metrics(history: list[dict[str, Any]]) -> dict[str, float]:
    out: dict[str, float] = {}
    l1 = [float(row["ratio_l1_nu"]) for row in history if "ratio_l1_nu" in row and np.isfinite(float(row["ratio_l1_nu"]))]
    residual = [
        float(row["bellman_flow_residual_l1"])
        for row in history
        if "bellman_flow_residual_l1" in row and np.isfinite(float(row["bellman_flow_residual_l1"]))
    ]
    if l1:
        out["history_ratio_l1_initial"] = float(l1[0])
        out["history_ratio_l1_final"] = float(l1[-1])
        out["history_ratio_l1_min"] = float(np.min(l1))
    if residual:
        out["history_flow_residual_final"] = float(residual[-1])
        out["history_flow_residual_min"] = float(np.min(residual))
    return out


def weighted_quantile(values: Array, weights: Array, q: float) -> float:
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    order = np.argsort(v)
    v = v[order]
    w = w[order]
    cdf = np.cumsum(w) / max(float(np.sum(w)), 1e-12)
    return float(v[min(np.searchsorted(cdf, float(q), side="left"), v.size - 1)])
