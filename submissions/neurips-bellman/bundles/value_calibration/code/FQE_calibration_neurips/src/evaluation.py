from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from .calibration.targets import apply_calibration, policy_value_predictions
from .data import TransitionBatch
from .environments import NonlinearMDP
from .policies import SoftmaxPolicy
from .utils import kfold_indices, mse


@dataclass
class EvaluationBundle:
    initial_states: np.ndarray
    test_batch: TransitionBatch
    oracle_value: float


def calibrated_q_values(
    model: object,
    states: np.ndarray,
    actions: np.ndarray,
    calibrator: object | None = None,
    calibration_target: str = "value_bellman",
) -> np.ndarray:
    q = model.predict_q(states, actions)
    if calibrator is not None and calibration_target != "value_bellman":
        q = apply_calibration(q, calibrator, calibration_target)
    return np.asarray(q, dtype=float)


def weighted_mse(prediction: np.ndarray, target: np.ndarray, weights: np.ndarray | None = None) -> float:
    pred = np.asarray(prediction, dtype=float)
    y = np.asarray(target, dtype=float)
    mask = np.isfinite(pred) & np.isfinite(y)
    if weights is not None:
        w = np.asarray(weights, dtype=float)
        mask &= np.isfinite(w) & (w >= 0)
        w = w[mask]
    else:
        w = None
    pred, y = pred[mask], y[mask]
    if pred.size == 0:
        return float("nan")
    err = (pred - y) ** 2
    if w is None:
        return float(np.mean(err))
    total = float(np.sum(w))
    return float(np.sum(w * err) / max(total, 1e-12))


def _rank_average(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(arr.size, dtype=float)
    sorted_values = arr[order]
    start = 0
    while start < arr.size:
        end = start + 1
        while end < arr.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def value_prediction_diagnostics(
    prediction: np.ndarray,
    true_value: np.ndarray | None,
    *,
    n_bins: int = 10,
) -> dict[str, float | str]:
    """Calibration diagnostics for value predictions against oracle V values."""
    if true_value is None:
        return {
            "value_oracle_pearson": float("nan"),
            "value_oracle_spearman": float("nan"),
            "value_calibration_slope": float("nan"),
            "value_calibration_intercept": float("nan"),
            "value_reliability_curve_json": "",
        }
    pred = np.asarray(prediction, dtype=float).reshape(-1)
    true = np.asarray(true_value, dtype=float).reshape(-1)
    finite = np.isfinite(pred) & np.isfinite(true)
    pred, true = pred[finite], true[finite]
    if pred.size < 2:
        return {
            "value_oracle_pearson": float("nan"),
            "value_oracle_spearman": float("nan"),
            "value_calibration_slope": float("nan"),
            "value_calibration_intercept": float("nan"),
            "value_reliability_curve_json": "",
        }
    pred_std = float(np.std(pred))
    true_std = float(np.std(true))
    pearson = float(np.corrcoef(pred, true)[0, 1]) if pred_std > 0 and true_std > 0 else float("nan")
    pred_rank = _rank_average(pred)
    true_rank = _rank_average(true)
    rank_std = float(np.std(pred_rank))
    true_rank_std = float(np.std(true_rank))
    spearman = (
        float(np.corrcoef(pred_rank, true_rank)[0, 1])
        if rank_std > 0 and true_rank_std > 0
        else float("nan")
    )
    slope = float("nan")
    intercept = float("nan")
    if pred_std > 0:
        slope = float(np.cov(pred, true, ddof=0)[0, 1] / max(float(np.var(pred)), 1e-12))
        intercept = float(np.mean(true) - slope * np.mean(pred))

    bins = min(max(int(n_bins), 1), pred.size)
    order = np.argsort(pred, kind="mergesort")
    curve = []
    for idx in np.array_split(order, bins):
        if idx.size:
            curve.append(
                {
                    "pred_mean": float(np.mean(pred[idx])),
                    "true_mean": float(np.mean(true[idx])),
                    "count": int(idx.size),
                }
            )
    return {
        "value_oracle_pearson": pearson,
        "value_oracle_spearman": spearman,
        "value_calibration_slope": slope,
        "value_calibration_intercept": intercept,
        "value_reliability_curve_json": json.dumps(curve, separators=(",", ":")),
    }


def estimate_policy_value_at_states(
    model: object,
    states: np.ndarray,
    target_policy: SoftmaxPolicy,
    calibrator: object | None = None,
    calibration_target: str = "value_bellman",
) -> np.ndarray:
    if calibration_target == "value_bellman":
        if hasattr(model, "predict_value"):
            return np.asarray(model.predict_value(states, target_policy, calibrator, calibration_target), dtype=float)
        values = policy_value_predictions(model, states, target_policy)
        if calibrator is not None:
            values = calibrator.predict(values)
        return np.asarray(values, dtype=float)
    probs = target_policy.action_probabilities(states)
    q_cols = []
    for action in range(probs.shape[1]):
        actions = np.full(states.shape[0], action, dtype=int)
        q_cols.append(calibrated_q_values(model, states, actions, calibrator, calibration_target))
    return np.sum(probs * np.column_stack(q_cols), axis=1)


def estimate_value(
    model: object,
    initial_states: np.ndarray,
    target_policy: SoftmaxPolicy,
    calibrator: object | None = None,
    calibration_target: str = "value_bellman",
) -> float:
    values = estimate_policy_value_at_states(model, initial_states, target_policy, calibrator, calibration_target)
    return float(np.mean(values))


def bootstrap_value_interval(
    value_predictions: np.ndarray,
    oracle_value: float,
    seed: int,
    n_bootstrap: int = 200,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Diagnostic bootstrap interval for the mean over evaluation states.

    This is intentionally a lightweight diagnostic over the independent initial
    evaluation-state set, not a formal OPE confidence interval.
    """

    values = np.asarray(value_predictions, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0 or int(n_bootstrap) <= 1:
        return {
            "interval_lower_95": float("nan"),
            "interval_upper_95": float("nan"),
            "interval_length_95": float("nan"),
            "interval_coverage_95": float("nan"),
        }
    rng = np.random.default_rng(int(seed))
    idx = rng.integers(0, values.size, size=(int(n_bootstrap), values.size))
    means = np.mean(values[idx], axis=1)
    lower = float(np.quantile(means, alpha / 2.0))
    upper = float(np.quantile(means, 1.0 - alpha / 2.0))
    return {
        "interval_lower_95": lower,
        "interval_upper_95": upper,
        "interval_length_95": float(upper - lower),
        "interval_coverage_95": float(lower <= float(oracle_value) <= upper),
    }


def bellman_residual(
    model: object,
    batch: TransitionBatch,
    gamma: float,
    calibrator: object | None,
    target_type: str,
    target_policy: SoftmaxPolicy | None = None,
    weights: np.ndarray | None = None,
) -> float:
    if target_type == "value_bellman":
        if target_policy is None:
            raise ValueError("target_policy is required for value-space Bellman residuals.")
        pred = estimate_policy_value_at_states(model, batch.states, target_policy, calibrator, target_type)
        target = batch.rewards + float(gamma) * estimate_policy_value_at_states(
            model, batch.next_states, target_policy, calibrator, target_type
        )
        return weighted_mse(pred, target, weights)
    pred = model.predict_q(batch.states, batch.actions)
    next_q = model.predict_q(batch.next_states, batch.next_actions)
    if calibrator is not None:
        pred = apply_calibration(pred, calibrator, target_type)
        next_q = apply_calibration(next_q, calibrator, target_type)
    target = batch.rewards + float(gamma) * next_q
    return mse(pred, target)


def calibration_error(model: object, batch: TransitionBatch, gamma: float, calibrator: object | None, target_type: str) -> float:
    pred = model.predict_q(batch.states, batch.actions)
    target = batch.rewards + float(gamma) * model.predict_q(batch.next_states, batch.next_actions)
    if calibrator is not None:
        pred = apply_calibration(pred, calibrator, target_type)
    return mse(pred, target)


def bellman_outcome_with_true_reward(
    model: object,
    batch: TransitionBatch,
    env: NonlinearMDP,
    target_policy: SoftmaxPolicy,
    gamma: float,
    calibrator: object | None,
    target_type: str,
) -> np.ndarray:
    continuation = estimate_policy_value_at_states(model, batch.next_states, target_policy, calibrator, target_type)
    return env.expected_reward(batch.states, batch.actions) + float(gamma) * continuation


def bellman_outcome_mse(
    model: object,
    batch: TransitionBatch,
    env: NonlinearMDP,
    target_policy: SoftmaxPolicy,
    gamma: float,
    calibrator: object | None,
    target_type: str,
    weights: np.ndarray | None = None,
) -> float:
    if target_type == "value_bellman":
        pred = estimate_policy_value_at_states(model, batch.states, target_policy, calibrator, target_type)
    else:
        pred = calibrated_q_values(model, batch.states, batch.actions, calibrator, target_type)
    outcome = bellman_outcome_with_true_reward(model, batch, env, target_policy, gamma, calibrator, target_type)
    return weighted_mse(pred, outcome, weights)


def true_q_function_mse(
    model: object,
    batch: TransitionBatch,
    true_q_values: np.ndarray | None,
    calibrator: object | None,
    target_type: str,
) -> float:
    if true_q_values is None:
        return float("nan")
    pred = calibrated_q_values(model, batch.states, batch.actions, calibrator, target_type)
    return mse(pred, np.asarray(true_q_values, dtype=float))


def true_v_function_mse(
    model: object,
    states: np.ndarray,
    true_v_values: np.ndarray | None,
    target_policy: SoftmaxPolicy,
    calibrator: object | None,
    target_type: str,
) -> float:
    if true_v_values is None:
        return float("nan")
    pred = estimate_policy_value_at_states(model, states, target_policy, calibrator, target_type)
    return mse(pred, np.asarray(true_v_values, dtype=float))


def _weighted_quantile(values: np.ndarray, weights: np.ndarray | None, quantiles: np.ndarray) -> np.ndarray:
    if weights is None or float(np.sum(weights)) <= 0:
        return np.quantile(values, quantiles)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    sorted_weights = weights[order]
    cdf = np.cumsum(sorted_weights) / max(float(np.sum(sorted_weights)), 1e-12)
    return np.interp(quantiles, cdf, sorted_values, left=sorted_values[0], right=sorted_values[-1])


def _quantile_bin_ids(
    values: np.ndarray,
    n_bins: int,
    min_bin_size: int,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, int]:
    arr = np.asarray(values, dtype=float)
    n = arr.size
    if n == 0:
        return np.zeros(0, dtype=int), 0
    max_bins = max(n // max(int(min_bin_size), 1), 1)
    bins = min(max(int(n_bins), 1), max_bins, n)
    if weights is None:
        order = np.argsort(arr, kind="mergesort")
        ids = np.zeros(n, dtype=int)
        for bin_id, idx in enumerate(np.array_split(order, bins)):
            ids[idx] = bin_id
        return ids, bins
    w = np.asarray(weights, dtype=float)
    quantiles = np.linspace(0.0, 1.0, bins + 1)
    edges = _weighted_quantile(arr, w, quantiles)
    edges[0], edges[-1] = -np.inf, np.inf
    edges = np.unique(np.maximum.accumulate(edges))
    if edges.size < 3:
        return np.zeros(n, dtype=int), 1
    ids = np.searchsorted(edges[1:-1], arr, side="right")
    bins = int(edges.size - 1)
    return ids, bins


def _fit_quantile_bin_curve(
    predictions: np.ndarray,
    outcomes: np.ndarray,
    n_bins: int,
    min_bin_size: int,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    pred = np.asarray(predictions, dtype=float)
    outcome = np.asarray(outcomes, dtype=float)
    finite = np.isfinite(pred) & np.isfinite(outcome)
    w = None
    if weights is not None:
        w_all = np.asarray(weights, dtype=float)
        finite &= np.isfinite(w_all) & (w_all >= 0)
        w = w_all[finite]
    pred, outcome = pred[finite], outcome[finite]
    if pred.size == 0:
        return np.array([-np.inf, np.inf]), np.array([float("nan")]), float("nan"), 1
    max_bins = max(pred.size // max(int(min_bin_size), 1), 1)
    bins = min(max(int(n_bins), 1), max_bins, pred.size)
    quantiles = np.linspace(0.0, 1.0, bins + 1)
    if w is None or float(np.sum(w)) <= 0:
        edges = np.quantile(pred, quantiles)
    else:
        order = np.argsort(pred)
        sorted_pred = pred[order]
        sorted_w = w[order]
        cdf = np.cumsum(sorted_w) / max(float(np.sum(sorted_w)), 1e-12)
        edges = np.interp(quantiles, cdf, sorted_pred, left=sorted_pred[0], right=sorted_pred[-1])
    edges[0], edges[-1] = -np.inf, np.inf
    edges = np.maximum.accumulate(edges)
    if w is None or float(np.sum(w)) <= 0:
        global_mean = float(np.mean(outcome))
    else:
        global_mean = float(np.sum(w * outcome) / max(float(np.sum(w)), 1e-12))
    bin_ids = np.searchsorted(edges[1:-1], pred, side="right")
    means = np.full(bins, global_mean, dtype=float)
    for bin_id in range(bins):
        mask = bin_ids == bin_id
        if np.any(mask):
            if w is None:
                means[bin_id] = float(np.mean(outcome[mask]))
            else:
                means[bin_id] = float(np.sum(w[mask] * outcome[mask]) / max(float(np.sum(w[mask])), 1e-12))
    return edges, means, global_mean, bins


def _predict_quantile_bin_curve(
    predictions: np.ndarray,
    edges: np.ndarray,
    means: np.ndarray,
    fallback: float,
) -> np.ndarray:
    pred = np.asarray(predictions, dtype=float)
    bin_ids = np.searchsorted(edges[1:-1], pred, side="right")
    out = means[np.clip(bin_ids, 0, len(means) - 1)]
    out = np.asarray(out, dtype=float)
    out[~np.isfinite(out)] = fallback
    return out


def bellman_calibration_error_from_arrays(
    predictions: np.ndarray,
    outcomes: np.ndarray,
    n_bins: int = 50,
    min_bin_size: int = 20,
    n_folds: int = 5,
    weights: np.ndarray | None = None,
) -> dict[str, float | int]:
    """Estimate plug-in and cross-fitted debiased binned calibration errors."""
    pred = np.asarray(predictions, dtype=float)
    outcome = np.asarray(outcomes, dtype=float)
    finite = np.isfinite(pred) & np.isfinite(outcome)
    w = None
    if weights is not None:
        w_all = np.asarray(weights, dtype=float)
        finite &= np.isfinite(w_all) & (w_all >= 0)
        w = w_all[finite]
    pred, outcome = pred[finite], outcome[finite]
    if pred.size == 0:
        return {
            "bellman_calibration_error": float("nan"),
            "bellman_calibration_error_debiased_raw": float("nan"),
            "bellman_calibration_error_plugin": float("nan"),
            "bellman_calibration_bins": 0,
            "bellman_calibration_test_size": 0,
        }

    bin_ids, actual_bins = _quantile_bin_ids(pred, n_bins, min_bin_size, w)
    plugin = 0.0
    total_weight = float(np.sum(w)) if w is not None else float(pred.size)
    for bin_id in range(actual_bins):
        mask = bin_ids == bin_id
        if np.any(mask):
            if w is None:
                share = float(np.mean(mask))
                out_mean = float(np.mean(outcome[mask]))
                pred_mean = float(np.mean(pred[mask]))
            else:
                share = float(np.sum(w[mask]) / max(total_weight, 1e-12))
                out_mean = float(np.sum(w[mask] * outcome[mask]) / max(float(np.sum(w[mask])), 1e-12))
                pred_mean = float(np.sum(w[mask] * pred[mask]) / max(float(np.sum(w[mask])), 1e-12))
            plugin += share * float((out_mean - pred_mean) ** 2)

    if pred.size < 2:
        return {
            "bellman_calibration_error": float(plugin),
            "bellman_calibration_error_debiased_raw": float(plugin),
            "bellman_calibration_error_plugin": float(plugin),
            "bellman_calibration_bins": int(actual_bins),
            "bellman_calibration_test_size": int(pred.size),
        }

    folds = min(max(int(n_folds), 2), pred.size)
    gamma_hat = np.full(pred.size, float("nan"), dtype=float)
    for train_idx, hold_idx in kfold_indices(pred.size, folds, seed=91037):
        train_w = None if w is None else w[train_idx]
        edges, means, fallback, _ = _fit_quantile_bin_curve(
            pred[train_idx], outcome[train_idx], n_bins, min_bin_size, weights=train_w
        )
        gamma_hat[hold_idx] = _predict_quantile_bin_curve(pred[hold_idx], edges, means, fallback)
    score = (outcome - pred) * (gamma_hat - pred)
    if w is None:
        raw = float(np.nanmean(score))
    else:
        raw = float(np.nansum(w * score) / max(float(np.nansum(w)), 1e-12))
    return {
        "bellman_calibration_error": float(max(raw, 0.0)) if np.isfinite(raw) else float("nan"),
        "bellman_calibration_error_debiased_raw": raw,
        "bellman_calibration_error_plugin": float(plugin),
        "bellman_calibration_bins": int(actual_bins),
        "bellman_calibration_test_size": int(pred.size),
    }


def bellman_calibration_error_50bin(
    model: object,
    batch: TransitionBatch,
    env: NonlinearMDP,
    target_policy: SoftmaxPolicy,
    gamma: float,
    calibrator: object | None,
    target_type: str,
    n_bins: int = 50,
    min_bin_size: int = 20,
    n_folds: int = 5,
    weights: np.ndarray | None = None,
) -> dict[str, float | int]:
    """Estimate L2 Bellman calibration error on an independent diagnostic set.

    The plug-in estimate bins calibrated predictions into quantile bins and
    measures the squared gap between bin-average Bellman outcomes and bin-average
    predictions. The debiased estimate follows the robust calibration-error form
    E[(Y - Delta)(gamma(Delta) - Delta)] with a cross-fitted binned calibration
    curve, where Y = r_0 + gamma * V_hat(S').
    """
    if target_type == "value_bellman":
        pred = estimate_policy_value_at_states(model, batch.states, target_policy, calibrator, target_type)
    else:
        pred = calibrated_q_values(model, batch.states, batch.actions, calibrator, target_type)
    outcome = bellman_outcome_with_true_reward(model, batch, env, target_policy, gamma, calibrator, target_type)
    return bellman_calibration_error_from_arrays(
        pred,
        outcome,
        n_bins=n_bins,
        min_bin_size=min_bin_size,
        n_folds=n_folds,
        weights=weights,
    )


def coverage_stratified_errors(
    model: object,
    batch: TransitionBatch,
    gamma: float,
    calibrator: object | None,
    target_type: str,
    n_quantiles: int = 5,
    target_policy: SoftmaxPolicy | None = None,
    env: NonlinearMDP | None = None,
) -> dict[str, float | int]:
    """Held-out Bellman error stratified by behavior-target coverage.

    The stratification uses only metadata already present on the independent
    test transition batch. It is diagnostic-only and is never fed back into
    fitting, calibration, or model selection.
    """
    if target_type == "value_bellman":
        if target_policy is None:
            raise ValueError("target_policy is required for value-space coverage diagnostics.")
        pred = estimate_policy_value_at_states(model, batch.states, target_policy, calibrator, target_type)
        target = batch.rewards + float(gamma) * estimate_policy_value_at_states(
            model, batch.next_states, target_policy, calibrator, target_type
        )
        calibration_target = (
            bellman_outcome_with_true_reward(model, batch, env, target_policy, gamma, calibrator, target_type)
            if env is not None
            else target
        )
    else:
        pred = model.predict_q(batch.states, batch.actions)
        next_q = model.predict_q(batch.next_states, batch.next_actions)
        if calibrator is not None:
            pred = apply_calibration(pred, calibrator, target_type)
            next_q = apply_calibration(next_q, calibrator, target_type)
        target = batch.rewards + float(gamma) * next_q
        calibration_target = target
    errors = (np.asarray(pred, dtype=float) - np.asarray(target, dtype=float)) ** 2
    calibration_errors = (np.asarray(pred, dtype=float) - np.asarray(calibration_target, dtype=float)) ** 2
    strata = coverage_strata(batch, n_quantiles=n_quantiles)
    ratio = np.asarray(batch.target_probs, dtype=float) / np.maximum(np.asarray(batch.behavior_probs, dtype=float), 1e-8)
    out: dict[str, float | int] = {}
    for stratum in range(int(n_quantiles)):
        mask = strata == stratum
        out[f"coverage_stratum_{stratum}_count"] = int(np.sum(mask))
        if np.any(mask):
            out[f"coverage_stratum_{stratum}_error"] = float(np.mean(errors[mask]))
            out[f"coverage_stratum_{stratum}_calibration_error"] = float(np.mean(calibration_errors[mask]))
            out[f"coverage_stratum_{stratum}_mean_ratio"] = float(np.mean(ratio[mask]))
        else:
            out[f"coverage_stratum_{stratum}_error"] = float("nan")
            out[f"coverage_stratum_{stratum}_calibration_error"] = float("nan")
            out[f"coverage_stratum_{stratum}_mean_ratio"] = float("nan")
    return out


def diagnostic_warning(
    model: object,
    batch: TransitionBatch,
    initial_states: np.ndarray,
    target_policy: SoftmaxPolicy,
    value_estimate: float,
    oracle_value: float,
    calibration_error_value: float,
    true_v_mse_value: float = float("nan"),
    true_q_mse_value: float = float("nan"),
    bellman_outcome_mse_value: float = float("nan"),
    bias_threshold: float = 5.0,
    prediction_threshold: float = 1e4,
) -> tuple[bool, str]:
    messages: list[str] = []
    hard_failure = False
    q_test = model.predict_q(batch.states, batch.actions)
    q_init = []
    for action in range(target_policy.action_probabilities(initial_states).shape[1]):
        q_init.append(model.predict_q(initial_states, np.full(initial_states.shape[0], action, dtype=int)))
    q_all = np.concatenate([np.asarray(q_test, dtype=float), *[np.asarray(q, dtype=float) for q in q_init]])
    if not np.all(np.isfinite(q_all)) or not np.isfinite(value_estimate):
        messages.append("nonfinite_prediction_or_value")
        hard_failure = True
    if q_all.size and float(np.nanmax(np.abs(q_all))) > prediction_threshold:
        messages.append("exploding_predictions")
        hard_failure = True
    if abs(float(value_estimate) - float(oracle_value)) > bias_threshold:
        messages.append("extreme_value_bias")
    if not np.isfinite(calibration_error_value):
        messages.append("nonfinite_calibration_error")
        hard_failure = True
    if not np.isfinite(true_v_mse_value):
        messages.append("nonfinite_true_v_mse")
    if not np.isfinite(true_q_mse_value):
        messages.append("nonfinite_true_q_mse")
    if not np.isfinite(bellman_outcome_mse_value):
        messages.append("nonfinite_bellman_outcome_mse")
    return hard_failure, ";".join(messages)


def coverage_strata(batch: TransitionBatch, n_quantiles: int = 5) -> np.ndarray:
    ratio = np.asarray(batch.target_probs, dtype=float) / np.maximum(np.asarray(batch.behavior_probs, dtype=float), 1e-8)
    edges = np.unique(np.quantile(ratio, np.linspace(0.0, 1.0, n_quantiles + 1)))
    if edges.size < 3:
        return np.zeros(len(batch), dtype=int)
    return np.searchsorted(edges[1:-1], ratio, side="right")


def make_result_row(
    *,
    env: NonlinearMDP,
    seed: int,
    sample_size: int,
    coverage: str,
    reward_noise: float,
    learner: str,
    calibrated: bool,
    protocol: str,
    calibrator_name: str,
    calibration_target: str,
    all_data: bool,
    sample_splitting: bool,
    train_fraction: float,
    calibration_fraction: float,
    value_estimate: float,
    oracle_value: float,
    bellman_residual_value: float,
    calibration_error_value: float,
    runtime: float,
    failure_flag: bool,
    true_v_mse_value: float = float("nan"),
    true_q_mse_value: float = float("nan"),
    bellman_outcome_mse_value: float = float("nan"),
    bellman_calibration_plugin_value: float = float("nan"),
    bellman_calibration_raw_value: float = float("nan"),
    bellman_calibration_bins: int = 0,
    bellman_calibration_test_size: int = 0,
    interval_lower_95: float = float("nan"),
    interval_upper_95: float = float("nan"),
    interval_length_95: float = float("nan"),
    interval_coverage_95: float = float("nan"),
    failure_reason: str = "",
    diagnostic_warning_message: str = "",
    extra: dict[str, float | str] | None = None,
) -> dict[str, float | str | bool | int]:
    value_error = float(value_estimate - oracle_value)
    row: dict[str, float | str | bool | int] = {
        "environment_name": env.config.name,
        "replication_seed": int(seed),
        "sample_size": int(sample_size),
        "state_dimension": int(env.state_dim),
        "coverage_setting": coverage,
        "reward_noise_setting": float(reward_noise),
        "baseline_learner": learner,
        "calibrated": bool(calibrated),
        "calibration_protocol": protocol,
        "calibrator": calibrator_name,
        "calibration_target": calibration_target,
        "base_learner_used_all_data": bool(all_data),
        "sample_splitting_used": bool(sample_splitting),
        "train_fraction": float(train_fraction),
        "calibration_fraction": float(calibration_fraction),
        "value_estimate": float(value_estimate),
        "oracle_value": float(oracle_value),
        "value_error": value_error,
        "squared_error": value_error**2,
        "bellman_residual": float(bellman_residual_value),
        "calibration_error": float(calibration_error_value),
        "true_v_mse": float(true_v_mse_value),
        "true_value_function_mse": float(true_v_mse_value),
        "true_q_mse": float(true_q_mse_value),
        "true_function_mse": float(true_q_mse_value),
        "bellman_outcome_mse": float(bellman_outcome_mse_value),
        "brier_score": float(bellman_outcome_mse_value),
        "bellman_brier_score": float(bellman_outcome_mse_value),
        "bellman_calibration_error": float(calibration_error_value),
        "bellman_calibration_error_plugin": float(bellman_calibration_plugin_value),
        "bellman_calibration_error_debiased_raw": float(bellman_calibration_raw_value),
        "bellman_calibration_bins": int(bellman_calibration_bins),
        "bellman_calibration_test_size": int(bellman_calibration_test_size),
        "interval_lower_95": float(interval_lower_95),
        "interval_upper_95": float(interval_upper_95),
        "interval_length_95": float(interval_length_95),
        "interval_coverage_95": float(interval_coverage_95),
        "runtime": float(runtime),
        "failure_flag": bool(failure_flag),
        "failure_reason": failure_reason,
        "diagnostic_warning_message": diagnostic_warning_message,
    }
    if extra:
        row.update(extra)
    return row
