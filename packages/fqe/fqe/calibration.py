from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from fqe.bellman import build_bellman_target, validate_action_weights, validate_bootstrap_inputs, weighted_action_expectation


Array = np.ndarray
CalibrationMethod = Literal[
    "histogram_constant",
    "histogram_rescale",
    "isotonic_histogram_constant",
    "isotonic_histogram_rescale",
]
BinStrategy = Literal["quantile", "equal_width"]

__all__ = [
    "BellmanCalibrator",
    "fit_bellman_calibrator",
    "fit_q_bellman_calibrator",
    "fit_value_bellman_calibrator",
    "bellman_calibration_diagnostics",
    "recommend_bellman_calibration",
    "plot_bellman_calibration_diagnostics",
]


@dataclass
class BellmanCalibrator:
    """Post-hoc one-dimensional Bellman calibration map.

    The map is fitted from raw FQE predictions to Bellman targets and can be
    applied to current predictions and next predictions. Histogram-rescale
    methods correct the bin mean while preserving raw within-bin deviations.
    """

    method: CalibrationMethod
    bin_edges: Array
    bin_prediction_mean: Array
    bin_target_mean: Array
    bin_counts: Array
    n_iterations: int
    bin_strategy: str
    min_bin_size: int
    prediction_min: float
    prediction_max: float
    diagnostics: dict[str, Any]

    def predict(self, predictions: Array) -> Array:
        pred = _as_finite_vector(predictions, "predictions")
        bins = self.bin_indices(pred)
        if self.method.endswith("_constant"):
            return self.bin_target_mean[bins].astype(np.float64, copy=True)
        return (self.bin_target_mean[bins] + (pred - self.bin_prediction_mean[bins])).astype(np.float64, copy=False)

    def bin_indices(self, predictions: Array) -> Array:
        pred = _as_finite_vector(predictions, "predictions")
        return np.clip(np.searchsorted(self.bin_edges[1:-1], pred, side="right"), 0, self.bin_target_mean.shape[0] - 1)

    def extrapolation_counts(self, predictions: Array) -> dict[str, int]:
        pred = _as_finite_vector(predictions, "predictions")
        return {
            "below_training_range": int(np.sum(pred < self.prediction_min)),
            "above_training_range": int(np.sum(pred > self.prediction_max)),
        }


def fit_bellman_calibrator(
    predictions: Array,
    next_predictions: Array,
    rewards: Array,
    gamma: float,
    *,
    terminals: Array | None = None,
    timeouts: Array | None = None,
    continuation: Array | None = None,
    sample_weight: Array | None = None,
    method: CalibrationMethod = "histogram_rescale",
    n_bins: int = 10,
    min_bin_size: int = 30,
    n_iterations: int = 4,
    bin_strategy: BinStrategy = "quantile",
) -> BellmanCalibrator:
    """Fit a post-hoc Bellman calibrator from arrays.

    Iteration 1 uses ``reward + gamma * (1 - terminal) * next_prediction`` as
    the target. Later iterations use calibrated next predictions from the
    previous iteration.
    """

    pred, next_pred, rew, term, weight = _validated_transition_vectors(
        predictions,
        next_predictions,
        rewards,
        terminals=terminals,
        timeouts=timeouts,
        continuation=continuation,
        sample_weight=sample_weight,
    )
    gamma_f = _validate_gamma(gamma)
    method = _validate_method(method)
    if bin_strategy not in {"quantile", "equal_width"}:
        raise ValueError("bin_strategy must be 'quantile' or 'equal_width'.")
    bins_requested = _validate_positive_int(n_bins, "n_bins")
    min_size = _validate_positive_int(min_bin_size, "min_bin_size")
    iterations = _validate_positive_int(n_iterations, "n_iterations")

    calibrator: BellmanCalibrator | None = None
    losses: list[float] = []
    targets = build_bellman_target(rewards=rew, gamma=gamma_f, next_predictions=next_pred, continuation=1.0 - term)
    for _ in range(iterations):
        if calibrator is not None:
            targets = build_bellman_target(
                rewards=rew,
                gamma=gamma_f,
                next_predictions=calibrator.predict(next_pred),
                continuation=1.0 - term,
            )
        calibrator = _fit_single_step_calibrator(
            pred,
            targets,
            sample_weight=weight,
            method=method,
            n_bins=bins_requested,
            min_bin_size=min_size,
            bin_strategy=bin_strategy,
            n_iterations=iterations,
        )
        calibrated = calibrator.predict(pred)
        losses.append(_weighted_mean((calibrated - targets) ** 2, weight))

    assert calibrator is not None
    calibrator.diagnostics.update(
        {
            "bellman_calibration_loss_first": float(losses[0]) if losses else np.nan,
            "bellman_calibration_loss_last": float(losses[-1]) if losses else np.nan,
            "gamma": float(gamma_f),
            "n_iterations": int(iterations),
            "n_samples": int(pred.shape[0]),
        }
    )
    return calibrator


def fit_q_bellman_calibrator(
    model: Any,
    states: Array,
    actions: Array,
    next_states: Array,
    next_actions: Array,
    rewards: Array,
    gamma: float,
    *,
    terminals: Array | None = None,
    timeouts: Array | None = None,
    continuation: Array | None = None,
    sample_weight: Array | None = None,
    next_action_weights: Array | None = None,
    method: CalibrationMethod = "histogram_rescale",
    n_bins: int = 10,
    min_bin_size: int = 30,
    n_iterations: int = 4,
    bin_strategy: BinStrategy = "quantile",
) -> BellmanCalibrator:
    """Fit a Bellman calibrator for a Q-mode FQE model."""

    pred = np.asarray(model.predict_q(states, actions), dtype=np.float64).reshape(-1)
    next_pred = _predict_q_next_average(model, next_states, next_actions, next_action_weights=next_action_weights)
    return fit_bellman_calibrator(
        pred,
        next_pred,
        rewards,
        gamma,
        terminals=terminals,
        timeouts=timeouts,
        continuation=continuation,
        sample_weight=sample_weight,
        method=method,
        n_bins=n_bins,
        min_bin_size=min_bin_size,
        n_iterations=n_iterations,
        bin_strategy=bin_strategy,
    )


def fit_value_bellman_calibrator(
    model: Any,
    states: Array,
    next_states: Array,
    rewards: Array,
    gamma: float,
    *,
    terminals: Array | None = None,
    timeouts: Array | None = None,
    continuation: Array | None = None,
    sample_weight: Array | None = None,
    method: CalibrationMethod = "histogram_rescale",
    n_bins: int = 10,
    min_bin_size: int = 30,
    n_iterations: int = 4,
    bin_strategy: BinStrategy = "quantile",
) -> BellmanCalibrator:
    """Fit a Bellman calibrator for a value-mode FQE model."""

    pred = np.asarray(model.predict_value(states), dtype=np.float64).reshape(-1)
    next_pred = np.asarray(model.predict_value(next_states), dtype=np.float64).reshape(-1)
    return fit_bellman_calibrator(
        pred,
        next_pred,
        rewards,
        gamma,
        terminals=terminals,
        timeouts=timeouts,
        continuation=continuation,
        sample_weight=sample_weight,
        method=method,
        n_bins=n_bins,
        min_bin_size=min_bin_size,
        n_iterations=n_iterations,
        bin_strategy=bin_strategy,
    )


def bellman_calibration_diagnostics(
    predictions: Array,
    next_predictions: Array,
    rewards: Array,
    gamma: float,
    *,
    calibrator: BellmanCalibrator | None = None,
    terminals: Array | None = None,
    timeouts: Array | None = None,
    continuation: Array | None = None,
    sample_weight: Array | None = None,
    n_bins: int = 20,
    min_bin_size: int = 30,
    n_folds: int = 5,
) -> dict[str, Any]:
    """Compute user-facing Bellman calibration diagnostics.

    Reports plug-in, fixed-bin debiased, and cross-fitted debiased calibration
    error before and after applying ``calibrator``.
    """

    pred, next_pred, rew, term, weight = _validated_transition_vectors(
        predictions,
        next_predictions,
        rewards,
        terminals=terminals,
        timeouts=timeouts,
        continuation=continuation,
        sample_weight=sample_weight,
    )
    gamma_f = _validate_gamma(gamma)
    bins_requested = _validate_positive_int(n_bins, "n_bins")
    min_size = _validate_positive_int(min_bin_size, "min_bin_size")
    folds_requested = _validate_positive_int(n_folds, "n_folds")

    outcome_before = build_bellman_target(rewards=rew, gamma=gamma_f, next_predictions=next_pred, continuation=1.0 - term)
    if calibrator is None:
        pred_after = pred.copy()
        next_after = next_pred.copy()
        extrapolation = {"below_training_range": 0, "above_training_range": 0}
    else:
        pred_after = calibrator.predict(pred)
        next_after = calibrator.predict(next_pred)
        extrapolation = calibrator.extrapolation_counts(pred)
    outcome_after = build_bellman_target(rewards=rew, gamma=gamma_f, next_predictions=next_after, continuation=1.0 - term)

    before = _calibration_error_summary(pred, outcome_before, weight, bins_requested, min_size, folds_requested)
    after = _calibration_error_summary(pred_after, outcome_after, weight, bins_requested, min_size, folds_requested)
    residual_before = _weighted_mean((pred - outcome_before) ** 2, weight)
    residual_after = _weighted_mean((pred_after - outcome_after) ** 2, weight)
    finite_after = bool(np.all(np.isfinite(pred_after)) and np.all(np.isfinite(next_after)))

    diagnostics: dict[str, Any] = {
        "method": "none" if calibrator is None else calibrator.method,
        "gamma": float(gamma_f),
        "n_samples": int(pred.shape[0]),
        "n_bins_requested": int(bins_requested),
        "min_bin_size": int(min_size),
        "n_folds": int(min(max(folds_requested, 2), pred.shape[0])) if pred.shape[0] >= 2 else 1,
        "bellman_residual_mse_before": float(residual_before),
        "bellman_residual_mse_after": float(residual_after),
        "bellman_residual_mse_change": float(residual_after - residual_before),
        "bellman_calibration_error_plugin_before": float(before["plugin"]),
        "bellman_calibration_error_plugin_after": float(after["plugin"]),
        "bellman_calibration_error_debiased_before": float(before["debiased"]),
        "bellman_calibration_error_debiased_after": float(after["debiased"]),
        "bellman_calibration_error_crossfit_before": float(before["crossfit"]),
        "bellman_calibration_error_crossfit_after": float(after["crossfit"]),
        "bellman_calibration_bins_before": int(before["bins"]),
        "bellman_calibration_bins_after": int(after["bins"]),
        "bellman_calibration_test_size": int(pred.shape[0]),
        "bin_table_before": before["bin_table"],
        "bin_table_after": after["bin_table"],
        "bin_table": after["bin_table"],
        "finite_calibrated_predictions": finite_after,
        "extrapolation_below_training_range": int(extrapolation["below_training_range"]),
        "extrapolation_above_training_range": int(extrapolation["above_training_range"]),
    }
    if calibrator is not None:
        calibrator_min_bin_size = int(getattr(calibrator, "min_bin_size", min_size))
        diagnostics.update(
            {
                "calibrator_effective_bins": int(calibrator.bin_target_mean.shape[0]),
                "calibrator_min_bin_size": int(calibrator_min_bin_size),
                "calibrator_small_bin_count": int(np.sum(calibrator.bin_counts < calibrator_min_bin_size)),
                "calibrator_empty_bin_count": int(np.sum(calibrator.bin_counts <= 0)),
                "calibrator_prediction_min": float(calibrator.prediction_min),
                "calibrator_prediction_max": float(calibrator.prediction_max),
            }
        )
    diagnostics.update(recommend_bellman_calibration(diagnostics))
    return diagnostics


def recommend_bellman_calibration(
    diagnostics: dict[str, Any],
    *,
    min_error_reduction: float = 1e-3,
    min_absolute_error_reduction: float = 1e-4,
    max_residual_mse_increase_fraction: float = 1e-3,
    max_small_bin_fraction: float = 0.5,
) -> dict[str, Any]:
    """Return a conservative recommendation for applying calibration."""

    plugin_before = float(diagnostics.get("bellman_calibration_error_plugin_before", np.nan))
    plugin_after = float(diagnostics.get("bellman_calibration_error_plugin_after", np.nan))
    debiased_before = float(diagnostics.get("bellman_calibration_error_debiased_before", np.nan))
    debiased_after = float(diagnostics.get("bellman_calibration_error_debiased_after", np.nan))
    cross_before = float(diagnostics.get("bellman_calibration_error_crossfit_before", np.nan))
    cross_after = float(diagnostics.get("bellman_calibration_error_crossfit_after", np.nan))
    resid_before = float(diagnostics.get("bellman_residual_mse_before", np.nan))
    resid_after = float(diagnostics.get("bellman_residual_mse_after", np.nan))
    effective_bins = int(diagnostics.get("calibrator_effective_bins", diagnostics.get("bellman_calibration_bins_after", 1)))
    small_bins = int(diagnostics.get("calibrator_small_bin_count", 0))
    finite_ok = bool(diagnostics.get("finite_calibrated_predictions", True))

    plugin_drop = _safe_relative_drop(plugin_before, plugin_after)
    debiased_drop = _safe_relative_drop(debiased_before, debiased_after)
    cross_drop = _safe_relative_drop(cross_before, cross_after)
    plugin_abs_drop = _safe_absolute_drop(plugin_before, plugin_after)
    debiased_abs_drop = _safe_absolute_drop(debiased_before, debiased_after)
    cross_abs_drop = _safe_absolute_drop(cross_before, cross_after)
    residual_increase = _safe_relative_increase(resid_before, resid_after)
    small_bin_fraction = float(small_bins / max(effective_bins, 1))
    best_relative_drop = max(_finite_or_neg_inf(debiased_drop), _finite_or_neg_inf(cross_drop), _finite_or_neg_inf(plugin_drop))
    best_absolute_drop = max(
        _finite_or_neg_inf(debiased_abs_drop),
        _finite_or_neg_inf(cross_abs_drop),
        _finite_or_neg_inf(plugin_abs_drop),
    )

    reasons: list[str] = []
    if not finite_ok:
        recommendation = "do_not_apply"
        reasons.append("Calibrated predictions or calibrated Bellman targets are not finite.")
    elif small_bin_fraction > float(max_small_bin_fraction):
        recommendation = "do_not_apply"
        reasons.append("Too many calibration bins are below the configured minimum bin size.")
    elif np.isfinite(cross_drop) and cross_drop < -float(min_error_reduction):
        recommendation = "do_not_apply"
        reasons.append("Cross-fitted debiased calibration error worsens after calibration.")
    elif np.isfinite(debiased_drop) and debiased_drop < -float(min_error_reduction):
        recommendation = "do_not_apply"
        reasons.append("Fixed-bin debiased calibration error worsens after calibration.")
    elif np.isfinite(residual_increase) and residual_increase > float(max_residual_mse_increase_fraction):
        recommendation = "do_not_apply"
        reasons.append("Bellman residual MSE increases beyond the configured tolerance.")
    elif best_relative_drop >= float(min_error_reduction) and best_absolute_drop >= float(min_absolute_error_reduction):
        recommendation = "apply"
        reasons.append("Calibration reduces Bellman calibration error and residual diagnostics remain within tolerance.")
    else:
        recommendation = "neutral"
        reasons.append("Calibration diagnostics are stable, but the estimated improvement is too small to matter.")

    return {
        "calibration_recommendation": recommendation,
        "calibration_recommendation_reasons": reasons,
        "plugin_error_reduction_fraction": float(plugin_drop),
        "debiased_error_reduction_fraction": float(debiased_drop),
        "crossfit_error_reduction_fraction": float(cross_drop),
        "plugin_error_absolute_reduction": float(plugin_abs_drop),
        "debiased_error_absolute_reduction": float(debiased_abs_drop),
        "crossfit_error_absolute_reduction": float(cross_abs_drop),
        "residual_mse_increase_fraction": float(residual_increase),
        "small_bin_fraction": float(small_bin_fraction),
        "recommendation_thresholds": {
            "min_error_reduction": float(min_error_reduction),
            "min_absolute_error_reduction": float(min_absolute_error_reduction),
            "max_residual_mse_increase_fraction": float(max_residual_mse_increase_fraction),
            "max_small_bin_fraction": float(max_small_bin_fraction),
        },
    }


def plot_bellman_calibration_diagnostics(
    diagnostics: dict[str, Any],
    *,
    path: str | None = None,
    show: bool = False,
):
    """Plot Bellman calibration diagnostics with lazy matplotlib import."""

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional dependency.
        raise ImportError("matplotlib is required for Bellman calibration diagnostic plots.") from exc

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    metric_names = ["plugin", "debiased", "crossfit"]
    before = [float(diagnostics.get(f"bellman_calibration_error_{name}_before", np.nan)) for name in metric_names]
    after = [float(diagnostics.get(f"bellman_calibration_error_{name}_after", np.nan)) for name in metric_names]
    x = np.arange(len(metric_names))
    axes[0, 0].bar(x - 0.18, before, width=0.36, label="before", color="#777777")
    axes[0, 0].bar(x + 0.18, after, width=0.36, label="after", color="#2b6cb0")
    axes[0, 0].set_xticks(x, metric_names)
    axes[0, 0].set_ylabel("calibration error")
    axes[0, 0].set_title("Binned Bellman Calibration")
    axes[0, 0].legend()

    resid_before = float(diagnostics.get("bellman_residual_mse_before", np.nan))
    resid_after = float(diagnostics.get("bellman_residual_mse_after", np.nan))
    axes[0, 1].bar(["before", "after"], [resid_before, resid_after], color=["#777777", "#2b6cb0"])
    axes[0, 1].set_title("Bellman Residual MSE")

    table = diagnostics.get("bin_table_after") or diagnostics.get("bin_table") or []
    bins = [int(row["bin"]) for row in table]
    gaps = [float(row["gap_after"]) for row in table]
    counts = [int(row["count"]) for row in table]
    axes[1, 0].axhline(0.0, color="#444444", linewidth=0.8)
    axes[1, 0].bar(bins, gaps, color="#2b6cb0")
    axes[1, 0].set_title("After Calibration Bin Gaps")
    axes[1, 0].set_xlabel("prediction bin")
    axes[1, 0].set_ylabel("target mean - prediction mean")

    axes[1, 1].bar(bins, counts, color="#4a5568")
    axes[1, 1].set_title("Bin Counts")
    axes[1, 1].set_xlabel("prediction bin")
    axes[1, 1].set_ylabel("count")

    recommendation = diagnostics.get("calibration_recommendation")
    if recommendation:
        fig.suptitle(f"Bellman Calibration Diagnostics: {recommendation}", fontsize=13)
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def _fit_single_step_calibrator(
    predictions: Array,
    targets: Array,
    *,
    sample_weight: Array | None,
    method: CalibrationMethod,
    n_bins: int,
    min_bin_size: int,
    bin_strategy: str,
    n_iterations: int,
) -> BellmanCalibrator:
    pred = _as_finite_vector(predictions, "predictions")
    target = _as_finite_vector(targets, "targets")
    if pred.shape != target.shape:
        raise ValueError("predictions and targets must have the same length.")
    weight = _optional_weights(sample_weight, pred.shape[0], "sample_weight")
    finite = np.isfinite(pred) & np.isfinite(target) & np.isfinite(weight) & (weight >= 0.0)
    pred = pred[finite]
    target = target[finite]
    weight = weight[finite]
    if pred.size < 2:
        raise ValueError("Need at least two finite calibration rows.")
    if float(np.sum(weight)) <= 0.0:
        weight = np.ones_like(pred)

    if method.startswith("isotonic_"):
        bin_id, edges = _isotonic_bin_ids(pred, target, weight, n_bins=n_bins, min_bin_size=min_bin_size)
    else:
        bin_id, edges = _histogram_bin_ids(pred, weight, n_bins=n_bins, min_bin_size=min_bin_size, strategy=bin_strategy)
    n_effective_bins = int(np.max(bin_id)) + 1
    pred_mean = np.zeros(n_effective_bins, dtype=np.float64)
    target_mean = np.zeros(n_effective_bins, dtype=np.float64)
    counts = np.zeros(n_effective_bins, dtype=np.int64)
    global_pred_mean = _weighted_mean(pred, weight)
    global_target_mean = _weighted_mean(target, weight)
    for idx in range(n_effective_bins):
        mask = bin_id == idx
        counts[idx] = int(np.sum(mask))
        if np.any(mask):
            pred_mean[idx] = _weighted_mean(pred[mask], weight[mask])
            target_mean[idx] = _weighted_mean(target[mask], weight[mask])
        else:
            pred_mean[idx] = global_pred_mean
            target_mean[idx] = global_target_mean

    return BellmanCalibrator(
        method=method,
        bin_edges=edges,
        bin_prediction_mean=pred_mean,
        bin_target_mean=target_mean,
        bin_counts=counts,
        n_iterations=int(n_iterations),
        bin_strategy=str(bin_strategy),
        min_bin_size=int(min_bin_size),
        prediction_min=float(np.min(pred)),
        prediction_max=float(np.max(pred)),
        diagnostics={
            "method": method,
            "effective_bins": int(n_effective_bins),
            "bin_counts": counts.copy(),
            "bin_prediction_mean": pred_mean.copy(),
            "bin_target_mean": target_mean.copy(),
            "bin_edges": edges.copy(),
            "small_bin_count": int(np.sum(counts < int(min_bin_size))),
            "empty_bin_count": int(np.sum(counts <= 0)),
        },
    )


def _calibration_error_summary(
    predictions: Array,
    outcomes: Array,
    weights: Array,
    n_bins: int,
    min_bin_size: int,
    n_folds: int,
) -> dict[str, Any]:
    pred = _as_finite_vector(predictions, "predictions")
    outcome = _as_finite_vector(outcomes, "outcomes")
    if pred.shape != outcome.shape:
        raise ValueError("predictions and outcomes must have the same length.")
    weight = _optional_weights(weights, pred.shape[0], "weights")
    bin_id, _ = _histogram_bin_ids(pred, weight, n_bins=n_bins, min_bin_size=min_bin_size, strategy="quantile")
    actual_bins = int(np.max(bin_id)) + 1
    total_weight = max(float(np.sum(weight)), 1e-12)
    plugin = 0.0
    debiased = 0.0
    table: list[dict[str, Any]] = []
    for idx in range(actual_bins):
        mask = bin_id == idx
        if not np.any(mask):
            continue
        w_b = weight[mask]
        pred_b = pred[mask]
        out_b = outcome[mask]
        weight_sum = float(np.sum(w_b))
        share = weight_sum / total_weight
        pred_mean = _weighted_mean(pred_b, w_b)
        out_mean = _weighted_mean(out_b, w_b)
        gap = out_mean - pred_mean
        plugin += share * gap * gap
        diff = out_b - pred_b
        denom = weight_sum * weight_sum - float(np.sum(w_b * w_b))
        if denom > 1e-12:
            u_stat = (float(np.sum(w_b * diff)) ** 2 - float(np.sum((w_b * diff) ** 2))) / denom
        else:
            u_stat = gap * gap
        debiased += share * u_stat
        table.append(
            {
                "bin": int(idx),
                "count": int(np.sum(mask)),
                "weight_sum": float(weight_sum),
                "prediction_min": float(np.min(pred_b)),
                "prediction_max": float(np.max(pred_b)),
                "prediction_mean": float(pred_mean),
                "target_mean": float(out_mean),
                "calibrated_mean": float(pred_mean),
                "gap_after": float(gap),
                "weighted_gap": float(share * gap),
                "plugin_contribution": float(share * gap * gap),
                "debiased_contribution_raw": float(share * u_stat),
            }
        )
    crossfit = _crossfit_debiased_error(pred, outcome, weight, n_bins=n_bins, min_bin_size=min_bin_size, n_folds=n_folds)
    return {
        "plugin": float(plugin),
        "debiased": float(debiased),
        "crossfit": float(crossfit),
        "bins": int(actual_bins),
        "bin_table": table,
    }


def _crossfit_debiased_error(
    predictions: Array,
    outcomes: Array,
    weights: Array,
    *,
    n_bins: int,
    min_bin_size: int,
    n_folds: int,
) -> float:
    pred = _as_finite_vector(predictions, "predictions")
    outcome = _as_finite_vector(outcomes, "outcomes")
    weight = _optional_weights(weights, pred.shape[0], "weights")
    if pred.size < 2:
        return float("nan")
    folds = min(max(int(n_folds), 2), pred.size)
    gamma_hat = np.full(pred.shape[0], np.nan, dtype=np.float64)
    for train_idx, hold_idx in _kfold_indices(pred.shape[0], folds, seed=91037):
        edges, means, fallback = _fit_bin_curve(pred[train_idx], outcome[train_idx], weight[train_idx], n_bins, min_bin_size)
        gamma_hat[hold_idx] = _predict_bin_curve(pred[hold_idx], edges, means, fallback)
    score = (outcome - pred) * (gamma_hat - pred)
    return _weighted_mean(score[np.isfinite(score)], weight[np.isfinite(score)])


def _fit_bin_curve(predictions: Array, outcomes: Array, weights: Array, n_bins: int, min_bin_size: int) -> tuple[Array, Array, float]:
    pred = _as_finite_vector(predictions, "predictions")
    outcome = _as_finite_vector(outcomes, "outcomes")
    weight = _optional_weights(weights, pred.shape[0], "weights")
    bin_id, edges = _histogram_bin_ids(pred, weight, n_bins=n_bins, min_bin_size=min_bin_size, strategy="quantile")
    bins = int(np.max(bin_id)) + 1
    fallback = _weighted_mean(outcome, weight)
    means = np.full(bins, fallback, dtype=np.float64)
    for idx in range(bins):
        mask = bin_id == idx
        if np.any(mask):
            means[idx] = _weighted_mean(outcome[mask], weight[mask])
    return edges, means, fallback


def _predict_bin_curve(predictions: Array, edges: Array, means: Array, fallback: float) -> Array:
    pred = _as_finite_vector(predictions, "predictions")
    ids = np.clip(np.searchsorted(edges[1:-1], pred, side="right"), 0, means.shape[0] - 1)
    out = means[ids].astype(np.float64, copy=True)
    out[~np.isfinite(out)] = float(fallback)
    return out


def _histogram_bin_ids(
    predictions: Array,
    weights: Array,
    *,
    n_bins: int,
    min_bin_size: int,
    strategy: str,
) -> tuple[Array, Array]:
    pred = _as_finite_vector(predictions, "predictions")
    bins = min(max(int(n_bins), 1), pred.shape[0], max(1, pred.shape[0] // max(int(min_bin_size), 1)))
    if bins == 1:
        return np.zeros(pred.shape[0], dtype=np.int64), np.array([-np.inf, np.inf], dtype=np.float64)
    if strategy == "equal_width":
        if float(np.max(pred)) <= float(np.min(pred)) + 1e-12:
            edges = np.array([-np.inf, np.inf], dtype=np.float64)
            return np.zeros(pred.shape[0], dtype=np.int64), edges
        edges = np.linspace(float(np.min(pred)), float(np.max(pred)), bins + 1)
    else:
        edges = _weighted_quantile(pred, weights, np.linspace(0.0, 1.0, bins + 1))
    edges = _sanitize_edges(edges, pred)
    bin_id = np.searchsorted(edges[1:-1], pred, side="right").astype(np.int64)
    return bin_id, edges


def _isotonic_bin_ids(predictions: Array, targets: Array, weights: Array, *, n_bins: int, min_bin_size: int) -> tuple[Array, Array]:
    pred = _as_finite_vector(predictions, "predictions")
    target = _as_finite_vector(targets, "targets")
    weight = _optional_weights(weights, pred.shape[0], "weights")
    order = np.argsort(pred, kind="mergesort")
    sorted_pred = pred[order]
    sorted_target = target[order]
    sorted_weight = weight[order]
    blocks = _pava_blocks(sorted_target, sorted_weight)
    max_bins = min(max(int(n_bins), 1), pred.shape[0], max(1, pred.shape[0] // max(int(min_bin_size), 1)))
    if len(blocks) > max_bins:
        block_score = np.asarray([block["value"] for block in blocks], dtype=np.float64)
        block_weight = np.asarray([block["weight"] for block in blocks], dtype=np.float64)
        block_bin, _ = _histogram_bin_ids(block_score, block_weight, n_bins=max_bins, min_bin_size=1, strategy="quantile")
        merged: list[dict[str, Any]] = []
        for idx in range(int(np.max(block_bin)) + 1):
            members = [block for keep, block in zip(block_bin == idx, blocks) if keep]
            merged.append({"start": members[0]["start"], "end": members[-1]["end"]})
        blocks = merged
    sorted_bin = np.empty(pred.shape[0], dtype=np.int64)
    for idx, block in enumerate(blocks):
        sorted_bin[int(block["start"]) : int(block["end"])] = idx
    bin_id = np.empty_like(sorted_bin)
    bin_id[order] = sorted_bin
    edges = _edges_from_sorted_blocks(sorted_pred, blocks)
    return bin_id, edges


def _pava_blocks(sorted_targets: Array, sorted_weights: Array) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for idx, (target, weight) in enumerate(zip(sorted_targets, sorted_weights)):
        w = max(float(weight), 1e-12)
        blocks.append({"start": idx, "end": idx + 1, "weight": w, "sum": w * float(target), "value": float(target)})
        while len(blocks) >= 2 and float(blocks[-2]["value"]) > float(blocks[-1]["value"]):
            right = blocks.pop()
            left = blocks.pop()
            merged_weight = float(left["weight"]) + float(right["weight"])
            merged_sum = float(left["sum"]) + float(right["sum"])
            blocks.append(
                {
                    "start": int(left["start"]),
                    "end": int(right["end"]),
                    "weight": merged_weight,
                    "sum": merged_sum,
                    "value": merged_sum / max(merged_weight, 1e-12),
                }
            )
    return blocks


def _edges_from_sorted_blocks(sorted_predictions: Array, blocks: list[dict[str, Any]]) -> Array:
    edges = [-np.inf]
    for left, right in zip(blocks[:-1], blocks[1:]):
        left_value = float(sorted_predictions[int(left["end"]) - 1])
        right_value = float(sorted_predictions[int(right["start"])])
        edges.append(0.5 * (left_value + right_value))
    edges.append(np.inf)
    return np.asarray(edges, dtype=np.float64)


def _sanitize_edges(edges: Array, predictions: Array) -> Array:
    finite_edges = np.asarray(edges, dtype=np.float64)
    finite_edges = np.maximum.accumulate(finite_edges)
    finite_edges = np.unique(finite_edges)
    if finite_edges.size < 2:
        return np.array([-np.inf, np.inf], dtype=np.float64)
    finite_edges[0] = -np.inf
    finite_edges[-1] = np.inf
    if finite_edges.size < 3 and float(np.max(predictions)) > float(np.min(predictions)) + 1e-12:
        mid = 0.5 * (float(np.min(predictions)) + float(np.max(predictions)))
        return np.array([-np.inf, mid, np.inf], dtype=np.float64)
    return finite_edges


def _weighted_quantile(values: Array, weights: Array, quantiles: Array) -> Array:
    values = _as_finite_vector(values, "values")
    weights = _optional_weights(weights, values.shape[0], "weights")
    q = np.asarray(quantiles, dtype=np.float64)
    if float(np.sum(weights)) <= 0.0:
        return np.quantile(values, q)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    sorted_weights = weights[order]
    cdf = np.cumsum(sorted_weights) / max(float(np.sum(sorted_weights)), 1e-12)
    return np.interp(q, cdf, sorted_values, left=sorted_values[0], right=sorted_values[-1])


def _predict_q_next_average(
    model: Any,
    next_states: Array,
    next_actions: Array,
    *,
    next_action_weights: Array | None = None,
) -> Array:
    states = np.asarray(next_states, dtype=np.float64)
    actions = np.asarray(next_actions, dtype=np.float64)
    if actions.ndim == 2:
        if next_action_weights is not None:
            validate_action_weights(
                next_action_weights,
                n_rows=actions.shape[0],
                n_actions=1,
                name="next_action_weights",
            )
        return np.asarray(model.predict_q(states, actions), dtype=np.float64).reshape(-1)
    if actions.ndim != 3:
        raise ValueError("next_actions must have shape (n, action_dim) or (n, n_action_samples, action_dim).")
    n, n_samples, _ = actions.shape
    if states.shape[0] != n:
        raise ValueError("next_states and next_actions must have the same number of rows.")
    repeated_states = np.repeat(states, int(n_samples), axis=0)
    flat_actions = actions.reshape(n * int(n_samples), -1)
    pred = np.asarray(model.predict_q(repeated_states, flat_actions), dtype=np.float64).reshape(n, int(n_samples))
    weights = validate_action_weights(
        next_action_weights,
        n_rows=n,
        n_actions=int(n_samples),
        name="next_action_weights",
    )
    return weighted_action_expectation(pred, weights)


def _validated_transition_vectors(
    predictions: Array,
    next_predictions: Array,
    rewards: Array,
    *,
    terminals: Array | None,
    timeouts: Array | None,
    continuation: Array | None,
    sample_weight: Array | None,
) -> tuple[Array, Array, Array, Array, Array]:
    pred = _as_finite_vector(predictions, "predictions")
    next_pred = _as_finite_vector(next_predictions, "next_predictions")
    rew = _as_finite_vector(rewards, "rewards")
    if pred.shape != next_pred.shape or pred.shape != rew.shape:
        raise ValueError("predictions, next_predictions, and rewards must have the same length.")
    bootstrap = validate_bootstrap_inputs(
        n_rows=pred.shape[0],
        terminals=terminals,
        timeouts=timeouts,
        continuation=continuation,
    )
    term = bootstrap.terminals
    weight = _optional_weights(sample_weight, pred.shape[0], "sample_weight")
    return pred, next_pred, rew, term, weight


def _optional_weights(weights: Array | None, n: int, name: str) -> Array:
    if weights is None:
        return np.ones(int(n), dtype=np.float64)
    arr = _as_finite_vector(weights, name)
    if arr.shape[0] != int(n):
        raise ValueError(f"{name} must have {int(n)} rows.")
    if np.any(arr < 0.0):
        raise ValueError(f"{name} must be nonnegative.")
    if float(np.sum(arr)) <= 0.0:
        raise ValueError(f"{name} must have positive total weight.")
    return arr.astype(np.float64, copy=False)


def _as_finite_vector(value: Array, name: str) -> Array:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} must be nonempty.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values.")
    return arr


def _validate_gamma(gamma: float) -> float:
    value = float(gamma)
    if not (0.0 <= value < 1.0):
        raise ValueError("gamma must be in [0, 1).")
    return value


def _validate_method(method: str) -> CalibrationMethod:
    valid = {
        "histogram_constant",
        "histogram_rescale",
        "isotonic_histogram_constant",
        "isotonic_histogram_rescale",
    }
    if method not in valid:
        raise ValueError(
            "method must be one of 'histogram_constant', 'histogram_rescale', "
            "'isotonic_histogram_constant', or 'isotonic_histogram_rescale'."
        )
    return method  # type: ignore[return-value]


def _validate_positive_int(value: int, name: str) -> int:
    out = int(value)
    if out <= 0:
        raise ValueError(f"{name} must be positive.")
    return out


def _weighted_mean(values: Array, weights: Array | None) -> float:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return float("nan")
    if weights is None:
        return float(np.mean(arr))
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    total = float(np.sum(w))
    if total <= 0.0:
        return float(np.mean(arr))
    return float(np.sum(w * arr) / total)


def _kfold_indices(n: int, n_folds: int, seed: int) -> list[tuple[Array, Array]]:
    rng = np.random.default_rng(int(seed))
    indices = np.arange(int(n), dtype=np.int64)
    rng.shuffle(indices)
    folds = np.array_split(indices, int(n_folds))
    out = []
    for fold in folds:
        hold = np.asarray(fold, dtype=np.int64)
        train = np.setdiff1d(indices, hold, assume_unique=False)
        if train.size and hold.size:
            out.append((train, hold))
    return out


def _safe_relative_drop(before: float, after: float) -> float:
    if not np.isfinite(before) or not np.isfinite(after):
        return float("nan")
    denom = max(abs(float(before)), 1e-12)
    return float((before - after) / denom)


def _safe_relative_increase(before: float, after: float) -> float:
    if not np.isfinite(before) or not np.isfinite(after):
        return float("nan")
    denom = max(abs(float(before)), 1e-12)
    return float((after - before) / denom)


def _safe_absolute_drop(before: float, after: float) -> float:
    if not np.isfinite(before) or not np.isfinite(after):
        return float("nan")
    return float(before - after)


def _finite_or_neg_inf(value: float) -> float:
    return float(value) if np.isfinite(value) else float("-inf")
