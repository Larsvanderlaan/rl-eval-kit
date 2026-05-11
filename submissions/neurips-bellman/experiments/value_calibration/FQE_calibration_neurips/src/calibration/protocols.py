from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..data import TransitionBatch
from ..estimators.baselines import fit_estimator
from ..evaluation import (
    bellman_calibration_error_50bin,
    bellman_calibration_error_from_arrays,
    bellman_outcome_mse,
    bellman_residual,
    bootstrap_value_interval,
    coverage_stratified_errors,
    diagnostic_warning,
    estimate_policy_value_at_states,
    make_result_row,
    true_q_function_mse,
    true_v_function_mse,
    value_prediction_diagnostics,
)
from ..policies import SoftmaxPolicy
from ..utils import kfold_indices, timed, train_calibration_split
from .calibrators import fit_calibrator, fit_iterated_bellman_calibrator, fit_value_bellman_calibrator, is_iterated_bellman_calibrator
from .targets import (
    action_importance_weights,
    calibration_xy,
    importance_weight_diagnostics,
    policy_value_predictions,
    value_calibration_arrays,
)


@dataclass
class ProtocolContext:
    batch: TransitionBatch
    test_batch: TransitionBatch
    initial_states: np.ndarray
    oracle_value: float
    env: object
    target_policy: SoftmaxPolicy
    learner: str
    learner_params: dict[str, Any]
    gamma: float
    seed: int
    coverage: str
    reward_noise: float
    diagnostic_batch: TransitionBatch | None = None
    diagnostic_true_q_values: np.ndarray | None = None
    diagnostic_true_v_values: np.ndarray | None = None
    calibration_error_bins: int = 50
    calibration_error_min_bin_size: int = 20
    calibration_error_folds: int = 5
    importance_weight_scheme: str = "action_ratio"
    importance_weight_clip: float = 20.0
    normalize_importance_weights: bool = True
    value_calibration_iterations: int = 4
    interval_bootstrap_reps: int = 200


class MedianFoldValueModel:
    """Pointwise-median aggregation of fold-trained value predictors.

    Strict cross-calibration evaluates new states with only the fold-trained
    first-stage learners used to build out-of-fold calibration pairs. For
    value-space calibration, the causal-calibration-style aggregation is the
    pointwise median of calibrated fold values.
    """

    def __init__(self, models: list[object]):
        if not models:
            raise ValueError("MedianFoldValueModel requires at least one fold model.")
        self.models = list(models)
        self.diagnostics = self._combine_diagnostics(self.models)
        self._value_cache: dict[tuple[int, tuple[int, ...]], np.ndarray] = {}

    @staticmethod
    def _combine_diagnostics(models: list[object]) -> dict[str, float | str]:
        out: dict[str, float | str] = {
            "cross_aggregation": "pointwise_median",
            "cross_n_fold_models": float(len(models)),
            "cross_final_refit": 0.0,
        }
        keys = sorted({key for model in models for key in dict(getattr(model, "diagnostics", {}))})
        for key in keys:
            vals = [dict(getattr(model, "diagnostics", {})).get(key) for model in models]
            numeric = []
            for value in vals:
                try:
                    numeric.append(float(value))
                except (TypeError, ValueError):
                    continue
            if numeric:
                out[key] = float(np.nanmedian(np.asarray(numeric, dtype=float)))
            elif vals:
                out[key] = str(vals[0])
        return out

    def predict_q(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        q_values = [np.asarray(model.predict_q(states, actions), dtype=float) for model in self.models]
        return np.nanmedian(np.vstack(q_values), axis=0)

    def predict_value(
        self,
        states: np.ndarray,
        target_policy: SoftmaxPolicy,
        calibrator: object | None = None,
        calibration_target: str = "value_bellman",
    ) -> np.ndarray:
        states_arr = np.asarray(states, dtype=float)
        cache_key = (id(states), tuple(states_arr.shape))
        raw = self._value_cache.get(cache_key)
        if raw is None:
            raw = np.vstack(
                [
                    np.asarray(policy_value_predictions(model, states_arr, target_policy), dtype=float)
                    for model in self.models
                ]
            )
            self._value_cache[cache_key] = raw
        if calibrator is not None and calibration_target == "value_bellman":
            calibrated = np.vstack([np.asarray(calibrator.predict(row), dtype=float) for row in raw])
            return np.nanmedian(calibrated, axis=0)
        return np.nanmedian(raw, axis=0)


def _one_shot_calibrator_params(params: dict[str, Any]) -> dict[str, Any]:
    allowed = {"n_bins", "bin_strategy", "min_bin_size"}
    return {key: value for key, value in dict(params or {}).items() if key in allowed}


def _iterated_calibrator_params(params: dict[str, Any]) -> dict[str, Any]:
    out = _one_shot_calibrator_params(params)
    for source, target in [("n_iterations", "n_iterations"), ("bellman_iterations", "n_iterations")]:
        if source in params:
            out[target] = params[source]
    return out


def _fit_protocol_calibrator(
    calibrator_name: str,
    model: object,
    batch: TransitionBatch,
    ctx: ProtocolContext,
    calibration_target: str,
    calibrator_params: dict[str, Any],
) -> object:
    if calibration_target == "value_bellman":
        values, next_values, rewards = value_calibration_arrays(model, batch, ctx.target_policy)
        weights = action_importance_weights(batch, ctx.importance_weight_clip, ctx.normalize_importance_weights)
        cal = fit_value_bellman_calibrator(
            calibrator_name,
            values,
            next_values,
            rewards,
            ctx.gamma,
            weights,
            n_iterations=int(
                calibrator_params.get(
                    "value_calibration_iterations",
                    calibrator_params.get("bellman_iterations", ctx.value_calibration_iterations),
                )
            ),
            **_one_shot_calibrator_params(calibrator_params),
        )
        cal.diagnostics.update(importance_weight_diagnostics(weights, ctx.importance_weight_clip, ctx.normalize_importance_weights))
        return cal
    if is_iterated_bellman_calibrator(calibrator_name):
        return fit_iterated_bellman_calibrator(
            calibrator_name,
            model.predict_q(batch.states, batch.actions),
            model.predict_q(batch.next_states, batch.next_actions),
            batch.rewards,
            ctx.gamma,
            calibration_target,
            **_iterated_calibrator_params(calibrator_params),
        )
    pred, target = calibration_xy(model, batch, ctx.gamma, calibration_target)
    return fit_calibrator(calibrator_name, pred, target, **_one_shot_calibrator_params(calibrator_params))


def _fit_array_calibrator(
    calibrator_name: str,
    predictions: np.ndarray,
    targets: np.ndarray,
    calibrator_params: dict[str, Any],
    *,
    next_predictions: np.ndarray | None = None,
    rewards: np.ndarray | None = None,
    weights: np.ndarray | None = None,
    gamma: float | None = None,
    calibration_target: str = "value_bellman",
) -> object:
    if calibration_target == "value_bellman":
        if next_predictions is None or rewards is None or gamma is None:
            raise ValueError("Value-space Bellman calibration requires values, next values, rewards, and gamma.")
        return fit_value_bellman_calibrator(
            calibrator_name,
            predictions,
            next_predictions,
            rewards,
            gamma,
            weights,
            n_iterations=int(calibrator_params.get("value_calibration_iterations", calibrator_params.get("bellman_iterations", 4))),
            **_one_shot_calibrator_params(calibrator_params),
        )
    if is_iterated_bellman_calibrator(calibrator_name):
        if next_predictions is None or rewards is None or gamma is None:
            raise ValueError("Iterated Bellman calibration requires next predictions, rewards, and gamma.")
        return fit_iterated_bellman_calibrator(
            calibrator_name,
            predictions,
            next_predictions,
            rewards,
            gamma,
            calibration_target,
            **_iterated_calibrator_params(calibrator_params),
        )
    return fit_calibrator(calibrator_name, predictions, targets, **_one_shot_calibrator_params(calibrator_params))


def _evaluate_row(ctx: ProtocolContext, model: object, calibrator: object | None, meta: dict[str, Any], runtime: float) -> dict[str, Any]:
    target_type = meta["calibration_target"]
    diagnostic_batch = ctx.diagnostic_batch if ctx.diagnostic_batch is not None else ctx.test_batch
    diagnostic_weights = action_importance_weights(
        diagnostic_batch, ctx.importance_weight_clip, ctx.normalize_importance_weights
    )
    initial_value_predictions = estimate_policy_value_at_states(
        model, ctx.initial_states, ctx.target_policy, calibrator, target_type
    )
    value = float(np.mean(initial_value_predictions))
    interval_diag = bootstrap_value_interval(
        initial_value_predictions,
        ctx.oracle_value,
        seed=ctx.seed + 104729,
        n_bootstrap=ctx.interval_bootstrap_reps,
    )
    test_weights = action_importance_weights(ctx.test_batch, ctx.importance_weight_clip, ctx.normalize_importance_weights)
    if target_type == "value_bellman":
        raw_diag_pred = estimate_policy_value_at_states(model, diagnostic_batch.states, ctx.target_policy, None, target_type)
        test_pred = estimate_policy_value_at_states(model, ctx.test_batch.states, ctx.target_policy, calibrator, target_type)
        test_next = estimate_policy_value_at_states(model, ctx.test_batch.next_states, ctx.target_policy, calibrator, target_type)
        bellman_value = float(
            np.sum(test_weights * (test_pred - (ctx.test_batch.rewards + ctx.gamma * test_next)) ** 2)
            / max(float(np.sum(test_weights)), 1e-12)
        )
        # Reuse the large diagnostic predictions across all value-space metrics.
        diag_pred = estimate_policy_value_at_states(model, diagnostic_batch.states, ctx.target_policy, calibrator, target_type)
        diag_next = estimate_policy_value_at_states(model, diagnostic_batch.next_states, ctx.target_policy, calibrator, target_type)
        diag_outcome = ctx.env.expected_reward(diagnostic_batch.states, diagnostic_batch.actions) + ctx.gamma * diag_next
        bellman_outcome_mse_value = float(
            np.sum(diagnostic_weights * (diag_pred - diag_outcome) ** 2) / max(float(np.sum(diagnostic_weights)), 1e-12)
        )
        if ctx.diagnostic_true_v_values is None:
            true_v_mse_value = float("nan")
        else:
            true_v_mse_value = float(np.mean((diag_pred - np.asarray(ctx.diagnostic_true_v_values, dtype=float)) ** 2))
        true_q_mse_value = true_q_function_mse(
            model, diagnostic_batch, ctx.diagnostic_true_q_values, calibrator, target_type
        )
        bellman_cal = bellman_calibration_error_from_arrays(
            diag_pred,
            diag_outcome,
            n_bins=ctx.calibration_error_bins,
            min_bin_size=ctx.calibration_error_min_bin_size,
            n_folds=ctx.calibration_error_folds,
            weights=diagnostic_weights,
        )
    else:
        raw_diag_pred = estimate_policy_value_at_states(model, diagnostic_batch.states, ctx.target_policy, None, "value_bellman")
        bellman_value = bellman_residual(
            model, ctx.test_batch, ctx.gamma, calibrator, target_type, ctx.target_policy, test_weights
        )
        bellman_outcome_mse_value = bellman_outcome_mse(
            model, diagnostic_batch, ctx.env, ctx.target_policy, ctx.gamma, calibrator, target_type, diagnostic_weights
        )
        true_v_mse_value = true_v_function_mse(
            model, diagnostic_batch.states, ctx.diagnostic_true_v_values, ctx.target_policy, calibrator, target_type
        )
        true_q_mse_value = true_q_function_mse(
            model, diagnostic_batch, ctx.diagnostic_true_q_values, calibrator, target_type
        )
        bellman_cal = bellman_calibration_error_50bin(
            model,
            diagnostic_batch,
            ctx.env,
            ctx.target_policy,
            ctx.gamma,
            calibrator,
            target_type,
            n_bins=ctx.calibration_error_bins,
            min_bin_size=ctx.calibration_error_min_bin_size,
            n_folds=ctx.calibration_error_folds,
            weights=diagnostic_weights,
        )
    cal_error = float(bellman_cal["bellman_calibration_error"])
    warn_flag, warn_msg = diagnostic_warning(
        model,
        ctx.test_batch,
        ctx.initial_states,
        ctx.target_policy,
        value,
        ctx.oracle_value,
        cal_error,
        true_v_mse_value,
        true_q_mse_value,
        bellman_outcome_mse_value,
    )
    coverage_extra = coverage_stratified_errors(
        model,
        ctx.test_batch,
        ctx.gamma,
        calibrator,
        target_type,
        target_policy=ctx.target_policy,
        env=ctx.env,
    )
    model_diagnostics = dict(getattr(model, "diagnostics", {}))
    calibrator_diagnostics = dict(getattr(calibrator, "diagnostics", {})) if calibrator is not None else {}
    diag_failure = bool(float(model_diagnostics.get("saddle_nan_flag", 0.0) or 0.0)) or bool(
        float(model_diagnostics.get("saddle_exploding_flag", 0.0) or 0.0)
    )
    if diag_failure:
        warn_flag = True
        warning = str(model_diagnostics.get("failure_reason", "saddle_instability")).strip() or "saddle_instability"
        warn_msg = (warn_msg + ";" + warning).strip(";") if warn_msg else warning
    extra = dict(coverage_extra)
    value_diag = value_prediction_diagnostics(
        diag_pred if target_type == "value_bellman" else estimate_policy_value_at_states(
            model, diagnostic_batch.states, ctx.target_policy, calibrator, target_type
        ),
        ctx.diagnostic_true_v_values,
    )
    raw_value_diag = value_prediction_diagnostics(raw_diag_pred, ctx.diagnostic_true_v_values)
    for key, value_extra in value_diag.items():
        extra[key] = value_extra
    for key, value_extra in raw_value_diag.items():
        extra[f"raw_{key}"] = value_extra
    extra.update(importance_weight_diagnostics(diagnostic_weights, ctx.importance_weight_clip, ctx.normalize_importance_weights))
    for key, value_extra in model_diagnostics.items():
        if isinstance(value_extra, (int, float, str, bool)):
            extra[f"model_diag_{key}"] = value_extra
    for key, value_extra in calibrator_diagnostics.items():
        if isinstance(value_extra, (int, float, str, bool)):
            extra[f"calibrator_diag_{key}"] = value_extra
            if key in {
                "calibration_object",
                "calibration_weight_scheme",
                "importance_weight_clip",
                "importance_weight_normalized",
                "importance_weight_ess",
                "importance_weight_mean",
                "importance_weight_max",
            }:
                extra[key] = value_extra
    return make_result_row(
        env=ctx.env,
        seed=ctx.seed,
        sample_size=len(ctx.batch),
        coverage=ctx.coverage,
        reward_noise=ctx.reward_noise,
        learner=ctx.learner,
        calibrated=meta["calibrated"],
        protocol=meta["protocol"],
        calibrator_name=meta["calibrator"],
        calibration_target=target_type,
        all_data=meta["all_data"],
        sample_splitting=meta["sample_splitting"],
        train_fraction=meta["train_fraction"],
        calibration_fraction=meta["calibration_fraction"],
        value_estimate=value,
        oracle_value=ctx.oracle_value,
        bellman_residual_value=bellman_value,
        calibration_error_value=cal_error,
        true_v_mse_value=true_v_mse_value,
        true_q_mse_value=true_q_mse_value,
        bellman_outcome_mse_value=bellman_outcome_mse_value,
        bellman_calibration_plugin_value=float(bellman_cal["bellman_calibration_error_plugin"]),
        bellman_calibration_raw_value=float(bellman_cal["bellman_calibration_error_debiased_raw"]),
        bellman_calibration_bins=int(bellman_cal["bellman_calibration_bins"]),
        bellman_calibration_test_size=int(bellman_cal["bellman_calibration_test_size"]),
        interval_lower_95=float(interval_diag["interval_lower_95"]),
        interval_upper_95=float(interval_diag["interval_upper_95"]),
        interval_length_95=float(interval_diag["interval_length_95"]),
        interval_coverage_95=float(interval_diag["interval_coverage_95"]),
        runtime=runtime,
        failure_flag=warn_flag,
        failure_reason=warn_msg,
        diagnostic_warning_message=warn_msg,
        extra=extra,
    )


def run_uncalibrated_all_data(ctx: ProtocolContext, calibration_target: str) -> dict[str, Any]:
    with timed() as tb:
        result = fit_estimator(ctx.learner, ctx.batch, ctx.env.n_actions, ctx.target_policy, ctx.gamma, ctx.learner_params, ctx.seed)
    return _evaluate_row(ctx, result.model, None, {
        "calibrated": False,
        "protocol": "uncalibrated_all_data",
        "calibrator": "none",
        "calibration_target": calibration_target,
        "all_data": True,
        "sample_splitting": False,
        "train_fraction": 1.0,
        "calibration_fraction": 0.0,
    }, tb.seconds)


def run_same_fraction_uncalibrated(ctx: ProtocolContext, train_fraction: float, calibration_target: str) -> dict[str, Any]:
    train_idx, _ = train_calibration_split(len(ctx.batch), train_fraction, ctx.seed + 17)
    with timed() as tb:
        result = fit_estimator(
            ctx.learner, ctx.batch.subset(train_idx), ctx.env.n_actions, ctx.target_policy, ctx.gamma, ctx.learner_params, ctx.seed
        )
    return _evaluate_row(ctx, result.model, None, {
        "calibrated": False,
        "protocol": "uncalibrated_same_fraction",
        "calibrator": "none",
        "calibration_target": calibration_target,
        "all_data": False,
        "sample_splitting": True,
        "train_fraction": train_fraction,
        "calibration_fraction": 1.0 - train_fraction,
    }, tb.seconds)


def run_no_split(ctx: ProtocolContext, calibrator_name: str, calibration_target: str, calibrator_params: dict[str, Any]) -> dict[str, Any]:
    with timed() as tb:
        result = fit_estimator(ctx.learner, ctx.batch, ctx.env.n_actions, ctx.target_policy, ctx.gamma, ctx.learner_params, ctx.seed)
        cal = _fit_protocol_calibrator(calibrator_name, result.model, ctx.batch, ctx, calibration_target, calibrator_params)
    return _evaluate_row(ctx, result.model, cal, {
        "calibrated": True,
        "protocol": "no_split",
        "calibrator": calibrator_name,
        "calibration_target": calibration_target,
        "all_data": True,
        "sample_splitting": False,
        "train_fraction": 1.0,
        "calibration_fraction": 1.0,
    }, tb.seconds)


def run_split(
    ctx: ProtocolContext,
    calibrator_name: str,
    calibration_target: str,
    train_fraction: float,
    calibrator_params: dict[str, Any],
) -> dict[str, Any]:
    train_idx, cal_idx = train_calibration_split(len(ctx.batch), train_fraction, ctx.seed + 29)
    with timed() as tb:
        result = fit_estimator(
            ctx.learner, ctx.batch.subset(train_idx), ctx.env.n_actions, ctx.target_policy, ctx.gamma, ctx.learner_params, ctx.seed
        )
        cal = _fit_protocol_calibrator(
            calibrator_name,
            result.model,
            ctx.batch.subset(cal_idx),
            ctx,
            calibration_target,
            calibrator_params,
        )
    return _evaluate_row(ctx, result.model, cal, {
        "calibrated": True,
        "protocol": "split",
        "calibrator": calibrator_name,
        "calibration_target": calibration_target,
        "all_data": False,
        "sample_splitting": True,
        "train_fraction": train_fraction,
        "calibration_fraction": 1.0 - train_fraction,
    }, tb.seconds)


def _prepare_cross_calibration(
    ctx: ProtocolContext,
    calibration_target: str,
    n_folds: int,
) -> dict[str, Any]:
    preds = np.zeros(len(ctx.batch), dtype=float)
    targets = np.zeros(len(ctx.batch), dtype=float)
    next_preds = np.zeros(len(ctx.batch), dtype=float)
    rewards = np.zeros(len(ctx.batch), dtype=float)
    weights = np.ones(len(ctx.batch), dtype=float)
    fold_models: list[object] = []
    with timed() as tb:
        for fold_id, (train_idx, hold_idx) in enumerate(kfold_indices(len(ctx.batch), n_folds, ctx.seed + 43)):
            fold = fit_estimator(
                ctx.learner,
                ctx.batch.subset(train_idx),
                ctx.env.n_actions,
                ctx.target_policy,
                ctx.gamma,
                ctx.learner_params,
                ctx.seed + fold_id,
            )
            fold_models.append(fold.model)
            hold_batch = ctx.batch.subset(hold_idx)
            if calibration_target == "value_bellman":
                p, p_next, r = value_calibration_arrays(fold.model, hold_batch, ctx.target_policy)
                preds[hold_idx] = p
                next_preds[hold_idx] = p_next
                rewards[hold_idx] = r
                weights[hold_idx] = action_importance_weights(
                    hold_batch,
                    ctx.importance_weight_clip,
                    normalize=False,
                )
            else:
                p, y = calibration_xy(fold.model, hold_batch, ctx.gamma, calibration_target)
                preds[hold_idx] = p
                targets[hold_idx] = y
                next_preds[hold_idx] = fold.model.predict_q(hold_batch.next_states, hold_batch.next_actions)
                rewards[hold_idx] = hold_batch.rewards
        if calibration_target == "value_bellman" and ctx.normalize_importance_weights:
            mean_weight = float(np.mean(weights))
            if mean_weight > 0:
                weights = weights / mean_weight
        median_model = MedianFoldValueModel(fold_models)
    return {
        "preds": preds,
        "targets": targets,
        "next_preds": next_preds,
        "rewards": rewards,
        "weights": weights,
        "median_model": median_model,
        "prepare_seconds": tb.seconds,
        "n_folds": max(int(n_folds), 2),
    }


def _fit_cross_row_from_prepared(
    ctx: ProtocolContext,
    prepared: dict[str, Any],
    calibrator_name: str,
    calibration_target: str,
    calibrator_params: dict[str, Any],
) -> dict[str, Any]:
    with timed() as tb:
        cal = _fit_array_calibrator(
            calibrator_name,
            prepared["preds"],
            prepared["targets"],
            calibrator_params,
            next_predictions=prepared["next_preds"],
            rewards=prepared["rewards"],
            weights=prepared["weights"],
            gamma=ctx.gamma,
            calibration_target=calibration_target,
        )
        if calibration_target == "value_bellman":
            cal.diagnostics.update(
                importance_weight_diagnostics(
                    prepared["weights"], ctx.importance_weight_clip, ctx.normalize_importance_weights
                )
            )
            cal.diagnostics["cross_aggregation"] = "pointwise_median"
            cal.diagnostics["cross_final_refit"] = 0.0
    return _evaluate_row(ctx, prepared["median_model"], cal, {
        "calibrated": True,
        "protocol": "cross",
        "calibrator": calibrator_name,
        "calibration_target": calibration_target,
        "all_data": False,
        "sample_splitting": True,
        "train_fraction": float((int(prepared["n_folds"]) - 1) / int(prepared["n_folds"])),
        "calibration_fraction": 1.0,
    }, float(prepared["prepare_seconds"]) + tb.seconds)


def run_cross_calibration(
    ctx: ProtocolContext,
    calibrator_name: str,
    calibration_target: str,
    n_folds: int,
    calibrator_params: dict[str, Any],
) -> dict[str, Any]:
    prepared = _prepare_cross_calibration(ctx, calibration_target, n_folds)
    return _fit_cross_row_from_prepared(ctx, prepared, calibrator_name, calibration_target, calibrator_params)


def run_cross_calibrations(
    ctx: ProtocolContext,
    calibrator_names: list[str],
    calibration_target: str,
    n_folds: int,
    calibrator_params: dict[str, Any],
) -> list[dict[str, Any]]:
    prepared = _prepare_cross_calibration(ctx, calibration_target, n_folds)
    return [
        _fit_cross_row_from_prepared(ctx, prepared, calibrator_name, calibration_target, calibrator_params)
        for calibrator_name in calibrator_names
    ]
