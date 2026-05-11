from __future__ import annotations

from typing import Any

import numpy as np

from fqe_benchmark.types import BenchmarkDataset, FittedEstimator


Array = np.ndarray


def evaluate_fitted_estimator(dataset: BenchmarkDataset, fitted: FittedEstimator) -> dict[str, Any]:
    """Evaluate a fitted estimator on the dataset's fixed evaluation samples."""
    q_target = fitted.predict_q(dataset.target_eval_states, dataset.target_eval_actions)
    q_behavior = fitted.predict_q(dataset.behavior_eval_states, dataset.behavior_eval_actions)
    value_estimate = fitted.estimate_policy_value(dataset.initial_states, dataset.initial_actions)
    row: dict[str, Any] = {
        "policy_value_estimate": float(value_estimate),
        "runtime_sec": float(fitted.runtime_sec),
        "tuning_runtime_sec": float(fitted.tuning_runtime_sec),
    }
    if dataset.true_q_fn is not None:
        true_target = dataset.true_q_fn(dataset.target_eval_states, dataset.target_eval_actions)
        true_behavior = dataset.true_q_fn(dataset.behavior_eval_states, dataset.behavior_eval_actions)
        true_initial = dataset.true_q_fn(dataset.initial_states, dataset.initial_actions)
        row.update(
            {
                "target_q_mse": float(np.mean((q_target - true_target) ** 2)),
                "behavior_q_mse": float(np.mean((q_behavior - true_behavior) ** 2)),
                "initial_v_mse": float(np.mean((fitted.predict_q(dataset.initial_states, dataset.initial_actions) - true_initial) ** 2)),
            }
        )
    if dataset.true_policy_value is not None:
        error = float(value_estimate - dataset.true_policy_value)
        row.update(
            {
                "policy_value_true": float(dataset.true_policy_value),
                "policy_value_error": error,
                "policy_value_absolute_error": abs(error),
                "policy_value_squared_error": error * error,
            }
        )
    row.update(_bellman_residual_metrics(dataset, fitted))
    return row


def _bellman_residual_metrics(dataset: BenchmarkDataset, fitted: FittedEstimator) -> dict[str, float]:
    train_current = fitted.predict_q(dataset.states, dataset.actions)
    train_next = fitted.predict_q(dataset.next_states, _flatten_next_actions(dataset.next_actions))
    target = np.asarray(dataset.rewards, dtype=np.float64).reshape(-1) + dataset.gamma * (
        1.0 - np.asarray(dataset.terminals, dtype=np.float64).reshape(-1)
    ) * train_next
    behavior_residual = train_current - target
    # Target residual is approximate because only transition samples from the
    # training batch are available in this generic wrapper.
    target_current = fitted.predict_q(dataset.target_eval_states, dataset.target_eval_actions)
    target_proxy = dataset.true_q_fn(dataset.target_eval_states, dataset.target_eval_actions) if dataset.true_q_fn else target_current
    return {
        "behavior_bellman_residual_mse": float(np.mean(behavior_residual**2)),
        "target_bellman_residual_mse": float(np.mean((target_current - target_proxy) ** 2)),
    }


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    numeric_keys = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if key not in {"seed", "sample_size"} and _is_number(value)
        }
    )
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    group_keys = ("stage", "dataset", "domain", "estimator", "gamma", "sample_size", "policy_shift")
    for row in rows:
        if row.get("status") != "ok":
            continue
        key = tuple(row.get(group_key, "") for group_key in group_keys)
        groups.setdefault(key, []).append(row)
    summary: list[dict[str, Any]] = []
    for key, group in sorted(groups.items(), key=lambda item: item[0]):
        out = {group_key: value for group_key, value in zip(group_keys, key)}
        out["n_runs"] = len(group)
        for metric in numeric_keys:
            values = np.asarray([float(row[metric]) for row in group if _is_number(row.get(metric))], dtype=np.float64)
            if values.size:
                out[f"{metric}_mean"] = float(np.mean(values))
                out[f"{metric}_std"] = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
        summary.append(out)
    return summary


def _flatten_next_actions(next_actions: Array) -> Array:
    arr = np.asarray(next_actions, dtype=np.float64)
    return arr[:, 0, :] if arr.ndim == 3 else arr


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(float(value))
