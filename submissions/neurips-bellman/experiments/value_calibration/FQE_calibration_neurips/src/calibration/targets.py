from __future__ import annotations

import numpy as np

from ..data import TransitionBatch
from ..policies import SoftmaxPolicy


def raw_predictions(model: object, batch: TransitionBatch) -> np.ndarray:
    return model.predict_q(batch.states, batch.actions)


def bellman_targets(model: object, batch: TransitionBatch, gamma: float) -> np.ndarray:
    next_q = model.predict_q(batch.next_states, batch.next_actions)
    return np.asarray(batch.rewards, dtype=float) + float(gamma) * next_q


def calibration_xy(model: object, batch: TransitionBatch, gamma: float, target_type: str) -> tuple[np.ndarray, np.ndarray]:
    pred = raw_predictions(model, batch)
    bellman = bellman_targets(model, batch, gamma)
    if target_type in {"bellman_target", "q_value"}:
        return pred, bellman
    if target_type == "td_residual":
        return pred, bellman - pred
    if target_type == "final_value":
        return pred, bellman
    raise ValueError(f"Unknown calibration target '{target_type}'.")


def apply_calibration(prediction: np.ndarray, calibrator: object, target_type: str) -> np.ndarray:
    prediction = np.asarray(prediction, dtype=float)
    calibrated = calibrator.predict(prediction)
    if target_type == "td_residual":
        return prediction + calibrated
    return calibrated


def policy_value_predictions(model: object, states: np.ndarray, target_policy: SoftmaxPolicy) -> np.ndarray:
    """Average a fitted Q baseline over the target policy to get V_hat(S)."""
    probs = target_policy.action_probabilities(states)
    q_cols = []
    for action in range(probs.shape[1]):
        actions = np.full(states.shape[0], action, dtype=int)
        q_cols.append(model.predict_q(states, actions))
    return np.sum(probs * np.column_stack(q_cols), axis=1).astype(float)


def value_calibration_arrays(
    model: object,
    batch: TransitionBatch,
    target_policy: SoftmaxPolicy,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        policy_value_predictions(model, batch.states, target_policy),
        policy_value_predictions(model, batch.next_states, target_policy),
        np.asarray(batch.rewards, dtype=float),
    )


def action_importance_weights(
    batch: TransitionBatch,
    clip: float = 20.0,
    normalize: bool = True,
) -> np.ndarray:
    weights = np.asarray(batch.target_probs, dtype=float) / np.maximum(np.asarray(batch.behavior_probs, dtype=float), 1e-12)
    clip = float(clip)
    if np.isfinite(clip) and clip > 0:
        weights = np.minimum(weights, clip)
    weights = np.where(np.isfinite(weights) & (weights >= 0), weights, 0.0)
    if normalize:
        mean = float(np.mean(weights))
        if mean > 0:
            weights = weights / mean
    return weights.astype(float)


def importance_weight_diagnostics(weights: np.ndarray, clip: float, normalize: bool) -> dict[str, float | str]:
    w = np.asarray(weights, dtype=float)
    total_sq = float(np.sum(w**2))
    ess = float((np.sum(w) ** 2) / max(total_sq, 1e-12) / max(w.size, 1))
    return {
        "calibration_object": "value",
        "calibration_weight_scheme": "action_ratio",
        "importance_weight_clip": float(clip),
        "importance_weight_normalized": float(bool(normalize)),
        "importance_weight_ess": ess,
        "importance_weight_mean": float(np.mean(w)) if w.size else float("nan"),
        "importance_weight_max": float(np.max(w)) if w.size else float("nan"),
    }
