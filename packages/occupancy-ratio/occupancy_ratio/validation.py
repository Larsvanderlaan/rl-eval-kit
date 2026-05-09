"""Shared boosted estimator input validation and mode-resolution helpers."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np


Array = np.ndarray

__all__ = [
    "_as_2d",
    "_optional_binary_vector",
    "_postprocess_known_action_ratios",
    "_prepare_target_action_samples",
    "_resolve_continuation",
    "_resolve_initial_ratio_mode",
    "_resolve_known_action_ratio_inputs",
    "_resolve_one_step_ratio_mode",
    "_validate_aligned_inputs",
    "_validate_base_transition_inputs",
    "_validate_initial_action_inputs",
    "_validate_initial_state_inputs",
    "_validate_next_target_actions",
    "_validate_occupancy_stabilization_config",
    "_validate_ratio_prediction_config",
    "_validate_target_action_rows",
]


def _validate_ratio_prediction_config(
    *,
    prediction_max: Optional[float],
    prediction_power: float,
    moment_calibration: str = "none",
    crossfit_folds: int = 1,
    density_ratio_loss: str = "lsif",
    logistic_logit_clip: Optional[float] = 20.0,
) -> None:
    if prediction_max is not None and prediction_max <= 0.0:
        raise ValueError("prediction_max must be positive when supplied.")
    if not (0.0 < float(prediction_power) <= 1.0):
        raise ValueError("prediction_power must be in (0, 1].")
    if str(moment_calibration) not in {"none", "scalar"}:
        raise ValueError("moment_calibration must be 'none' or 'scalar'.")
    if int(crossfit_folds) < 1:
        raise ValueError("crossfit_folds must be >= 1.")
    if str(density_ratio_loss).strip().lower() not in {"lsif", "logistic"}:
        raise ValueError("density_ratio_loss must be 'lsif' or 'logistic'.")
    if logistic_logit_clip is not None and float(logistic_logit_clip) <= 0.0:
        raise ValueError("logistic_logit_clip must be positive when supplied.")


def _validate_occupancy_stabilization_config(
    *,
    fixed_point_damping: float,
    occupancy_ratio_max: Optional[float],
    occupancy_projection_eps: float,
    pseudo_outcome_max: Optional[float],
    pseudo_outcome_upper_quantile: float,
    pseudo_outcome_min: float,
    transition_cache_norm_eps: float,
    occupancy_sample_weight_mode: str,
    occupancy_sample_weight_max: Optional[float],
    fixed_point_tol: Optional[float],
    fixed_point_patience: int,
    min_outer_iterations: int,
) -> None:
    if not (0.0 < float(fixed_point_damping) <= 1.0):
        raise ValueError("fixed_point_damping must be in (0, 1].")
    if occupancy_ratio_max is not None and occupancy_ratio_max <= 0.0:
        raise ValueError("occupancy_ratio_max must be positive when supplied.")
    if occupancy_projection_eps <= 0.0:
        raise ValueError("occupancy_projection_eps must be positive.")
    if pseudo_outcome_max is not None and pseudo_outcome_max <= 0.0:
        raise ValueError("pseudo_outcome_max must be positive when supplied.")
    if not (0.0 < float(pseudo_outcome_upper_quantile) < 1.0):
        raise ValueError("pseudo_outcome_upper_quantile must be in (0, 1).")
    if pseudo_outcome_min < 0.0:
        raise ValueError("pseudo_outcome_min must be nonnegative.")
    if transition_cache_norm_eps <= 0.0:
        raise ValueError("transition_cache_norm_eps must be positive.")
    allowed_weight_modes = {"uniform", "sqrt_action_ratio", "action_ratio", "sqrt_target", "target"}
    if str(occupancy_sample_weight_mode) not in allowed_weight_modes:
        raise ValueError(f"occupancy_sample_weight_mode must be one of {sorted(allowed_weight_modes)}.")
    if occupancy_sample_weight_max is not None and occupancy_sample_weight_max <= 0.0:
        raise ValueError("occupancy_sample_weight_max must be positive when supplied.")
    if fixed_point_tol is not None and fixed_point_tol <= 0.0:
        raise ValueError("fixed_point_tol must be positive when supplied.")
    if fixed_point_patience <= 0:
        raise ValueError("fixed_point_patience must be positive.")
    if min_outer_iterations < 0:
        raise ValueError("min_outer_iterations must be nonnegative.")


def _as_2d(x: Array, name: str) -> Array:
    x = np.asarray(x)
    if x.ndim == 1:
        return x.reshape(-1, 1)
    if x.ndim == 2:
        return x
    raise ValueError(f"{name} must be 1D or 2D.")


def _prepare_target_action_samples(S: Array, A: Array, actions: Array, *, name: str) -> Dict[str, Array | int]:
    arr = np.asarray(actions)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim == 2:
        if arr.shape[0] != S.shape[0]:
            raise ValueError(f"{name} must match states rows.")
        return {
            "actions": arr,
            "states": S,
            "row_index": np.arange(S.shape[0], dtype=np.int64),
            "num_samples": 1,
        }
    if arr.ndim == 3:
        if arr.shape[0] != S.shape[0]:
            raise ValueError(f"{name} must match states rows.")
        if arr.shape[2] != A.shape[1]:
            raise ValueError(f"{name} must have the same feature dimension as actions.")
        n, m, d = arr.shape
        row_index = np.repeat(np.arange(n, dtype=np.int64), m)
        return {
            "actions": arr.reshape(n * m, d),
            "states": np.repeat(S, m, axis=0),
            "row_index": row_index,
            "num_samples": int(m),
        }
    raise ValueError(f"{name} must be 1D, 2D, or 3D.")


def _validate_base_transition_inputs(*, S: Array, A: Array, S_next: Array) -> None:
    n = S.shape[0]
    if A.shape[0] != n or S_next.shape[0] != n:
        raise ValueError("S, A, and S_next must all have the same number of rows.")
    if S_next.shape[1] != S.shape[1]:
        raise ValueError("S_next must have the same feature dimension as S.")


def _validate_target_action_rows(*, A: Array, A_pi: Array, name: str) -> None:
    if A_pi.shape[1] != A.shape[1]:
        raise ValueError(f"{name} must have the same feature dimension as actions.")


def _validate_aligned_inputs(*, S: Array, A: Array, S_next: Array, A_pi: Array) -> None:
    n = S.shape[0]
    if A.shape[0] != n or S_next.shape[0] != n or A_pi.shape[0] != n:
        raise ValueError("S, A, S_next, A_pi must all have the same number of rows.")
    if S_next.shape[1] != S.shape[1]:
        raise ValueError("S_next must have the same feature dimension as S.")
    if A_pi.shape[1] != A.shape[1]:
        raise ValueError("A_pi must have the same feature dimension as A.")


def _validate_initial_state_inputs(
    *,
    S: Array,
    S_initial: Optional[Array],
    initial_weights: Optional[Array],
) -> None:
    if S_initial is None:
        if initial_weights is not None:
            raise ValueError("initial_weights requires initial_states.")
        return
    if S_initial.shape[1] != S.shape[1]:
        raise ValueError("initial_states must have the same feature dimension as states.")
    if S_initial.shape[0] == 0:
        raise ValueError("initial_states must contain at least one row.")
    if initial_weights is not None:
        weights = np.asarray(initial_weights, dtype=np.float64).reshape(-1)
        if weights.shape[0] != S_initial.shape[0]:
            raise ValueError("initial_weights must match initial_states rows.")
        if not np.any(np.isfinite(weights) & (weights > 0.0)):
            raise ValueError("initial_weights must contain at least one positive finite value.")


def _validate_initial_action_inputs(*, A: Array, S_initial: Optional[Array], A_initial: Optional[Array]) -> None:
    if A_initial is None:
        return
    if S_initial is None:
        raise ValueError("initial_actions requires initial_states.")
    if A_initial.shape[0] != S_initial.shape[0]:
        raise ValueError("initial_actions must match initial_states rows.")
    if A_initial.shape[1] != A.shape[1]:
        raise ValueError("initial_actions must have the same feature dimension as actions.")


def _validate_next_target_actions(*, A: Array, S: Array, A_pi_next: Optional[Array]) -> None:
    if A_pi_next is None:
        return
    if A_pi_next.shape[0] != S.shape[0]:
        raise ValueError("target_next_actions must match states rows.")
    if A_pi_next.shape[1] != A.shape[1]:
        raise ValueError("target_next_actions must have the same feature dimension as actions.")


def _resolve_initial_ratio_mode(
    mode: str,
    *,
    S_initial: Optional[Array],
    A_initial: Optional[Array],
) -> str:
    normalized = str(mode).strip().lower()
    if normalized not in {"auto", "joint", "factored"}:
        raise ValueError("initial_ratio_mode must be 'auto', 'joint', or 'factored'.")
    if normalized == "auto":
        return "joint" if S_initial is not None and A_initial is not None else "factored"
    if normalized == "joint" and (S_initial is None or A_initial is None):
        raise ValueError("initial_ratio_mode='joint' requires initial_states and initial_actions.")
    return normalized


def _resolve_one_step_ratio_mode(mode: str, *, A_pi_next: Optional[Array]) -> str:
    normalized = str(mode).strip().lower()
    if normalized not in {"auto", "direct", "factored"}:
        raise ValueError("one_step_ratio_mode must be 'auto', 'direct', or 'factored'.")
    if normalized == "auto":
        return "direct" if A_pi_next is not None else "factored"
    if normalized == "direct" and A_pi_next is None:
        raise ValueError("one_step_ratio_mode='direct' requires target_next_actions.")
    return normalized


def _resolve_continuation(
    *,
    terminals: Optional[Array],
    timeouts: Optional[Array],
    handle_timeouts: str,
    absorbing_state: bool,
    n_rows: int,
) -> Array:
    normalized = str(handle_timeouts).strip().lower()
    if normalized not in {"nonterminal", "terminal", "error"}:
        raise ValueError("handle_timeouts must be 'nonterminal', 'terminal', or 'error'.")
    terminal_arr = _optional_binary_vector(terminals, n_rows, "terminals")
    timeout_arr = _optional_binary_vector(timeouts, n_rows, "timeouts")
    if normalized == "error" and np.any(timeout_arr > 0.0):
        raise ValueError("timeouts were supplied and handle_timeouts='error'.")
    if bool(absorbing_state):
        return np.ones(int(n_rows), dtype=np.float64)
    continuation = 1.0 - terminal_arr
    if normalized == "terminal":
        continuation = continuation * (1.0 - timeout_arr)
    return np.clip(continuation, 0.0, 1.0).astype(np.float64, copy=False)


def _optional_binary_vector(values: Optional[Array], n_rows: int, name: str) -> Array:
    if values is None:
        return np.zeros(int(n_rows), dtype=np.float64)
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.shape[0] != int(n_rows):
        raise ValueError(f"{name} must have {n_rows} rows.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values.")
    if np.any((arr < 0.0) | (arr > 1.0)):
        raise ValueError(f"{name} must be in [0, 1].")
    return arr.astype(np.float64, copy=False)


def _resolve_known_action_ratio_inputs(
    *,
    action_ratio_values: Optional[Array],
    behavior_log_prob: Optional[Array],
    target_log_prob: Optional[Array],
    n_rows: int,
    q_rows: int,
    n_target_rows: int,
    prediction_max: Optional[float],
    normalize: bool,
) -> tuple[Optional[Array], Optional[Array]]:
    supplied = action_ratio_values is not None
    supplied_logs = behavior_log_prob is not None or target_log_prob is not None
    if supplied and supplied_logs:
        raise ValueError("Provide either action_ratio_values or behavior/target log-probs, not both.")
    if behavior_log_prob is None and target_log_prob is None and action_ratio_values is None:
        return None, None
    if (behavior_log_prob is None) != (target_log_prob is None):
        raise ValueError("behavior_log_prob and target_log_prob must be supplied together.")
    if action_ratio_values is not None:
        raw = np.asarray(action_ratio_values, dtype=np.float64).reshape(-1)
    else:
        raw = np.exp(
            np.asarray(target_log_prob, dtype=np.float64).reshape(-1)
            - np.asarray(behavior_log_prob, dtype=np.float64).reshape(-1)
        )
    if raw.shape[0] not in {int(n_rows), int(q_rows)}:
        raise ValueError("known action ratios must have either behavior rows or all query rows.")
    values = _postprocess_known_action_ratios(raw, prediction_max=prediction_max, normalize=normalize)
    if raw.shape[0] == int(q_rows):
        return values[-int(n_rows) :], values
    query = np.concatenate([np.ones(int(n_target_rows), dtype=np.float64), values])
    return values, query


def _postprocess_known_action_ratios(values: Array, *, prediction_max: Optional[float], normalize: bool) -> Array:
    out = np.asarray(values, dtype=np.float64).reshape(-1).copy()
    finite_pos = float(prediction_max) if prediction_max is not None else np.finfo(np.float64).max / 16.0
    out = np.nan_to_num(out, nan=0.0, posinf=finite_pos, neginf=0.0)
    np.maximum(out, 0.0, out=out)
    if prediction_max is not None:
        np.minimum(out, float(prediction_max), out=out)
    if normalize:
        mean = float(np.mean(out)) if out.size else 0.0
        if np.isfinite(mean) and mean > 1e-12:
            out = out / mean
    return out.astype(np.float64, copy=False)
