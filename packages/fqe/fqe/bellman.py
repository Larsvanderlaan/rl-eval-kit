from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class BootstrapInfo:
    """Validated Bellman bootstrap continuation metadata."""

    continuation: Array
    terminals: Array
    timeouts: Array
    mode: str
    diagnostics: dict[str, float | str]


def validate_bootstrap_inputs(
    *,
    n_rows: int,
    terminals: Array | None = None,
    timeouts: Array | None = None,
    continuation: Array | None = None,
) -> BootstrapInfo:
    """Validate terminal/timeout semantics for Bellman targets.

    ``terminals`` means true absorbing termination and blocks bootstrapping.
    ``timeouts`` means exogenous truncation and does not block bootstrapping.
    ``continuation`` is an explicit bootstrap multiplier in ``[0, 1]`` and is
    mutually exclusive with terminal/timeout masks.
    """

    n = int(n_rows)
    if continuation is not None and (terminals is not None or timeouts is not None):
        raise ValueError("continuation is mutually exclusive with terminals/timeouts.")
    if continuation is not None:
        cont = _as_unit_interval_vector(continuation, n, "continuation")
        term = 1.0 - cont
        timeout = np.zeros(n, dtype=np.float64)
        mode = "explicit_continuation"
    else:
        term = np.zeros(n, dtype=np.float64) if terminals is None else _as_unit_interval_vector(terminals, n, "terminals")
        timeout = np.zeros(n, dtype=np.float64) if timeouts is None else _as_unit_interval_vector(timeouts, n, "timeouts")
        cont = 1.0 - term
        mode = "terminals_timeouts"
    diagnostics = {
        "bootstrap_mode": mode,
        "terminal_fraction": float(np.mean(term)) if n else 0.0,
        "timeout_fraction": float(np.mean(timeout)) if n else 0.0,
        "continuation_mean": float(np.mean(cont)) if n else 0.0,
        "continuation_min": float(np.min(cont)) if n else 0.0,
        "continuation_max": float(np.max(cont)) if n else 0.0,
    }
    return BootstrapInfo(
        continuation=np.ascontiguousarray(cont, dtype=np.float64),
        terminals=np.ascontiguousarray(term, dtype=np.float64),
        timeouts=np.ascontiguousarray(timeout, dtype=np.float64),
        mode=mode,
        diagnostics=diagnostics,
    )


def validate_action_weights(weights: Array | None, *, n_rows: int, n_actions: int, name: str) -> Array:
    """Return row-normalized target-action weights for sampled next/initial actions."""

    n = int(n_rows)
    m = int(n_actions)
    if weights is None:
        return np.full((n, m), 1.0 / max(m, 1), dtype=np.float64)
    arr = np.asarray(weights, dtype=np.float64)
    if arr.ndim == 1:
        if m != 1:
            raise ValueError(f"{name} must have shape ({n}, {m}) for multiple action samples.")
        arr = arr.reshape(n, 1)
    elif arr.ndim != 2:
        raise ValueError(f"{name} must be a 1D or 2D array.")
    if arr.shape != (n, m):
        raise ValueError(f"{name} must have shape ({n}, {m}).")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values.")
    if np.any(arr < 0.0):
        raise ValueError(f"{name} must be nonnegative.")
    row_sum = np.sum(arr, axis=1)
    if np.any(row_sum <= 0.0):
        raise ValueError(f"{name} rows must have positive total weight.")
    return np.ascontiguousarray(arr / row_sum[:, None], dtype=np.float64)


def weighted_action_expectation(values: Array, weights: Array | None = None) -> Array:
    """Average a ``(n, m)`` action-value matrix with row-normalized weights."""

    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("values must be a 2D (n, n_action_samples) array.")
    weight = validate_action_weights(weights, n_rows=matrix.shape[0], n_actions=matrix.shape[1], name="action_weights")
    return np.sum(matrix * weight, axis=1)


def build_bellman_target(
    *,
    rewards: Array,
    gamma: float,
    next_predictions: Array,
    continuation: Array,
    target_min: float | None = None,
    target_max: float | None = None,
) -> Array:
    target = np.asarray(rewards, dtype=np.float64).reshape(-1) + float(gamma) * np.asarray(
        continuation,
        dtype=np.float64,
    ).reshape(-1) * np.asarray(next_predictions, dtype=np.float64).reshape(-1)
    return clip_targets(target, target_min, target_max)


def bellman_residual(
    *,
    predictions: Array,
    rewards: Array,
    gamma: float,
    next_predictions: Array,
    continuation: Array,
    target_min: float | None = None,
    target_max: float | None = None,
) -> Array:
    target = build_bellman_target(
        rewards=rewards,
        gamma=gamma,
        next_predictions=next_predictions,
        continuation=continuation,
        target_min=target_min,
        target_max=target_max,
    )
    return np.asarray(predictions, dtype=np.float64).reshape(-1) - target


def bellman_risk(
    *,
    predictions: Array,
    rewards: Array,
    gamma: float,
    next_predictions: Array,
    continuation: Array,
    sample_weight: Array,
    target_min: float | None = None,
    target_max: float | None = None,
) -> float:
    err = bellman_residual(
        predictions=predictions,
        rewards=rewards,
        gamma=gamma,
        next_predictions=next_predictions,
        continuation=continuation,
        target_min=target_min,
        target_max=target_max,
    )
    return weighted_mean(err * err, sample_weight)


def clip_targets(targets: Array, target_min: float | None, target_max: float | None) -> Array:
    out = np.asarray(targets, dtype=np.float64)
    if target_min is not None or target_max is not None:
        out = np.clip(
            out,
            -np.inf if target_min is None else float(target_min),
            np.inf if target_max is None else float(target_max),
        )
    return out


def weighted_mean(values: Array, weights: Array | None) -> float:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return float("nan")
    if weights is None:
        return float(np.mean(arr))
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if w.shape[0] != arr.shape[0]:
        raise ValueError("weights must match values length.")
    if not np.all(np.isfinite(w)):
        raise ValueError("weights must contain only finite values.")
    if np.any(w < 0.0):
        raise ValueError("weights must be nonnegative.")
    total = float(np.sum(w))
    if total <= 0.0:
        raise ValueError("weights must have positive total weight.")
    return float(np.average(arr, weights=w))


def weight_diagnostics(weights: Array, *, prefix: str = "weight") -> dict[str, float]:
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if w.size == 0 or not np.all(np.isfinite(w)):
        return {
            f"{prefix}_mean": float("nan"),
            f"{prefix}_std": float("nan"),
            f"{prefix}_min": float("nan"),
            f"{prefix}_max": float("nan"),
            f"{prefix}_p99": float("nan"),
            f"{prefix}_ess_fraction": 0.0,
        }
    denom = float(np.sum(w * w))
    ess = 0.0 if denom <= 0.0 else float((np.sum(w) ** 2 / denom) / w.size)
    return {
        f"{prefix}_mean": float(np.mean(w)),
        f"{prefix}_std": float(np.std(w)),
        f"{prefix}_min": float(np.min(w)),
        f"{prefix}_max": float(np.max(w)),
        f"{prefix}_p99": float(np.quantile(w, 0.99)),
        f"{prefix}_ess_fraction": ess,
    }


def package_versions(names: tuple[str, ...]) -> dict[str, str | None]:
    try:
        from importlib import metadata
    except Exception:  # pragma: no cover
        return {name: None for name in names}
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _as_unit_interval_vector(value: Array, n_rows: int, name: str) -> Array:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.shape[0] != int(n_rows):
        raise ValueError(f"{name} must have {int(n_rows)} rows.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values.")
    if np.any((arr < 0.0) | (arr > 1.0)):
        raise ValueError(f"{name} must be in [0, 1].")
    return arr


def serializable_config(config: Any) -> dict[str, Any]:
    from dataclasses import asdict, is_dataclass

    if is_dataclass(config):
        return asdict(config)
    if isinstance(config, dict):
        return dict(config)
    raise TypeError("config must be a dataclass or dict.")
