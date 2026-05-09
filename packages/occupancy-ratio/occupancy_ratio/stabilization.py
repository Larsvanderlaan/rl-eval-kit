"""Stabilization, projection, loss, and weight-summary helpers for FORI."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import lightgbm as lgb
import numpy as np


Array = np.ndarray

__all__ = [
    "_adaptive_huber_growth",
    "_adaptive_huber_quantile_cap",
    "_clip_pseudo_outcomes",
    "_damped_update",
    "_ess",
    "_make_occupancy_objective",
    "_make_occupancy_sample_weights",
    "_make_stabilized_fixed_point_target",
    "_nonnegative",
    "_normalize_occupancy_loss",
    "_occupancy_loss_value",
    "_project_nonnegative_normalized",
    "_quantile_or_nan",
    "_resolve_huber_delta",
    "_safe_divide",
    "_squared_error_objective",
    "_summarize_ratio_predictions",
    "_summarize_vector",
    "_summarize_weights",
    "_weighted_mean",
]


def _normalize_occupancy_loss(loss: str) -> str:
    normalized = str(loss).strip().lower()
    aliases = {
        "l2": "squared",
        "mse": "squared",
        "squared_error": "squared",
        "squared": "squared",
        "huber": "huber",
        "robust": "huber",
    }
    if normalized not in aliases:
        raise ValueError("loss must be 'squared' or 'huber'.")
    return aliases[normalized]


def _resolve_huber_delta(
    residuals: Array,
    *,
    loss: str,
    huber_delta: Optional[float],
    huber_delta_scale: float,
    huber_delta_quantile_power: Optional[float],
    huber_delta_min_quantile: float,
) -> Optional[float]:
    if loss != "huber":
        return None
    if huber_delta is not None:
        # A fixed threshold estimates a conditional Huber location. The
        # adaptive default below lets the threshold diverge with n so robust
        # finite-sample fitting still targets the conditional mean asymptotically.
        return float(huber_delta)

    resid = np.asarray(residuals, dtype=np.float64).reshape(-1)
    resid = resid[np.isfinite(resid)]
    if resid.size == 0:
        return 1.0
    centered = resid - float(np.median(resid))
    mad = float(np.median(np.abs(centered)))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale <= 0.0:
        scale = float(np.std(resid))
    if not np.isfinite(scale) or scale <= 0.0:
        q75, q25 = np.percentile(resid, [75.0, 25.0])
        scale = float((q75 - q25) / 1.349)
    if not np.isfinite(scale) or scale <= 0.0:
        scale = max(float(np.mean(np.abs(resid))), 1.0)
    growth = _adaptive_huber_growth(resid.size)
    delta = float(huber_delta_scale) * scale * growth
    quantile_cap = _adaptive_huber_quantile_cap(
        resid,
        quantile_power=huber_delta_quantile_power,
        min_quantile=huber_delta_min_quantile,
    )
    if quantile_cap is not None:
        delta = min(delta, quantile_cap)
    return max(delta, 1e-8)


def _adaptive_huber_growth(n_eff: int) -> float:
    """Increasing Huber threshold multiplier for mean-consistent robust fitting."""
    n_eff = int(n_eff)
    if n_eff <= 2:
        return 1.0
    return float(np.sqrt(n_eff / np.log(n_eff)))


def _adaptive_huber_quantile_cap(
    residuals: Array,
    *,
    quantile_power: Optional[float],
    min_quantile: float,
) -> Optional[float]:
    """Finite-sample cap whose quantile level tends to one as sample size grows."""
    if quantile_power is None:
        return None
    abs_resid = np.abs(np.asarray(residuals, dtype=np.float64).reshape(-1))
    abs_resid = abs_resid[np.isfinite(abs_resid)]
    if abs_resid.size == 0:
        return None
    level = max(float(min_quantile), 1.0 - abs_resid.size ** (-float(quantile_power)))
    level = min(level, 1.0 - 1.0 / max(float(abs_resid.size), 2.0))
    cap = float(np.quantile(abs_resid, level))
    return max(cap, 1e-8) if np.isfinite(cap) else None


def _make_occupancy_objective(
    *,
    loss: str,
    huber_delta: Optional[float],
    huber_hessian_floor: float,
) -> Callable[[Array, lgb.Dataset], tuple[Array, Array]]:
    if loss == "squared":
        return _squared_error_objective

    if huber_delta is None:
        raise ValueError("huber_delta is required for Huber occupancy loss.")
    delta = float(huber_delta)
    hessian_floor = float(huber_hessian_floor)

    def huber_objective(preds: Array, train_data: lgb.Dataset) -> tuple[Array, Array]:
        resid = preds - train_data.get_label()
        abs_resid = np.abs(resid)
        grad = np.clip(resid, -delta, delta)
        hess = np.where(abs_resid <= delta, 1.0, hessian_floor)
        return grad, hess

    return huber_objective


def _squared_error_objective(preds: Array, train_data: lgb.Dataset) -> tuple[Array, Array]:
    resid = preds - train_data.get_label()
    return resid, np.ones_like(resid)


def _occupancy_loss_value(
    preds: Array,
    labels: Array,
    *,
    loss: str,
    huber_delta: Optional[float],
) -> float:
    resid = np.asarray(preds, dtype=np.float64) - np.asarray(labels, dtype=np.float64)
    if loss == "squared":
        return float(0.5 * np.mean(resid**2))
    if huber_delta is None:
        raise ValueError("huber_delta is required for Huber occupancy loss.")
    abs_resid = np.abs(resid)
    quadratic = abs_resid <= float(huber_delta)
    values = np.empty_like(abs_resid, dtype=np.float64)
    values[quadratic] = 0.5 * resid[quadratic] ** 2
    values[~quadratic] = float(huber_delta) * (abs_resid[~quadratic] - 0.5 * float(huber_delta))
    return float(np.mean(values))


def _nonnegative(x: Array) -> Array:
    return np.maximum(np.asarray(x, dtype=np.float64), 0.0)


def _safe_divide(numerator: Array, denominator: Array, *, eps: float = 1e-12) -> Array:
    numerator = np.asarray(numerator, dtype=np.float64)
    denominator = np.asarray(denominator, dtype=np.float64)
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan, dtype=np.float64),
        where=denominator > eps,
    )


def _project_nonnegative_normalized(
    values: Array,
    reference_weights: Optional[Array] = None,
    max_value: Optional[float] = None,
    normalize: bool = True,
    eps: float = 1e-12,
    *,
    normalization_scale: Optional[float] = None,
    return_info: bool = False,
) -> Array | tuple[Array, Dict[str, float]]:
    """Project ratio estimates onto a nonnegative, optionally bounded scale.

    The occupancy ratio has unit mean under the reference distribution. The
    empirical normalization enforces that moment on the current reference batch;
    passing ``normalization_scale`` lets prediction use a training-time scale
    instead of depending on the arbitrary batch supplied by a caller.
    """
    x_raw = np.asarray(values, dtype=np.float64).reshape(-1)
    cap = None if max_value is None else float(max_value)
    posinf = cap if cap is not None else np.finfo(np.float64).max / 16.0
    negative_fraction = float(np.mean(x_raw < 0.0)) if x_raw.size else 0.0
    nonfinite_fraction = float(np.mean(~np.isfinite(x_raw))) if x_raw.size else 0.0
    x = np.nan_to_num(x_raw, nan=0.0, posinf=posinf, neginf=0.0)
    np.maximum(x, 0.0, out=x)
    clipped_fraction = 0.0
    post_normalization_clipped_fraction = 0.0
    if cap is not None:
        clipped_fraction = float(np.mean(x_raw > cap)) if x.size else 0.0
        np.minimum(x, cap, out=x)

    scale = 1.0
    if normalize:
        if normalization_scale is not None:
            scale = float(normalization_scale)
        else:
            scale = _weighted_mean(x, reference_weights)
        if np.isfinite(scale) and scale > eps:
            x = x / scale
            if cap is not None:
                post_normalization_clipped_fraction = float(np.mean(x > cap)) if x.size else 0.0
                np.minimum(x, cap, out=x)
        else:
            fill = 1.0 if cap is None else min(1.0, cap)
            x = np.full_like(x, fill, dtype=np.float64)
            scale = 0.0

    info = dict(
        projection_clipped_fraction=float(clipped_fraction),
        projection_post_normalization_clipped_fraction=float(post_normalization_clipped_fraction),
        projection_negative_fraction=float(negative_fraction),
        projection_nonfinite_fraction=float(nonfinite_fraction),
        projection_normalization_scale=float(scale),
        projection_max_value=float(cap) if cap is not None else float("nan"),
    )
    if return_info:
        return x.astype(np.float64, copy=False), info
    return x.astype(np.float64, copy=False)


def _weighted_mean(values: Array, weights: Optional[Array]) -> float:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return 0.0
    if weights is None:
        return float(np.mean(x))
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if w.shape[0] != x.shape[0]:
        raise ValueError("reference_weights must match values length.")
    w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    w = np.maximum(w, 0.0)
    denom = float(np.sum(w))
    if denom <= 0.0:
        return float(np.mean(x))
    return float(np.sum(w * x) / denom)


def _clip_pseudo_outcomes(
    values: Array,
    *,
    enabled: bool,
    pseudo_outcome_max: Optional[float],
    pseudo_outcome_upper_quantile: float,
    pseudo_outcome_min: float,
    target_min: Optional[float],
    target_max: Optional[float],
) -> tuple[Array, Dict[str, float]]:
    raw = np.asarray(values, dtype=np.float64).reshape(-1)
    y = np.nan_to_num(raw, nan=float(pseudo_outcome_min), posinf=0.0, neginf=float(pseudo_outcome_min))
    lower = max(float(pseudo_outcome_min), float(target_min) if target_min is not None else float(pseudo_outcome_min))
    upper_candidates: List[float] = []
    if target_max is not None:
        upper_candidates.append(float(target_max))
    if enabled:
        if pseudo_outcome_max is not None:
            upper_candidates.append(float(pseudo_outcome_max))
        else:
            finite = y[np.isfinite(y)]
            finite = finite[finite >= lower]
            if finite.size:
                upper_candidates.append(float(np.quantile(finite, float(pseudo_outcome_upper_quantile))))
    cap = min(upper_candidates) if upper_candidates else None
    if cap is not None:
        cap = max(float(cap), lower)
    before = y.copy()
    np.maximum(y, lower, out=y)
    if cap is not None:
        np.minimum(y, cap, out=y)
    clipped = before != y
    diag = dict(
        pseudo_outcome_cap=float(cap) if cap is not None else float("nan"),
        pseudo_outcome_min=float(lower),
        pseudo_outcome_clipped_fraction=float(np.mean(clipped)) if clipped.size else 0.0,
        pseudo_outcome_p95=_quantile_or_nan(y, 0.95),
        pseudo_outcome_p99=_quantile_or_nan(y, 0.99),
        pseudo_outcome_max=float(np.max(y)) if y.size else float("nan"),
        pseudo_outcome_mean=float(np.mean(y)) if y.size else float("nan"),
    )
    return y, diag


def _make_stabilized_fixed_point_target(
    *,
    raw_target: Array,
    current: Array,
    eta: float,
    normalize: bool,
    occupancy_ratio_max: Optional[float],
    eps: float,
    clip_pseudo_outcomes: bool,
    pseudo_outcome_max: Optional[float],
    pseudo_outcome_upper_quantile: float,
    pseudo_outcome_min: float,
    target_min: Optional[float],
    target_max: Optional[float],
) -> tuple[Array, Dict[str, Any]]:
    clipped, clip_diag = _clip_pseudo_outcomes(
        raw_target,
        enabled=clip_pseudo_outcomes,
        pseudo_outcome_max=pseudo_outcome_max,
        pseudo_outcome_upper_quantile=pseudo_outcome_upper_quantile,
        pseudo_outcome_min=pseudo_outcome_min,
        target_min=target_min,
        target_max=target_max,
    )
    projected, projection_diag = _project_nonnegative_normalized(
        clipped,
        max_value=occupancy_ratio_max,
        normalize=normalize,
        eps=eps,
        return_info=True,
    )
    damped = _damped_update(current, projected, eta)
    diag: Dict[str, Any] = {}
    diag.update(clip_diag)
    diag.update({f"target_raw_{key}": val for key, val in _summarize_vector(raw_target).items()})
    diag.update({f"target_projected_{key}": val for key, val in _summarize_vector(projected).items()})
    diag.update({f"target_damped_{key}": val for key, val in _summarize_vector(damped).items()})
    diag.update({f"pseudo_{key}": val for key, val in projection_diag.items()})
    return damped, diag


def _damped_update(current: Array, projected_update: Array, eta: float) -> Array:
    current_arr = np.asarray(current, dtype=np.float64)
    update_arr = np.asarray(projected_update, dtype=np.float64)
    return (1.0 - float(eta)) * current_arr + float(eta) * update_arr


def _make_occupancy_sample_weights(
    *,
    mode: str,
    action_ratio: Optional[Array],
    target: Array,
    max_value: Optional[float],
    eps: float = 1e-12,
) -> tuple[Array, Dict[str, float]]:
    mode = str(mode)
    target_arr = _project_nonnegative_normalized(target, max_value=None, normalize=False, eps=eps)
    if mode == "uniform":
        weights = np.ones_like(target_arr, dtype=np.float64)
    elif mode in {"sqrt_action_ratio", "action_ratio"}:
        if action_ratio is None:
            raise ValueError(f"action_ratio is required for occupancy_sample_weight_mode='{mode}'.")
        base = _project_nonnegative_normalized(action_ratio, max_value=max_value, normalize=False, eps=eps)
        weights = np.sqrt(base) if mode == "sqrt_action_ratio" else base
    elif mode in {"sqrt_target", "target"}:
        weights = np.sqrt(target_arr) if mode == "sqrt_target" else target_arr
    else:
        raise ValueError("Unknown occupancy sample-weight mode.")

    weights = np.nan_to_num(weights, nan=1.0, posinf=max_value if max_value is not None else 1.0, neginf=0.0)
    np.maximum(weights, 0.0, out=weights)
    clipped_fraction = 0.0
    if max_value is not None:
        clipped_fraction = float(np.mean(weights > float(max_value))) if weights.size else 0.0
        np.minimum(weights, float(max_value), out=weights)
    mean = float(np.mean(weights)) if weights.size else 0.0
    if np.isfinite(mean) and mean > eps:
        weights = weights / mean
    else:
        weights = np.ones_like(weights, dtype=np.float64)
    if max_value is not None:
        np.minimum(weights, float(max_value), out=weights)
        mean = float(np.mean(weights)) if weights.size else 0.0
        if np.isfinite(mean) and mean > eps:
            weights = weights / mean
            np.minimum(weights, float(max_value), out=weights)
    diag = _summarize_weights(weights)
    diag["sample_weight_clipped_fraction"] = float(clipped_fraction)
    diag["sample_weight_mode"] = mode
    return weights.astype(np.float64, copy=False), diag


def _ess(weights: Array, *, eps: float = 1e-12) -> float:
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    w = w[np.isfinite(w)]
    denom = float(np.sum(w**2))
    if w.size == 0 or denom <= eps:
        return 0.0
    return float(np.sum(w) ** 2 / denom)


def _summarize_vector(values: Array) -> Dict[str, float]:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return dict(mean=float("nan"), std=float("nan"), p95=float("nan"), p99=float("nan"), max=float("nan"))
    return dict(
        mean=float(np.mean(x)),
        std=float(np.std(x)),
        p95=float(np.quantile(x, 0.95)),
        p99=float(np.quantile(x, 0.99)),
        max=float(np.max(x)),
    )


def _summarize_ratio_predictions(values: Array) -> Dict[str, float]:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {
            "min": float("nan"),
            "p50": float("nan"),
            "p90": float("nan"),
            "p95": float("nan"),
            "p99": float("nan"),
            "max": float("nan"),
            "clipped_fraction": 0.0,
            "normalization_scale": 1.0,
        }
    return {
        "min": float(np.min(x)),
        "p50": float(np.quantile(x, 0.50)),
        "p90": float(np.quantile(x, 0.90)),
        "p95": float(np.quantile(x, 0.95)),
        "p99": float(np.quantile(x, 0.99)),
        "max": float(np.max(x)),
        "clipped_fraction": 0.0,
        "normalization_scale": 1.0,
    }


def _summarize_weights(weights: Array) -> Dict[str, float]:
    summary = _summarize_vector(weights)
    return {
        "sample_weight_mean": summary["mean"],
        "sample_weight_std": summary["std"],
        "sample_weight_p95": summary["p95"],
        "sample_weight_p99": summary["p99"],
        "sample_weight_max": summary["max"],
    }


def _quantile_or_nan(values: Array, q: float) -> float:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    return float(np.quantile(x, q)) if x.size else float("nan")
