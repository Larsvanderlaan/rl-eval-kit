from __future__ import annotations

from typing import Any, Optional

import numpy as np

from occupancy_ratio.diagnostics import postprocess_weights, weight_summary


Array = np.ndarray

__all__ = ["compare_fori_to_google_dualdice"]


def compare_fori_to_google_dualdice(
    fori_model: Any,
    google_model: Any,
    states: Array,
    actions: Array,
    target_actions: Optional[Array] = None,
    cap: Optional[float] = None,
    normalize: bool = False,
    rewards: Optional[Array] = None,
) -> dict[str, Any]:
    """Compare FORI and Google DualDICE state-action ratios under matched postprocessing.

    ``target_actions`` is accepted for caller convenience but this utility
    compares the state-action object on the supplied ``(states, actions)`` rows.
    """
    del target_actions
    fori_raw = _predict_state_action_ratio(fori_model, states, actions)
    google_raw = _predict_state_action_ratio(google_model, states, actions)
    fori = postprocess_weights(fori_raw, cap=cap, normalize=normalize)
    google = postprocess_weights(google_raw, cap=cap, normalize=normalize)

    out = {
        "postprocessing": {
            "cap": None if cap is None else float(cap),
            "normalize": bool(normalize),
        },
        "object": "state_action_ratio",
        "fori": weight_summary(fori, cap=cap),
        "google": weight_summary(google, cap=cap),
    }
    out.update(_pairwise_metrics(fori, google))
    if rewards is not None:
        rewards_arr = np.asarray(rewards, dtype=np.float64).reshape(-1)
        if rewards_arr.shape[0] != fori.shape[0]:
            raise ValueError("rewards must have the same number of rows as states/actions.")
        out["fori"]["weighted_reward_mean"] = float(np.mean(fori * rewards_arr))
        out["google"]["weighted_reward_mean"] = float(np.mean(google * rewards_arr))
        out["reward_value_fori"] = out["fori"]["weighted_reward_mean"]
        out["reward_value_google"] = out["google"]["weighted_reward_mean"]
    return _json_ready(out)


def _predict_state_action_ratio(model: Any, states: Array, actions: Array) -> Array:
    if not hasattr(model, "predict_state_action_ratio"):
        raise ValueError("model must expose predict_state_action_ratio(states, actions, clip=False).")
    return np.asarray(model.predict_state_action_ratio(states, actions, clip=False), dtype=np.float64).reshape(-1)


def _pairwise_metrics(left: Array, right: Array, eps: float = 1e-12) -> dict[str, float]:
    x = np.asarray(left, dtype=np.float64).reshape(-1)
    y = np.asarray(right, dtype=np.float64).reshape(-1)
    if x.shape[0] != y.shape[0]:
        raise ValueError("FORI and Google predictions must have the same length.")
    finite = np.isfinite(x) & np.isfinite(y)
    if not np.any(finite):
        return {
            "pearson_corr": float("nan"),
            "spearman_corr": float("nan"),
            "correlation": float("nan"),
            "rank_correlation": float("nan"),
            "mean_abs_diff": float("nan"),
            "median_abs_diff": float("nan"),
            "mean_abs_log_diff": float("nan"),
            "mean_abs_log_ratio_gap": float("nan"),
            "top_1pct_overlap": float("nan"),
            "top_5pct_overlap": float("nan"),
        }
    x = x[finite]
    y = y[finite]
    pearson = _corrcoef(x, y)
    spearman = _corrcoef(_rankdata(x), _rankdata(y))
    log_gap = np.abs(np.log(np.maximum(x, eps)) - np.log(np.maximum(y, eps)))
    return {
        "pearson_corr": pearson,
        "spearman_corr": spearman,
        "correlation": pearson,
        "rank_correlation": spearman,
        "mean_abs_diff": float(np.mean(np.abs(x - y))),
        "median_abs_diff": float(np.median(np.abs(x - y))),
        "mean_abs_log_diff": float(np.mean(log_gap)),
        "mean_abs_log_ratio_gap": float(np.mean(log_gap)),
        "top_1pct_overlap": _top_fraction_overlap(x, y, 0.01),
        "top_5pct_overlap": _top_fraction_overlap(x, y, 0.05),
    }


def _corrcoef(x: Array, y: Array) -> float:
    if x.size < 2 or y.size < 2:
        return float("nan")
    if float(np.std(x)) <= 0.0 or float(np.std(y)) <= 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _rankdata(x: Array) -> Array:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(x.shape[0], dtype=np.float64)
    ranks[order] = np.arange(x.shape[0], dtype=np.float64)
    return ranks


def _top_fraction_overlap(x: Array, y: Array, fraction: float) -> float:
    if x.size == 0:
        return float("nan")
    k = max(1, int(np.ceil(float(fraction) * x.size)))
    x_top = set(np.argsort(x)[-k:].tolist())
    y_top = set(np.argsort(y)[-k:].tolist())
    return float(len(x_top & y_top) / k)


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(val) for val in value]
    if isinstance(value, np.ndarray):
        return [_json_ready(val) for val in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
