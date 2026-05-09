from __future__ import annotations

from typing import Any, Optional

import numpy as np


Array = np.ndarray

__all__ = [
    "postprocess_weights",
    "regularization_path_report",
    "weight_summary",
]


def weight_summary(w: Array, *, cap: Optional[float] = None, eps: float = 1e-12) -> dict[str, Any]:
    """Summarize a vector of nonnegative or raw importance weights."""
    values = np.asarray(w, dtype=np.float64).reshape(-1)
    finite = np.isfinite(values)
    out: dict[str, Any] = {
        "n": int(values.size),
        "nonfinite_fraction": float(1.0 - np.mean(finite)) if values.size else 0.0,
    }
    if not np.any(finite):
        out.update(
            mean=float("nan"),
            std=float("nan"),
            cv=float("nan"),
            ess=0.0,
            ess_fraction=0.0,
            min=float("nan"),
            p01=float("nan"),
            p05=float("nan"),
            p10=float("nan"),
            p50=float("nan"),
            p90=float("nan"),
            p95=float("nan"),
            p99=float("nan"),
            max=float("nan"),
            zero_fraction=0.0,
            clipped_fraction=None if cap is None else 0.0,
        )
        return out

    x = values[finite]
    mean = float(np.mean(x))
    std = float(np.std(x))
    denom = float(np.sum(x**2))
    ess = float((np.sum(x) ** 2) / (denom + eps)) if x.size else 0.0
    q01, q05, q10, q50, q90, q95, q99 = np.quantile(x, [0.01, 0.05, 0.10, 0.50, 0.90, 0.95, 0.99])
    out.update(
        mean=mean,
        std=std,
        cv=float(std / (abs(mean) + eps)),
        ess=ess,
        ess_fraction=float(ess / max(x.size, 1)),
        min=float(np.min(x)),
        p01=float(q01),
        p05=float(q05),
        p10=float(q10),
        p50=float(q50),
        p90=float(q90),
        p95=float(q95),
        p99=float(q99),
        max=float(np.max(x)),
        zero_fraction=float(np.mean(x <= eps)),
        clipped_fraction=None if cap is None else float(np.mean(x >= float(cap) - 1e-9)),
    )
    return out


def postprocess_weights(
    w: Array,
    *,
    cap: Optional[float] = None,
    normalize: bool = False,
    eps: float = 1e-12,
) -> Array:
    """Apply identical safety clipping and optional mean normalization to weights."""
    out = np.asarray(w, dtype=np.float64).reshape(-1).copy()
    finite_pos = float(cap) if cap is not None else np.finfo(np.float64).max / 16.0
    out = np.nan_to_num(out, nan=0.0, posinf=finite_pos, neginf=0.0)
    np.maximum(out, 0.0, out=out)
    if cap is not None:
        np.minimum(out, float(cap), out=out)
    if normalize:
        mean = float(np.mean(out)) if out.size else 0.0
        if np.isfinite(mean) and mean > eps:
            out = out / mean
            if cap is not None:
                np.minimum(out, float(cap), out=out)
    return out.astype(np.float64, copy=False)


def regularization_path_report(
    result_or_model: Any,
    *,
    data: Optional[dict[str, Any]] = None,
    google_model: Any = None,
) -> dict[str, Any]:
    """Build a compact, JSON-serializable report of available FORI weight stages."""
    result = result_or_model.to_legacy_dict() if hasattr(result_or_model, "to_legacy_dict") else dict(result_or_model)
    cap = result.get("occupancy_ratio_max")
    normalize = bool(result.get("normalize_occupancy", False))
    report: dict[str, Any] = {
        "config": _json_ready(
            {
                "loss": result.get("loss"),
                "fixed_point_damping": result.get("fixed_point_damping"),
                "normalize_occupancy": normalize,
                "occupancy_ratio_max": cap,
                "clip_pseudo_outcomes": result.get("clip_pseudo_outcomes"),
                "direct_adjoint_num_boost_round": result.get("direct_adjoint_num_boost_round"),
                "direct_adjoint_loss": result.get("direct_adjoint_loss"),
            }
        ),
        "summaries": {},
        "history_last": _json_ready(result.get("history", [{}])[-1] if result.get("history") else {}),
    }
    summaries: dict[str, Any] = report["summaries"]
    for name, key, key_cap in (
        ("action_ratio_postprocessed_behavior", "pred_iw", result.get("iw_prediction_max")),
        ("action_ratio_postprocessed_query", "pred_iw_query", result.get("iw_prediction_max")),
        ("occupancy_raw_behavior", "pred_state_action_ratio_beh_raw", None),
        ("occupancy_stabilized_behavior", "pred_state_action_ratio_beh", cap),
        ("occupancy_raw_target", "pred_state_action_ratio_pi_raw", None),
        ("occupancy_stabilized_target", "pred_state_action_ratio_pi", cap),
        ("occupancy_query_stabilized", "pred_query_stabilized", cap),
        ("occupancy_behavior_in_query_stabilized", "pred_sa_iw_in_query_clipped", cap),
    ):
        values = result.get(key)
        if values is not None:
            summaries[name] = weight_summary(values, cap=key_cap)

    if data is not None and hasattr(result_or_model, "predict_state_action_ratio"):
        states = data.get("states")
        actions = data.get("actions")
        if states is not None and actions is not None:
            raw = result_or_model.predict_state_action_ratio(states, actions, clip=False)
            final = result_or_model.predict_state_action_ratio(states, actions, clip=True)
            summaries["public_raw_prediction"] = weight_summary(raw)
            summaries["public_final_prediction"] = weight_summary(final, cap=cap)
            if google_model is not None and hasattr(google_model, "predict_state_action_ratio"):
                google_raw = google_model.predict_state_action_ratio(states, actions, clip=False)
                google_pp = postprocess_weights(google_raw, cap=cap, normalize=normalize)
                summaries["google_state_action_ratio_matched_postprocess"] = weight_summary(google_pp, cap=cap)

    return _json_ready(report)


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
