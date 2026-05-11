from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

import numpy as np


Array = np.ndarray


def effective_sample_size(weights: Array) -> float:
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    return float((w.sum() ** 2) / max(float(np.sum(w**2)), 1e-12))


def _weight_cv(weights: Array) -> float:
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    return float(np.std(w) / max(abs(float(np.mean(w))), 1e-12))


def summarize_weights(weights: Array, *, raw_weights: Array | None = None) -> dict[str, float]:
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    raw = w if raw_weights is None else np.asarray(raw_weights, dtype=np.float64).reshape(-1)
    if raw.shape != w.shape:
        raise ValueError("raw_weights must have the same shape as weights.")
    finite_raw = np.isfinite(raw)
    safety_clipped = (~finite_raw) | (raw < 0.0)
    postprocessed = ~np.isclose(raw, w, rtol=1e-10, atol=1e-12, equal_nan=True)
    q50 = float(np.quantile(w, 0.50))
    q99 = float(np.quantile(w, 0.99))
    return {
        "weight_mean": float(np.mean(w)),
        "weight_std": float(np.std(w)),
        "weight_cv": _weight_cv(w),
        "weight_min": float(np.min(w)),
        "weight_max": float(np.max(w)),
        "weight_q50": q50,
        "weight_q90": float(np.quantile(w, 0.90)),
        "weight_q95": float(np.quantile(w, 0.95)),
        "weight_q99": q99,
        "weight_q99_to_median": float(q99 / max(q50, 1e-12)),
        "effective_sample_size": effective_sample_size(w),
        "effective_sample_size_fraction": float(effective_sample_size(w) / max(w.shape[0], 1)),
        "negative_raw_fraction": float(np.mean(raw < 0.0)),
        "nonfinite_raw_fraction": float(np.mean(~finite_raw)),
        "clipping_fraction": float(np.mean(safety_clipped)),
        "postprocessing_changed_fraction": float(np.mean(postprocessed)),
    }


def truth_heterogeneity_diagnostics(true_ratio: Array, estimated_ratio: Array) -> dict[str, float]:
    truth = np.asarray(true_ratio, dtype=np.float64).reshape(-1)
    pred = np.asarray(estimated_ratio, dtype=np.float64).reshape(-1)
    if truth.shape != pred.shape:
        raise ValueError("true_ratio and estimated_ratio must have the same shape.")
    n = max(int(truth.shape[0]), 1)
    truth_ess = float(effective_sample_size(truth) / n)
    pred_ess = float(effective_sample_size(pred) / n)
    truth_q99 = float(np.quantile(truth, 0.99))
    pred_q99 = float(np.quantile(pred, 0.99))
    truth_q50 = float(np.quantile(truth, 0.50))
    pred_q50 = float(np.quantile(pred, 0.50))
    truth_cv = _weight_cv(truth)
    pred_cv = _weight_cv(pred)
    truth_q99_to_median = float(truth_q99 / max(truth_q50, 1e-12))
    pred_q99_to_median = float(pred_q99 / max(pred_q50, 1e-12))
    return {
        "true_effective_sample_size_fraction": truth_ess,
        "ess_fraction_abs_error_to_truth": float(abs(pred_ess - truth_ess)),
        "ess_fraction_ratio_to_truth": float(pred_ess / max(truth_ess, 1e-12)),
        "true_weight_cv": truth_cv,
        "weight_cv_abs_error_to_truth": float(abs(pred_cv - truth_cv)),
        "weight_cv_ratio_to_truth": float(pred_cv / max(truth_cv, 1e-12)),
        "true_weight_q99": truth_q99,
        "weight_q99_abs_error_to_truth": float(abs(pred_q99 - truth_q99)),
        "weight_q99_ratio_to_truth": float(pred_q99 / max(truth_q99, 1e-12)),
        "true_weight_q99_to_median": truth_q99_to_median,
        "weight_q99_to_median_abs_error_to_truth": float(abs(pred_q99_to_median - truth_q99_to_median)),
        "weight_q99_to_median_ratio_to_truth": float(
            pred_q99_to_median / max(truth_q99_to_median, 1e-12)
        ),
        "true_weight_max": float(np.max(truth)),
        "weight_max_abs_error_to_truth": float(abs(float(np.max(pred)) - float(np.max(truth)))),
    }


def ratio_quality(true_ratio: Array, estimated_ratio: Array) -> dict[str, float]:
    truth = np.asarray(true_ratio, dtype=np.float64).reshape(-1)
    pred = np.asarray(estimated_ratio, dtype=np.float64).reshape(-1)
    if truth.shape != pred.shape:
        raise ValueError("true_ratio and estimated_ratio must have the same shape.")
    truth_pos = np.maximum(truth, 1e-12)
    pred_pos = np.maximum(pred, 1e-12)
    truth_sd = float(np.std(truth_pos))
    pred_sd = float(np.std(pred_pos))
    corr = np.nan
    if truth_sd > 1e-12 and pred_sd > 1e-12:
        corr = float(np.corrcoef(truth_pos, pred_pos)[0, 1])
    elif np.allclose(truth_pos, pred_pos):
        corr = 1.0
    return {
        "ratio_rmse": float(np.sqrt(np.mean((pred - truth) ** 2))),
        "ratio_mae": float(np.mean(np.abs(pred - truth))),
        "ratio_l1": float(np.mean(np.abs(pred - truth))),
        "ratio_normalized_l1": float(np.mean(np.abs(pred - truth)) / max(float(np.mean(np.abs(truth))), 1e-12)),
        "ratio_tv": float(0.5 * np.mean(np.abs(pred - truth))),
        "ratio_bias": float(np.mean(pred - truth)),
        "ratio_rel_mse": float(np.mean((pred - truth) ** 2) / max(float(np.mean(truth**2)), 1e-12)),
        "log_ratio_rmse": float(np.sqrt(np.mean((np.log(pred_pos) - np.log(truth_pos)) ** 2))),
        "ratio_corr": corr,
    }


def calibration_by_quantile(
    true_ratio: Array,
    estimated_ratio: Array,
    *,
    n_bins: int = 10,
) -> dict[str, float]:
    truth = np.asarray(true_ratio, dtype=np.float64).reshape(-1)
    pred = np.asarray(estimated_ratio, dtype=np.float64).reshape(-1)
    order = np.argsort(pred)
    bins = np.array_split(order, int(n_bins))
    gaps = []
    for indices in bins:
        if indices.size == 0:
            continue
        gaps.append(abs(float(np.mean(pred[indices]) - np.mean(truth[indices]))))
    return {
        "calibration_abs_gap_mean": float(np.mean(gaps)) if gaps else np.nan,
        "calibration_abs_gap_max": float(np.max(gaps)) if gaps else np.nan,
    }


def normalization_error(weights: Array, reference_weights: Array | None = None) -> float:
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if reference_weights is None:
        return float(np.mean(w) - 1.0)
    ref = np.asarray(reference_weights, dtype=np.float64).reshape(-1)
    return float(np.sum(ref * w) / max(float(np.sum(ref)), 1e-12) - 1.0)


def bellman_flow_residual(
    weights: Array,
    true_ratio: Array,
    *,
    feature_matrix: Array | None = None,
) -> float:
    """A finite-sample moment discrepancy against oracle ratios.

    Exact Bellman residuals are setting-specific. This common diagnostic
    measures whether the estimator matches oracle weighted feature moments on
    the benchmark rows; tabular and Gaussian settings use rich enough default
    features for this to catch flow-scale errors.
    """
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    truth = np.asarray(true_ratio, dtype=np.float64).reshape(-1)
    if feature_matrix is None:
        feature_matrix = np.column_stack([np.ones_like(w), truth])
    phi = np.asarray(feature_matrix, dtype=np.float64)
    diff = np.mean(phi * (w - truth)[:, None], axis=0)
    return float(np.linalg.norm(diff))


def estimator_diagnostics(
    *,
    true_ratio: Array,
    estimated_ratio: Array,
    raw_ratio: Array | None = None,
    reference_weights: Array | None = None,
    feature_matrix: Array | None = None,
) -> dict[str, float]:
    raw = estimated_ratio if raw_ratio is None else raw_ratio
    out = {}
    out.update(ratio_quality(true_ratio, estimated_ratio))
    out.update(calibration_by_quantile(true_ratio, estimated_ratio))
    out.update(summarize_weights(estimated_ratio, raw_weights=raw))
    out.update(truth_heterogeneity_diagnostics(true_ratio, estimated_ratio))
    out["normalization_error"] = normalization_error(estimated_ratio, reference_weights)
    out["bellman_flow_residual_l2"] = bellman_flow_residual(
        estimated_ratio,
        true_ratio,
        feature_matrix=feature_matrix,
    )
    out["ratio_truth_available"] = 1.0
    return out


def estimator_diagnostics_optional(
    *,
    true_ratio: Array | None,
    estimated_ratio: Array,
    raw_ratio: Array | None = None,
    reference_weights: Array | None = None,
    feature_matrix: Array | None = None,
) -> dict[str, float]:
    if true_ratio is not None:
        return estimator_diagnostics(
            true_ratio=true_ratio,
            estimated_ratio=estimated_ratio,
            raw_ratio=raw_ratio,
            reference_weights=reference_weights,
            feature_matrix=feature_matrix,
        )
    raw = estimated_ratio if raw_ratio is None else raw_ratio
    out = summarize_weights(estimated_ratio, raw_weights=raw)
    out["normalization_error"] = normalization_error(estimated_ratio, reference_weights)
    out["ratio_truth_available"] = 0.0
    return out


def _finite_float_or_none(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def summarize_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    keys = ("profile", "stage", "setting", "policy_shift", "estimator", "gamma", "sample_size", "status")
    for row in rows:
        groups[tuple(row.get(key) for key in keys)].append(row)

    summary = []
    for group_key, group_rows in groups.items():
        out = {key: value for key, value in zip(keys, group_key)}
        out["n_runs"] = len(group_rows)
        numeric_keys = sorted(
            {
                key
                for row in group_rows
                for key, value in row.items()
                if _finite_float_or_none(value) is not None
            }
        )
        for key in numeric_keys:
            vals = np.asarray(
                [
                    numeric
                    for row in group_rows
                    if key in row
                    for numeric in (_finite_float_or_none(row[key]),)
                    if numeric is not None
                ]
            )
            if vals.size == 0:
                continue
            out[f"{key}_mean"] = float(np.mean(vals))
            out[f"{key}_std"] = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
        summary.append(out)
    return summary
