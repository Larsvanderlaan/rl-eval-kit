from __future__ import annotations

from typing import Any

import numpy as np


Array = np.ndarray


def calibrate_occupancy_bellman_binning(
    omega_hat: Array,
    h: Array,
    h_next: Array,
    init_moments: Array,
    gamma: float,
    score: Array | None = None,
    n_bins: int = 10,
    min_bin_size: int = 30,
    w_max: float | None = None,
    lambda_bellman: float = 1.0,
    lambda_shrink: float = 1.0,
    ridge: float = 1e-6,
    normalize: bool = True,
    return_diagnostics: bool = True,
) -> dict[str, Any] | Array:
    """Post-hoc backward occupancy Bellman-moment calibration.

    This conservative calibration step adjusts a fitted occupancy ratio
    estimate with a small number of nonnegative bin-level multipliers. The
    multipliers are chosen to reduce empirical backward occupancy Bellman
    moment residuals while shrinking toward one, optionally respecting a final
    weight cap. Under limited coverage this only stabilizes supported weighted
    moments; it does not recover unsupported target occupancy mass.

    Parameters
    ----------
    omega_hat:
        Fitted occupancy ratio estimates on calibration rows.
    h:
        Bellman test functions evaluated at the current state-action rows.
    h_next:
        Bellman test functions evaluated at next-state/target-action rows.
    init_moments:
        Initial-law moments for the same test functions.
    gamma:
        Discount factor in ``[0, 1)``.
    score:
        Optional one-dimensional score used for quantile binning. When omitted,
        ``log1p(max(omega_hat, 0))`` is used.
    n_bins:
        Requested number of score bins.
    min_bin_size:
        Minimum bin size when the sample size permits it.
    w_max:
        Optional cap for final calibrated weights.
    lambda_bellman:
        Weight on squared backward occupancy Bellman residuals.
    lambda_shrink:
        Weight on bin-count weighted shrinkage toward multiplier one.
    ridge:
        Additional ridge stabilization on multipliers.
    normalize:
        Whether to rescale calibrated weights toward unit mean after solving.
    return_diagnostics:
        When true, return a dictionary with weights and diagnostics. Otherwise
        return only the calibrated weights.
    """

    omega = _as_finite_vector(omega_hat, "omega_hat")
    n = int(omega.shape[0])
    h_arr = _as_finite_matrix(h, "h", n)
    h_next_arr = _as_finite_matrix(h_next, "h_next", n)
    if h_arr.shape != h_next_arr.shape:
        raise ValueError("h and h_next must have the same shape.")
    init = _as_finite_vector(init_moments, "init_moments")
    if init.shape[0] != h_arr.shape[1]:
        raise ValueError("init_moments length must match the number of columns in h.")
    gamma_f = float(gamma)
    if not (0.0 <= gamma_f < 1.0):
        raise ValueError("gamma must be in [0, 1).")
    bins_requested = int(n_bins)
    min_size = int(min_bin_size)
    if bins_requested <= 0:
        raise ValueError("n_bins must be positive.")
    if min_size <= 0:
        raise ValueError("min_bin_size must be positive.")
    lambda_bellman_f = _as_nonnegative_scalar(lambda_bellman, "lambda_bellman")
    lambda_shrink_f = _as_nonnegative_scalar(lambda_shrink, "lambda_shrink")
    ridge_f = _as_nonnegative_scalar(ridge, "ridge")
    if lambda_bellman_f + lambda_shrink_f + ridge_f <= 0.0:
        raise ValueError("At least one objective weight must be positive.")
    if w_max is not None:
        w_max_f = float(w_max)
        if not np.isfinite(w_max_f) or w_max_f <= 0.0:
            raise ValueError("w_max must be positive when supplied.")
    else:
        w_max_f = None

    if score is None:
        score_arr = np.log1p(np.maximum(omega, 0.0))
    else:
        score_arr = _as_finite_vector(score, "score")
        if score_arr.shape[0] != n:
            raise ValueError("score must have the same length as omega_hat.")

    bin_id = _make_quantile_bins(score_arr, n_bins=bins_requested, min_bin_size=min_size)
    counts = np.bincount(bin_id, minlength=int(np.max(bin_id)) + 1).astype(np.float64)
    n_effective_bins = int(counts.shape[0])

    delta_h = h_arr - gamma_f * h_next_arr
    moments_by_bin = np.zeros((h_arr.shape[1], n_effective_bins), dtype=np.float64)
    for bin_idx in range(n_effective_bins):
        mask = bin_id == bin_idx
        if np.any(mask):
            moments_by_bin[:, bin_idx] = np.sum(omega[mask, None] * delta_h[mask], axis=0) / float(n)
    target = (1.0 - gamma_f) * init

    lower = np.zeros(n_effective_bins, dtype=np.float64)
    upper = np.full(n_effective_bins, np.inf, dtype=np.float64)
    if w_max_f is not None:
        for bin_idx in range(n_effective_bins):
            max_omega = float(np.max(omega[bin_id == bin_idx]))
            if max_omega > 0.0:
                upper[bin_idx] = w_max_f / max_omega

    raw_multipliers = _solve_box_qp_coordinate_descent(
        moments_by_bin,
        target,
        counts,
        lower=lower,
        upper=upper,
        lambda_bellman=lambda_bellman_f,
        lambda_shrink=lambda_shrink_f,
        ridge=ridge_f,
    )
    raw_weights = raw_multipliers[bin_id] * omega
    normalization_scale = 1.0
    if normalize:
        raw_mean = float(np.mean(raw_weights))
        if np.isfinite(raw_mean) and raw_mean > 1e-12:
            normalization_scale = 1.0 / raw_mean
            if w_max_f is not None:
                max_raw = float(np.max(raw_weights))
                if max_raw > 0.0:
                    normalization_scale = min(normalization_scale, w_max_f / max_raw)
    multipliers = raw_multipliers * normalization_scale
    omega_cal = multipliers[bin_id] * omega
    if w_max_f is not None:
        omega_cal = np.minimum(omega_cal, w_max_f)
        clipped_fraction = float(np.mean(raw_weights * normalization_scale > w_max_f + 1e-12))
    else:
        clipped_fraction = np.nan

    residual_before = _bellman_residual(moments_by_bin, np.ones(n_effective_bins, dtype=np.float64), target)
    residual_after = np.mean(omega_cal[:, None] * delta_h, axis=0) - target
    bin_contrib_before = moments_by_bin
    bin_contrib_after = moments_by_bin * multipliers.reshape(1, -1)
    bin_weight_mean_before = np.zeros(n_effective_bins, dtype=np.float64)
    bin_weight_mean_after = np.zeros(n_effective_bins, dtype=np.float64)
    bin_weight_max_before = np.zeros(n_effective_bins, dtype=np.float64)
    bin_weight_max_after = np.zeros(n_effective_bins, dtype=np.float64)
    for bin_idx in range(n_effective_bins):
        mask = bin_id == bin_idx
        if np.any(mask):
            bin_weight_mean_before[bin_idx] = float(np.mean(omega[mask]))
            bin_weight_mean_after[bin_idx] = float(np.mean(omega_cal[mask]))
            bin_weight_max_before[bin_idx] = float(np.max(omega[mask]))
            bin_weight_max_after[bin_idx] = float(np.max(omega_cal[mask]))
    objective_value = _objective_value(
        moments_by_bin,
        target,
        multipliers,
        counts,
        lambda_bellman=lambda_bellman_f,
        lambda_shrink=lambda_shrink_f,
        ridge=ridge_f,
    )
    diagnostics = {
        "multipliers": multipliers,
        "raw_multipliers": raw_multipliers,
        "normalization_scale": float(normalization_scale),
        "bin_id": bin_id,
        "bin_counts": counts.astype(np.int64),
        "objective_value": float(objective_value),
        "bellman_residual_before": residual_before,
        "bellman_residual_after": residual_after,
        "residual_norm_before": float(np.linalg.norm(residual_before)),
        "residual_norm_after": float(np.linalg.norm(residual_after)),
        "bin_moment_contribution_norm_before": np.linalg.norm(bin_contrib_before, axis=0),
        "bin_moment_contribution_norm_after": np.linalg.norm(bin_contrib_after, axis=0),
        "bin_weight_mean_before": bin_weight_mean_before,
        "bin_weight_mean_after": bin_weight_mean_after,
        "bin_weight_max_before": bin_weight_max_before,
        "bin_weight_max_after": bin_weight_max_after,
        "ess_before": _effective_sample_size(omega),
        "ess_after": _effective_sample_size(omega_cal),
        "weight_q95_before": float(np.quantile(omega, 0.95)),
        "weight_q95_after": float(np.quantile(omega_cal, 0.95)),
        "weight_q99_before": float(np.quantile(omega, 0.99)),
        "weight_q99_after": float(np.quantile(omega_cal, 0.99)),
        "max_weight_before": float(np.max(omega)),
        "max_weight_after": float(np.max(omega_cal)),
    }
    diagnostics.update(
        recommend_occupancy_bellman_calibration(
            diagnostics,
            omega_before=omega,
            omega_after=omega_cal,
        )
    )
    if w_max_f is not None:
        diagnostics["clipped_fraction"] = clipped_fraction
    if not return_diagnostics:
        return omega_cal
    return {"omega_cal": omega_cal, "diagnostics": diagnostics}


def occupancy_bellman_calibration_diagnostics(
    omega_before: Array,
    omega_after: Array,
    h: Array,
    h_next: Array,
    init_moments: Array,
    gamma: float,
    *,
    score: Array | None = None,
    n_bins: int = 10,
    min_bin_size: int = 30,
) -> dict[str, Any]:
    """Diagnose a post-hoc Bellman-moment calibration without oracle truth.

    The diagnostics report global and score-bin backward occupancy Bellman
    moment imbalance before and after calibration, weight-tail changes, ESS
    changes, and a non-oracle recommendation on whether to apply the calibrated
    weights. The recommendation is conservative: it favors applying calibration
    only when Bellman residuals improve without a material ESS or tail cost.
    """

    before = _as_finite_vector(omega_before, "omega_before")
    after = _as_finite_vector(omega_after, "omega_after")
    if before.shape != after.shape:
        raise ValueError("omega_before and omega_after must have the same shape.")
    n = int(before.shape[0])
    h_arr = _as_finite_matrix(h, "h", n)
    h_next_arr = _as_finite_matrix(h_next, "h_next", n)
    if h_arr.shape != h_next_arr.shape:
        raise ValueError("h and h_next must have the same shape.")
    init = _as_finite_vector(init_moments, "init_moments")
    if init.shape[0] != h_arr.shape[1]:
        raise ValueError("init_moments length must match the number of columns in h.")
    gamma_f = float(gamma)
    if not (0.0 <= gamma_f < 1.0):
        raise ValueError("gamma must be in [0, 1).")
    if score is None:
        score_arr = np.log1p(np.maximum(before, 0.0))
    else:
        score_arr = _as_finite_vector(score, "score")
        if score_arr.shape[0] != n:
            raise ValueError("score must have the same length as omega_before.")
    bin_id = _make_quantile_bins(score_arr, n_bins=int(n_bins), min_bin_size=int(min_bin_size))
    bin_count = int(np.max(bin_id)) + 1
    counts = np.bincount(bin_id, minlength=bin_count).astype(np.int64)
    delta_h = h_arr - gamma_f * h_next_arr
    target = (1.0 - gamma_f) * init
    residual_before = np.mean(before[:, None] * delta_h, axis=0) - target
    residual_after = np.mean(after[:, None] * delta_h, axis=0) - target
    table = []
    for bin_idx in range(bin_count):
        mask = bin_id == bin_idx
        b_resid = np.mean(before[mask, None] * delta_h[mask], axis=0) if np.any(mask) else np.zeros_like(target)
        a_resid = np.mean(after[mask, None] * delta_h[mask], axis=0) if np.any(mask) else np.zeros_like(target)
        table.append(
            {
                "bin": int(bin_idx),
                "count": int(counts[bin_idx]),
                "score_min": float(np.min(score_arr[mask])) if np.any(mask) else np.nan,
                "score_max": float(np.max(score_arr[mask])) if np.any(mask) else np.nan,
                "weight_mean_before": float(np.mean(before[mask])) if np.any(mask) else np.nan,
                "weight_mean_after": float(np.mean(after[mask])) if np.any(mask) else np.nan,
                "weight_q99_before": float(np.quantile(before[mask], 0.99)) if np.any(mask) else np.nan,
                "weight_q99_after": float(np.quantile(after[mask], 0.99)) if np.any(mask) else np.nan,
                "weight_max_before": float(np.max(before[mask])) if np.any(mask) else np.nan,
                "weight_max_after": float(np.max(after[mask])) if np.any(mask) else np.nan,
                "bellman_contribution_norm_before": float(np.linalg.norm(b_resid)),
                "bellman_contribution_norm_after": float(np.linalg.norm(a_resid)),
            }
        )
    diagnostics: dict[str, Any] = {
        "bin_id": bin_id,
        "bin_counts": counts,
        "bin_table": table,
        "bellman_residual_before": residual_before,
        "bellman_residual_after": residual_after,
        "residual_norm_before": float(np.linalg.norm(residual_before)),
        "residual_norm_after": float(np.linalg.norm(residual_after)),
        "ess_before": _effective_sample_size(before),
        "ess_after": _effective_sample_size(after),
        "weight_mean_before": float(np.mean(before)),
        "weight_mean_after": float(np.mean(after)),
        "weight_q95_before": float(np.quantile(before, 0.95)),
        "weight_q95_after": float(np.quantile(after, 0.95)),
        "weight_q99_before": float(np.quantile(before, 0.99)),
        "weight_q99_after": float(np.quantile(after, 0.99)),
        "max_weight_before": float(np.max(before)),
        "max_weight_after": float(np.max(after)),
    }
    diagnostics.update(
        recommend_occupancy_bellman_calibration(
            diagnostics,
            omega_before=before,
            omega_after=after,
        )
    )
    return diagnostics


def recommend_occupancy_bellman_calibration(
    diagnostics: dict[str, Any],
    *,
    omega_before: Array | None = None,
    omega_after: Array | None = None,
    min_residual_reduction: float = 1e-3,
    max_ess_loss_fraction: float = 0.02,
    max_q99_increase_fraction: float = 0.05,
    max_max_weight_increase_fraction: float = 0.05,
) -> dict[str, Any]:
    """Return a conservative recommendation for applying calibration.

    The rule uses only empirical Bellman-moment and weight-tail diagnostics. It
    does not use oracle ratios or target values. The result is intentionally
    conservative under limited coverage: calibration is recommended only when
    it gives a meaningful Bellman residual reduction without material ESS or
    tail degradation.
    """

    before_norm = float(diagnostics.get("residual_norm_before", np.nan))
    after_norm = float(diagnostics.get("residual_norm_after", np.nan))
    ess_before = float(diagnostics.get("ess_before", np.nan))
    ess_after = float(diagnostics.get("ess_after", np.nan))
    if omega_before is not None:
        w_before = _as_finite_vector(omega_before, "omega_before")
        q99_before = float(np.quantile(w_before, 0.99))
        max_before = float(np.max(w_before))
    else:
        q99_before = float(diagnostics.get("weight_q99_before", np.nan))
        max_before = float(diagnostics.get("max_weight_before", np.nan))
    if omega_after is not None:
        w_after = _as_finite_vector(omega_after, "omega_after")
        q99_after = float(np.quantile(w_after, 0.99))
        max_after = float(np.max(w_after))
    else:
        q99_after = float(diagnostics.get("weight_q99_after", np.nan))
        max_after = float(diagnostics.get("max_weight_after", np.nan))

    residual_reduction = _safe_relative_drop(before_norm, after_norm)
    ess_loss_fraction = _safe_relative_drop(ess_before, ess_after)
    q99_increase_fraction = _safe_relative_increase(q99_before, q99_after)
    max_weight_increase = _safe_relative_increase(max_before, max_after)

    reasons = []
    if not np.isfinite(residual_reduction):
        recommendation = "do_not_apply"
        reasons.append("Bellman residual diagnostics are not finite.")
    elif after_norm > before_norm + 1e-12:
        recommendation = "do_not_apply"
        reasons.append("Bellman residual norm increases after calibration.")
    elif ess_loss_fraction > max_ess_loss_fraction:
        recommendation = "do_not_apply"
        reasons.append("Effective sample size loss exceeds the configured tolerance.")
    elif q99_increase_fraction > max_q99_increase_fraction:
        recommendation = "do_not_apply"
        reasons.append("99th-percentile weight tail increases beyond the configured tolerance.")
    elif max_weight_increase > max_max_weight_increase_fraction:
        recommendation = "do_not_apply"
        reasons.append("Maximum weight increases beyond the configured tolerance.")
    elif residual_reduction >= min_residual_reduction:
        recommendation = "apply"
        reasons.append("Bellman residual reduction is meaningful and tail/ESS costs are within tolerance.")
    else:
        recommendation = "neutral"
        reasons.append("Calibration is safe by diagnostics but the Bellman residual reduction is too small to matter.")

    return {
        "calibration_recommendation": recommendation,
        "calibration_recommendation_reasons": reasons,
        "residual_reduction_fraction": float(residual_reduction),
        "ess_loss_fraction": float(ess_loss_fraction),
        "q99_increase_fraction": float(q99_increase_fraction),
        "max_weight_increase_fraction": float(max_weight_increase),
        "recommendation_thresholds": {
            "min_residual_reduction": float(min_residual_reduction),
            "max_ess_loss_fraction": float(max_ess_loss_fraction),
            "max_q99_increase_fraction": float(max_q99_increase_fraction),
            "max_weight_increase_fraction": float(max_max_weight_increase_fraction),
        },
    }


def plot_occupancy_bellman_calibration_diagnostics(
    diagnostics: dict[str, Any],
    *,
    path: str | None = None,
    show: bool = False,
):
    """Plot user-facing backward occupancy Bellman calibration diagnostics.

    The figure summarizes global Bellman residual change, ESS/tail changes,
    and score-bin Bellman contribution norms. Matplotlib is imported lazily so
    the core package does not require plotting dependencies at import time.
    """

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required for calibration diagnostic plots.") from exc

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    before_norm = float(diagnostics.get("residual_norm_before", np.nan))
    after_norm = float(diagnostics.get("residual_norm_after", np.nan))
    axes[0, 0].bar(["before", "after"], [before_norm, after_norm], color=["#777777", "#2b6cb0"])
    axes[0, 0].set_title("Bellman Moment Residual")
    axes[0, 0].set_ylabel("L2 norm")

    ess_before = float(diagnostics.get("ess_before", np.nan))
    ess_after = float(diagnostics.get("ess_after", np.nan))
    axes[0, 1].bar(["before", "after"], [ess_before, ess_after], color=["#777777", "#2b6cb0"])
    axes[0, 1].set_title("Effective Sample Size")

    q99_before = float(diagnostics.get("weight_q99_before", np.nan))
    q99_after = float(diagnostics.get("weight_q99_after", np.nan))
    if not np.isfinite(q99_before):
        q99_before = float(np.nan)
    if not np.isfinite(q99_after):
        q99_after = float(np.nan)
    max_before = float(diagnostics.get("max_weight_before", np.nan))
    max_after = float(diagnostics.get("max_weight_after", np.nan))
    x = np.arange(2)
    axes[1, 0].bar(x - 0.18, [q99_before, max_before], width=0.36, label="before", color="#777777")
    axes[1, 0].bar(x + 0.18, [q99_after, max_after], width=0.36, label="after", color="#2b6cb0")
    axes[1, 0].set_xticks(x, ["q99", "max"])
    axes[1, 0].set_title("Weight Tail")
    axes[1, 0].legend()

    table = diagnostics.get("bin_table")
    if table:
        bins = [row["bin"] for row in table]
        contrib_before = [row["bellman_contribution_norm_before"] for row in table]
        contrib_after = [row["bellman_contribution_norm_after"] for row in table]
    else:
        bins = list(range(len(diagnostics.get("bin_moment_contribution_norm_before", []))))
        contrib_before = list(diagnostics.get("bin_moment_contribution_norm_before", []))
        contrib_after = list(diagnostics.get("bin_moment_contribution_norm_after", []))
    bin_positions = np.asarray(bins, dtype=np.float64)
    axes[1, 1].bar(bin_positions - 0.18, contrib_before, width=0.36, label="before", color="#777777")
    axes[1, 1].bar(bin_positions + 0.18, contrib_after, width=0.36, label="after", color="#2b6cb0")
    axes[1, 1].set_title("Bin Bellman Contribution Norm")
    axes[1, 1].set_xlabel("score bin")
    axes[1, 1].legend()

    recommendation = diagnostics.get("calibration_recommendation")
    if recommendation:
        fig.suptitle(f"Bellman-Moment Calibration: {recommendation}", fontsize=13)
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def estimate_ope_bellman_control_variate(
    weights: Array,
    rewards: Array,
    h: Array,
    h_next: Array,
    init_moments: Array,
    gamma: float,
    *,
    ridge: float = 1e-6,
) -> dict[str, Any]:
    """Estimate OPE value with a Bellman-moment control variate.

    This optional diagnostic estimator uses the empirical identity
    ``E[omega * (h - gamma h_next)] = (1 - gamma) E_0[h]`` as a control
    variate. It is useful for sensitivity analysis of downstream OPE, but it
    is not an oracle guarantee and should be reported beside the raw weighted
    estimate.
    """

    w = _as_finite_vector(weights, "weights")
    r = _as_finite_vector(rewards, "rewards")
    if r.shape != w.shape:
        raise ValueError("rewards and weights must have the same length.")
    h_arr = _as_finite_matrix(h, "h", w.shape[0])
    h_next_arr = _as_finite_matrix(h_next, "h_next", w.shape[0])
    if h_arr.shape != h_next_arr.shape:
        raise ValueError("h and h_next must have the same shape.")
    init = _as_finite_vector(init_moments, "init_moments")
    if init.shape[0] != h_arr.shape[1]:
        raise ValueError("init_moments length must match the number of columns in h.")
    gamma_f = float(gamma)
    if not (0.0 <= gamma_f < 1.0):
        raise ValueError("gamma must be in [0, 1).")
    ridge_f = _as_nonnegative_scalar(ridge, "ridge")

    delta_h = h_arr - gamma_f * h_next_arr
    x = w[:, None] * delta_h
    y = w * r
    x_centered = x - np.mean(x, axis=0, keepdims=True)
    y_centered = y - float(np.mean(y))
    gram = x_centered.T @ x_centered / float(w.shape[0]) + ridge_f * np.eye(x.shape[1], dtype=np.float64)
    rhs = x_centered.T @ y_centered / float(w.shape[0])
    beta = np.linalg.solve(gram, rhs)
    raw_estimate = float(np.mean(y))
    observed_moment = np.mean(x, axis=0)
    target_moment = (1.0 - gamma_f) * init
    corrected_estimate = float(raw_estimate - beta @ (observed_moment - target_moment))
    return {
        "raw_weighted_value": raw_estimate,
        "bellman_control_variate_value": corrected_estimate,
        "correction": float(corrected_estimate - raw_estimate),
        "beta": beta,
        "moment_residual": observed_moment - target_moment,
        "moment_residual_norm": float(np.linalg.norm(observed_moment - target_moment)),
    }


def _as_finite_vector(value: Array, name: str) -> Array:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} must be nonempty.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values.")
    return arr


def _as_finite_matrix(value: Array, name: str, n_rows: int) -> Array:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 1D or 2D array.")
    if arr.shape[0] != int(n_rows):
        raise ValueError(f"{name} must have {int(n_rows)} rows.")
    if arr.shape[1] == 0:
        raise ValueError(f"{name} must have at least one column.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values.")
    return arr


def _as_nonnegative_scalar(value: float, name: str) -> float:
    out = float(value)
    if not np.isfinite(out) or out < 0.0:
        raise ValueError(f"{name} must be nonnegative.")
    return out


def _make_quantile_bins(score: Array, *, n_bins: int, min_bin_size: int) -> Array:
    n = int(score.shape[0])
    if n <= 0:
        raise ValueError("score must be nonempty.")
    max_bins_by_size = max(1, n // int(min_bin_size))
    effective_bins = min(int(n_bins), n, max_bins_by_size)
    order = np.argsort(score, kind="mergesort")
    bin_id = np.empty(n, dtype=np.int64)
    for bin_idx, indices in enumerate(np.array_split(order, effective_bins)):
        bin_id[indices] = int(bin_idx)
    return bin_id


def _solve_box_qp_coordinate_descent(
    moments_by_bin: Array,
    target: Array,
    counts: Array,
    *,
    lower: Array,
    upper: Array,
    lambda_bellman: float,
    lambda_shrink: float,
    ridge: float,
    max_iter: int = 10_000,
    tol: float = 1e-11,
) -> Array:
    b = int(counts.shape[0])
    a = np.clip(np.ones(b, dtype=np.float64), lower, upper)
    hessian = lambda_bellman * (moments_by_bin.T @ moments_by_bin)
    hessian += np.diag(lambda_shrink * counts + ridge)
    rhs = lambda_bellman * (moments_by_bin.T @ target) + lambda_shrink * counts

    diag = np.diag(hessian)
    for _ in range(int(max_iter)):
        max_change = 0.0
        for idx in range(b):
            if diag[idx] <= 1e-18:
                continue
            partial = float(hessian[idx] @ a - diag[idx] * a[idx])
            proposal = (float(rhs[idx]) - partial) / float(diag[idx])
            proposal = min(max(proposal, float(lower[idx])), float(upper[idx]))
            change = abs(proposal - float(a[idx]))
            if change > max_change:
                max_change = change
            a[idx] = proposal
        if max_change <= tol:
            break
    return a


def _bellman_residual(moments_by_bin: Array, multipliers: Array, target: Array) -> Array:
    return moments_by_bin @ multipliers - target


def _objective_value(
    moments_by_bin: Array,
    target: Array,
    multipliers: Array,
    counts: Array,
    *,
    lambda_bellman: float,
    lambda_shrink: float,
    ridge: float,
) -> float:
    residual = _bellman_residual(moments_by_bin, multipliers, target)
    shrink = multipliers - 1.0
    return float(
        lambda_bellman * np.sum(residual**2)
        + lambda_shrink * np.sum(counts * shrink**2)
        + ridge * np.sum(multipliers**2)
    )


def _effective_sample_size(weights: Array) -> float:
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    return float((np.sum(w) ** 2) / max(float(np.sum(w**2)), 1e-12))


def _safe_relative_drop(before: float, after: float) -> float:
    if not (np.isfinite(before) and np.isfinite(after)):
        return float("nan")
    denom = max(abs(float(before)), 1e-12)
    return float((float(before) - float(after)) / denom)


def _safe_relative_increase(before: float, after: float) -> float:
    if not (np.isfinite(before) and np.isfinite(after)):
        return float("nan")
    denom = max(abs(float(before)), 1e-12)
    return float((float(after) - float(before)) / denom)
