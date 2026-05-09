from __future__ import annotations

import numpy as np
import pytest

from occupancy_ratio import (
    calibrate_occupancy_bellman_binning,
    estimate_ope_bellman_control_variate,
    occupancy_bellman_calibration_diagnostics,
    plot_occupancy_bellman_calibration_diagnostics,
)
from occupancy_ratio_benchmark.config import OccupancyRatioBenchmarkConfig
from occupancy_ratio_benchmark.discrete import make_discrete_dataset
from occupancy_ratio_benchmark.estimators import _bellman_moment_test_arrays
from occupancy_ratio_benchmark.runner import _expanded_estimators


def test_bellman_calibration_validates_dimensions() -> None:
    omega = np.ones(5)
    h = np.ones((5, 2))
    h_next = np.ones((4, 2))
    init = np.ones(2)
    with pytest.raises(ValueError, match="h_next"):
        calibrate_occupancy_bellman_binning(omega, h, h_next, init, gamma=0.9)
    with pytest.raises(ValueError, match="init_moments"):
        calibrate_occupancy_bellman_binning(omega, h, np.ones((5, 2)), np.ones(3), gamma=0.9)
    with pytest.raises(ValueError, match="gamma"):
        calibrate_occupancy_bellman_binning(omega, h, np.ones((5, 2)), init, gamma=1.0)
    with pytest.raises(ValueError, match="w_max"):
        calibrate_occupancy_bellman_binning(omega, h, np.ones((5, 2)), init, gamma=0.9, w_max=0.0)


def test_bellman_calibration_small_n_too_many_bins_does_not_crash() -> None:
    omega = np.linspace(0.5, 1.5, 7)
    h = np.column_stack([np.ones(7), np.linspace(-1.0, 1.0, 7)])
    out = calibrate_occupancy_bellman_binning(
        omega,
        h,
        0.25 * h,
        np.array([1.0, 0.0]),
        gamma=0.5,
        n_bins=20,
        min_bin_size=30,
    )
    assert out["omega_cal"].shape == omega.shape
    assert out["diagnostics"]["bin_counts"].sum() == omega.shape[0]


def test_bellman_calibration_multipliers_are_nonnegative_and_weights_finite() -> None:
    rng = np.random.default_rng(1)
    omega = np.exp(rng.normal(scale=0.25, size=80))
    h = np.column_stack([np.ones(80), rng.normal(size=80)])
    out = calibrate_occupancy_bellman_binning(
        omega,
        h,
        0.5 * h,
        np.array([1.0, 0.0]),
        gamma=0.4,
        n_bins=8,
        min_bin_size=10,
    )
    assert np.all(out["diagnostics"]["multipliers"] >= 0.0)
    assert np.all(np.isfinite(out["omega_cal"]))


def test_bellman_calibration_respects_w_max_after_normalization() -> None:
    omega = np.linspace(0.2, 20.0, 100)
    h = np.ones((100, 1))
    out = calibrate_occupancy_bellman_binning(
        omega,
        h,
        np.zeros_like(h),
        np.array([1.0]),
        gamma=0.0,
        n_bins=5,
        min_bin_size=10,
        w_max=3.0,
        normalize=True,
    )
    assert float(np.max(out["omega_cal"])) <= 3.0 + 1e-10
    assert "clipped_fraction" in out["diagnostics"]


def test_bellman_residual_norm_decreases_in_synthetic_example() -> None:
    omega = np.full(120, 0.5)
    h = np.ones((120, 1))
    out = calibrate_occupancy_bellman_binning(
        omega,
        h,
        np.zeros_like(h),
        np.array([1.0]),
        gamma=0.0,
        n_bins=4,
        min_bin_size=10,
        lambda_bellman=1.0,
        lambda_shrink=0.0,
        ridge=1e-9,
        normalize=True,
    )
    diag = out["diagnostics"]
    assert diag["residual_norm_after"] < diag["residual_norm_before"]


def test_bellman_calibration_normalize_gives_unit_mean_without_cap() -> None:
    omega = np.full(90, 0.75)
    h = np.ones((90, 1))
    out = calibrate_occupancy_bellman_binning(
        omega,
        h,
        np.zeros_like(h),
        np.array([1.0]),
        gamma=0.0,
        n_bins=3,
        min_bin_size=10,
        normalize=True,
    )
    assert np.isclose(np.mean(out["omega_cal"]), 1.0, atol=1e-10)


def test_benchmark_bellman_helper_builds_matching_arrays() -> None:
    dataset = make_discrete_dataset(setting="discrete_chain", gamma=0.9, sample_size=25, seed=11)
    h, h_next, init_moments = _bellman_moment_test_arrays(dataset)
    assert h.shape == h_next.shape
    assert h.shape[0] == dataset.n
    assert init_moments.shape == (h.shape[1],)
    assert np.allclose(h[:, 0], 1.0)
    assert np.isclose(init_moments[0], 1.0)


def test_estimator_expansion_includes_bellman_moment_presets() -> None:
    config = OccupancyRatioBenchmarkConfig(
        estimators=("boosted_tree", "neural_network"),
        boosted_estimator_presets=("bellman_moment_calibrated",),
        neural_estimator_presets=("bellman_moment_calibrated",),
    )
    assert _expanded_estimators(config) == (
        "boosted_tree_bellman_moment_calibrated",
        "neural_network_bellman_moment_calibrated",
    )


def test_calibration_diagnostics_recommendation_is_user_facing() -> None:
    omega = np.full(120, 0.5)
    h = np.ones((120, 1))
    h_next = np.zeros_like(h)
    init = np.array([1.0])
    calibrated = calibrate_occupancy_bellman_binning(
        omega,
        h,
        h_next,
        init,
        gamma=0.0,
        n_bins=4,
        min_bin_size=10,
        lambda_bellman=1.0,
        lambda_shrink=0.01,
        ridge=1e-6,
    )
    diag = occupancy_bellman_calibration_diagnostics(
        omega,
        calibrated["omega_cal"],
        h,
        h_next,
        init,
        gamma=0.0,
        n_bins=4,
        min_bin_size=10,
    )
    assert diag["calibration_recommendation"] in {"apply", "neutral", "do_not_apply"}
    assert diag["residual_reduction_fraction"] > 0.0
    assert len(diag["bin_table"]) == 4
    assert "calibration_recommendation_reasons" in diag


def test_calibration_diagnostics_flags_tail_cost() -> None:
    omega = np.ones(100)
    omega_bad = omega.copy()
    omega_bad[-1] = 50.0
    h = np.ones((100, 1))
    init = np.array([1.0])
    diag = occupancy_bellman_calibration_diagnostics(
        omega,
        omega_bad,
        h,
        np.zeros_like(h),
        init,
        gamma=0.0,
        n_bins=5,
        min_bin_size=10,
    )
    assert diag["calibration_recommendation"] == "do_not_apply"


def test_bellman_control_variate_value_returns_correction() -> None:
    x = np.linspace(-1.0, 1.0, 80)
    weights = np.ones_like(x)
    rewards = 2.0 + x
    h = np.column_stack([np.ones_like(x), x])
    h_next = np.zeros_like(h)
    init = np.array([1.0, 0.0])
    out = estimate_ope_bellman_control_variate(
        weights,
        rewards,
        h,
        h_next,
        init,
        gamma=0.0,
        ridge=1e-6,
    )
    assert np.isfinite(out["bellman_control_variate_value"])
    assert out["beta"].shape == (2,)
    assert abs(out["bellman_control_variate_value"] - 2.0) < abs(out["raw_weighted_value"] - 2.0) + 1e-8


def test_plot_calibration_diagnostics_smoke(tmp_path) -> None:
    pytest.importorskip("matplotlib")
    omega = np.full(60, 0.75)
    h = np.ones((60, 1))
    cal = calibrate_occupancy_bellman_binning(
        omega,
        h,
        np.zeros_like(h),
        np.array([1.0]),
        gamma=0.0,
        n_bins=3,
        min_bin_size=10,
    )
    path = tmp_path / "calibration_diagnostics.png"
    fig = plot_occupancy_bellman_calibration_diagnostics(cal["diagnostics"], path=str(path))
    assert path.exists()
    assert fig is not None
