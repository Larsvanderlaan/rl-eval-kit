from __future__ import annotations

import numpy as np
import pytest

from fqe import (
    BellmanCalibrator,
    bellman_calibration_diagnostics,
    fit_bellman_calibrator,
    fit_q_bellman_calibrator,
    fit_value_bellman_calibrator,
    plot_bellman_calibration_diagnostics,
    recommend_bellman_calibration,
)


def test_public_calibration_exports() -> None:
    assert BellmanCalibrator.__name__ == "BellmanCalibrator"
    assert callable(fit_bellman_calibrator)
    assert callable(bellman_calibration_diagnostics)
    assert callable(recommend_bellman_calibration)


def test_calibration_validates_inputs() -> None:
    pred = np.ones(5)
    next_pred = np.ones(4)
    rewards = np.ones(5)
    with pytest.raises(ValueError, match="same length"):
        fit_bellman_calibrator(pred, next_pred, rewards, gamma=0.9)
    with pytest.raises(ValueError, match="gamma"):
        fit_bellman_calibrator(pred, np.ones(5), rewards, gamma=1.0)
    with pytest.raises(ValueError, match="method"):
        fit_bellman_calibrator(pred, np.ones(5), rewards, gamma=0.9, method="bad")
    with pytest.raises(ValueError, match="nonnegative"):
        fit_bellman_calibrator(pred, np.ones(5), rewards, gamma=0.9, sample_weight=np.array([1, 1, -1, 1, 1]))


def test_small_sample_too_many_bins_falls_back() -> None:
    pred = np.linspace(-1.0, 1.0, 7)
    rewards = pred - 0.5
    cal = fit_bellman_calibrator(pred, np.zeros_like(pred), rewards, gamma=0.0, n_bins=50, min_bin_size=30)
    assert cal.predict(pred).shape == pred.shape
    assert int(np.sum(cal.bin_counts)) == pred.shape[0]
    diag = bellman_calibration_diagnostics(pred, np.zeros_like(pred), rewards, 0.0, calibrator=cal)
    assert diag["bellman_calibration_test_size"] == 7


def test_miscalibrated_shift_improves_and_recommends_apply() -> None:
    pred = np.linspace(-2.0, 2.0, 160)
    rewards = pred - 1.0
    cal = fit_bellman_calibrator(
        pred,
        np.zeros_like(pred),
        rewards,
        gamma=0.0,
        method="histogram_rescale",
        n_bins=8,
        min_bin_size=10,
    )
    diag = bellman_calibration_diagnostics(
        pred,
        np.zeros_like(pred),
        rewards,
        gamma=0.0,
        calibrator=cal,
        n_bins=8,
        min_bin_size=10,
    )
    assert diag["bellman_residual_mse_after"] < diag["bellman_residual_mse_before"]
    assert diag["bellman_calibration_error_plugin_after"] < diag["bellman_calibration_error_plugin_before"]
    assert diag["calibration_recommendation"] == "apply"


def test_already_calibrated_is_neutral_not_harmful() -> None:
    pred = np.linspace(-1.0, 1.0, 120)
    rewards = pred.copy()
    cal = fit_bellman_calibrator(pred, np.zeros_like(pred), rewards, gamma=0.0, n_bins=6, min_bin_size=10)
    diag = bellman_calibration_diagnostics(
        pred,
        np.zeros_like(pred),
        rewards,
        0.0,
        calibrator=cal,
        n_bins=6,
        min_bin_size=10,
    )
    assert diag["calibration_recommendation"] in {"neutral", "apply"}
    assert diag["bellman_residual_mse_after"] <= diag["bellman_residual_mse_before"] + 1e-12


def test_histogram_rescale_preserves_within_bin_shape() -> None:
    pred = np.linspace(0.0, 9.9, 100)
    rewards = pred + np.where(pred < 5.0, 2.0, -3.0)
    cal = fit_bellman_calibrator(
        pred,
        np.zeros_like(pred),
        rewards,
        gamma=0.0,
        method="histogram_rescale",
        n_bins=2,
        min_bin_size=10,
    )
    calibrated = cal.predict(pred)
    bin_id = cal.bin_indices(pred)
    for idx in range(int(np.max(bin_id)) + 1):
        mask = bin_id == idx
        raw_centered = pred[mask] - float(np.mean(pred[mask]))
        cal_centered = calibrated[mask] - float(np.mean(calibrated[mask]))
        assert np.allclose(cal_centered, raw_centered)


def test_terminal_mask_blocks_next_prediction_in_targets() -> None:
    pred = np.full(80, 10.0)
    next_pred = np.full(80, 100.0)
    rewards = np.full(80, 2.0)
    cal = fit_bellman_calibrator(
        pred,
        next_pred,
        rewards,
        gamma=0.9,
        terminals=np.ones_like(pred),
        method="histogram_constant",
        n_bins=4,
        min_bin_size=10,
    )
    assert np.allclose(cal.predict(pred), 2.0)


def test_sample_weights_affect_bin_target_means() -> None:
    pred = np.concatenate([np.zeros(10), np.ones(10)])
    rewards = np.concatenate([np.zeros(10), np.full(10, 10.0)])
    weights = np.concatenate([np.ones(10), np.full(10, 5.0)])
    cal = fit_bellman_calibrator(
        pred,
        np.zeros_like(pred),
        rewards,
        gamma=0.0,
        sample_weight=weights,
        method="histogram_constant",
        n_bins=1,
        min_bin_size=1,
    )
    assert np.isclose(cal.predict(np.array([0.5]))[0], np.average(rewards, weights=weights))


def test_isotonic_histogram_learns_monotone_bins() -> None:
    rng = np.random.default_rng(7)
    pred = np.linspace(-2.0, 2.0, 120)
    rewards = pred + 0.1 * rng.normal(size=pred.shape[0])
    cal = fit_bellman_calibrator(
        pred,
        np.zeros_like(pred),
        rewards,
        gamma=0.0,
        method="isotonic_histogram_constant",
        n_bins=8,
        min_bin_size=8,
    )
    assert np.all(np.diff(cal.bin_target_mean) >= -1e-12)
    assert np.all(np.isfinite(cal.predict(pred)))


class _DummyQModel:
    def predict_q(self, states, actions):
        return np.asarray(states, dtype=float).reshape(len(states), -1)[:, 0] + np.asarray(actions, dtype=float).reshape(
            len(states), -1
        )[:, 0]


class _DummyValueModel:
    def predict_value(self, states):
        return np.asarray(states, dtype=float).reshape(len(states), -1)[:, 0]


def test_q_and_value_model_helpers_with_dummy_models() -> None:
    states = np.linspace(0.0, 1.0, 40).reshape(-1, 1)
    actions = np.ones((40, 1))
    next_states = states + 0.1
    next_actions = np.stack([np.zeros((40, 1)), np.ones((40, 1))], axis=1)
    rewards = states.reshape(-1)
    q_cal = fit_q_bellman_calibrator(
        _DummyQModel(),
        states,
        actions,
        next_states,
        next_actions,
        rewards,
        gamma=0.5,
        n_bins=4,
        min_bin_size=5,
    )
    assert np.all(np.isfinite(q_cal.predict(np.linspace(0.0, 1.0, 5))))

    value_cal = fit_value_bellman_calibrator(
        _DummyValueModel(),
        states,
        next_states,
        rewards,
        gamma=0.5,
        n_bins=4,
        min_bin_size=5,
    )
    assert np.all(np.isfinite(value_cal.predict(np.linspace(0.0, 1.0, 5))))


def test_plot_bellman_calibration_diagnostics_smoke(tmp_path) -> None:
    pytest.importorskip("matplotlib")
    pred = np.linspace(-1.0, 1.0, 80)
    rewards = pred - 0.25
    cal = fit_bellman_calibrator(pred, np.zeros_like(pred), rewards, gamma=0.0, n_bins=4, min_bin_size=10)
    diag = bellman_calibration_diagnostics(pred, np.zeros_like(pred), rewards, 0.0, calibrator=cal)
    path = tmp_path / "bellman_calibration.png"
    fig = plot_bellman_calibration_diagnostics(diag, path=str(path))
    assert path.exists()
    assert fig is not None
