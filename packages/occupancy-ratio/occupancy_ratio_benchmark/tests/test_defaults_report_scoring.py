from __future__ import annotations

from occupancy_ratio_benchmark.config import OccupancyRatioBenchmarkConfig
from occupancy_ratio_benchmark.defaults_report import _balanced_score, _balanced_winners
from occupancy_ratio_benchmark.runner import _expanded_estimators


def test_low_ess_matching_truth_beats_over_smoothed_weights() -> None:
    truth_matching = {
        "profile": "medium",
        "setting": "linear_gaussian",
        "gamma": 0.99,
        "sample_size": 1000,
        "seed": 0,
        "estimator": "neural_network_relaxed_tail",
        "status": "ok",
        "ratio_truth_available": 1.0,
        "ratio_normalized_l1": 0.05,
        "log_ratio_rmse": 0.10,
        "ope_value_abs_error": 0.10,
        "effective_sample_size_fraction": 0.06,
        "true_effective_sample_size_fraction": 0.05,
        "ess_fraction_abs_error_to_truth": 0.01,
    }
    over_smoothed = {
        **truth_matching,
        "estimator": "neural_network_stable",
        "ratio_normalized_l1": 0.30,
        "log_ratio_rmse": 0.50,
        "effective_sample_size_fraction": 0.95,
        "ess_fraction_abs_error_to_truth": 0.90,
    }

    assert _balanced_score(truth_matching) < _balanced_score(over_smoothed)


def test_ratio_l1_drives_truth_available_winner_not_ess() -> None:
    rows = [
        {
            "profile": "medium",
            "setting": "discrete_grid",
            "gamma": 0.99,
            "sample_size": 1000,
            "seed": 0,
            "estimator": "boosted_tree_stable",
            "status": "ok",
            "ratio_truth_available": 1.0,
            "ratio_normalized_l1": 0.25,
            "log_ratio_rmse": 0.40,
            "ope_value_abs_error": 0.01,
            "effective_sample_size_fraction": 0.95,
            "true_effective_sample_size_fraction": 0.08,
        },
        {
            "profile": "medium",
            "setting": "discrete_grid",
            "gamma": 0.99,
            "sample_size": 1000,
            "seed": 0,
            "estimator": "boosted_tree_relaxed_tail",
            "status": "ok",
            "ratio_truth_available": 1.0,
            "ratio_normalized_l1": 0.05,
            "log_ratio_rmse": 0.12,
            "ope_value_abs_error": 0.02,
            "effective_sample_size_fraction": 0.09,
            "true_effective_sample_size_fraction": 0.08,
        },
    ]

    winners = _balanced_winners(rows)

    assert winners[0]["winning_estimator"] == "boosted_tree_relaxed_tail"


def test_ope_drives_winner_when_ratio_truth_unavailable() -> None:
    rows = [
        {
            "profile": "high_stakes",
            "setting": "gym_mountain_car_continuous",
            "gamma": 0.99,
            "sample_size": 1000,
            "seed": 0,
            "estimator": "neural_network_stable",
            "status": "ok",
            "ratio_truth_available": 0.0,
            "ope_value_abs_error": 0.40,
            "ope_value_abs_error_se_units": 3.0,
            "effective_sample_size_fraction": 0.90,
        },
        {
            "profile": "high_stakes",
            "setting": "gym_mountain_car_continuous",
            "gamma": 0.99,
            "sample_size": 1000,
            "seed": 0,
            "estimator": "neural_network_relaxed_tail",
            "status": "ok",
            "ratio_truth_available": 0.0,
            "ope_value_abs_error": 0.20,
            "ope_value_abs_error_se_units": 1.5,
            "effective_sample_size_fraction": 0.05,
        },
    ]

    winners = _balanced_winners(rows)

    assert winners[0]["winning_estimator"] == "neural_network_relaxed_tail"


def test_relaxed_tail_presets_expand_for_both_families() -> None:
    config = OccupancyRatioBenchmarkConfig(
        estimators=("boosted_tree", "neural_network"),
        boosted_estimator_presets=("relaxed_tail",),
        neural_estimator_presets=("relaxed_tail",),
    )

    assert _expanded_estimators(config) == ("boosted_tree_relaxed_tail", "neural_network_relaxed_tail")
