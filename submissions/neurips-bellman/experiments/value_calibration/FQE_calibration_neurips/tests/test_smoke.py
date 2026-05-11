from __future__ import annotations

import math
from pathlib import Path

from FQE_calibration_neurips.scripts.run_experiment import run_config


def test_debug_experiment_end_to_end(tmp_path: Path) -> None:
    config = {
        "seed": 11,
        "replications": 1,
        "gamma": 0.9,
        "horizon": 12,
        "n_actions": 3,
        "sample_sizes": [90],
        "state_dimensions": [4],
        "coverage_settings": ["good"],
        "reward_noise_settings": [0.1],
        "test_transitions": 80,
        "diagnostic_test_transitions": 120,
        "true_q_rollouts_per_state": 1,
        "calibration_error_bins": 6,
        "calibration_error_min_bin_size": 5,
        "calibration_error_folds": 3,
        "initial_eval_states": 80,
        "oracle_rollouts": 60,
        "policy_shift": {"good": 0.1, "moderate": 0.7, "severe": 1.2, "extrapolation": 1.5},
        "baseline_learners": ["random_feature_fqe", "neural_fqe"],
        "calibration_protocols": ["cross", "split", "no_split"],
        "calibrators": ["linear", "histogram", "isotonic", "isotonic_histogram"],
        "calibration_targets": ["value_bellman"],
        "cross_folds": 3,
        "split_fractions": [0.8],
        "split_comparators": ["offset_correction", "residual_correction"],
        "calibrator_params": {"n_bins": 4, "bin_strategy": "quantile", "min_bin_size": 5},
        "comparator_params": {"n_components": 16, "ridge": 0.01},
        "learner_params": {
            "random_feature_fqe": {"n_components": 24, "n_iters": 3, "ridge": 0.01},
            "neural_fqe": {
                "hidden_dims": [12],
                "n_iters": 2,
                "epochs_per_iter": 1,
                "batch_size": 32,
                "lr": 0.002,
                "device": "cpu",
            },
        },
    }
    rows = run_config(config, tmp_path, debug=False)
    assert rows
    required = {
        "environment_name",
        "replication_seed",
        "sample_size",
        "state_dimension",
        "coverage_setting",
        "baseline_learner",
        "calibrated",
        "calibration_protocol",
        "calibrator",
        "calibration_target",
        "base_learner_used_all_data",
        "sample_splitting_used",
        "train_fraction",
        "calibration_fraction",
        "value_estimate",
        "oracle_value",
        "value_error",
        "squared_error",
        "bellman_residual",
        "calibration_error",
        "true_v_mse",
        "true_value_function_mse",
        "true_q_mse",
        "true_function_mse",
        "bellman_outcome_mse",
        "brier_score",
        "bellman_brier_score",
        "bellman_calibration_error",
        "bellman_calibration_error_plugin",
        "bellman_calibration_error_debiased_raw",
        "bellman_calibration_bins",
        "bellman_calibration_test_size",
        "runtime",
        "failure_flag",
        "run_mode",
        "suite_name",
        "environment_tier",
        "policy_shift_setting",
        "misspecification_setting",
        "learner_variant",
        "learner_quality_regime",
        "calibration_difficulty",
        "main_figure_role",
        "oracle_value_method",
        "train_data_provenance",
        "calibration_data_provenance",
        "test_data_provenance",
        "split_fraction",
        "main_evidence_eligible",
        "calibration_object",
        "calibration_weight_scheme",
        "importance_weight_ess",
        "importance_weight_clip",
        "importance_weight_max",
        "interval_coverage_95",
        "interval_length_95",
    }
    protocols = {row["calibration_protocol"] for row in rows}
    calibrators = {row["calibrator"] for row in rows}
    learners = {row["baseline_learner"] for row in rows}
    assert {"uncalibrated_all_data", "cross", "split", "no_split"}.issubset(protocols)
    assert {"linear", "histogram", "isotonic", "isotonic_histogram"}.issubset(calibrators)
    assert {"random_feature_fqe", "neural_fqe"}.issubset(learners)
    for row in rows:
        assert required.issubset(row)
        if not row["failure_flag"]:
            for key in [
                "value_estimate",
                "oracle_value",
                "squared_error",
                "bellman_residual",
                "calibration_error",
                "true_v_mse",
                "true_q_mse",
                "bellman_outcome_mse",
                "brier_score",
                "bellman_calibration_error",
                "bellman_calibration_error_plugin",
                "interval_length_95",
            ]:
                assert math.isfinite(float(row[key]))
            assert int(row["bellman_calibration_bins"]) >= 1
            assert int(row["bellman_calibration_test_size"]) == 120


def test_linear_calibration_improves_affine_distortion() -> None:
    import numpy as np

    from FQE_calibration_neurips.src.calibration.calibrators import fit_calibrator

    x = np.linspace(-2.0, 2.0, 200)
    y = 1.5 + 0.7 * x
    cal = fit_calibrator("linear", x, y)
    before = float(np.mean((x - y) ** 2))
    after = float(np.mean((cal.predict(x) - y) ** 2))
    assert after < 0.01 * before


def test_isotonic_calibration_improves_monotone_distortion() -> None:
    import numpy as np

    from FQE_calibration_neurips.src.calibration.calibrators import fit_calibrator

    truth = np.linspace(-3.0, 3.0, 200)
    distorted = np.sign(truth) * np.sqrt(np.abs(truth) + 1.0)
    cal = fit_calibrator("isotonic", distorted, truth)
    before = float(np.mean((distorted - truth) ** 2))
    after = float(np.mean((cal.predict(distorted) - truth) ** 2))
    assert after < 0.1 * before
