from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from fqe_benchmark.adapters import _stationary_neural_ratio_configs, estimator_registry
from fqe_benchmark.data import make_dataset
from fqe_benchmark.runner import _resolved_estimators, run_benchmark
from fqe_benchmark.types import BenchmarkConfig


LIGHTGBM_AVAILABLE = importlib.util.find_spec("lightgbm") is not None
TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
GYMNASIUM_AVAILABLE = importlib.util.find_spec("gymnasium") is not None


def test_tabular_gamma_zero_truth_matches_rewards() -> None:
    dataset = make_dataset(
        name="tabular_chain",
        sample_size=50,
        gamma=0.0,
        seed=0,
        n_eval=64,
        n_initial_eval=32,
    )
    q_train = dataset.true_q_fn(dataset.states, dataset.actions)
    assert np.allclose(q_train, dataset.rewards)
    assert np.isfinite(dataset.true_policy_value)


def test_linear_gaussian_dataset_shapes_and_truth() -> None:
    dataset = make_dataset(
        name="linear_gaussian",
        sample_size=40,
        gamma=0.5,
        seed=1,
        policy_shift=0.4,
        n_eval=32,
        n_initial_eval=16,
    )
    assert dataset.states.shape == (40, 2)
    assert dataset.actions.shape == (40, 1)
    assert dataset.target_actions.shape == (40, 1)
    assert dataset.next_actions.shape == (40, 1)
    q_eval = dataset.true_q_fn(dataset.target_eval_states, dataset.target_eval_actions)
    assert q_eval.shape == (32,)
    assert np.all(np.isfinite(q_eval))


@pytest.mark.skipif(not GYMNASIUM_AVAILABLE, reason="gymnasium is not installed")
def test_gym_mountain_car_dataset_shapes_and_mc_truth() -> None:
    dataset = make_dataset(
        name="gym_mountain_car_continuous",
        sample_size=24,
        gamma=0.95,
        seed=2,
        n_eval=16,
        n_initial_eval=12,
    )
    assert dataset.domain == "gym_control"
    assert dataset.states.shape == (24, 2)
    assert dataset.actions.shape == (24, 1)
    assert dataset.target_actions.shape == (24, 1)
    assert dataset.next_actions.shape == (24, 1)
    assert dataset.target_eval_states.shape == (16, 2)
    assert np.isfinite(dataset.true_policy_value)


def test_optional_adapter_preflight_reports_missing_or_unsupported() -> None:
    config = BenchmarkConfig.for_stage("smoke")
    registry = estimator_registry()
    assert registry["stationary_weighted_fqe"] is registry["google_dualdice_weighted_fqe"]
    assert registry["stationary_weighted_neural_fqe"] is registry["google_dualdice_weighted_neural_fqe"]
    assert registry["stationary_weighted_fori_fqe"] is registry["ours_stationary_weighted_fqe"]
    assert registry["stationary_weighted_fori_neural_fqe"] is registry["ours_stationary_weighted_neural_fqe"]
    assert registry["stationary_weighted_minimax_fqe"] is registry["ours_minimax_weighted_fqe"]
    assert registry["stationary_weighted_google_dice_rl_dualdice_exact_fqe"] is registry[
        "ours_google_dice_rl_dualdice_exact_weighted_fqe"
    ]
    assert registry["stationary_weighted_scope_rl_minimax_state_action_fqe"] is registry[
        "ours_scope_rl_minimax_state_action_weighted_fqe"
    ]
    assert registry["boosted_fqe_staged_cv"] is registry["ours_boosted_fqe_staged_cv"]
    assert registry["neural_fqe_staged_cv"] is registry["ours_neural_fqe_staged_cv"]
    preflight = registry["d3rlpy_fqe"].preflight(config, None)
    assert preflight.status in {"missing_dependency", "unsupported_setting", "ok"}
    google = registry["google_policy_eval_fqe_l2"].preflight(config, None)
    assert google.status in {"missing_dependency", "unsupported_setting"}


def test_fqe_benchmark_public_module() -> None:
    import importlib

    assert importlib.import_module("fqe.boosted").fit_fqe_lgbm
    assert importlib.import_module("fqe.neural").fit_fqe_neural
    assert importlib.import_module("fqe_benchmark.run").main


def test_staged_cv_benchmark_config_and_estimator_resolution() -> None:
    config = BenchmarkConfig(
        stage="smoke",
        estimators=("boosted_fqe", "neural_fqe"),
        staged_cv=True,
        staged_cv_iterations=(1, 2),
    )
    assert _resolved_estimators(config) == ("boosted_fqe_staged_cv", "neural_fqe_staged_cv")


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch is not installed")
def test_stationary_neural_ratio_benchmark_config_is_not_tiny() -> None:
    config = BenchmarkConfig.for_stage("smoke")
    ratio_configs = _stationary_neural_ratio_configs(config, seed=0)
    assert ratio_configs["one_step_ratio_mode"] == "factored"
    assert ratio_configs["occupancy_config"].num_iterations >= 8
    assert ratio_configs["occupancy_config"].gradient_steps_per_iteration >= 4
    assert ratio_configs["action_ratio_config"].max_steps >= 80
    assert ratio_configs["transition_ratio_config"].permutation_samples >= 4
    core_configs = _stationary_neural_ratio_configs(BenchmarkConfig.for_stage("core"), seed=0)
    assert tuple(core_configs["occupancy_config"].hidden_dims) == (64, 64)
    assert core_configs["occupancy_config"].activation == "silu"
    assert core_configs["occupancy_config"].num_iterations == 60
    assert core_configs["occupancy_config"].gradient_steps_per_iteration == 6
    assert core_configs["occupancy_config"].mcmc_samples == 24
    assert core_configs["occupancy_config"].direct_adjoint_steps == 128
    assert core_configs["action_ratio_config"].max_steps == 800
    assert core_configs["source_state_ratio_config"].max_steps == 800
    assert core_configs["transition_ratio_config"].max_steps == 1000
    assert core_configs["occupancy_config"].fixed_point_damping == 0.5
    assert core_configs["action_ratio_config"].density_ratio_loss == "lsif"
    assert core_configs["action_ratio_config"].moment_calibration == "scalar"
    google_parity = _stationary_neural_ratio_configs(
        BenchmarkConfig.for_stage("core"),
        seed=0,
        preset="google_parity",
    )
    assert tuple(google_parity["occupancy_config"].hidden_dims) == (256, 256)
    assert google_parity["occupancy_config"].activation == "relu"


def test_benchmark_smoke_writes_outputs(tmp_path) -> None:
    estimators = ["controlled_linear_fqe", "stationary_weighted_fqe", "stationary_weighted_fori_fqe"]
    if LIGHTGBM_AVAILABLE:
        estimators.append("boosted_fqe_auto")
        estimators.append("ours_boosted_fqe")
    if TORCH_AVAILABLE:
        estimators.append("ours_neural_fqe")
    estimators.append("d3rlpy_fqe")
    config = BenchmarkConfig(
        stage="smoke",
        output_root=tmp_path,
        seeds=(0,),
        datasets=("tabular_chain",),
        estimators=tuple(estimators),
        sample_sizes=(50,),
        gammas=(0.0,),
        output_plots=False,
        boosted_num_iterations=6,
        neural_num_iterations=3,
        neural_gradient_steps_per_iteration=4,
        boosted_tune_num_iterations=4,
        automl_tuning="fast",
    )
    result = run_benchmark(config)
    assert result.results_path.exists()
    assert result.summary_path.exists()
    assert result.diagnostics_path.exists()
    assert result.manifest_path.exists()
    assert result.tuning_results_path.exists()
    statuses = {row["estimator"]: row["status"] for row in result.rows}
    assert statuses["controlled_linear_fqe"] == "ok"
    assert statuses["d3rlpy_fqe"] in {"missing_dependency", "unsupported_setting", "ok"}
    assert statuses["stationary_weighted_fqe"] in {"ok", "missing_dependency"}
    assert statuses["stationary_weighted_fori_fqe"] in {"ok", "missing_dependency"}
    if LIGHTGBM_AVAILABLE:
        assert statuses["boosted_fqe_auto"] == "ok"
    ok_rows = [row for row in result.rows if row["status"] == "ok"]
    assert ok_rows
    assert all("policy_value_absolute_error" in row for row in ok_rows)
    if LIGHTGBM_AVAILABLE:
        assert result.tuning_rows
