from __future__ import annotations

import json
import importlib.util

import numpy as np
import pytest

from genpqr import DeepGenPQRConfig, DiscreteNormalizationPolicy, available_q_estimators, fit_deep_genpqr
from genpqr.benchmarks import (
    evaluate_reward_matrix,
    make_soft_grid_reward_recovery,
    run_reward_recovery_benchmark,
    run_tiny_production_validation,
)


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


def test_tiny_production_validation_schema_is_json_safe() -> None:
    report = run_tiny_production_validation(n_steps=16, seed=31)
    json.dumps(report)
    assert set(report) == {"tabular_chain", "gridworld", "linear_gaussian"}
    assert report["tabular_chain"]["fit_genpqr"]["reward_finite_fraction"] == 1.0
    assert report["tabular_chain"]["deepgenpqr_pooled_fqe"]["q_backend"] == "benchmark_pooled_q"
    assert "crossfit" in report["tabular_chain"]
    assert report["linear_gaussian"]["deepgenpqr_pooled_fqe"]["reward_finite_fraction"] == 1.0


def test_soft_grid_reward_truth_and_oracle_policy() -> None:
    problem = make_soft_grid_reward_recovery(n_x=5, n_y=4, sample_size=120, seed=41, gamma=0.0)
    assert np.allclose(problem.true_rewards[:, 0], 0.0)
    probs = problem.oracle_policy.predict_proba(problem.grid_states)
    assert probs.shape == problem.true_rewards.shape
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert np.all(probs > 0.0)
    action_counts = np.bincount(problem.dataset.actions, minlength=problem.n_actions)
    assert action_counts[problem.anchor_action] > 0


def test_soft_grid_dataset_generation_is_seed_deterministic() -> None:
    first = make_soft_grid_reward_recovery(n_x=5, n_y=5, sample_size=140, seed=91, gamma=0.0)
    second = make_soft_grid_reward_recovery(n_x=5, n_y=5, sample_size=140, seed=91, gamma=0.0)
    assert np.allclose(first.true_rewards, second.true_rewards)
    assert np.allclose(first.oracle_policy.probabilities, second.oracle_policy.probabilities)
    assert np.allclose(first.dataset.states, second.dataset.states)
    assert np.array_equal(first.dataset.actions, second.dataset.actions)
    assert np.allclose(first.dataset.next_states, second.dataset.next_states)


def test_soft_grid_reward_recovery_smoke_includes_deepgenpqr_metrics() -> None:
    report = run_reward_recovery_benchmark(
        n_x=5,
        n_y=5,
        sample_size=180,
        seed=77,
        gamma=0.0,
        include_pooled_fqe=False,
        include_native_bc=False,
        include_neural_deeppqr=False,
    )
    json.dumps(report)
    rows = {row["name"]: row for row in report["rows"]}
    anchor = rows["oracle_policy_anchor_deeppqr"]
    assert anchor["status"] == "ok"
    assert "benchmark_reward_rmse" in anchor["summary"]
    assert "benchmark_anchor_rmse" in anchor["summary"]
    assert anchor["summary"]["anchor_support"]["anchor_count"] > 0
    assert anchor["metrics"]["reward_rmse"] < rows["constant_mean_reward"]["metrics"]["reward_rmse"]
    assert anchor["metrics"]["reward_rmse"] < rows["random_reward"]["metrics"]["reward_rmse"]
    assert anchor["metrics"]["anchor_rmse"] < rows["constant_mean_reward"]["metrics"]["anchor_rmse"]
    assert anchor["metrics"]["anchor_rmse"] < rows["random_reward"]["metrics"]["anchor_rmse"]


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="action-head neural FQE requires torch")
def test_soft_grid_pooled_action_head_fqe_smoke() -> None:
    assert "fqe_action_head_neural" in available_q_estimators()
    report = run_reward_recovery_benchmark(
        n_x=4,
        n_y=4,
        sample_size=80,
        seed=5,
        gamma=0.0,
        include_pooled_fqe=True,
        include_native_bc=False,
        include_neural_deeppqr=False,
        neural_fast=True,
    )
    rows = {row["name"]: row for row in report["rows"]}
    pooled = rows["oracle_policy_pooled_action_head_fqe"]
    assert pooled["status"] == "ok"
    assert pooled["summary"]["q_backend"] == "action_head_neural_fqe"
    assert pooled["metrics"]["reward_rmse"] < rows["constant_mean_reward"]["metrics"]["reward_rmse"]


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="auto neural FQE discrete route requires torch")
def test_deepgenpqr_default_pooled_fqe_uses_action_head_for_discrete() -> None:
    problem = make_soft_grid_reward_recovery(n_x=4, n_y=4, sample_size=80, seed=11, gamma=0.0)
    result = fit_deep_genpqr(
        dataset=problem.dataset,
        gamma=problem.gamma,
        normalization_policy=DiscreteNormalizationPolicy.anchor(problem.n_actions, problem.anchor_action),
        anchor_function=0.0,
        config=DeepGenPQRConfig(
            policy=problem.oracle_policy,
            q_mode="pooled_fqe",
            seed=11,
            n_action_samples=problem.n_actions,
            q_config={
                "config_overrides": {
                    "hidden_dims": (64, 64),
                    "head_hidden_dims": (32,),
                    "batch_size": 64,
                    "num_iterations": 20,
                    "gradient_steps_per_iteration": 8,
                    "patience": 5,
                    "seed": 11,
                },
                "n_next_action_samples": problem.n_actions,
            },
        ),
    )
    assert DeepGenPQRConfig().q_backend == "auto_neural_fqe"
    assert result.summary()["q_backend"] == "action_head_neural_fqe"


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="auto neural FQE discrete route requires torch")
def test_soft_grid_rare_anchor_default_pooled_auto_beats_constant() -> None:
    problem = make_soft_grid_reward_recovery(
        n_x=4,
        n_y=4,
        sample_size=160,
        seed=17,
        gamma=0.0,
        anchor_behavior_scale=0.08,
        min_anchor_fraction=0.0,
        behavior_floor=0.015,
        behavior_mixture=0.25,
        goal_width=0.55,
        decoy_width=0.55,
        barrier_width=0.55,
        interaction_scale=0.02,
        process_noise=0.04,
    )
    result = fit_deep_genpqr(
        dataset=problem.dataset,
        gamma=problem.gamma,
        normalization_policy=DiscreteNormalizationPolicy.anchor(problem.n_actions, problem.anchor_action),
        anchor_function=0.0,
        config=DeepGenPQRConfig(
            policy=problem.oracle_policy,
            q_mode="pooled_fqe",
            seed=17,
            n_action_samples=problem.n_actions,
            q_config={
                "config_overrides": {
                    "hidden_dims": (64, 64),
                    "head_hidden_dims": (32,),
                    "batch_size": 64,
                    "num_iterations": 30,
                    "gradient_steps_per_iteration": 10,
                    "patience": 6,
                    "seed": 17,
                },
                "n_next_action_samples": problem.n_actions,
            },
        ),
    )
    all_states = np.repeat(problem.grid_states, problem.n_actions, axis=0)
    all_actions = np.tile(np.arange(problem.n_actions), problem.n_states)
    reward_matrix = result.predict_reward(all_states, all_actions).reshape(problem.n_states, problem.n_actions)
    metrics = evaluate_reward_matrix(reward_matrix, problem)
    constant_value = float(np.mean(problem.true_rewards[:, 1:]))
    constant_metrics = evaluate_reward_matrix(np.full_like(problem.true_rewards, constant_value), problem)
    assert result.summary()["q_backend"] == "action_head_neural_fqe"
    assert metrics["anchor_support_fraction"] < 0.05
    assert metrics["reward_rmse"] < constant_metrics["reward_rmse"]
    assert np.all(np.isfinite(metrics["per_action_rmse"]))
