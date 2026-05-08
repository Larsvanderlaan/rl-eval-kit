from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from fqe import (
    NeuralFQEConfig,
    NeuralFQEModel,
    fit_fqe_neural,
    fit_fqe_neural_from_policy,
    fit_value_neural,
    tune_fqe_neural_cv,
)


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
pytestmark_torch = pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is not installed")


def _small_config(**overrides):
    params = {
        "hidden_dims": (32, 32),
        "activation": "relu",
        "learning_rate": 2e-3,
        "weight_decay": 0.0,
        "batch_size": 32,
        "num_iterations": 8,
        "gradient_steps_per_iteration": 12,
        "target_update_tau": 0.5,
        "validation_fraction": 0.25,
        "patience": 5,
        "infer_value_bounds": False,
        "standardize_inputs": True,
        "device": "cpu",
        "seed": 11,
        "show_progress": False,
    }
    params.update(overrides)
    return NeuralFQEConfig.stable_defaults(**params)


def test_neural_import_public_api() -> None:
    assert NeuralFQEModel.__name__ == "NeuralFQEModel"
    assert fit_fqe_neural.__name__ == "fit_fqe_neural"


def test_neural_config_validation() -> None:
    with pytest.raises(ValueError, match="hidden_dims"):
        NeuralFQEConfig(hidden_dims=())
    with pytest.raises(ValueError, match="activation"):
        NeuralFQEConfig(activation="bad")
    with pytest.raises(ValueError, match="target_update_tau"):
        NeuralFQEConfig(target_update_tau=0.0)
    with pytest.raises(ValueError, match="loss"):
        NeuralFQEConfig(loss="bad")


def test_neural_shape_and_weight_validation() -> None:
    states = np.zeros((4, 2))
    actions = np.zeros((4, 1))
    rewards = np.ones(4)
    with pytest.raises(ValueError, match="gamma"):
        fit_value_neural(states, states, rewards, gamma=1.0)
    with pytest.raises(ValueError, match="actions"):
        fit_fqe_neural(states, np.zeros((3, 1)), states, actions, rewards, gamma=0.5)
    with pytest.raises(ValueError, match="nonnegative"):
        fit_value_neural(states, states, rewards, gamma=0.5, sample_weight=np.array([1.0, -1.0, 1.0, 1.0]))
    with pytest.raises(ValueError, match="action dimension"):
        fit_fqe_neural(states, actions, states, np.zeros((4, 2)), rewards, gamma=0.5)


@pytestmark_torch
def test_gamma_zero_neural_value_fits_immediate_rewards() -> None:
    states = np.linspace(-1.0, 1.0, 40).reshape(-1, 1)
    rewards = 1.0 + 2.0 * states.reshape(-1)
    model = fit_value_neural(
        states,
        states,
        rewards,
        gamma=0.0,
        config=_small_config(loss="squared", num_iterations=14, gradient_steps_per_iteration=20, early_stopping=False),
    )
    pred = model.predict_value(states)
    assert pred.shape == rewards.shape
    assert np.mean((pred - rewards) ** 2) < 0.35
    assert model.history
    assert model.diagnostics["mode"] == "value"
    assert model.to_legacy_dict()["mode"] == "value"


@pytestmark_torch
def test_constant_reward_neural_value_approaches_discounted_value() -> None:
    states = np.zeros((48, 1))
    rewards = np.ones(48)
    model = fit_value_neural(
        states,
        states,
        rewards,
        gamma=0.5,
        terminals=np.zeros(48),
        config=_small_config(
            loss="squared",
            num_iterations=18,
            gradient_steps_per_iteration=18,
            target_min=0.0,
            target_max=3.0,
            early_stopping=False,
        ),
    )
    value = model.predict_value(np.zeros((4, 1)))
    assert np.all(np.isfinite(value))
    assert np.allclose(value, 2.0, atol=0.75)


@pytestmark_torch
def test_terminal_mask_blocks_neural_bootstrap() -> None:
    states = np.zeros((36, 1))
    rewards = np.ones(36)
    model = fit_value_neural(
        states,
        states,
        rewards,
        gamma=0.9,
        terminals=np.ones(36),
        config=_small_config(loss="squared", num_iterations=10, gradient_steps_per_iteration=14, early_stopping=False),
    )
    assert np.allclose(model.predict_value(np.zeros((3, 1))), 1.0, atol=0.55)


@pytestmark_torch
def test_neural_value_mode_matches_q_mode_with_constant_actions() -> None:
    states = np.linspace(-1.0, 1.0, 36).reshape(-1, 1)
    actions = np.zeros((36, 1))
    rewards = 0.5 + states.reshape(-1)
    config = _small_config(loss="squared", num_iterations=10, gradient_steps_per_iteration=14, early_stopping=False, seed=21)
    value_model = fit_value_neural(states, states, rewards, gamma=0.0, config=config)
    q_model = fit_fqe_neural(states, actions, states, actions, rewards, gamma=0.0, config=config)
    assert np.mean((value_model.predict_value(states) - q_model.predict_q(states, actions)) ** 2) < 0.25


@pytestmark_torch
def test_neural_multi_sample_next_actions_and_policy_sampler() -> None:
    rng = np.random.default_rng(4)
    states = rng.normal(size=(42, 2))
    actions = rng.normal(size=(42, 1))
    next_states = rng.normal(size=(42, 2))
    rewards = states[:, 0] - 0.25 * actions[:, 0]
    next_actions = np.stack([np.zeros((42, 1)), np.ones((42, 1))], axis=1)
    model = fit_fqe_neural(
        states,
        actions,
        next_states,
        next_actions,
        rewards,
        gamma=0.2,
        config=_small_config(num_iterations=5, gradient_steps_per_iteration=8),
    )
    pred = model.predict(states, actions)
    assert pred.shape == (42,)
    assert np.all(np.isfinite(pred))

    def sampler(next_states_arg, rng_arg, n_samples):
        assert n_samples == 2
        assert next_states_arg.shape == next_states.shape
        return np.stack(
            [np.zeros((next_states_arg.shape[0], 1)), np.ones((next_states_arg.shape[0], 1))],
            axis=1,
        )

    sampled_model = fit_fqe_neural_from_policy(
        states,
        actions,
        next_states,
        rewards,
        0.2,
        sampler,
        n_next_action_samples=2,
        config=_small_config(num_iterations=4, gradient_steps_per_iteration=6),
    )
    assert np.isfinite(sampled_model.estimate_policy_value(states[:5], actions[:5]))


@pytestmark_torch
def test_tune_fqe_neural_cv_smoke() -> None:
    states = np.linspace(0.0, 1.0, 28).reshape(-1, 1)
    rewards = states.reshape(-1)
    result = tune_fqe_neural_cv(
        param_grid=({"hidden_dims": (16,)}, {"hidden_dims": (24,)}),
        states=states,
        next_states=states,
        rewards=rewards,
        gamma=0.0,
        base_config=_small_config(num_iterations=4, gradient_steps_per_iteration=5, early_stopping=False),
        fit_final=True,
    )
    assert "best_params" in result
    assert np.isfinite(result["best_score"])
    assert result["model"] is not None
