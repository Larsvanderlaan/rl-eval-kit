from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from fqe import BoostedFQEConfig, fit_fqe_from_policy, fit_fqe_lgbm, fit_value_lgbm, load_fqe_lgbm, tune_fqe_cv


LIGHTGBM_AVAILABLE = importlib.util.find_spec("lightgbm") is not None
pytestmark_lgbm = pytest.mark.skipif(not LIGHTGBM_AVAILABLE, reason="LightGBM is not installed")


def _small_config(**overrides):
    params = {
        "num_iterations": 8,
        "trees_per_iteration": 1,
        "validation_fraction": 0.25,
        "patience": 4,
        "refit_on_all_data": True,
        "infer_value_bounds": False,
        "show_progress": False,
        "seed": 17,
        "lgb_params": {
            "learning_rate": 0.2,
            "num_leaves": 15,
            "min_data_in_leaf": 1,
            "lambda_l2": 0.0,
            "verbosity": -1,
            "num_threads": 1,
        },
    }
    params.update(overrides)
    return BoostedFQEConfig.stable_defaults(**params)


def test_import_public_api() -> None:
    from fqe import FQEModel, fit_fqe_lgbm as imported_fit

    assert imported_fit is fit_fqe_lgbm
    assert FQEModel.__name__ == "FQEModel"


def test_config_validation() -> None:
    with pytest.raises(ValueError, match="gamma"):
        fit_value_lgbm(np.zeros((2, 1)), np.zeros((2, 1)), np.ones(2), 1.0)
    with pytest.raises(ValueError, match="loss"):
        BoostedFQEConfig(loss="bad")
    with pytest.raises(ValueError, match="validation_fraction"):
        BoostedFQEConfig(validation_fraction=1.0)


def test_q_mode_requires_aligned_actions() -> None:
    states = np.zeros((4, 2))
    actions = np.zeros((3, 1))
    next_states = np.zeros((4, 2))
    next_actions = np.zeros((4, 1))
    rewards = np.ones(4)
    with pytest.raises(ValueError, match="actions"):
        fit_fqe_lgbm(states, actions, next_states, next_actions, rewards, 0.9)


def test_next_actions_shape_validation() -> None:
    states = np.zeros((4, 2))
    actions = np.zeros((4, 1))
    next_states = np.zeros((4, 2))
    rewards = np.ones(4)
    with pytest.raises(ValueError, match="action dimension"):
        fit_fqe_lgbm(states, actions, next_states, np.zeros((4, 2)), rewards, 0.9)


def test_sample_weight_validation() -> None:
    states = np.zeros((4, 1))
    rewards = np.ones(4)
    with pytest.raises(ValueError, match="nonnegative"):
        fit_value_lgbm(states, states, rewards, 0.9, sample_weight=np.array([1.0, -1.0, 1.0, 1.0]))


@pytestmark_lgbm
def test_gamma_zero_value_fits_immediate_rewards() -> None:
    states = np.arange(12, dtype=float).reshape(-1, 1)
    rewards = 2.0 + 0.5 * states.reshape(-1)
    model = fit_value_lgbm(
        states,
        states,
        rewards,
        gamma=0.0,
        config=_small_config(loss="squared", num_iterations=30, early_stopping=False),
    )
    pred = model.predict_value(states)
    assert pred.shape == rewards.shape
    assert np.mean((pred - rewards) ** 2) < 0.5
    assert model.history
    assert model.diagnostics["mode"] == "value"
    assert model.to_legacy_dict()["mode"] == "value"


@pytestmark_lgbm
def test_constant_reward_value_approaches_discounted_value() -> None:
    states = np.zeros((40, 1))
    rewards = np.ones(40)
    model = fit_value_lgbm(
        states,
        states,
        rewards,
        gamma=0.5,
        terminals=np.zeros(40),
        config=_small_config(loss="squared", num_iterations=20, target_min=0.0, target_max=3.0),
    )
    value = model.predict_value(np.zeros((3, 1)))
    assert np.all(np.isfinite(value))
    assert np.allclose(value, 2.0, atol=0.6)


@pytestmark_lgbm
def test_gym_timeouts_continue_bootstrap() -> None:
    states = np.zeros((40, 1))
    rewards = np.ones(40)
    model = fit_value_lgbm(
        states,
        states,
        rewards,
        gamma=0.5,
        timeouts=np.ones(40),
        config=_small_config(loss="squared", num_iterations=20, target_min=0.0, target_max=3.0),
    )
    assert np.allclose(model.predict_value(np.zeros((3, 1))), 2.0, atol=0.6)
    assert model.diagnostics["timeout_fraction"] == pytest.approx(1.0)


@pytestmark_lgbm
def test_value_mode_matches_q_mode_with_constant_actions() -> None:
    states = np.linspace(-1.0, 1.0, 30).reshape(-1, 1)
    actions = np.zeros((30, 1))
    rewards = 1.0 + states.reshape(-1)
    config = _small_config(loss="squared", num_iterations=10, seed=22)
    value_model = fit_value_lgbm(states, states, rewards, gamma=0.0, config=config)
    q_model = fit_fqe_lgbm(states, actions, states, actions, rewards, gamma=0.0, config=config)
    assert np.allclose(value_model.predict_value(states), q_model.predict_q(states, actions), atol=0.4)


@pytestmark_lgbm
def test_multi_sample_next_actions_and_policy_sampler() -> None:
    rng = np.random.default_rng(4)
    states = rng.normal(size=(50, 2))
    actions = rng.normal(size=(50, 1))
    next_states = rng.normal(size=(50, 2))
    rewards = states[:, 0] - 0.25 * actions[:, 0]
    next_actions = np.stack([np.zeros((50, 1)), np.ones((50, 1))], axis=1)
    model = fit_fqe_lgbm(
        states,
        actions,
        next_states,
        next_actions,
        rewards,
        gamma=0.2,
        config=_small_config(num_iterations=6),
    )
    pred = model.predict(states, actions)
    assert pred.shape == (50,)
    assert np.all(np.isfinite(pred))

    def sampler(next_states_arg, rng_arg, n_samples):
        assert n_samples == 2
        assert next_states_arg.shape == next_states.shape
        return np.stack(
            [np.zeros((next_states_arg.shape[0], 1)), np.ones((next_states_arg.shape[0], 1))],
            axis=1,
        )

    sampled_model = fit_fqe_from_policy(
        states,
        actions,
        next_states,
        rewards,
        0.2,
        sampler,
        n_next_action_samples=2,
        config=_small_config(num_iterations=4),
    )
    assert np.isfinite(sampled_model.estimate_policy_value(states[:5], actions[:5]))


@pytestmark_lgbm
def test_discrete_action_spec_and_weighted_action_expectations() -> None:
    states = np.repeat(np.linspace(0.0, 1.0, 24).reshape(-1, 1), 2, axis=0)
    actions = np.tile(np.array([0, 1]), 24)
    rewards = states.reshape(-1) + actions.astype(float)
    next_actions = np.tile(np.array([[0, 1]]), (states.shape[0], 1))
    model = fit_fqe_lgbm(
        states,
        actions,
        states,
        next_actions,
        rewards,
        gamma=0.1,
        next_action_weights=np.column_stack([np.full(states.shape[0], 0.25), np.full(states.shape[0], 0.75)]),
        action_spec={"type": "discrete_index", "n_actions": 2},
        config=_small_config(loss="squared", num_iterations=10, seed=29),
    )
    pred = model.predict_q(states[:4], np.array([0, 1, 0, 1]))
    assert pred.shape == (4,)
    assert model.diagnostics["target_action_expectation"] == "weighted"
    value = model.estimate_policy_value(
        states[:3],
        np.tile(np.array([[0, 1]]), (3, 1)),
        initial_action_weights=np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]]),
    )
    assert np.isfinite(value)


@pytestmark_lgbm
def test_terminal_mask_blocks_bootstrap() -> None:
    states = np.zeros((30, 1))
    next_states = np.zeros((30, 1))
    rewards = np.ones(30)
    terminal_model = fit_value_lgbm(
        states,
        next_states,
        rewards,
        gamma=0.9,
        terminals=np.ones(30),
        config=_small_config(loss="squared", num_iterations=10, infer_value_bounds=False),
    )
    assert np.allclose(terminal_model.predict_value(np.zeros((4, 1))), 1.0, atol=0.4)


@pytestmark_lgbm
def test_boosted_serialization_round_trip(tmp_path) -> None:
    states = np.linspace(-1.0, 1.0, 36).reshape(-1, 1)
    actions = np.zeros((36, 1))
    rewards = 1.0 + states.reshape(-1)
    model = fit_fqe_lgbm(
        states,
        actions,
        states,
        actions,
        rewards,
        gamma=0.0,
        config=_small_config(loss="squared", num_iterations=8, seed=31),
    )
    path = tmp_path / "boosted_fqe.fqe"
    model.save(path)
    loaded = load_fqe_lgbm(path)
    assert loaded.diagnostics["mode"] == "q"
    assert np.allclose(loaded.predict_q(states, actions), model.predict_q(states, actions))


@pytestmark_lgbm
def test_tune_fqe_cv_smoke() -> None:
    states = np.linspace(0.0, 1.0, 24).reshape(-1, 1)
    rewards = states.reshape(-1)
    result = tune_fqe_cv(
        param_grid=({"lgb_params": {"num_leaves": 3}}, {"lgb_params": {"num_leaves": 7}}),
        states=states,
        next_states=states,
        rewards=rewards,
        gamma=0.0,
        base_config=_small_config(num_iterations=4, refit_on_all_data=False),
        fit_final=True,
    )
    assert "best_params" in result
    assert np.isfinite(result["best_score"])
    assert result["model"] is not None
