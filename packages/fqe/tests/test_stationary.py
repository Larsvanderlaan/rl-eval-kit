from __future__ import annotations

import builtins

import numpy as np
import pytest

from fqe import fit_stationary_weighted_fqe
from fqe import GoogleDualDICEConfig
from fqe.fit_fqe import BoostedFQEConfig
from fqe import stationary


def _arrays(n: int = 8):
    states = np.zeros((n, 1), dtype=np.float64)
    actions = np.zeros((n, 1), dtype=np.float64)
    rewards = np.linspace(0.0, 1.0, n)
    return states, actions, rewards


def test_gamma_ratio_must_be_below_one() -> None:
    states, actions, rewards = _arrays()
    with pytest.raises(ValueError, match="discount below one"):
        fit_stationary_weighted_fqe(
            states=states,
            actions=actions,
            next_states=states,
            target_actions=actions,
            next_actions=actions,
            rewards=rewards,
            gamma=0.0,
            gamma_ratio=1.0,
        )


def test_optional_occupancy_import_error_is_clear(monkeypatch) -> None:
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "occupancy_ratio":
            raise ModuleNotFoundError("no occupancy_ratio")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ModuleNotFoundError, match="Stationary weighted FQE requires"):
        stationary._load_occupancy_api()


def test_stationary_weight_combination_normalizes_to_mean_one() -> None:
    weights, diagnostics = stationary._combine_weights(
        sample_weight=np.array([1.0, 3.0]),
        ratio_weights=np.array([2.0, 2.0]),
        normalize=True,
    )
    assert np.allclose(weights, np.array([0.5, 1.5]))
    assert np.isclose(np.mean(weights), 1.0)
    assert np.isclose(diagnostics["fqe_weight_unnormalized_mean"], 4.0)
    assert diagnostics["fqe_weight_ess_fraction"] < 1.0


def test_stationary_weighted_fqe_uses_ratio_weights_and_target_next_actions(monkeypatch) -> None:
    states, actions, rewards = _arrays(10)
    captured = {}

    class FakeOccupancyModel:
        def predict_state_action_ratio(self, states_arg, actions_arg, *, clip=True):
            captured["clip"] = clip
            return np.linspace(1.0, 3.0, states_arg.shape[0])

    def fake_fit(**kwargs):
        captured.update(kwargs)
        return FakeOccupancyModel()

    monkeypatch.setattr(stationary, "_load_occupancy_api", lambda family="boosted": fake_fit)
    result = fit_stationary_weighted_fqe(
        states=states,
        actions=actions,
        next_states=states,
        target_actions=actions,
        next_actions=actions,
        rewards=rewards,
        gamma=0.0,
        gamma_ratio=0.95,
        ratio_backend="occupancy_ratio",
        sample_weight=np.ones(10),
        fqe_config=BoostedFQEConfig.stable_defaults(num_iterations=4, validation_fraction=0.3, infer_value_bounds=False),
    )
    assert captured["gamma"] == 0.95
    assert captured["target_next_actions"] is not None
    assert captured["clip"] is True
    assert np.isclose(np.mean(result.sample_weight), 1.0)
    assert np.allclose(result.ratio_weights, np.linspace(1.0, 3.0, 10))
    assert result.diagnostics["gamma_ratio"] == 0.95
    assert np.isfinite(result.estimate_policy_value(states[:2], actions[:2]))


def test_stationary_uniform_ratio_falls_back_to_action_ratio(monkeypatch) -> None:
    states, actions, rewards = _arrays(10)

    class FakeOccupancyModel:
        def predict_state_action_ratio(self, states_arg, actions_arg, *, clip=True):
            return np.ones(states_arg.shape[0])

        def predict_action_ratio(self, states_arg, actions_arg, *, clip=True):
            return np.linspace(1.0, 2.0, states_arg.shape[0])

    monkeypatch.setattr(stationary, "_load_occupancy_api", lambda family="boosted": lambda **kwargs: FakeOccupancyModel())
    result = fit_stationary_weighted_fqe(
        states=states,
        actions=actions,
        next_states=states,
        target_actions=actions,
        next_actions=actions,
        rewards=rewards,
        gamma=0.0,
        gamma_ratio=0.95,
        ratio_backend="occupancy_ratio",
        sample_weight=np.ones(10),
        fqe_config=BoostedFQEConfig.stable_defaults(num_iterations=4, validation_fraction=0.3, infer_value_bounds=False),
    )
    assert result.diagnostics["ratio_weight_source"] == "action_ratio_fallback"
    assert result.diagnostics["ratio_weight_fallback_reason"] == "near_uniform"
    assert np.allclose(result.ratio_weights, np.linspace(1.0, 2.0, 10))
    assert np.isclose(np.mean(result.sample_weight), 1.0)


def test_invalid_stationary_ratio_degrades_to_uniform_weights(monkeypatch) -> None:
    states, actions, rewards = _arrays(10)

    class FakeOccupancyModel:
        def predict_state_action_ratio(self, states_arg, actions_arg, *, clip=True):
            return np.zeros(states_arg.shape[0])

    monkeypatch.setattr(stationary, "_load_occupancy_api", lambda family="boosted": lambda **kwargs: FakeOccupancyModel())
    result = fit_stationary_weighted_fqe(
        states=states,
        actions=actions,
        next_states=states,
        target_actions=actions,
        next_actions=actions,
        rewards=rewards,
        gamma=0.0,
        gamma_ratio=0.95,
        ratio_backend="occupancy_ratio",
        sample_weight=np.ones(10),
        fqe_config=BoostedFQEConfig.stable_defaults(num_iterations=4, validation_fraction=0.3, infer_value_bounds=False),
    )
    assert result.diagnostics["ratio_weight_source"] == "uniform_invalid_ratio_fallback"
    assert result.diagnostics["ratio_weight_invalid_reason"] == "nonpositive_total"
    assert result.diagnostics["ratio_weight_degraded"] is True
    assert np.allclose(result.sample_weight, np.ones(10))


def test_stationary_neural_family_loads_neural_ratio_weights(monkeypatch) -> None:
    states, actions, rewards = _arrays(8)
    captured = {}

    class FakeOccupancyModel:
        def predict_state_action_ratio(self, states_arg, actions_arg, *, clip=True):
            return np.linspace(1.0, 2.0, states_arg.shape[0])

        def predict_action_ratio(self, states_arg, actions_arg, *, clip=True):
            return np.linspace(2.0, 3.0, states_arg.shape[0])

    class FakeNeuralModel:
        diagnostics = {"fit": "fake"}

        def predict(self, states_arg, actions_arg=None):
            return np.zeros(states_arg.shape[0])

        def predict_q(self, states_arg, actions_arg):
            return np.zeros(states_arg.shape[0])

        def estimate_policy_value(self, initial_states, initial_actions=None, initial_weights=None):
            return 0.0

    def fake_load(family="boosted"):
        captured["ratio_family"] = family
        return lambda **kwargs: FakeOccupancyModel()

    def fake_fit_neural(**kwargs):
        captured["fqe_sample_weight"] = kwargs["sample_weight"]
        return FakeNeuralModel()

    monkeypatch.setattr(stationary, "_load_occupancy_api", fake_load)
    monkeypatch.setattr(stationary, "fit_fqe_neural", fake_fit_neural)
    result = fit_stationary_weighted_fqe(
        states=states,
        actions=actions,
        next_states=states,
        target_actions=actions,
        next_actions=actions,
        rewards=rewards,
        gamma=0.0,
        gamma_ratio=0.95,
        family="neural",
        ratio_backend="occupancy_ratio",
        sample_weight=np.ones(8),
    )
    assert captured["ratio_family"] == "neural"
    assert result.diagnostics["family"] == "neural"
    assert result.diagnostics["ratio_family"] == "neural"
    assert np.allclose(result.ratio_weights, np.linspace(1.0, 2.0, 8))
    assert np.isclose(np.mean(captured["fqe_sample_weight"]), 1.0)


def test_stationary_occupancy_source_failure_retries_without_initial_states(monkeypatch) -> None:
    states, actions, rewards = _arrays(10)
    calls = []

    class FakeOccupancyModel:
        def predict_state_action_ratio(self, states_arg, actions_arg, *, clip=True):
            return np.linspace(1.0, 2.0, states_arg.shape[0])

        def predict_action_ratio(self, states_arg, actions_arg, *, clip=True):
            return np.linspace(1.0, 2.0, states_arg.shape[0])

    def fake_fit(**kwargs):
        calls.append(kwargs)
        if kwargs["initial_states"] is not None:
            raise RuntimeError("source ratio failed")
        return FakeOccupancyModel()

    monkeypatch.setattr(stationary, "_load_occupancy_api", lambda family="boosted": fake_fit)
    result = fit_stationary_weighted_fqe(
        states=states,
        actions=actions,
        next_states=states,
        target_actions=actions,
        next_actions=actions,
        rewards=rewards,
        gamma=0.0,
        gamma_ratio=0.95,
        ratio_backend="occupancy_ratio",
        initial_states=states[:2],
        initial_actions=actions[:2],
        sample_weight=np.ones(10),
        fqe_config=BoostedFQEConfig.stable_defaults(num_iterations=4, validation_fraction=0.3, infer_value_bounds=False),
    )
    assert len(calls) == 2
    assert calls[0]["initial_states"] is not None
    assert calls[1]["initial_states"] is None
    assert result.diagnostics["occupancy_fit_fallback_used"] is True
    assert "source ratio failed" in result.diagnostics["occupancy_fit_fallback_reason"]


def test_stationary_google_dualdice_backend_uses_google_weights(monkeypatch) -> None:
    states, actions, rewards = _arrays(10)
    captured = {}

    class FakeGoogleModel:
        diagnostics = {"google_dualdice_num_updates": 7.0}

        def predict_state_action_ratio(self, states_arg, actions_arg, *, clip=True):
            captured["clip"] = clip
            return np.linspace(1.0, 2.0, states_arg.shape[0])

        def predict_action_ratio(self, states_arg, actions_arg, *, clip=True):
            return np.ones(states_arg.shape[0])

    def fake_fit_google(**kwargs):
        captured.update(kwargs)
        return FakeGoogleModel(), {
            "occupancy_fit_fallback_used": False,
            "occupancy_fit_fallback_reason": None,
        }

    monkeypatch.setattr(stationary, "_fit_google_dualdice_ratio_model", fake_fit_google)
    result = fit_stationary_weighted_fqe(
        states=states,
        actions=actions,
        next_states=states,
        target_actions=actions,
        next_actions=actions,
        rewards=rewards,
        gamma=0.0,
        gamma_ratio=0.99,
        initial_states=states[:2],
        initial_actions=actions[:2],
        config=stationary.StationaryWeightedFQEConfig(
            google_dualdice_config=GoogleDualDICEConfig(num_updates=7, batch_size=4),
            fqe_config=BoostedFQEConfig.stable_defaults(num_iterations=4, validation_fraction=0.3, infer_value_bounds=False),
        ),
    )
    assert captured["gamma_ratio"] == 0.99
    assert captured["target_next_actions"] is not None
    assert result.diagnostics["ratio_backend"] == "google_dualdice"
    assert result.diagnostics["occupancy_google_dualdice_num_updates"] == 7.0
    assert np.allclose(result.ratio_weights, np.linspace(1.0, 2.0, 10))


def test_stationary_minimax_weight_backend_uses_common_facade(monkeypatch) -> None:
    states, actions, rewards = _arrays(10)
    captured = {}

    class FakeMinimaxModel:
        diagnostics = {"minimax_method": "scope_rl_minimax_state_action"}
        method = "scope_rl_minimax_state_action"

        def predict_state_action_ratio(self, states_arg, actions_arg, *, clip=True):
            captured["clip"] = clip
            return np.linspace(1.0, 3.0, states_arg.shape[0])

        def predict_action_ratio(self, states_arg, actions_arg, *, clip=True):
            return np.ones(states_arg.shape[0])

    def fake_fit_minimax(**kwargs):
        captured.update(kwargs)
        return FakeMinimaxModel(), {
            "occupancy_fit_fallback_used": False,
            "occupancy_fit_fallback_reason": None,
            "minimax_weight_method_resolved": "scope_rl_minimax_state_action",
        }

    monkeypatch.setattr(stationary, "_fit_minimax_weight_ratio_model", fake_fit_minimax)
    result = fit_stationary_weighted_fqe(
        states=states,
        actions=actions,
        next_states=states,
        target_actions=actions,
        next_actions=actions,
        rewards=rewards,
        gamma=0.0,
        gamma_ratio=0.95,
        initial_states=states[:2],
        initial_actions=actions[:2],
        config=stationary.StationaryWeightedFQEConfig(
            ratio_backend="minimax_weight",
            minimax_weight_method="scope_rl_minimax_state_action",
            fqe_config=BoostedFQEConfig.stable_defaults(num_iterations=4, validation_fraction=0.3, infer_value_bounds=False),
        ),
        episode_ids=np.repeat(np.arange(2), 5),
        timesteps=np.tile(np.arange(5), 2),
        step_per_trajectory=5,
    )
    assert captured["method"] == "scope_rl_minimax_state_action"
    assert captured["gamma_ratio"] == 0.95
    assert captured["step_per_trajectory"] == 5
    assert result.diagnostics["ratio_backend"] == "minimax_weight"
    assert result.diagnostics["minimax_weight_method"] == "scope_rl_minimax_state_action"
    assert result.diagnostics["occupancy_minimax_method"] == "scope_rl_minimax_state_action"
    assert np.allclose(result.ratio_weights, np.linspace(1.0, 3.0, 10))


def test_default_dualdice_backend_requires_initial_state_actions() -> None:
    states, actions, rewards = _arrays(10)
    with pytest.raises(ValueError, match="Google DualDICE is the default"):
        fit_stationary_weighted_fqe(
            states=states,
            actions=actions,
            next_states=states,
            target_actions=actions,
            next_actions=actions,
            rewards=rewards,
            gamma=0.0,
            fqe_config=BoostedFQEConfig.stable_defaults(num_iterations=4, validation_fraction=0.3, infer_value_bounds=False),
        )


def test_google_dualdice_preflight_reports_missing_source(tmp_path) -> None:
    available, reason = stationary.preflight_google_dualdice(tmp_path / "missing-google-research")
    assert available is False
    assert "DualDICE source" in reason
