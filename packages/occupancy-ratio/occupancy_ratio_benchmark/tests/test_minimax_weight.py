from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from occupancy_ratio import (
    DEFAULT_MINIMAX_WEIGHT_METHOD,
    GOOGLE_DICE_RL_DUALDICE_EXACT_FLAGS,
    GOOGLE_DICE_RL_RECOMMENDED_FLAGS,
    GoogleDICERLConfig,
    MinimaxWeightConfig,
    ScopeRLMinimaxWeightConfig,
    fit_minimax_weight,
    preflight_google_dice_rl,
    preflight_scope_rl,
)
import occupancy_ratio.minimax_weight as minimax_weight


def _arrays(n: int = 6):
    states = np.arange(n * 2, dtype=np.float64).reshape(n, 2)
    actions = np.linspace(-1.0, 1.0, n).reshape(n, 1)
    return states, actions


def test_minimax_weight_config_validation() -> None:
    assert DEFAULT_MINIMAX_WEIGHT_METHOD == "google_policy_eval_dualdice"
    with pytest.raises(ValueError, match="num_steps"):
        GoogleDICERLConfig(num_steps=0)
    with pytest.raises(ValueError, match="hidden_dims"):
        GoogleDICERLConfig(hidden_dims=())
    with pytest.raises(ValueError, match="n_steps"):
        ScopeRLMinimaxWeightConfig(n_steps=0)
    with pytest.raises(ValueError, match="method"):
        MinimaxWeightConfig(method="not-a-method")


def test_minimax_google_policy_eval_dispatches_to_existing_wrapper(monkeypatch) -> None:
    states, actions = _arrays()
    captured = {}

    class FakeGoogleModel:
        diagnostics = {"num_updates": 3.0}

        def predict_state_action_ratio(self, states_arg, actions_arg, *, clip=True):
            return np.linspace(-1.0, 2.0, states_arg.shape[0]) if not clip else np.linspace(0.0, 2.0, states_arg.shape[0])

        def predict_action_ratio(self, states_arg, actions_arg, *, clip=True):
            return np.ones(states_arg.shape[0])

    def fake_fit_google(**kwargs):
        captured.update(kwargs)
        return FakeGoogleModel()

    monkeypatch.setattr(minimax_weight, "fit_google_dualdice_occupancy_ratio", fake_fit_google)
    model = fit_minimax_weight(
        states=states,
        actions=actions,
        next_states=states + 0.1,
        target_actions=actions * 0.5,
        target_next_actions=actions * 0.25,
        gamma=0.95,
        initial_states=states[:2],
        initial_actions=actions[:2],
        method="google_policy_eval_dualdice",
    )
    assert captured["gamma"] == 0.95
    assert captured["initial_states"].shape[0] == 2
    assert model.method == "google_policy_eval_dualdice"
    assert model.diagnostics["minimax_backend"] == "google_policy_eval_dualdice"
    assert np.allclose(model.predict_action_ratio(states, actions), np.ones(states.shape[0]))


def test_minimax_google_dice_rl_methods_use_official_flag_sets(monkeypatch) -> None:
    states, actions = _arrays()
    calls = []

    def fake_fit_google_dice_rl(common, cfg, *, method, flags):
        calls.append((method, dict(flags)))
        return minimax_weight.MinimaxWeightModel(
            backend_model=object(),
            method=method,
            gamma=common["gamma"],
            state_dim=common["S"].shape[1],
            action_dim=common["A"].shape[1],
            diagnostics={"minimax_method": method},
            config=cfg,
            _state_action_predictor=lambda s, a: np.ones(s.shape[0]),
        )

    monkeypatch.setattr(minimax_weight, "_fit_google_dice_rl", fake_fit_google_dice_rl)
    for method in ("google_dice_rl_dualdice_exact", "google_dice_rl_recommended"):
        fit_minimax_weight(
            states=states,
            actions=actions,
            next_states=states + 0.1,
            target_actions=actions * 0.5,
            target_next_actions=actions * 0.25,
            gamma=0.95,
            initial_states=states[:2],
            initial_actions=actions[:2],
            method=method,
        )
    assert calls[0] == ("google_dice_rl_dualdice_exact", GOOGLE_DICE_RL_DUALDICE_EXACT_FLAGS)
    assert calls[1] == ("google_dice_rl_recommended", GOOGLE_DICE_RL_RECOMMENDED_FLAGS)


def test_minimax_scope_rl_state_action_with_fake_package(monkeypatch) -> None:
    states, actions = _arrays()
    captured = {}

    class FakeStateActionWeightFunction:
        def __init__(self, **kwargs):
            captured["function_kwargs"] = kwargs

    class FakeStateWeightFunction:
        def __init__(self, **kwargs):
            captured["state_function_kwargs"] = kwargs

    class FakeLearner:
        def __init__(self, **kwargs):
            captured["learner_kwargs"] = kwargs

        def fit(self, **kwargs):
            captured["fit_kwargs"] = kwargs

        def predict_weight(self, state, action):
            return np.linspace(1.0, 2.0, np.asarray(state).shape[0])

    scope_rl = types.ModuleType("scope_rl")
    ope = types.ModuleType("scope_rl.ope")
    wvl = types.ModuleType("scope_rl.ope.weight_value_learning")
    function = types.ModuleType("scope_rl.ope.weight_value_learning.function")
    wvl.ContinuousMinimaxStateActionWeightLearning = FakeLearner
    wvl.ContinuousMinimaxStateWeightLearning = FakeLearner
    function.ContinuousStateActionWeightFunction = FakeStateActionWeightFunction
    function.StateWeightFunction = FakeStateWeightFunction
    monkeypatch.setitem(sys.modules, "scope_rl", scope_rl)
    monkeypatch.setitem(sys.modules, "scope_rl.ope", ope)
    monkeypatch.setitem(sys.modules, "scope_rl.ope.weight_value_learning", wvl)
    monkeypatch.setitem(sys.modules, "scope_rl.ope.weight_value_learning.function", function)

    preflight = preflight_scope_rl(method="scope_rl_minimax_state_action")
    assert preflight.available is True
    model = fit_minimax_weight(
        states=states,
        actions=actions,
        next_states=states + 0.1,
        target_actions=actions * 0.5,
        target_next_actions=actions * 0.25,
        gamma=0.95,
        method="scope_rl_minimax_state_action",
        config=MinimaxWeightConfig(
            method="scope_rl_minimax_state_action",
            scope_rl=ScopeRLMinimaxWeightConfig(n_steps=3, n_steps_per_epoch=3, batch_size=2, hidden_dim=7),
        ),
        step_per_trajectory=3,
    )
    assert captured["function_kwargs"]["hidden_dim"] == 7
    assert captured["fit_kwargs"]["step_per_trajectory"] == 3
    assert captured["fit_kwargs"]["evaluation_policy_action"].shape == (6, 1)
    assert model.diagnostics["minimax_backend"] == "scope_rl"
    assert np.allclose(model.predict_state_action_ratio(states, actions), np.linspace(1.0, 2.0, 6))


def test_optional_minimax_preflights_report_missing_sources(tmp_path) -> None:
    dice = preflight_google_dice_rl(tmp_path / "missing-dice-rl")
    assert dice.available is False
    assert "DICE-RL source" in dice.reason
    scope = preflight_scope_rl(tmp_path / "missing-scope-rl")
    assert scope.available is False
    assert "SCOPE-RL import failed" in scope.reason or "Missing SCOPE-RL package" in scope.reason
