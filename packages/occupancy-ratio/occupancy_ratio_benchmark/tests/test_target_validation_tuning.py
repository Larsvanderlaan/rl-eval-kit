from __future__ import annotations

import inspect

import numpy as np
import pytest

from occupancy_ratio import (
    ActionRatioConfig,
    OccupancyRegressionConfig,
    OccupancySearchSpace,
    OccupancyTargetValidationResult,
    OccupancyTuningConfig,
    SourceStateRatioConfig,
    TransitionRatioConfig,
    tune_occupancy_ratio_with_target_validation,
)


class _FakeRatioModel:
    def __init__(self, weights: np.ndarray) -> None:
        self.weights = np.asarray(weights, dtype=np.float64).reshape(-1)

    def predict_state_action_ratio(self, states, actions, clip=True):
        n = np.asarray(states).shape[0]
        if n == self.weights.shape[0]:
            return self.weights.copy()
        return np.resize(self.weights, n).astype(np.float64, copy=False)


def _space() -> OccupancySearchSpace:
    return OccupancySearchSpace(
        boosted_occupancy=OccupancyRegressionConfig(
            num_iterations=1,
            mcmc_samples=1,
            occupancy_ratio_max=50.0,
            show_progress=False,
            lgb_params={"min_data_in_leaf": 1, "num_leaves": 3, "verbose": -1},
        ),
        boosted_action_ratio=ActionRatioConfig(
            num_boost_round=1,
            early_stopping_rounds=0,
            refit_on_all_data=False,
            show_progress=False,
            lgb_params={"min_data_in_leaf": 1, "num_leaves": 3, "verbose": -1},
        ),
        boosted_source_state_ratio=SourceStateRatioConfig(
            num_boost_round=1,
            early_stopping_rounds=0,
            refit_on_all_data=False,
            show_progress=False,
            lgb_params={"min_data_in_leaf": 1, "num_leaves": 3, "verbose": -1},
        ),
        boosted_transition_ratio=TransitionRatioConfig(
            num_boost_round=1,
            permutation_samples=1,
            early_stopping_rounds=0,
            refit_on_all_data=False,
            show_progress=False,
            lgb_params={"min_data_in_leaf": 1, "num_leaves": 3, "verbose": -1},
        ),
        boosted_candidates=(
            {"occupancy": {"fixed_point_damping": 0.5}},
            {"occupancy": {"fixed_point_damping": 0.75}},
        ),
    )


def _base_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    states = np.array([[0.0], [0.0], [1.0], [1.0]])
    actions = np.zeros((4, 1))
    next_states = states.copy()
    target_actions = actions.copy()
    rewards = np.ones(4)
    return states, actions, next_states, target_actions, rewards


def test_occupancy_target_validation_discounted_moments_selects_moment_match(monkeypatch) -> None:
    import occupancy_ratio._tuning_impl as tuning_impl

    def fake_fit_family(**kwargs):
        damping = float(kwargs["configs"]["occupancy"].fixed_point_damping)
        return _FakeRatioModel(np.array([2.0, 2.0, 0.0, 0.0]) if damping == 0.75 else np.ones(4))

    monkeypatch.setattr(tuning_impl, "_fit_family", fake_fit_family)
    states, actions, next_states, target_actions, rewards = _base_arrays()
    result = tune_occupancy_ratio_with_target_validation(
        states=states,
        actions=actions,
        next_states=next_states,
        target_actions=target_actions,
        rewards=rewards,
        gamma=0.5,
        search_space=_space(),
        config=OccupancyTuningConfig(families=("boosted",), max_candidates=2, stagewise=False, seed=11),
        validation_states=np.zeros((4, 1)),
        validation_actions=np.zeros((4, 1)),
        validation_rewards=np.ones(4),
        validation_episode_ids=np.array([0, 0, 1, 1]),
        validation_timestep=np.array([0, 1, 0, 1]),
        validation_terminals=np.array([0, 1, 0, 1]),
    )
    assert isinstance(result, OccupancyTargetValidationResult)
    assert result.selected_candidate_id == "boosted_001"
    rows = {row["candidate_id"]: row for row in result.candidate_rows()}
    assert rows["boosted_001"]["score"] < rows["boosted_000"]["score"]
    assert rows["boosted_001"]["metric_guardrail_passed"] == 1.0


def test_occupancy_target_validation_scalar_ope_keeps_guardrails(monkeypatch) -> None:
    import occupancy_ratio._tuning_impl as tuning_impl

    def fake_fit_family(**kwargs):
        damping = float(kwargs["configs"]["occupancy"].fixed_point_damping)
        return _FakeRatioModel(np.ones(4) if damping == 0.75 else np.array([100.0, 0.0, 0.0, 0.0]))

    monkeypatch.setattr(tuning_impl, "_fit_family", fake_fit_family)
    states, actions, next_states, target_actions, _ = _base_arrays()
    rewards = np.array([1.0, 0.0, 0.0, 0.0])
    result = tune_occupancy_ratio_with_target_validation(
        states=states,
        actions=actions,
        next_states=next_states,
        target_actions=target_actions,
        rewards=rewards,
        gamma=0.5,
        search_space=_space(),
        config=OccupancyTuningConfig(families=("boosted",), max_candidates=2, stagewise=False, seed=12),
        score_mode="scalar_ope",
        target_value=50.0,
        target_value_se=0.0,
    )
    rows = {row["candidate_id"]: row for row in result.candidate_rows()}
    assert rows["boosted_000"]["guardrail_passed"] == 0.0
    assert result.selected_candidate_id == "boosted_001"
    assert result.validation_diagnostics["validation_label_scope"] == "scalar_ope_only"


def test_occupancy_target_validation_selection_rule_defaults_to_min_score(monkeypatch) -> None:
    import occupancy_ratio._tuning_impl as tuning_impl

    def fake_fit_family(**kwargs):
        damping = float(kwargs["configs"]["occupancy"].fixed_point_damping)
        return _FakeRatioModel(np.array([2.0, 1.0, 0.5, 0.5]) if damping == 0.75 else np.ones(4))

    monkeypatch.setattr(tuning_impl, "_fit_family", fake_fit_family)
    states, actions, next_states, target_actions, _ = _base_arrays()
    rewards = np.array([1.0, 0.0, 0.0, 0.0])
    result = tune_occupancy_ratio_with_target_validation(
        states=states,
        actions=actions,
        next_states=next_states,
        target_actions=target_actions,
        rewards=rewards,
        gamma=0.5,
        search_space=_space(),
        config=OccupancyTuningConfig(families=("boosted",), max_candidates=2, stagewise=False, seed=13),
        score_mode="scalar_ope",
        target_value=1.0,
        target_value_se=10.0,
    )
    assert result.selected_candidate_id == "boosted_001"
    assert result.selection_rule == "min_score"
    assert result.validation_diagnostics["validation_selection_rule"] == "guardrails_then_min_score"
    assert result.validation_diagnostics["selected_min_score_candidate_id"] == "boosted_001"
    assert result.validation_diagnostics["selected_one_se_candidate_id"] == "boosted_000"

    one_se_result = tune_occupancy_ratio_with_target_validation(
        states=states,
        actions=actions,
        next_states=next_states,
        target_actions=target_actions,
        rewards=rewards,
        gamma=0.5,
        search_space=_space(),
        config=OccupancyTuningConfig(families=("boosted",), max_candidates=2, stagewise=False, seed=13),
        score_mode="scalar_ope",
        selection_rule="one_se",
        target_value=1.0,
        target_value_se=10.0,
    )
    assert one_se_result.selected_candidate_id == "boosted_000"
    assert one_se_result.selection_rule == "one_se"
    assert one_se_result.validation_diagnostics["validation_selection_rule"] == "guardrails_then_one_se"


def test_occupancy_target_validation_rejects_invalid_selection_rule() -> None:
    states, actions, next_states, target_actions, rewards = _base_arrays()
    with pytest.raises(ValueError, match="selection_rule"):
        tune_occupancy_ratio_with_target_validation(
            states=states,
            actions=actions,
            next_states=next_states,
            target_actions=target_actions,
            rewards=rewards,
            gamma=0.5,
            search_space=_space(),
            config=OccupancyTuningConfig(families=("boosted",), max_candidates=1, stagewise=False, seed=14),
            selection_rule="stable",  # type: ignore[arg-type]
            validation_states=states,
            validation_actions=actions,
            validation_rewards=rewards,
            validation_episode_ids=np.array([0, 0, 1, 1]),
            validation_timestep=np.array([0, 1, 0, 1]),
            validation_terminals=np.array([0, 1, 0, 1]),
        )


def test_target_validation_exports() -> None:
    import occupancy_ratio
    import occupancy_ratio.tuning as tuning

    assert occupancy_ratio.tune_occupancy_ratio_with_target_validation
    assert tuning.tune_occupancy_ratio_with_target_validation
    assert occupancy_ratio.OccupancyTargetValidationResult.__module__ == "occupancy_ratio.tuning"
    assert inspect.signature(occupancy_ratio.tune_occupancy_ratio_with_target_validation).parameters["selection_rule"].default == "min_score"
    assert inspect.signature(tuning.tune_occupancy_ratio_with_target_validation).parameters["selection_rule"].default == "min_score"
