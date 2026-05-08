from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from FQE_calibration_neurips.scripts.run_experiment import run_config
from FQE_calibration_neurips.src.calibration.calibrators import fit_calibrator, fit_value_bellman_calibrator
from FQE_calibration_neurips.src.calibration.protocols import ProtocolContext, run_cross_calibration, run_split
from FQE_calibration_neurips.src.calibration.targets import action_importance_weights, policy_value_predictions
from FQE_calibration_neurips.src.data import sample_initial_eval_states, sample_transition_batch
from FQE_calibration_neurips.src.environments import NonlinearMDP, NonlinearMDPConfig, monte_carlo_oracle_value
from FQE_calibration_neurips.src.estimators.baselines import fit_estimator
from FQE_calibration_neurips.src.estimators.model_wrappers import PredictionDistortionConfig, PredictionDistortionWrapper
from FQE_calibration_neurips.src.estimators.saddle_point_bellman import (
    IterativeSaddlePointBellmanConfig,
    fit_iterative_saddle_point_bellman,
)
from FQE_calibration_neurips.src.policies import make_policy_pair
from FQE_calibration_neurips.src.utils import kfold_indices, train_calibration_split


def test_all_calibrators_fit_and_predict_finite() -> None:
    x = np.linspace(-2.0, 2.0, 80)
    y = 0.5 + np.tanh(x)
    for name in ["linear", "histogram", "isotonic", "isotonic_histogram"]:
        cal = fit_calibrator(name, x, y, n_bins=6, min_bin_size=4)
        pred = cal.predict(x)
        assert pred.shape == x.shape
        assert np.all(np.isfinite(pred))


def test_value_bellman_calibrators_fit_and_predict_finite() -> None:
    x = np.linspace(-1.5, 1.5, 80)
    next_x = 0.8 * x + 0.2
    rewards = 0.25 + 0.4 * x
    weights = np.linspace(0.2, 2.0, x.size)
    for name in ["linear", "histogram", "isotonic", "isotonic_histogram"]:
        cal = fit_value_bellman_calibrator(
            name,
            x,
            next_x,
            rewards,
            gamma=0.9,
            sample_weight=weights,
            n_iterations=3,
            n_bins=5,
            min_bin_size=4,
        )
        pred = cal.predict(x)
        assert pred.shape == x.shape
        assert np.all(np.isfinite(pred))
        assert cal.diagnostics["calibration_object"] == "value"


def test_policy_value_predictions_average_q_under_target_policy() -> None:
    class ActionModel:
        def predict_q(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
            return states[:, 0] + 10.0 * actions

    class FixedPolicy:
        def action_probabilities(self, states: np.ndarray) -> np.ndarray:
            return np.tile(np.array([[0.25, 0.75]]), (states.shape[0], 1))

    states = np.array([[1.0, 0.0], [2.0, 0.0], [-1.0, 0.0]])
    values = policy_value_predictions(ActionModel(), states, FixedPolicy())  # type: ignore[arg-type]
    assert np.allclose(values, states[:, 0] + 7.5)


def test_reward_shift_defaults_preserve_expected_reward() -> None:
    states = np.array([[0.1, -0.2, 0.3], [0.4, 0.0, -0.5]])
    actions = np.array([0, 1])
    base = NonlinearMDP(NonlinearMDPConfig(state_dim=3, n_actions=2, reward_noise=0.0, seed=313))
    identity = NonlinearMDP(
        NonlinearMDPConfig(
            state_dim=3,
            n_actions=2,
            reward_noise=0.0,
            seed=313,
            reward_shift_intercept=0.0,
            reward_shift_scale=1.0,
        )
    )
    assert np.allclose(base.expected_reward(states, actions), identity.expected_reward(states, actions))


def test_reward_shift_applies_after_shared_base_reward() -> None:
    states = np.array([[0.1, -0.2, 0.3], [0.4, 0.0, -0.5], [-0.7, 0.2, 0.1]])
    actions = np.array([0, 1, 0])
    old = NonlinearMDP(NonlinearMDPConfig(state_dim=3, n_actions=2, reward_noise=0.0, seed=414))
    current = NonlinearMDP(
        NonlinearMDPConfig(
            state_dim=3,
            n_actions=2,
            reward_noise=0.0,
            seed=414,
            reward_shift_intercept=0.6,
            reward_shift_scale=1.25,
        )
    )
    assert np.allclose(current.expected_reward(states, actions), 0.6 + 1.25 * old.expected_reward(states, actions))


def test_action_importance_weights_are_clipped_and_normalized() -> None:
    from FQE_calibration_neurips.src.data import TransitionBatch

    batch = TransitionBatch(
        states=np.zeros((3, 2)),
        actions=np.zeros(3, dtype=int),
        rewards=np.zeros(3),
        next_states=np.zeros((3, 2)),
        next_actions=np.zeros(3, dtype=int),
        behavior_probs=np.array([0.5, 0.1, 0.01]),
        target_probs=np.array([0.25, 0.2, 0.5]),
    )
    raw = np.array([0.5, 2.0, 50.0])
    clipped = np.minimum(raw, 20.0)
    expected = clipped / clipped.mean()
    assert np.allclose(action_importance_weights(batch, clip=20.0, normalize=True), expected)


def test_split_indices_are_disjoint_and_cover_dataset() -> None:
    train, cal = train_calibration_split(101, 0.7, 123)
    assert set(train).isdisjoint(set(cal))
    assert sorted(np.concatenate([train, cal]).tolist()) == list(range(101))


def test_cross_calibration_fold_indices_are_out_of_fold() -> None:
    seen_holdouts: list[int] = []
    for train, holdout in kfold_indices(53, 5, 19):
        assert set(train).isdisjoint(set(holdout))
        seen_holdouts.extend(holdout.tolist())
    assert sorted(seen_holdouts) == list(range(53))


class _FakeModel:
    def predict_q(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        return np.asarray(states)[:, 0] * 0.0 + np.asarray(actions) * 0.0


class _FakeResult:
    model = _FakeModel()


def _tiny_context() -> ProtocolContext:
    env = NonlinearMDP(NonlinearMDPConfig(state_dim=3, n_actions=2, gamma=0.9, horizon=5, seed=77))
    target, behavior = make_policy_pair(3, 2, shift=0.1, coverage="good", seed=78)
    train = sample_transition_batch(env, behavior, target, 40, seed=79)
    test = sample_transition_batch(env, behavior, target, 30, seed=80)
    init = sample_initial_eval_states(env, 20, seed=81)
    oracle = monte_carlo_oracle_value(env, target, 20, seed=82, initial_states=init)
    return ProtocolContext(
        batch=train,
        test_batch=test,
        initial_states=init,
        oracle_value=oracle,
        env=env,
        target_policy=target,
        learner="random_feature_fqe",
        learner_params={},
        gamma=env.gamma,
        seed=83,
        coverage="good",
        reward_noise=0.0,
    )


def test_split_protocol_does_not_use_test_set_for_fit_or_calibration(monkeypatch: pytest.MonkeyPatch) -> None:
    import FQE_calibration_neurips.src.calibration.protocols as protocols

    ctx = _tiny_context()
    fit_batches: list[int] = []
    calibration_batches: list[int] = []

    def fake_fit(_learner, batch, *_args, **_kwargs):
        fit_batches.append(id(batch))
        return _FakeResult()

    def fake_value_arrays(_model, batch, *_args, **_kwargs):
        calibration_batches.append(id(batch))
        return np.zeros(len(batch)), np.zeros(len(batch)), np.asarray(batch.rewards)

    monkeypatch.setattr(protocols, "fit_estimator", fake_fit)
    monkeypatch.setattr(protocols, "value_calibration_arrays", fake_value_arrays)
    run_split(ctx, "linear", "value_bellman", 0.75, {})
    assert id(ctx.test_batch) not in fit_batches
    assert id(ctx.test_batch) not in calibration_batches
    assert len(fit_batches) == 1
    assert len(calibration_batches) == 1


def test_iterated_split_protocol_uses_only_calibration_split(monkeypatch: pytest.MonkeyPatch) -> None:
    import FQE_calibration_neurips.src.calibration.protocols as protocols

    ctx = _tiny_context()
    _, cal_idx = train_calibration_split(len(ctx.batch), 0.75, ctx.seed + 29)
    value_fit_lengths: list[int] = []

    def fake_fit(_learner, batch, *_args, **_kwargs):
        return _FakeResult()

    def fake_value_arrays(_model, batch, *_args, **_kwargs):
        return np.zeros(len(batch)), np.ones(len(batch)), np.asarray(batch.rewards)

    def fake_value_calibrator(_name, values, next_values, rewards, *_args, **_kwargs):
        value_fit_lengths.append(len(values))
        assert len(values) == len(next_values) == len(rewards)
        return fit_value_bellman_calibrator("linear", values, next_values, rewards, gamma=0.9, n_iterations=1)

    monkeypatch.setattr(protocols, "fit_estimator", fake_fit)
    monkeypatch.setattr(protocols, "value_calibration_arrays", fake_value_arrays)
    monkeypatch.setattr(protocols, "fit_value_bellman_calibrator", fake_value_calibrator)
    run_split(ctx, "iterated_isotonic_bellman", "value_bellman", 0.75, {"bellman_iterations": 2})
    assert value_fit_lengths == [len(cal_idx)]
    assert value_fit_lengths[0] != len(ctx.test_batch)


def test_cross_protocol_uses_fold_models_with_pointwise_median_and_no_test_calibration(monkeypatch: pytest.MonkeyPatch) -> None:
    import FQE_calibration_neurips.src.calibration.protocols as protocols

    ctx = _tiny_context()
    fit_lengths: list[int] = []
    calibration_batches: list[int] = []

    def fake_fit(_learner, batch, *_args, **_kwargs):
        fit_lengths.append(len(batch))
        return _FakeResult()

    def fake_value_arrays(_model, batch, *_args, **_kwargs):
        calibration_batches.append(id(batch))
        return np.zeros(len(batch)), np.ones(len(batch)), np.asarray(batch.rewards)

    monkeypatch.setattr(protocols, "fit_estimator", fake_fit)
    monkeypatch.setattr(protocols, "value_calibration_arrays", fake_value_arrays)
    row = run_cross_calibration(ctx, "linear", "value_bellman", 4, {})
    assert len(fit_lengths) == 4
    assert all(length < len(ctx.batch) for length in fit_lengths)
    assert id(ctx.test_batch) not in calibration_batches
    assert len(calibration_batches) == 4
    assert row["base_learner_used_all_data"] is False
    assert row["model_diag_cross_aggregation"] == "pointwise_median"
    assert float(row["model_diag_cross_final_refit"]) == 0.0


def test_result_schema_consistent_across_estimators(tmp_path: Path) -> None:
    config = {
        "seed": 101,
        "replications": 1,
        "gamma": 0.9,
        "horizon": 8,
        "n_actions": 2,
        "sample_sizes": [80],
        "state_dimensions": [3],
        "coverage_settings": ["good"],
        "reward_noise_settings": [0.01],
        "transition_noise": 0.0,
        "test_transitions": 60,
        "initial_eval_states": 60,
        "oracle_rollouts": 50,
        "misspecification": "well_specified_linear",
        "policy_shift": {"good": 0.05, "moderate": 0.5, "severe": 1.0, "extrapolation": 1.5},
        "baseline_learners": ["linear_fqe", "regularized_bellman", "saddle_point_bellman", "ensemble_fqe", "neural_fqe"],
        "calibration_protocols": ["split"],
        "calibrators": ["linear"],
        "calibration_targets": ["value_bellman"],
        "cross_folds": 2,
        "split_fractions": [0.8],
        "split_comparators": [],
        "calibrator_params": {"n_bins": 4, "min_bin_size": 4},
        "learner_params": {
            "linear_fqe": {"feature_type": "linear", "ridge": 0.0001, "n_iters": 3},
            "regularized_bellman": {"feature_type": "linear", "ridge": 0.0001},
            "saddle_point_bellman": {"feature_type": "linear", "ridge": 0.0001, "critic_ridge": 0.0001},
            "ensemble_fqe": {"feature_type": "linear", "n_members": 2, "ridge": 0.0001, "n_iters": 3},
            "neural_fqe": {"hidden_dims": [8], "n_iters": 2, "epochs_per_iter": 1, "batch_size": 32, "device": "cpu"},
        },
    }
    rows = run_config(config, tmp_path, debug=False)
    keyset = set(rows[0])
    for row in rows:
        assert set(row) == keyset
        assert np.isfinite(float(row["value_estimate"]))
        assert np.isfinite(float(row["oracle_value"]))
        assert "diagnostic_warning_message" in row
        assert row["calibration_object"] == "value"
        assert row["calibration_weight_scheme"] == "action_ratio"
        assert np.isfinite(float(row["true_v_mse"]))
        assert np.isfinite(float(row["importance_weight_ess"]))


def test_legacy_q_calibration_targets_rejected_by_default(tmp_path: Path) -> None:
    config = {
        "seed": 303,
        "replications": 1,
        "sample_sizes": [20],
        "state_dimensions": [2],
        "coverage_settings": ["good"],
        "reward_noise_settings": [0.0],
        "baseline_learners": ["linear_fqe"],
        "calibration_protocols": ["cross"],
        "calibrators": ["linear"],
        "calibration_targets": ["q_value"],
    }
    with pytest.raises(ValueError, match="Q-space calibration targets are disabled"):
        run_config(config, tmp_path, debug=False)


def test_iterative_saddle_returns_finite_values_in_stable_mode() -> None:
    env = NonlinearMDP(NonlinearMDPConfig(state_dim=3, n_actions=2, gamma=0.9, horizon=6, seed=701))
    target, behavior = make_policy_pair(3, 2, shift=0.1, coverage="good", seed=702)
    batch = sample_transition_batch(env, behavior, target, 60, seed=703)
    model = fit_iterative_saddle_point_bellman(
        batch,
        env.n_actions,
        target,
        IterativeSaddlePointBellmanConfig(
            gamma=env.gamma,
            n_components=10,
            max_iters=8,
            step_size=0.002,
            q_ridge=0.01,
            critic_ridge=0.01,
            gradient_clip=10.0,
        ),
        seed=704,
    )
    q = model.predict_q(batch.states[:10], batch.actions[:10])
    assert np.all(np.isfinite(q))
    assert model.diagnostics["saddle_iterations_completed"] >= 1
    assert model.diagnostics["saddle_exploding_flag"] == 0.0


def test_unstable_iterative_saddle_sets_failure_diagnostics() -> None:
    env = NonlinearMDP(NonlinearMDPConfig(state_dim=3, n_actions=2, gamma=0.9, horizon=6, seed=711))
    target, behavior = make_policy_pair(3, 2, shift=0.1, coverage="good", seed=712)
    batch = sample_transition_batch(env, behavior, target, 50, seed=713)
    model = fit_iterative_saddle_point_bellman(
        batch,
        env.n_actions,
        target,
        IterativeSaddlePointBellmanConfig(
            gamma=env.gamma,
            n_components=10,
            max_iters=5,
            step_size=1.0,
            q_ridge=1e-10,
            critic_ridge=1e-10,
            gradient_clip=1e6,
            divergence_threshold=1e-6,
            averaging=False,
        ),
        seed=714,
    )
    assert model.diagnostics["saddle_exploding_flag"] == 1.0
    assert model.diagnostics["failure_reason"] == "nonfinite_or_exploding_saddle_parameters"


def test_under_iterated_neural_fqe_reports_fewer_bellman_updates() -> None:
    env = NonlinearMDP(NonlinearMDPConfig(state_dim=3, n_actions=2, gamma=0.9, horizon=6, seed=721))
    target, behavior = make_policy_pair(3, 2, shift=0.1, coverage="good", seed=722)
    batch = sample_transition_batch(env, behavior, target, 50, seed=723)
    weak = fit_estimator(
        "neural_fqe",
        batch,
        env.n_actions,
        target,
        env.gamma,
        {"hidden_dims": [6], "n_iters": 1, "epochs_per_iter": 1, "bootstrap_on_first_iter": False, "batch_size": 16},
        seed=724,
    )
    strong = fit_estimator(
        "neural_fqe",
        batch,
        env.n_actions,
        target,
        env.gamma,
        {"hidden_dims": [6], "n_iters": 3, "epochs_per_iter": 1, "batch_size": 16},
        seed=725,
    )
    assert weak.diagnostics["actual_bellman_iterations"] == 1.0
    assert strong.diagnostics["actual_bellman_iterations"] == 3.0
    assert weak.diagnostics["bootstrap_on_first_iter"] == 0.0


def test_prediction_distortion_wrapper_changes_q_and_preserves_interface() -> None:
    base = _FakeModel()
    wrapper = PredictionDistortionWrapper(base, PredictionDistortionConfig(kind="affine", intercept=1.0, slope=2.0))
    states = np.ones((4, 2))
    actions = np.zeros(4, dtype=int)
    base_q = base.predict_q(states, actions)
    wrapped_q = wrapper.predict_q(states, actions)
    assert np.allclose(wrapped_q, 1.0 + 2.0 * base_q)
    assert wrapper.n_actions is None
    assert wrapper.diagnostics["prediction_distortion_kind"] == "affine"
