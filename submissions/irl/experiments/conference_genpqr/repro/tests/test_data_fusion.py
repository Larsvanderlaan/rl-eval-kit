"""Correctness tests for the complementary-source experiment."""

from __future__ import annotations

import numpy as np
import pytest

from submissions.irl.experiments.conference_genpqr.repro.data_fusion import (
    fit_state_reward_regressor,
    generate_data_fusion_world,
    recover_reward_matrix,
)
from submissions.irl.experiments.conference_genpqr.repro.data_generation import SimulationConfig
from submissions.irl.experiments.conference_genpqr.repro.deeppqr_baseline import fit_deeppqr_baseline
from submissions.irl.experiments.conference_genpqr.repro.policy_estimation import (
    fit_behavior_cloning_policy,
)
from submissions.irl.experiments.conference_genpqr.repro.q_evaluation import fit_fqe_neural


class FixedPolicy:
    def __init__(self, probabilities: np.ndarray):
        self.probabilities = np.asarray(probabilities, dtype=float)
        self.n_actions = self.probabilities.shape[0]

    def predict_proba(self, states: np.ndarray) -> np.ndarray:
        return np.tile(self.probabilities[None, :], (states.shape[0], 1))


class FixedQ:
    def __init__(self, matrix: np.ndarray):
        self.matrix = np.asarray(matrix, dtype=float)

    def predict_all_actions(self, states: np.ndarray) -> np.ndarray:
        if states.shape[0] != self.matrix.shape[0]:
            raise ValueError("Unexpected state count.")
        return self.matrix


@pytest.fixture
def tiny_world():
    config = SimulationConfig(
        state_dim=3,
        n_actions=4,
        horizon=3,
        gamma=0.9,
        seed=17,
    )
    return generate_data_fusion_world(
        seed=17,
        config=config,
        behavior_trajectories=40,
        outcome_trajectories=50,
        test_trajectories=20,
    )


def test_sources_are_complementary_independent_and_nested(tiny_world) -> None:
    outcome = tiny_world.outcome
    behavior = tiny_world.behavior
    assert not hasattr(outcome, "actions")
    assert not hasattr(behavior, "rewards")
    assert outcome.states.shape == (150, 3)
    assert behavior.states.shape == behavior.next_states.shape == (120, 3)
    assert not np.array_equal(outcome.states[:120], behavior.states)

    subset = outcome.subset_trajectories(7)
    assert subset.states.shape == (21, 3)
    assert set(np.unique(subset.trajectory_ids)) == set(range(7))
    counts = np.bincount(subset.trajectory_ids, minlength=7)
    np.testing.assert_array_equal(counts, np.full(7, 3))


def test_oracle_normalization_and_anchor_identities(tiny_world) -> None:
    for states in [
        tiny_world.outcome.states,
        tiny_world.behavior.states,
        tiny_world.test.states,
    ]:
        reward = tiny_world.oracle.reward_matrix(states)
        policy = tiny_world.oracle.policy_probs(states)
        g_values = tiny_world.oracle.g(states)
        np.testing.assert_allclose(reward[:, 0], 0.0, atol=1e-10)
        np.testing.assert_allclose(np.sum(policy, axis=1), 1.0, atol=1e-12)
        np.testing.assert_allclose(g_values, np.sum(policy * reward, axis=1), atol=1e-12)


def test_recovery_enforces_supplied_normalization() -> None:
    rng = np.random.default_rng(9)
    states = rng.normal(size=(13, 3))
    q_matrix = rng.normal(size=(13, 4))
    raw_policy = rng.uniform(size=(13, 4))
    policy = raw_policy / raw_policy.sum(axis=1, keepdims=True)
    g_values = rng.normal(size=13)
    recovered = recover_reward_matrix(
        q_estimate=FixedQ(q_matrix),
        normalization_policy=lambda _: policy,
        normalization_function=lambda _: g_values,
        states=states,
    )
    np.testing.assert_allclose(np.sum(policy * recovered, axis=1), g_values, atol=1e-12)


def test_g_regressor_uses_state_outcome_source_and_is_deterministic(tiny_world) -> None:
    source = tiny_world.outcome.subset_trajectories(20)
    first = fit_state_reward_regressor(source, seed=33, n_estimators=10)
    second = fit_state_reward_regressor(source, seed=33, n_estimators=10)
    first_prediction = first.predict(tiny_world.test.states)
    second_prediction = second.predict(tiny_world.test.states)
    assert first_prediction.shape == (tiny_world.test.states.shape[0],)
    assert np.all(np.isfinite(first_prediction))
    np.testing.assert_allclose(first_prediction, second_prediction, atol=0.0, rtol=0.0)


def test_torch_estimators_respect_declared_seeds(tiny_world) -> None:
    behavior = tiny_world.behavior
    first_policy = fit_behavior_cloning_policy(
        behavior.states,
        behavior.actions,
        n_actions=4,
        n_epochs=2,
        seed=101,
    )
    second_policy = fit_behavior_cloning_policy(
        behavior.states,
        behavior.actions,
        n_actions=4,
        n_epochs=2,
        seed=101,
    )
    np.testing.assert_allclose(
        first_policy.predict_proba(tiny_world.test.states),
        second_policy.predict_proba(tiny_world.test.states),
        atol=0.0,
        rtol=0.0,
    )

    fixed_policy = FixedPolicy(np.full(4, 0.25))
    pseudo_rewards = np.linspace(-0.5, 0.5, behavior.actions.shape[0])
    first_q = fit_fqe_neural(
        states=behavior.states,
        actions=behavior.actions,
        rewards=pseudo_rewards,
        next_states=behavior.next_states,
        dones=behavior.dones,
        policy=fixed_policy,
        n_actions=4,
        gamma=0.9,
        n_fqe_iters=1,
        epochs_per_iter=1,
        seed=202,
    )
    second_q = fit_fqe_neural(
        states=behavior.states,
        actions=behavior.actions,
        rewards=pseudo_rewards,
        next_states=behavior.next_states,
        dones=behavior.dones,
        policy=fixed_policy,
        n_actions=4,
        gamma=0.9,
        n_fqe_iters=1,
        epochs_per_iter=1,
        seed=202,
    )
    np.testing.assert_allclose(
        first_q.predict_all_actions(tiny_world.test.states),
        second_q.predict_all_actions(tiny_world.test.states),
        atol=0.0,
        rtol=0.0,
    )


def test_deeppqr_ignores_withheld_reward_values_and_rejects_no_anchor(tiny_world) -> None:
    behavior = tiny_world.behavior
    policy = fit_behavior_cloning_policy(
        behavior.states,
        behavior.actions,
        n_actions=4,
        n_epochs=2,
        seed=303,
    )
    common = {
        "states": behavior.states,
        "actions": behavior.actions,
        "next_states": behavior.next_states,
        "dones": behavior.dones,
        "n_actions": 4,
        "gamma": 0.9,
        "g_values": np.zeros(behavior.actions.shape[0]),
        "behavior_policy": policy,
        "seed": 404,
        "anchor_n_iters": 2,
        "anchor_epochs_per_iter": 1,
        "reward_n_epochs": 2,
    }
    first = fit_deeppqr_baseline(
        rewards=np.zeros(behavior.actions.shape[0]),
        **common,
    )
    second = fit_deeppqr_baseline(
        rewards=np.linspace(-100.0, 100.0, behavior.actions.shape[0]),
        **common,
    )
    test_states = tiny_world.test.states
    np.testing.assert_allclose(
        first.full_q.predict_all_actions(test_states),
        second.full_q.predict_all_actions(test_states),
        atol=0.0,
        rtol=0.0,
    )
    for action in range(4):
        actions = np.full(test_states.shape[0], action, dtype=int)
        np.testing.assert_allclose(
            first.reward_network.predict(test_states, actions),
            second.reward_network.predict(test_states, actions),
            atol=0.0,
            rtol=0.0,
        )

    no_anchor = dict(common)
    no_anchor["actions"] = np.ones_like(behavior.actions)
    with pytest.raises(ValueError, match="No observations"):
        fit_deeppqr_baseline(
            rewards=np.zeros(behavior.actions.shape[0]),
            **no_anchor,
        )
