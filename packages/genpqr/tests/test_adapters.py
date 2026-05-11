from __future__ import annotations

import numpy as np
import pytest

from genpqr import (
    ActionSpaceSpec,
    D3RLPYFQEstimator,
    DiscreteNormalizationPolicy,
    GenPQRAdapterError,
    ReusableScopeRLQEstimator,
    ScopeRLDatasetBoundQEstimator,
)
from genpqr.datasets import TransitionDataset
from genpqr.policies import D3RLPYPolicyAdapter
from genpqr.q_estimators import scope_logged_dataset_from_transitions


def test_d3rlpy_preflight_rejects_ambiguous_rowwise_data() -> None:
    estimator = D3RLPYFQEstimator(algo=object(), allow_ordered_episodes=True)
    with pytest.raises(Exception, match="strict ordered episode"):
        estimator.preflight(episode_ids=np.array([0, 0]), dataset_metadata={})


def test_d3rlpy_preflight_accepts_strict_ordered_metadata() -> None:
    dataset = TransitionDataset.from_arrays(
        states=np.zeros((2, 1)),
        actions=np.array([0, 1]),
        next_states=np.zeros((2, 1)),
        terminals=np.array([0.0, 1.0]),
        action_space=ActionSpaceSpec.discrete(2),
        episode_ids=np.array([0, 0]),
        strict_episodes=True,
    )
    estimator = D3RLPYFQEstimator(algo=object(), allow_ordered_episodes=True)
    estimator.preflight(episode_ids=dataset.episode_ids, dataset_metadata=dataset.metadata)


def test_scope_rl_dataset_bound_name_and_guard() -> None:
    estimator = ScopeRLDatasetBoundQEstimator(method="mql", env=object(), evaluation_policies=object())
    with pytest.raises(GenPQRAdapterError, match="dataset-bound"):
        estimator.preflight()


def test_scope_rl_logged_payload_shape_and_action_encoding() -> None:
    payload = scope_logged_dataset_from_transitions(
        states=np.zeros((3, 2)),
        actions=np.array([0, 1, 0]),
        rewards=np.array([0.1, 0.2, 0.3]),
        terminals=np.array([0.0, 0.0, 1.0]),
        action_space=ActionSpaceSpec.discrete(2),
    )
    assert payload["state"].shape == (3, 1, 2)
    assert payload["action"].tolist() == [[0], [1], [0]]
    assert payload["reward"].shape == (3, 1)


def test_reusable_scope_rl_fake_model_receives_encoded_payload() -> None:
    class FakeReusableModel:
        def __init__(self) -> None:
            self.actions_seen = None

        def fit(self, *, actions, **kwargs):
            del kwargs
            self.actions_seen = np.asarray(actions)

        def predict_q(self, states, actions):
            del actions
            return np.zeros(np.asarray(states).shape[0])

    model = FakeReusableModel()
    estimator = ReusableScopeRLQEstimator(model, action_encoding="index")
    fitted = estimator.fit(
        states=np.zeros((3, 2)),
        actions=np.array([0, 1, 0]),
        next_states=np.zeros((3, 2)),
        pseudo_rewards=np.zeros(3),
        normalization_policy=DiscreteNormalizationPolicy.uniform(2),
        gamma=0.9,
    )
    assert model.actions_seen.tolist() == [0, 1, 0]
    assert fitted.backend == "scope_rl_reusable"


def test_d3rlpy_policy_requires_calibrated_log_prob_by_default() -> None:
    class DeterministicAlgo:
        def predict(self, states):
            return np.zeros(np.asarray(states).shape[0], dtype=np.int64)

    adapter = D3RLPYPolicyAdapter(DeterministicAlgo(), ActionSpaceSpec.discrete(2))
    with pytest.raises(GenPQRAdapterError, match="predict_proba"):
        adapter.log_prob(np.zeros((2, 1)), np.array([0, 1]))

    approximate = D3RLPYPolicyAdapter(
        DeterministicAlgo(),
        ActionSpaceSpec.discrete(2),
        allow_approximate_log_prob=True,
    )
    assert np.all(np.isfinite(approximate.log_prob(np.zeros((2, 1)), np.array([0, 1]))))
    assert approximate.diagnostics["approximate_log_prob"] is True


def test_d3rlpy_continuous_policy_requires_log_density_by_default() -> None:
    class MeanAlgo:
        def predict(self, states):
            return np.zeros((np.asarray(states).shape[0], 1))

    adapter = D3RLPYPolicyAdapter(MeanAlgo(), ActionSpaceSpec.continuous(1))
    with pytest.raises(GenPQRAdapterError, match="log_prob"):
        adapter.log_prob(np.zeros((2, 1)), np.zeros((2, 1)))


def test_d3rlpy_ordered_episodes_require_shifted_next_states() -> None:
    with pytest.raises(ValueError, match="next_states"):
        TransitionDataset.from_arrays(
            states=np.array([[0.0], [1.0]]),
            actions=np.array([0, 1]),
            next_states=np.array([[0.0], [1.0]]),
            terminals=np.array([0.0, 1.0]),
            action_space=ActionSpaceSpec.discrete(2),
            episode_ids=np.array([0, 0]),
            strict_episodes=True,
        )
