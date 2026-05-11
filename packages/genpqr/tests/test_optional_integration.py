from __future__ import annotations

import importlib.util
import os

import numpy as np
import pytest

from genpqr import (
    ActionSpaceSpec,
    DeepGenPQRConfig,
    DiscreteNormalizationPolicy,
    D3RLPYFQEstimator,
    fit_deep_genpqr,
)
from genpqr.benchmarks import make_tabular_chain
from genpqr.datasets import TransitionDataset


RUN_OPTIONAL = os.environ.get("GENPQR_RUN_OPTIONAL_INTEGRATION") == "1"


@pytest.mark.optional_integration
@pytest.mark.skipif(not RUN_OPTIONAL, reason="optional integration disabled")
def test_optional_integration_placeholder() -> None:
    """Placeholder proving optional integrations are opt-in in CI."""

    assert True


@pytest.mark.optional_integration
@pytest.mark.skipif(not RUN_OPTIONAL, reason="optional integration disabled")
@pytest.mark.skipif(
    importlib.util.find_spec("gymnasium") is None
    or importlib.util.find_spec("imitation") is None
    or importlib.util.find_spec("stable_baselines3") is None,
    reason="imitation/SB3/Gymnasium stack is not installed",
)
@pytest.mark.parametrize("policy", ["deep_airl", "deep_gail"])
def test_live_imitation_airl_gail_tiny_env(policy: str) -> None:
    import gymnasium as gym

    env = gym.make("CartPole-v1")
    dataset = make_tabular_chain(8, seed=7)
    result = fit_deep_genpqr(
        dataset=dataset,
        gamma=0.0,
        env=env,
        normalization_policy=DiscreteNormalizationPolicy.uniform(2),
        config=DeepGenPQRConfig(
            policy=policy,
            q_backend="deeppqr_linear",
            policy_config={"total_timesteps": 1, "demo_batch_size": 4},
            q_config={"n_iterations": 1},
        ),
    )
    assert result.action_space == ActionSpaceSpec.discrete(2)
    env.close()


@pytest.mark.optional_integration
@pytest.mark.skipif(not RUN_OPTIONAL, reason="optional integration disabled")
@pytest.mark.skipif(importlib.util.find_spec("d3rlpy") is None, reason="d3rlpy is not installed")
def test_live_d3rlpy_ordered_episode_payload_smoke() -> None:
    dataset = TransitionDataset.from_arrays(
        states=np.zeros((2, 1)),
        actions=np.array([0, 1]),
        next_states=np.zeros((2, 1)),
        terminals=np.array([0.0, 1.0]),
        action_space=ActionSpaceSpec.discrete(2),
        episode_ids=np.array([0, 0]),
        strict_episodes=True,
    )
    estimator = D3RLPYFQEstimator(algo=object(), allow_ordered_episodes=True, n_steps=1)
    estimator.preflight(episode_ids=dataset.episode_ids, dataset_metadata=dataset.metadata)
    kwargs = dataset.to_d3rlpy_kwargs()
    assert kwargs["observations"].shape == (2, 1)
    assert kwargs["actions"].shape[0] == 2


@pytest.mark.optional_integration
@pytest.mark.skipif(not RUN_OPTIONAL, reason="optional integration disabled")
@pytest.mark.skipif(importlib.util.find_spec("scope_rl") is None, reason="SCOPE-RL is not installed")
def test_live_scope_rl_import_smoke() -> None:
    import scope_rl

    assert scope_rl is not None
