from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from FQE_neurips.utils import TransitionBatch

from .envs import LinearGaussianEnv
from .policies import GaussianLinearPolicy
from .truth import PolicyMomentSequence, build_policy_moment_sequence


@dataclass
class ContinuousTransitionBatch:
    """Continuous offline batch plus auxiliary time-index metadata."""

    batch: TransitionBatch
    time_indices: np.ndarray


def _sample_states_at_times(
    sequence: PolicyMomentSequence,
    time_indices: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    states = np.zeros((time_indices.shape[0], 2), dtype=np.float64)
    for t in np.unique(time_indices):
        mask = time_indices == t
        if t <= sequence.head_steps:
            mean_t = sequence.state_means[int(t)]
            cov_t = sequence.state_covariances[int(t)]
        else:
            mean_t = sequence.stationary_mean
            cov_t = sequence.stationary_cov
        states[mask] = rng.multivariate_normal(mean=mean_t, cov=cov_t, size=int(mask.sum()))
    return states


def sample_discounted_behavior_batch(
    env: LinearGaussianEnv,
    behavior_policy: GaussianLinearPolicy,
    target_policy: GaussianLinearPolicy,
    gamma: float,
    n_samples: int,
    seed: int,
    sequence: PolicyMomentSequence | None = None,
) -> ContinuousTransitionBatch:
    rng = np.random.default_rng(seed)
    if sequence is None:
        sequence = build_policy_moment_sequence(env=env, policy=behavior_policy, gamma=gamma)
    time_indices = rng.geometric(p=1.0 - gamma, size=n_samples) - 1
    states = _sample_states_at_times(sequence, time_indices, rng)
    actions = behavior_policy.sample_actions(states, rng)
    rewards = env.sample_reward(states, actions, rng)
    next_states = env.step(states, actions, rng)
    next_actions = target_policy.sample_actions(next_states, rng)
    return ContinuousTransitionBatch(
        batch=TransitionBatch(
            states=states,
            actions=actions,
            rewards=rewards,
            next_states=next_states,
            next_actions=next_actions,
        ),
        time_indices=time_indices.astype(np.int64),
    )


def sample_stationary_behavior_batch(
    env: LinearGaussianEnv,
    behavior_policy: GaussianLinearPolicy,
    target_policy: GaussianLinearPolicy,
    n_samples: int,
    seed: int,
    sequence: PolicyMomentSequence | None = None,
) -> ContinuousTransitionBatch:
    """Sample iid transitions from the behavior stationary occupancy."""

    rng = np.random.default_rng(seed)
    if sequence is None:
        sequence = build_policy_moment_sequence(env=env, policy=behavior_policy, gamma=1.0)
    states = rng.multivariate_normal(
        mean=sequence.stationary_mean,
        cov=sequence.stationary_cov,
        size=n_samples,
    )
    actions = behavior_policy.sample_actions(states, rng)
    rewards = env.sample_reward(states, actions, rng)
    next_states = env.step(states, actions, rng)
    next_actions = target_policy.sample_actions(next_states, rng)
    return ContinuousTransitionBatch(
        batch=TransitionBatch(
            states=states,
            actions=actions,
            rewards=rewards,
            next_states=next_states,
            next_actions=next_actions,
        ),
        time_indices=-np.ones(n_samples, dtype=np.int64),
    )


def sample_reference_behavior_batch(
    env: LinearGaussianEnv,
    behavior_policy: GaussianLinearPolicy,
    target_policy: GaussianLinearPolicy,
    ratio_gamma: float,
    n_samples: int,
    seed: int,
    sequence: PolicyMomentSequence | None = None,
) -> ContinuousTransitionBatch:
    """Sample from the behavior distribution matched to `ratio_gamma`."""

    if ratio_gamma >= 1.0 - 1e-12:
        return sample_stationary_behavior_batch(
            env=env,
            behavior_policy=behavior_policy,
            target_policy=target_policy,
            n_samples=n_samples,
            seed=seed,
            sequence=sequence,
        )
    return sample_discounted_behavior_batch(
        env=env,
        behavior_policy=behavior_policy,
        target_policy=target_policy,
        gamma=ratio_gamma,
        n_samples=n_samples,
        seed=seed,
        sequence=sequence,
    )


def sample_behavior_trajectory_batch(
    env: LinearGaussianEnv,
    behavior_policy: GaussianLinearPolicy,
    target_policy: GaussianLinearPolicy,
    n_samples: int,
    seed: int,
    trajectory_length: int = 40,
) -> ContinuousTransitionBatch:
    """Optional diagnostic sampler using explicit trajectories instead of discounted IID draws."""

    rng = np.random.default_rng(seed)
    states = []
    actions = []
    rewards = []
    next_states = []
    time_indices = []
    while len(states) < n_samples:
        state = rng.multivariate_normal(mean=env.config.initial_mean, cov=env.config.initial_cov, size=1)
        for t in range(trajectory_length):
            action = behavior_policy.sample_actions(state, rng)
            reward = env.sample_reward(state, action, rng)
            next_state = env.step(state, action, rng)
            states.append(state.reshape(-1))
            actions.append(action.reshape(-1))
            rewards.append(float(reward[0]))
            next_states.append(next_state.reshape(-1))
            time_indices.append(t)
            state = next_state
            if len(states) >= n_samples:
                break
    states_arr = np.asarray(states, dtype=np.float64)
    next_states_arr = np.asarray(next_states, dtype=np.float64)
    next_actions_arr = target_policy.sample_actions(next_states_arr, rng)
    return ContinuousTransitionBatch(
        batch=TransitionBatch(
            states=states_arr,
            actions=np.asarray(actions, dtype=np.float64),
            rewards=np.asarray(rewards, dtype=np.float64),
            next_states=next_states_arr,
            next_actions=next_actions_arr,
        ),
        time_indices=np.asarray(time_indices, dtype=np.int64),
    )
