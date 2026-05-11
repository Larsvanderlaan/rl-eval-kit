from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .environments import NonlinearMDP
from .policies import SoftmaxPolicy


Array = np.ndarray


@dataclass
class TransitionBatch:
    states: Array
    actions: Array
    rewards: Array
    next_states: Array
    next_actions: Array
    behavior_probs: Array
    target_probs: Array

    def __len__(self) -> int:
        return int(self.states.shape[0])

    def subset(self, indices: Array) -> "TransitionBatch":
        idx = np.asarray(indices, dtype=int)
        return TransitionBatch(
            states=self.states[idx],
            actions=self.actions[idx],
            rewards=self.rewards[idx],
            next_states=self.next_states[idx],
            next_actions=self.next_actions[idx],
            behavior_probs=self.behavior_probs[idx],
            target_probs=self.target_probs[idx],
        )


def sample_transition_batch(
    env: NonlinearMDP,
    behavior_policy: SoftmaxPolicy,
    target_policy: SoftmaxPolicy,
    n_samples: int,
    seed: int,
    shifted_initial_states: bool = False,
) -> TransitionBatch:
    rng = np.random.default_rng(seed)
    states = env.initial_states(n_samples, rng, shifted=shifted_initial_states)
    behavior_all = behavior_policy.action_probabilities(states)
    target_all = target_policy.action_probabilities(states)
    actions = behavior_policy.sample(states, rng)
    rewards = env.reward(states, actions, rng)
    next_states = env.step(states, actions, rng)
    next_actions = target_policy.sample(next_states, rng)
    rows = np.arange(n_samples)
    return TransitionBatch(
        states=states,
        actions=actions,
        rewards=rewards,
        next_states=next_states,
        next_actions=next_actions,
        behavior_probs=behavior_all[rows, actions],
        target_probs=target_all[rows, actions],
    )


def sample_initial_eval_states(env: NonlinearMDP, n: int, seed: int, shifted: bool = False) -> Array:
    return env.initial_states(n, np.random.default_rng(seed), shifted=shifted)
