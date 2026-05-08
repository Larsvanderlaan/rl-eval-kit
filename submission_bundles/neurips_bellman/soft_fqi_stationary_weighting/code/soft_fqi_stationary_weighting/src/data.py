from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .env import GridMDP


@dataclass
class TransitionBatch:
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray


def sample_transition_batch(
    mdp: GridMDP,
    behavior_state_dist: np.ndarray,
    behavior_policy: np.ndarray,
    n_samples: int,
    seed: int,
) -> TransitionBatch:
    rng = np.random.default_rng(seed)
    state_dist = np.asarray(behavior_state_dist, dtype=np.float64).reshape(-1)
    state_dist = state_dist / np.maximum(state_dist.sum(), 1e-300)
    states = rng.choice(mdp.n_states, size=int(n_samples), p=state_dist)
    actions = np.zeros(int(n_samples), dtype=np.int64)
    next_states = np.zeros(int(n_samples), dtype=np.int64)
    for state in np.unique(states):
        mask = states == state
        actions[mask] = rng.choice(mdp.n_actions, size=int(mask.sum()), p=behavior_policy[state])
    for state in np.unique(states):
        state_mask = states == state
        for action in range(mdp.n_actions):
            mask = state_mask & (actions == action)
            if np.any(mask):
                next_states[mask] = rng.choice(mdp.n_states, size=int(mask.sum()), p=mdp.transition[state, action])
    rewards = mdp.reward[states, actions].astype(np.float64)
    return TransitionBatch(
        states=states.astype(np.int64),
        actions=actions.astype(np.int64),
        rewards=rewards,
        next_states=next_states.astype(np.int64),
    )
