from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GridConfig:
    n_x: int = 31
    n_y: int = 31
    low: float = -1.0
    high: float = 1.0


@dataclass(frozen=True)
class EnvConfig:
    action_scale: float = 0.23
    drift_scale: float = 0.13
    interaction_scale: float = 0.055
    process_noise: float = 0.075
    teleport_prob: float = 0.002
    goal: tuple[float, float] = (0.62, 0.62)
    decoy: tuple[float, float] = (-0.55, 0.45)
    goal_width: float = 0.20
    decoy_width: float = 0.23
    barrier_width: float = 0.22
    action_cost: float = 0.035


@dataclass
class GridMDP:
    grid: GridConfig
    config: EnvConfig
    states: np.ndarray
    actions: np.ndarray
    transition: np.ndarray
    reward: np.ndarray
    decoy_reward: np.ndarray

    @property
    def n_states(self) -> int:
        return int(self.states.shape[0])

    @property
    def n_actions(self) -> int:
        return int(self.actions.shape[0])


def make_action_vectors(action_scale: float) -> np.ndarray:
    return np.asarray(
        [
            [0.0, 0.0],
            [action_scale, 0.0],
            [-action_scale, 0.0],
            [0.0, action_scale],
            [0.0, -action_scale],
        ],
        dtype=np.float64,
    )


def build_states(grid: GridConfig) -> np.ndarray:
    xs = np.linspace(grid.low, grid.high, grid.n_x)
    ys = np.linspace(grid.low, grid.high, grid.n_y)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    return np.column_stack([xx.ravel(), yy.ravel()]).astype(np.float64)


def nonlinear_drift(states: np.ndarray, config: EnvConfig) -> np.ndarray:
    x = states[:, 0]
    y = states[:, 1]
    drift = np.column_stack(
        [
            np.sin(np.pi * y) + 0.35 * x * (1.0 - y**2),
            -0.85 * np.sin(np.pi * x) + 0.25 * y * (1.0 - x**2),
        ]
    )
    return config.drift_scale * drift


def transition_mean(states: np.ndarray, action_vec: np.ndarray, config: EnvConfig) -> np.ndarray:
    states_arr = np.asarray(states, dtype=np.float64).reshape(-1, 2)
    action = np.asarray(action_vec, dtype=np.float64).reshape(1, 2)
    interaction = np.column_stack(
        [
            action[0, 1] * states_arr[:, 0],
            -action[0, 0] * states_arr[:, 1],
        ]
    )
    mean = states_arr + nonlinear_drift(states_arr, config) + action
    mean = mean + config.interaction_scale * interaction
    return np.clip(mean, config.low if hasattr(config, "low") else -1.0, config.high if hasattr(config, "high") else 1.0)


def _radial_bump(points: np.ndarray, center: tuple[float, float], width: float) -> np.ndarray:
    center_arr = np.asarray(center, dtype=np.float64).reshape(1, 2)
    sq = np.sum((points - center_arr) ** 2, axis=1)
    return np.exp(-0.5 * sq / (width**2))


def reward_for_center(
    states: np.ndarray,
    action_vec: np.ndarray,
    *,
    primary: tuple[float, float],
    secondary: tuple[float, float],
    config: EnvConfig,
) -> np.ndarray:
    mean_next = transition_mean(states, action_vec, config)
    goal_bump = _radial_bump(mean_next, primary, config.goal_width)
    decoy_bump = _radial_bump(mean_next, secondary, config.decoy_width)
    barrier = _radial_bump(mean_next, (0.05, -0.10), config.barrier_width)
    action = np.asarray(action_vec, dtype=np.float64).reshape(2)
    interaction = 0.055 * (
        np.sin(2.0 * np.pi * states[:, 0]) * action[0]
        - np.cos(np.pi * states[:, 1]) * action[1]
    )
    cost = config.action_cost * float(action @ action) / max(config.action_scale**2, 1e-12)
    return 1.20 * goal_bump + 0.55 * decoy_bump - 0.22 * barrier - cost + interaction


def build_transition_kernel(states: np.ndarray, actions: np.ndarray, config: EnvConfig) -> np.ndarray:
    n_states = states.shape[0]
    n_actions = actions.shape[0]
    transition = np.zeros((n_states, n_actions, n_states), dtype=np.float64)
    noise_var = max(config.process_noise**2, 1e-10)
    for action_idx, action_vec in enumerate(actions):
        means = transition_mean(states, action_vec, config)
        for state_idx in range(n_states):
            diff = states - means[state_idx]
            logits = -0.5 * np.sum(diff * diff, axis=1) / noise_var
            logits -= float(np.max(logits))
            probs = np.exp(logits)
            probs /= np.maximum(probs.sum(), 1e-300)
            if config.teleport_prob > 0.0:
                probs = (1.0 - config.teleport_prob) * probs + config.teleport_prob / n_states
            transition[state_idx, action_idx] = probs / probs.sum()
    return transition


def build_reward_tables(states: np.ndarray, actions: np.ndarray, config: EnvConfig) -> tuple[np.ndarray, np.ndarray]:
    reward = np.zeros((states.shape[0], actions.shape[0]), dtype=np.float64)
    decoy_reward = np.zeros_like(reward)
    for action_idx, action_vec in enumerate(actions):
        reward[:, action_idx] = reward_for_center(
            states,
            action_vec,
            primary=config.goal,
            secondary=config.decoy,
            config=config,
        )
        decoy_reward[:, action_idx] = reward_for_center(
            states,
            action_vec,
            primary=config.decoy,
            secondary=config.goal,
            config=config,
        )
    return reward, decoy_reward


def build_grid_mdp(grid: GridConfig, config: EnvConfig) -> GridMDP:
    states = build_states(grid)
    actions = make_action_vectors(config.action_scale)
    transition = build_transition_kernel(states, actions, config)
    reward, decoy_reward = build_reward_tables(states, actions, config)
    return GridMDP(
        grid=grid,
        config=config,
        states=states,
        actions=actions,
        transition=transition,
        reward=reward,
        decoy_reward=decoy_reward,
    )
