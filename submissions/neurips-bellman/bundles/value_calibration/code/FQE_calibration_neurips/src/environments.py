from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .policies import SoftmaxPolicy


Array = np.ndarray


@dataclass
class NonlinearMDPConfig:
    name: str = "nonlinear_discrete_action"
    state_dim: int = 6
    n_actions: int = 3
    gamma: float = 0.95
    reward_noise: float = 0.2
    transition_noise: float = 0.25
    horizon: int = 80
    extrapolation_scale: float = 0.0
    reference_shift_scale: float = 0.0
    reward_shift_intercept: float = 0.0
    reward_shift_scale: float = 1.0
    misspecification: str = "none"
    seed: int = 0


class NonlinearMDP:
    """Nonlinear continuous-state MDP with discrete actions."""

    def __init__(self, config: NonlinearMDPConfig):
        self.config = config
        rng = np.random.default_rng(config.seed)
        d, a = config.state_dim, config.n_actions
        base = rng.normal(scale=0.22, size=(d, d))
        self.transition_matrix = 0.55 * np.eye(d) + base / np.sqrt(max(d, 1))
        self.action_effects = rng.normal(scale=0.45, size=(a, d))
        self.reward_linear = rng.normal(scale=0.65, size=d)
        self.reward_action = rng.normal(scale=0.35, size=a)
        self.reward_interaction = rng.normal(scale=0.25, size=(a, d))
        self.initial_mean = rng.normal(scale=0.4, size=d)
        self.well_specified_theta = rng.normal(scale=0.15, size=d + a + d)
        if config.misspecification == "well_specified_linear":
            self.transition_matrix = np.zeros_like(self.transition_matrix)
            self.action_effects = np.zeros_like(self.action_effects)
            self.initial_mean = np.zeros(d, dtype=float)
        if config.misspecification == "finite_iteration_scale":
            self.transition_matrix = 0.92 * np.eye(d)
            self.action_effects = rng.normal(scale=0.08, size=(a, d))
            self.reward_linear = np.zeros(d, dtype=float)
            self.reward_linear[0] = 0.8
            if d > 1:
                self.reward_linear[1] = 0.25
            self.reward_action = np.linspace(-0.15, 0.20, a)
            self.reward_interaction = rng.normal(scale=0.04, size=(a, d))
            self.initial_mean = np.zeros(d, dtype=float)

    @property
    def gamma(self) -> float:
        return float(self.config.gamma)

    @property
    def state_dim(self) -> int:
        return int(self.config.state_dim)

    @property
    def n_actions(self) -> int:
        return int(self.config.n_actions)

    def initial_states(self, n: int, rng: np.random.Generator, shifted: bool = False) -> Array:
        mean = self.initial_mean.copy()
        if shifted:
            shift = float(self.config.reference_shift_scale or self.config.extrapolation_scale)
            mean[0] += shift
        elif self.config.extrapolation_scale > 0:
            mean[0] += float(self.config.extrapolation_scale)
        return rng.normal(loc=mean, scale=1.0, size=(n, self.state_dim))

    def step_mean(self, states: Array, actions: Array) -> Array:
        x = np.asarray(states, dtype=float)
        a = np.asarray(actions, dtype=int)
        if self.config.misspecification == "well_specified_linear":
            return np.zeros_like(x)
        if self.config.misspecification == "finite_iteration_scale":
            return 0.92 * x + self.action_effects[a] + 0.03 * np.sin(x)
        nonlinear = 0.22 * np.sin(x) + 0.08 * np.cos(x[:, ::-1])
        return np.tanh(x @ self.transition_matrix.T + self.action_effects[a] + nonlinear)

    def step(self, states: Array, actions: Array, rng: np.random.Generator) -> Array:
        mean = self.step_mean(states, actions)
        return mean + self.config.transition_noise * rng.normal(size=mean.shape)

    def expected_reward(self, states: Array, actions: Array) -> Array:
        x = np.asarray(states, dtype=float)
        a = np.asarray(actions, dtype=int)
        if self.config.misspecification == "well_specified_linear":
            one_hot = np.zeros((x.shape[0], self.n_actions), dtype=float)
            one_hot[np.arange(x.shape[0]), a] = 1.0
            features = np.hstack([x, one_hot, x * (a[:, None] + 1.0) / max(self.n_actions, 1)])
            return features @ self.well_specified_theta
        if self.config.misspecification == "finite_iteration_scale":
            x0 = x[:, 0]
            x1 = x[:, 1 % self.state_dim]
            interactions = np.sum(x * self.reward_interaction[a], axis=1)
            return (
                0.8
                + 0.9 * np.tanh(x0)
                + 0.25 * x1
                + self.reward_action[a]
                + 0.15 * np.tanh(x0 + 0.5 * x1)
                + interactions
            ).astype(float)
        base = x @ self.reward_linear
        nonlinear = 0.65 * np.sin(x[:, 0]) + 0.35 * np.cos(x[:, 1 % self.state_dim])
        interactions = np.sum(x * self.reward_interaction[a], axis=1)
        penalty = -0.06 * np.sum(x**2, axis=1)
        reward = base + nonlinear + interactions + self.reward_action[a] + penalty
        if self.config.misspecification == "bellman_incomplete":
            x1 = x[:, 1 % self.state_dim]
            x2 = x[:, 2 % self.state_dim]
            action_scale = (a + 1.0) / max(self.n_actions, 1)
            reward = (
                0.35 * reward
                + 1.10 * np.sin(x[:, 0] * x1 + 0.4 * action_scale)
                + 0.75 * np.tanh(x2 - 0.5 * action_scale)
                - 0.10 * np.sum(np.abs(x[:, : min(4, self.state_dim)]), axis=1)
            )
        if self.config.misspecification == "monotone_distortion":
            reward = np.sign(reward) * np.sqrt(np.abs(reward) + 1.0)
        elif self.config.misspecification == "nonmonotone":
            reward = reward + 0.9 * np.sin(2.0 * reward)
        elif self.config.misspecification == "affine":
            reward = 1.25 + 0.7 * reward
        reward = float(self.config.reward_shift_intercept) + float(self.config.reward_shift_scale) * reward
        return reward.astype(float)

    def reward(self, states: Array, actions: Array, rng: np.random.Generator) -> Array:
        mean = self.expected_reward(states, actions)
        if self.config.reward_noise <= 0:
            return mean
        return mean + self.config.reward_noise * rng.normal(size=mean.shape)


def monte_carlo_oracle_value(
    env: NonlinearMDP,
    target_policy: SoftmaxPolicy,
    n_rollouts: int,
    seed: int,
    initial_states: Array | None = None,
) -> float:
    rng = np.random.default_rng(seed)
    states = env.initial_states(n_rollouts, rng) if initial_states is None else np.asarray(initial_states, dtype=float)
    total = np.zeros(states.shape[0], dtype=float)
    discount = 1.0
    for _ in range(env.config.horizon):
        actions = target_policy.sample(states, rng)
        total += discount * env.expected_reward(states, actions)
        states = env.step(states, actions, rng)
        discount *= env.gamma
    return float(np.mean(total))


def monte_carlo_q_values(
    env: NonlinearMDP,
    target_policy: SoftmaxPolicy,
    states: Array,
    actions: Array,
    n_rollouts_per_state: int,
    seed: int,
) -> Array:
    """Approximate the true target-policy Q-function at fixed state-actions.

    The first reward is the environment's noiseless conditional mean reward
    r_0(s, a). Future rewards are also evaluated with conditional mean rewards
    while transition noise and target-policy actions are Monte Carlo averaged.
    This keeps the diagnostic focused on function approximation error rather
    than reward observation noise.
    """
    x = np.asarray(states, dtype=float)
    a = np.asarray(actions, dtype=int)
    n = x.shape[0]
    reps = max(int(n_rollouts_per_state), 1)
    rng = np.random.default_rng(seed)

    rollout_states = np.repeat(x, reps, axis=0)
    rollout_actions = np.repeat(a, reps)
    total = env.expected_reward(rollout_states, rollout_actions)
    rollout_states = env.step(rollout_states, rollout_actions, rng)
    discount = env.gamma
    for _ in range(max(int(env.config.horizon) - 1, 0)):
        rollout_actions = target_policy.sample(rollout_states, rng)
        total += discount * env.expected_reward(rollout_states, rollout_actions)
        rollout_states = env.step(rollout_states, rollout_actions, rng)
        discount *= env.gamma
    return total.reshape(n, reps).mean(axis=1)


def monte_carlo_v_values(
    env: NonlinearMDP,
    target_policy: SoftmaxPolicy,
    states: Array,
    n_rollouts_per_state: int,
    seed: int,
) -> Array:
    probs = target_policy.action_probabilities(np.asarray(states, dtype=float))
    values = np.zeros(probs.shape[0], dtype=float)
    for action in range(env.n_actions):
        actions = np.full(probs.shape[0], action, dtype=int)
        values += probs[:, action] * monte_carlo_q_values(
            env, target_policy, states, actions, n_rollouts_per_state, seed + 7919 + action
        )
    return values


def monte_carlo_v_values_direct(
    env: NonlinearMDP,
    target_policy: SoftmaxPolicy,
    states: Array,
    n_rollouts_per_state: int,
    seed: int,
) -> Array:
    """Approximate V^pi(s) by direct target-policy rollouts from fixed states.

    This is lower cost than enumerating all initial actions via Q-values and is
    intended for large near-population diagnostic sets. Rewards use conditional
    means, matching the other oracle diagnostics.
    """
    x = np.asarray(states, dtype=float)
    n = x.shape[0]
    reps = max(int(n_rollouts_per_state), 1)
    rng = np.random.default_rng(seed)
    rollout_states = np.repeat(x, reps, axis=0)
    total = np.zeros(rollout_states.shape[0], dtype=float)
    discount = 1.0
    for _ in range(env.config.horizon):
        actions = target_policy.sample(rollout_states, rng)
        total += discount * env.expected_reward(rollout_states, actions)
        rollout_states = env.step(rollout_states, actions, rng)
        discount *= env.gamma
    return total.reshape(n, reps).mean(axis=1)


def rollout_states(
    env: NonlinearMDP,
    policy: SoftmaxPolicy,
    n_rollouts: int,
    seed: int,
) -> Array:
    rng = np.random.default_rng(seed)
    states = env.initial_states(n_rollouts, rng)
    collected = []
    for _ in range(env.config.horizon):
        collected.append(states.copy())
        actions = policy.sample(states, rng)
        states = env.step(states, actions, rng)
    return np.vstack(collected)
