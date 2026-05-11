"""DeepPQR-style simulation code with generalized normalization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List

import numpy as np

from utils import set_random_seed, softmax


PolicyFunction = Callable[[np.ndarray], np.ndarray]
NormalizerFunction = Callable[[np.ndarray], np.ndarray]


@dataclass
class SimulationConfig:
    """Configuration for the synthetic infinite-horizon simulator."""

    state_dim: int = 5
    n_actions: int = 5
    horizon: int = 20
    gamma: float = 0.9
    behavior_temperature: float = 1.0
    behavior_logit_scale: float = 1.0
    anchor_logit_shift: float = 0.0
    process_noise: float = 0.35
    reward_noise: float = 0.05
    initial_state_scale: float = 1.0
    planner_num_states: int = 2048
    planner_state_scale: float = 1.5
    planner_ridge: float = 1e-3
    planner_iters: int = 60
    seed: int = 0


@dataclass
class LinearGaussianDynamics:
    """Action-specific linear-Gaussian transition dynamics."""

    transition_matrices: np.ndarray
    transition_offsets: np.ndarray
    reward_state_weights: np.ndarray
    reward_action_bias: np.ndarray
    policy_state_weights: np.ndarray
    policy_action_bias: np.ndarray
    time_action_bias: np.ndarray
    omega: np.ndarray


@dataclass
class ExactSoftQPlanner:
    """Exact soft-Q object used to define a policy-consistent anchor-action MDP."""

    policy_weights: np.ndarray
    policy_bias: np.ndarray
    alpha: float = 1.0
    gamma: float = 0.9
    anchor_action: int = 0

    def _scaled_states(self, states: np.ndarray) -> np.ndarray:
        states = np.asarray(states, dtype=float)
        return states / np.sqrt(max(states.shape[1], 1))

    def score_matrix(self, states: np.ndarray) -> np.ndarray:
        scaled = self._scaled_states(states)
        raw = scaled @ self.policy_weights.T + self.policy_bias[None, :]
        curved = raw + 0.35 * np.sin(1.5 * raw) + 0.20 * ((scaled**2) @ np.abs(self.policy_weights).T)
        return 0.8 * np.tanh(curved)

    def predict_proba(self, states: np.ndarray) -> np.ndarray:
        scores = self.score_matrix(states)
        return softmax(scores / max(self.alpha, 1e-8), axis=1)

    def predict_anchor_q(self, states: np.ndarray) -> np.ndarray:
        pi_anchor = np.clip(self.predict_proba(states)[:, self.anchor_action], 1e-8, 1.0)
        scale = self.gamma * self.alpha / max(1.0 - self.gamma, 1e-8)
        return -scale * np.log(pi_anchor)

    def predict_q(self, states: np.ndarray, time_step: int | None = None) -> np.ndarray:
        del time_step
        pi = np.clip(self.predict_proba(states), 1e-8, 1.0)
        anchor_q = self.predict_anchor_q(states)
        return self.alpha * (np.log(pi) - np.log(pi[:, [self.anchor_action]])) + anchor_q[:, None]


def make_anchor_mu(anchor_action: int, n_actions: int) -> PolicyFunction:
    """Return the anchor-action policy used in DeepPQR as a special case."""

    def mu(states: np.ndarray) -> np.ndarray:
        probs = np.zeros((states.shape[0], n_actions), dtype=float)
        probs[:, anchor_action] = 1.0
        return probs

    return mu


def make_uniform_mu(n_actions: int) -> PolicyFunction:
    """Return a uniform normalization policy."""

    def mu(states: np.ndarray) -> np.ndarray:
        return np.full((states.shape[0], n_actions), 1.0 / n_actions, dtype=float)

    return mu


def make_constant_g(value: float = 0.0) -> NormalizerFunction:
    """Return the state-only normalizer g(s) = value."""

    def g(states: np.ndarray) -> np.ndarray:
        return np.full(states.shape[0], value, dtype=float)

    return g


def make_zero_g() -> NormalizerFunction:
    """Return the canonical anchor-value choice g(s) = 0."""

    def g(states: np.ndarray) -> np.ndarray:
        return np.zeros(states.shape[0], dtype=float)

    return g


def make_anchor_g(anchor_action: int, n_actions: int) -> NormalizerFunction:
    """Backward-compatible alias for the common anchor-action choice g(s) = 0."""
    del anchor_action, n_actions
    return make_zero_g()


def sample_simulation_parameters(config: SimulationConfig) -> LinearGaussianDynamics:
    """Sample a reproducible family of transition, reward, and policy parameters."""
    rng = set_random_seed(config.seed)
    transition_matrices = np.empty((config.n_actions, config.state_dim, config.state_dim), dtype=float)
    transition_offsets = rng.normal(scale=0.25, size=(config.n_actions, config.state_dim))
    reward_state_weights = rng.normal(scale=0.2, size=(config.n_actions, config.state_dim))
    reward_action_bias = np.linspace(-0.15, 0.15, config.n_actions)
    policy_state_weights = rng.normal(scale=0.7, size=(config.n_actions, config.state_dim))
    policy_action_bias = np.linspace(0.5, -0.3, config.n_actions)
    time_action_bias = rng.normal(scale=0.15, size=(config.horizon, config.n_actions))
    for action in range(config.n_actions):
        raw = rng.normal(scale=0.15, size=(config.state_dim, config.state_dim))
        transition_matrices[action] = 0.65 * np.eye(config.state_dim) + raw
    return LinearGaussianDynamics(
        transition_matrices=transition_matrices,
        transition_offsets=transition_offsets,
        reward_state_weights=reward_state_weights,
        reward_action_bias=reward_action_bias,
        policy_state_weights=policy_state_weights,
        policy_action_bias=policy_action_bias,
        time_action_bias=time_action_bias,
        omega=rng.uniform(0.5, 1.5, size=config.state_dim + 1),
    )


def base_reward_matrix(
    states: np.ndarray,
    time_step: int,
    params: LinearGaussianDynamics,
) -> np.ndarray:
    """Construct a fallback DeepPQR-style reward matrix."""
    states = np.asarray(states, dtype=float)
    del time_step
    action_values = np.arange(params.reward_action_bias.shape[0], dtype=float)
    scaled_states = states / max(states.shape[1], 1)
    omega_state = params.omega[:-1]
    omega_action = params.omega[-1]
    logits = scaled_states @ omega_state[:, None] + (action_values[None, :] / 4.0) * omega_action
    numerator = action_values[None, :] * np.tanh(logits)
    denominator = 4.0 * np.sum(params.omega)
    return numerator / max(denominator, 1e-8)


def normalize_reward_matrix(
    states: np.ndarray,
    reward_matrix: np.ndarray,
    mu: PolicyFunction,
    g: NormalizerFunction,
) -> np.ndarray:
    """Shift rewards so the paper's normalization `sum_a mu(a|s) r(s,a) = g(s)` holds."""
    mu_probs = np.asarray(mu(states), dtype=float)
    g_values = np.asarray(g(states), dtype=float).reshape(-1, 1)
    mu_average = np.sum(mu_probs * reward_matrix, axis=1, keepdims=True)
    return reward_matrix - mu_average + g_values


def behavior_policy_matrix(
    states: np.ndarray,
    time_step: int,
    params: LinearGaussianDynamics,
    temperature: float,
    reward_matrix: np.ndarray | None = None,
    planner: ExactSoftQPlanner | None = None,
) -> np.ndarray:
    """Softmax behavior policy used to generate trajectories.

    By default we align the logged policy with the normalized reward signal,
    which is the relevant object for the anchor-action PQR comparisons.
    """
    if planner is not None:
        return planner.predict_proba(states)
    if reward_matrix is not None:
        logits = np.asarray(reward_matrix, dtype=float)
        return softmax(logits / max(temperature, 1e-6), axis=1)

    logits = states @ params.policy_state_weights.T
    logits = logits + params.policy_action_bias[None, :] + params.time_action_bias[time_step][None, :]
    return softmax(logits / max(temperature, 1e-6), axis=1)


def transition_step(
    states: np.ndarray,
    actions: np.ndarray,
    params: LinearGaussianDynamics,
    config: SimulationConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample the DeepPQR synthetic transition."""
    states = np.asarray(states, dtype=float)
    actions = np.asarray(actions, dtype=int).reshape(-1)
    state_bound = float(config.state_dim)
    action_offsets = np.concatenate(
        [
            np.array([0.0], dtype=float),
            np.linspace(-0.35, 0.35, max(config.n_actions - 1, 1), dtype=float),
        ]
    )[: config.n_actions]
    shift = np.repeat(action_offsets[actions][:, None], config.state_dim, axis=1)
    proposed = states + shift
    in_bounds = np.all(np.abs(proposed) <= state_bound, axis=1)
    next_states = proposed.copy()
    if np.any(~in_bounds):
        next_states[~in_bounds] = rng.uniform(
            low=-state_bound,
            high=state_bound,
            size=(np.sum(~in_bounds), config.state_dim),
        )
    return next_states


def deterministic_transition_mean(
    states: np.ndarray,
    actions: np.ndarray,
    params: LinearGaussianDynamics,
) -> np.ndarray:
    """Return the conditional mean transition used for AIRL rollouts."""
    states = np.asarray(states, dtype=float)
    actions = np.asarray(actions, dtype=int).reshape(-1)
    state_bound = float(states.shape[1])
    n_actions = params.transition_matrices.shape[0]
    action_offsets = np.concatenate(
        [
            np.array([0.0], dtype=float),
            np.linspace(-0.35, 0.35, max(n_actions - 1, 1), dtype=float),
        ]
    )[:n_actions]
    shift = np.repeat(action_offsets[actions][:, None], states.shape[1], axis=1)
    proposed = states + shift
    in_bounds = np.all(np.abs(proposed) <= state_bound, axis=1)
    next_states = proposed.copy()
    next_states[~in_bounds] = 0.0
    return next_states


def make_exact_softq_planner(config: SimulationConfig, params: LinearGaussianDynamics) -> ExactSoftQPlanner:
    """Construct an exact soft-Q object for the anchor-action experiment."""
    policy_bias = params.policy_action_bias.copy()
    policy_bias[0] += config.anchor_logit_shift
    return ExactSoftQPlanner(
        policy_weights=params.policy_state_weights,
        policy_bias=policy_bias,
        alpha=config.behavior_temperature / max(config.behavior_logit_scale, 1e-8),
        gamma=config.gamma,
        anchor_action=0,
    )


def exact_anchor_pqr_reward_matrix(
    states: np.ndarray,
    planner: ExactSoftQPlanner,
    params: LinearGaussianDynamics,
    config: SimulationConfig,
) -> np.ndarray:
    """Construct rewards that exactly satisfy the DeepPQR Bellman identity."""
    states = np.asarray(states, dtype=float)
    q_matrix = planner.predict_q(states)
    reward_matrix = np.empty_like(q_matrix)
    for action in range(config.n_actions):
        action_vec = np.full(states.shape[0], action, dtype=int)
        next_states = deterministic_transition_mean(states, action_vec, params)
        pi_next = np.clip(planner.predict_proba(next_states), 1e-8, 1.0)
        anchor_q_next = planner.predict_anchor_q(next_states)
        continuation = -planner.alpha * np.log(pi_next[:, planner.anchor_action]) + anchor_q_next
        reward_matrix[:, action] = q_matrix[:, action] - config.gamma * continuation
    return reward_matrix


def generate_deeppqr_style_data(
    n_trajectories: int,
    config: SimulationConfig | None = None,
    mu: PolicyFunction | None = None,
    g: NormalizerFunction | None = None,
    simulation_parameters: LinearGaussianDynamics | None = None,
) -> Dict[str, np.ndarray]:
    """Generate trajectories under a generalized normalization condition.

    The reward is built by first constructing an unrestricted action-value-like
    signal and then shifting it so that

        sum_a mu(a | s) r(s, a) = g(s)

    for every visited state. Choosing `mu` as a point mass on an anchor action
    and `g(s) = 0` recovers the anchor-action normalization used by DeepPQR.
    """
    config = config or SimulationConfig()
    params = simulation_parameters or sample_simulation_parameters(config)
    rng = set_random_seed(config.seed + 1)
    mu = mu or make_uniform_mu(config.n_actions)
    g = g or make_constant_g(0.0)
    planner = make_exact_softq_planner(config=config, params=params)

    states_list = []
    actions_list = []
    rewards_list = []
    next_states_list = []
    dones_list = []
    times_list = []
    traj_ids = []
    behavior_prob_list = []
    normalized_reward_matrix_list = []
    g_value_list = []

    states = rng.normal(scale=config.initial_state_scale, size=(n_trajectories, config.state_dim))
    for time_step in range(config.horizon):
        normalized_reward_matrix = exact_anchor_pqr_reward_matrix(
            states=states,
            planner=planner,
            params=params,
            config=config,
        )
        policy_probs = behavior_policy_matrix(
            states=states,
            time_step=time_step,
            params=params,
            temperature=config.behavior_temperature,
            reward_matrix=normalized_reward_matrix,
            planner=planner,
        )
        actions = np.array(
            [rng.choice(config.n_actions, p=policy_probs[i]) for i in range(n_trajectories)],
            dtype=int,
        )
        reward_noise = rng.normal(scale=config.reward_noise, size=n_trajectories)
        rewards = normalized_reward_matrix[np.arange(n_trajectories), actions] + reward_noise
        next_states = deterministic_transition_mean(states, actions, params)
        dones = np.zeros(n_trajectories, dtype=int)

        states_list.append(states.copy())
        actions_list.append(actions)
        rewards_list.append(rewards)
        next_states_list.append(next_states.copy())
        dones_list.append(dones)
        times_list.append(np.full(n_trajectories, time_step, dtype=int))
        traj_ids.append(np.arange(n_trajectories, dtype=int))
        behavior_prob_list.append(policy_probs[np.arange(n_trajectories), actions])
        normalized_reward_matrix_list.append(normalized_reward_matrix)
        g_value_list.append(np.asarray(g(states), dtype=float))

        states = next_states

    dataset = {
        "states": np.concatenate(states_list, axis=0),
        "actions": np.concatenate(actions_list, axis=0),
        "rewards": np.concatenate(rewards_list, axis=0),
        "next_states": np.concatenate(next_states_list, axis=0),
        "dones": np.concatenate(dones_list, axis=0),
        "time_index": np.concatenate(times_list, axis=0),
        "trajectory_id": np.concatenate(traj_ids, axis=0),
        "behavior_action_prob": np.concatenate(behavior_prob_list, axis=0),
        "normalized_reward_matrix": np.concatenate(normalized_reward_matrix_list, axis=0),
        "g_values": np.concatenate(g_value_list, axis=0),
        "config": config,
        "simulation_parameters": params,
        "mu": mu,
        "g": g,
        "planner": planner,
    }
    return dataset
