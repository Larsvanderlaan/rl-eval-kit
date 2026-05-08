from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from .fqe import FQEConfig, fit_weighted_fqe_nn, predict_q_values
from .fqe_linear import LinearFQEConfig, fit_weighted_linear_fqe, predict_linear_q_values
from .neural_rkhs_weights import KernelConfig, NeuralRKHSWeightsConfig, estimate_ratio_neural_rkhs
from .ratio_estimation import (
    NeuralRatioConfig,
    estimate_ratio_closed_form_linear,
    estimate_ratio_saddle_neural,
    positive_linear_ratio_weights,
)
from .utils import (
    DiscreteMDP,
    TransitionBatch,
    exact_ratio_against_behavior,
    evaluate_policy_tabular,
    sample_actions,
    sample_next_states,
    set_random_seed,
    stabilize_weights,
    stationary_state_action_distribution,
)


@dataclass
class LatentGarnetConfig:
    """Configuration for a realistic finite-state benchmark with latent geometry."""

    n_states: int = 100
    n_actions: int = 4
    latent_dim: int = 3
    branching_factor: int = 5
    transition_bandwidth: float = 0.35
    teleport_mass: float = 0.03
    observation_mode: str = "rich"
    compact_obs_dim: int = 8
    obs_rff_dim: int = 16
    basic_linear_raw_dims: int = 4
    flexible_linear_rff_dim: int = 16
    flexible_linear_bandwidth: float | str | None = "median"
    flexible_linear_include_raw: bool = True
    linear_q_state_rff_dim: int = 2
    linear_q_raw_dims: int = 1
    reward_scale: float = 1.0
    goal_reward_scale: float = 6.0
    goal_bandwidth: float = 0.35
    action_goal_bonus: float = 2.0
    policy_goal_scale: float = 8.0
    start_bandwidth: float = 0.45
    policy_temperature: float = 0.8
    behavior_temperature: float = 0.8
    behavior_coverage: float = 0.3
    dataset_size: int = 10_000
    burn_in: int = 2_000
    data_mode: str = "mixed"
    n_trajectories: int = 200
    trajectory_horizon: int | None = None
    iid_fraction: float = 0.5
    seed: int = 0


@dataclass
class LatentGarnetBenchmark:
    """Finite exact MDP with continuous-like latent structure and nonlinear observations."""

    config: LatentGarnetConfig
    mdp: DiscreteMDP
    latent_states: np.ndarray
    observed_state_features: np.ndarray
    linear_basic_state_features: np.ndarray
    flexible_linear_state_features: np.ndarray
    linear_q_state_features: np.ndarray
    initial_state_distribution: np.ndarray
    target_policy_logits: np.ndarray
    distractor_policy_logits: np.ndarray

    def featurize_state_actions(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        feature_set: str = "raw",
    ) -> np.ndarray:
        """Continuous-style observed features for linear/neural ratio models."""

        states = np.asarray(states, dtype=np.int64).reshape(-1)
        actions = np.asarray(actions, dtype=np.int64).reshape(-1)
        if feature_set == "raw":
            obs = self.observed_state_features[states]
        elif feature_set == "linear_basic":
            obs = self.linear_basic_state_features[states]
        elif feature_set == "flexible_linear":
            obs = self.flexible_linear_state_features[states]
        elif feature_set == "linear_q":
            obs = self.linear_q_state_features[states]
        else:
            raise ValueError(f"Unknown feature_set '{feature_set}'.")
        one_hot_actions = np.eye(self.config.n_actions, dtype=np.float64)[actions]
        tensor_features = np.einsum("nd,na->nda", obs, one_hot_actions).reshape(len(states), -1)
        return tensor_features

    def exact_state_action_ratio(self, gamma_ratio: float) -> np.ndarray:
        return exact_ratio_against_behavior(self.mdp, gamma_ratio=gamma_ratio)

    def stationary_overlap_metrics(self) -> dict[str, float]:
        mu_pi = stationary_state_action_distribution(self.mdp, self.mdp.target_policy)
        nu_b = stationary_state_action_distribution(self.mdp, self.mdp.behavior_policy)
        support = mu_pi > 1e-12
        ratio = nu_b[support] / mu_pi[support]
        coverage_min = float(np.min(ratio))
        coverage_mean = float(np.sum(np.minimum(mu_pi, nu_b)))
        chi2_ratio = float(np.sum(((mu_pi - nu_b) ** 2) / np.maximum(nu_b, 1e-12)))
        return {
            "requested_behavior_coverage": float(self.config.behavior_coverage),
            "realized_min_density_ratio_nub_over_target": coverage_min,
            "realized_overlap_mass": coverage_mean,
            "chi2_target_vs_behavior": chi2_ratio,
        }


def _softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    z = logits / max(temperature, 1e-8)
    z = z - np.max(z, axis=1, keepdims=True)
    probs = np.exp(z)
    probs /= probs.sum(axis=1, keepdims=True)
    return probs


def _random_fourier_features(
    latent_states: np.ndarray,
    out_dim: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if out_dim <= 0:
        return np.empty((latent_states.shape[0], 0), dtype=np.float64)
    W = rng.normal(scale=1.0, size=(latent_states.shape[1], out_dim))
    b = rng.uniform(0.0, 2.0 * np.pi, size=out_dim)
    return np.cos(latent_states @ W + b)


def _build_observed_state_features(
    latent: np.ndarray,
    config: LatentGarnetConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    mode = config.observation_mode.lower()
    if mode == "rich":
        base_feats = [latent, latent**2, _random_fourier_features(latent, config.obs_rff_dim, rng)]
        return np.concatenate(base_feats, axis=1)
    if mode == "compact_nonlinear":
        w1 = rng.normal(size=(config.latent_dim, config.compact_obs_dim))
        b1 = rng.uniform(-np.pi, np.pi, size=config.compact_obs_dim)
        w2 = rng.normal(size=(config.latent_dim, config.compact_obs_dim))
        b2 = rng.uniform(-1.0, 1.0, size=config.compact_obs_dim)
        trig = np.sin(latent @ w1 + b1)
        tanh = np.tanh(latent @ w2 + b2)
        return np.concatenate([trig, tanh], axis=1)
    if mode == "raw":
        return latent.copy()
    raise ValueError(f"Unsupported observation_mode '{config.observation_mode}'.")


def _build_flexible_linear_state_features(
    observed_state_features: np.ndarray,
    config: LatentGarnetConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    x = np.asarray(observed_state_features, dtype=np.float64)
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale = np.where(scale < 1e-8, 1.0, scale)
    x_std = (x - mean) / scale

    bandwidth = config.flexible_linear_bandwidth
    if bandwidth is None or bandwidth == "median":
        if x_std.shape[0] > 512:
            idx = rng.choice(x_std.shape[0], size=512, replace=False)
            x_sub = x_std[idx]
        else:
            x_sub = x_std
        diffs = x_sub[:, None, :] - x_sub[None, :, :]
        dist = np.sqrt(np.sum(diffs**2, axis=-1))
        tri = dist[np.triu_indices(dist.shape[0], k=1)]
        tri = tri[tri > 0]
        bandwidth = float(np.median(tri)) if tri.size > 0 else 1.0
    bandwidth = max(float(bandwidth), 1e-8)

    W = rng.normal(scale=1.0 / bandwidth, size=(x_std.shape[1], config.flexible_linear_rff_dim))
    b = rng.uniform(0.0, 2.0 * np.pi, size=config.flexible_linear_rff_dim)
    rff = np.sqrt(2.0 / max(config.flexible_linear_rff_dim, 1)) * np.cos(x_std @ W + b)

    if config.flexible_linear_include_raw:
        return np.concatenate([x_std, rff], axis=1)
    return rff


def _build_linear_basic_state_features(
    observed_state_features: np.ndarray,
    config: LatentGarnetConfig,
) -> np.ndarray:
    """
    Small, simulator-agnostic default basis for the basic linear weight model.

    The intent is to mimic a reasonable prespecified baseline: a bias term, a
    few standardized observed coordinates, and a simple norm summary. This keeps
    the model heterogeneous across state-action pairs while remaining clearly
    misspecified in the nonlinear benchmark regimes.
    """

    x = np.asarray(observed_state_features, dtype=np.float64)
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale = np.where(scale < 1e-8, 1.0, scale)
    x_std = (x - mean) / scale

    keep = min(config.basic_linear_raw_dims, x_std.shape[1])
    raw_part = x_std[:, :keep]
    norm_part = np.linalg.norm(x_std, axis=1, keepdims=True) / np.sqrt(max(x_std.shape[1], 1))
    bias = np.ones((x_std.shape[0], 1), dtype=np.float64)
    return np.concatenate([bias, raw_part, norm_part], axis=1)


def _build_linear_q_state_features(
    observed_state_features: np.ndarray,
    config: LatentGarnetConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Very low-dimensional basis for linear FQE.

    This basis is intentionally compressed and therefore generally not Bellman
    complete under the nonlinear latent dynamics. It is meant to reflect a
    prespecified linear approximation that captures only coarse geometry:
    a bias term, a tiny number of raw coordinates, a radial summary, and a very
    small number of low-frequency random features.
    """

    x = np.asarray(observed_state_features, dtype=np.float64)
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale = np.where(scale < 1e-8, 1.0, scale)
    x_std = (x - mean) / scale

    keep = min(config.linear_q_raw_dims, x_std.shape[1])
    raw_part = x_std[:, :keep]
    radial_part = np.linalg.norm(x_std, axis=1, keepdims=True) / np.sqrt(max(x_std.shape[1], 1))
    if config.linear_q_state_rff_dim > 0:
        W = rng.normal(scale=0.5, size=(x_std.shape[1], config.linear_q_state_rff_dim))
        b = rng.uniform(0.0, 2.0 * np.pi, size=config.linear_q_state_rff_dim)
        rff = np.sqrt(2.0 / config.linear_q_state_rff_dim) * np.cos(x_std @ W + b)
    else:
        rff = np.empty((x_std.shape[0], 0), dtype=np.float64)
    bias = np.ones((x_std.shape[0], 1), dtype=np.float64)
    return np.concatenate([bias, raw_part, radial_part, rff], axis=1)


def build_latent_garnet_benchmark(config: LatentGarnetConfig | None = None) -> LatentGarnetBenchmark:
    """
    Build a finite-state benchmark that mimics continuous-state geometry.

    The underlying MDP is exactly known and remains tabular, but transitions and
    rewards are driven by a low-dimensional latent state embedding. Observed
    state-action features are nonlinear functions of the latent variables, so
    function approximation behaves more like a continuous-state problem.
    """

    if config is None:
        config = LatentGarnetConfig()

    rng = set_random_seed(config.seed)
    latent = rng.uniform(low=-1.0, high=1.0, size=(config.n_states, config.latent_dim))

    action_drifts = rng.normal(scale=0.4, size=(config.n_actions, config.latent_dim))
    transition_prob = np.zeros((config.n_states, config.n_actions, config.n_states), dtype=np.float64)

    for s in range(config.n_states):
        for a in range(config.n_actions):
            preferred = latent[s] + action_drifts[a] + 0.1 * rng.normal(size=config.latent_dim)
            dists = np.sum((latent - preferred[None, :]) ** 2, axis=1)
            support = np.argpartition(dists, config.branching_factor)[: config.branching_factor]
            local_logits = -dists[support] / (2.0 * config.transition_bandwidth**2)
            local_logits += 0.2 * rng.normal(size=support.shape[0])
            probs = np.exp(local_logits - np.max(local_logits))
            probs /= probs.sum()
            transition_prob[s, a, support] = probs
            transition_prob[s, a] = (1.0 - config.teleport_mass) * transition_prob[s, a] + config.teleport_mass / config.n_states

    observed_state_features = _build_observed_state_features(latent, config, rng)
    linear_basic_state_features = _build_linear_basic_state_features(observed_state_features, config)
    flexible_linear_state_features = _build_flexible_linear_state_features(observed_state_features, config, rng)

    linear_q_state_features = _build_linear_q_state_features(observed_state_features, config, rng)

    goal_direction = rng.normal(size=config.latent_dim)
    goal_direction /= max(np.linalg.norm(goal_direction), 1e-8)
    goal_center = 0.8 * goal_direction
    start_center = -goal_center

    dist_to_goal_sq = np.sum((latent - goal_center[None, :]) ** 2, axis=1)
    goal_state_reward = config.goal_reward_scale * np.exp(
        -dist_to_goal_sq / (2.0 * config.goal_bandwidth**2)
    )

    next_goal_dist_sq = np.sum(
        (latent[:, None, :] + action_drifts[None, :, :] - goal_center[None, None, :]) ** 2,
        axis=2,
    )
    action_goal_bonus = config.action_goal_bonus * np.exp(
        -next_goal_dist_sq / (2.0 * config.goal_bandwidth**2)
    )
    reward_proj = rng.normal(size=(observed_state_features.shape[1], config.n_actions))
    reward_nonlinear = 0.25 * np.sin(observed_state_features @ reward_proj)
    rewards = (
        config.reward_scale * reward_nonlinear
        + goal_state_reward[:, None]
        + action_goal_bonus
    )

    policy_goal_scores = -next_goal_dist_sq / max(2.0 * config.goal_bandwidth**2, 1e-8)
    policy_proj = rng.normal(size=(observed_state_features.shape[1], config.n_actions))
    policy_nonlinear = 0.5 * np.tanh(observed_state_features @ policy_proj)
    policy_logits = config.policy_goal_scale * policy_goal_scores + policy_nonlinear

    distractor_proj = rng.normal(size=(observed_state_features.shape[1], config.n_actions))
    distractor_nonlinear = 0.5 * np.tanh(observed_state_features @ distractor_proj)
    distractor_logits = -config.policy_goal_scale * policy_goal_scores + distractor_nonlinear

    target_policy = _softmax(policy_logits, temperature=config.policy_temperature)
    distractor_policy = _softmax(distractor_logits, temperature=config.behavior_temperature)
    behavior_policy = config.behavior_coverage * target_policy + (1.0 - config.behavior_coverage) * distractor_policy
    behavior_policy /= behavior_policy.sum(axis=1, keepdims=True)

    start_dist_logits = -np.sum((latent - start_center[None, :]) ** 2, axis=1) / max(
        2.0 * config.start_bandwidth**2,
        1e-8,
    )
    start_dist = np.exp(start_dist_logits - np.max(start_dist_logits))
    initial_state_distribution = start_dist / np.sum(start_dist)

    mdp = DiscreteMDP(
        transition_prob=transition_prob,
        rewards=rewards,
        target_policy=target_policy,
        behavior_policy=behavior_policy,
    )
    return LatentGarnetBenchmark(
        config=config,
        mdp=mdp,
        latent_states=latent,
        observed_state_features=observed_state_features,
        linear_basic_state_features=linear_basic_state_features,
        flexible_linear_state_features=flexible_linear_state_features,
        linear_q_state_features=linear_q_state_features,
        initial_state_distribution=initial_state_distribution,
        target_policy_logits=policy_logits,
        distractor_policy_logits=distractor_logits,
    )


def _simulate_long_behavior_trajectory(
    benchmark: LatentGarnetBenchmark,
    dataset_size: int,
    rng: np.random.Generator,
    burn_in: int,
) -> TransitionBatch:
    """Generate one long behavior-policy trajectory after a burn-in period."""

    n = int(dataset_size)
    burn = int(burn_in)
    state = int(rng.integers(benchmark.config.n_states))

    states = np.zeros(n, dtype=np.int64)
    actions = np.zeros(n, dtype=np.int64)
    rewards = np.zeros(n, dtype=np.float64)
    next_states = np.zeros(n, dtype=np.int64)

    for t in range(n + burn):
        action = int(sample_actions(benchmark.mdp.behavior_policy, np.array([state]), rng)[0])
        next_state = int(sample_next_states(benchmark.mdp.transition_prob, np.array([state]), np.array([action]), rng)[0])
        reward = float(benchmark.mdp.rewards[state, action])
        if t >= burn:
            idx = t - burn
            states[idx] = state
            actions[idx] = action
            rewards[idx] = reward
            next_states[idx] = next_state
        state = next_state

    next_actions = sample_actions(benchmark.mdp.target_policy, next_states, rng)
    return TransitionBatch(
        states=states,
        actions=actions,
        rewards=rewards,
        next_states=next_states,
        next_actions=next_actions,
    )


def _simulate_multi_trajectory_behavior(
    benchmark: LatentGarnetBenchmark,
    dataset_size: int,
    rng: np.random.Generator,
    n_trajectories: int,
    trajectory_horizon: int | None,
) -> TransitionBatch:
    """Generate many shorter behavior trajectories from the benchmark start distribution."""

    n = int(dataset_size)
    n_traj = max(int(n_trajectories), 1)
    horizon = trajectory_horizon
    if horizon is None:
        horizon = int(np.ceil(n / n_traj))
    horizon = max(int(horizon), 1)

    states_list: list[int] = []
    actions_list: list[int] = []
    rewards_list: list[float] = []
    next_states_list: list[int] = []

    for _ in range(n_traj):
        if len(states_list) >= n:
            break
        state = int(rng.choice(benchmark.config.n_states, p=benchmark.initial_state_distribution))
        for _ in range(horizon):
            action = int(sample_actions(benchmark.mdp.behavior_policy, np.array([state]), rng)[0])
            next_state = int(sample_next_states(benchmark.mdp.transition_prob, np.array([state]), np.array([action]), rng)[0])
            reward = float(benchmark.mdp.rewards[state, action])
            states_list.append(state)
            actions_list.append(action)
            rewards_list.append(reward)
            next_states_list.append(next_state)
            state = next_state
            if len(states_list) >= n:
                break

    states = np.asarray(states_list[:n], dtype=np.int64)
    actions = np.asarray(actions_list[:n], dtype=np.int64)
    rewards = np.asarray(rewards_list[:n], dtype=np.float64)
    next_states = np.asarray(next_states_list[:n], dtype=np.int64)
    next_actions = sample_actions(benchmark.mdp.target_policy, next_states, rng)
    return TransitionBatch(
        states=states,
        actions=actions,
        rewards=rewards,
        next_states=next_states,
        next_actions=next_actions,
    )


def _simulate_stationary_iid_behavior(
    benchmark: LatentGarnetBenchmark,
    dataset_size: int,
    rng: np.random.Generator,
) -> TransitionBatch:
    """Generate transitions i.i.d. from the behavior stationary state-action distribution."""

    n = int(dataset_size)
    nu_b = stationary_state_action_distribution(benchmark.mdp, benchmark.mdp.behavior_policy)
    flat_index = rng.choice(nu_b.shape[0], size=n, p=nu_b)
    states = flat_index // benchmark.config.n_actions
    actions = flat_index % benchmark.config.n_actions
    rewards = benchmark.mdp.rewards[states, actions]
    next_states = sample_next_states(benchmark.mdp.transition_prob, states, actions, rng)
    next_actions = sample_actions(benchmark.mdp.target_policy, next_states, rng)
    return TransitionBatch(
        states=np.asarray(states, dtype=np.int64),
        actions=np.asarray(actions, dtype=np.int64),
        rewards=np.asarray(rewards, dtype=np.float64),
        next_states=np.asarray(next_states, dtype=np.int64),
        next_actions=np.asarray(next_actions, dtype=np.int64),
    )


def _concat_batches(batches: list[TransitionBatch], rng: np.random.Generator) -> TransitionBatch:
    states = np.concatenate([batch.states for batch in batches], axis=0)
    actions = np.concatenate([batch.actions for batch in batches], axis=0)
    rewards = np.concatenate([batch.rewards for batch in batches], axis=0)
    next_states = np.concatenate([batch.next_states for batch in batches], axis=0)
    next_actions = np.concatenate([batch.next_actions for batch in batches], axis=0)
    perm = rng.permutation(states.shape[0])
    return TransitionBatch(
        states=states[perm],
        actions=actions[perm],
        rewards=rewards[perm],
        next_states=next_states[perm],
        next_actions=next_actions[perm],
    )


def simulate_behavior_dataset(
    benchmark: LatentGarnetBenchmark,
    dataset_size: int | None = None,
    seed: int | None = None,
    burn_in: int | None = None,
    data_mode: str | None = None,
    n_trajectories: int | None = None,
    trajectory_horizon: int | None = None,
    iid_fraction: float | None = None,
) -> TransitionBatch:
    """
    Generate offline data using one long chain, many shorter trajectories, stationary i.i.d.
    sampling, or a configurable mixture of trajectory and i.i.d. data.
    """

    n = benchmark.config.dataset_size if dataset_size is None else int(dataset_size)
    burn = benchmark.config.burn_in if burn_in is None else int(burn_in)
    mode = getattr(benchmark.config, "data_mode", "trajectory") if data_mode is None else data_mode
    n_traj = getattr(benchmark.config, "n_trajectories", 1) if n_trajectories is None else int(n_trajectories)
    traj_horizon = getattr(benchmark.config, "trajectory_horizon", None) if trajectory_horizon is None else int(trajectory_horizon)
    iid_mix = getattr(benchmark.config, "iid_fraction", 0.0) if iid_fraction is None else float(iid_fraction)
    rng = set_random_seed(benchmark.config.seed if seed is None else seed)

    if mode == "trajectory":
        return _simulate_long_behavior_trajectory(benchmark, dataset_size=n, rng=rng, burn_in=burn)
    if mode == "multi_trajectory":
        return _simulate_multi_trajectory_behavior(
            benchmark,
            dataset_size=n,
            rng=rng,
            n_trajectories=n_traj,
            trajectory_horizon=traj_horizon,
        )
    if mode == "stationary_iid":
        return _simulate_stationary_iid_behavior(benchmark, dataset_size=n, rng=rng)
    if mode == "mixed":
        n_iid = int(round(np.clip(iid_mix, 0.0, 1.0) * n))
        n_traj_samples = max(n - n_iid, 0)
        batches: list[TransitionBatch] = []
        if n_traj_samples > 0:
            batches.append(
                _simulate_multi_trajectory_behavior(
                    benchmark,
                    dataset_size=n_traj_samples,
                    rng=rng,
                    n_trajectories=n_traj,
                    trajectory_horizon=traj_horizon,
                )
            )
        if n_iid > 0:
            batches.append(_simulate_stationary_iid_behavior(benchmark, dataset_size=n_iid, rng=rng))
        if not batches:
            raise ValueError("The mixed data generator received zero total samples.")
        if len(batches) == 1:
            return batches[0]
        return _concat_batches(batches, rng)
    raise ValueError(f"Unsupported latent benchmark data_mode '{mode}'.")


def simulate_behavior_trajectory(
    benchmark: LatentGarnetBenchmark,
    dataset_size: int | None = None,
    seed: int | None = None,
    burn_in: int | None = None,
) -> TransitionBatch:
    """Backward-compatible alias for the benchmark's configured data generator."""

    return simulate_behavior_dataset(
        benchmark=benchmark,
        dataset_size=dataset_size,
        seed=seed,
        burn_in=burn_in,
    )


def evaluate_weight_estimators_on_benchmark(
    benchmark: LatentGarnetBenchmark,
    gamma_ratio: float = 1.0,
    seed: int | None = None,
) -> dict[str, object]:
    """Compare exact-ratio recovery for linear, neural, and RKHS-critic estimators."""

    batch = simulate_behavior_trajectory(benchmark, seed=benchmark.config.seed if seed is None else seed)
    phi_basic = benchmark.featurize_state_actions(batch.states, batch.actions, feature_set="linear_basic")
    phi_basic_next = benchmark.featurize_state_actions(batch.next_states, batch.next_actions, feature_set="linear_basic")
    phi_raw = benchmark.featurize_state_actions(batch.states, batch.actions, feature_set="raw")
    phi_raw_next = benchmark.featurize_state_actions(batch.next_states, batch.next_actions, feature_set="raw")
    phi_flex = benchmark.featurize_state_actions(batch.states, batch.actions, feature_set="flexible_linear")
    phi_flex_next = benchmark.featurize_state_actions(batch.next_states, batch.next_actions, feature_set="flexible_linear")

    exact_ratio = benchmark.exact_state_action_ratio(gamma_ratio=gamma_ratio)
    nu_b = stationary_state_action_distribution(benchmark.mdp, benchmark.mdp.behavior_policy)

    linear_basic = estimate_ratio_closed_form_linear(
        weight_features=phi_basic,
        critic_features=phi_basic,
        next_critic_features=phi_basic_next,
        gamma_ratio=gamma_ratio,
        ridge_primal=1e-5,
        ridge_dual=1e-5,
        normalization_penalty=10.0,
    )
    linear_flexible = estimate_ratio_closed_form_linear(
        weight_features=phi_flex,
        critic_features=phi_flex,
        next_critic_features=phi_flex_next,
        gamma_ratio=gamma_ratio,
        ridge_primal=1e-5,
        ridge_dual=1e-5,
        normalization_penalty=10.0,
    )
    neural = estimate_ratio_saddle_neural(
        weight_features=phi_raw,
        critic_features=phi_raw,
        next_critic_features=phi_raw_next,
        gamma_ratio=gamma_ratio,
        config=NeuralRatioConfig(
            max_steps=2000,
            batch_size=512,
            step_size=1e-3,
            ridge_weight=1e-4,
            ridge_critic=1e-4,
            normalization_penalty=10.0,
            valid_fraction=0.1,
            early_stopping_patience=15,
            uniform_mix=0.02,
            seed=benchmark.config.seed if seed is None else seed,
        ),
    )
    neural_rkhs = estimate_ratio_neural_rkhs(
        weight_features=phi_raw,
        critic_features=phi_raw,
        next_critic_features=phi_raw_next,
        gamma_ratio=gamma_ratio,
        config=NeuralRKHSWeightsConfig(
            max_steps=1500,
            learning_rate=1e-3,
            weight_decay=1e-4,
            critic_ridge=1e-4,
            normalization_penalty=10.0,
            valid_fraction=0.1,
            early_stopping_patience=10,
            uniform_mix=0.02,
            seed=benchmark.config.seed if seed is None else seed,
            kernel=KernelConfig(kernel="rbf", bandwidth="median", max_anchors=256),
        ),
    )

    grid_states = np.repeat(np.arange(benchmark.config.n_states), benchmark.config.n_actions)
    grid_actions = np.tile(np.arange(benchmark.config.n_actions), benchmark.config.n_states)
    grid_phi_basic = benchmark.featurize_state_actions(grid_states, grid_actions, feature_set="linear_basic")
    grid_phi_raw = benchmark.featurize_state_actions(grid_states, grid_actions, feature_set="raw")
    grid_phi_flex = benchmark.featurize_state_actions(grid_states, grid_actions, feature_set="flexible_linear")

    linear_basic_grid_ratio = positive_linear_ratio_weights(
        alpha=linear_basic.alpha,
        features=grid_phi_basic,
    )
    linear_flexible_grid_ratio = positive_linear_ratio_weights(
        alpha=linear_flexible.alpha,
        features=grid_phi_flex,
    )
    with torch.no_grad():
        grid_phi_t = torch.tensor(grid_phi_raw, dtype=torch.float32)
        neural_grid = neural.weight_model(grid_phi_t).cpu().numpy().astype(np.float64)
        neural_grid /= np.sum(neural_grid * nu_b)
        rkhs_grid = neural_rkhs.weight_model(grid_phi_t).cpu().numpy().astype(np.float64)
        rkhs_grid /= np.sum(rkhs_grid * nu_b)

    def metrics(estimator: str, estimated_ratio: np.ndarray) -> dict[str, float]:
        err = estimated_ratio - exact_ratio
        return {
            "estimator": estimator,
            "weighted_rmse": float(np.sqrt(np.sum(nu_b * err**2))),
            "unweighted_rmse": float(np.sqrt(np.mean(err**2))),
            "max_abs_error": float(np.max(np.abs(err))),
            "corr": float(np.corrcoef(exact_ratio, estimated_ratio)[0, 1]) if np.std(estimated_ratio) > 0 else 0.0,
        }

    q_star = evaluate_policy_tabular(benchmark.mdp, gamma=0.99)
    return {
        "overlap_metrics": benchmark.stationary_overlap_metrics(),
        "exact_ratio_summary": {
            "min": float(exact_ratio.min()),
            "max": float(exact_ratio.max()),
            "mean_under_behavior": float(np.sum(exact_ratio * nu_b)),
            "q_star_mean": float(np.mean(q_star)),
            "q_star_std": float(np.std(q_star)),
        },
        "linear_basic": metrics("linear_basic", linear_basic_grid_ratio),
        "linear_flexible": metrics("linear_flexible", linear_flexible_grid_ratio),
        "neural_saddle": metrics("neural_saddle", neural_grid),
        "neural_rkhs": metrics("neural_rkhs", rkhs_grid),
        "linear_basic_diagnostics": linear_basic.diagnostics,
        "linear_flexible_diagnostics": linear_flexible.diagnostics,
        "neural_diagnostics": neural.diagnostics,
        "neural_rkhs_diagnostics": neural_rkhs.diagnostics,
    }


def estimate_weight_methods_on_benchmark(
    benchmark: LatentGarnetBenchmark,
    gamma_ratio: float = 1.0,
    seed: int | None = None,
    quick: bool = False,
) -> dict[str, object]:
    """Fit all weight estimators once and return reusable artifacts for evaluation."""

    used_seed = benchmark.config.seed if seed is None else seed
    batch = simulate_behavior_trajectory(benchmark, seed=used_seed)
    phi_basic = benchmark.featurize_state_actions(batch.states, batch.actions, feature_set="linear_basic")
    phi_next_basic = benchmark.featurize_state_actions(batch.next_states, batch.next_actions, feature_set="linear_basic")
    phi_raw = benchmark.featurize_state_actions(batch.states, batch.actions, feature_set="raw")
    phi_next_raw = benchmark.featurize_state_actions(batch.next_states, batch.next_actions, feature_set="raw")
    phi_flex = benchmark.featurize_state_actions(batch.states, batch.actions, feature_set="flexible_linear")
    phi_next_flex = benchmark.featurize_state_actions(batch.next_states, batch.next_actions, feature_set="flexible_linear")

    exact_ratio = benchmark.exact_state_action_ratio(gamma_ratio=gamma_ratio)
    nu_b = stationary_state_action_distribution(benchmark.mdp, benchmark.mdp.behavior_policy)

    linear_basic = estimate_ratio_closed_form_linear(
        weight_features=phi_basic,
        critic_features=phi_basic,
        next_critic_features=phi_next_basic,
        gamma_ratio=gamma_ratio,
        ridge_primal=1e-5,
        ridge_dual=1e-5,
        normalization_penalty=10.0,
    )
    linear_flexible = estimate_ratio_closed_form_linear(
        weight_features=phi_flex,
        critic_features=phi_flex,
        next_critic_features=phi_next_flex,
        gamma_ratio=gamma_ratio,
        ridge_primal=1e-5,
        ridge_dual=1e-5,
        normalization_penalty=10.0,
    )
    neural = estimate_ratio_saddle_neural(
        weight_features=phi_raw,
        critic_features=phi_raw,
        next_critic_features=phi_next_raw,
        gamma_ratio=gamma_ratio,
        config=NeuralRatioConfig(
            max_steps=600 if quick else 2000,
            batch_size=256 if quick else 512,
            step_size=1e-3,
            ridge_weight=1e-4,
            ridge_critic=1e-4,
            normalization_penalty=10.0,
            valid_fraction=0.1,
            early_stopping_patience=8 if quick else 15,
            uniform_mix=0.02,
            seed=used_seed,
        ),
    )
    neural_rkhs = estimate_ratio_neural_rkhs(
        weight_features=phi_raw,
        critic_features=phi_raw,
        next_critic_features=phi_next_raw,
        gamma_ratio=gamma_ratio,
        config=NeuralRKHSWeightsConfig(
            max_steps=500 if quick else 1500,
            learning_rate=1e-3,
            weight_decay=1e-4,
            critic_ridge=1e-4,
            normalization_penalty=10.0,
            valid_fraction=0.1,
            early_stopping_patience=6 if quick else 10,
            uniform_mix=0.02,
            seed=used_seed,
            kernel=KernelConfig(kernel="rbf", bandwidth="median", max_anchors=128 if quick else 256),
        ),
    )
    return {
        "batch": batch,
        "state_action_features_basic": phi_basic,
        "next_state_action_features_basic": phi_next_basic,
        "state_action_features_raw": phi_raw,
        "next_state_action_features_raw": phi_next_raw,
        "state_action_features_flexible": phi_flex,
        "next_state_action_features_flexible": phi_next_flex,
        "exact_ratio": exact_ratio,
        "nu_b": nu_b,
        "linear_basic": linear_basic,
        "linear_flexible": linear_flexible,
        "neural": neural,
        "neural_rkhs": neural_rkhs,
    }


def _summarize_sample_weights(weights: np.ndarray) -> dict[str, float]:
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    ess = float((w.sum() ** 2) / max(np.sum(w**2), 1e-12))
    return {
        "min": float(w.min()),
        "max": float(w.max()),
        "mean": float(w.mean()),
        "std": float(w.std()),
        "cv": float(w.std() / max(w.mean(), 1e-12)),
        "q95": float(np.quantile(w, 0.95)),
        "q99": float(np.quantile(w, 0.99)),
        "effective_sample_size": ess,
        "effective_sample_size_fraction": float(ess / max(len(w), 1)),
    }


def evaluate_fqe_methods_on_benchmark(
    benchmark: LatentGarnetBenchmark,
    gamma_eval: float = 0.99,
    gamma_ratio: float = 1.0,
    seed: int | None = None,
    fqe_config: FQEConfig | None = None,
    flexible_fqe_config: FQEConfig | None = None,
    linear_fqe_config: LinearFQEConfig | None = None,
    quick: bool = False,
) -> dict[str, object]:
    """
    Compare weighted and unweighted FQE across linear and neural function classes.

    The study is designed to be swept from on-policy to off-policy by varying
    `behavior_coverage`. It reports:
    - neural FQE on a richer nonlinear observation map,
    - linear FQE on a compressed basis that is intentionally not Bellman complete,
    - oracle exact-ratio weighting,
    - learned weighting baselines,
    - weight-stability summaries to expose the correction/stability tradeoff.
    """

    if fqe_config is None:
        fqe_config = FQEConfig(
            gamma=gamma_eval,
            hidden_dims=(64, 64),
            n_outer_iters=12 if quick else 30,
            epochs_per_iter=10 if quick else 20,
            batch_size=256,
            learning_rate=5e-4,
            weight_decay=5e-4,
            grad_clip_norm=5.0,
            target_update_tau=0.1,
            valid_fraction=0.1,
            early_stopping_patience=5,
            min_improvement=1e-5,
            device="cpu",
        )
    if flexible_fqe_config is None:
        flexible_fqe_config = FQEConfig(
            gamma=gamma_eval,
            hidden_dims=(256, 256),
            n_outer_iters=15 if quick else 35,
            epochs_per_iter=12 if quick else 25,
            batch_size=256,
            learning_rate=5e-4,
            weight_decay=1e-4,
            grad_clip_norm=5.0,
            target_update_tau=0.1,
            valid_fraction=0.1,
            early_stopping_patience=6,
            min_improvement=1e-5,
            device="cpu",
        )
    if linear_fqe_config is None:
        linear_fqe_config = LinearFQEConfig(
            gamma=gamma_eval,
            ridge=5e-3,
            n_outer_iters=30 if quick else 60,
            target_update_tau=0.35,
            valid_fraction=0.1,
            early_stopping_patience=8,
            min_improvement=1e-6,
            tol=1e-8,
            use_averaging=True,
            averaging_start_iter=5,
        )

    used_seed = benchmark.config.seed if seed is None else seed
    weight_artifacts = estimate_weight_methods_on_benchmark(
        benchmark,
        gamma_ratio=gamma_ratio,
        seed=used_seed,
        quick=quick,
    )
    batch = weight_artifacts["batch"]
    x_linear = benchmark.featurize_state_actions(batch.states, batch.actions, feature_set="linear_q")
    x_linear_next = benchmark.featurize_state_actions(batch.next_states, batch.next_actions, feature_set="linear_q")
    x_sa = weight_artifacts["state_action_features_raw"]
    x_next = weight_artifacts["next_state_action_features_raw"]
    exact_ratio = np.asarray(weight_artifacts["exact_ratio"], dtype=np.float64)

    batch_indices = batch.states * benchmark.config.n_actions + batch.actions
    oracle_weights, oracle_meta = stabilize_weights(exact_ratio[batch_indices], return_metadata=True)
    raw_policy_ratio = (
        benchmark.mdp.target_policy[batch.states, batch.actions]
        / np.maximum(benchmark.mdp.behavior_policy[batch.states, batch.actions], 1e-12)
    )
    policy_ratio_weights, policy_ratio_meta = stabilize_weights(raw_policy_ratio, return_metadata=True)

    sample_weight_map = {
        "unweighted": None,
        "oracle": oracle_weights,
        "weighted_policy_ratio": policy_ratio_weights,
        "weighted_linear_basic": weight_artifacts["linear_basic"].weights,
        "weighted_linear_flexible": weight_artifacts["linear_flexible"].weights,
        "weighted_neural": weight_artifacts["neural"].weights,
        "weighted_neural_rkhs": weight_artifacts["neural_rkhs"].weights,
    }

    neural_results = {
        name: fit_weighted_fqe_nn(
            batch=batch,
            n_states=benchmark.config.n_states,
            n_actions=benchmark.config.n_actions,
            weights=weights,
            state_action_features=x_sa,
            next_state_action_features=x_next,
            config=fqe_config,
            seed=used_seed,
        )
        for name, weights in sample_weight_map.items()
    }
    flexible_neural_results = {
        name: fit_weighted_fqe_nn(
            batch=batch,
            n_states=benchmark.config.n_states,
            n_actions=benchmark.config.n_actions,
            weights=weights,
            state_action_features=x_sa,
            next_state_action_features=x_next,
            config=flexible_fqe_config,
            seed=used_seed,
        )
        for name, weights in sample_weight_map.items()
    }
    linear_results = {
        name: fit_weighted_linear_fqe(
            batch=batch,
            state_action_features=x_linear,
            next_state_action_features=x_linear_next,
            weights=weights,
            config=linear_fqe_config,
            seed=used_seed,
        )
        for name, weights in sample_weight_map.items()
    }

    grid_states = np.repeat(np.arange(benchmark.config.n_states), benchmark.config.n_actions)
    grid_actions = np.tile(np.arange(benchmark.config.n_actions), benchmark.config.n_states)
    grid_features_neural = benchmark.featurize_state_actions(grid_states, grid_actions, feature_set="raw")
    grid_features_linear = benchmark.featurize_state_actions(grid_states, grid_actions, feature_set="linear_q")
    q_star = evaluate_policy_tabular(benchmark.mdp, gamma=gamma_eval).reshape(-1)
    mu_pi = stationary_state_action_distribution(benchmark.mdp, benchmark.mdp.target_policy)
    nu_b = stationary_state_action_distribution(benchmark.mdp, benchmark.mdp.behavior_policy)
    mu_pi_state = mu_pi.reshape(benchmark.config.n_states, benchmark.config.n_actions).sum(axis=1)
    nu_b_state = nu_b.reshape(benchmark.config.n_states, benchmark.config.n_actions).sum(axis=1)
    q_star_rms = float(np.sqrt(np.sum(mu_pi * q_star**2)))

    initial_state_dist = np.asarray(benchmark.initial_state_distribution, dtype=np.float64)
    v_star = (benchmark.mdp.target_policy * q_star.reshape(benchmark.config.n_states, benchmark.config.n_actions)).sum(axis=1)
    initial_value_true = float(initial_state_dist @ v_star)

    def summarize_neural(name: str, result) -> dict[str, float]:
        preds = predict_q_values(
            result.model,
            grid_states,
            grid_actions,
            benchmark.config.n_states,
            benchmark.config.n_actions,
            state_action_features=grid_features_neural,
            device=fqe_config.device,
        )
        err = preds - q_star
        q_hat = preds.reshape(benchmark.config.n_states, benchmark.config.n_actions)
        v_hat = (benchmark.mdp.target_policy * q_hat).sum(axis=1)
        v_err = v_hat - v_star
        value_error = float(np.sum(mu_pi * preds) - np.sum(mu_pi * q_star))
        initial_value_estimate = float(initial_state_dist @ v_hat)
        return {
            "method": name,
            "stationary_q_rmse": float(np.sqrt(np.sum(mu_pi * err**2))),
            "stationary_rmse": float(np.sqrt(np.sum(mu_pi * err**2))),
            "target_policy_rmse": float(np.sqrt(np.sum(mu_pi * err**2))),
            "target_policy_relative_rmse": float(np.sqrt(np.sum(mu_pi * err**2)) / max(q_star_rms, 1e-12)),
            "behavior_q_rmse": float(np.sqrt(np.sum(nu_b * err**2))),
            "behavior_rmse": float(np.sqrt(np.sum(nu_b * err**2))),
            "behavior_policy_rmse": float(np.sqrt(np.sum(nu_b * err**2))),
            "stationary_v_rmse": float(np.sqrt(np.sum(mu_pi_state * v_err**2))),
            "behavior_v_rmse": float(np.sqrt(np.sum(nu_b_state * v_err**2))),
            "uniform_rmse": float(np.sqrt(np.mean(err**2))),
            "stationary_value_error": value_error,
            "initial_state_value_rmse": float(np.sqrt(initial_state_dist @ ((v_hat - v_star) ** 2))),
            "initial_policy_value_estimate": initial_value_estimate,
            "initial_policy_value_true": initial_value_true,
            "initial_policy_value_error": float(initial_value_estimate - initial_value_true),
            "initial_policy_value_abs_error": float(abs(initial_value_estimate - initial_value_true)),
            "initial_policy_value_relative_abs_error": float(abs(initial_value_estimate - initial_value_true) / max(abs(initial_value_true), 1e-12)),
        }

    def summarize_linear(name: str, result) -> dict[str, float]:
        preds = predict_linear_q_values(result.theta, grid_features_linear)
        err = preds - q_star
        q_hat = preds.reshape(benchmark.config.n_states, benchmark.config.n_actions)
        v_hat = (benchmark.mdp.target_policy * q_hat).sum(axis=1)
        v_err = v_hat - v_star
        value_error = float(np.sum(mu_pi * preds) - np.sum(mu_pi * q_star))
        initial_value_estimate = float(initial_state_dist @ v_hat)
        return {
            "method": name,
            "stationary_q_rmse": float(np.sqrt(np.sum(mu_pi * err**2))),
            "stationary_rmse": float(np.sqrt(np.sum(mu_pi * err**2))),
            "target_policy_rmse": float(np.sqrt(np.sum(mu_pi * err**2))),
            "target_policy_relative_rmse": float(np.sqrt(np.sum(mu_pi * err**2)) / max(q_star_rms, 1e-12)),
            "behavior_q_rmse": float(np.sqrt(np.sum(nu_b * err**2))),
            "behavior_rmse": float(np.sqrt(np.sum(nu_b * err**2))),
            "behavior_policy_rmse": float(np.sqrt(np.sum(nu_b * err**2))),
            "stationary_v_rmse": float(np.sqrt(np.sum(mu_pi_state * v_err**2))),
            "behavior_v_rmse": float(np.sqrt(np.sum(nu_b_state * v_err**2))),
            "uniform_rmse": float(np.sqrt(np.mean(err**2))),
            "stationary_value_error": value_error,
            "initial_state_value_rmse": float(np.sqrt(initial_state_dist @ ((v_hat - v_star) ** 2))),
            "initial_policy_value_estimate": initial_value_estimate,
            "initial_policy_value_true": initial_value_true,
            "initial_policy_value_error": float(initial_value_estimate - initial_value_true),
            "initial_policy_value_abs_error": float(abs(initial_value_estimate - initial_value_true)),
            "initial_policy_value_relative_abs_error": float(abs(initial_value_estimate - initial_value_true) / max(abs(initial_value_true), 1e-12)),
        }

    weight_stability = {
        "oracle": {
            **_summarize_sample_weights(oracle_weights),
            "effective_max_weight": oracle_meta["effective_max_weight"],
        },
        "policy_ratio": {
            **_summarize_sample_weights(policy_ratio_weights),
            "effective_max_weight": policy_ratio_meta["effective_max_weight"],
        },
        "linear_basic": _summarize_sample_weights(weight_artifacts["linear_basic"].weights),
        "linear_flexible": _summarize_sample_weights(weight_artifacts["linear_flexible"].weights),
        "neural": _summarize_sample_weights(weight_artifacts["neural"].weights),
        "neural_rkhs": _summarize_sample_weights(weight_artifacts["neural_rkhs"].weights),
    }

    return {
        "overlap_metrics": benchmark.stationary_overlap_metrics(),
        "exact_ratio_summary": {
            "min": float(exact_ratio.min()),
            "max": float(exact_ratio.max()),
            "mean_under_behavior": float(np.sum(exact_ratio * nu_b)),
            "target_q_rms": q_star_rms,
        },
        "policy_evaluation_setup": {
            "initial_state_distribution": "benchmark_goal_avoiding_start_distribution",
            "true_initial_policy_value": initial_value_true,
        },
        "linear_fqe_metrics": {
            name: summarize_linear(name, result) for name, result in linear_results.items()
        },
        "neural_fqe_metrics": {
            name: summarize_neural(name, result) for name, result in neural_results.items()
        },
        "neural_fqe_flexible_metrics": {
            name: summarize_neural(name, result) for name, result in flexible_neural_results.items()
        },
        "weight_stability": weight_stability,
        "weight_diagnostics": {
            "oracle": {
                "effective_max_weight": oracle_meta["effective_max_weight"],
                "chosen_uniform_mix": float(oracle_meta["chosen_uniform_mix"]),
                "ess_fraction_before_mix": float(oracle_meta["ess_fraction_before_mix"]),
                "ess_fraction_after_mix": float(oracle_meta["ess_fraction_after_mix"]),
                "target_ess_fraction": oracle_meta["target_ess_fraction"],
            },
            "policy_ratio": {
                "min_weight_raw": float(raw_policy_ratio.min()),
                "max_weight_raw": float(raw_policy_ratio.max()),
                "min_weight_processed": float(policy_ratio_weights.min()),
                "max_weight_processed": float(policy_ratio_weights.max()),
                "effective_max_weight": policy_ratio_meta["effective_max_weight"],
                "chosen_uniform_mix": float(policy_ratio_meta["chosen_uniform_mix"]),
                "ess_fraction_before_mix": float(policy_ratio_meta["ess_fraction_before_mix"]),
                "ess_fraction_after_mix": float(policy_ratio_meta["ess_fraction_after_mix"]),
                "target_ess_fraction": policy_ratio_meta["target_ess_fraction"],
            },
            "linear_basic": weight_artifacts["linear_basic"].diagnostics,
            "linear_flexible": weight_artifacts["linear_flexible"].diagnostics,
            "neural": weight_artifacts["neural"].diagnostics,
            "neural_rkhs": weight_artifacts["neural_rkhs"].diagnostics,
        },
    }
