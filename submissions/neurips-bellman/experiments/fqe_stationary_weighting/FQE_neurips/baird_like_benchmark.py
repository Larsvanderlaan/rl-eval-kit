from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .latent_garnet_benchmark import evaluate_fqe_methods_on_benchmark, evaluate_weight_estimators_on_benchmark
from .utils import (
    DiscreteMDP,
    exact_ratio_against_behavior,
    one_hot,
    stationary_state_action_distribution,
)


@dataclass
class BairdLikeConfig:
    """Baird-inspired PE benchmark with a goal state heavily favored by the target policy."""

    n_spokes: int = 6
    teleport_mass: float = 0.01
    behavior_solid_prob: float = 0.05
    target_solid_prob: float = 0.98
    goal_stickiness_target: float = 0.98
    goal_exit_behavior: float = 0.85
    reward_goal: float = 1.0
    reward_hub_bonus: float = 0.2
    dataset_size: int = 1_000
    burn_in: int = 1_000
    seed: int = 0

    @property
    def n_states(self) -> int:
        return self.n_spokes + 2

    @property
    def n_actions(self) -> int:
        return 2


@dataclass
class BairdLikeBenchmark:
    config: BairdLikeConfig
    mdp: DiscreteMDP
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
        states = np.asarray(states, dtype=np.int64).reshape(-1)
        actions = np.asarray(actions, dtype=np.int64).reshape(-1)
        if feature_set == "raw":
            obs = self.observed_state_features[states]
            a_one_hot = one_hot(actions, self.config.n_actions)
            return np.einsum("nd,na->nda", obs, a_one_hot).reshape(len(states), -1)
        elif feature_set == "linear_basic":
            obs = self.linear_basic_state_features[states]
            a_one_hot = one_hot(actions, self.config.n_actions)
            return np.einsum("nd,na->nda", obs, a_one_hot).reshape(len(states), -1)
        elif feature_set == "flexible_linear":
            obs = self.flexible_linear_state_features[states]
            a_one_hot = one_hot(actions, self.config.n_actions)
            return np.einsum("nd,na->nda", obs, a_one_hot).reshape(len(states), -1)
        elif feature_set == "linear_q":
            obs = self.linear_q_state_features[states]
            a_one_hot = one_hot(actions, self.config.n_actions)
            # For linear FQE we deliberately share state geometry across actions.
            return np.concatenate([obs, a_one_hot], axis=1)
        else:
            raise ValueError(f"Unknown feature_set '{feature_set}'.")

    def exact_state_action_ratio(self, gamma_ratio: float) -> np.ndarray:
        return exact_ratio_against_behavior(self.mdp, gamma_ratio=gamma_ratio)

    def stationary_overlap_metrics(self) -> dict[str, float]:
        mu_pi = stationary_state_action_distribution(self.mdp, self.mdp.target_policy)
        nu_b = stationary_state_action_distribution(self.mdp, self.mdp.behavior_policy)
        support = mu_pi > 1e-12
        ratio = nu_b[support] / mu_pi[support]
        return {
            "requested_behavior_coverage": float(1.0 - (1.0 - self.config.behavior_solid_prob)),
            "realized_min_density_ratio_nub_over_target": float(np.min(ratio)),
            "realized_overlap_mass": float(np.sum(np.minimum(mu_pi, nu_b))),
            "chi2_target_vs_behavior": float(np.sum(((mu_pi - nu_b) ** 2) / np.maximum(nu_b, 1e-12))),
        }


def _state_coords(config: BairdLikeConfig) -> np.ndarray:
    n_spokes = config.n_spokes
    angles = np.linspace(0.0, 2.0 * np.pi, num=n_spokes, endpoint=False)
    spokes = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    hub = np.array([[0.0, 0.0]])
    goal = np.array([[0.15, 0.05]])
    return np.vstack([spokes, hub, goal]).astype(np.float64)


def _build_observed_features(coords: np.ndarray) -> np.ndarray:
    radius = np.linalg.norm(coords, axis=1, keepdims=True)
    angle_feats = np.stack(
        [np.cos(np.arctan2(coords[:, 1], coords[:, 0])), np.sin(np.arctan2(coords[:, 1], coords[:, 0]))],
        axis=1,
    )
    return np.concatenate([coords, radius, angle_feats], axis=1)


def _build_linear_basic_features(obs: np.ndarray) -> np.ndarray:
    x = np.asarray(obs, dtype=np.float64)
    mean = x.mean(axis=0)
    std = np.where(x.std(axis=0) < 1e-8, 1.0, x.std(axis=0))
    x_std = (x - mean) / std
    bias = np.ones((x.shape[0], 1), dtype=np.float64)
    return np.concatenate([bias, x_std[:, :2], np.linalg.norm(x_std, axis=1, keepdims=True)], axis=1)


def _build_flexible_linear_features(obs: np.ndarray, seed: int, out_dim: int = 24) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.asarray(obs, dtype=np.float64)
    mean = x.mean(axis=0)
    std = np.where(x.std(axis=0) < 1e-8, 1.0, x.std(axis=0))
    x_std = (x - mean) / std
    W = rng.normal(scale=0.75, size=(x_std.shape[1], out_dim))
    b = rng.uniform(0.0, 2.0 * np.pi, size=out_dim)
    rff = np.sqrt(2.0 / out_dim) * np.cos(x_std @ W + b)
    return np.concatenate([x_std, rff], axis=1)


def _build_baird_linear_q_features(config: BairdLikeConfig) -> np.ndarray:
    n_spokes = config.n_spokes
    n_states = config.n_states
    feats = np.zeros((n_states, n_spokes + 2), dtype=np.float64)
    for i in range(n_spokes):
        feats[i, i] = 2.0
        feats[i, -2] = 1.0
    hub = n_spokes
    goal = n_spokes + 1
    feats[hub, -2] = 2.0
    feats[hub, -1] = 1.0
    feats[goal, -2] = 1.5
    feats[goal, -1] = 2.0
    return feats


def build_baird_like_benchmark(config: BairdLikeConfig | None = None) -> BairdLikeBenchmark:
    if config is None:
        config = BairdLikeConfig()

    rng = np.random.default_rng(config.seed)
    n_spokes = config.n_spokes
    hub = n_spokes
    goal = n_spokes + 1
    n_states = config.n_states
    n_actions = config.n_actions

    transition_prob = np.zeros((n_states, n_actions, n_states), dtype=np.float64)

    # Action 0 = solid, Action 1 = dashed.
    for s in range(n_spokes):
        transition_prob[s, 0, hub] = 1.0
        transition_prob[s, 1, :n_spokes] = 1.0 / n_spokes

    transition_prob[hub, 0, goal] = 1.0
    transition_prob[hub, 1, :n_spokes] = 1.0 / n_spokes

    transition_prob[goal, 0, goal] = config.goal_stickiness_target
    transition_prob[goal, 0, :n_spokes] = (1.0 - config.goal_stickiness_target) / n_spokes
    transition_prob[goal, 1, :n_spokes] = config.goal_exit_behavior / n_spokes
    transition_prob[goal, 1, goal] = 1.0 - config.goal_exit_behavior

    transition_prob = (1.0 - config.teleport_mass) * transition_prob
    transition_prob[:, :, :n_spokes] += config.teleport_mass / n_spokes

    rewards = np.zeros((n_states, n_actions), dtype=np.float64)
    rewards[hub, 0] = config.reward_hub_bonus
    rewards[goal, :] = config.reward_goal

    target_policy = np.zeros((n_states, n_actions), dtype=np.float64)
    behavior_policy = np.zeros((n_states, n_actions), dtype=np.float64)
    target_policy[:, 0] = config.target_solid_prob
    target_policy[:, 1] = 1.0 - config.target_solid_prob
    behavior_policy[:, 0] = config.behavior_solid_prob
    behavior_policy[:, 1] = 1.0 - config.behavior_solid_prob
    behavior_policy[goal, 1] = config.goal_exit_behavior
    behavior_policy[goal, 0] = 1.0 - config.goal_exit_behavior

    mdp = DiscreteMDP(
        transition_prob=transition_prob,
        rewards=rewards,
        target_policy=target_policy,
        behavior_policy=behavior_policy,
    )

    coords = _state_coords(config)
    observed = _build_observed_features(coords)
    linear_basic = _build_linear_basic_features(observed)
    flexible_linear = _build_flexible_linear_features(observed, seed=config.seed)
    linear_q = _build_baird_linear_q_features(config)
    initial_state_distribution = np.zeros(n_states, dtype=np.float64)
    initial_state_distribution[:n_spokes] = 1.0 / n_spokes

    target_logits = np.log(np.maximum(target_policy, 1e-12))
    distractor_logits = np.log(np.maximum(behavior_policy, 1e-12))

    return BairdLikeBenchmark(
        config=config,
        mdp=mdp,
        observed_state_features=observed,
        linear_basic_state_features=linear_basic,
        flexible_linear_state_features=flexible_linear,
        linear_q_state_features=linear_q,
        initial_state_distribution=initial_state_distribution,
        target_policy_logits=target_logits,
        distractor_policy_logits=distractor_logits,
    )


def run_baird_like_sanity(
    config: BairdLikeConfig | None = None,
    gamma_eval: float = 0.99,
    gamma_ratio: float = 1.0,
) -> dict[str, object]:
    benchmark = build_baird_like_benchmark(config)
    return {
        "weight_results": evaluate_weight_estimators_on_benchmark(benchmark, gamma_ratio=gamma_ratio, seed=benchmark.config.seed),
        "fqe_results": evaluate_fqe_methods_on_benchmark(benchmark, gamma_eval=gamma_eval, gamma_ratio=gamma_ratio, seed=benchmark.config.seed),
    }
