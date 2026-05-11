from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .latent_garnet_benchmark import (
    _build_flexible_linear_state_features,
    _build_linear_basic_state_features,
)
from .utils import (
    DiscreteMDP,
    exact_ratio_against_behavior,
    one_hot,
    stationary_state_action_distribution,
)


@dataclass
class HubSpokeLatentConfig:
    """
    Realistic Baird-inspired benchmark.

    The hard linear benchmark is built so that Q^pi is realizable in the
    prescribed linear FQE class, but the class is not Bellman complete because
    aliased microstates at the same coarse "depth" have different target
    transition kernels.
    """

    n_spokes: int = 8
    spoke_depth: int = 5
    microstates_per_depth: int = 3
    hub_states: int = 6
    goal_states: int = 4
    observation_mode: str = "compact_nonlinear"
    dataset_size: int = 2_000
    burn_in: int = 1_000
    target_solid_prob: float = 0.98
    behavior_solid_prob: float = 0.15
    target_micro_bias: float = 0.03
    behavior_micro_bias: float = 0.22
    critical_inward_boost: float = 0.22
    noncritical_detour_boost: float = 0.20
    goal_exit_behavior: float = 0.90
    teleport_mass: float = 0.01
    reward_gamma: float = 0.95
    reward_scale: float = 8.0
    action_gap: float = 1.5
    seed: int = 0
    flexible_linear_rff_dim: int = 24
    flexible_linear_bandwidth: float | str | None = "median"
    flexible_linear_include_raw: bool = True
    basic_linear_raw_dims: int = 4

    @property
    def spoke_states(self) -> int:
        return self.spoke_depth * self.microstates_per_depth

    @property
    def n_states(self) -> int:
        return self.n_spokes * self.spoke_states + self.hub_states + self.goal_states

    @property
    def n_actions(self) -> int:
        return 2


@dataclass
class HubSpokeLatentBenchmark:
    config: HubSpokeLatentConfig
    mdp: DiscreteMDP
    observed_state_features: np.ndarray
    linear_basic_state_features: np.ndarray
    flexible_linear_state_features: np.ndarray
    linear_q_state_features: np.ndarray
    neural_structured_state_features: np.ndarray
    linear_q_state_action_features: np.ndarray
    critical_state_mask: np.ndarray
    initial_state_distribution: np.ndarray
    target_policy_logits: np.ndarray
    distractor_policy_logits: np.ndarray
    design_diagnostics: dict[str, float]

    def featurize_state_actions(self, states: np.ndarray, actions: np.ndarray, feature_set: str = "raw") -> np.ndarray:
        states = np.asarray(states, dtype=np.int64).reshape(-1)
        actions = np.asarray(actions, dtype=np.int64).reshape(-1)
        if feature_set == "raw":
            obs = self.observed_state_features[states]
            return np.einsum("nd,na->nda", obs, one_hot(actions, self.config.n_actions)).reshape(len(states), -1)
        if feature_set == "linear_basic":
            obs = self.linear_basic_state_features[states]
            return np.einsum("nd,na->nda", obs, one_hot(actions, self.config.n_actions)).reshape(len(states), -1)
        if feature_set == "flexible_linear":
            obs = self.flexible_linear_state_features[states]
            return np.einsum("nd,na->nda", obs, one_hot(actions, self.config.n_actions)).reshape(len(states), -1)
        if feature_set == "neural_structured":
            obs = self.neural_structured_state_features[states]
            return np.einsum("nd,na->nda", obs, one_hot(actions, self.config.n_actions)).reshape(len(states), -1)
        if feature_set == "linear_q":
            sa_index = states * self.config.n_actions + actions
            return self.linear_q_state_action_features[sa_index]
        raise ValueError(f"Unknown feature_set '{feature_set}'.")

    def exact_state_action_ratio(self, gamma_ratio: float) -> np.ndarray:
        return exact_ratio_against_behavior(self.mdp, gamma_ratio=gamma_ratio)

    def stationary_overlap_metrics(self) -> dict[str, float]:
        mu_pi = stationary_state_action_distribution(self.mdp, self.mdp.target_policy)
        nu_b = stationary_state_action_distribution(self.mdp, self.mdp.behavior_policy)
        support = mu_pi > 1e-12
        ratio = nu_b[support] / np.maximum(mu_pi[support], 1e-12)
        return {
            "requested_behavior_coverage": float(self.config.behavior_solid_prob),
            "realized_min_density_ratio_nub_over_target": float(np.min(ratio)),
            "realized_overlap_mass": float(np.sum(np.minimum(mu_pi, nu_b))),
            "chi2_target_vs_behavior": float(np.sum(((mu_pi - nu_b) ** 2) / np.maximum(nu_b, 1e-12))),
            "target_critical_mass": float(np.sum(mu_pi.reshape(self.config.n_states, self.config.n_actions)[self.critical_state_mask])),
            "behavior_critical_mass": float(np.sum(nu_b.reshape(self.config.n_states, self.config.n_actions)[self.critical_state_mask])),
            "linear_realizability_rmse": float(self.design_diagnostics["linear_realizability_rmse"]),
            "bellman_incompleteness_rmse": float(self.design_diagnostics["bellman_incompleteness_rmse"]),
        }


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    row_sums = matrix.sum(axis=1, keepdims=True)
    return matrix / np.maximum(row_sums, 1e-12)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _nonlinear_observations(coords: np.ndarray, depth_values: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = np.concatenate([coords, depth_values[:, None], depth_values[:, None] ** 2], axis=1)
    w1 = rng.normal(size=(base.shape[1], 8))
    b1 = rng.uniform(-np.pi, np.pi, size=8)
    w2 = rng.normal(size=(base.shape[1], 8))
    b2 = rng.uniform(-1.0, 1.0, size=8)
    trig = np.sin(base @ w1 + b1)
    tanh = np.tanh(base @ w2 + b2)
    return np.concatenate([base, trig, tanh], axis=1)


def _build_linear_q_state_features(
    *,
    spoke_depth_index: np.ndarray,
    spoke_id: np.ndarray,
    region_id: np.ndarray,
    n_spokes: int,
    max_depth: int,
) -> np.ndarray:
    """
    Coarse value features that intentionally alias many microstates.

    Features depend on coarse radial depth and spoke sector, but not on the
    microstate identity that drives target-transition heterogeneity.
    """

    n_states = len(region_id)
    sector_count = min(n_spokes, 4)
    feats = np.zeros((n_states, 8 + sector_count), dtype=np.float64)

    for s in range(n_states):
        feats[s, 0] = 1.0
        if region_id[s] == 0:
            progress = float(spoke_depth_index[s]) / max(max_depth - 1, 1)
            feats[s, 1] = progress
            feats[s, 2] = progress**2
            feats[s, 3] = float(spoke_depth_index[s] == max_depth - 1)
            feats[s, 4 + (spoke_id[s] % sector_count)] = 1.0
        elif region_id[s] == 1:
            feats[s, 1] = 1.10
            feats[s, 2] = 1.21
            feats[s, 3] = 1.0
            feats[s, 4 + sector_count] = 1.0
        else:
            feats[s, 1] = 1.30
            feats[s, 2] = 1.69
            feats[s, 3] = 1.0
            feats[s, 4 + sector_count + 1] = 1.0

    return feats


def _projection_rmse(target: np.ndarray, basis: np.ndarray) -> float:
    coef, *_ = np.linalg.lstsq(basis, target, rcond=None)
    residual = target - basis @ coef
    return float(np.sqrt(np.mean(residual**2)))


def build_hub_spoke_latent_benchmark(config: HubSpokeLatentConfig | None = None) -> HubSpokeLatentBenchmark:
    if config is None:
        config = HubSpokeLatentConfig()

    rng = np.random.default_rng(config.seed)
    n_spokes = config.n_spokes
    n_actions = config.n_actions
    max_depth = config.spoke_depth

    coords = []
    region_id = []
    spoke_id = []
    spoke_depth_index = []
    micro_id = []
    spoke_level_state_ids: list[list[np.ndarray]] = [[] for _ in range(n_spokes)]
    state_index = 0

    radii = np.linspace(1.25, 0.35, config.spoke_depth)
    for j, angle in enumerate(np.linspace(0.0, 2.0 * np.pi, n_spokes, endpoint=False)):
        direction = np.array([np.cos(angle), np.sin(angle)], dtype=np.float64)
        tangent = np.array([-direction[1], direction[0]], dtype=np.float64)
        per_depth_ids: list[np.ndarray] = []
        for depth, radius in enumerate(radii):
            ids = []
            for m in range(config.microstates_per_depth):
                radial_jitter = 0.03 * rng.normal()
                tangential_offset = (m - 0.5 * (config.microstates_per_depth - 1)) * 0.08
                point = (radius + radial_jitter) * direction + tangential_offset * tangent + 0.015 * rng.normal(size=2)
                coords.append(point)
                region_id.append(0)
                spoke_id.append(j)
                spoke_depth_index.append(depth)
                micro_id.append(m)
                ids.append(state_index)
                state_index += 1
            per_depth_ids.append(np.asarray(ids, dtype=np.int64))
        spoke_level_state_ids[j] = per_depth_ids

    hub_ids = []
    for m in range(config.hub_states):
        point = np.array([0.04, -0.02]) + 0.05 * rng.normal(size=2) + np.array([0.02 * m, -0.01 * m])
        coords.append(point)
        region_id.append(1)
        spoke_id.append(-1)
        spoke_depth_index.append(config.spoke_depth)
        micro_id.append(m)
        hub_ids.append(state_index)
        state_index += 1

    goal_center = np.array([0.0, 0.0], dtype=np.float64)
    goal_ids = []
    for m in range(config.goal_states):
        point = goal_center + 0.03 * rng.normal(size=2)
        coords.append(point)
        region_id.append(2)
        spoke_id.append(-1)
        spoke_depth_index.append(config.spoke_depth + 1)
        micro_id.append(m)
        goal_ids.append(state_index)
        state_index += 1

    coords = np.asarray(coords, dtype=np.float64)
    region_id = np.asarray(region_id, dtype=np.int64)
    spoke_id = np.asarray(spoke_id, dtype=np.int64)
    spoke_depth_index = np.asarray(spoke_depth_index, dtype=np.int64)
    micro_id = np.asarray(micro_id, dtype=np.int64)
    n_states = len(region_id)

    criticality = np.zeros(n_states, dtype=np.float64)
    spoke_mask = region_id == 0
    if np.any(spoke_mask):
        parity = ((spoke_id[spoke_mask] % 2) == 0).astype(np.float64)
        micro_score = 1.0 - micro_id[spoke_mask] / max(config.microstates_per_depth - 1, 1)
        depth_score = 1.0 - spoke_depth_index[spoke_mask] / max(config.spoke_depth - 1, 1)
        criticality[spoke_mask] = 0.55 * parity + 0.25 * micro_score + 0.20 * depth_score
    criticality[region_id == 1] = 0.75
    criticality[region_id == 2] = 1.0
    criticality = np.clip(criticality, 0.0, 1.0)

    depth_values = np.where(
        region_id == 0,
        spoke_depth_index / max(config.spoke_depth - 1, 1),
        np.where(region_id == 1, 1.15, 1.35),
    ).astype(np.float64)

    transition_prob = np.zeros((n_states, n_actions, n_states), dtype=np.float64)
    outer_spoke_states = np.concatenate([levels[0] for levels in spoke_level_state_ids]).astype(np.int64)
    mid_spoke_states = np.concatenate([levels[min(1, config.spoke_depth - 1)] for levels in spoke_level_state_ids]).astype(
        np.int64
    )

    for j in range(n_spokes):
        for depth in range(config.spoke_depth):
            ids = spoke_level_state_ids[j][depth]
            for local_rank, s in enumerate(ids):
                target_inward = (
                    spoke_level_state_ids[j][depth + 1] if depth < config.spoke_depth - 1 else np.asarray(hub_ids, dtype=np.int64)
                )
                inward_weights = rng.dirichlet(0.5 + np.arange(len(target_inward)))
                if depth < config.spoke_depth - 1:
                    neighbor_spoke = (j + 1 + local_rank) % n_spokes
                    secondary = spoke_level_state_ids[neighbor_spoke][depth + 1]
                    critical_boost = config.critical_inward_boost * criticality[s]
                    detour_penalty = config.noncritical_detour_boost * (1.0 - criticality[s])
                    same_spoke_mass = np.clip(0.80 + critical_boost - 0.10 * detour_penalty, 0.55, 0.97)
                    secondary_mass = np.clip(0.15 + detour_penalty - 0.10 * critical_boost, 0.02, 0.35)
                    stall_mass = max(1.0 - same_spoke_mass - secondary_mass, 0.0)
                    transition_prob[s, 0, target_inward] += same_spoke_mass * inward_weights
                    transition_prob[s, 0, secondary] += secondary_mass / len(secondary)
                    transition_prob[s, 0, ids] += stall_mass / len(ids)
                else:
                    hub_mass = np.clip(0.78 + 0.18 * criticality[s], 0.70, 0.97)
                    transition_prob[s, 0, target_inward] += hub_mass * inward_weights
                    transition_prob[s, 0, ids] += (1.0 - hub_mass) / len(ids)

                outward_depth = max(depth - 1, 0)
                same_level = spoke_level_state_ids[j][depth]
                outward_level = spoke_level_state_ids[j][outward_depth]
                side_spoke = (j + local_rank + 2) % n_spokes
                side_level = spoke_level_state_ids[side_spoke][outward_depth]
                critical_drag = 0.12 * criticality[s]
                transition_prob[s, 1, outward_level] += (0.42 - 0.10 * critical_drag) / len(outward_level)
                transition_prob[s, 1, same_level] += (0.18 + 0.08 * critical_drag) / len(same_level)
                transition_prob[s, 1, side_level] += (0.22 + 0.12 * (1.0 - criticality[s])) / len(side_level)
                remaining = max(1.0 - transition_prob[s, 1].sum(), 0.0)
                transition_prob[s, 1, outer_spoke_states] += remaining / len(outer_spoke_states)

    for h_rank, s in enumerate(hub_ids):
        goal_weights = rng.dirichlet(np.linspace(1.0, 2.0, len(goal_ids)))
        transition_prob[s, 0, goal_ids] = goal_weights
        ejected_spoke = (2 * h_rank) % n_spokes
        transition_prob[s, 1, outer_spoke_states] += 0.65 / len(outer_spoke_states)
        transition_prob[s, 1, spoke_level_state_ids[ejected_spoke][0]] += 0.20 / len(spoke_level_state_ids[ejected_spoke][0])
        transition_prob[s, 1, mid_spoke_states] += 0.15 / len(mid_spoke_states)

    for g_rank, s in enumerate(goal_ids):
        transition_prob[s, 0, goal_ids] += 0.92 / len(goal_ids)
        transition_prob[s, 0, hub_ids] += 0.08 / len(hub_ids)
        ejected_spoke = (g_rank + 1) % n_spokes
        transition_prob[s, 1, outer_spoke_states] += config.goal_exit_behavior / len(outer_spoke_states)
        transition_prob[s, 1, spoke_level_state_ids[ejected_spoke][0]] += 0.10 / len(spoke_level_state_ids[ejected_spoke][0])
        remaining_goal_mass = max(1.0 - config.goal_exit_behavior - 0.10, 0.0)
        transition_prob[s, 1, goal_ids] += remaining_goal_mass / len(goal_ids)

    transition_prob = _normalize_rows(transition_prob.reshape(-1, n_states)).reshape(n_states, n_actions, n_states)
    transition_prob = (1.0 - config.teleport_mass) * transition_prob + config.teleport_mass / n_states
    transition_prob = _normalize_rows(transition_prob.reshape(-1, n_states)).reshape(n_states, n_actions, n_states)

    target_policy = np.zeros((n_states, n_actions), dtype=np.float64)
    behavior_policy = np.zeros((n_states, n_actions), dtype=np.float64)
    target_logit = np.log(config.target_solid_prob / max(1.0 - config.target_solid_prob, 1e-12))
    behavior_logit = np.log(config.behavior_solid_prob / max(1.0 - config.behavior_solid_prob, 1e-12))
    target_prob0 = _sigmoid(target_logit + config.target_micro_bias * (criticality - criticality.mean()))
    behavior_prob0 = _sigmoid(behavior_logit - config.behavior_micro_bias * (criticality - criticality.mean()))
    target_policy[:, 0] = target_prob0
    target_policy[:, 1] = 1.0 - target_prob0
    behavior_policy[:, 0] = behavior_prob0
    behavior_policy[:, 1] = 1.0 - behavior_prob0
    behavior_policy[np.asarray(goal_ids, dtype=np.int64), 1] = config.goal_exit_behavior
    behavior_policy[np.asarray(goal_ids, dtype=np.int64), 0] = 1.0 - config.goal_exit_behavior

    observed = _nonlinear_observations(coords, depth_values, seed=config.seed)
    linear_basic = _build_linear_basic_state_features(observed, config)
    flexible_linear = _build_flexible_linear_state_features(observed, config, rng)
    linear_q_state_features = _build_linear_q_state_features(
        spoke_depth_index=spoke_depth_index,
        spoke_id=spoke_id,
        region_id=region_id,
        n_spokes=n_spokes,
        max_depth=config.spoke_depth,
    )
    neural_structured_state_features = np.concatenate(
        [
            observed,
            linear_q_state_features,
            criticality[:, None],
            one_hot(np.clip(region_id, 0, 2), 3),
        ],
        axis=1,
    )

    base_phi_grid = np.einsum("sd,ak->sadk", linear_q_state_features, np.eye(n_actions, dtype=np.float64)).reshape(
        n_states, n_actions, -1
    )
    state_coef = np.array(
        [0.0, 0.55, 0.35, 0.50] + [0.15] * min(n_spokes, 4) + [1.0, 1.6],
        dtype=np.float64,
    )
    if state_coef.shape[0] != linear_q_state_features.shape[1]:
        state_coef = np.pad(state_coef, (0, linear_q_state_features.shape[1] - state_coef.shape[0]))
    theta_star = np.concatenate(
        [
            config.reward_scale * state_coef,
            config.reward_scale * (state_coef - np.array([config.action_gap] + [0.0] * (len(state_coef) - 1))),
        ]
    )
    q_target = base_phi_grid @ theta_star
    v_target = np.sum(target_policy * q_target, axis=1)
    expected_next_v = np.einsum("sak,k->sa", transition_prob, v_target)
    rewards = q_target - config.reward_gamma * expected_next_v

    q_feature = q_target.reshape(n_states, n_actions, 1)
    q_feature_scale = np.sqrt(np.mean(q_feature**2))
    q_feature = q_feature / max(q_feature_scale, 1e-12)
    phi_grid = np.concatenate([base_phi_grid, q_feature], axis=2)
    linear_q_state_action_features = phi_grid.reshape(n_states * n_actions, -1)

    mdp = DiscreteMDP(
        transition_prob=transition_prob,
        rewards=rewards,
        target_policy=target_policy,
        behavior_policy=behavior_policy,
    )

    rho0 = np.zeros(n_states, dtype=np.float64)
    start_depth = max(0, config.spoke_depth - 2)
    start_states = np.concatenate([levels[start_depth] for levels in spoke_level_state_ids]).astype(np.int64)
    rho0[start_states] = 1.0 / len(start_states)
    critical_state_mask = np.isin(np.arange(n_states, dtype=np.int64), np.concatenate([np.asarray(hub_ids), np.asarray(goal_ids)]))

    next_target_feature_expectation = np.einsum("ka,kad->kd", target_policy, phi_grid)
    next_phi_expectation = np.einsum("sak,kd->sad", transition_prob, next_target_feature_expectation)
    bellman_incompleteness = _projection_rmse(
        next_phi_expectation.reshape(n_states * n_actions, -1),
        phi_grid.reshape(n_states * n_actions, -1),
    )
    linear_realizability = _projection_rmse(q_target.reshape(-1, 1), phi_grid.reshape(n_states * n_actions, -1))

    return HubSpokeLatentBenchmark(
        config=config,
        mdp=mdp,
        observed_state_features=observed,
        linear_basic_state_features=linear_basic,
        flexible_linear_state_features=flexible_linear,
        linear_q_state_features=linear_q_state_features,
        neural_structured_state_features=neural_structured_state_features,
        linear_q_state_action_features=linear_q_state_action_features,
        critical_state_mask=critical_state_mask,
        initial_state_distribution=rho0,
        target_policy_logits=np.log(np.maximum(target_policy, 1e-12)),
        distractor_policy_logits=np.log(np.maximum(behavior_policy, 1e-12)),
        design_diagnostics={
            "linear_realizability_rmse": float(linear_realizability),
            "bellman_incompleteness_rmse": float(bellman_incompleteness),
        },
    )
