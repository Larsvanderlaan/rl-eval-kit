from __future__ import annotations

import numpy as np

from .env import GridMDP
from .features import QFeatureMap, linear_q_features
from .soft_dp import bellman_operator, evaluate_soft_policy_value, softmax_policy


def weighted_rmse(values: np.ndarray, weights: np.ndarray) -> float:
    vals = np.asarray(values, dtype=np.float64).reshape(-1)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    w = w / np.maximum(w.sum(), 1e-300)
    return float(np.sqrt(np.sum(w * vals * vals)))


def advantage_centered_rmse(diff: np.ndarray, state_action_dist: np.ndarray) -> float:
    d = np.asarray(diff, dtype=np.float64).reshape(state_action_dist.shape)
    dist = np.asarray(state_action_dist, dtype=np.float64)
    state_dist = np.sum(dist, axis=1)
    policy = dist / np.maximum(state_dist[:, None], 1e-300)
    centered = d - np.sum(policy * d, axis=1, keepdims=True)
    return weighted_rmse(centered.reshape(-1), dist.reshape(-1))


def weighted_design_condition_number(features: np.ndarray, weights: np.ndarray, ridge: float) -> float:
    x = np.asarray(features, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    w = w / np.maximum(np.mean(w), 1e-300)
    gram = (x.T @ (w[:, None] * x)) / max(x.shape[0], 1) + float(ridge) * np.eye(x.shape[1])
    try:
        return float(np.linalg.cond(gram))
    except np.linalg.LinAlgError:
        return float("inf")


def _weighted_linear_projection(
    values: np.ndarray,
    *,
    mdp: GridMDP,
    weights: np.ndarray,
    ridge: float,
    q_feature_map: QFeatureMap | None = None,
) -> np.ndarray:
    state_ids, action_ids = np.meshgrid(np.arange(mdp.n_states), np.arange(mdp.n_actions), indexing="ij")
    flat_states = state_ids.reshape(-1)
    flat_actions = action_ids.reshape(-1)
    if q_feature_map is None:
        phi = linear_q_features(mdp.states[flat_states], flat_actions, mdp.actions)
    else:
        phi = q_feature_map.transform(mdp.states[flat_states], flat_actions)
    y = np.asarray(values, dtype=np.float64).reshape(-1)
    w = np.maximum(np.asarray(weights, dtype=np.float64).reshape(-1), 1e-12)
    w = w / np.maximum(np.mean(w), 1e-300)
    gram = (phi.T @ (w[:, None] * phi)) / max(phi.shape[0], 1)
    rhs = (phi.T @ (w * y)) / max(phi.shape[0], 1)
    system = gram + float(ridge) * np.eye(phi.shape[1], dtype=np.float64)
    try:
        theta = np.linalg.solve(system, rhs)
    except np.linalg.LinAlgError:
        theta = np.linalg.lstsq(system, rhs, rcond=None)[0]
    return (phi @ theta).reshape(mdp.n_states, mdp.n_actions)


def compute_q_metrics(
    q_values: np.ndarray,
    *,
    mdp: GridMDP,
    q_star: np.ndarray,
    target_sa_dist: np.ndarray,
    behavior_sa_dist: np.ndarray,
    gamma: float,
    tau_final: float,
    reference_value: float,
    rho0: np.ndarray,
    compute_value: bool,
    projection_ridge: float = 1e-8,
    q_feature_map: QFeatureMap | None = None,
) -> dict[str, float]:
    q = np.asarray(q_values, dtype=np.float64).reshape(mdp.n_states, mdp.n_actions)
    diff = q - q_star
    target_flat = target_sa_dist.reshape(-1)
    behavior_flat = behavior_sa_dist.reshape(-1)
    bellman = bellman_operator(mdp.transition, mdp.reward, q, gamma, tau_final)
    residual = bellman - q
    target_projected = _weighted_linear_projection(
        bellman,
        mdp=mdp,
        weights=target_flat,
        ridge=projection_ridge,
        q_feature_map=q_feature_map,
    )
    behavior_projected = _weighted_linear_projection(
        bellman,
        mdp=mdp,
        weights=behavior_flat,
        ridge=projection_ridge,
        q_feature_map=q_feature_map,
    )
    target_projected_residual = target_projected - q
    behavior_projected_residual = behavior_projected - q
    out = {
        "stationary_q_rmse": weighted_rmse(diff.reshape(-1), target_flat),
        "behavior_q_rmse": weighted_rmse(diff.reshape(-1), behavior_flat),
        "stationary_advantage_q_rmse": advantage_centered_rmse(diff, target_sa_dist),
        "behavior_advantage_q_rmse": advantage_centered_rmse(diff, behavior_sa_dist),
        "stationary_bellman_rmse": weighted_rmse(residual.reshape(-1), target_flat),
        "behavior_bellman_rmse": weighted_rmse(residual.reshape(-1), behavior_flat),
        "stationary_projected_bellman_rmse": weighted_rmse(target_projected_residual.reshape(-1), target_flat),
        "behavior_projected_bellman_rmse": weighted_rmse(behavior_projected_residual.reshape(-1), behavior_flat),
        "cross_behavior_projected_bellman_rmse": weighted_rmse(behavior_projected_residual.reshape(-1), target_flat),
        "cross_stationary_projected_bellman_rmse": weighted_rmse(target_projected_residual.reshape(-1), behavior_flat),
        "max_abs_q": float(np.max(np.abs(q))),
    }
    target_state_dist = np.sum(target_sa_dist, axis=1)
    behavior_state_dist = np.sum(behavior_sa_dist, axis=1)
    learned_action = np.argmax(q, axis=1)
    optimal_action = np.argmax(q_star, axis=1)
    action_match = (learned_action == optimal_action).astype(np.float64)
    out["stationary_optimal_action_agreement"] = float(np.sum(target_state_dist * action_match))
    out["behavior_optimal_action_agreement"] = float(np.sum(behavior_state_dist * action_match))
    out["norm_mismatch_ratio"] = float(out["stationary_q_rmse"] / max(out["behavior_q_rmse"], 1e-12))
    out["diverged"] = float((not np.all(np.isfinite(q))) or out["max_abs_q"] > 1e5)
    if compute_value:
        pi_q = softmax_policy(q, tau_final)
        value = evaluate_soft_policy_value(mdp.transition, mdp.reward, pi_q, gamma, tau_final, rho0)
        out["policy_value"] = value
        out["policy_value_error"] = float(reference_value - value)
    else:
        out["policy_value"] = float("nan")
        out["policy_value_error"] = float("nan")
    return out
