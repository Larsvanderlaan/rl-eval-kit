from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SoftValueIterationResult:
    q: np.ndarray
    v: np.ndarray
    n_iters: int
    sup_delta: float
    converged: bool


def soft_value(q_values: np.ndarray, tau: float) -> np.ndarray:
    q = np.asarray(q_values, dtype=np.float64)
    scaled = q / float(tau)
    max_scaled = np.max(scaled, axis=1, keepdims=True)
    return float(tau) * (
        max_scaled[:, 0] + np.log(np.maximum(np.sum(np.exp(scaled - max_scaled), axis=1), 1e-300))
    )


def softmax_policy(q_values: np.ndarray, tau: float) -> np.ndarray:
    q = np.asarray(q_values, dtype=np.float64)
    scaled = q / float(tau)
    scaled -= np.max(scaled, axis=1, keepdims=True)
    exp_q = np.exp(np.clip(scaled, -745.0, 80.0))
    denom = np.maximum(np.sum(exp_q, axis=1, keepdims=True), 1e-300)
    return exp_q / denom


def bellman_operator(transition: np.ndarray, reward: np.ndarray, q_values: np.ndarray, gamma: float, tau: float) -> np.ndarray:
    v_next = soft_value(q_values, tau)
    expected_next = transition.reshape(-1, transition.shape[-1]) @ v_next
    return reward + float(gamma) * expected_next.reshape(reward.shape)


def soft_value_iteration(
    transition: np.ndarray,
    reward: np.ndarray,
    gamma: float,
    tau: float,
    *,
    tol: float = 1e-10,
    max_iter: int = 20_000,
) -> SoftValueIterationResult:
    q = np.zeros_like(reward, dtype=np.float64)
    sup_delta = float("inf")
    converged = False
    for iteration in range(1, max_iter + 1):
        q_new = bellman_operator(transition, reward, q, gamma, tau)
        sup_delta = float(np.max(np.abs(q_new - q)))
        q = q_new
        if sup_delta < tol:
            converged = True
            break
    return SoftValueIterationResult(
        q=q,
        v=soft_value(q, tau),
        n_iters=iteration,
        sup_delta=sup_delta,
        converged=converged,
    )


def policy_transition(transition: np.ndarray, policy: np.ndarray) -> np.ndarray:
    return np.einsum("sa,sat->st", policy, transition, optimize=True)


def stationary_state_distribution(
    transition: np.ndarray,
    policy: np.ndarray,
    *,
    tol: float = 1e-12,
    max_iter: int = 100_000,
) -> tuple[np.ndarray, float, bool]:
    p_policy = policy_transition(transition, policy)
    dist = np.ones(policy.shape[0], dtype=np.float64) / policy.shape[0]
    residual = float("inf")
    converged = False
    for _ in range(max_iter):
        next_dist = dist @ p_policy
        residual = float(np.max(np.abs(next_dist - dist)))
        dist = next_dist
        if residual < tol:
            converged = True
            break
    dist = np.maximum(dist, 0.0)
    dist /= np.maximum(dist.sum(), 1e-300)
    residual = float(np.max(np.abs(dist @ p_policy - dist)))
    return dist, residual, converged


def state_action_distribution(state_dist: np.ndarray, policy: np.ndarray) -> np.ndarray:
    dist = np.asarray(state_dist, dtype=np.float64).reshape(-1, 1) * np.asarray(policy, dtype=np.float64)
    dist = np.maximum(dist, 0.0)
    return dist / np.maximum(dist.sum(), 1e-300)


def evaluate_soft_policy_value(
    transition: np.ndarray,
    reward: np.ndarray,
    policy: np.ndarray,
    gamma: float,
    tau: float,
    rho0: np.ndarray,
) -> float:
    p_policy = policy_transition(transition, policy)
    entropy = -np.sum(policy * np.log(np.maximum(policy, 1e-300)), axis=1)
    r_policy = np.sum(policy * reward, axis=1) + float(tau) * entropy
    system = np.eye(policy.shape[0], dtype=np.float64) - float(gamma) * p_policy
    try:
        value = np.linalg.solve(system, r_policy)
    except np.linalg.LinAlgError:
        value = np.linalg.lstsq(system, r_policy, rcond=None)[0]
    return float(np.asarray(rho0, dtype=np.float64).reshape(-1) @ value)


def discounted_state_distribution(
    transition: np.ndarray,
    policy: np.ndarray,
    rho0: np.ndarray,
    gamma_weight: float,
) -> np.ndarray:
    rho0_arr = np.asarray(rho0, dtype=np.float64).reshape(-1)
    rho0_arr = rho0_arr / np.maximum(rho0_arr.sum(), 1e-300)
    if gamma_weight >= 1.0 - 1e-12:
        dist, _resid, _conv = stationary_state_distribution(transition, policy)
        return dist
    p_policy = policy_transition(transition, policy)
    system = np.eye(policy.shape[0], dtype=np.float64) - float(gamma_weight) * p_policy.T
    rhs = (1.0 - float(gamma_weight)) * rho0_arr
    try:
        dist = np.linalg.solve(system, rhs)
    except np.linalg.LinAlgError:
        dist = np.linalg.lstsq(system, rhs, rcond=None)[0]
    dist = np.maximum(dist, 0.0)
    return dist / np.maximum(dist.sum(), 1e-300)
