from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class TransitionBatch:
    """Offline transition dataset with next actions sampled from the target policy."""

    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray
    next_actions: np.ndarray

    def __len__(self) -> int:
        return int(self.states.shape[0])


def set_random_seed(seed: int) -> np.random.Generator:
    """Create a reproducible NumPy generator."""

    return np.random.default_rng(seed)


def one_hot(indices: np.ndarray, size: int) -> np.ndarray:
    """Return a dense one-hot matrix."""

    indices = np.asarray(indices, dtype=np.int64).reshape(-1)
    out = np.zeros((indices.shape[0], size), dtype=np.float64)
    out[np.arange(indices.shape[0]), indices] = 1.0
    return out


def state_action_one_hot(
    states: np.ndarray,
    actions: np.ndarray,
    n_states: int,
    n_actions: int,
) -> np.ndarray:
    """Tabular state-action basis, useful for ratio estimation or diagnostics."""

    states = np.asarray(states, dtype=np.int64).reshape(-1)
    actions = np.asarray(actions, dtype=np.int64).reshape(-1)
    indices = states * n_actions + actions
    return one_hot(indices, n_states * n_actions)


def clip_normalize_weights(
    weights: np.ndarray,
    min_weight: float = 1e-8,
    max_weight: Optional[float] = None,
) -> np.ndarray:
    """Clip weights for stability and normalize them to empirical mean one."""

    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if max_weight is None:
        w = np.maximum(w, min_weight)
    else:
        w = np.clip(w, min_weight, max_weight)
    mean_w = float(w.mean())
    if mean_w <= 0.0:
        raise ValueError("Weights must have positive empirical mean after clipping.")
    return w / mean_w


def effective_sample_size(weights: np.ndarray) -> float:
    """Effective sample size of a nonnegative weight vector."""

    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    return float((w.sum() ** 2) / max(np.sum(w**2), 1e-12))


def _apply_uniform_mix(weights: np.ndarray, uniform_mix: float) -> np.ndarray:
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if uniform_mix <= 0.0:
        return w / np.mean(w)
    mixed = (1.0 - uniform_mix) * w + uniform_mix * np.ones_like(w)
    return mixed / np.mean(mixed)


def interpolate_with_uniform(weights: np.ndarray, weight_strength: float) -> np.ndarray:
    """
    Interpolate between unweighted and fully weighted samples.

    - `weight_strength = 0` gives the uniform weight vector.
    - `weight_strength = 1` gives the input weights after mean normalization.
    """

    strength = float(np.clip(weight_strength, 0.0, 1.0))
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    w = w / np.mean(w)
    mixed = (1.0 - strength) * np.ones_like(w) + strength * w
    return mixed / np.mean(mixed)


def stabilize_weights(
    weights: np.ndarray,
    min_weight: float = 1e-8,
    max_weight: Optional[float] = 20.0,
    clip_quantile: float | None = 0.995,
    uniform_mix: float = 0.02,
    target_ess_fraction: float | None = 0.4,
    max_uniform_mix: float = 0.5,
    return_metadata: bool = False,
) -> tuple[np.ndarray, Optional[float]] | tuple[np.ndarray, dict[str, float | None]]:
    """
    Stabilize importance / density-ratio weights for downstream FQE.

    Processing order:
    1. enforce positivity,
    2. cap very large weights using an optional quantile-based threshold and/or hard max,
    3. normalize to mean one,
    4. optionally increase shrinkage toward uniform until a target ESS fraction is reached.
    """

    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    upper = None
    if clip_quantile is not None:
        upper = float(np.quantile(w, clip_quantile))
    if max_weight is not None:
        upper = min(max_weight, upper) if upper is not None else max_weight
    w = clip_normalize_weights(w, min_weight=min_weight, max_weight=upper)
    ess_before = effective_sample_size(w) / max(len(w), 1)

    chosen_uniform_mix = float(max(uniform_mix, 0.0))
    if target_ess_fraction is not None and ess_before < target_ess_fraction:
        lo = chosen_uniform_mix
        hi = max(lo, max_uniform_mix)
        if effective_sample_size(_apply_uniform_mix(w, hi)) / max(len(w), 1) < target_ess_fraction and hi < 1.0:
            hi = 1.0
        for _ in range(30):
            mid = 0.5 * (lo + hi)
            cand = _apply_uniform_mix(w, mid)
            ess_mid = effective_sample_size(cand) / max(len(cand), 1)
            if ess_mid >= target_ess_fraction:
                hi = mid
            else:
                lo = mid
        chosen_uniform_mix = hi

    w = _apply_uniform_mix(w, chosen_uniform_mix)
    ess_after = effective_sample_size(w) / max(len(w), 1)

    if return_metadata:
        return w, {
            "effective_max_weight": None if upper is None else float(upper),
            "chosen_uniform_mix": float(chosen_uniform_mix),
            "ess_fraction_before_mix": float(ess_before),
            "ess_fraction_after_mix": float(ess_after),
            "target_ess_fraction": None if target_ess_fraction is None else float(target_ess_fraction),
        }
    return w, upper


def train_valid_split(
    n: int,
    valid_fraction: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Simple random train/validation split."""

    indices = np.arange(n)
    if valid_fraction <= 0.0:
        return indices, np.empty(0, dtype=np.int64)
    perm = rng.permutation(indices)
    n_valid = max(1, int(round(valid_fraction * n)))
    valid_idx = np.sort(perm[:n_valid])
    train_idx = np.sort(perm[n_valid:])
    return train_idx, valid_idx


@dataclass
class DiscreteMDP:
    """Small discrete MDP container for the end-to-end experiment."""

    transition_prob: np.ndarray
    rewards: np.ndarray
    target_policy: np.ndarray
    behavior_policy: np.ndarray

    @property
    def n_states(self) -> int:
        return int(self.transition_prob.shape[0])

    @property
    def n_actions(self) -> int:
        return int(self.transition_prob.shape[1])


def sample_actions(policy: np.ndarray, states: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Sample actions from a discrete stochastic policy."""

    probs = np.asarray(policy, dtype=np.float64)[np.asarray(states, dtype=np.int64)]
    cdf = np.cumsum(probs, axis=1)
    draws = rng.random(size=probs.shape[0])[:, None]
    return (draws > cdf[:, :-1]).sum(axis=1).astype(np.int64)


def sample_next_states(
    transition_prob: np.ndarray,
    states: np.ndarray,
    actions: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample next states from a discrete transition kernel."""

    probs = transition_prob[np.asarray(states, dtype=np.int64), np.asarray(actions, dtype=np.int64)]
    cdf = np.cumsum(probs, axis=1)
    draws = rng.random(size=probs.shape[0])[:, None]
    return (draws > cdf[:, :-1]).sum(axis=1).astype(np.int64)


def stationary_distribution(transition_matrix: np.ndarray, n_iters: int = 10_000) -> np.ndarray:
    """Power-iteration estimate of a stationary state distribution."""

    n_states = transition_matrix.shape[0]
    dist = np.ones(n_states, dtype=np.float64) / n_states
    for _ in range(n_iters):
        new_dist = dist @ transition_matrix
        if np.max(np.abs(new_dist - dist)) < 1e-12:
            break
        dist = new_dist
    return dist / dist.sum()


def induced_state_transition(mdp: DiscreteMDP, policy: np.ndarray) -> np.ndarray:
    """State-only transition matrix induced by a policy."""

    return np.einsum("sa,san->sn", policy, mdp.transition_prob)


def induced_state_action_transition(mdp: DiscreteMDP, policy: np.ndarray) -> np.ndarray:
    """State-action transition matrix induced by the environment and a policy."""

    n_states, n_actions = mdp.n_states, mdp.n_actions
    n_sa = n_states * n_actions
    transition = np.zeros((n_sa, n_sa), dtype=np.float64)
    for s in range(n_states):
        for a in range(n_actions):
            row = s * n_actions + a
            for sp in range(n_states):
                p_sp = mdp.transition_prob[s, a, sp]
                if p_sp == 0.0:
                    continue
                for ap in range(n_actions):
                    col = sp * n_actions + ap
                    transition[row, col] += p_sp * policy[sp, ap]
    return transition


def stationary_state_action_distribution(mdp: DiscreteMDP, policy: np.ndarray) -> np.ndarray:
    """Exact stationary distribution over state-action pairs for a discrete policy."""

    state_dist = stationary_distribution(induced_state_transition(mdp, policy))
    sa_dist = (state_dist[:, None] * policy).reshape(-1)
    return sa_dist / sa_dist.sum()


def exact_ratio_against_behavior(
    mdp: DiscreteMDP,
    gamma_ratio: float,
) -> np.ndarray:
    """
    Exact stationary or discounted occupancy ratio relative to the behavior stationary distribution.

    - gamma_ratio = 1 gives the stationary target ratio mu_pi / nu_b.
    - gamma_ratio < 1 gives the discounted resolvent ratio for base measure nu_b.
    """

    nu_b = stationary_state_action_distribution(mdp, mdp.behavior_policy)
    if gamma_ratio == 1.0:
        target = stationary_state_action_distribution(mdp, mdp.target_policy)
    else:
        transition_target = induced_state_action_transition(mdp, mdp.target_policy)
        system = np.eye(transition_target.shape[0]) - gamma_ratio * transition_target.T
        rhs = (1.0 - gamma_ratio) * nu_b
        target = np.linalg.solve(system, rhs)
        target = np.maximum(target, 0.0)
        target = target / target.sum()
    ratio = target / np.maximum(nu_b, 1e-12)
    return ratio


def evaluate_policy_tabular(mdp: DiscreteMDP, gamma: float) -> np.ndarray:
    """Solve the exact action-value function for a discrete target policy."""

    n_states, n_actions = mdp.n_states, mdp.n_actions
    n_sa = n_states * n_actions
    bellman_matrix = np.eye(n_sa, dtype=np.float64)
    rhs = mdp.rewards.reshape(-1).astype(np.float64).copy()

    for s in range(n_states):
        for a in range(n_actions):
            row = s * n_actions + a
            for sp in range(n_states):
                p_sp = mdp.transition_prob[s, a, sp]
                if p_sp == 0.0:
                    continue
                for ap in range(n_actions):
                    bellman_matrix[row, sp * n_actions + ap] -= gamma * p_sp * mdp.target_policy[sp, ap]

    q_star = np.linalg.solve(bellman_matrix, rhs)
    return q_star.reshape(n_states, n_actions)
