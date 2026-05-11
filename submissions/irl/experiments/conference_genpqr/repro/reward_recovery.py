"""Reward and continuation recovery utilities."""

from __future__ import annotations

from typing import Callable, Dict

import numpy as np

from utils import EPS


PolicyFunction = Callable[[np.ndarray], np.ndarray]
NormalizerFunction = Callable[[np.ndarray], np.ndarray]


def recover_reward_and_continuation(
    policy_estimate,
    q_estimate,
    normalization_policy: PolicyFunction,
    normalization_function: NormalizerFunction,
    states: np.ndarray,
    actions: np.ndarray,
    gamma: float,
) -> Dict[str, np.ndarray]:
    """Recover reward and continuation terms using the GenPQR formulas.

    With ``u(s, a) = log pi(a | s)`` and ``Q`` denoting the fitted
    ``Q^mu_{u-g}``, the paper's recovery formulas are

        r(s, a) = Q(s, a) - sum_a' mu(a' | s) Q(s, a') + g(s)
        v(s, a) = (u(s, a) - g(s) - Q(s, a)) / gamma.
    """
    states = np.asarray(states, dtype=float)
    actions = np.asarray(actions, dtype=int).reshape(-1)

    q_matrix = q_estimate.predict_all_actions(states)
    q_sa = q_matrix[np.arange(states.shape[0]), actions]
    mu_probs = np.asarray(normalization_policy(states), dtype=float)
    mu_q = np.sum(mu_probs * q_matrix, axis=1)
    g_values = np.asarray(normalization_function(states), dtype=float).reshape(-1)

    reward_matrix = q_matrix - mu_q[:, None] + g_values[:, None]
    reward = reward_matrix[np.arange(states.shape[0]), actions]

    policy_probs = np.clip(policy_estimate.predict_proba(states), EPS, 1.0)
    u_matrix = np.log(policy_probs)
    u_sa = u_matrix[np.arange(states.shape[0]), actions]
    continuation = (u_sa - g_values - q_sa) / max(gamma, EPS)
    return {
        "reward": reward,
        "continuation_value": continuation,
        "q_sa": q_sa,
        "mu_q": mu_q,
        "reward_matrix": reward_matrix,
        "u_sa": u_sa,
    }
