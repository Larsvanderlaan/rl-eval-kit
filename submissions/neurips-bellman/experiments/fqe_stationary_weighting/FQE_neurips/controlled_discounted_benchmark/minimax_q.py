from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from FQE_neurips.utils import TransitionBatch

from .features import RatioFeatureMap, StateActionFeatureMap
from .policies import GaussianLinearPolicy


@dataclass
class MinimaxQOutput:
    theta: np.ndarray
    q_function: object
    diagnostics: dict[str, float | int | str]


def _solve_stable(matrix: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(matrix, rhs, rcond=None)[0]


def _minimax_matrices(
    batch: TransitionBatch,
    *,
    feature_map: object,
    critic_feature_map: RatioFeatureMap,
    target_policy: GaussianLinearPolicy,
    gamma: float,
    sample_weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    phi = feature_map.transform(batch.states, batch.actions)
    phi_next = feature_map.expected_features_given_state(batch.next_states, target_policy)
    critic = critic_feature_map.transform(batch.states, batch.actions)
    rewards = np.asarray(batch.rewards, dtype=np.float64).reshape(-1)
    delta_phi = phi - float(gamma) * phi_next
    n = max(phi.shape[0], 1)
    if sample_weights is None:
        weights = np.ones(n, dtype=np.float64)
    else:
        weights = np.maximum(np.asarray(sample_weights, dtype=np.float64).reshape(-1), 1e-12)
        weights = weights / np.maximum(float(np.mean(weights)), 1e-12)
    moment_matrix = (critic.T @ (weights[:, None] * delta_phi)) / n
    reward_moment = (critic.T @ (weights * rewards)) / n
    critic_gram = (critic.T @ (weights[:, None] * critic)) / n
    return moment_matrix, reward_moment, critic_gram, critic


def _moment_risk(
    theta: np.ndarray,
    *,
    batch: TransitionBatch,
    feature_map: object,
    critic_feature_map: RatioFeatureMap,
    target_policy: GaussianLinearPolicy,
    gamma: float,
    critic_ridge: float,
    sample_weights: np.ndarray | None = None,
) -> float:
    moment_matrix, reward_moment, critic_gram, _critic = _minimax_matrices(
        batch,
        feature_map=feature_map,
        critic_feature_map=critic_feature_map,
        target_policy=target_policy,
        gamma=gamma,
        sample_weights=sample_weights,
    )
    moment = moment_matrix @ theta - reward_moment
    h = critic_gram + float(critic_ridge) * np.eye(critic_gram.shape[0], dtype=np.float64)
    return float(moment @ _solve_stable(h, moment))


def fit_minimax_linear_q(
    batch: TransitionBatch,
    *,
    feature_map: object,
    critic_feature_map: RatioFeatureMap,
    target_policy: GaussianLinearPolicy,
    gamma: float,
    q_ridge: float,
    critic_ridge: float,
    sample_weights: np.ndarray | None = None,
    initial_feature_expectation: np.ndarray | None = None,
    occupancy_gamma: float | None = None,
) -> MinimaxQOutput:
    """Closed-form minimax Bellman Q estimator.

    The Q/actor class is the same linear feature class used by affine FQE.  The
    adversarial critic is the same finite RBF-polynomial feature class used for
    stationary-ratio moment fitting.  The estimator minimizes the squared
    Bellman moment in the critic norm with Tikhonov regularization,

        min_theta || E_n[psi(S,A){phi(S,A)-gamma E_pi phi(S',A')}^T theta
                    - E_n[psi(S,A)R] ||_{H^{-1}}^2
                  + lambda ||theta||_2^2,

    where H is the empirical critic Gram matrix plus critic ridge.

    If `initial_feature_expectation` is supplied, the objective adds the
    DICE-style discounted occupancy value term
    `(1 - occupancy_gamma) E_{S0,A0~pi}[phi(S0,A0)]^T theta`.  Then the dual
    critic is the occupancy-ratio-like object rather than an externally
    supplied set of sample weights.
    """

    moment_matrix, reward_moment, critic_gram, _critic = _minimax_matrices(
        batch,
        feature_map=feature_map,
        critic_feature_map=critic_feature_map,
        target_policy=target_policy,
        gamma=gamma,
        sample_weights=sample_weights,
    )
    h = critic_gram + float(critic_ridge) * np.eye(critic_gram.shape[0], dtype=np.float64)
    h_inv_g = _solve_stable(h, moment_matrix)
    h_inv_b = _solve_stable(h, reward_moment)
    if initial_feature_expectation is None:
        initial_term = np.zeros(moment_matrix.shape[1], dtype=np.float64)
        effective_occupancy_gamma = np.nan
    else:
        if occupancy_gamma is None:
            raise ValueError("occupancy_gamma is required with initial_feature_expectation.")
        initial_term = (1.0 - float(occupancy_gamma)) * np.asarray(
            initial_feature_expectation,
            dtype=np.float64,
        ).reshape(moment_matrix.shape[1])
        effective_occupancy_gamma = float(occupancy_gamma)
    system = moment_matrix.T @ h_inv_g + float(q_ridge) * np.eye(
        moment_matrix.shape[1],
        dtype=np.float64,
    )
    rhs = moment_matrix.T @ h_inv_b - initial_term
    theta = _solve_stable(system, rhs)
    moment = moment_matrix @ theta - reward_moment
    moment_norm = float(moment @ _solve_stable(h, moment))
    try:
        critic_condition = float(np.linalg.cond(h))
    except np.linalg.LinAlgError:
        critic_condition = float("inf")
    try:
        primal_condition = float(np.linalg.cond(system))
    except np.linalg.LinAlgError:
        primal_condition = float("inf")
    if hasattr(feature_map, "quadratic_form_from_theta"):
        q_function = feature_map.quadratic_form_from_theta(theta)
    else:
        q_function = feature_map.function_from_theta(theta)
    return MinimaxQOutput(
        theta=theta.copy(),
        q_function=q_function,
        diagnostics={
            "solver": "minimax_q_rbf_critic",
            "moment_violation_l2": float(np.linalg.norm(moment)),
            "critic_norm_moment_risk": moment_norm,
            "normalization_error": 0.0,
            "q_ridge": float(q_ridge),
            "critic_ridge": float(critic_ridge),
            "occupancy_gamma": effective_occupancy_gamma,
            "initial_value_feature_norm": float(np.linalg.norm(initial_term)),
            "critic_condition_number": critic_condition,
            "primal_condition_number": primal_condition,
            "critic_dimension": int(moment_matrix.shape[0]),
            "q_dimension": int(moment_matrix.shape[1]),
        },
    )


def select_cv_minimax_linear_q(
    batch: TransitionBatch,
    *,
    feature_map: StateActionFeatureMap,
    critic_feature_map: RatioFeatureMap,
    target_policy: GaussianLinearPolicy,
    gamma: float,
    candidate_ridges: Sequence[float],
    seed: int,
    n_folds: int = 3,
) -> MinimaxQOutput:
    """Select a shared primal/critic ridge by held-out Bellman moment risk."""

    n = batch.states.shape[0]
    rng = np.random.default_rng(seed)
    folds = [
        np.asarray(fold, dtype=np.int64)
        for fold in np.array_split(rng.permutation(n), max(2, min(int(n_folds), n)))
        if len(fold) > 0
    ]
    all_idx = np.arange(n)
    ridge_stats: list[dict[str, float]] = []
    for ridge in candidate_ridges:
        fold_losses: list[float] = []
        for val_idx in folds:
            train_mask = np.ones(n, dtype=bool)
            train_mask[val_idx] = False
            train_idx = all_idx[train_mask]
            train_batch = TransitionBatch(
                states=batch.states[train_idx],
                actions=batch.actions[train_idx],
                rewards=batch.rewards[train_idx],
                next_states=batch.next_states[train_idx],
                next_actions=batch.next_actions[train_idx],
            )
            val_batch = TransitionBatch(
                states=batch.states[val_idx],
                actions=batch.actions[val_idx],
                rewards=batch.rewards[val_idx],
                next_states=batch.next_states[val_idx],
                next_actions=batch.next_actions[val_idx],
            )
            fit = fit_minimax_linear_q(
                train_batch,
                feature_map=feature_map,
                critic_feature_map=critic_feature_map,
                target_policy=target_policy,
                gamma=gamma,
                q_ridge=float(ridge),
                critic_ridge=float(ridge),
                sample_weights=None,
            )
            fold_losses.append(
                _moment_risk(
                    fit.theta,
                    batch=val_batch,
                    feature_map=feature_map,
                    critic_feature_map=critic_feature_map,
                    target_policy=target_policy,
                    gamma=gamma,
                    critic_ridge=float(ridge),
                )
            )
        losses = np.asarray(fold_losses, dtype=np.float64)
        ridge_stats.append(
            {
                "ridge": float(ridge),
                "mean": float(np.mean(losses)),
                "se": float(np.std(losses, ddof=1) / np.sqrt(max(len(losses), 1)))
                if len(losses) > 1
                else 0.0,
            }
        )
    ridge_stats.sort(key=lambda item: (item["mean"], item["ridge"]))
    best = ridge_stats[0]
    final = fit_minimax_linear_q(
        batch,
        feature_map=feature_map,
        critic_feature_map=critic_feature_map,
        target_policy=target_policy,
        gamma=gamma,
        q_ridge=float(best["ridge"]),
        critic_ridge=float(best["ridge"]),
    )
    final.diagnostics.update(
        {
            "solver": "minimax_q_rbf_critic_cv_tikhonov",
            "cv_selected_ridge": float(best["ridge"]),
            "cv_selected_ridge_min": float(best["ridge"]),
            "cv_selected_ridge_one_se": float("nan"),
            "cv_validation_ratio_moment_risk": float(best["mean"]),
            "cv_validation_ratio_moment_risk_se": float(best["se"]),
            "cv_n_folds": len(folds),
        }
    )
    return final
