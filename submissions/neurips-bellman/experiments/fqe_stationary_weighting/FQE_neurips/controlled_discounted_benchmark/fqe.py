from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from FQE_neurips.fqe_linear import LinearFQEConfig, LinearFQEResult
from FQE_neurips.utils import TransitionBatch

from .configs import FQESolverConfig
from .features import QuadraticStateActionFunction, StateActionFeatureMap
from .policies import GaussianLinearPolicy


@dataclass
class LinearFQEOutput:
    theta: np.ndarray
    backend_result: LinearFQEResult
    q_function: object
    feature_map: object


def build_linear_fqe_config(config: FQESolverConfig, gamma: float) -> LinearFQEConfig:
    return LinearFQEConfig(
        solver="iterative",
        gamma=gamma,
        ridge=config.ridge,
        n_outer_iters=config.n_outer_iters,
        target_update_tau=config.target_update_tau,
        valid_fraction=config.valid_fraction,
        early_stopping_patience=None,
        min_improvement=1e-8,
        tol=1e-10,
        use_averaging=False,
        selection_mode="last_iter",
        initial_theta_mode="zero",
        reduce_rank=False,
    )


def fit_linear_fqe(
    batch: TransitionBatch,
    *,
    feature_map: object,
    target_policy: GaussianLinearPolicy,
    gamma: float,
    solver_config: FQESolverConfig,
    sample_weights: np.ndarray | None,
    seed: int,
) -> LinearFQEOutput:
    state_action_features = feature_map.transform(batch.states, batch.actions)
    next_expected_features = feature_map.expected_features_given_state(batch.next_states, target_policy)
    rewards = np.asarray(batch.rewards, dtype=np.float64).reshape(-1)
    phi = np.asarray(state_action_features, dtype=np.float64)
    phi_next = np.asarray(next_expected_features, dtype=np.float64)
    if sample_weights is None:
        weights = np.ones(phi.shape[0], dtype=np.float64)
    else:
        weights = np.maximum(np.asarray(sample_weights, dtype=np.float64).reshape(-1), 1e-12)
        weights = weights / np.maximum(np.mean(weights), 1e-12)

    gram = phi.T @ (weights[:, None] * phi)
    gram = gram + solver_config.ridge * np.eye(phi.shape[1], dtype=np.float64)
    reward_rhs = phi.T @ (weights * rewards)
    next_rhs = phi.T @ (weights[:, None] * phi_next)

    theta = np.zeros(phi.shape[1], dtype=np.float64)
    target_theta = theta.copy()
    history = {
        "train_loss": [],
        "valid_loss": [],
        "bellman_residual": [],
        "parameter_change": [],
    }
    for _ in range(solver_config.n_outer_iters):
        rhs = reward_rhs + gamma * (next_rhs @ target_theta)
        try:
            theta_new = np.linalg.solve(gram, rhs)
        except np.linalg.LinAlgError:
            theta_new = np.linalg.lstsq(gram, rhs, rcond=None)[0]
        residual = phi @ theta_new - (rewards + gamma * (phi_next @ theta_new))
        train_loss = float(np.mean(weights * residual**2))
        parameter_change = float(np.linalg.norm(theta_new - theta))
        history["train_loss"].append(train_loss)
        history["valid_loss"].append(train_loss)
        history["bellman_residual"].append(train_loss)
        history["parameter_change"].append(parameter_change)
        theta = theta_new
        target_theta = (
            (1.0 - solver_config.target_update_tau) * target_theta
            + solver_config.target_update_tau * theta_new
        )
        if parameter_change < solver_config.ridge * 1e-6:
            break

    result = LinearFQEResult(
        theta=theta.copy(),
        history=history,
        theta_iterates=None,
        selected_iteration=len(history["train_loss"]) - 1,
        final_theta=theta.copy(),
    )
    if hasattr(feature_map, "quadratic_form_from_theta"):
        q_function = feature_map.quadratic_form_from_theta(theta)
    else:
        q_function = feature_map.function_from_theta(theta)
    return LinearFQEOutput(
        theta=theta.copy(),
        backend_result=result,
        q_function=q_function,
        feature_map=feature_map,
    )
