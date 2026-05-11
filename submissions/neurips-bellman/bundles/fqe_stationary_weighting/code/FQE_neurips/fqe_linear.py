from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .utils import TransitionBatch, clip_normalize_weights, train_valid_split


@dataclass
class LinearFQEConfig:
    """Configuration for weighted linear FQE with ridge regularization."""

    solver: str = "iterative"
    gamma: float = 0.99
    ridge: float = 1e-3
    n_outer_iters: int = 50
    target_update_tau: float = 0.3
    valid_fraction: float = 0.1
    early_stopping_patience: int | None = 8
    min_improvement: float = 1e-6
    tol: float = 1e-8
    use_averaging: bool = True
    averaging_start_iter: int = 5
    initial_theta_mode: str = "zero"
    random_init_scale: float = 1.0
    initial_theta: np.ndarray | None = None
    selection_mode: str = "best_valid"
    track_iterates: bool = False
    reduce_rank: bool = True
    rank_tol: float = 1e-10


@dataclass
class LinearFQEResult:
    """Output from weighted linear FQE."""

    theta: np.ndarray
    history: dict[str, list[float]]
    theta_iterates: np.ndarray | None = None
    selected_iteration: int = -1
    final_theta: np.ndarray | None = None


def _solve_projected_fixed_point(
    features: np.ndarray,
    next_features: np.ndarray,
    rewards: np.ndarray,
    weights: np.ndarray,
    gamma: float,
    ridge: float,
) -> np.ndarray:
    weighted_features = features * weights[:, None]
    lhs = features.T @ (weighted_features - gamma * (next_features * weights[:, None]))
    lhs = lhs + ridge * np.eye(features.shape[1], dtype=np.float64)
    rhs = features.T @ (weights * rewards)
    try:
        return np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(lhs, rhs, rcond=None)[0]


def _solve_weighted_ridge(
    features: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
    ridge: float,
) -> np.ndarray:
    weighted_features = features * weights[:, None]
    gram = features.T @ weighted_features
    gram = gram + ridge * np.eye(features.shape[1], dtype=np.float64)
    rhs = features.T @ (weights * targets)
    try:
        return np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(gram, rhs, rcond=None)[0]


def _bellman_residual_mse(
    theta: np.ndarray,
    rewards: np.ndarray,
    features: np.ndarray,
    next_features: np.ndarray,
    weights: np.ndarray,
    gamma: float,
) -> float:
    residual = features @ theta - (rewards + gamma * (next_features @ theta))
    return float(np.mean(weights * residual**2))


def _initialize_theta(config: LinearFQEConfig, dim: int, rng: np.random.Generator) -> np.ndarray:
    if config.initial_theta is not None:
        theta0 = np.asarray(config.initial_theta, dtype=np.float64).reshape(-1)
        if theta0.shape[0] != dim:
            raise ValueError("config.initial_theta must have the same dimension as the feature width.")
        return theta0.copy()
    if config.initial_theta_mode == "zero":
        return np.zeros(dim, dtype=np.float64)
    if config.initial_theta_mode == "random":
        return config.random_init_scale * rng.normal(size=dim)
    raise ValueError(f"Unsupported initial_theta_mode '{config.initial_theta_mode}'.")


def _reduce_feature_rank(
    features: np.ndarray,
    next_features: np.ndarray,
    rank_tol: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Collapse redundant feature directions while preserving the represented
    value-function class on the sampled design.

    If `features = U S V^T`, we work in the reduced coordinates
    `z = features @ T`, `z_next = next_features @ T` with
    `T = V_r / S_r`, so that `z = U_r`. A parameter vector `beta` in the
    reduced coordinates maps back to the original parameterization via
    `theta = T beta`.
    """

    u, s, vt = np.linalg.svd(features, full_matrices=False)
    keep = s > rank_tol
    if not np.any(keep):
        raise ValueError("State-action features are numerically rank zero.")
    transform = (vt[keep].T / s[keep])
    reduced = features @ transform
    reduced_next = next_features @ transform
    return reduced, reduced_next, transform


def _map_theta_to_reduced(theta: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Project an original-parameter vector into reduced coordinates."""

    return np.linalg.lstsq(transform, np.asarray(theta, dtype=np.float64), rcond=None)[0]


def fit_linear_fqe(
    batch: TransitionBatch,
    state_action_features: np.ndarray,
    next_state_action_features: np.ndarray,
    weights: np.ndarray | None = None,
    config: LinearFQEConfig | None = None,
    seed: int = 0,
) -> LinearFQEResult:
    """
    Fit weighted linear FQE by iterated ridge-regression Bellman updates.

    The basis is passed in explicitly so the caller can study misspecified and
    non-Bellman-complete feature maps. Stability comes from:
    - Tikhonov regularization via `ridge`,
    - damped target updates,
    - validation-based iterate selection,
    - optional iterate averaging.
    """

    if config is None:
        config = LinearFQEConfig()

    phi = np.asarray(state_action_features, dtype=np.float64)
    phi_next = np.asarray(next_state_action_features, dtype=np.float64)
    rewards = np.asarray(batch.rewards, dtype=np.float64).reshape(-1)
    if phi.ndim != 2 or phi_next.ndim != 2:
        raise ValueError("Feature arrays must be 2D.")
    if phi.shape != phi_next.shape:
        raise ValueError("Current and next feature arrays must have the same shape.")
    if phi.shape[0] != len(batch):
        raise ValueError("Feature arrays must have one row per transition.")

    if weights is None:
        sample_weights = np.ones(phi.shape[0], dtype=np.float64)
    else:
        # Weighted FQE should consume the caller's stabilized weights directly.
        # We only enforce positivity / mean normalization here rather than
        # reapplying ESS-based shrinkage inside the FQE solver.
        sample_weights = clip_normalize_weights(np.asarray(weights, dtype=np.float64))

    original_feature_dim = phi.shape[1]
    transform = np.eye(original_feature_dim, dtype=np.float64)
    if config.reduce_rank:
        phi, phi_next, transform = _reduce_feature_rank(phi, phi_next, rank_tol=config.rank_tol)

    rng = np.random.default_rng(seed)
    train_idx, valid_idx = train_valid_split(phi.shape[0], config.valid_fraction, rng)
    if train_idx.size == 0:
        train_idx = np.arange(phi.shape[0], dtype=np.int64)

    if config.solver == "direct":
        beta_direct = _solve_projected_fixed_point(
            features=phi[train_idx],
            next_features=phi_next[train_idx],
            rewards=rewards[train_idx],
            weights=sample_weights[train_idx],
            gamma=config.gamma,
            ridge=config.ridge,
        )
        valid_loss = _bellman_residual_mse(
            theta=beta_direct,
            rewards=rewards[valid_idx] if valid_idx.size > 0 else rewards[train_idx],
            features=phi[valid_idx] if valid_idx.size > 0 else phi[train_idx],
            next_features=phi_next[valid_idx] if valid_idx.size > 0 else phi_next[train_idx],
            weights=sample_weights[valid_idx] if valid_idx.size > 0 else sample_weights[train_idx],
            gamma=config.gamma,
        )
        theta_direct = transform @ beta_direct
        history = {
            "train_loss": [
                _bellman_residual_mse(
                    theta=beta_direct,
                    rewards=rewards[train_idx],
                    features=phi[train_idx],
                    next_features=phi_next[train_idx],
                    weights=sample_weights[train_idx],
                    gamma=config.gamma,
                )
            ],
            "valid_loss": [float(valid_loss)],
            "bellman_residual": [float(valid_loss)],
            "parameter_change": [0.0],
        }
        theta_iterates = theta_direct[None, :] if config.track_iterates else None
        return LinearFQEResult(
            theta=theta_direct.copy(),
            history=history,
            theta_iterates=theta_iterates,
            selected_iteration=0,
            final_theta=theta_direct.copy(),
        )
    if config.solver != "iterative":
        raise ValueError(f"Unsupported linear FQE solver '{config.solver}'.")

    theta0_orig = _initialize_theta(config, original_feature_dim, rng)
    beta = _map_theta_to_reduced(theta0_orig, transform)
    beta_target = beta.copy()
    beta_avg = np.zeros_like(beta)
    n_avg = 0
    best_beta = beta.copy()
    best_valid = float("inf")
    patience = 0
    selected_iteration = -1
    theta_path: list[np.ndarray] = []

    history: dict[str, list[float]] = {
        "train_loss": [],
        "valid_loss": [],
        "bellman_residual": [],
        "parameter_change": [],
    }

    for step in range(config.n_outer_iters):
        targets = rewards + config.gamma * (phi_next @ beta_target)
        beta_new = _solve_weighted_ridge(
            features=phi[train_idx],
            targets=targets[train_idx],
            weights=sample_weights[train_idx],
            ridge=config.ridge,
        )

        if config.use_averaging and step >= config.averaging_start_iter:
            n_avg += 1
            beta_avg = ((n_avg - 1) * beta_avg + beta_new) / max(n_avg, 1)
            beta_eval = beta_avg.copy()
        else:
            beta_eval = beta_new.copy()

        train_loss = _bellman_residual_mse(
            theta=beta_eval,
            rewards=rewards[train_idx],
            features=phi[train_idx],
            next_features=phi_next[train_idx],
            weights=sample_weights[train_idx],
            gamma=config.gamma,
        )
        if valid_idx.size > 0:
            valid_loss = _bellman_residual_mse(
                theta=beta_eval,
                rewards=rewards[valid_idx],
                features=phi[valid_idx],
                next_features=phi_next[valid_idx],
                weights=sample_weights[valid_idx],
                gamma=config.gamma,
            )
        else:
            valid_loss = train_loss

        parameter_change = float(np.linalg.norm(beta_new - beta))
        history["train_loss"].append(float(train_loss))
        history["valid_loss"].append(float(valid_loss))
        history["bellman_residual"].append(float(valid_loss))
        history["parameter_change"].append(parameter_change)
        if config.track_iterates:
            theta_path.append((transform @ beta_eval).copy())

        if valid_loss + config.min_improvement < best_valid:
            best_valid = float(valid_loss)
            best_beta = beta_eval.copy()
            selected_iteration = step
            patience = 0
        else:
            patience += 1

        beta = beta_new
        beta_target = (1.0 - config.target_update_tau) * beta_target + config.target_update_tau * beta_new

        should_stop_on_patience = config.early_stopping_patience is not None and patience >= config.early_stopping_patience
        if parameter_change < config.tol or should_stop_on_patience:
            break

    if config.selection_mode == "last_iter":
        selected_theta = transform @ beta_eval
        selected_iteration = len(history["train_loss"]) - 1
    elif config.selection_mode == "best_valid":
        selected_theta = transform @ best_beta
    else:
        raise ValueError(f"Unsupported selection_mode '{config.selection_mode}'.")

    return LinearFQEResult(
        theta=selected_theta,
        history=history,
        theta_iterates=np.stack(theta_path, axis=0) if theta_path else None,
        selected_iteration=selected_iteration,
        final_theta=(transform @ beta_eval).copy(),
    )


def fit_weighted_linear_fqe(
    batch: TransitionBatch,
    state_action_features: np.ndarray,
    next_state_action_features: np.ndarray,
    weights: np.ndarray | None = None,
    config: LinearFQEConfig | None = None,
    seed: int = 0,
) -> LinearFQEResult:
    """Thin alias emphasizing that linear FQE accepts sample weights directly."""

    return fit_linear_fqe(
        batch=batch,
        state_action_features=state_action_features,
        next_state_action_features=next_state_action_features,
        weights=weights,
        config=config,
        seed=seed,
    )


def predict_linear_q_values(theta: np.ndarray, state_action_features: np.ndarray) -> np.ndarray:
    """Predict action values from a fitted linear FQE parameter vector."""

    return np.asarray(state_action_features, dtype=np.float64) @ np.asarray(theta, dtype=np.float64)
