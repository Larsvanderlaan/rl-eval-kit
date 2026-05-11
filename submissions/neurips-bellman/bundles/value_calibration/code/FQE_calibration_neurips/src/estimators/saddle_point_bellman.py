from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.kernel_approximation import RBFSampler

from ..data import TransitionBatch
from ..policies import SoftmaxPolicy
from .random_feature_fqe import IdentityFeatureMap, state_action_matrix
from .regularized_bellman import RegularizedBellmanConfig, RegularizedBellmanModel, fit_regularized_bellman


@dataclass
class SaddlePointBellmanConfig:
    gamma: float = 0.95
    n_components: int = 128
    bandwidth: float = 0.6
    ridge: float = 1e-2
    feature_type: str = "rbf"
    critic_ridge: float = 1e-2
    max_iters: int = 250
    step_size: float = 1e-2


@dataclass
class IterativeSaddlePointBellmanConfig:
    gamma: float = 0.95
    n_components: int = 128
    bandwidth: float = 0.6
    q_ridge: float = 1e-3
    critic_ridge: float = 1e-3
    feature_type: str = "rbf"
    max_iters: int = 500
    step_size: float = 5e-2
    averaging: bool = True
    gradient_clip: float = 100.0
    divergence_threshold: float = 1e6


def _feature_map(config: IterativeSaddlePointBellmanConfig, seed: int):
    if config.feature_type == "linear":
        return IdentityFeatureMap()
    return RBFSampler(
        gamma=1.0 / max(2.0 * float(config.bandwidth) ** 2, 1e-8),
        n_components=int(config.n_components),
        random_state=int(seed),
    )


class IterativeSaddlePointBellmanModel:
    def __init__(self, featurizer: RBFSampler | IdentityFeatureMap, theta: np.ndarray, n_actions: int, diagnostics: dict):
        self.featurizer = featurizer
        self.theta = np.asarray(theta, dtype=float)
        self.n_actions = int(n_actions)
        self.diagnostics = diagnostics

    def _features(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        return self.featurizer.transform(state_action_matrix(states, actions, self.n_actions))

    def predict_q(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        return self._features(states, actions) @ self.theta

    def value(self, states: np.ndarray, policy: SoftmaxPolicy) -> np.ndarray:
        probs = policy.action_probabilities(states)
        vals = np.column_stack(
            [
                self.predict_q(states, np.full(states.shape[0], action, dtype=int))
                for action in range(self.n_actions)
            ]
        )
        return np.sum(probs * vals, axis=1)


def fit_saddle_point_bellman(
    batch: TransitionBatch,
    n_actions: int,
    policy: SoftmaxPolicy,
    config: SaddlePointBellmanConfig,
    seed: int,
) -> RegularizedBellmanModel:
    """Approximate adversarial Bellman residual via stabilized random-feature residual minimization.

    The critic class is represented by the same random Fourier basis as the Q
    model. The resulting closed-form regularized residual solve is the stable
    finite-dimensional saddle objective solution for this linearized critic.
    """

    base_cfg = RegularizedBellmanConfig(
        gamma=config.gamma,
        n_components=config.n_components,
        bandwidth=config.bandwidth,
        ridge=config.ridge + config.critic_ridge,
        feature_type=config.feature_type,
    )
    return fit_regularized_bellman(batch, n_actions, policy, base_cfg, seed)


def fit_iterative_saddle_point_bellman(
    batch: TransitionBatch,
    n_actions: int,
    policy: SoftmaxPolicy,
    config: IterativeSaddlePointBellmanConfig,
    seed: int,
) -> IterativeSaddlePointBellmanModel:
    """Iterative random-feature min-max Bellman residual approximation.

    This solves the finite-dimensional saddle objective

        min_theta max_beta E[(r - (phi-gamma phi')theta) psi beta]
        - critic_ridge ||beta||^2 / 2 + q_ridge ||theta||^2 / 2

    with simultaneous gradient descent/ascent. It is intentionally less
    stabilized than the closed-form approximation and therefore exposes the
    instability diagnostics expected from saddle-style Bellman training.
    """

    featurizer = _feature_map(config, seed)
    phi = featurizer.fit_transform(state_action_matrix(batch.states, batch.actions, n_actions))
    phi_next = featurizer.transform(state_action_matrix(batch.next_states, batch.next_actions, n_actions))
    design = phi - float(config.gamma) * phi_next
    rewards = np.asarray(batch.rewards, dtype=float)
    n, d = design.shape
    rng = np.random.default_rng(seed)
    theta = rng.normal(scale=1e-3, size=d)
    beta = rng.normal(scale=1e-3, size=d)
    avg_theta = np.zeros_like(theta)
    objective_history: list[float] = []
    grad_history: list[float] = []
    diverged = False
    divergence_reason = ""

    spectral = float(np.linalg.norm(design, ord=2) ** 2 / max(n, 1))
    condition_proxy = float(np.linalg.cond(design.T @ design / max(n, 1) + float(config.q_ridge) * np.eye(d)))

    for it in range(1, int(config.max_iters) + 1):
        residual = rewards - design @ theta
        critic_score = design @ beta
        grad_theta = -(design.T @ critic_score) / max(n, 1) + float(config.q_ridge) * theta
        grad_beta = (design.T @ residual) / max(n, 1) - float(config.critic_ridge) * beta
        grad_norm = float(np.sqrt(np.linalg.norm(grad_theta) ** 2 + np.linalg.norm(grad_beta) ** 2))
        if np.isfinite(grad_norm) and grad_norm > float(config.gradient_clip):
            scale = float(config.gradient_clip) / max(grad_norm, 1e-12)
            grad_theta = grad_theta * scale
            grad_beta = grad_beta * scale
            grad_norm = float(config.gradient_clip)
        theta = theta - float(config.step_size) * grad_theta
        beta = beta + float(config.step_size) * grad_beta
        if config.averaging:
            weight = 1.0 / float(it)
            avg_theta = (1.0 - weight) * avg_theta + weight * theta
        if it == 1 or it % max(1, int(config.max_iters) // 10) == 0 or it == int(config.max_iters):
            residual = rewards - design @ theta
            objective = float(
                np.mean(residual * (design @ beta))
                - 0.5 * float(config.critic_ridge) * float(beta @ beta)
                + 0.5 * float(config.q_ridge) * float(theta @ theta)
            )
            objective_history.append(objective)
            grad_history.append(grad_norm)
        max_abs = max(float(np.nanmax(np.abs(theta))), float(np.nanmax(np.abs(beta))))
        if (not np.isfinite(max_abs)) or max_abs > float(config.divergence_threshold):
            diverged = True
            divergence_reason = "nonfinite_or_exploding_saddle_parameters"
            break

    theta_out = avg_theta if bool(config.averaging) and not diverged else theta
    q_train = phi @ theta_out
    diagnostics = {
        "saddle_solver": "simultaneous_gradient_descent_ascent",
        "saddle_iterations_completed": float(it),
        "saddle_step_size": float(config.step_size),
        "saddle_primal_norm": float(np.linalg.norm(theta_out)),
        "saddle_critic_norm": float(np.linalg.norm(beta)),
        "saddle_gradient_norm_last": float(grad_history[-1]) if grad_history else float("nan"),
        "saddle_objective_last": float(objective_history[-1]) if objective_history else float("nan"),
        "saddle_objective_path_min": float(np.nanmin(objective_history)) if objective_history else float("nan"),
        "saddle_objective_path_max": float(np.nanmax(objective_history)) if objective_history else float("nan"),
        "saddle_condition_proxy": condition_proxy,
        "saddle_design_spectral_norm": spectral,
        "saddle_nan_flag": float(not np.all(np.isfinite(theta_out)) or not np.all(np.isfinite(beta))),
        "saddle_exploding_flag": float(diverged),
        "q_train_min": float(np.nanmin(q_train)) if q_train.size else float("nan"),
        "q_train_max": float(np.nanmax(q_train)) if q_train.size else float("nan"),
        "q_train_std": float(np.nanstd(q_train)) if q_train.size else float("nan"),
        "failure_reason": divergence_reason,
    }
    return IterativeSaddlePointBellmanModel(featurizer, theta_out, n_actions, diagnostics)
