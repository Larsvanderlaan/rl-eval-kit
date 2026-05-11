from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.kernel_approximation import RBFSampler

from ..data import TransitionBatch
from ..policies import SoftmaxPolicy
from .random_feature_fqe import IdentityFeatureMap, state_action_matrix


Array = np.ndarray


@dataclass
class RegularizedBellmanConfig:
    gamma: float = 0.95
    n_components: int = 128
    bandwidth: float = 0.6
    ridge: float = 1e-2
    feature_type: str = "rbf"


class RegularizedBellmanModel:
    def __init__(
        self,
        featurizer: RBFSampler | IdentityFeatureMap,
        theta: Array,
        n_actions: int,
        diagnostics: dict[str, float | str] | None = None,
    ):
        self.featurizer = featurizer
        self.theta = np.asarray(theta, dtype=float)
        self.n_actions = int(n_actions)
        self.diagnostics = diagnostics or {}

    def _features(self, states: Array, actions: Array) -> Array:
        return self.featurizer.transform(state_action_matrix(states, actions, self.n_actions))

    def predict_q(self, states: Array, actions: Array) -> Array:
        return self._features(states, actions) @ self.theta

    def value(self, states: Array, policy: SoftmaxPolicy) -> Array:
        probs = policy.action_probabilities(states)
        vals = np.column_stack([
            self.predict_q(states, np.full(states.shape[0], a, dtype=int))
            for a in range(self.n_actions)
        ])
        return np.sum(probs * vals, axis=1)


def fit_regularized_bellman(
    batch: TransitionBatch,
    n_actions: int,
    policy: SoftmaxPolicy,
    config: RegularizedBellmanConfig,
    seed: int,
) -> RegularizedBellmanModel:
    if config.feature_type == "linear":
        featurizer = IdentityFeatureMap()
    else:
        featurizer = RBFSampler(
            gamma=1.0 / max(2.0 * config.bandwidth**2, 1e-8),
            n_components=int(config.n_components),
            random_state=int(seed),
        )
    phi = featurizer.fit_transform(state_action_matrix(batch.states, batch.actions, n_actions))
    phi_next = featurizer.transform(state_action_matrix(batch.next_states, batch.next_actions, n_actions))
    design = phi - float(config.gamma) * phi_next
    lhs = design.T @ design + float(config.ridge) * np.eye(design.shape[1])
    rhs = design.T @ np.asarray(batch.rewards, dtype=float)
    try:
        theta = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        theta = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
    q_train = phi @ theta
    diagnostics = {
        "feature_dimension": float(phi.shape[1]),
        "ridge_alpha": float(config.ridge),
        "bellman_design_condition_proxy": float(np.linalg.cond(lhs)) if lhs.size else float("nan"),
        "q_train_min": float(np.nanmin(q_train)) if q_train.size else float("nan"),
        "q_train_max": float(np.nanmax(q_train)) if q_train.size else float("nan"),
        "q_train_std": float(np.nanstd(q_train)) if q_train.size else float("nan"),
    }
    return RegularizedBellmanModel(featurizer, theta, n_actions, diagnostics=diagnostics)
