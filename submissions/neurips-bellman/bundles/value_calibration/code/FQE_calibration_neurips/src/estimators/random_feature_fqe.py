from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import Ridge

from ..data import TransitionBatch
from ..policies import SoftmaxPolicy


Array = np.ndarray


def state_action_matrix(states: Array, actions: Array, n_actions: int) -> Array:
    x = np.asarray(states, dtype=float)
    a = np.asarray(actions, dtype=int)
    one_hot = np.zeros((x.shape[0], n_actions), dtype=float)
    one_hot[np.arange(x.shape[0]), a] = 1.0
    return np.hstack([x, one_hot, x * (a[:, None] + 1.0) / max(n_actions, 1)])


class IdentityFeatureMap:
    def fit_transform(self, x: Array) -> Array:
        return np.asarray(x, dtype=float)

    def transform(self, x: Array) -> Array:
        return np.asarray(x, dtype=float)


@dataclass
class RandomFeatureFQEConfig:
    gamma: float = 0.95
    n_components: int = 128
    bandwidth: float = 0.6
    ridge: float = 1e-3
    n_iters: int = 35
    feature_type: str = "rbf"


class RandomFeatureFQEModel:
    def __init__(
        self,
        featurizer: RBFSampler | IdentityFeatureMap,
        ridge: Ridge,
        n_actions: int,
        diagnostics: dict[str, float | str] | None = None,
    ):
        self.featurizer = featurizer
        self.ridge = ridge
        self.n_actions = n_actions
        self.diagnostics = diagnostics or {}

    def _features(self, states: Array, actions: Array) -> Array:
        return self.featurizer.transform(state_action_matrix(states, actions, self.n_actions))

    def predict_q(self, states: Array, actions: Array) -> Array:
        return self.ridge.predict(self._features(states, actions)).astype(float)

    def value(self, states: Array, policy: SoftmaxPolicy) -> Array:
        probs = policy.action_probabilities(states)
        vals = np.column_stack([
            self.predict_q(states, np.full(states.shape[0], a, dtype=int))
            for a in range(self.n_actions)
        ])
        return np.sum(probs * vals, axis=1)


def fit_random_feature_fqe(
    batch: TransitionBatch,
    n_actions: int,
    policy: SoftmaxPolicy,
    config: RandomFeatureFQEConfig,
    seed: int,
) -> RandomFeatureFQEModel:
    base_x = state_action_matrix(batch.states, batch.actions, n_actions)
    if config.feature_type == "linear":
        featurizer = IdentityFeatureMap()
    else:
        featurizer = RBFSampler(
            gamma=1.0 / max(2.0 * config.bandwidth**2, 1e-8),
            n_components=int(config.n_components),
            random_state=int(seed),
        )
    phi = featurizer.fit_transform(base_x)
    phi_next = featurizer.transform(state_action_matrix(batch.next_states, batch.next_actions, n_actions))
    ridge = Ridge(alpha=float(config.ridge), fit_intercept=True)
    target = np.asarray(batch.rewards, dtype=float).copy()
    for _ in range(int(config.n_iters)):
        ridge.fit(phi, target)
        target = batch.rewards + config.gamma * ridge.predict(phi_next)
    ridge.fit(phi, target)
    q_train = ridge.predict(phi).astype(float)
    diagnostics = {
        "actual_bellman_iterations": float(config.n_iters),
        "feature_dimension": float(phi.shape[1]),
        "ridge_alpha": float(config.ridge),
        "q_train_min": float(np.nanmin(q_train)) if q_train.size else float("nan"),
        "q_train_max": float(np.nanmax(q_train)) if q_train.size else float("nan"),
        "q_train_std": float(np.nanstd(q_train)) if q_train.size else float("nan"),
    }
    return RandomFeatureFQEModel(featurizer, ridge, n_actions, diagnostics=diagnostics)
