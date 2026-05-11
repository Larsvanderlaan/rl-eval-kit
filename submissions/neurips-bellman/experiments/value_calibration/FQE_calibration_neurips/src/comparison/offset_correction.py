from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import Ridge

from ..data import TransitionBatch
from ..estimators.random_feature_fqe import state_action_matrix
from ..policies import SoftmaxPolicy


@dataclass
class OffsetCorrectionModel:
    base_model: object
    featurizer: RBFSampler
    residual_model: Ridge
    n_actions: int

    def _features(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        return self.featurizer.transform(state_action_matrix(states, actions, self.n_actions))

    def predict_q(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        base = self.base_model.predict_q(states, actions)
        return base + self.residual_model.predict(self._features(states, actions))

    def value(self, states: np.ndarray, policy: SoftmaxPolicy) -> np.ndarray:
        probs = policy.action_probabilities(states)
        vals = np.column_stack([
            self.predict_q(states, np.full(states.shape[0], a, dtype=int))
            for a in range(self.n_actions)
        ])
        return np.sum(probs * vals, axis=1)


def fit_offset_correction(
    base_model: object,
    calibration_batch: TransitionBatch,
    gamma: float,
    n_actions: int,
    seed: int,
    residual_scale: float = 1.0,
    n_components: int = 64,
    ridge: float = 1e-2,
) -> OffsetCorrectionModel:
    pred = base_model.predict_q(calibration_batch.states, calibration_batch.actions)
    target = calibration_batch.rewards + float(gamma) * base_model.predict_q(
        calibration_batch.next_states, calibration_batch.next_actions
    )
    residual = float(residual_scale) * (target - pred)
    featurizer = RBFSampler(gamma=0.5, n_components=int(n_components), random_state=int(seed))
    phi = featurizer.fit_transform(state_action_matrix(calibration_batch.states, calibration_batch.actions, n_actions))
    model = Ridge(alpha=float(ridge)).fit(phi, residual)
    return OffsetCorrectionModel(base_model, featurizer, model, n_actions)
