from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..policies import SoftmaxPolicy
from .offset_correction import fit_offset_correction


@dataclass
class ShrunkCorrectionModel:
    base_model: object
    correction_model: object
    shrinkage: float = 0.5

    def predict_q(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        base = self.base_model.predict_q(states, actions)
        corrected = self.correction_model.predict_q(states, actions)
        return (1.0 - self.shrinkage) * base + self.shrinkage * corrected

    def value(self, states: np.ndarray, policy: SoftmaxPolicy) -> np.ndarray:
        probs = policy.action_probabilities(states)
        vals = np.column_stack([
            self.predict_q(states, np.full(states.shape[0], a, dtype=int))
            for a in range(probs.shape[1])
        ])
        return np.sum(probs * vals, axis=1)


def fit_regularized_toward_first_stage(base_model: object, *args, shrinkage: float = 0.5, **kwargs) -> ShrunkCorrectionModel:
    correction = fit_offset_correction(base_model, *args, residual_scale=1.0, **kwargs)
    return ShrunkCorrectionModel(base_model=base_model, correction_model=correction, shrinkage=float(shrinkage))
