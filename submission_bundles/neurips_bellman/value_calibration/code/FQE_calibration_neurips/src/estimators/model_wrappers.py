from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PredictionDistortionConfig:
    kind: str = "none"
    intercept: float = 0.0
    slope: float = 1.0
    strength: float = 0.25
    clip: float | None = None


class PredictionDistortionWrapper:
    """Diagnostic-only wrapper for mechanism studies of calibratable errors."""

    def __init__(self, base_model: object, config: PredictionDistortionConfig):
        self.base_model = base_model
        self.config = config
        self.n_actions = getattr(base_model, "n_actions", None)
        base_diag = dict(getattr(base_model, "diagnostics", {}))
        base_diag.update(
            {
                "prediction_distortion_kind": config.kind,
                "prediction_distortion_intercept": float(config.intercept),
                "prediction_distortion_slope": float(config.slope),
                "prediction_distortion_strength": float(config.strength),
            }
        )
        self.diagnostics = base_diag

    def _distort(self, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        kind = str(self.config.kind)
        if kind in {"none", "", "identity"}:
            out = q
        elif kind == "affine":
            out = float(self.config.intercept) + float(self.config.slope) * q
        elif kind == "monotone":
            out = q + float(self.config.strength) * np.tanh(q)
        elif kind == "saturation":
            scale = max(float(self.config.strength), 1e-6)
            out = scale * np.tanh(q / scale)
        else:
            raise ValueError(f"Unknown prediction distortion kind '{kind}'.")
        if self.config.clip is not None:
            out = np.clip(out, -float(self.config.clip), float(self.config.clip))
        return np.asarray(out, dtype=float)

    def predict_q(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        return self._distort(self.base_model.predict_q(states, actions))

    def value(self, states: np.ndarray, policy: object) -> np.ndarray:
        probs = policy.action_probabilities(states)
        vals = np.column_stack(
            [
                self.predict_q(states, np.full(states.shape[0], action, dtype=int))
                for action in range(probs.shape[1])
            ]
        )
        return np.sum(probs * vals, axis=1)


def maybe_wrap_prediction_distortion(model: object, params: dict) -> object:
    raw = params.get("prediction_distortion")
    if not raw:
        return model
    cfg = PredictionDistortionConfig(**dict(raw))
    return PredictionDistortionWrapper(model, cfg)
