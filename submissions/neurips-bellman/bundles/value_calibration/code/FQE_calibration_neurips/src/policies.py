from __future__ import annotations

from dataclasses import dataclass

import numpy as np


Array = np.ndarray


@dataclass
class SoftmaxPolicy:
    """Discrete-action softmax policy over continuous states."""

    weights: Array
    temperature: float = 1.0
    name: str = "softmax"

    def action_probabilities(self, states: Array) -> Array:
        x = np.asarray(states, dtype=float)
        logits = x @ self.weights.T
        logits = logits / max(float(self.temperature), 1e-8)
        logits = logits - np.max(logits, axis=1, keepdims=True)
        probs = np.exp(logits)
        return probs / np.sum(probs, axis=1, keepdims=True)

    def sample(self, states: Array, rng: np.random.Generator) -> Array:
        probs = self.action_probabilities(states)
        cdf = np.cumsum(probs, axis=1)
        draws = rng.random(size=probs.shape[0])[:, None]
        return (draws > cdf[:, :-1]).sum(axis=1).astype(int)


def make_policy_pair(
    state_dim: int,
    n_actions: int,
    shift: float,
    coverage: str,
    seed: int,
) -> tuple[SoftmaxPolicy, SoftmaxPolicy]:
    """Create target and behavior policies with a controlled shift."""

    rng = np.random.default_rng(seed)
    target_w = rng.normal(scale=0.55, size=(n_actions, state_dim))
    direction = rng.normal(scale=0.45, size=(n_actions, state_dim))
    behavior_w = target_w - float(shift) * direction
    if coverage == "good":
        target_temp, behavior_temp = 1.2, 1.4
    elif coverage == "moderate":
        target_temp, behavior_temp = 0.85, 1.05
    elif coverage == "severe":
        target_temp, behavior_temp = 0.55, 0.75
    elif coverage == "extrapolation":
        target_temp, behavior_temp = 0.45, 1.5
    else:
        raise ValueError(f"Unknown coverage setting '{coverage}'.")
    return (
        SoftmaxPolicy(target_w, temperature=target_temp, name="target"),
        SoftmaxPolicy(behavior_w, temperature=behavior_temp, name=f"behavior_{coverage}"),
    )
