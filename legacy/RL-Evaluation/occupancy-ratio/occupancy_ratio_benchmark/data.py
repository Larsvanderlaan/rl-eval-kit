from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


Array = np.ndarray


@dataclass
class BenchmarkDataset:
    """One benchmark dataset and all truth needed for diagnostics."""

    setting: str
    states: Array
    actions: Array
    next_states: Array
    target_actions: Array
    next_target_actions: Array
    rewards: Array
    true_ratio: Array | None
    initial_states: Array
    initial_actions: Array
    initial_weights: Array
    masks: Array
    gamma: float
    seed: int
    sample_size: int
    true_action_ratio: Array | None = None
    true_transition_ratio: Array | None = None
    reference_weights: Array | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = int(np.asarray(self.states).shape[0])
        for name in (
            "actions",
            "next_states",
            "target_actions",
            "next_target_actions",
            "rewards",
            "masks",
        ):
            value = np.asarray(getattr(self, name))
            if value.shape[0] != n:
                raise ValueError(f"{name} must have {n} rows.")
        if self.true_ratio is not None:
            value = np.asarray(self.true_ratio)
            if value.shape[0] != n:
                raise ValueError(f"true_ratio must have {n} rows.")

    @property
    def n(self) -> int:
        return int(np.asarray(self.states).shape[0])

    @property
    def state_dim(self) -> int:
        return int(np.asarray(self.states).reshape(self.n, -1).shape[1])

    @property
    def action_dim(self) -> int:
        return int(np.asarray(self.actions).reshape(self.n, -1).shape[1])


def as_2d(x: Array) -> Array:
    arr = np.asarray(x)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    if arr.ndim == 2:
        return arr
    raise ValueError("Expected a 1D or 2D array.")


def one_hot(indices: Array, size: int) -> Array:
    idx = np.asarray(indices, dtype=np.int64).reshape(-1)
    out = np.zeros((idx.shape[0], int(size)), dtype=np.float64)
    out[np.arange(idx.shape[0]), idx] = 1.0
    return out


def state_action_indices(states: Array, actions: Array, n_actions: int) -> Array:
    return np.asarray(states, dtype=np.int64).reshape(-1) * int(n_actions) + np.asarray(
        actions,
        dtype=np.int64,
    ).reshape(-1)
