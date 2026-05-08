from __future__ import annotations

from dataclasses import dataclass

import numpy as np


Array = np.ndarray


def _as_2d_float(name: str, value: Array) -> Array:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array, got shape {arr.shape}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values.")
    return np.ascontiguousarray(arr)


def _as_next_float(value: Array, n: int, p: int) -> Array:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim == 2:
        if arr.shape != (n, p):
            raise ValueError(f"X_next must have shape {(n, p)} or (n, m, {p}), got {arr.shape}.")
        out = np.ascontiguousarray(arr)
    elif arr.ndim == 3:
        if arr.shape[0] != n or arr.shape[2] != p:
            raise ValueError(f"X_next must have shape {(n, p)} or (n, m, {p}), got {arr.shape}.")
        out = np.ascontiguousarray(arr)
    else:
        raise ValueError(f"X_next must be 2D or 3D, got shape {arr.shape}.")
    if not np.all(np.isfinite(out)):
        raise ValueError("X_next contains non-finite values.")
    return out


def _as_1d_float(name: str, value: Array, n: int) -> Array:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.shape[0] != n:
        raise ValueError(f"{name} must have length {n}, got {arr.shape[0]}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values.")
    return np.ascontiguousarray(arr)


@dataclass(frozen=True)
class BellmanTransitionData:
    """Offline fixed-policy-evaluation transitions in state-action feature form."""

    X: Array
    reward: Array
    X_next: Array
    sample_weight: Array | None = None
    groups: Array | None = None

    def __post_init__(self) -> None:
        x = _as_2d_float("X", self.X)
        r = _as_1d_float("reward", self.reward, x.shape[0])
        xp = _as_next_float(self.X_next, x.shape[0], x.shape[1])
        w = None if self.sample_weight is None else _as_1d_float("sample_weight", self.sample_weight, x.shape[0])
        groups = None
        if self.groups is not None:
            groups = np.asarray(self.groups).reshape(-1)
            if groups.shape[0] != x.shape[0]:
                raise ValueError(f"groups must have length {x.shape[0]}, got {groups.shape[0]}.")
        object.__setattr__(self, "X", x)
        object.__setattr__(self, "reward", r)
        object.__setattr__(self, "X_next", xp)
        object.__setattr__(self, "sample_weight", w)
        object.__setattr__(self, "groups", groups)

    @property
    def n_samples(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.X.shape[1])

    def subset(self, indices: Array) -> "BellmanTransitionData":
        idx = np.asarray(indices, dtype=np.int64)
        return BellmanTransitionData(
            X=self.X[idx],
            reward=self.reward[idx],
            X_next=self.X_next[idx],
            sample_weight=None if self.sample_weight is None else self.sample_weight[idx],
            groups=None if self.groups is None else self.groups[idx],
        )
