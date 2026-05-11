"""Public contracts and action-space utilities for GenPQR."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal, Protocol, runtime_checkable

import numpy as np

from genpqr.exceptions import GenPQRConfigurationError


Array = np.ndarray
ActionKind = Literal["discrete", "continuous"]


@dataclass(frozen=True)
class ActionSpaceSpec:
    """Description of the action space visible to GenPQR.

    Parameters
    ----------
    kind:
        Either ``"discrete"`` or ``"continuous"``.
    n_actions:
        Number of finite actions for a discrete action space.
    action_dim:
        Width of the encoded action vector used by Q estimators. For discrete
        actions this is ``n_actions`` because actions are encoded one-hot before
        being passed to generic FQE backends.
    """

    kind: ActionKind
    n_actions: int | None = None
    action_dim: int | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"discrete", "continuous"}:
            raise ValueError("kind must be either 'discrete' or 'continuous'.")
        if self.kind == "discrete":
            if self.n_actions is None or int(self.n_actions) <= 1:
                raise ValueError("discrete action spaces require n_actions > 1.")
            object.__setattr__(self, "n_actions", int(self.n_actions))
            object.__setattr__(self, "action_dim", int(self.n_actions))
        else:
            if self.action_dim is None or int(self.action_dim) <= 0:
                raise ValueError("continuous action spaces require action_dim > 0.")
            object.__setattr__(self, "action_dim", int(self.action_dim))
            object.__setattr__(self, "n_actions", None)

    @classmethod
    def discrete(cls, n_actions: int) -> "ActionSpaceSpec":
        """Create a finite action-space spec."""

        return cls(kind="discrete", n_actions=int(n_actions))

    @classmethod
    def continuous(cls, action_dim: int) -> "ActionSpaceSpec":
        """Create a continuous action-space spec."""

        return cls(kind="continuous", action_dim=int(action_dim))

    @classmethod
    def infer(cls, actions: Array, *, n_actions: int | None = None) -> "ActionSpaceSpec":
        """Infer a conservative action-space spec from observed actions.

        Integer-valued one-dimensional actions are treated as discrete. All
        other arrays are treated as continuous unless ``n_actions`` is supplied.
        """

        arr = np.asarray(actions)
        if n_actions is not None:
            return cls.discrete(int(n_actions))
        if arr.ndim == 1 and _is_integer_like(arr):
            return cls.discrete(int(np.max(arr)) + 1)
        if arr.ndim == 2 and arr.shape[1] == 1 and _is_integer_like(arr.reshape(-1)):
            return cls.discrete(int(np.max(arr)) + 1)
        if arr.ndim != 2:
            raise ValueError("continuous actions must be a 2D array; pass action_space for discrete action matrices.")
        return cls.continuous(arr.shape[1])

    def validate_actions(self, actions: Array, *, n_rows: int | None = None, name: str = "actions") -> None:
        """Validate actions against this action space."""

        if self.kind == "discrete":
            indices = self.action_indices(actions, n_rows=n_rows, name=name)
            if indices.size and (indices.min() < 0 or indices.max() >= int(self.n_actions)):
                raise ValueError(f"{name} contains action indices outside [0, {int(self.n_actions) - 1}].")
            return
        arr = self.action_matrix(actions, n_rows=n_rows, name=name)
        if arr.shape[1] != int(self.action_dim):
            raise ValueError(f"{name} must have action_dim={int(self.action_dim)} columns.")

    def action_indices(self, actions: Array, *, n_rows: int | None = None, name: str = "actions") -> Array:
        """Return integer action indices for a discrete action array."""

        if self.kind != "discrete":
            raise GenPQRConfigurationError("action_indices is only available for discrete action spaces.")
        arr = np.asarray(actions)
        if arr.ndim == 1:
            idx = arr
        elif arr.ndim == 2 and arr.shape[1] == 1:
            idx = arr.reshape(-1)
        elif arr.ndim == 2 and arr.shape[1] == int(self.n_actions):
            _validate_one_hot_matrix(arr, name=name)
            idx = np.argmax(arr, axis=1)
        else:
            raise ValueError(f"{name} must be a 1D index array or one-hot matrix for a discrete action space.")
        if n_rows is not None and idx.shape[0] != int(n_rows):
            raise ValueError(f"{name} must have {int(n_rows)} rows.")
        if not _is_integer_like(idx):
            raise ValueError(f"{name} must contain integer action indices.")
        idx = np.asarray(idx, dtype=np.int64).reshape(-1)
        if idx.size and (idx.min() < 0 or idx.max() >= int(self.n_actions)):
            raise ValueError(f"{name} contains action indices outside [0, {int(self.n_actions) - 1}].")
        return idx

    def action_matrix(self, actions: Array, *, n_rows: int | None = None, name: str = "actions") -> Array:
        """Return the encoded 2D action matrix used by Q estimators."""

        if self.kind == "discrete":
            return self.one_hot(actions, n_rows=n_rows, name=name)
        arr = np.asarray(actions, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if arr.ndim != 2:
            raise ValueError(f"{name} must be a 2D continuous-action array.")
        if n_rows is not None and arr.shape[0] != int(n_rows):
            raise ValueError(f"{name} must have {int(n_rows)} rows.")
        if arr.shape[1] != int(self.action_dim):
            raise ValueError(f"{name} must have action_dim={int(self.action_dim)} columns.")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} must contain only finite values.")
        return arr

    def one_hot(self, actions: Array, *, n_rows: int | None = None, name: str = "actions") -> Array:
        """Encode discrete action indices as a one-hot matrix."""

        idx = self.action_indices(actions, n_rows=n_rows, name=name)
        out = np.zeros((idx.shape[0], int(self.n_actions)), dtype=np.float64)
        if idx.size:
            out[np.arange(idx.shape[0]), idx] = 1.0
        return out

    def encode_samples(self, actions: Array, *, n_rows: int, name: str = "actions") -> Array:
        """Encode policy samples for FQE.

        Returns shape ``(n, action_dim)`` for one sample and
        ``(n, n_samples, action_dim)`` for multiple samples.
        """

        arr = np.asarray(actions)
        if self.kind == "continuous":
            arr = np.asarray(actions, dtype=np.float64)
            if arr.ndim == 2:
                self.action_matrix(arr, n_rows=n_rows, name=name)
                return arr
            if arr.ndim == 3:
                if arr.shape[0] != int(n_rows) or arr.shape[2] != int(self.action_dim):
                    raise ValueError(f"{name} must have shape (n, n_samples, action_dim).")
                if not np.all(np.isfinite(arr)):
                    raise ValueError(f"{name} must contain only finite values.")
                return arr
            raise ValueError(f"{name} must have shape (n, action_dim) or (n, n_samples, action_dim).")

        if arr.ndim == 1 or (arr.ndim == 2 and arr.shape[1] == 1):
            return self.one_hot(arr, n_rows=n_rows, name=name)
        if arr.ndim == 2:
            if arr.shape[0] != int(n_rows):
                raise ValueError(f"{name} must have {int(n_rows)} rows.")
            if arr.shape[1] == int(self.n_actions) and _is_one_hot_matrix(arr):
                return self.one_hot(arr, n_rows=n_rows, name=name)
            encoded = np.zeros((arr.shape[0], arr.shape[1], int(self.n_actions)), dtype=np.float64)
            for j in range(arr.shape[1]):
                encoded[:, j, :] = self.one_hot(arr[:, j], n_rows=n_rows, name=name)
            return encoded
        if arr.ndim == 3 and arr.shape[2] == int(self.n_actions):
            if arr.shape[0] != int(n_rows):
                raise ValueError(f"{name} must have {int(n_rows)} rows.")
            if not np.all(np.isfinite(arr)) or not np.all((arr == 0) | (arr == 1)):
                raise ValueError(f"{name} one-hot samples must contain only 0/1 values.")
            if not np.allclose(arr.sum(axis=2), 1.0):
                raise ValueError(f"{name} one-hot samples must have exactly one active action.")
            return np.asarray(arr, dtype=np.float64)
        raise ValueError(f"{name} has unsupported sample shape for a discrete action space.")

    def all_actions(self, n_rows: int) -> Array:
        """Return a repeated finite-action grid with shape ``(n_rows * n_actions,)``."""

        if self.kind != "discrete":
            raise GenPQRConfigurationError("all_actions is only available for discrete action spaces.")
        return np.tile(np.arange(int(self.n_actions), dtype=np.int64), int(n_rows))


@runtime_checkable
class EstimatedPolicy(Protocol):
    """Protocol for fitted behavior-policy estimates."""

    action_space: ActionSpaceSpec

    def log_prob(self, states: Array, actions: Array) -> Array:
        """Return log probabilities or log densities for state-action rows."""

    def sample(self, states: Array, rng: np.random.Generator, n_samples: int = 1) -> Array:
        """Sample actions for each state."""


@runtime_checkable
class PolicyEstimator(Protocol):
    """Protocol for first-stage policy estimators."""

    def fit(
        self,
        *,
        states: Array,
        actions: Array,
        next_states: Array | None,
        terminals: Array | None,
        action_space: ActionSpaceSpec,
        sample_weight: Array | None = None,
        env: Any | None = None,
    ) -> EstimatedPolicy:
        """Fit a policy estimator."""


@runtime_checkable
class NormalizationPolicy(Protocol):
    """Protocol for the GenPQR normalization policy ``mu``."""

    action_space: ActionSpaceSpec

    def sample(self, states: Array, rng: np.random.Generator, n_samples: int = 1) -> Array:
        """Sample normalization-policy actions for each state."""


@runtime_checkable
class FittedQFunction(Protocol):
    """Protocol for fitted Q functions used by reward recovery."""

    action_space: ActionSpaceSpec

    def predict_q(self, states: Array, actions: Array) -> Array:
        """Predict Q-values for state-action rows."""

    def expected_q(
        self,
        states: Array,
        normalization_policy: NormalizationPolicy,
        *,
        n_action_samples: int,
        rng: np.random.Generator,
    ) -> Array:
        """Estimate ``E_mu[Q(s, A)]`` for each state."""


@runtime_checkable
class QEstimator(Protocol):
    """Protocol for second-stage Q estimators."""

    def fit(
        self,
        *,
        states: Array,
        actions: Array,
        next_states: Array,
        pseudo_rewards: Array,
        normalization_policy: NormalizationPolicy,
        gamma: float,
        terminals: Array | None = None,
        sample_weight: Array | None = None,
        policy: EstimatedPolicy | None = None,
    ) -> FittedQFunction:
        """Fit a Q estimator for the pseudo-reward ``u-g``."""


@runtime_checkable
class RewardFunction(Protocol):
    """Protocol for recovered reward functions."""

    def predict_reward(self, states: Array, actions: Array) -> Array:
        """Predict recovered rewards for state-action rows."""


AnchorFunction = Callable[[Array], Array]


def _is_integer_like(arr: Array) -> bool:
    values = np.asarray(arr)
    if not np.all(np.isfinite(values)):
        return False
    return np.issubdtype(values.dtype, np.integer) or np.allclose(values, np.round(values))


def _is_one_hot_matrix(values: Array) -> bool:
    arr = np.asarray(values)
    return bool(
        arr.ndim == 2
        and np.all(np.isfinite(arr))
        and np.all((arr == 0) | (arr == 1))
        and np.allclose(arr.sum(axis=1), 1.0)
    )


def _validate_one_hot_matrix(values: Array, *, name: str) -> None:
    arr = np.asarray(values)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values.")
    if not np.all((arr == 0) | (arr == 1)):
        raise ValueError(f"{name} one-hot matrix must contain only 0/1 values.")
    if arr.shape[0] and not np.allclose(arr.sum(axis=1), 1.0):
        raise ValueError(f"{name} must be one-hot encoded when supplied as a matrix.")
