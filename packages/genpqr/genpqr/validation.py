"""Input validation helpers for GenPQR."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from genpqr.types import ActionSpaceSpec, Array


@dataclass(frozen=True)
class TransitionBatch:
    """Validated transition arrays used internally by GenPQR."""

    states: Array
    actions: Array
    encoded_actions: Array
    next_states: Array
    terminals: Array
    sample_weight: Array | None
    action_space: ActionSpaceSpec


def as_2d_float(values: Array, name: str, *, n_rows: int | None = None) -> Array:
    """Return a finite 2D float array."""

    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array.")
    if n_rows is not None and arr.shape[0] != int(n_rows):
        raise ValueError(f"{name} must have {int(n_rows)} rows.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values.")
    return arr


def as_1d_float(values: Array, name: str, *, n_rows: int | None = None) -> Array:
    """Return a finite row vector.

    Valid inputs are ``(n,)`` and ``(n, 1)``. Wider matrices are rejected rather
    than flattened so backend shape bugs cannot silently broadcast downstream.
    """

    raw = np.asarray(values, dtype=np.float64)
    if raw.ndim == 0:
        arr = raw.reshape(1)
    elif raw.ndim == 1:
        arr = raw
    elif raw.ndim == 2 and raw.shape[1] == 1:
        arr = raw[:, 0]
    else:
        raise ValueError(f"{name} must have shape (n,) or (n, 1).")
    if n_rows is not None and arr.shape[0] != int(n_rows):
        raise ValueError(f"{name} must have {int(n_rows)} rows.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values.")
    return arr


def optional_terminals(terminals: Array | None, n_rows: int) -> Array:
    """Validate terminal flags or return zeros."""

    if terminals is None:
        return np.zeros(int(n_rows), dtype=np.float64)
    arr = as_1d_float(terminals, "terminals", n_rows=n_rows)
    if np.any((arr < 0.0) | (arr > 1.0)):
        raise ValueError("terminals must be binary or probabilities in [0, 1].")
    return arr


def optional_weights(sample_weight: Array | None, n_rows: int) -> Array | None:
    """Validate optional nonnegative row weights."""

    if sample_weight is None:
        return None
    arr = as_1d_float(sample_weight, "sample_weight", n_rows=n_rows)
    if np.any(arr < 0.0):
        raise ValueError("sample_weight must be nonnegative.")
    if not np.any(arr > 0.0):
        raise ValueError("sample_weight must contain at least one positive value.")
    return arr


def validate_gamma(gamma: float) -> float:
    """Validate a discounted infinite-horizon gamma."""

    value = float(gamma)
    if not (0.0 <= value < 1.0):
        raise ValueError("gamma must be in [0, 1).")
    return value


def prepare_transition_batch(
    *,
    states: Array,
    actions: Array,
    next_states: Array,
    terminals: Array | None,
    gamma: float,
    action_space: ActionSpaceSpec | None = None,
    sample_weight: Array | None = None,
) -> TransitionBatch:
    """Validate transition arrays and infer the action space when needed."""

    del gamma
    states_2d = as_2d_float(states, "states")
    n_rows = states_2d.shape[0]
    if n_rows == 0:
        raise ValueError("states must be nonempty.")
    next_states_2d = as_2d_float(next_states, "next_states", n_rows=n_rows)
    if next_states_2d.shape[1] != states_2d.shape[1]:
        raise ValueError("next_states must have the same number of columns as states.")
    spec = ActionSpaceSpec.infer(actions) if action_space is None else action_space
    spec.validate_actions(actions, n_rows=n_rows)
    encoded_actions = spec.action_matrix(actions, n_rows=n_rows)
    return TransitionBatch(
        states=states_2d,
        actions=np.asarray(actions),
        encoded_actions=encoded_actions,
        next_states=next_states_2d,
        terminals=optional_terminals(terminals, n_rows),
        sample_weight=optional_weights(sample_weight, n_rows),
        action_space=spec,
    )


def normalize_anchor_values(anchor_values: Array | float | int, states: Array) -> Array:
    """Coerce anchor/normalization values to one value per state."""

    n_rows = np.asarray(states).shape[0]
    if np.isscalar(anchor_values):
        return np.full(n_rows, float(anchor_values), dtype=np.float64)
    return as_1d_float(np.asarray(anchor_values), "anchor_function(states)", n_rows=n_rows)
