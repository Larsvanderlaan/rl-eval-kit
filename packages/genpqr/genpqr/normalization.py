"""Normalization-policy implementations for GenPQR."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from genpqr.exceptions import GenPQRConfigurationError
from genpqr.types import ActionSpaceSpec, Array
from genpqr.validation import as_1d_float, as_2d_float


ProbabilityProvider = Array | Callable[[Array], Array]
Sampler = Callable[[Array, np.random.Generator, int], Array]


@dataclass
class DiscreteNormalizationPolicy:
    """Finite-action normalization policy ``mu``.

    Parameters
    ----------
    n_actions:
        Number of finite actions.
    probabilities:
        Either a fixed probability vector, a per-row matrix, or a callable from
        states to a probability matrix.
    """

    n_actions: int
    probabilities: ProbabilityProvider

    def __post_init__(self) -> None:
        self.action_space = ActionSpaceSpec.discrete(int(self.n_actions))

    @classmethod
    def uniform(cls, n_actions: int) -> "DiscreteNormalizationPolicy":
        """Create a uniform finite-action normalization policy."""

        return cls(n_actions=int(n_actions), probabilities=np.full(int(n_actions), 1.0 / int(n_actions)))

    @classmethod
    def anchor(cls, n_actions: int, anchor_action: int) -> "DiscreteNormalizationPolicy":
        """Create an anchor-action normalization policy."""

        if anchor_action < 0 or anchor_action >= int(n_actions):
            raise ValueError("anchor_action is out of bounds.")
        probs = np.zeros(int(n_actions), dtype=np.float64)
        probs[int(anchor_action)] = 1.0
        return cls(n_actions=int(n_actions), probabilities=probs)

    def predict_proba(self, states: Array) -> Array:
        """Return ``mu(a | s)`` for every state and finite action."""

        states_2d = as_2d_float(states, "states")
        if callable(self.probabilities):
            probs = np.asarray(self.probabilities(states_2d), dtype=np.float64)
        else:
            probs = np.asarray(self.probabilities, dtype=np.float64)
        if probs.ndim == 1:
            if probs.shape[0] != self.n_actions:
                raise ValueError("probability vector has the wrong number of actions.")
            probs = np.tile(probs.reshape(1, -1), (states_2d.shape[0], 1))
        if probs.ndim != 2 or probs.shape != (states_2d.shape[0], self.n_actions):
            raise ValueError("normalization probabilities must have shape (n, n_actions).")
        if not np.all(np.isfinite(probs)):
            raise ValueError("normalization probabilities must be finite.")
        if np.any(probs < 0.0):
            raise ValueError("normalization probabilities must be nonnegative.")
        row_sums = probs.sum(axis=1, keepdims=True)
        if np.any(row_sums <= 0.0):
            raise ValueError("normalization probabilities must have positive row sums.")
        return probs / row_sums

    def log_prob(self, states: Array, actions: Array) -> Array:
        """Return log probabilities for state-action rows."""

        probs = self.predict_proba(states)
        idx = self.action_space.action_indices(actions, n_rows=probs.shape[0])
        return np.log(np.clip(probs[np.arange(probs.shape[0]), idx], 1e-300, None))

    def sample(self, states: Array, rng: np.random.Generator, n_samples: int = 1) -> Array:
        """Sample finite actions from ``mu``."""

        if n_samples <= 0:
            raise ValueError("n_samples must be positive.")
        probs = self.predict_proba(states)
        draws = np.empty((probs.shape[0], int(n_samples)), dtype=np.int64)
        choices = np.arange(self.n_actions, dtype=np.int64)
        for i, row in enumerate(probs):
            draws[i] = rng.choice(choices, size=int(n_samples), p=row)
        return draws.reshape(-1) if int(n_samples) == 1 else draws


@dataclass
class ContinuousNormalizationPolicy:
    """Continuous-action normalization policy backed by a sampler.

    The sampler must return either ``(n, action_dim)`` for one sample or
    ``(n, n_samples, action_dim)`` for multiple samples.
    """

    action_dim: int
    sampler: Sampler
    log_density: Callable[[Array, Array], Array] | None = None

    def __post_init__(self) -> None:
        self.action_space = ActionSpaceSpec.continuous(int(self.action_dim))

    def sample(self, states: Array, rng: np.random.Generator, n_samples: int = 1) -> Array:
        """Sample continuous actions from ``mu``."""

        if n_samples <= 0:
            raise ValueError("n_samples must be positive.")
        states_2d = as_2d_float(states, "states")
        draws = np.asarray(self.sampler(states_2d, rng, int(n_samples)), dtype=np.float64)
        return self.action_space.encode_samples(draws, n_rows=states_2d.shape[0], name="normalization samples")

    def log_prob(self, states: Array, actions: Array) -> Array:
        """Return log densities when the policy provides them."""

        if self.log_density is None:
            raise GenPQRConfigurationError("This continuous normalization policy does not expose log_density.")
        states_2d = as_2d_float(states, "states")
        actions_2d = self.action_space.action_matrix(actions, n_rows=states_2d.shape[0])
        return as_1d_float(
            self.log_density(states_2d, actions_2d),
            "log_density",
            n_rows=states_2d.shape[0],
        )


def resolve_normalization_policy(
    normalization_policy: object | None,
    action_space: ActionSpaceSpec,
) -> DiscreteNormalizationPolicy | ContinuousNormalizationPolicy:
    """Resolve a user normalization policy, choosing a finite-action default."""

    if normalization_policy is not None:
        if not hasattr(normalization_policy, "sample"):
            raise GenPQRConfigurationError("normalization_policy must expose sample(states, rng, n_samples).")
        policy_space = getattr(normalization_policy, "action_space", None)
        if policy_space is None:
            raise GenPQRConfigurationError("normalization_policy must expose an action_space attribute.")
        if policy_space is not None and policy_space != action_space:
            raise GenPQRConfigurationError("normalization_policy.action_space does not match action_space.")
        return normalization_policy  # type: ignore[return-value]
    if action_space.kind == "discrete":
        return DiscreteNormalizationPolicy.uniform(int(action_space.n_actions))
    raise GenPQRConfigurationError(
        "Continuous-action GenPQR requires an explicit normalization_policy with a sampler."
    )
