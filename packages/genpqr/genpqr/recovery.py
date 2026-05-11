"""Reward recovery for GenPQR."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from genpqr.types import ActionSpaceSpec, Array, FittedQFunction, NormalizationPolicy
from genpqr.validation import as_1d_float, as_2d_float, normalize_anchor_values


@dataclass
class GenPQRRewardFunction:
    """Recovered normalized reward function.

    Parameters
    ----------
    q_function:
        Fitted Q function for the pseudo-reward ``u-g`` under the normalization
        policy.
    normalization_policy:
        Policy ``mu`` defining the reward normalization.
    anchor_function:
        Callable returning ``g(s)``.
    n_action_samples:
        Monte Carlo sample count for continuous-action ``mu Q``.
    seed:
        Seed used for deterministic reward prediction under sampled
        normalization policies.
    """

    q_function: FittedQFunction
    normalization_policy: NormalizationPolicy
    anchor_function: Callable[[Array], Array] | float = 0.0
    n_action_samples: int = 32
    seed: int = 123
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def action_space(self) -> ActionSpaceSpec:
        """Return the action-space spec."""

        return self.q_function.action_space

    def predict_reward(self, states: Array, actions: Array) -> Array:
        """Predict normalized rewards for state-action rows."""

        states_2d = as_2d_float(states, "states")
        self.action_space.validate_actions(actions, n_rows=states_2d.shape[0])
        q_sa = as_1d_float(self.q_function.predict_q(states_2d, actions), "q_function.predict_q", n_rows=states_2d.shape[0])
        rng = np.random.default_rng(int(self.seed))
        mu_q = as_1d_float(
            self.q_function.expected_q(
                states_2d,
                self.normalization_policy,
                n_action_samples=int(self.n_action_samples),
                rng=rng,
            ),
            "q_function.expected_q",
            n_rows=states_2d.shape[0],
        )
        g = self._anchor_values(states_2d)
        return q_sa - mu_q + g

    def predict_reward_matrix(self, states: Array) -> Array:
        """Predict rewards for every finite action.

        Raises
        ------
        GenPQRConfigurationError
            If called for continuous actions.
        """

        if self.action_space.kind != "discrete":
            raise ValueError("predict_reward_matrix is only available for discrete action spaces.")
        states_2d = as_2d_float(states, "states")
        if hasattr(self.q_function, "predict_q_matrix"):
            q_matrix = self.q_function.predict_q_matrix(states_2d)  # type: ignore[attr-defined]
            q_matrix = np.asarray(q_matrix, dtype=np.float64)
            expected_shape = (states_2d.shape[0], int(self.action_space.n_actions))
            if q_matrix.shape != expected_shape or not np.all(np.isfinite(q_matrix)):
                raise FloatingPointError("q_function.predict_q_matrix returned invalid Q predictions.")
        else:
            cols = []
            for action in range(int(self.action_space.n_actions)):
                cols.append(
                    as_1d_float(
                        self.q_function.predict_q(states_2d, np.full(states_2d.shape[0], action)),
                        "q_function.predict_q",
                        n_rows=states_2d.shape[0],
                    )
                )
            q_matrix = np.stack(cols, axis=1)
        rng = np.random.default_rng(int(self.seed))
        mu_q = as_1d_float(
            self.q_function.expected_q(
                states_2d,
                self.normalization_policy,
                n_action_samples=int(self.n_action_samples),
                rng=rng,
            ),
            "q_function.expected_q",
            n_rows=states_2d.shape[0],
        )
        return q_matrix - mu_q[:, None] + self._anchor_values(states_2d)[:, None]

    def normalization_residual(self, states: Array) -> Array:
        """Return ``E_mu[r(s,A)] - g(s)`` for finite-action diagnostics."""

        if self.action_space.kind != "discrete" or not hasattr(self.normalization_policy, "predict_proba"):
            return np.full(as_2d_float(states, "states").shape[0], np.nan, dtype=np.float64)
        states_2d = as_2d_float(states, "states")
        rewards = self.predict_reward_matrix(states_2d)
        probs = self.normalization_policy.predict_proba(states_2d)  # type: ignore[attr-defined]
        return np.sum(probs * rewards, axis=1) - self._anchor_values(states_2d)

    def mc_standard_error(self, states: Array) -> Array:
        """Estimate Monte Carlo standard error for sampled ``E_mu[Q]`` terms.

        For finite-action normalization policies with exact probabilities, this
        returns ``nan`` because no Monte Carlo approximation is used.
        """

        states_2d = as_2d_float(states, "states")
        if self.action_space.kind == "discrete" and hasattr(self.normalization_policy, "predict_proba"):
            return np.full(states_2d.shape[0], np.nan, dtype=np.float64)
        n_samples = int(self.n_action_samples)
        if n_samples <= 1:
            return np.full(states_2d.shape[0], np.nan, dtype=np.float64)
        rng = np.random.default_rng(int(self.seed))
        samples = self.normalization_policy.sample(states_2d, rng, n_samples)
        arr = self.action_space.encode_samples(samples, n_rows=states_2d.shape[0], name="normalization samples")
        if arr.ndim == 3 and self.action_space.kind == "discrete":
            values = [
                as_1d_float(
                    self.q_function.predict_q(states_2d, arr[:, j, :]),
                    "q_function.predict_q",
                    n_rows=states_2d.shape[0],
                )
                for j in range(arr.shape[1])
            ]
        elif arr.ndim == 2 and self.action_space.kind == "discrete":
            return np.full(states_2d.shape[0], np.nan, dtype=np.float64)
        elif arr.ndim == 3:
            values = [
                as_1d_float(
                    self.q_function.predict_q(states_2d, arr[:, j, :]),
                    "q_function.predict_q",
                    n_rows=states_2d.shape[0],
                )
                for j in range(arr.shape[1])
            ]
        else:
            return np.full(states_2d.shape[0], np.nan, dtype=np.float64)
        stacked = np.stack(values, axis=1)
        return np.std(stacked, axis=1, ddof=1) / np.sqrt(stacked.shape[1])

    def _anchor_values(self, states: Array) -> Array:
        if callable(self.anchor_function):
            raw = self.anchor_function(states)
        else:
            raw = float(self.anchor_function)
        return normalize_anchor_values(raw, states)
