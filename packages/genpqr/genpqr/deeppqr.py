"""DeepPQR anchor-Q backend."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from genpqr.exceptions import GenPQRConfigurationError
from genpqr.types import ActionSpaceSpec, Array, EstimatedPolicy, NormalizationPolicy
from genpqr.validation import as_1d_float, as_2d_float, optional_terminals, optional_weights


def _standardize_fit(states: Array) -> tuple[Array, Array]:
    mean = np.mean(states, axis=0)
    std = np.std(states, axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return mean, std


def _features(states: Array, mean: Array, std: Array) -> Array:
    x = (as_2d_float(states, "states") - mean) / std
    return np.concatenate([np.ones((x.shape[0], 1)), x, x**2], axis=1)


@dataclass
class DeepPQRAnchorQEstimator:
    """DeepPQR-style state-only anchor-Q estimator.

    The estimator fits ``W(s)=Q(s,a_anchor)`` only on rows whose observed action
    is ``anchor_action``. It then reconstructs the full stratified Q function as

    ``Q(s,a) = W(s) + alpha * [log pi(a|s) - log pi(a_anchor|s)]``.
    """

    anchor_action: int = 0
    alpha: float = 1.0
    ridge: float = 1e-3
    n_iterations: int = 80
    n_action_samples: int = 16
    weak_anchor_fraction: float = 0.05
    diagnostics: dict[str, Any] = field(default_factory=dict)

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
    ) -> "DeepPQRStratifiedQFunction":
        """Fit the DeepPQR state-only anchor value and return stratified Q."""

        if not np.isclose(float(self.alpha), 1.0):
            raise GenPQRConfigurationError(
                "DeepPQRAnchorQEstimator currently supports alpha=1.0 only. "
                "For temperature-scaled variants, pass a policy whose log_prob already includes the desired scaling."
            )
        if policy is None:
            raise GenPQRConfigurationError("DeepPQRAnchorQEstimator requires the fitted behavior policy.")
        action_space = normalization_policy.action_space
        if action_space.kind != "discrete":
            raise GenPQRConfigurationError("DeepPQRAnchorQEstimator only supports discrete actions.")
        if self.anchor_action < 0 or self.anchor_action >= int(action_space.n_actions):
            raise ValueError("anchor_action is out of bounds.")
        states_2d = as_2d_float(states, "states")
        next_states_2d = as_2d_float(next_states, "next_states", n_rows=states_2d.shape[0])
        action_idx = action_space.action_indices(actions, n_rows=states_2d.shape[0])
        n_rows = states_2d.shape[0]
        pseudo_rewards_1d = as_1d_float(pseudo_rewards, "pseudo_rewards", n_rows=n_rows)
        terminals_1d = optional_terminals(terminals, n_rows)
        weights = np.ones(n_rows, dtype=np.float64) if sample_weight is None else optional_weights(sample_weight, n_rows)
        mask = action_idx == int(self.anchor_action)
        anchor_count = int(np.sum(mask))
        if anchor_count == 0:
            raise GenPQRConfigurationError("DeepPQR anchor-Q fit has zero anchor-action rows.")
        mean, std = _standardize_fit(states_2d)
        x_anchor = _features(states_2d[mask], mean, std)
        x_next_anchor = _features(next_states_2d[mask], mean, std)
        u_minus_g_anchor = pseudo_rewards_1d[mask]
        done_anchor = terminals_1d[mask]
        w_anchor = weights[mask]
        if not np.any(w_anchor > 0.0):
            raise GenPQRConfigurationError("DeepPQR anchor-Q fit has zero positive-weight anchor-action rows.")
        shift_next = _normalization_log_ratio_shift(
            policy=policy,
            states=next_states_2d[mask],
            normalization_policy=normalization_policy,
            action_space=action_space,
            anchor_action=int(self.anchor_action),
            n_action_samples=int(self.n_action_samples),
        )
        coef = np.zeros(x_anchor.shape[1], dtype=np.float64)
        penalty = float(self.ridge) * np.eye(x_anchor.shape[1], dtype=np.float64)
        for _ in range(int(self.n_iterations)):
            mu_q_next = x_next_anchor @ coef + shift_next
            targets = u_minus_g_anchor + float(gamma) * (1.0 - done_anchor) * mu_q_next
            xtw = x_anchor.T * w_anchor.reshape(1, -1)
            coef = np.linalg.solve(xtw @ x_anchor + penalty, xtw @ targets)
        anchor_probs = _anchor_probabilities(policy, states_2d, action_space, int(self.anchor_action))
        diagnostics = {
            "backend": "deep_pqr_anchor",
            "anchor_action": int(self.anchor_action),
            "anchor_count": anchor_count,
            "weighted_anchor_count": float(np.sum(w_anchor)),
            "anchor_fraction": float(anchor_count / states_2d.shape[0]),
            "mean_anchor_policy_probability": float(np.mean(anchor_probs)),
            "weak_anchor_support": bool(anchor_count / states_2d.shape[0] < float(self.weak_anchor_fraction)),
            "normalization_log_ratio_shift_mean": float(np.mean(shift_next)),
        }
        self.diagnostics = diagnostics
        return DeepPQRStratifiedQFunction(
            coef=coef,
            input_mean=mean,
            input_std=std,
            policy=policy,
            action_space=action_space,
            anchor_action=int(self.anchor_action),
            alpha=float(self.alpha),
            diagnostics=diagnostics,
        )


@dataclass
class DeepPQRStratifiedQFunction:
    """Fitted DeepPQR stratified Q function."""

    coef: Array
    input_mean: Array
    input_std: Array
    policy: EstimatedPolicy
    action_space: ActionSpaceSpec
    anchor_action: int
    alpha: float = 1.0
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def predict_anchor_value(self, states: Array) -> Array:
        """Predict the state-only anchor value ``W(s)``."""

        return _features(states, self.input_mean, self.input_std) @ self.coef

    def predict_q(self, states: Array, actions: Array) -> Array:
        """Predict reconstructed DeepPQR Q-values."""

        states_2d = as_2d_float(states, "states")
        idx = self.action_space.action_indices(actions, n_rows=states_2d.shape[0])
        anchor = np.full(states_2d.shape[0], int(self.anchor_action), dtype=np.int64)
        logp_action = self.policy.log_prob(states_2d, idx)
        logp_anchor = self.policy.log_prob(states_2d, anchor)
        return self.predict_anchor_value(states_2d) + float(self.alpha) * (logp_action - logp_anchor)

    def predict_q_matrix(self, states: Array) -> Array:
        """Predict all finite-action Q-values."""

        states_2d = as_2d_float(states, "states")
        cols = []
        for action in range(int(self.action_space.n_actions)):
            cols.append(self.predict_q(states_2d, np.full(states_2d.shape[0], action, dtype=np.int64)))
        return np.stack(cols, axis=1)

    def expected_q(
        self,
        states: Array,
        normalization_policy: NormalizationPolicy,
        *,
        n_action_samples: int,
        rng: np.random.Generator,
    ) -> Array:
        """Estimate ``E_mu[Q(s, A)]``."""

        states_2d = as_2d_float(states, "states")
        if hasattr(normalization_policy, "predict_proba"):
            probs = normalization_policy.predict_proba(states_2d)  # type: ignore[attr-defined]
            return np.sum(probs * self.predict_q_matrix(states_2d), axis=1)
        samples = normalization_policy.sample(states_2d, rng, int(n_action_samples))
        encoded = self.action_space.encode_samples(samples, n_rows=states_2d.shape[0], name="normalization samples")
        if encoded.ndim == 2:
            return self.predict_q(states_2d, encoded)
        values = [self.predict_q(states_2d, encoded[:, j, :]) for j in range(encoded.shape[1])]
        return np.mean(np.stack(values, axis=1), axis=1)


def _anchor_probabilities(policy: EstimatedPolicy, states: Array, action_space: ActionSpaceSpec, anchor_action: int | Array) -> Array:
    if action_space.kind == "discrete" and hasattr(policy, "predict_proba"):
        probs = policy.predict_proba(states)  # type: ignore[attr-defined]
        return np.asarray(probs, dtype=np.float64)[:, int(anchor_action)]
    states_2d = as_2d_float(states, "states")
    if action_space.kind == "discrete":
        anchor = np.full(states_2d.shape[0], int(anchor_action), dtype=np.int64)
    else:
        anchor = action_space.action_matrix(anchor_action, n_rows=states_2d.shape[0], name="anchor_action")
    return np.exp(policy.log_prob(states_2d, anchor))


def _normalization_log_ratio_shift(
    *,
    policy: EstimatedPolicy,
    states: Array,
    normalization_policy: NormalizationPolicy,
    action_space: ActionSpaceSpec,
    anchor_action: int | Array,
    n_action_samples: int,
) -> Array:
    states_2d = as_2d_float(states, "states")
    if action_space.kind == "discrete":
        anchor = np.full(states_2d.shape[0], int(anchor_action), dtype=np.int64)
    else:
        anchor = action_space.action_matrix(anchor_action, n_rows=states_2d.shape[0], name="anchor_action")
    logp_anchor = policy.log_prob(states_2d, anchor)
    if action_space.kind == "discrete" and hasattr(normalization_policy, "predict_proba"):
        probs = normalization_policy.predict_proba(states_2d)  # type: ignore[attr-defined]
        logp_cols = []
        for action in range(int(action_space.n_actions)):
            action_vec = np.full(states_2d.shape[0], action, dtype=np.int64)
            logp_cols.append(policy.log_prob(states_2d, action_vec))
        expected_logp = np.sum(probs * np.stack(logp_cols, axis=1), axis=1)
        return expected_logp - logp_anchor
    rng = np.random.default_rng(123)
    samples = action_space.encode_samples(
        normalization_policy.sample(states_2d, rng, int(n_action_samples)),
        n_rows=states_2d.shape[0],
        name="normalization samples",
    )
    if samples.ndim == 2:
        return policy.log_prob(states_2d, samples) - logp_anchor
    values = []
    for j in range(samples.shape[1]):
        sample_actions = samples[:, j] if action_space.kind == "discrete" else samples[:, j, :]
        values.append(policy.log_prob(states_2d, sample_actions) - logp_anchor)
    return np.mean(np.stack(values, axis=1), axis=1)
