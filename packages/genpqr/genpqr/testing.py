"""Contract checks for GenPQR extension authors."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np

from genpqr.exceptions import GenPQRConfigurationError
from genpqr.normalization import DiscreteNormalizationPolicy
from genpqr.types import ActionSpaceSpec, Array, EstimatedPolicy, FittedQFunction, NormalizationPolicy
from genpqr.validation import as_1d_float


def check_estimated_policy_contract(
    policy: EstimatedPolicy,
    *,
    states: Array,
    actions: Array,
    action_space: ActionSpaceSpec,
) -> None:
    """Validate a fitted policy object against the GenPQR protocol."""

    if getattr(policy, "action_space", None) != action_space:
        raise AssertionError("policy.action_space does not match action_space.")
    n_rows = np.asarray(states).shape[0]
    try:
        logp = as_1d_float(policy.log_prob(states, actions), "policy.log_prob", n_rows=n_rows)
    except ValueError as exc:
        raise AssertionError("policy.log_prob must return one finite value per row.") from exc
    if not np.all(np.isfinite(logp)):
        raise AssertionError("policy.log_prob must return one finite value per row.")
    sample_one = policy.sample(states, np.random.default_rng(0), n_samples=1)
    action_space.validate_actions(sample_one, n_rows=n_rows, name="policy samples")
    sample_many = policy.sample(states, np.random.default_rng(0), n_samples=3)
    if action_space.kind == "discrete":
        arr = np.asarray(sample_many)
        if arr.shape != (n_rows, 3):
            raise AssertionError("discrete policy samples must have shape (n, n_samples).")
        if hasattr(policy, "predict_proba"):
            probs = np.asarray(policy.predict_proba(states), dtype=np.float64)
            if probs.shape != (n_rows, int(action_space.n_actions)):
                raise AssertionError("predict_proba must have shape (n, n_actions).")
            if np.any(probs < 0.0) or not np.allclose(probs.sum(axis=1), 1.0):
                raise AssertionError("predict_proba rows must be valid probabilities.")
    else:
        arr = np.asarray(sample_many)
        if arr.shape != (n_rows, 3, int(action_space.action_dim)):
            raise AssertionError("continuous policy samples must have shape (n, n_samples, action_dim).")


def check_policy_estimator_contract(
    estimator: Any,
    *,
    states: Array,
    actions: Array,
    next_states: Array,
    terminals: Array,
    action_space: ActionSpaceSpec,
) -> None:
    """Fit and validate a policy estimator."""

    if not hasattr(estimator, "fit"):
        raise AssertionError("policy estimator must expose fit(...).")
    fitted = estimator.fit(
        states=states,
        actions=actions,
        next_states=next_states,
        terminals=terminals,
        action_space=action_space,
    )
    check_estimated_policy_contract(fitted, states=states, actions=actions, action_space=action_space)


def check_fitted_q_contract(
    q_function: FittedQFunction,
    *,
    states: Array,
    actions: Array,
    action_space: ActionSpaceSpec,
    normalization_policy: NormalizationPolicy | None = None,
) -> None:
    """Validate a fitted Q function object against the GenPQR protocol."""

    if getattr(q_function, "action_space", None) != action_space:
        raise AssertionError("q_function.action_space does not match action_space.")
    n_rows = np.asarray(states).shape[0]
    try:
        q_values = as_1d_float(q_function.predict_q(states, actions), "q_function.predict_q", n_rows=n_rows)
    except ValueError as exc:
        raise AssertionError("predict_q must return one finite value per row.") from exc
    if not np.all(np.isfinite(q_values)):
        raise AssertionError("predict_q must return one finite value per row.")
    mu = normalization_policy
    if mu is None:
        if action_space.kind != "discrete":
            raise GenPQRConfigurationError("continuous Q contract checks require normalization_policy.")
        mu = DiscreteNormalizationPolicy.uniform(int(action_space.n_actions))
    try:
        expected = as_1d_float(
            q_function.expected_q(states, mu, n_action_samples=3, rng=np.random.default_rng(0)),
            "q_function.expected_q",
            n_rows=n_rows,
        )
    except ValueError as exc:
        raise AssertionError("expected_q must return one finite value per row.") from exc
    if not np.all(np.isfinite(expected)):
        raise AssertionError("expected_q must return one finite value per row.")
    if action_space.kind == "discrete" and hasattr(mu, "predict_proba") and hasattr(q_function, "predict_q_matrix"):
        manual = np.sum(mu.predict_proba(states) * q_function.predict_q_matrix(states), axis=1)  # type: ignore[attr-defined]
        if not np.allclose(expected, manual):
            raise AssertionError("expected_q must agree with finite-action averaging.")


def check_q_estimator_contract(
    estimator: Any,
    *,
    states: Array,
    actions: Array,
    next_states: Array,
    pseudo_rewards: Array,
    normalization_policy: NormalizationPolicy,
    gamma: float,
    terminals: Array,
    policy: EstimatedPolicy | None = None,
) -> None:
    """Fit and validate a Q estimator."""

    if not hasattr(estimator, "fit"):
        raise AssertionError("Q estimator must expose fit(...).")
    fitted = estimator.fit(
        states=states,
        actions=actions,
        next_states=next_states,
        pseudo_rewards=pseudo_rewards,
        normalization_policy=normalization_policy,
        gamma=gamma,
        terminals=terminals,
        policy=policy,
    )
    check_fitted_q_contract(
        fitted,
        states=states,
        actions=actions,
        action_space=normalization_policy.action_space,
        normalization_policy=normalization_policy,
    )


def assert_deepcopyable(obj: Any) -> None:
    """Assert that an object can be deep-copied for cross-fitting."""

    try:
        copy.deepcopy(obj)
    except Exception as exc:  # pragma: no cover - depends on user objects.
        raise AssertionError("object must be deepcopy-able for cross-fitting.") from exc
