"""High-level GenPQR fitting API."""

from __future__ import annotations

from dataclasses import dataclass, field
import inspect
import time
from typing import Any, Callable, Sequence

import numpy as np

from genpqr.datasets import EpisodeDataset, TransitionDataset, ensure_transition_dataset
from genpqr.deeppqr import DeepPQRAnchorQEstimator
from genpqr.diagnostics import GenPQRDiagnostics, build_diagnostics
from genpqr.exceptions import GenPQRConfigurationError
from genpqr.neural_deeppqr import NeuralDeepPQRAnchorQEstimator
from genpqr.normalization import resolve_normalization_policy
from genpqr.policies import policy_estimator_accepts_parameter, resolve_policy_estimator
from genpqr.q_estimators import resolve_q_estimator
from genpqr.recovery import GenPQRRewardFunction
from genpqr.types import ActionSpaceSpec, Array, EstimatedPolicy, FittedQFunction, NormalizationPolicy
from genpqr.validation import as_1d_float, normalize_anchor_values, validate_gamma


_PRESETS: dict[str, dict[str, Any]] = {
    "airl_fast": {
        "policy": "imitation_airl",
        "q": "neural_fqe",
        "policy_config": {"total_timesteps": 10_000, "demo_batch_size": 256},
    },
    "airl_balanced": {
        "policy": "imitation_airl",
        "q": "neural_fqe",
        "policy_config": {"total_timesteps": 100_000, "demo_batch_size": 512},
    },
    "airl_paper": {
        "policy": "imitation_airl",
        "q": "neural_fqe",
        "policy_config": {"total_timesteps": 500_000, "demo_batch_size": 1024},
        "n_action_samples": 64,
    },
    "gail_fast": {
        "policy": "imitation_gail",
        "q": "neural_fqe",
        "policy_config": {"total_timesteps": 10_000, "demo_batch_size": 256},
    },
    "gail_balanced": {
        "policy": "imitation_gail",
        "q": "neural_fqe",
        "policy_config": {"total_timesteps": 100_000, "demo_batch_size": 512},
    },
    "bc_boosted_fast": {"policy": "behavior_cloning_native", "q": "fqe_boosted"},
    "bc_neural_balanced": {"policy": "behavior_cloning_native", "q": "fqe_neural"},
    "deeppqr_linear": {
        "policy": "behavior_cloning_native",
        "q": DeepPQRAnchorQEstimator(anchor_action=0),
    },
    "deeppqr_neural": {
        "policy": "behavior_cloning_native",
        "q": NeuralDeepPQRAnchorQEstimator(anchor_action=0),
    },
}


def list_presets() -> tuple[str, ...]:
    """Return available GenPQR configuration preset names."""

    return tuple(sorted(_PRESETS))


@dataclass
class GenPQRConfig:
    """Configuration for :func:`fit_genpqr`.

    Parameters
    ----------
    policy:
        Named policy estimator, policy-estimator object, or fitted policy
        exposing ``log_prob``. The default is AIRL through the lazy
        ``imitation`` adapter.
    q:
        Named Q estimator or Q-estimator object. The default is neural FQE.
    n_action_samples:
        Number of normalization-policy samples for continuous-action
        expectations.
    seed:
        Random seed used for deterministic sampling and native estimators.
    policy_config, q_config:
        Keyword arguments passed to named policy/Q estimator constructors.
    """

    policy: str | Any = "airl"
    q: str | Any = "neural_fqe"
    n_action_samples: int = 32
    seed: int = 123
    policy_config: dict[str, Any] = field(default_factory=dict)
    q_config: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_preset(cls, name: str, **overrides: Any) -> "GenPQRConfig":
        """Create a configuration from a named preset."""

        key = str(name).strip().lower()
        if key not in _PRESETS:
            raise GenPQRConfigurationError(f"Unknown GenPQR preset '{name}'.")
        values = dict(_PRESETS[key])
        for dict_key in ("policy_config", "q_config"):
            if dict_key in overrides and dict_key in values:
                merged = dict(values[dict_key])
                merged.update(overrides.pop(dict_key))
                values[dict_key] = merged
        values.update(overrides)
        return cls(**values)


@dataclass
class GenPQRResult:
    """Result returned by :func:`fit_genpqr`."""

    policy: EstimatedPolicy
    q_function: FittedQFunction
    reward_function: GenPQRRewardFunction
    config: GenPQRConfig
    action_space: ActionSpaceSpec
    normalization_policy: NormalizationPolicy
    diagnostics: dict[str, Any]
    diagnostics_report: GenPQRDiagnostics | None = None

    def predict_reward(self, states: Array, actions: Array) -> Array:
        """Convenience wrapper for ``reward_function.predict_reward``."""

        return self.reward_function.predict_reward(states, actions)

    def save(self, path: str) -> None:
        """Save this result to a directory."""

        from genpqr.serialization import save_genpqr_result

        save_genpqr_result(self, path)

    def summary(self) -> dict[str, Any]:
        """Return a compact JSON-safe result summary."""

        return {
            "n_rows": self.diagnostics.get("n_rows"),
            "action_space": self.action_space.kind,
            "policy_neg_log_likelihood": self.diagnostics.get("policy_neg_log_likelihood"),
            "q_backend": self.diagnostics.get("q_backend"),
            "reward_mean": self.diagnostics.get("reward_mean"),
            "reward_std": self.diagnostics.get("reward_std"),
            "normalization_residual_abs_mean": self.diagnostics.get("normalization_residual_abs_mean"),
            "warnings": self.diagnostics.get("warnings", []),
            "warning_codes": self.diagnostics.get("warning_codes", []),
        }


def fit_genpqr(
    *,
    dataset: TransitionDataset | EpisodeDataset | None = None,
    states: Array | None = None,
    actions: Array | None = None,
    next_states: Array | None = None,
    terminals: Array | None = None,
    gamma: float,
    sample_weight: Array | None = None,
    episode_ids: Array | None = None,
    initial_states: Array | None = None,
    initial_actions: Array | None = None,
    env: Any | None = None,
    action_space: ActionSpaceSpec | None = None,
    normalization_policy: NormalizationPolicy | None = None,
    anchor_function: Callable[[Array], Array] | float = 0.0,
    config: GenPQRConfig | None = None,
) -> GenPQRResult:
    """Fit GenPQR and recover a normalized reward.

    Parameters
    ----------
    states, actions, next_states:
        Transition arrays. Discrete actions may be integer indices or one-hot
        rows. Continuous actions must be a 2D array.
    terminals:
        Terminal flags. Omitted flags are treated as nonterminal.
    gamma:
        Discount factor in ``[0, 1)``.
    sample_weight:
        Optional transition-row weights.
    episode_ids, initial_states:
        Accepted for API symmetry and future diagnostics; they are not required
        by the core GenPQR identity.
    env:
        Environment required by adversarial/generator-based policy estimators.
    action_space:
        Explicit action-space spec. If omitted, GenPQR infers discrete actions
        from integer arrays and otherwise treats actions as continuous.
    normalization_policy:
        Normalization policy ``mu``. Finite actions default to uniform; continuous
        actions require an explicit sampler.
    anchor_function:
        Anchor function ``g(s)`` or scalar anchor value.
    config:
        GenPQR configuration. Defaults to AIRL + neural FQE.

    Returns
    -------
    GenPQRResult
        Fitted policy, Q function, reward function, and diagnostics.
    """

    start_time = time.perf_counter()
    cfg = GenPQRConfig() if config is None else config
    gamma_value = validate_gamma(gamma)
    batch = ensure_transition_dataset(
        dataset=dataset,
        states=states,
        actions=actions,
        next_states=next_states,
        terminals=terminals,
        action_space=action_space,
        sample_weight=sample_weight,
        episode_ids=episode_ids,
        initial_states=initial_states,
        initial_actions=initial_actions,
    )
    mu = resolve_normalization_policy(normalization_policy, batch.action_space)
    if mu.action_space != batch.action_space:
        raise GenPQRConfigurationError("normalization_policy action space does not match transition action space.")
    fitted_policy = _fit_or_use_policy(
        cfg.policy,
        cfg.policy_config,
        states=batch.states,
        actions=batch.actions,
        next_states=batch.next_states,
        terminals=batch.terminals,
        action_space=batch.action_space,
        sample_weight=batch.sample_weight,
        env=env,
        seed=cfg.seed,
    )
    _validate_fitted_policy(fitted_policy, batch.action_space, batch.states, batch.actions)
    return _fit_genpqr_with_policy(
        fitted_policy=fitted_policy,
        config=cfg,
        batch=batch,
        normalization_policy=mu,
        gamma_value=gamma_value,
        anchor_function=anchor_function,
        env=env,
        start_time=start_time,
    )


def _fit_genpqr_with_policy(
    *,
    fitted_policy: EstimatedPolicy,
    config: GenPQRConfig,
    batch: TransitionDataset,
    normalization_policy: NormalizationPolicy,
    gamma_value: float,
    anchor_function: Callable[[Array], Array] | float,
    env: Any | None = None,
    start_time: float | None = None,
) -> GenPQRResult:
    """Fit the Q/reward stage with an already fitted policy."""

    cfg = config
    mu = normalization_policy
    g_values = _anchor_values(anchor_function, batch.states)
    log_policy = fitted_policy.log_prob(batch.states, batch.actions)
    log_policy = _validate_row_vector(log_policy, batch.states.shape[0], "policy.log_prob")
    pseudo_rewards = log_policy - g_values
    q_estimator = resolve_q_estimator(cfg.q, **cfg.q_config)
    if not hasattr(q_estimator, "fit"):
        raise GenPQRConfigurationError("q must be a named estimator or expose fit(...).")
    if hasattr(q_estimator, "preflight"):
        q_estimator.preflight(episode_ids=batch.episode_ids, dataset_metadata=batch.metadata, env=env)
    q_function = _fit_q_estimator(
        q_estimator,
        states=batch.states,
        actions=batch.actions,
        next_states=batch.next_states,
        pseudo_rewards=pseudo_rewards,
        normalization_policy=mu,
        gamma=gamma_value,
        terminals=batch.terminals,
        sample_weight=batch.sample_weight,
        policy=fitted_policy,
        episode_ids=batch.episode_ids,
        dataset_metadata=batch.metadata,
    )
    _validate_q_function(q_function, batch.action_space, batch.states, batch.actions, mu)
    reward_function = GenPQRRewardFunction(
        q_function=q_function,
        normalization_policy=mu,
        anchor_function=anchor_function,
        n_action_samples=int(cfg.n_action_samples),
        seed=int(cfg.seed),
    )
    diagnostics_report = build_diagnostics(
        policy=fitted_policy,
        q_function=q_function,
        reward_function=reward_function,
        states=batch.states,
        actions=batch.actions,
        log_policy=log_policy,
        pseudo_rewards=pseudo_rewards,
    )
    diagnostics = diagnostics_report.to_dict()
    diagnostics["fit_time_seconds"] = None if start_time is None else float(time.perf_counter() - start_time)
    diagnostics_report.extra["fit_time_seconds"] = diagnostics["fit_time_seconds"]
    return GenPQRResult(
        policy=fitted_policy,
        q_function=q_function,
        reward_function=reward_function,
        config=cfg,
        action_space=batch.action_space,
        normalization_policy=mu,
        diagnostics=diagnostics,
        diagnostics_report=diagnostics_report,
    )


def fit_genpqr_auto(
    *,
    dataset: TransitionDataset | EpisodeDataset | None = None,
    states: Array | None = None,
    actions: Array | None = None,
    next_states: Array | None = None,
    terminals: Array | None = None,
    gamma: float,
    policy_candidates: Sequence[str | Any] = ("behavior_cloning_native",),
    q_candidates: Sequence[str | Any] = ("fqe_boosted",),
    n_action_samples: int = 32,
    seed: int = 123,
    policy_config: dict[str, Any] | None = None,
    q_config: dict[str, Any] | None = None,
    **kwargs: Any,
) -> GenPQRResult:
    """Fit a small proxy-only GenPQR candidate sweep.

    The selector uses internal diagnostics only: finite predictions, reward
    normalization residuals when available, and backend Bellman risk if exposed.
    It never uses oracle rewards or target-policy Monte Carlo values.
    """

    best_result: GenPQRResult | None = None
    best_score = np.inf
    errors: list[str] = []
    for policy in policy_candidates:
        for q in q_candidates:
            try:
                cfg = GenPQRConfig(
                    policy=policy,
                    q=q,
                    n_action_samples=int(n_action_samples),
                    seed=int(seed),
                    policy_config=dict(policy_config or {}),
                    q_config=dict(q_config or {}),
                )
                result = fit_genpqr(
                    dataset=dataset,
                    states=states,
                    actions=actions,
                    next_states=next_states,
                    terminals=terminals,
                    gamma=gamma,
                    config=cfg,
                    **kwargs,
                )
                score = _auto_score(result)
            except Exception as exc:  # pragma: no cover - exercised by integration users.
                errors.append(f"{policy}/{q}: {exc}")
                continue
            if score < best_score:
                best_score = score
                best_result = result
    if best_result is None:
        raise GenPQRConfigurationError("All GenPQR auto candidates failed: " + "; ".join(errors))
    best_result.diagnostics["auto_score"] = float(best_score)
    best_result.diagnostics["auto_errors"] = errors
    return best_result


def _fit_or_use_policy(
    policy: str | Any,
    policy_config: dict[str, Any],
    *,
    states: Array,
    actions: Array,
    next_states: Array,
    terminals: Array,
    action_space: ActionSpaceSpec,
    sample_weight: Array | None,
    env: Any | None,
    seed: int,
) -> EstimatedPolicy:
    if not isinstance(policy, str) and hasattr(policy, "log_prob") and hasattr(policy, "sample"):
        if getattr(policy, "action_space", action_space) != action_space:
            raise GenPQRConfigurationError("Provided fitted policy action_space does not match action_space.")
        return policy
    config = dict(policy_config)
    if isinstance(policy, str) and policy_estimator_accepts_parameter(policy, "seed"):
        config.setdefault("seed", seed)
    estimator = resolve_policy_estimator(policy, **config)
    if not hasattr(estimator, "fit"):
        raise GenPQRConfigurationError("policy must be a fitted policy or expose fit(...).")
    if hasattr(estimator, "preflight"):
        estimator.preflight(env=env)
    return estimator.fit(
        states=states,
        actions=actions,
        next_states=next_states,
        terminals=terminals,
        action_space=action_space,
        sample_weight=sample_weight,
        env=env,
    )


def _anchor_values(anchor_function: Callable[[Array], Array] | float, states: Array) -> Array:
    raw = anchor_function(states) if callable(anchor_function) else float(anchor_function)
    return normalize_anchor_values(raw, states)


def _validate_fitted_policy(policy: EstimatedPolicy, action_space: ActionSpaceSpec, states: Array, actions: Array) -> None:
    if getattr(policy, "action_space", None) != action_space:
        raise GenPQRConfigurationError("Fitted policy action_space does not match the transition action_space.")
    if not hasattr(policy, "log_prob") or not hasattr(policy, "sample"):
        raise GenPQRConfigurationError("Fitted policy must expose log_prob(...) and sample(...).")
    _validate_row_vector(policy.log_prob(states[: min(5, states.shape[0])], actions[: min(5, states.shape[0])]), min(5, states.shape[0]), "policy.log_prob")


def _validate_q_function(
    q_function: FittedQFunction,
    action_space: ActionSpaceSpec,
    states: Array,
    actions: Array,
    normalization_policy: NormalizationPolicy,
) -> None:
    if getattr(q_function, "action_space", None) != action_space:
        raise GenPQRConfigurationError("Fitted Q function action_space does not match the transition action_space.")
    if not hasattr(q_function, "predict_q") or not hasattr(q_function, "expected_q"):
        raise GenPQRConfigurationError("Fitted Q function must expose predict_q(...) and expected_q(...).")
    n_probe = min(5, states.shape[0])
    _validate_row_vector(q_function.predict_q(states[:n_probe], actions[:n_probe]), n_probe, "q_function.predict_q")
    _validate_row_vector(
        q_function.expected_q(
            states[:n_probe],
            normalization_policy,
            n_action_samples=2,
            rng=np.random.default_rng(0),
        ),
        n_probe,
        "q_function.expected_q",
    )


def _validate_row_vector(values: Array, n_rows: int, name: str) -> Array:
    try:
        arr = as_1d_float(values, name, n_rows=n_rows)
    except ValueError:
        raise GenPQRConfigurationError(f"{name} must return one value per row.")
    return arr


def _auto_score(result: GenPQRResult) -> float:
    residual = result.diagnostics.get("normalization_residual_abs_mean")
    score = 0.0 if residual is None else float(residual)
    q_risk = result.diagnostics.get("q_validation_bellman_risk")
    if q_risk is not None and np.isfinite(q_risk):
        score += 0.01 * float(q_risk)
    return score


def _fit_q_estimator(q_estimator: Any, **kwargs: Any) -> FittedQFunction:
    signature = inspect.signature(q_estimator.fit)
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return q_estimator.fit(**kwargs)
    accepted = {name: value for name, value in kwargs.items() if name in signature.parameters}
    return q_estimator.fit(**accepted)


__all__ = ["DeepPQRAnchorQEstimator", "GenPQRConfig", "GenPQRResult", "fit_genpqr", "fit_genpqr_auto", "list_presets"]
