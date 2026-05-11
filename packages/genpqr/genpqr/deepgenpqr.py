"""Neural DeepGenPQR user-facing API."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import time
from typing import Any, Callable, Literal

from genpqr.api import GenPQRConfig, GenPQRResult, _fit_genpqr_with_policy, _fit_or_use_policy, _validate_fitted_policy, fit_genpqr
from genpqr.datasets import EpisodeDataset, TransitionDataset, ensure_transition_dataset
from genpqr.deeppqr import DeepPQRAnchorQEstimator
from genpqr.exceptions import GenPQRConfigurationError
from genpqr.neural_deeppqr import NeuralDeepPQRAnchorQEstimator
from genpqr.normalization import ContinuousNormalizationPolicy, DiscreteNormalizationPolicy, resolve_normalization_policy
from genpqr.types import ActionSpaceSpec, Array, NormalizationPolicy
from genpqr.validation import validate_gamma


DeepGenPQRQMode = Literal["pooled_fqe", "anchor_deeppqr"]
DeepGenPQRAnchorFallback = Literal["error", "pooled_fqe"]


_DEEPGENPQR_PRESETS: dict[str, dict[str, Any]] = {
    "deepgenpqr_airl_fqe_fast": {
        "policy": "deep_airl",
        "q_mode": "pooled_fqe",
        "q_backend": "auto_neural_fqe",
        "policy_config": {"total_timesteps": 10_000, "demo_batch_size": 256},
        "q_config": {"n_next_action_samples": 8},
    },
    "deepgenpqr_airl_fqe_balanced": {
        "policy": "deep_airl",
        "q_mode": "pooled_fqe",
        "q_backend": "auto_neural_fqe",
        "policy_config": {"total_timesteps": 100_000, "demo_batch_size": 512},
        "q_config": {"n_next_action_samples": 16},
        "n_action_samples": 64,
    },
    "deepgenpqr_airl_anchor_fast": {
        "policy": "deep_airl",
        "q_mode": "anchor_deeppqr",
        "anchor_backend": "neural_deeppqr",
        "policy_config": {"total_timesteps": 10_000, "demo_batch_size": 256},
        "q_config": {"max_epochs": 100, "patience": 10},
    },
    "deepgenpqr_airl_anchor_balanced": {
        "policy": "deep_airl",
        "q_mode": "anchor_deeppqr",
        "anchor_backend": "neural_deeppqr",
        "policy_config": {"total_timesteps": 100_000, "demo_batch_size": 512},
        "q_config": {"max_epochs": 500, "patience": 20},
        "n_action_samples": 64,
    },
    "deepgenpqr_gail_fqe_balanced": {
        "policy": "deep_gail",
        "q_mode": "pooled_fqe",
        "q_backend": "auto_neural_fqe",
        "policy_config": {"total_timesteps": 100_000, "demo_batch_size": 512},
        "q_config": {"n_next_action_samples": 16},
        "n_action_samples": 64,
    },
    "deepgenpqr_bc_fqe_debug": {
        "policy": "behavior_cloning_native",
        "q_mode": "pooled_fqe",
        "q_backend": "auto_neural_fqe",
        "policy_config": {"n_epochs": 50},
        "q_config": {"n_next_action_samples": 4},
    },
}


def list_deepgenpqr_presets() -> tuple[str, ...]:
    """Return available DeepGenPQR preset names."""

    return tuple(sorted(_DEEPGENPQR_PRESETS))


@dataclass
class DeepGenPQRConfig:
    """Configuration for :func:`fit_deep_genpqr`.

    Parameters
    ----------
    policy:
        Deep policy estimator name, estimator object, or fitted policy. The
        default ``"deep_airl"`` resolves to the lazy HumanCompatibleAI
        ``imitation`` AIRL adapter.
    q_mode:
        ``"pooled_fqe"`` fits neural FQE over all rows. ``"anchor_deeppqr"``
        fits a DeepPQR-style state-only anchor Q backend on anchor rows.
    q_backend:
        Named Q estimator or object used in pooled mode. The default
        ``"auto_neural_fqe"`` routes finite-action data to the action-head
        neural FQE backend and continuous-action data to generic neural FQE.
    anchor_backend:
        Named Q estimator or object used in anchor mode.
    anchor_action:
        Discrete anchor action index, continuous fixed anchor action, or a
        callable returning continuous anchor actions for query states.
    anchor_selector:
        Optional continuous-action callback returning a boolean mask of rows to
        use for the anchor-value fit.
    anchor_tolerance:
        Absolute tolerance used to identify continuous fixed-anchor rows when
        ``anchor_selector`` is not supplied.
    min_anchor_count:
        Minimum positive number of anchor rows required by the neural anchor
        backend.
    anchor_fallback:
        ``"error"`` keeps anchor-mode fitting strict. ``"pooled_fqe"`` refits
        with pooled neural FQE if anchor support is unavailable or weak.
    """

    policy: str | Any = "deep_airl"
    q_mode: DeepGenPQRQMode = "pooled_fqe"
    q_backend: str | Any = "auto_neural_fqe"
    anchor_backend: str | Any = "neural_deeppqr"
    anchor_action: int | float | Array | Callable[[Array], Array] = 0
    anchor_selector: Callable[[Array, Array], Array] | None = None
    anchor_tolerance: float = 1e-8
    min_anchor_count: int = 5
    anchor_fallback: DeepGenPQRAnchorFallback = "error"
    seed: int = 123
    n_action_samples: int = 32
    policy_config: dict[str, Any] = field(default_factory=dict)
    q_config: dict[str, Any] = field(default_factory=dict)
    normalization_config: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_preset(cls, name: str, **overrides: Any) -> "DeepGenPQRConfig":
        """Create a DeepGenPQR configuration from a named preset."""

        key = str(name).strip().lower()
        if key not in _DEEPGENPQR_PRESETS:
            raise GenPQRConfigurationError(f"Unknown DeepGenPQR preset '{name}'.")
        values = dict(_DEEPGENPQR_PRESETS[key])
        for dict_key in ("policy_config", "q_config", "normalization_config"):
            if dict_key in overrides and dict_key in values:
                merged = dict(values[dict_key])
                merged.update(overrides.pop(dict_key))
                values[dict_key] = merged
        values.update(overrides)
        return cls(**values)


@dataclass
class DeepGenPQRResult:
    """Result returned by :func:`fit_deep_genpqr`."""

    genpqr_result: GenPQRResult
    config: DeepGenPQRConfig
    q_mode: DeepGenPQRQMode
    policy_backend: str
    q_backend: str
    diagnostics: dict[str, Any]

    @property
    def policy(self) -> Any:
        """Return the fitted policy."""

        return self.genpqr_result.policy

    @property
    def q_function(self) -> Any:
        """Return the fitted Q function."""

        return self.genpqr_result.q_function

    @property
    def reward_function(self) -> Any:
        """Return the recovered reward function."""

        return self.genpqr_result.reward_function

    @property
    def action_space(self) -> ActionSpaceSpec:
        """Return the fitted action space."""

        return self.genpqr_result.action_space

    @property
    def normalization_policy(self) -> NormalizationPolicy:
        """Return the normalization policy."""

        return self.genpqr_result.normalization_policy

    def predict_reward(self, states: Array, actions: Array) -> Array:
        """Predict recovered rewards for state-action rows."""

        return self.genpqr_result.predict_reward(states, actions)

    def save(self, path: str) -> None:
        """Save this DeepGenPQR result to a directory."""

        from genpqr.serialization import save_deep_genpqr_result

        save_deep_genpqr_result(self, path)

    def summary(self) -> dict[str, Any]:
        """Return a compact JSON-safe DeepGenPQR summary."""

        base = self.genpqr_result.summary()
        base.update(
            {
                "deepgenpqr_mode": self.q_mode,
                "policy_backend": self.policy_backend,
                "q_backend": self.q_backend,
                "pooled_actions": self.diagnostics.get("pooled_actions"),
                "anchor_enabled": self.diagnostics.get("anchor_enabled"),
                "anchor_support": self.diagnostics.get("anchor_support"),
                "normalization_mc_se": self.diagnostics.get("normalization_mc_se"),
            }
        )
        for key in (
            "benchmark_reward_rmse",
            "benchmark_reward_mae",
            "benchmark_reward_correlation",
            "benchmark_anchor_rmse",
            "benchmark_action_ranking_accuracy",
            "benchmark_anchor_support_fraction",
        ):
            if key in self.diagnostics:
                base[key] = self.diagnostics.get(key)
        return base


def fit_deep_genpqr(
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
    config: DeepGenPQRConfig | None = None,
) -> DeepGenPQRResult:
    """Fit the neural DeepGenPQR workflow.

    DeepGenPQR is a production neural wrapper over :func:`fit_genpqr`.
    ``q_mode="pooled_fqe"`` uses neural FQE over all action rows.
    ``q_mode="anchor_deeppqr"`` uses the neural DeepPQR anchor-Q backend.
    """

    start_time = time.perf_counter()
    cfg = DeepGenPQRConfig() if config is None else config
    if cfg.q_mode not in {"pooled_fqe", "anchor_deeppqr"}:
        raise GenPQRConfigurationError("DeepGenPQR q_mode must be 'pooled_fqe' or 'anchor_deeppqr'.")
    if cfg.anchor_fallback not in {"error", "pooled_fqe"}:
        raise GenPQRConfigurationError("DeepGenPQR anchor_fallback must be 'error' or 'pooled_fqe'.")
    gen_config = _to_genpqr_config(cfg)
    resolved_normalization_policy = _normalization_from_config(
        explicit_policy=normalization_policy,
        normalization_config=cfg.normalization_config,
        dataset=dataset,
        action_space=action_space,
    )
    fit_kwargs = {
        "dataset": dataset,
        "states": states,
        "actions": actions,
        "next_states": next_states,
        "terminals": terminals,
        "gamma": gamma,
        "sample_weight": sample_weight,
        "episode_ids": episode_ids,
        "initial_states": initial_states,
        "initial_actions": initial_actions,
        "env": env,
        "action_space": action_space,
        "normalization_policy": resolved_normalization_policy,
        "anchor_function": anchor_function,
    }
    fallback_reason = None
    active_cfg = cfg
    if cfg.q_mode == "anchor_deeppqr":
        result, active_cfg, gen_config, fallback_reason = _fit_anchor_mode_with_policy_reuse(
            cfg=cfg,
            gen_config=gen_config,
            fit_kwargs=fit_kwargs,
            gamma=gamma,
            anchor_function=anchor_function,
            start_time=start_time,
        )
    else:
        result = fit_genpqr(config=gen_config, **fit_kwargs)
    diagnostics = _build_deep_diagnostics(
        result=result,
        config=active_cfg,
        fit_time_seconds=float(time.perf_counter() - start_time),
        requested_config=cfg,
        fallback_reason=fallback_reason,
    )
    result.diagnostics.update(diagnostics)
    if result.diagnostics_report is not None:
        result.diagnostics_report.extra.update(diagnostics)
    return DeepGenPQRResult(
        genpqr_result=result,
        config=active_cfg,
        q_mode=active_cfg.q_mode,
        policy_backend=_backend_name(gen_config.policy),
        q_backend=result.diagnostics.get("q_backend") or _backend_name(gen_config.q),
        diagnostics=result.diagnostics,
    )


def _to_genpqr_config(config: DeepGenPQRConfig) -> GenPQRConfig:
    policy = _resolve_deep_policy_name(config.policy)
    q_config = dict(config.q_config)
    if config.q_mode == "pooled_fqe":
        _reject_anchor_backend_in_pooled_mode(config.q_backend)
        q = config.q_backend
    else:
        q = _resolve_anchor_backend_name(config.anchor_backend)
        if isinstance(q, str):
            q_config.update(_anchor_q_config(config, q))
    return GenPQRConfig(
        policy=policy,
        q=q,
        n_action_samples=int(config.n_action_samples),
        seed=int(config.seed),
        policy_config=dict(config.policy_config),
        q_config=q_config,
    )


def _anchor_q_config(config: DeepGenPQRConfig, backend_name: str) -> dict[str, Any]:
    base: dict[str, Any] = {
        "anchor_action": config.anchor_action,
        "n_action_samples": config.n_action_samples,
    }
    if backend_name in {"deeppqr_neural", "neural_deeppqr", "deep_pqr_neural"}:
        base.update(
            {
                "anchor_selector": config.anchor_selector,
                "anchor_tolerance": config.anchor_tolerance,
                "min_anchor_count": config.min_anchor_count,
                "seed": config.seed,
            }
        )
    return base


def _pooled_fallback_config(config: DeepGenPQRConfig) -> DeepGenPQRConfig:
    return replace(
        config,
        q_mode="pooled_fqe",
        q_config={},
        anchor_fallback="error",
    )


def _fit_anchor_mode_with_policy_reuse(
    *,
    cfg: DeepGenPQRConfig,
    gen_config: GenPQRConfig,
    fit_kwargs: dict[str, Any],
    gamma: float,
    anchor_function: Callable[[Array], Array] | float,
    start_time: float,
) -> tuple[GenPQRResult, DeepGenPQRConfig, GenPQRConfig, str | None]:
    batch = ensure_transition_dataset(
        dataset=fit_kwargs["dataset"],
        states=fit_kwargs["states"],
        actions=fit_kwargs["actions"],
        next_states=fit_kwargs["next_states"],
        terminals=fit_kwargs["terminals"],
        action_space=fit_kwargs["action_space"],
        sample_weight=fit_kwargs["sample_weight"],
        episode_ids=fit_kwargs["episode_ids"],
        initial_states=fit_kwargs["initial_states"],
        initial_actions=fit_kwargs["initial_actions"],
    )
    gamma_value = validate_gamma(gamma)
    mu = resolve_normalization_policy(fit_kwargs["normalization_policy"], batch.action_space)
    if mu.action_space != batch.action_space:
        raise GenPQRConfigurationError("normalization_policy action space does not match transition action space.")
    fitted_policy = _fit_or_use_policy(
        gen_config.policy,
        gen_config.policy_config,
        states=batch.states,
        actions=batch.actions,
        next_states=batch.next_states,
        terminals=batch.terminals,
        action_space=batch.action_space,
        sample_weight=batch.sample_weight,
        env=fit_kwargs["env"],
        seed=gen_config.seed,
    )
    _validate_fitted_policy(fitted_policy, batch.action_space, batch.states, batch.actions)
    fallback_reason = None
    active_cfg = cfg
    active_gen_config = gen_config
    try:
        result = _fit_genpqr_with_policy(
            fitted_policy=fitted_policy,
            config=active_gen_config,
            batch=batch,
            normalization_policy=mu,
            gamma_value=gamma_value,
            anchor_function=anchor_function,
            env=fit_kwargs["env"],
            start_time=start_time,
        )
    except GenPQRConfigurationError as exc:
        if cfg.anchor_fallback != "pooled_fqe" or not _is_anchor_support_error(exc):
            raise
        fallback_reason = str(exc)
        active_cfg = _pooled_fallback_config(cfg)
        active_gen_config = _to_genpqr_config(active_cfg)
        result = _fit_genpqr_with_policy(
            fitted_policy=fitted_policy,
            config=active_gen_config,
            batch=batch,
            normalization_policy=mu,
            gamma_value=gamma_value,
            anchor_function=anchor_function,
            env=fit_kwargs["env"],
            start_time=start_time,
        )
    if active_cfg.q_mode == "anchor_deeppqr" and _anchor_support_is_weak(result):
        if cfg.anchor_fallback == "error":
            raise GenPQRConfigurationError(
                "DeepGenPQR anchor support is weak; increase anchor coverage or set anchor_fallback='pooled_fqe'."
            )
        fallback_reason = "weak_anchor_support"
        active_cfg = _pooled_fallback_config(cfg)
        active_gen_config = _to_genpqr_config(active_cfg)
        result = _fit_genpqr_with_policy(
            fitted_policy=fitted_policy,
            config=active_gen_config,
            batch=batch,
            normalization_policy=mu,
            gamma_value=gamma_value,
            anchor_function=anchor_function,
            env=fit_kwargs["env"],
            start_time=start_time,
        )
    return result, active_cfg, active_gen_config, fallback_reason


def _reject_anchor_backend_in_pooled_mode(q_backend: str | Any) -> None:
    if isinstance(q_backend, str):
        key = q_backend.lower()
        if key in {"deeppqr_linear", "deeppqr_neural", "neural_deeppqr", "deep_pqr_neural"}:
            raise GenPQRConfigurationError(
                "DeepGenPQR pooled_fqe mode cannot use a DeepPQR anchor backend; set q_mode='anchor_deeppqr'."
            )
        return
    if isinstance(q_backend, (DeepPQRAnchorQEstimator, NeuralDeepPQRAnchorQEstimator)):
        raise GenPQRConfigurationError(
            "DeepGenPQR pooled_fqe mode cannot use a DeepPQR anchor backend; set q_mode='anchor_deeppqr'."
        )


def _is_anchor_support_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "anchor" in message or "deeppqr" in message or "deep pqr" in message


def _anchor_support_is_weak(result: GenPQRResult) -> bool:
    diagnostics = dict(getattr(result.q_function, "diagnostics", {}))
    return bool(diagnostics.get("weak_anchor_support", False))


def _normalization_from_config(
    *,
    explicit_policy: NormalizationPolicy | None,
    normalization_config: dict[str, Any],
    dataset: TransitionDataset | EpisodeDataset | None,
    action_space: ActionSpaceSpec | None,
) -> NormalizationPolicy | None:
    if explicit_policy is not None:
        return explicit_policy
    if not normalization_config:
        return None
    config = dict(normalization_config)
    kind = str(config.pop("kind", "uniform")).lower()
    space = action_space or getattr(dataset, "action_space", None)
    if kind in {"uniform", "anchor"}:
        if space is None:
            raise GenPQRConfigurationError("normalization_config requires action_space or a dataset with action_space.")
        if space.kind != "discrete":
            raise GenPQRConfigurationError(f"normalization_config kind='{kind}' requires discrete actions.")
        if kind == "uniform":
            return DiscreteNormalizationPolicy.uniform(int(space.n_actions))
        if "anchor_action" not in config:
            raise GenPQRConfigurationError("normalization_config kind='anchor' requires anchor_action.")
        return DiscreteNormalizationPolicy.anchor(int(space.n_actions), int(config["anchor_action"]))
    if kind == "continuous":
        sampler = config.get("sampler")
        if sampler is None:
            raise GenPQRConfigurationError("normalization_config kind='continuous' requires sampler.")
        action_dim = int(config.get("action_dim", getattr(space, "action_dim", 0) or 0))
        if action_dim <= 0:
            raise GenPQRConfigurationError("normalization_config kind='continuous' requires action_dim.")
        return ContinuousNormalizationPolicy(
            action_dim=action_dim,
            sampler=sampler,
            log_density=config.get("log_density"),
        )
    raise GenPQRConfigurationError(f"Unknown normalization_config kind '{kind}'.")


def _resolve_deep_policy_name(policy: str | Any) -> str | Any:
    if not isinstance(policy, str):
        return policy
    aliases = {
        "deep_airl": "imitation_airl",
        "deep_gail": "imitation_gail",
        "deep_bc": "imitation_bc",
    }
    return aliases.get(policy.lower(), policy)


def _resolve_anchor_backend_name(q_backend: str | Any) -> str | Any:
    if not isinstance(q_backend, str):
        return q_backend
    aliases = {
        "neural_deeppqr": "deeppqr_neural",
        "deep_pqr_neural": "deeppqr_neural",
    }
    return aliases.get(q_backend.lower(), q_backend)


def _build_deep_diagnostics(
    *,
    result: GenPQRResult,
    config: DeepGenPQRConfig,
    fit_time_seconds: float,
    requested_config: DeepGenPQRConfig | None = None,
    fallback_reason: str | None = None,
) -> dict[str, Any]:
    q_diag = dict(getattr(result.q_function, "diagnostics", {}))
    anchor_support = None
    if config.q_mode == "anchor_deeppqr":
        anchor_support = {
            "anchor_count": q_diag.get("anchor_count"),
            "weighted_anchor_count": q_diag.get("weighted_anchor_count"),
            "anchor_fraction": q_diag.get("anchor_fraction"),
            "weak_anchor_support": q_diag.get("weak_anchor_support"),
            "anchor_kind": q_diag.get("anchor_kind"),
            "anchor_tolerance": q_diag.get("anchor_tolerance"),
        }
    diagnostics = {
        "deepgenpqr_mode": config.q_mode,
        "deepgenpqr_requested_mode": (requested_config or config).q_mode,
        "deepgenpqr_policy_backend": _backend_name(_resolve_deep_policy_name(config.policy)),
        "deepgenpqr_q_backend": result.diagnostics.get("q_backend") or _backend_name(result.q_function),
        "policy_backend": _backend_name(_resolve_deep_policy_name(config.policy)),
        "q_backend": result.diagnostics.get("q_backend") or _backend_name(result.q_function),
        "pooled_actions": bool(config.q_mode == "pooled_fqe"),
        "anchor_enabled": bool(config.q_mode == "anchor_deeppqr"),
        "anchor_support": anchor_support,
        "normalization_mc_se": result.diagnostics.get("continuous_mc_standard_error_mean"),
        "deepgenpqr_fit_time_seconds": float(fit_time_seconds),
        "anchor_fallback": (requested_config or config).anchor_fallback,
        "anchor_fallback_used": fallback_reason is not None,
        "anchor_fallback_reason": fallback_reason,
    }
    if fallback_reason is not None:
        warnings = list(result.diagnostics.get("warnings", []))
        warning_codes = list(result.diagnostics.get("warning_codes", []))
        warnings.append("DeepGenPQR anchor mode fell back to pooled FQE.")
        warning_codes.append("anchor_fallback_to_pooled_fqe")
        diagnostics["warnings"] = warnings
        diagnostics["warning_codes"] = warning_codes
    return diagnostics


def _backend_name(obj: Any) -> str:
    if isinstance(obj, str):
        return obj
    return type(obj).__name__


__all__ = [
    "DeepGenPQRAnchorFallback",
    "DeepGenPQRConfig",
    "DeepGenPQRQMode",
    "DeepGenPQRResult",
    "fit_deep_genpqr",
    "list_deepgenpqr_presets",
]
