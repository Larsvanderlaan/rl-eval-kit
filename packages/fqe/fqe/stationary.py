from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Sequence

import numpy as np

from fqe.fit_fqe import (
    Array,
    BoostedFQEConfig,
    FQEModel,
    _as_1d_float,
    _as_2d_float,
    _as_next_actions,
    _optional_terminals,
    _optional_weights,
    fit_fqe_lgbm,
)
from fqe.fit_neural_fqe import NeuralFQEConfig, NeuralFQEModel, fit_fqe_neural

if TYPE_CHECKING:
    from occupancy_ratio.google_dualdice import GoogleDualDICEOccupancyRatioModel


__all__ = [
    "StationaryWeightedFQEConfig",
    "StationaryWeightedFQEResult",
    "GoogleDualDICEConfig",
    "fit_stationary_weighted_fqe",
    "preflight_google_dualdice",
    "preflight_minimax_weight",
]


@dataclass(frozen=True)
class GoogleDualDICEConfig:
    """Configuration for optional Google Research DualDICE ratio weights."""

    google_research_path: str | Path = Path("/tmp/google-research")
    num_updates: int = 1000
    batch_size: int = 128
    weight_decay: float = 1e-5
    seed: int = 123
    limit_tf_threads: bool = True


@dataclass(frozen=True)
class StationaryWeightedFQEConfig:
    """Controls for occupancy-ratio weighted FQE."""

    family: Literal["boosted", "neural"] = "boosted"
    ratio_backend: Literal["occupancy_ratio", "google_dualdice", "minimax_weight"] = "google_dualdice"
    ratio_family: Literal["auto", "boosted", "neural"] = "auto"
    minimax_weight_method: str = "auto"
    normalize_weights: bool = True
    ratio_clip: bool = True
    fallback_uniform_ratio_to_action_ratio: bool = True
    fallback_source_ratio_failure: bool = True
    uniform_ratio_std_tol: float = 1e-8
    initial_ratio_mode: str = "factored"
    one_step_ratio_mode: str = "auto"
    fqe_config: BoostedFQEConfig | None = None
    neural_config: NeuralFQEConfig | None = None
    occupancy_config: Any | None = None
    action_ratio_config: Any | None = None
    source_state_ratio_config: Any | None = None
    transition_ratio_config: Any | None = None
    google_dualdice_config: GoogleDualDICEConfig | None = None
    minimax_weight_config: Any | None = None


@dataclass
class StationaryWeightedFQEResult:
    """Result bundle for stationary occupancy-weighted FQE."""

    fqe_model: FQEModel | NeuralFQEModel
    occupancy_model: Any
    ratio_weights: Array
    sample_weight: Array
    diagnostics: dict[str, Any]

    def predict(self, states: Array, actions: Array | None = None) -> Array:
        return self.fqe_model.predict(states, actions)

    def predict_q(self, states: Array, actions: Array) -> Array:
        return self.fqe_model.predict_q(states, actions)

    def estimate_policy_value(
        self,
        initial_states: Array,
        initial_actions: Array | None = None,
        initial_weights: Array | None = None,
    ) -> float:
        return self.fqe_model.estimate_policy_value(initial_states, initial_actions, initial_weights)

def fit_stationary_weighted_fqe(
    *,
    states: Array,
    actions: Array,
    next_states: Array,
    target_actions: Array,
    next_actions: Array,
    rewards: Array,
    gamma: float,
    gamma_ratio: float = 0.99,
    terminals: Array | None = None,
    sample_weight: Array | None = None,
    initial_states: Array | None = None,
    initial_actions: Array | None = None,
    initial_weights: Array | None = None,
    target_next_actions: Array | None = None,
    config: StationaryWeightedFQEConfig | None = None,
    fqe_config: BoostedFQEConfig | None = None,
    neural_config: NeuralFQEConfig | None = None,
    family: Literal["boosted", "neural"] | None = None,
    ratio_family: Literal["auto", "boosted", "neural"] | None = None,
    ratio_backend: Literal["occupancy_ratio", "google_dualdice", "minimax_weight"] | None = None,
    minimax_weight_method: str | None = None,
    occupancy_config: Any | None = None,
    action_ratio_config: Any | None = None,
    source_state_ratio_config: Any | None = None,
    transition_ratio_config: Any | None = None,
    google_dualdice_config: GoogleDualDICEConfig | None = None,
    minimax_weight_config: Any | None = None,
    episode_ids: Array | None = None,
    timesteps: Array | None = None,
    step_per_trajectory: int | None = None,
    behavior_action_pscore: Array | None = None,
    normalize_weights: bool | None = None,
    categorical_feature: Sequence[int | str] | None = None,
) -> StationaryWeightedFQEResult:
    """Fit FQE after reweighting rows by near-stationary occupancy ratios."""

    gamma_ratio = _validate_gamma_ratio(gamma_ratio)
    cfg = StationaryWeightedFQEConfig() if config is None else config
    fqe_family = str(cfg.family if family is None else family)
    if fqe_family not in {"boosted", "neural"}:
        raise ValueError("family must be 'boosted' or 'neural'.")
    resolved_ratio_backend = str(cfg.ratio_backend if ratio_backend is None else ratio_backend)
    if resolved_ratio_backend not in {"occupancy_ratio", "google_dualdice", "minimax_weight"}:
        raise ValueError("ratio_backend must be 'occupancy_ratio', 'google_dualdice', or 'minimax_weight'.")
    resolved_minimax_method = str(cfg.minimax_weight_method if minimax_weight_method is None else minimax_weight_method)
    resolved_ratio_family = _resolve_ratio_family(
        fqe_family=fqe_family,
        ratio_family=cfg.ratio_family if ratio_family is None else ratio_family,
    )
    rewards_1d = _as_1d_float(rewards, "rewards")
    states_2d = _as_2d_float(states, "states", n_rows=rewards_1d.shape[0])
    actions_2d = _as_2d_float(actions, "actions", n_rows=rewards_1d.shape[0])
    next_states_2d = _as_2d_float(next_states, "next_states", n_rows=rewards_1d.shape[0])
    target_actions_2d = _as_2d_float(target_actions, "target_actions", n_rows=rewards_1d.shape[0], expected_cols=actions_2d.shape[1])
    next_actions_3d = _as_next_actions(next_actions, n_rows=rewards_1d.shape[0], action_dim=actions_2d.shape[1])
    terminals_1d = _optional_terminals(terminals, rewards_1d.shape[0])
    user_weight = _optional_weights(sample_weight, rewards_1d.shape[0], "sample_weight")
    initial_states_2d = None if initial_states is None else _as_2d_float(initial_states, "initial_states")
    initial_actions_2d = None
    if initial_actions is not None:
        initial_actions_2d = _as_2d_float(
            initial_actions,
            "initial_actions",
            n_rows=None if initial_states_2d is None else initial_states_2d.shape[0],
            expected_cols=actions_2d.shape[1],
        )
    if resolved_ratio_backend == "google_dualdice" and (initial_states_2d is None or initial_actions_2d is None):
        raise ValueError(
            "Google DualDICE is the default stationary ratio backend and requires "
            "initial_states and initial_actions. Pass ratio_backend='occupancy_ratio' "
            "to use the FORI backend without joint initial state-action rows."
        )
    ratio_target_next_actions = _resolve_target_next_actions(
        target_next_actions=target_next_actions,
        next_actions_3d=next_actions_3d,
        n_rows=rewards_1d.shape[0],
        action_dim=actions_2d.shape[1],
    )
    if resolved_ratio_backend == "google_dualdice":
        occupancy_model, occupancy_fit_diag = _fit_google_dualdice_ratio_model(
            states=states_2d,
            actions=actions_2d,
            next_states=next_states_2d,
            target_actions=target_actions_2d,
            target_next_actions=ratio_target_next_actions,
            terminals=terminals_1d,
            sample_weight=user_weight,
            initial_states=initial_states_2d,
            initial_actions=initial_actions_2d,
            initial_weights=initial_weights,
            gamma_ratio=gamma_ratio,
            config=google_dualdice_config if google_dualdice_config is not None else cfg.google_dualdice_config,
        )
    elif resolved_ratio_backend == "minimax_weight":
        occupancy_model, occupancy_fit_diag = _fit_minimax_weight_ratio_model(
            states=states_2d,
            actions=actions_2d,
            next_states=next_states_2d,
            target_actions=target_actions_2d,
            target_next_actions=ratio_target_next_actions,
            terminals=terminals_1d,
            rewards=rewards_1d,
            sample_weight=user_weight,
            initial_states=initial_states_2d,
            initial_actions=initial_actions_2d,
            initial_weights=initial_weights,
            gamma_ratio=gamma_ratio,
            method=resolved_minimax_method,
            config=minimax_weight_config if minimax_weight_config is not None else cfg.minimax_weight_config,
            episode_ids=episode_ids,
            timesteps=timesteps,
            step_per_trajectory=step_per_trajectory,
            behavior_action_pscore=behavior_action_pscore,
            google_dualdice_config=google_dualdice_config if google_dualdice_config is not None else cfg.google_dualdice_config,
        )
    else:
        fit_discounted_occupancy_ratio = _load_occupancy_api(resolved_ratio_family)
        occupancy_kwargs = dict(
            states=states_2d,
            actions=actions_2d,
            next_states=next_states_2d,
            target_actions=target_actions_2d,
            gamma=gamma_ratio,
            initial_states=initial_states_2d,
            initial_actions=initial_actions_2d,
            initial_weights=initial_weights,
            target_next_actions=ratio_target_next_actions,
            initial_ratio_mode=str(cfg.initial_ratio_mode),
            one_step_ratio_mode=str(cfg.one_step_ratio_mode),
            occupancy=occupancy_config if occupancy_config is not None else cfg.occupancy_config,
            action_ratio=action_ratio_config if action_ratio_config is not None else cfg.action_ratio_config,
            source_state_ratio=source_state_ratio_config if source_state_ratio_config is not None else cfg.source_state_ratio_config,
            transition_ratio=transition_ratio_config if transition_ratio_config is not None else cfg.transition_ratio_config,
        )
        occupancy_model, occupancy_fit_diag = _fit_occupancy_with_source_fallback(
            fit_discounted_occupancy_ratio,
            occupancy_kwargs=occupancy_kwargs,
            allow_fallback=bool(cfg.fallback_source_ratio_failure),
        )
    ratio_weights, ratio_diag = _select_ratio_weights(
        occupancy_model,
        states_2d,
        actions_2d,
        clip=bool(cfg.ratio_clip),
        fallback_to_action_ratio=bool(cfg.fallback_uniform_ratio_to_action_ratio),
        uniform_std_tol=float(cfg.uniform_ratio_std_tol),
    )
    final_weight, weight_diag = _combine_weights(
        sample_weight=user_weight,
        ratio_weights=ratio_weights,
        normalize=bool(cfg.normalize_weights if normalize_weights is None else normalize_weights),
    )
    if fqe_family == "boosted":
        model = fit_fqe_lgbm(
            states=states_2d,
            actions=actions_2d,
            next_states=next_states_2d,
            next_actions=next_actions_3d,
            rewards=rewards_1d,
            gamma=gamma,
            terminals=terminals_1d,
            sample_weight=final_weight,
            config=fqe_config if fqe_config is not None else cfg.fqe_config,
            categorical_feature=categorical_feature,
        )
    else:
        model = fit_fqe_neural(
            states=states_2d,
            actions=actions_2d,
            next_states=next_states_2d,
            next_actions=next_actions_3d,
            rewards=rewards_1d,
            gamma=gamma,
            terminals=terminals_1d,
            sample_weight=final_weight,
            config=neural_config if neural_config is not None else cfg.neural_config,
        )
    diagnostics: dict[str, Any] = {
        "family": fqe_family,
        "ratio_backend": resolved_ratio_backend,
        "ratio_family": resolved_ratio_family,
        "minimax_weight_method": resolved_minimax_method if resolved_ratio_backend == "minimax_weight" else None,
        "gamma": float(gamma),
        "gamma_ratio": float(gamma_ratio),
        "normalize_weights": bool(cfg.normalize_weights if normalize_weights is None else normalize_weights),
        "ratio_clip": bool(cfg.ratio_clip),
        "target_next_actions_used": bool(ratio_target_next_actions is not None),
    }
    diagnostics.update(occupancy_fit_diag)
    diagnostics.update(ratio_diag)
    diagnostics.update(weight_diag)
    diagnostics.update({f"fqe_{key}": value for key, value in dict(model.diagnostics).items()})
    diagnostics.update({f"occupancy_{key}": value for key, value in _model_diagnostics(occupancy_model).items()})
    return StationaryWeightedFQEResult(
        fqe_model=model,
        occupancy_model=occupancy_model,
        ratio_weights=ratio_weights,
        sample_weight=final_weight,
        diagnostics=diagnostics,
    )


def _load_occupancy_api(family: Literal["boosted", "neural"] = "boosted"):
    try:
        if family == "neural":
            from occupancy_ratio import fit_discounted_occupancy_ratio_neural

            return fit_discounted_occupancy_ratio_neural
        from occupancy_ratio import fit_discounted_occupancy_ratio
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Stationary weighted FQE requires the optional occupancy-ratio package. "
            "Install the stationary extra with `pip install fqe[stationary]` or install "
            "`occupancy-ratio` alongside fqe."
        ) from exc
    return fit_discounted_occupancy_ratio


def preflight_google_dualdice(google_research_path: str | Path = Path("/tmp/google-research")) -> tuple[bool, str]:
    try:
        from occupancy_ratio import preflight_google_dualdice as occupancy_preflight
    except ModuleNotFoundError as exc:
        return False, f"occupancy-ratio is required for Google DualDICE weights: {exc}"
    preflight = occupancy_preflight(google_research_path)
    return bool(preflight.available), str(preflight.reason)


def preflight_minimax_weight(method: str = "auto", config: Any | None = None) -> tuple[bool, str]:
    try:
        from occupancy_ratio import preflight_minimax_weight as occupancy_preflight
    except ModuleNotFoundError as exc:
        return False, f"occupancy-ratio is required for minimax weights: {exc}"
    preflight = occupancy_preflight(method=method, config=config)
    return bool(preflight.available), str(preflight.reason)


def _fit_google_dualdice_ratio_model(
    *,
    states: Array,
    actions: Array,
    next_states: Array,
    target_actions: Array,
    target_next_actions: Array | None,
    terminals: Array,
    sample_weight: Array,
    initial_states: Array | None,
    initial_actions: Array | None,
    initial_weights: Array | None,
    gamma_ratio: float,
    config: GoogleDualDICEConfig | None,
) -> tuple[GoogleDualDICEOccupancyRatioModel, dict[str, Any]]:
    cfg = GoogleDualDICEConfig() if config is None else config
    try:
        from occupancy_ratio import GoogleDualDICEConfig as OccupancyGoogleDualDICEConfig
        from occupancy_ratio import fit_google_dualdice_occupancy_ratio
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Google DualDICE weights require the optional occupancy-ratio package. "
            "Install fqe[stationary] and occupancy-ratio[google-dualdice]."
        ) from exc

    occupancy_config = OccupancyGoogleDualDICEConfig(
        google_research_path=cfg.google_research_path,
        num_updates=int(cfg.num_updates),
        batch_size=int(cfg.batch_size),
        weight_decay=float(cfg.weight_decay),
        seed=int(cfg.seed),
        limit_tf_threads=bool(cfg.limit_tf_threads),
    )
    model = fit_google_dualdice_occupancy_ratio(
        states=states,
        actions=actions,
        next_states=next_states,
        target_actions=target_actions,
        gamma=float(gamma_ratio),
        initial_states=initial_states,
        initial_actions=initial_actions,
        initial_weights=initial_weights,
        target_next_actions=target_next_actions,
        terminals=terminals,
        sample_weight=sample_weight,
        config=occupancy_config,
    )
    return model, {
        "occupancy_fit_fallback_used": False,
        "occupancy_fit_fallback_reason": None,
        "google_dualdice_num_updates": float(cfg.num_updates),
        "google_dualdice_batch_size": float(min(int(cfg.batch_size), states.shape[0])),
    }


def _fit_minimax_weight_ratio_model(
    *,
    states: Array,
    actions: Array,
    next_states: Array,
    target_actions: Array,
    target_next_actions: Array | None,
    terminals: Array,
    rewards: Array,
    sample_weight: Array,
    initial_states: Array | None,
    initial_actions: Array | None,
    initial_weights: Array | None,
    gamma_ratio: float,
    method: str,
    config: Any | None,
    episode_ids: Array | None,
    timesteps: Array | None,
    step_per_trajectory: int | None,
    behavior_action_pscore: Array | None,
    google_dualdice_config: GoogleDualDICEConfig | None,
) -> tuple[Any, dict[str, Any]]:
    try:
        from occupancy_ratio import GoogleDualDICEConfig as OccupancyGoogleDualDICEConfig
        from occupancy_ratio import MinimaxWeightConfig, fit_minimax_weight
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Minimax stationary weights require the optional occupancy-ratio package. "
            "Install fqe[stationary] or occupancy-ratio with the requested minimax backend extras."
        ) from exc

    resolved_config = config
    if resolved_config is None and google_dualdice_config is not None:
        resolved_config = MinimaxWeightConfig(
            method=method,
            google_policy_eval=OccupancyGoogleDualDICEConfig(
                google_research_path=google_dualdice_config.google_research_path,
                num_updates=int(google_dualdice_config.num_updates),
                batch_size=int(google_dualdice_config.batch_size),
                weight_decay=float(google_dualdice_config.weight_decay),
                seed=int(google_dualdice_config.seed),
                limit_tf_threads=bool(google_dualdice_config.limit_tf_threads),
            ),
        )
    elif resolved_config is None:
        resolved_config = MinimaxWeightConfig(method=method)

    model = fit_minimax_weight(
        states=states,
        actions=actions,
        next_states=next_states,
        target_actions=target_actions,
        target_next_actions=target_next_actions,
        gamma=float(gamma_ratio),
        rewards=rewards,
        initial_states=initial_states,
        initial_actions=initial_actions,
        initial_weights=initial_weights,
        terminals=terminals,
        sample_weight=sample_weight,
        method=method,
        config=resolved_config,
        episode_ids=episode_ids,
        timesteps=timesteps,
        step_per_trajectory=step_per_trajectory,
        behavior_action_pscore=behavior_action_pscore,
    )
    return model, {
        "occupancy_fit_fallback_used": False,
        "occupancy_fit_fallback_reason": None,
        "minimax_weight_method_resolved": str(getattr(model, "method", method)),
    }


def _resolve_ratio_family(
    *,
    fqe_family: str,
    ratio_family: Literal["auto", "boosted", "neural"],
) -> Literal["boosted", "neural"]:
    if ratio_family == "auto":
        return "neural" if fqe_family == "neural" else "boosted"
    if ratio_family not in {"boosted", "neural"}:
        raise ValueError("ratio_family must be 'auto', 'boosted', or 'neural'.")
    return ratio_family


def _validate_gamma_ratio(gamma_ratio: float) -> float:
    value = float(gamma_ratio)
    if not (0.0 <= value < 1.0):
        raise ValueError("gamma_ratio must be in [0, 1); stationary approximation must use a discount below one.")
    return value


def _resolve_target_next_actions(
    *,
    target_next_actions: Array | None,
    next_actions_3d: Array,
    n_rows: int,
    action_dim: int,
) -> Array | None:
    if target_next_actions is not None:
        return _as_2d_float(target_next_actions, "target_next_actions", n_rows=n_rows, expected_cols=action_dim)
    if next_actions_3d.shape[1] == 1:
        return np.ascontiguousarray(next_actions_3d[:, 0, :])
    return None


def _fit_occupancy_with_source_fallback(
    fit_discounted_occupancy_ratio,
    *,
    occupancy_kwargs: dict[str, Any],
    allow_fallback: bool,
):
    try:
        return fit_discounted_occupancy_ratio(**occupancy_kwargs), {
            "occupancy_fit_fallback_used": False,
            "occupancy_fit_fallback_reason": None,
        }
    except Exception as exc:
        if not allow_fallback or occupancy_kwargs.get("initial_states") is None:
            raise
        retry_kwargs = dict(occupancy_kwargs)
        retry_kwargs.update(initial_states=None, initial_actions=None, initial_weights=None, initial_ratio_mode="factored")
        try:
            model = fit_discounted_occupancy_ratio(**retry_kwargs)
        except Exception as retry_exc:
            raise retry_exc from exc
        return model, {
            "occupancy_fit_fallback_used": True,
            "occupancy_fit_fallback_reason": f"{type(exc).__name__}: {exc}",
        }


def _select_ratio_weights(
    model: Any,
    states: Array,
    actions: Array,
    *,
    clip: bool,
    fallback_to_action_ratio: bool,
    uniform_std_tol: float,
) -> tuple[Array, dict[str, Any]]:
    primary = np.asarray(model.predict_state_action_ratio(states, actions, clip=clip), dtype=np.float64).reshape(-1)
    issue = _ratio_weight_issue(primary, uniform_std_tol=uniform_std_tol)
    diagnostics: dict[str, Any] = {
        "ratio_weight_source": "occupancy",
        "ratio_weight_fallback_used": False,
        "ratio_weight_fallback_reason": None,
        "ratio_weight_invalid_reason": issue,
        "primary_ratio_weight_mean": float(np.mean(primary)) if primary.size else float("nan"),
        "primary_ratio_weight_std": float(np.std(primary)) if primary.size else float("nan"),
        "primary_ratio_weight_min": _quantile(primary, 0.0),
        "primary_ratio_weight_max": _quantile(primary, 1.0),
        "primary_ratio_weight_p99": _quantile(primary, 0.99),
        "primary_ratio_weight_ess_fraction": _ess_fraction(primary),
    }
    if issue is None:
        return np.ascontiguousarray(primary, dtype=np.float64), diagnostics
    if not fallback_to_action_ratio:
        if issue is not None:
            diagnostics["ratio_weight_fallback_reason"] = issue
        return np.ascontiguousarray(primary, dtype=np.float64), diagnostics

    predict_action_ratio = getattr(model, "predict_action_ratio", None)
    if not callable(predict_action_ratio):
        diagnostics.update(
            {
                "ratio_weight_source": "uniform_invalid_ratio_fallback",
                "ratio_weight_fallback_used": True,
                "ratio_weight_fallback_reason": f"{issue}; action-ratio predictor unavailable",
                "ratio_weight_degraded": True,
            }
        )
        return np.ones_like(primary, dtype=np.float64), diagnostics

    fallback = np.asarray(predict_action_ratio(states, actions, clip=clip), dtype=np.float64).reshape(-1)
    fallback_issue = _ratio_weight_issue(fallback, uniform_std_tol=uniform_std_tol)
    diagnostics.update(
        {
            "action_ratio_fallback_mean": float(np.mean(fallback)) if fallback.size else float("nan"),
            "action_ratio_fallback_std": float(np.std(fallback)) if fallback.size else float("nan"),
            "action_ratio_fallback_ess_fraction": _ess_fraction(fallback),
        }
    )
    if fallback_issue is not None:
        diagnostics.update(
            {
                "ratio_weight_source": "uniform_invalid_ratio_fallback",
                "ratio_weight_fallback_used": True,
                "ratio_weight_fallback_reason": f"{issue}; action-ratio fallback {fallback_issue}",
                "ratio_weight_degraded": True,
            }
        )
        return np.ones_like(primary, dtype=np.float64), diagnostics

    diagnostics.update(
        {
            "ratio_weight_source": "action_ratio_fallback",
            "ratio_weight_fallback_used": True,
            "ratio_weight_fallback_reason": issue,
            "ratio_weight_degraded": True,
        }
    )
    return np.ascontiguousarray(fallback, dtype=np.float64), diagnostics


def _ratio_weight_issue(weights: Array, *, uniform_std_tol: float) -> str | None:
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if w.size == 0:
        return "empty"
    if not np.all(np.isfinite(w)):
        return "nonfinite"
    if np.any(w < 0.0):
        return "negative"
    total = float(np.sum(w))
    if total <= 0.0 or not np.isfinite(total):
        return "nonpositive_total"
    scale = max(1.0, abs(float(np.mean(w))))
    if float(np.std(w)) <= max(0.0, float(uniform_std_tol)) * scale:
        return "near_uniform"
    return None


def _combine_weights(*, sample_weight: Array, ratio_weights: Array, normalize: bool) -> tuple[Array, dict[str, float]]:
    ratio = np.asarray(ratio_weights, dtype=np.float64).reshape(-1)
    base = np.asarray(sample_weight, dtype=np.float64).reshape(-1)
    if ratio.shape != base.shape:
        raise ValueError("ratio_weights and sample_weight must have the same shape.")
    if not np.all(np.isfinite(ratio)):
        raise ValueError("ratio_weights must contain only finite values.")
    if np.any(ratio < 0.0):
        raise ValueError("ratio_weights must be nonnegative.")
    combined = base * ratio
    total = float(np.sum(combined))
    if total <= 0.0 or not np.isfinite(total):
        raise ValueError("combined stationary FQE weights must have positive finite total weight.")
    unnormalized_mean = float(np.mean(combined))
    if normalize:
        combined = combined / max(unnormalized_mean, 1e-12)
    diag = {
        "ratio_weight_mean": float(np.mean(ratio)),
        "ratio_weight_std": float(np.std(ratio)),
        "ratio_weight_min": float(np.min(ratio)),
        "ratio_weight_max": float(np.max(ratio)),
        "ratio_weight_p99": _quantile(ratio, 0.99),
        "ratio_weight_ess_fraction": _ess_fraction(ratio),
        "fqe_weight_mean": float(np.mean(combined)),
        "fqe_weight_std": float(np.std(combined)),
        "fqe_weight_min": float(np.min(combined)),
        "fqe_weight_max": float(np.max(combined)),
        "fqe_weight_p99": _quantile(combined, 0.99),
        "fqe_weight_ess_fraction": _ess_fraction(combined),
        "fqe_weight_unnormalized_mean": unnormalized_mean,
    }
    return np.ascontiguousarray(combined, dtype=np.float64), diag


def _ess_fraction(weights: Array) -> float:
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    denom = float(np.sum(w * w))
    if w.size == 0 or denom <= 0.0 or not np.isfinite(denom):
        return 0.0
    return float((np.sum(w) ** 2 / denom) / w.size)


def _quantile(values: Array, q: float) -> float:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    return float(np.quantile(x, q)) if x.size else float("nan")


def _model_diagnostics(model: Any) -> dict[str, Any]:
    diagnostics = getattr(model, "diagnostics", None)
    if isinstance(diagnostics, dict):
        return dict(diagnostics)
    legacy = getattr(model, "legacy_result", None)
    if isinstance(legacy, dict):
        raw_diag = legacy.get("diagnostics")
        if isinstance(raw_diag, dict):
            return dict(raw_diag)
    return {}
