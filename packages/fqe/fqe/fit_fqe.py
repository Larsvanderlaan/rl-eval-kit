from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Sequence

import numpy as np

from fqe.bellman import (
    bellman_risk,
    build_bellman_target,
    package_versions,
    serializable_config,
    validate_action_weights,
    validate_bootstrap_inputs,
    weighted_action_expectation,
)

try:  # pragma: no cover - exercised when LightGBM is installed.
    import lightgbm as lgb
except ModuleNotFoundError:  # pragma: no cover - environment dependent.
    lgb = None


Array = np.ndarray
LossName = Literal["squared", "huber"]
ModeName = Literal["q", "value"]
NextActionSampler = Callable[[Array, np.random.Generator, int], Array]


__all__ = [
    "BoostedFQEConfig",
    "FQEModel",
    "fit_fqe_lgbm",
    "fit_value_lgbm",
    "fit_fqe_from_policy",
    "load_fqe_lgbm",
    "tune_fqe_cv",
]


@dataclass(frozen=True)
class BoostedFQEConfig:
    """Configuration for LightGBM fitted Q evaluation / value iteration."""

    lgb_params: dict[str, Any] = field(default_factory=dict)
    num_iterations: int = 200
    trees_per_iteration: int = 1
    validation_fraction: float = 0.2
    early_stopping: bool = True
    patience: int = 15
    min_improvement: float = 1e-6
    refit_on_all_data: bool = True
    loss: LossName = "huber"
    huber_delta: float | None = None
    huber_delta_scale: float = 1.345
    huber_hessian_floor: float = 1e-3
    target_min: float | None = None
    target_max: float | None = None
    infer_value_bounds: bool = True
    update_target_frequency: int = 1
    learning_rate_backoff: float = 0.5
    min_learning_rate: float | None = None
    seed: int = 123
    num_threads: int = 1
    show_progress: bool = False

    def __post_init__(self) -> None:
        if self.num_iterations <= 0:
            raise ValueError("num_iterations must be positive.")
        if self.trees_per_iteration <= 0:
            raise ValueError("trees_per_iteration must be positive.")
        if not (0.0 < self.validation_fraction < 1.0):
            raise ValueError("validation_fraction must be in (0, 1).")
        if self.patience < 0:
            raise ValueError("patience must be nonnegative.")
        if self.min_improvement < 0.0:
            raise ValueError("min_improvement must be nonnegative.")
        if self.loss not in {"squared", "huber"}:
            raise ValueError("loss must be 'squared' or 'huber'.")
        if self.huber_delta is not None and self.huber_delta <= 0.0:
            raise ValueError("huber_delta must be positive when supplied.")
        if self.huber_delta_scale <= 0.0:
            raise ValueError("huber_delta_scale must be positive.")
        if self.huber_hessian_floor < 0.0:
            raise ValueError("huber_hessian_floor must be nonnegative.")
        if self.target_min is not None and self.target_max is not None and self.target_min > self.target_max:
            raise ValueError("target_min must be <= target_max.")
        if self.update_target_frequency <= 0:
            raise ValueError("update_target_frequency must be positive.")
        if not (0.0 < self.learning_rate_backoff <= 1.0):
            raise ValueError("learning_rate_backoff must be in (0, 1].")
        if self.min_learning_rate is not None and self.min_learning_rate <= 0.0:
            raise ValueError("min_learning_rate must be positive when supplied.")
        if self.num_threads == 0:
            raise ValueError("num_threads must be nonzero.")

    @classmethod
    def stable_defaults(cls, **overrides: Any) -> "BoostedFQEConfig":
        """Construct conservative defaults for noisy offline RL data."""
        params: dict[str, Any] = {
            "loss": "huber",
            "num_iterations": 200,
            "trees_per_iteration": 1,
            "validation_fraction": 0.2,
            "early_stopping": True,
            "patience": 15,
            "min_improvement": 1e-6,
            "refit_on_all_data": True,
            "infer_value_bounds": True,
            "learning_rate_backoff": 0.5,
            "num_threads": 1,
            "lgb_params": {
                "learning_rate": 0.05,
                "num_leaves": 31,
                "min_data_in_leaf": 20,
                "lambda_l2": 1.0,
                "feature_fraction": 1.0,
                "bagging_fraction": 1.0,
                "bagging_freq": 0,
                "verbosity": -1,
            },
        }
        params.update(overrides)
        return cls(**params)


@dataclass
class FQEModel:
    """Fitted LightGBM FQE/FVI model with prediction helpers."""

    booster: Any
    mode: ModeName
    gamma: float
    state_dim: int
    action_dim: int | None
    config: BoostedFQEConfig
    history: list[dict[str, Any]]
    diagnostics: dict[str, Any]
    train_indices: Array
    validation_indices: Array
    categorical_feature: Sequence[int | str] | None = None
    action_spec: dict[str, Any] | None = None

    def predict(self, states: Array, actions: Array | None = None) -> Array:
        """Predict Q(s, a) in Q mode or V(s) in value mode."""
        if self.mode == "q":
            if actions is None:
                raise ValueError("actions are required when predicting with a Q-mode FQEModel.")
            return self.predict_q(states, actions)
        if actions is not None:
            raise ValueError("actions must not be supplied for a value-mode FQEModel.")
        return self.predict_value(states)

    def predict_q(self, states: Array, actions: Array) -> Array:
        """Predict fitted Q-values."""
        if self.mode != "q":
            raise ValueError("predict_q is only available for Q-mode FQEModel objects.")
        return _predict_booster(
            self.booster,
            _state_action_features(states, actions, self.state_dim, self.action_dim, self.action_spec),
        )

    def predict_value(self, states: Array) -> Array:
        """Predict fitted state values."""
        if self.mode != "value":
            raise ValueError("predict_value is only available for value-mode FQEModel objects.")
        return _predict_booster(self.booster, _as_2d_float(states, "states", expected_cols=self.state_dim))

    def estimate_policy_value(
        self,
        initial_states: Array,
        initial_actions: Array | None = None,
        initial_weights: Array | None = None,
        initial_action_weights: Array | None = None,
    ) -> float:
        """Estimate the policy value by averaging initial-state predictions."""
        if self.mode == "q":
            if initial_actions is None:
                raise ValueError("initial_actions are required for Q-mode policy value estimation.")
            states_2d = _as_2d_float(initial_states, "initial_states", expected_cols=self.state_dim)
            actions_3d = _as_initial_actions(
                initial_actions,
                n_rows=states_2d.shape[0],
                action_dim=self.action_dim,
                action_spec=self.action_spec,
            )
            action_weights = validate_action_weights(
                initial_action_weights,
                n_rows=states_2d.shape[0],
                n_actions=actions_3d.shape[1],
                name="initial_action_weights",
            )
            values_by_action = [
                self.predict_q(states_2d, actions_3d[:, action_idx, :])
                for action_idx in range(actions_3d.shape[1])
            ]
            values = weighted_action_expectation(np.stack(values_by_action, axis=1), action_weights)
        else:
            if initial_actions is not None:
                raise ValueError("initial_actions must not be supplied for value-mode policy value estimation.")
            values = self.predict_value(initial_states)
        weights = _optional_weights(initial_weights, values.shape[0], "initial_weights")
        return float(np.average(values, weights=weights))

    def to_legacy_dict(self) -> dict[str, Any]:
        """Return a dictionary payload compatible with older script-style code."""
        return {
            "model": self.booster,
            "mode": self.mode,
            "gamma": float(self.gamma),
            "state_dim": int(self.state_dim),
            "action_dim": None if self.action_dim is None else int(self.action_dim),
            "history": list(self.history),
            "diagnostics": dict(self.diagnostics),
            "train_indices": np.asarray(self.train_indices, dtype=np.int64),
            "validation_indices": np.asarray(self.validation_indices, dtype=np.int64),
            "action_spec": None if self.action_spec is None else dict(self.action_spec),
        }

    def save(self, path: str | Path) -> None:
        """Save a versioned boosted FQE model artifact."""
        _require_lightgbm()
        path_obj = Path(path)
        payload = {
            "schema_version": 1,
            "model_family": "boosted_lgbm",
            "mode": self.mode,
            "gamma": float(self.gamma),
            "state_dim": int(self.state_dim),
            "action_dim": None if self.action_dim is None else int(self.action_dim),
            "config": serializable_config(self.config),
            "history": list(self.history),
            "diagnostics": dict(self.diagnostics),
            "train_indices": np.asarray(self.train_indices, dtype=np.int64),
            "validation_indices": np.asarray(self.validation_indices, dtype=np.int64),
            "categorical_feature": None if self.categorical_feature is None else list(self.categorical_feature),
            "action_spec": None if self.action_spec is None else dict(self.action_spec),
            "versions": package_versions(("numpy", "lightgbm", "fqe")),
            "constant_value": None,
            "booster_model_string": None,
        }
        if isinstance(self.booster, _ConstantBooster):
            payload["constant_value"] = float(self.booster.value)
        else:
            payload["booster_model_string"] = self.booster.model_to_string()
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        with path_obj.open("wb") as handle:
            np.savez_compressed(handle, payload=np.asarray([payload], dtype=object))

    @classmethod
    def load(cls, path: str | Path) -> "FQEModel":
        """Load a boosted FQE artifact saved by :meth:`save`."""
        _require_lightgbm()
        with np.load(Path(path), allow_pickle=True) as data:
            payload = data["payload"].item()
        if int(payload.get("schema_version", -1)) != 1 or payload.get("model_family") != "boosted_lgbm":
            raise ValueError("Unsupported boosted FQE artifact.")
        config = BoostedFQEConfig(**dict(payload["config"]))
        constant_value = payload.get("constant_value")
        booster = _ConstantBooster(float(constant_value)) if constant_value is not None else lgb.Booster(
            model_str=str(payload["booster_model_string"])
        )
        return cls(
            booster=booster,
            mode=payload["mode"],
            gamma=float(payload["gamma"]),
            state_dim=int(payload["state_dim"]),
            action_dim=None if payload["action_dim"] is None else int(payload["action_dim"]),
            config=config,
            history=list(payload["history"]),
            diagnostics=dict(payload["diagnostics"]),
            train_indices=np.asarray(payload["train_indices"], dtype=np.int64),
            validation_indices=np.asarray(payload["validation_indices"], dtype=np.int64),
            categorical_feature=payload.get("categorical_feature"),
            action_spec=payload.get("action_spec"),
        )


def load_fqe_lgbm(path: str | Path) -> FQEModel:
    """Load a boosted FQE artifact saved by :meth:`FQEModel.save`."""
    return FQEModel.load(path)


@dataclass
class _ConstantBooster:
    value: float

    def predict(self, features: Array, **_: Any) -> Array:
        return np.full(np.asarray(features).shape[0], float(self.value), dtype=np.float64)

    def num_trees(self) -> int:
        return 0


def fit_fqe_lgbm(
    states: Array,
    actions: Array,
    next_states: Array,
    next_actions: Array,
    rewards: Array,
    gamma: float,
    terminals: Array | None = None,
    timeouts: Array | None = None,
    continuation: Array | None = None,
    sample_weight: Array | None = None,
    next_action_weights: Array | None = None,
    config: BoostedFQEConfig | None = None,
    categorical_feature: Sequence[int | str] | None = None,
    action_spec: dict[str, Any] | None = None,
) -> FQEModel:
    """Fit LightGBM FQE on state-action transitions."""
    config = BoostedFQEConfig.stable_defaults() if config is None else config
    rewards_1d = _as_1d_float(rewards, "rewards")
    states_2d = _as_2d_float(states, "states", n_rows=rewards_1d.shape[0])
    resolved_action_spec = _normalize_action_spec(action_spec)
    actions_2d = _as_action_features(actions, "actions", n_rows=rewards_1d.shape[0], action_spec=resolved_action_spec)
    next_states_2d = _as_2d_float(next_states, "next_states", n_rows=rewards_1d.shape[0])
    next_actions_3d = _as_next_action_features(
        next_actions,
        n_rows=rewards_1d.shape[0],
        action_dim=actions_2d.shape[1],
        action_spec=resolved_action_spec,
    )
    next_action_weight_2d = validate_action_weights(
        next_action_weights,
        n_rows=rewards_1d.shape[0],
        n_actions=next_actions_3d.shape[1],
        name="next_action_weights",
    )
    features = np.concatenate([states_2d, actions_2d], axis=1)
    next_features = _next_state_action_features(next_states_2d, next_actions_3d)
    return _fit_lgbm_fixed_point(
        features=features,
        next_features=next_features,
        next_action_weights=next_action_weight_2d,
        rewards=rewards_1d,
        gamma=gamma,
        terminals=terminals,
        timeouts=timeouts,
        continuation=continuation,
        sample_weight=sample_weight,
        config=config,
        mode="q",
        state_dim=states_2d.shape[1],
        action_dim=actions_2d.shape[1],
        categorical_feature=categorical_feature,
        action_spec=resolved_action_spec,
    )


def fit_value_lgbm(
    states: Array,
    next_states: Array,
    rewards: Array,
    gamma: float,
    terminals: Array | None = None,
    timeouts: Array | None = None,
    continuation: Array | None = None,
    sample_weight: Array | None = None,
    config: BoostedFQEConfig | None = None,
    categorical_feature: Sequence[int | str] | None = None,
) -> FQEModel:
    """Fit LightGBM fitted value iteration with no action inputs."""
    config = BoostedFQEConfig.stable_defaults() if config is None else config
    rewards_1d = _as_1d_float(rewards, "rewards")
    states_2d = _as_2d_float(states, "states", n_rows=rewards_1d.shape[0])
    next_states_2d = _as_2d_float(next_states, "next_states", n_rows=rewards_1d.shape[0])
    next_features = next_states_2d[:, None, :]
    return _fit_lgbm_fixed_point(
        features=states_2d,
        next_features=next_features,
        next_action_weights=np.ones((rewards_1d.shape[0], 1), dtype=np.float64),
        rewards=rewards_1d,
        gamma=gamma,
        terminals=terminals,
        timeouts=timeouts,
        continuation=continuation,
        sample_weight=sample_weight,
        config=config,
        mode="value",
        state_dim=states_2d.shape[1],
        action_dim=None,
        categorical_feature=categorical_feature,
        action_spec=None,
    )


def fit_fqe_from_policy(
    states: Array,
    actions: Array,
    next_states: Array,
    rewards: Array,
    gamma: float,
    next_action_sampler: NextActionSampler,
    *,
    n_next_action_samples: int = 1,
    terminals: Array | None = None,
    timeouts: Array | None = None,
    continuation: Array | None = None,
    sample_weight: Array | None = None,
    next_action_weights: Array | None = None,
    config: BoostedFQEConfig | None = None,
    categorical_feature: Sequence[int | str] | None = None,
    action_spec: dict[str, Any] | None = None,
) -> FQEModel:
    """Sample evaluation-policy next actions and fit Q-mode FQE."""
    if n_next_action_samples <= 0:
        raise ValueError("n_next_action_samples must be positive.")
    config = BoostedFQEConfig.stable_defaults() if config is None else config
    rewards_1d = _as_1d_float(rewards, "rewards")
    next_states_2d = _as_2d_float(next_states, "next_states", n_rows=rewards_1d.shape[0])
    rng = np.random.default_rng(config.seed)
    next_actions = next_action_sampler(next_states_2d, rng, int(n_next_action_samples))
    return fit_fqe_lgbm(
        states=states,
        actions=actions,
        next_states=next_states_2d,
        next_actions=next_actions,
        rewards=rewards_1d,
        gamma=gamma,
        terminals=terminals,
        timeouts=timeouts,
        continuation=continuation,
        sample_weight=sample_weight,
        next_action_weights=next_action_weights,
        config=config,
        categorical_feature=categorical_feature,
        action_spec=action_spec,
    )


def tune_fqe_cv(
    *,
    param_grid: Sequence[dict[str, Any]],
    states: Array,
    rewards: Array,
    gamma: float,
    actions: Array | None = None,
    next_states: Array,
    next_actions: Array | None = None,
    terminals: Array | None = None,
    timeouts: Array | None = None,
    continuation: Array | None = None,
    sample_weight: Array | None = None,
    next_action_weights: Array | None = None,
    action_spec: dict[str, Any] | None = None,
    base_config: BoostedFQEConfig | None = None,
    validation_fraction: float | None = None,
    seed: int | None = None,
    fit_final: bool = True,
) -> dict[str, Any]:
    """Tune config fields or LightGBM params by held-out Bellman risk."""
    if not param_grid:
        raise ValueError("param_grid must contain at least one candidate.")
    base_config = BoostedFQEConfig.stable_defaults() if base_config is None else base_config
    rewards_1d = _as_1d_float(rewards, "rewards")
    gamma = _validate_gamma(gamma)
    frac = base_config.validation_fraction if validation_fraction is None else float(validation_fraction)
    if not (0.0 < frac < 1.0):
        raise ValueError("validation_fraction must be in (0, 1).")
    split_seed = base_config.seed if seed is None else int(seed)
    train_idx, val_idx = _train_validation_indices(rewards_1d.shape[0], frac, split_seed)
    candidates: list[dict[str, Any]] = []
    for idx, updates in enumerate(param_grid):
        cfg = _config_with_updates(base_config, updates)
        cfg = replace(cfg, refit_on_all_data=False, validation_fraction=frac, seed=split_seed + idx)
        fit_kwargs = {
            "rewards": rewards_1d[train_idx],
            "gamma": gamma,
            "terminals": None if terminals is None else np.asarray(terminals)[train_idx],
            "timeouts": None if timeouts is None else np.asarray(timeouts)[train_idx],
            "continuation": None if continuation is None else np.asarray(continuation)[train_idx],
            "sample_weight": None if sample_weight is None else np.asarray(sample_weight)[train_idx],
            "config": cfg,
        }
        if actions is None and next_actions is None:
            model = fit_value_lgbm(
                states=np.asarray(states)[train_idx],
                next_states=np.asarray(next_states)[train_idx],
                **fit_kwargs,
            )
            val_pred = model.predict_value(np.asarray(states)[val_idx])
            val_next = model.predict_value(np.asarray(next_states)[val_idx])
        elif actions is not None and next_actions is not None:
            model = fit_fqe_lgbm(
                states=np.asarray(states)[train_idx],
                actions=np.asarray(actions)[train_idx],
                next_states=np.asarray(next_states)[train_idx],
                next_actions=_slice_next_actions(np.asarray(next_actions), train_idx),
                next_action_weights=None if next_action_weights is None else np.asarray(next_action_weights)[train_idx],
                action_spec=action_spec,
                **fit_kwargs,
            )
            val_pred = model.predict_q(np.asarray(states)[val_idx], np.asarray(actions)[val_idx])
            val_next = _predict_next_average(model.booster, _next_state_action_features(
                _as_2d_float(np.asarray(next_states)[val_idx], "next_states"),
                _as_next_action_features(
                    _slice_next_actions(np.asarray(next_actions), val_idx),
                    n_rows=len(val_idx),
                    action_dim=model.action_dim,
                    action_spec=model.action_spec,
                ),
            ), None if next_action_weights is None else np.asarray(next_action_weights)[val_idx])
        else:
            raise ValueError("actions and next_actions must either both be supplied or both be omitted.")
        val_rewards = rewards_1d[val_idx]
        val_bootstrap = validate_bootstrap_inputs(
            n_rows=len(val_idx),
            terminals=None if terminals is None else np.asarray(terminals)[val_idx],
            timeouts=None if timeouts is None else np.asarray(timeouts)[val_idx],
            continuation=None if continuation is None else np.asarray(continuation)[val_idx],
        )
        val_weights = _optional_weights(None if sample_weight is None else np.asarray(sample_weight)[val_idx], len(val_idx), "sample_weight")
        score = _bellman_risk(
            predictions=val_pred,
            next_predictions=val_next,
            rewards=val_rewards,
            gamma=gamma,
            terminals=1.0 - val_bootstrap.continuation,
            sample_weight=val_weights,
        )
        candidates.append({"index": idx, "params": dict(updates), "score": float(score), "model": model})
    best = min(candidates, key=lambda row: row["score"])
    final_model = None
    if fit_final:
        final_config = _config_with_updates(base_config, best["params"])
        if actions is None and next_actions is None:
            final_model = fit_value_lgbm(
                states=states,
                next_states=next_states,
                rewards=rewards,
                gamma=gamma,
                terminals=terminals,
                timeouts=timeouts,
                continuation=continuation,
                sample_weight=sample_weight,
                config=final_config,
            )
        else:
            final_model = fit_fqe_lgbm(
                states=states,
                actions=actions,
                next_states=next_states,
                next_actions=next_actions,
                rewards=rewards,
                gamma=gamma,
                terminals=terminals,
                timeouts=timeouts,
                continuation=continuation,
                sample_weight=sample_weight,
                next_action_weights=next_action_weights,
                action_spec=action_spec,
                config=final_config,
            )
    return {
        "best_params": dict(best["params"]),
        "best_score": float(best["score"]),
        "results": [{k: v for k, v in row.items() if k != "model"} for row in candidates],
        "model": final_model,
        "validation_indices": val_idx,
        "train_indices": train_idx,
    }


def _fit_lgbm_fixed_point(
    *,
    features: Array,
    next_features: Array,
    next_action_weights: Array,
    rewards: Array,
    gamma: float,
    terminals: Array | None,
    timeouts: Array | None,
    continuation: Array | None,
    sample_weight: Array | None,
    config: BoostedFQEConfig,
    mode: ModeName,
    state_dim: int,
    action_dim: int | None,
    categorical_feature: Sequence[int | str] | None,
    action_spec: dict[str, Any] | None,
) -> FQEModel:
    gamma = _validate_gamma(gamma)
    n = rewards.shape[0]
    bootstrap = validate_bootstrap_inputs(n_rows=n, terminals=terminals, timeouts=timeouts, continuation=continuation)
    continuation_1d = bootstrap.continuation
    weights_1d = _optional_weights(sample_weight, n, "sample_weight")
    _validate_features(features, next_features, rewards)
    if _is_constant_feature_matrix(features):
        return _fit_constant_fixed_point(
            rewards=rewards,
            gamma=gamma,
            continuation=continuation_1d,
            sample_weight=weights_1d,
            config=config,
            mode=mode,
            state_dim=state_dim,
            action_dim=action_dim,
            categorical_feature=categorical_feature,
            action_spec=action_spec,
            bootstrap_diagnostics=bootstrap.diagnostics,
        )
    _require_lightgbm()
    train_idx, val_idx = _train_validation_indices(n, config.validation_fraction, config.seed)
    target_min, target_max = _resolve_target_bounds(rewards, gamma, config)
    params = _default_lgb_params(config)
    learning_rate = float(params.get("learning_rate", 0.05))
    min_learning_rate = config.min_learning_rate
    if min_learning_rate is None:
        min_learning_rate = min(1e-5, learning_rate / 10_000.0)

    train_set = lgb.Dataset(
        features[train_idx],
        label=rewards[train_idx],
        weight=weights_1d[train_idx],
        free_raw_data=False,
        categorical_feature=categorical_feature,
    )
    all_set = lgb.Dataset(
        features,
        label=rewards,
        weight=weights_1d,
        free_raw_data=False,
        categorical_feature=categorical_feature,
    )

    booster = None
    booster_all = None
    next_pred_train = np.zeros(train_idx.shape[0], dtype=np.float64)
    next_pred_val = np.zeros(val_idx.shape[0], dtype=np.float64)
    next_pred_all = np.zeros(n, dtype=np.float64)
    best_risk = np.inf
    best_booster = None
    best_booster_all = None
    patience = 0
    stopped_early = False
    stop_reason = ""
    accepted_iterations = 0
    history: list[dict[str, Any]] = []

    iterator: Iterable[int] = range(config.num_iterations)
    if config.show_progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(iterator, desc="FQE boosting")
        except ModuleNotFoundError:
            pass

    for iteration in iterator:
        if iteration % config.update_target_frequency == 0:
            train_targets = build_bellman_target(
                rewards=rewards[train_idx],
                gamma=gamma,
                next_predictions=next_pred_train,
                continuation=continuation_1d[train_idx],
                target_min=target_min,
                target_max=target_max,
            )
            train_set.set_label(train_targets)
        params_iter = dict(params)
        params_iter["learning_rate"] = learning_rate
        with _lightgbm_verbosity(config.show_progress):
            start_iteration = 0 if booster is None else booster.num_trees()
            booster = lgb.train(
                params_iter,
                train_set,
                num_boost_round=int(config.trees_per_iteration),
                init_model=booster,
                keep_training_booster=True,
            )
        pred_train_candidate = _predict_booster(booster, features[train_idx])
        pred_val_candidate = _predict_booster(booster, features[val_idx])
        next_pred_train_candidate = _predict_next_average(booster, next_features[train_idx], next_action_weights[train_idx])
        next_pred_val_candidate = _predict_next_average(booster, next_features[val_idx], next_action_weights[val_idx])
        val_risk = _bellman_risk(
            predictions=pred_val_candidate,
            next_predictions=next_pred_val,
            rewards=rewards[val_idx],
            gamma=gamma,
            terminals=1.0 - continuation_1d[val_idx],
            sample_weight=weights_1d[val_idx],
        )
        train_risk = _bellman_risk(
            predictions=pred_train_candidate,
            next_predictions=next_pred_train,
            rewards=rewards[train_idx],
            gamma=gamma,
            terminals=1.0 - continuation_1d[train_idx],
            sample_weight=weights_1d[train_idx],
        )
        improved = (not config.early_stopping) or val_risk <= best_risk - float(config.min_improvement) or not np.isfinite(best_risk)
        if improved:
            best_risk = min(best_risk, val_risk)
            best_booster = booster.model_to_string()
            patience = 0
            next_pred_train = next_pred_train_candidate
            next_pred_val = next_pred_val_candidate
            accepted_iterations += 1
            if config.refit_on_all_data:
                all_targets = build_bellman_target(
                    rewards=rewards,
                    gamma=gamma,
                    next_predictions=next_pred_all,
                    continuation=continuation_1d,
                    target_min=target_min,
                    target_max=target_max,
                )
                all_targets = _clip_targets(
                    all_targets,
                    target_min,
                    target_max,
                )
                all_set.set_label(all_targets)
                params_all = dict(params_iter)
                with _lightgbm_verbosity(config.show_progress):
                    booster_all = lgb.train(
                        params_all,
                        all_set,
                        num_boost_round=int(config.trees_per_iteration),
                        init_model=booster_all,
                        keep_training_booster=True,
                    )
                next_pred_all = _predict_next_average(booster_all, next_features, next_action_weights)
                best_booster_all = booster_all.model_to_string()
        else:
            patience += 1
            if booster is not None:
                _rollback_trees(booster, booster.num_trees() - start_iteration)
            learning_rate *= float(config.learning_rate_backoff)
        history.append(
            {
                "iteration": int(iteration),
                "accepted": bool(improved),
                "train_bellman_risk": float(train_risk),
                "validation_bellman_risk": float(val_risk),
                "best_validation_bellman_risk": float(best_risk),
                "learning_rate": float(learning_rate),
                "trees": int(0 if booster is None else booster.num_trees()),
            }
        )
        if config.early_stopping and config.patience > 0 and patience >= config.patience:
            stopped_early = True
            stop_reason = "patience"
            break
        if config.early_stopping and learning_rate < float(min_learning_rate):
            stopped_early = True
            stop_reason = "min_learning_rate"
            break

    if config.refit_on_all_data and best_booster_all is not None:
        final_booster = lgb.Booster(model_str=best_booster_all)
    elif best_booster is not None:
        final_booster = lgb.Booster(model_str=best_booster)
    else:
        final_booster = booster_all if config.refit_on_all_data and booster_all is not None else booster
    if final_booster is None:
        raise RuntimeError("LightGBM FQE did not fit any trees.")
    final_current = _predict_booster(final_booster, features)
    final_next = _predict_next_average(final_booster, next_features, next_action_weights)
    final_self_risk = _bellman_risk(
        predictions=final_current,
        next_predictions=final_next,
        rewards=rewards,
        gamma=gamma,
        terminals=1.0 - continuation_1d,
        sample_weight=weights_1d,
    )
    diagnostics = {
        "mode": mode,
        "gamma": float(gamma),
        "loss": str(config.loss),
        "num_iterations_requested": int(config.num_iterations),
        "accepted_iterations": int(accepted_iterations),
        "trees": int(final_booster.num_trees()),
        "stopped_early": bool(stopped_early),
        "stop_reason": stop_reason,
        "best_validation_bellman_risk": float(best_risk),
        "final_train_bellman_risk": float(history[-1]["train_bellman_risk"]) if history else np.nan,
        "final_validation_bellman_risk": float(history[-1]["validation_bellman_risk"]) if history else np.nan,
        "final_self_bellman_risk": float(final_self_risk),
        "target_action_expectation": "weighted" if not _is_uniform_action_weights(next_action_weights) else "uniform",
        "target_min": target_min,
        "target_max": target_max,
        "n_samples": int(n),
        "n_train": int(train_idx.shape[0]),
        "n_validation": int(val_idx.shape[0]),
    }
    diagnostics.update(bootstrap.diagnostics)
    if action_spec is not None:
        diagnostics["action_spec_type"] = str(action_spec.get("type", "continuous"))
    return FQEModel(
        booster=final_booster,
        mode=mode,
        gamma=gamma,
        state_dim=int(state_dim),
        action_dim=None if action_dim is None else int(action_dim),
        config=config,
        history=history,
        diagnostics=diagnostics,
        train_indices=train_idx,
        validation_indices=val_idx,
        categorical_feature=categorical_feature,
        action_spec=action_spec,
    )


def _require_lightgbm() -> None:
    if lgb is None:
        raise ModuleNotFoundError(
            "LightGBM is required to fit fqe models. Install this package with its "
            "dependencies or run `pip install lightgbm`."
        )


def _is_constant_feature_matrix(features: Array) -> bool:
    return bool(np.all(np.ptp(np.asarray(features, dtype=np.float64), axis=0) <= 1e-12))


def _fit_constant_fixed_point(
    *,
    rewards: Array,
    gamma: float,
    continuation: Array,
    sample_weight: Array,
    config: BoostedFQEConfig,
    mode: ModeName,
    state_dim: int,
    action_dim: int | None,
    categorical_feature: Sequence[int | str] | None,
    action_spec: dict[str, Any] | None,
    bootstrap_diagnostics: dict[str, float | str],
) -> FQEModel:
    train_idx, val_idx = _train_validation_indices(rewards.shape[0], config.validation_fraction, config.seed)
    value = 0.0
    history: list[dict[str, Any]] = []
    best_risk = np.inf
    patience = 0
    stopped_early = False
    stop_reason = ""
    for iteration in range(config.num_iterations):
        target = build_bellman_target(
            rewards=rewards,
            gamma=gamma,
            next_predictions=np.full_like(rewards, value, dtype=np.float64),
            continuation=continuation,
            target_min=config.target_min,
            target_max=config.target_max,
        )
        candidate = float(np.average(target, weights=sample_weight))
        pred = np.full_like(rewards, candidate, dtype=np.float64)
        next_pred = np.full_like(rewards, value, dtype=np.float64)
        val_risk = _bellman_risk(
            predictions=pred[val_idx],
            next_predictions=next_pred[val_idx],
            rewards=rewards[val_idx],
            gamma=gamma,
            terminals=1.0 - continuation[val_idx],
            sample_weight=sample_weight[val_idx],
        )
        train_risk = _bellman_risk(
            predictions=pred[train_idx],
            next_predictions=next_pred[train_idx],
            rewards=rewards[train_idx],
            gamma=gamma,
            terminals=1.0 - continuation[train_idx],
            sample_weight=sample_weight[train_idx],
        )
        improved = True
        if improved:
            best_risk = min(best_risk, val_risk)
            patience = 0
            value = candidate
        else:
            patience += 1
        history.append(
            {
                "iteration": int(iteration),
                "accepted": bool(improved),
                "train_bellman_risk": float(train_risk),
                "validation_bellman_risk": float(val_risk),
                "best_validation_bellman_risk": float(best_risk),
                "learning_rate": 0.0,
                "trees": 0,
            }
        )
        if config.early_stopping and config.patience > 0 and patience >= config.patience:
            stopped_early = True
            stop_reason = "patience"
            break
    booster = _ConstantBooster(value=value)
    final_pred = np.full_like(rewards, value, dtype=np.float64)
    final_self_risk = _bellman_risk(
        predictions=final_pred,
        next_predictions=final_pred,
        rewards=rewards,
        gamma=gamma,
        terminals=1.0 - continuation,
        sample_weight=sample_weight,
    )
    diagnostics = {
        "mode": mode,
        "gamma": float(gamma),
        "loss": str(config.loss),
        "num_iterations_requested": int(config.num_iterations),
        "accepted_iterations": int(sum(1 for row in history if row["accepted"])),
        "trees": 0,
        "stopped_early": bool(stopped_early),
        "stop_reason": stop_reason,
        "best_validation_bellman_risk": float(best_risk),
        "final_train_bellman_risk": float(history[-1]["train_bellman_risk"]) if history else np.nan,
        "final_validation_bellman_risk": float(history[-1]["validation_bellman_risk"]) if history else np.nan,
        "final_self_bellman_risk": float(final_self_risk),
        "target_action_expectation": "none" if mode == "value" else "uniform",
        "target_min": config.target_min,
        "target_max": config.target_max,
        "n_samples": int(rewards.shape[0]),
        "n_train": int(train_idx.shape[0]),
        "n_validation": int(val_idx.shape[0]),
        "constant_feature_fallback": True,
    }
    diagnostics.update(bootstrap_diagnostics)
    if action_spec is not None:
        diagnostics["action_spec_type"] = str(action_spec.get("type", "continuous"))
    return FQEModel(
        booster=booster,
        mode=mode,
        gamma=float(gamma),
        state_dim=int(state_dim),
        action_dim=None if action_dim is None else int(action_dim),
        config=config,
        history=history,
        diagnostics=diagnostics,
        train_indices=train_idx,
        validation_indices=val_idx,
        categorical_feature=categorical_feature,
        action_spec=action_spec,
    )


def _default_lgb_params(config: BoostedFQEConfig) -> dict[str, Any]:
    params: dict[str, Any] = {
        "objective": "regression" if config.loss == "squared" else _huber_objective_factory(config),
        "metric": "None",
        "verbosity": -1,
        "seed": int(config.seed),
        "feature_pre_filter": False,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 20,
        "lambda_l1": 0.0,
        "lambda_l2": 1.0,
        "min_sum_hessian_in_leaf": 1e-3,
        "num_threads": int(config.num_threads),
    }
    params.update(dict(config.lgb_params))
    if config.loss == "huber":
        params["objective"] = _huber_objective_factory(config)
    return params


def _huber_objective_factory(config: BoostedFQEConfig):
    def objective(preds: Array, dataset: Any) -> tuple[Array, Array]:
        labels = dataset.get_label()
        weights = dataset.get_weight()
        if weights is None:
            weights = np.ones_like(preds)
        residual = preds - labels
        delta = _resolve_huber_delta(residual, config)
        abs_residual = np.abs(residual)
        grad = np.where(abs_residual <= delta, residual, delta * np.sign(residual))
        hess = np.where(abs_residual <= delta, 1.0, float(config.huber_hessian_floor))
        return grad * weights, hess * weights

    return objective


def _resolve_huber_delta(residual: Array, config: BoostedFQEConfig) -> float:
    if config.huber_delta is not None:
        return float(config.huber_delta)
    finite = np.asarray(residual, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 1.0
    mad = float(np.median(np.abs(finite - np.median(finite))))
    robust_sigma = max(1.4826 * mad, float(np.std(finite)), 1e-8)
    return float(config.huber_delta_scale * robust_sigma)


def _validate_gamma(gamma: float) -> float:
    value = float(gamma)
    if not (0.0 <= value < 1.0):
        raise ValueError("gamma must be in [0, 1).")
    return value


def _as_1d_float(x: Array, name: str) -> Array:
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} must be nonempty.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values.")
    return arr


def _as_2d_float(x: Array, name: str, n_rows: int | None = None, expected_cols: int | None = None) -> Array:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    elif arr.ndim != 2:
        raise ValueError(f"{name} must be a 1D or 2D array.")
    if n_rows is not None and arr.shape[0] != int(n_rows):
        raise ValueError(f"{name} must have {n_rows} rows.")
    if expected_cols is not None and arr.shape[1] != int(expected_cols):
        raise ValueError(f"{name} must have {expected_cols} columns.")
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError(f"{name} must be nonempty.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values.")
    return np.ascontiguousarray(arr)


def _normalize_action_spec(action_spec: dict[str, Any] | None) -> dict[str, Any] | None:
    if action_spec is None:
        return None
    spec = dict(action_spec)
    kind = str(spec.get("type", "continuous")).lower()
    aliases = {
        "continuous": "continuous",
        "onehot": "one_hot",
        "one_hot": "one_hot",
        "discrete": "discrete_index",
        "discrete_index": "discrete_index",
        "categorical": "categorical",
    }
    if kind not in aliases:
        raise ValueError("action_spec type must be continuous, one_hot, discrete_index, or categorical.")
    spec["type"] = aliases[kind]
    if spec.get("n_actions") is not None:
        n_actions = int(spec["n_actions"])
        if n_actions <= 0:
            raise ValueError("action_spec n_actions must be positive.")
        spec["n_actions"] = n_actions
    if spec["type"] == "discrete_index" and spec.get("n_actions") is None:
        raise ValueError("action_spec n_actions is required for discrete_index actions.")
    return spec


def _as_action_features(
    actions: Array,
    name: str,
    *,
    n_rows: int,
    action_spec: dict[str, Any] | None,
    expected_cols: int | None = None,
) -> Array:
    kind = "continuous" if action_spec is None else str(action_spec.get("type", "continuous"))
    if kind == "discrete_index":
        n_actions = int(action_spec["n_actions"])
        raw = np.asarray(actions)
        if raw.ndim == 2 and raw.shape[1] == n_actions:
            arr = _as_2d_float(actions, name, n_rows=n_rows, expected_cols=expected_cols)
            _validate_one_hot_rows(arr, name)
            return arr
        index = _as_index_matrix(actions, name, n_rows=n_rows, n_actions=n_actions)
        if index.shape[1] != 1:
            raise ValueError(f"{name} must contain exactly one action per row.")
        if expected_cols is not None and int(expected_cols) != n_actions:
            raise ValueError(f"{name} must have {expected_cols} columns after one-hot encoding.")
        return _one_hot_actions(index[:, 0], n_actions)
    if kind == "categorical":
        n_actions = None if action_spec is None else action_spec.get("n_actions")
        index = _as_index_matrix(actions, name, n_rows=n_rows, n_actions=None if n_actions is None else int(n_actions))
        if index.shape[1] != 1:
            raise ValueError(f"{name} must contain exactly one action per row.")
        if expected_cols is not None and int(expected_cols) != 1:
            raise ValueError(f"{name} must have one categorical action column.")
        return np.ascontiguousarray(index[:, :1], dtype=np.float64)
    arr = _as_2d_float(actions, name, n_rows=n_rows, expected_cols=expected_cols)
    if kind == "one_hot":
        n_actions = None if action_spec is None else action_spec.get("n_actions")
        if n_actions is not None and arr.shape[1] != int(n_actions):
            raise ValueError(f"{name} must have {int(n_actions)} one-hot action columns.")
        _validate_one_hot_rows(arr, name)
    return arr


def _as_next_action_features(
    next_actions: Array,
    *,
    n_rows: int,
    action_dim: int | None,
    action_spec: dict[str, Any] | None,
) -> Array:
    if action_dim is None:
        raise ValueError("action_dim is required for next action features.")
    kind = "continuous" if action_spec is None else str(action_spec.get("type", "continuous"))
    if kind == "discrete_index":
        n_actions = int(action_spec["n_actions"])
        if int(action_dim) != n_actions:
            raise ValueError("action_dim must match action_spec n_actions.")
        index = _as_index_matrix(next_actions, "next_actions", n_rows=n_rows, n_actions=n_actions)
        return _one_hot_actions(index.reshape(-1), n_actions).reshape(n_rows, index.shape[1], n_actions)
    if kind == "categorical":
        n_actions = None if action_spec is None else action_spec.get("n_actions")
        index = _as_index_matrix(
            next_actions,
            "next_actions",
            n_rows=n_rows,
            n_actions=None if n_actions is None else int(n_actions),
        )
        if int(action_dim) != 1:
            raise ValueError("categorical action features must have action_dim 1.")
        return np.ascontiguousarray(index[:, :, None], dtype=np.float64)
    arr = _as_next_actions(next_actions, n_rows, action_dim)
    if kind == "one_hot":
        n_actions = None if action_spec is None else action_spec.get("n_actions")
        if n_actions is not None and arr.shape[2] != int(n_actions):
            raise ValueError(f"next_actions must have {int(n_actions)} one-hot action columns.")
        _validate_one_hot_rows(arr.reshape(-1, arr.shape[2]), "next_actions")
    return arr


def _as_initial_actions(
    initial_actions: Array,
    *,
    n_rows: int,
    action_dim: int | None,
    action_spec: dict[str, Any] | None,
) -> Array:
    if action_dim is None:
        raise ValueError("action_dim is required for initial actions.")
    kind = "continuous" if action_spec is None else str(action_spec.get("type", "continuous"))
    if kind == "discrete_index":
        n_actions = int(action_spec["n_actions"])
        if int(action_dim) != n_actions:
            raise ValueError("action_dim must match action_spec n_actions.")
        index = _as_index_matrix(initial_actions, "initial_actions", n_rows=n_rows, n_actions=n_actions)
        return _one_hot_actions(index.reshape(-1), n_actions).reshape(n_rows, index.shape[1], n_actions)
    if kind == "categorical":
        n_actions = None if action_spec is None else action_spec.get("n_actions")
        index = _as_index_matrix(
            initial_actions,
            "initial_actions",
            n_rows=n_rows,
            n_actions=None if n_actions is None else int(n_actions),
        )
        if int(action_dim) != 1:
            raise ValueError("categorical action features must have action_dim 1.")
        return np.ascontiguousarray(index[:, :, None], dtype=np.float64)
    arr = _as_next_actions(initial_actions, n_rows, action_dim)
    if kind == "one_hot":
        n_actions = None if action_spec is None else action_spec.get("n_actions")
        if n_actions is not None and arr.shape[2] != int(n_actions):
            raise ValueError(f"initial_actions must have {int(n_actions)} one-hot action columns.")
        _validate_one_hot_rows(arr.reshape(-1, arr.shape[2]), "initial_actions")
    return arr


def _as_index_matrix(actions: Array, name: str, *, n_rows: int, n_actions: int | None) -> Array:
    arr = np.asarray(actions)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    elif arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]
    elif arr.ndim != 2:
        raise ValueError(f"{name} must be a 1D, 2D, or last-dimension-one 3D array of action indices.")
    if arr.shape[0] != int(n_rows):
        raise ValueError(f"{name} must have {n_rows} rows.")
    if arr.shape[1] <= 0:
        raise ValueError(f"{name} must contain at least one action column/sample.")
    as_float = np.asarray(arr, dtype=np.float64)
    if not np.all(np.isfinite(as_float)):
        raise ValueError(f"{name} must contain only finite action indices.")
    rounded = np.rint(as_float)
    if not np.allclose(as_float, rounded, rtol=0.0, atol=1e-8):
        raise ValueError(f"{name} must contain integer action indices.")
    index = rounded.astype(np.int64)
    if np.any(index < 0):
        raise ValueError(f"{name} action indices must be nonnegative.")
    if n_actions is not None and np.any(index >= int(n_actions)):
        raise ValueError(f"{name} action indices must be less than n_actions.")
    return np.ascontiguousarray(index)


def _one_hot_actions(index: Array, n_actions: int) -> Array:
    idx = np.asarray(index, dtype=np.int64).reshape(-1)
    out = np.zeros((idx.shape[0], int(n_actions)), dtype=np.float64)
    out[np.arange(idx.shape[0]), idx] = 1.0
    return np.ascontiguousarray(out)


def _validate_one_hot_rows(actions: Array, name: str) -> None:
    arr = np.asarray(actions, dtype=np.float64)
    if np.any((arr < -1e-8) | (arr > 1.0 + 1e-8)):
        raise ValueError(f"{name} one-hot rows must be in [0, 1].")
    row_sum = np.sum(arr, axis=1)
    if not np.allclose(row_sum, 1.0, rtol=1e-6, atol=1e-6):
        raise ValueError(f"{name} one-hot rows must sum to 1.")


def _as_next_actions(next_actions: Array, n_rows: int, action_dim: int | None) -> Array:
    arr = np.asarray(next_actions, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim == 2:
        arr = arr[:, None, :]
    elif arr.ndim != 3:
        raise ValueError("next_actions must be a 1D, 2D, or 3D array.")
    if arr.shape[0] != int(n_rows):
        raise ValueError(f"next_actions must have {n_rows} rows.")
    if arr.shape[1] <= 0 or arr.shape[2] <= 0:
        raise ValueError("next_actions must have positive action sample and feature dimensions.")
    if action_dim is not None and arr.shape[2] != int(action_dim):
        raise ValueError(f"next_actions must have action dimension {action_dim}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("next_actions must contain only finite values.")
    return np.ascontiguousarray(arr)


def _optional_terminals(terminals: Array | None, n_rows: int) -> Array:
    if terminals is None:
        return np.zeros(int(n_rows), dtype=np.float64)
    arr = np.asarray(terminals, dtype=np.float64).reshape(-1)
    if arr.shape[0] != int(n_rows):
        raise ValueError(f"terminals must have {n_rows} rows.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("terminals must contain only finite values.")
    if np.any((arr < 0.0) | (arr > 1.0)):
        raise ValueError("terminals must be in [0, 1].")
    return arr


def _optional_weights(weights: Array | None, n_rows: int, name: str) -> Array:
    if weights is None:
        return np.ones(int(n_rows), dtype=np.float64)
    arr = np.asarray(weights, dtype=np.float64).reshape(-1)
    if arr.shape[0] != int(n_rows):
        raise ValueError(f"{name} must have {n_rows} rows.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values.")
    if np.any(arr < 0.0):
        raise ValueError(f"{name} must be nonnegative.")
    if float(np.sum(arr)) <= 0.0:
        raise ValueError(f"{name} must have positive total weight.")
    return arr


def _state_action_features(
    states: Array,
    actions: Array,
    state_dim: int,
    action_dim: int | None,
    action_spec: dict[str, Any] | None = None,
) -> Array:
    if action_dim is None:
        raise ValueError("action_dim is required for state-action features.")
    states_2d = _as_2d_float(states, "states", expected_cols=state_dim)
    actions_2d = _as_action_features(
        actions,
        "actions",
        n_rows=states_2d.shape[0],
        action_spec=action_spec,
        expected_cols=action_dim,
    )
    return np.concatenate([states_2d, actions_2d], axis=1)


def _next_state_action_features(next_states: Array, next_actions: Array) -> Array:
    n, m, action_dim = next_actions.shape
    states_rep = np.repeat(next_states[:, None, :], repeats=m, axis=1)
    return np.concatenate([states_rep, next_actions], axis=2).reshape(n, m, next_states.shape[1] + action_dim)


def _validate_features(features: Array, next_features: Array, rewards: Array) -> None:
    if features.ndim != 2:
        raise ValueError("features must be 2D.")
    if next_features.ndim != 3:
        raise ValueError("next_features must be 3D.")
    if features.shape[0] != rewards.shape[0] or next_features.shape[0] != rewards.shape[0]:
        raise ValueError("features, next_features, and rewards must have aligned rows.")
    if next_features.shape[2] != features.shape[1]:
        raise ValueError("next_features must have the same feature dimension as features.")


def _train_validation_indices(n_rows: int, validation_fraction: float, seed: int) -> tuple[Array, Array]:
    if n_rows < 2:
        raise ValueError("at least two rows are required to create a validation split.")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_rows)
    n_val = min(n_rows - 1, max(1, int(round(float(validation_fraction) * n_rows))))
    val_idx = np.sort(perm[:n_val]).astype(np.int64)
    train_idx = np.sort(perm[n_val:]).astype(np.int64)
    if train_idx.size == 0:
        raise ValueError("validation split left no training rows.")
    return train_idx, val_idx


def _resolve_target_bounds(rewards: Array, gamma: float, config: BoostedFQEConfig) -> tuple[float | None, float | None]:
    target_min = config.target_min
    target_max = config.target_max
    if config.infer_value_bounds and gamma > 0.0:
        scale = max(1.0 - gamma, 1e-12)
        reward_min = float(np.min(rewards))
        reward_max = float(np.max(rewards))
        inferred_min = reward_min / scale
        inferred_max = reward_max / scale
        target_min = inferred_min if target_min is None else max(float(target_min), inferred_min)
        target_max = inferred_max if target_max is None else min(float(target_max), inferred_max)
    return target_min, target_max


def _clip_targets(targets: Array, target_min: float | None, target_max: float | None) -> Array:
    out = np.asarray(targets, dtype=np.float64)
    if target_min is not None or target_max is not None:
        out = np.clip(
            out,
            -np.inf if target_min is None else float(target_min),
            np.inf if target_max is None else float(target_max),
        )
    return out


def _predict_booster(booster: Any, features: Array) -> Array:
    pred = np.asarray(booster.predict(features), dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(pred)):
        raise FloatingPointError("model produced non-finite predictions.")
    return pred


def _predict_next_average(booster: Any, next_features: Array, next_action_weights: Array | None = None) -> Array:
    n, m, d = next_features.shape
    flat_pred = _predict_booster(booster, next_features.reshape(n * m, d))
    return weighted_action_expectation(flat_pred.reshape(n, m), next_action_weights)


def _bellman_risk(
    *,
    predictions: Array,
    next_predictions: Array,
    rewards: Array,
    gamma: float,
    terminals: Array,
    sample_weight: Array,
) -> float:
    terminal_1d = _optional_terminals(terminals, np.asarray(rewards).reshape(-1).shape[0])
    return bellman_risk(
        predictions=predictions,
        next_predictions=next_predictions,
        rewards=rewards,
        gamma=gamma,
        continuation=1.0 - terminal_1d,
        sample_weight=sample_weight,
    )


def _is_uniform_action_weights(weights: Array | None) -> bool:
    if weights is None:
        return True
    arr = np.asarray(weights, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] <= 0:
        return False
    return bool(np.allclose(arr, 1.0 / arr.shape[1], rtol=1e-8, atol=1e-10))


def _rollback_trees(booster: Any, n_trees: int) -> None:
    for _ in range(max(0, int(n_trees))):
        booster.rollback_one_iter()


class _lightgbm_verbosity:
    def __init__(self, verbose: bool) -> None:
        self.verbose = bool(verbose)

    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


def _slice_next_actions(next_actions: Array, indices: Array) -> Array:
    return next_actions[indices]


def _config_with_updates(base: BoostedFQEConfig, updates: dict[str, Any]) -> BoostedFQEConfig:
    updates = dict(updates)
    lgb_updates = updates.pop("lgb_params", None)
    if lgb_updates is not None:
        updates["lgb_params"] = {**base.lgb_params, **dict(lgb_updates)}
    known_fields = set(BoostedFQEConfig.__dataclass_fields__)
    config_updates = {}
    lgb_param_updates = {}
    for key, value in updates.items():
        if key in known_fields:
            config_updates[key] = value
        else:
            lgb_param_updates[key] = value
    if lgb_param_updates:
        config_updates["lgb_params"] = {**base.lgb_params, **lgb_param_updates}
    return replace(base, **config_updates)
