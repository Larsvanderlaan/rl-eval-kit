from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Literal, Sequence

import numpy as np

try:  # pragma: no cover - exercised when torch is installed.
    import torch
    from torch import nn
except ModuleNotFoundError:  # pragma: no cover - environment dependent.
    torch = None
    nn = None

from fqe.fit_fqe import (
    Array,
    LossName,
    ModeName,
    _as_1d_float,
    _as_2d_float,
    _as_action_features,
    _as_initial_actions,
    _as_next_action_features,
    _bellman_risk,
    _is_uniform_action_weights,
    _normalize_action_spec,
    _next_state_action_features,
    _optional_weights,
    _resolve_target_bounds,
    _slice_next_actions,
    _state_action_features,
    _train_validation_indices,
    _validate_features,
    _validate_gamma,
)
from fqe.bellman import (
    package_versions,
    serializable_config,
    validate_action_weights,
    validate_bootstrap_inputs,
    weighted_action_expectation,
)


NeuralNextActionSampler = Callable[[Array, np.random.Generator, int], Array]
ActivationName = Literal["relu", "tanh", "silu", "gelu"]

__all__ = [
    "NeuralFQEConfig",
    "NeuralFQEModel",
    "fit_fqe_neural",
    "fit_value_neural",
    "fit_fqe_neural_from_policy",
    "load_fqe_neural",
    "tune_fqe_neural_cv",
]


@dataclass(frozen=True)
class NeuralFQEConfig:
    """Configuration for neural fitted Q evaluation / value iteration."""

    hidden_dims: Sequence[int] = (256, 256)
    activation: ActivationName = "silu"
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    batch_size: int = 512
    num_iterations: int = 50
    gradient_steps_per_iteration: int = 20
    target_update_tau: float = 0.05
    loss: LossName = "huber"
    huber_delta: float | None = None
    huber_delta_scale: float = 1.345
    validation_fraction: float = 0.2
    early_stopping: bool = True
    patience: int = 8
    min_improvement: float = 1e-5
    grad_clip_norm: float | None = 5.0
    target_min: float | None = None
    target_max: float | None = None
    infer_value_bounds: bool = True
    standardize_inputs: bool = True
    device: str = "cpu"
    seed: int = 123
    show_progress: bool = False

    def __post_init__(self) -> None:
        if not tuple(self.hidden_dims) or any(int(width) <= 0 for width in self.hidden_dims):
            raise ValueError("hidden_dims must contain positive widths.")
        if self.activation not in {"relu", "tanh", "silu", "gelu"}:
            raise ValueError("activation must be one of 'relu', 'tanh', 'silu', or 'gelu'.")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive.")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay must be nonnegative.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.num_iterations <= 0:
            raise ValueError("num_iterations must be positive.")
        if self.gradient_steps_per_iteration <= 0:
            raise ValueError("gradient_steps_per_iteration must be positive.")
        if not (0.0 < self.target_update_tau <= 1.0):
            raise ValueError("target_update_tau must be in (0, 1].")
        if self.loss not in {"squared", "huber"}:
            raise ValueError("loss must be 'squared' or 'huber'.")
        if self.huber_delta is not None and self.huber_delta <= 0.0:
            raise ValueError("huber_delta must be positive when supplied.")
        if self.huber_delta_scale <= 0.0:
            raise ValueError("huber_delta_scale must be positive.")
        if not (0.0 < self.validation_fraction < 1.0):
            raise ValueError("validation_fraction must be in (0, 1).")
        if self.patience < 0:
            raise ValueError("patience must be nonnegative.")
        if self.min_improvement < 0.0:
            raise ValueError("min_improvement must be nonnegative.")
        if self.grad_clip_norm is not None and self.grad_clip_norm <= 0.0:
            raise ValueError("grad_clip_norm must be positive when supplied.")
        if self.target_min is not None and self.target_max is not None and self.target_min > self.target_max:
            raise ValueError("target_min must be <= target_max.")

    @classmethod
    def stable_defaults(cls, **overrides: Any) -> "NeuralFQEConfig":
        """Construct conservative neural defaults for offline RL data."""
        params: dict[str, Any] = {
            "hidden_dims": (256, 256),
            "activation": "silu",
            "learning_rate": 3e-4,
            "weight_decay": 1e-4,
            "batch_size": 512,
            "num_iterations": 50,
            "gradient_steps_per_iteration": 20,
            "target_update_tau": 0.05,
            "loss": "huber",
            "grad_clip_norm": 5.0,
            "validation_fraction": 0.2,
            "patience": 8,
            "min_improvement": 1e-5,
            "infer_value_bounds": True,
            "standardize_inputs": True,
            "device": "cpu",
            "seed": 123,
        }
        params.update(overrides)
        return cls(**params)


@dataclass
class NeuralFQEModel:
    """Fitted neural FQE/FVI model with prediction helpers."""

    network: Any
    target_network: Any
    mode: ModeName
    gamma: float
    state_dim: int
    action_dim: int | None
    config: NeuralFQEConfig
    input_mean: Array
    input_std: Array
    history: list[dict[str, Any]]
    diagnostics: dict[str, Any]
    train_indices: Array
    validation_indices: Array
    action_spec: dict[str, Any] | None = None

    def predict(self, states: Array, actions: Array | None = None) -> Array:
        """Predict Q(s, a) in Q mode or V(s) in value mode."""
        if self.mode == "q":
            if actions is None:
                raise ValueError("actions are required when predicting with a Q-mode NeuralFQEModel.")
            return self.predict_q(states, actions)
        if actions is not None:
            raise ValueError("actions must not be supplied for a value-mode NeuralFQEModel.")
        return self.predict_value(states)

    def predict_q(self, states: Array, actions: Array) -> Array:
        """Predict fitted Q-values."""
        if self.mode != "q":
            raise ValueError("predict_q is only available for Q-mode NeuralFQEModel objects.")
        features = _state_action_features(states, actions, self.state_dim, self.action_dim, self.action_spec)
        return self._predict_features(features)

    def predict_value(self, states: Array) -> Array:
        """Predict fitted state values."""
        if self.mode != "value":
            raise ValueError("predict_value is only available for value-mode NeuralFQEModel objects.")
        features = _as_2d_float(states, "states", expected_cols=self.state_dim)
        return self._predict_features(features)

    def estimate_policy_value(
        self,
        initial_states: Array,
        initial_actions: Array | None = None,
        initial_weights: Array | None = None,
        initial_action_weights: Array | None = None,
    ) -> float:
        """Estimate the policy value by averaging initial predictions."""
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
        """Return a dictionary payload compatible with script-style callers."""
        return {
            "model": self.network,
            "target_model": self.target_network,
            "mode": self.mode,
            "gamma": float(self.gamma),
            "state_dim": int(self.state_dim),
            "action_dim": None if self.action_dim is None else int(self.action_dim),
            "input_mean": np.asarray(self.input_mean, dtype=np.float64),
            "input_std": np.asarray(self.input_std, dtype=np.float64),
            "history": list(self.history),
            "diagnostics": dict(self.diagnostics),
            "train_indices": np.asarray(self.train_indices, dtype=np.int64),
            "validation_indices": np.asarray(self.validation_indices, dtype=np.int64),
            "action_spec": None if self.action_spec is None else dict(self.action_spec),
        }

    def save(self, path: str | Path) -> None:
        """Save a versioned neural FQE model artifact."""
        _require_torch()
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "model_family": "neural_torch",
            "mode": self.mode,
            "gamma": float(self.gamma),
            "state_dim": int(self.state_dim),
            "action_dim": None if self.action_dim is None else int(self.action_dim),
            "config": serializable_config(self.config),
            "input_mean": np.asarray(self.input_mean, dtype=np.float64),
            "input_std": np.asarray(self.input_std, dtype=np.float64),
            "history": list(self.history),
            "diagnostics": dict(self.diagnostics),
            "train_indices": np.asarray(self.train_indices, dtype=np.int64),
            "validation_indices": np.asarray(self.validation_indices, dtype=np.int64),
            "action_spec": None if self.action_spec is None else dict(self.action_spec),
            "versions": package_versions(("numpy", "torch", "fqe")),
            "network_state_dict": {key: value.detach().cpu() for key, value in self.network.state_dict().items()},
            "target_network_state_dict": {
                key: value.detach().cpu() for key, value in self.target_network.state_dict().items()
            },
        }
        torch.save(payload, path_obj)

    @classmethod
    def load(cls, path: str | Path) -> "NeuralFQEModel":
        """Load a neural FQE artifact saved by :meth:`save`."""
        _require_torch()
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        if int(payload.get("schema_version", -1)) != 1 or payload.get("model_family") != "neural_torch":
            raise ValueError("Unsupported neural FQE artifact.")
        config = NeuralFQEConfig(**dict(payload["config"]))
        state_dim = int(payload["state_dim"])
        action_dim = None if payload["action_dim"] is None else int(payload["action_dim"])
        input_dim = state_dim if payload["mode"] == "value" else state_dim + int(action_dim)
        network = _MLP(input_dim, tuple(config.hidden_dims), config.activation)
        target_network = _MLP(input_dim, tuple(config.hidden_dims), config.activation)
        network.load_state_dict(payload["network_state_dict"])
        target_network.load_state_dict(payload["target_network_state_dict"])
        device = torch.device(config.device)
        network.to(device).eval()
        target_network.to(device).eval()
        return cls(
            network=network,
            target_network=target_network,
            mode=payload["mode"],
            gamma=float(payload["gamma"]),
            state_dim=state_dim,
            action_dim=action_dim,
            config=config,
            input_mean=np.asarray(payload["input_mean"], dtype=np.float64),
            input_std=np.asarray(payload["input_std"], dtype=np.float64),
            history=list(payload["history"]),
            diagnostics=dict(payload["diagnostics"]),
            train_indices=np.asarray(payload["train_indices"], dtype=np.int64),
            validation_indices=np.asarray(payload["validation_indices"], dtype=np.int64),
            action_spec=payload.get("action_spec"),
        )

    def _predict_features(self, features: Array) -> Array:
        _require_torch()
        self.network.eval()
        with torch.no_grad():
            x = _features_to_tensor(features, self.input_mean, self.input_std, self.config.device)
            pred = self.network(x).detach().cpu().numpy().astype(np.float64).reshape(-1)
        if not np.all(np.isfinite(pred)):
            raise FloatingPointError("model produced non-finite predictions.")
        return pred


def load_fqe_neural(path: str | Path) -> NeuralFQEModel:
    """Load a neural FQE artifact saved by :meth:`NeuralFQEModel.save`."""
    return NeuralFQEModel.load(path)


def fit_fqe_neural(
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
    config: NeuralFQEConfig | None = None,
    action_spec: dict[str, Any] | None = None,
) -> NeuralFQEModel:
    """Fit neural FQE on state-action transitions."""
    config = NeuralFQEConfig.stable_defaults() if config is None else config
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
    return _fit_neural_fixed_point(
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
        action_spec=resolved_action_spec,
    )


def fit_value_neural(
    states: Array,
    next_states: Array,
    rewards: Array,
    gamma: float,
    terminals: Array | None = None,
    timeouts: Array | None = None,
    continuation: Array | None = None,
    sample_weight: Array | None = None,
    config: NeuralFQEConfig | None = None,
) -> NeuralFQEModel:
    """Fit neural fitted value iteration with no action inputs."""
    config = NeuralFQEConfig.stable_defaults() if config is None else config
    rewards_1d = _as_1d_float(rewards, "rewards")
    states_2d = _as_2d_float(states, "states", n_rows=rewards_1d.shape[0])
    next_states_2d = _as_2d_float(next_states, "next_states", n_rows=rewards_1d.shape[0])
    return _fit_neural_fixed_point(
        features=states_2d,
        next_features=next_states_2d[:, None, :],
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
        action_spec=None,
    )


def fit_fqe_neural_from_policy(
    states: Array,
    actions: Array,
    next_states: Array,
    rewards: Array,
    gamma: float,
    next_action_sampler: NeuralNextActionSampler,
    *,
    n_next_action_samples: int = 1,
    terminals: Array | None = None,
    timeouts: Array | None = None,
    continuation: Array | None = None,
    sample_weight: Array | None = None,
    next_action_weights: Array | None = None,
    config: NeuralFQEConfig | None = None,
    action_spec: dict[str, Any] | None = None,
) -> NeuralFQEModel:
    """Sample evaluation-policy next actions and fit Q-mode neural FQE."""
    if n_next_action_samples <= 0:
        raise ValueError("n_next_action_samples must be positive.")
    config = NeuralFQEConfig.stable_defaults() if config is None else config
    rewards_1d = _as_1d_float(rewards, "rewards")
    next_states_2d = _as_2d_float(next_states, "next_states", n_rows=rewards_1d.shape[0])
    rng = np.random.default_rng(config.seed)
    next_actions = next_action_sampler(next_states_2d, rng, int(n_next_action_samples))
    return fit_fqe_neural(
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
        action_spec=action_spec,
    )


def tune_fqe_neural_cv(
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
    base_config: NeuralFQEConfig | None = None,
    validation_fraction: float | None = None,
    seed: int | None = None,
    fit_final: bool = True,
) -> dict[str, Any]:
    """Tune neural config fields by held-out Bellman risk."""
    if not param_grid:
        raise ValueError("param_grid must contain at least one candidate.")
    base_config = NeuralFQEConfig.stable_defaults() if base_config is None else base_config
    rewards_1d = _as_1d_float(rewards, "rewards")
    gamma = _validate_gamma(gamma)
    frac = base_config.validation_fraction if validation_fraction is None else float(validation_fraction)
    if not (0.0 < frac < 1.0):
        raise ValueError("validation_fraction must be in (0, 1).")
    split_seed = base_config.seed if seed is None else int(seed)
    train_idx, val_idx = _train_validation_indices(rewards_1d.shape[0], frac, split_seed)
    candidates: list[dict[str, Any]] = []
    for idx, updates in enumerate(param_grid):
        cfg = replace(base_config, **dict(updates), validation_fraction=frac, seed=split_seed + idx)
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
            model = fit_value_neural(
                states=np.asarray(states)[train_idx],
                next_states=np.asarray(next_states)[train_idx],
                **fit_kwargs,
            )
            val_pred = model.predict_value(np.asarray(states)[val_idx])
            val_next = model.predict_value(np.asarray(next_states)[val_idx])
        elif actions is not None and next_actions is not None:
            model = fit_fqe_neural(
                states=np.asarray(states)[train_idx],
                actions=np.asarray(actions)[train_idx],
                next_states=np.asarray(next_states)[train_idx],
                next_actions=_slice_next_actions(np.asarray(next_actions), train_idx),
                next_action_weights=None if next_action_weights is None else np.asarray(next_action_weights)[train_idx],
                action_spec=action_spec,
                **fit_kwargs,
            )
            val_pred = model.predict_q(np.asarray(states)[val_idx], np.asarray(actions)[val_idx])
            val_next = _predict_next_average_neural(
                model.target_network,
                _next_state_action_features(
                    _as_2d_float(np.asarray(next_states)[val_idx], "next_states"),
                    _as_next_action_features(
                        _slice_next_actions(np.asarray(next_actions), val_idx),
                        n_rows=len(val_idx),
                        action_dim=model.action_dim,
                        action_spec=model.action_spec,
                    ),
                ),
                model.input_mean,
                model.input_std,
                model.config.device,
                None if next_action_weights is None else np.asarray(next_action_weights)[val_idx],
            )
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
        final_config = replace(base_config, **best["params"])
        if actions is None and next_actions is None:
            final_model = fit_value_neural(
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
            final_model = fit_fqe_neural(
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


def _fit_neural_fixed_point(
    *,
    features: Array,
    next_features: Array,
    next_action_weights: Array | None,
    rewards: Array,
    gamma: float,
    terminals: Array | None,
    timeouts: Array | None,
    continuation: Array | None,
    sample_weight: Array | None,
    config: NeuralFQEConfig,
    mode: ModeName,
    state_dim: int,
    action_dim: int | None,
    action_spec: dict[str, Any] | None,
) -> NeuralFQEModel:
    gamma = _validate_gamma(gamma)
    n = rewards.shape[0]
    bootstrap = validate_bootstrap_inputs(n_rows=n, terminals=terminals, timeouts=timeouts, continuation=continuation)
    continuation_1d = bootstrap.continuation
    next_action_weights_2d = validate_action_weights(
        next_action_weights,
        n_rows=n,
        n_actions=next_features.shape[1],
        name="next_action_weights",
    )
    weights_1d = _optional_weights(sample_weight, n, "sample_weight")
    _validate_features(features, next_features, rewards)
    _require_torch()
    _seed_torch(config.seed)

    train_idx, val_idx = _train_validation_indices(n, config.validation_fraction, config.seed)
    target_min, target_max = _resolve_target_bounds(rewards, gamma, config)
    input_mean, input_std = _normalizer(features[train_idx], config.standardize_inputs)
    device = torch.device(config.device)

    x_all = _features_to_tensor(features, input_mean, input_std, device)
    x_next_all = _next_features_to_tensor(next_features, input_mean, input_std, device)
    rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=device)
    continuation_t = torch.as_tensor(continuation_1d, dtype=torch.float32, device=device)
    next_action_weights_t = torch.as_tensor(next_action_weights_2d, dtype=torch.float32, device=device)
    weights_t = torch.as_tensor(weights_1d, dtype=torch.float32, device=device)
    train_idx_t = torch.as_tensor(train_idx, dtype=torch.long, device=device)
    val_idx_t = torch.as_tensor(val_idx, dtype=torch.long, device=device)

    network = _MLP(features.shape[1], tuple(config.hidden_dims), config.activation).to(device)
    target_network = deepcopy(network).to(device)
    optimizer = torch.optim.AdamW(network.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    batch_size = min(int(config.batch_size), int(train_idx.shape[0]))
    best_risk = np.inf
    best_state = deepcopy(network.state_dict())
    best_target_state = deepcopy(target_network.state_dict())
    patience = 0
    stopped_early = False
    stop_reason = ""
    history: list[dict[str, Any]] = []
    rng = np.random.default_rng(config.seed + 42_001)

    iterator = range(config.num_iterations)
    if config.show_progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(iterator, desc="Neural FQE")
        except ModuleNotFoundError:
            pass

    for iteration in iterator:
        network.train()
        with torch.no_grad():
            next_pred_all = _predict_next_tensor(target_network, x_next_all, next_action_weights_t)
            targets_all = rewards_t + float(gamma) * continuation_t * next_pred_all
            targets_all = _clip_targets_tensor(targets_all, target_min, target_max)
        last_loss = float("nan")
        for _ in range(config.gradient_steps_per_iteration):
            local = rng.integers(0, train_idx.shape[0], size=batch_size)
            batch_idx = train_idx_t[torch.as_tensor(local, dtype=torch.long, device=device)]
            pred = network(x_all[batch_idx])
            loss = _weighted_regression_loss(
                pred=pred,
                target=targets_all[batch_idx],
                weight=weights_t[batch_idx],
                config=config,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if config.grad_clip_norm is not None:
                nn.utils.clip_grad_norm_(network.parameters(), float(config.grad_clip_norm))
            optimizer.step()
            last_loss = float(loss.detach().cpu().item())
        _polyak_update(target_network, network, float(config.target_update_tau))
        network.eval()
        target_network.eval()
        with torch.no_grad():
            next_pred_eval = _predict_next_tensor(target_network, x_next_all, next_action_weights_t)
            pred_all = network(x_all)
            train_risk = _bellman_risk(
                predictions=pred_all[train_idx_t].detach().cpu().numpy().astype(np.float64),
                next_predictions=next_pred_eval[train_idx_t].detach().cpu().numpy().astype(np.float64),
                rewards=rewards[train_idx],
                gamma=gamma,
                terminals=1.0 - continuation_1d[train_idx],
                sample_weight=weights_1d[train_idx],
            )
            val_risk = _bellman_risk(
                predictions=pred_all[val_idx_t].detach().cpu().numpy().astype(np.float64),
                next_predictions=next_pred_eval[val_idx_t].detach().cpu().numpy().astype(np.float64),
                rewards=rewards[val_idx],
                gamma=gamma,
                terminals=1.0 - continuation_1d[val_idx],
                sample_weight=weights_1d[val_idx],
            )
        improved = (not config.early_stopping) or val_risk <= best_risk - float(config.min_improvement) or not np.isfinite(best_risk)
        if improved:
            best_risk = min(best_risk, val_risk)
            best_state = deepcopy(network.state_dict())
            best_target_state = deepcopy(target_network.state_dict())
            patience = 0
        else:
            patience += 1
        history.append(
            {
                "iteration": int(iteration),
                "accepted": bool(improved),
                "train_bellman_risk": float(train_risk),
                "validation_bellman_risk": float(val_risk),
                "best_validation_bellman_risk": float(best_risk),
                "train_loss_last": float(last_loss),
            }
        )
        if config.early_stopping and config.patience > 0 and patience >= config.patience:
            stopped_early = True
            stop_reason = "patience"
            break

    network.load_state_dict(best_state)
    target_network.load_state_dict(best_target_state)
    network.eval()
    target_network.eval()
    with torch.no_grad():
        pred_final = network(x_all).detach().cpu().numpy().astype(np.float64)
        next_final = _predict_next_tensor(target_network, x_next_all, next_action_weights_t).detach().cpu().numpy().astype(np.float64)
    final_self_risk = _bellman_risk(
        predictions=pred_final,
        next_predictions=next_final,
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
        "actual_iterations": int(len(history)),
        "accepted_iterations": int(sum(1 for row in history if row["accepted"])),
        "stopped_early": bool(stopped_early),
        "stop_reason": stop_reason,
        "best_validation_bellman_risk": float(best_risk),
        "final_train_bellman_risk": float(history[-1]["train_bellman_risk"]) if history else np.nan,
        "final_validation_bellman_risk": float(history[-1]["validation_bellman_risk"]) if history else np.nan,
        "final_self_bellman_risk": float(final_self_risk),
        "target_action_expectation": "weighted" if not _is_uniform_action_weights(next_action_weights_2d) else "uniform",
        "target_min": target_min,
        "target_max": target_max,
        "n_samples": int(n),
        "n_train": int(train_idx.shape[0]),
        "n_validation": int(val_idx.shape[0]),
        "device": str(device),
        "standardize_inputs": bool(config.standardize_inputs),
    }
    diagnostics.update(bootstrap.diagnostics)
    if action_spec is not None:
        diagnostics["action_spec_type"] = str(action_spec.get("type", "continuous"))
    return NeuralFQEModel(
        network=network,
        target_network=target_network,
        mode=mode,
        gamma=float(gamma),
        state_dim=int(state_dim),
        action_dim=None if action_dim is None else int(action_dim),
        config=config,
        input_mean=input_mean,
        input_std=input_std,
        history=history,
        diagnostics=diagnostics,
        train_indices=train_idx,
        validation_indices=val_idx,
        action_spec=action_spec,
    )


def _require_torch() -> None:
    if torch is None or nn is None:
        raise ModuleNotFoundError(
            "PyTorch is required to fit neural fqe models. Install the neural extra "
            "or run `pip install torch`."
        )


def _seed_torch(seed: int) -> None:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


class _MLP(nn.Module if nn is not None else object):
    def __init__(self, input_dim: int, hidden_dims: tuple[int, ...], activation: str) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        last = int(input_dim)
        for width in hidden_dims:
            layers.append(nn.Linear(last, int(width)))
            layers.append(_activation(activation))
            last = int(width)
        layers.append(nn.Linear(last, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _activation(name: str):
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "silu":
        return nn.SiLU()
    if name == "gelu":
        return nn.GELU()
    raise ValueError("Unknown activation.")


def _normalizer(features: Array, standardize: bool) -> tuple[Array, Array]:
    if not standardize:
        return np.zeros(features.shape[1], dtype=np.float64), np.ones(features.shape[1], dtype=np.float64)
    mean = np.mean(features, axis=0).astype(np.float64)
    std = np.std(features, axis=0).astype(np.float64)
    std = np.where(std > 1e-8, std, 1.0)
    return mean, std


def _features_to_tensor(features: Array, mean: Array, std: Array, device: Any):
    arr = (np.asarray(features, dtype=np.float32) - mean.astype(np.float32)) / std.astype(np.float32)
    return torch.as_tensor(arr, dtype=torch.float32, device=device)


def _next_features_to_tensor(next_features: Array, mean: Array, std: Array, device: Any):
    n, m, d = next_features.shape
    flat = next_features.reshape(n * m, d)
    return _features_to_tensor(flat, mean, std, device).reshape(n, m, d)


def _predict_next_tensor(model: Any, x_next: Any, action_weights: Any | None = None):
    n, m, d = x_next.shape
    pred = model(x_next.reshape(n * m, d)).reshape(n, m)
    if action_weights is None:
        return pred.mean(dim=1)
    return torch.sum(pred * action_weights, dim=1)


def _predict_next_average_neural(
    model: Any,
    next_features: Array,
    mean: Array,
    std: Array,
    device: str,
    next_action_weights: Array | None = None,
) -> Array:
    _require_torch()
    model.eval()
    with torch.no_grad():
        device_obj = torch.device(device)
        x_next = _next_features_to_tensor(next_features, mean, std, device_obj)
        weights_t = None
        if next_action_weights is not None:
            weights_t = torch.as_tensor(next_action_weights, dtype=torch.float32, device=device_obj)
        pred = _predict_next_tensor(model, x_next, weights_t).detach().cpu().numpy().astype(np.float64)
    return pred


def _clip_targets_tensor(targets: Any, target_min: float | None, target_max: float | None):
    if target_min is None and target_max is None:
        return targets
    return torch.clamp(
        targets,
        min=-float("inf") if target_min is None else float(target_min),
        max=float("inf") if target_max is None else float(target_max),
    )


def _weighted_regression_loss(pred: Any, target: Any, weight: Any, config: NeuralFQEConfig):
    if config.loss == "squared":
        per_row = (pred - target) ** 2
    else:
        delta = _resolve_huber_delta_tensor((pred - target).detach(), config)
        per_row = torch.nn.functional.huber_loss(pred, target, reduction="none", delta=delta)
    denom = torch.clamp(torch.sum(weight), min=1e-12)
    return torch.sum(per_row * weight) / denom


def _resolve_huber_delta_tensor(residual: Any, config: NeuralFQEConfig) -> float:
    if config.huber_delta is not None:
        return float(config.huber_delta)
    arr = residual.detach().cpu().numpy().astype(np.float64).reshape(-1)
    if arr.size == 0:
        return 1.0
    mad = float(np.median(np.abs(arr - np.median(arr))))
    robust_sigma = max(1.4826 * mad, float(np.std(arr)), 1e-8)
    return float(config.huber_delta_scale * robust_sigma)


def _polyak_update(target: Any, online: Any, tau: float) -> None:
    with torch.no_grad():
        for target_param, online_param in zip(target.parameters(), online.parameters()):
            target_param.data.mul_(1.0 - tau).add_(tau * online_param.data)
