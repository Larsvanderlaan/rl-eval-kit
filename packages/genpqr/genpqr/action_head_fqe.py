"""Action-head neural FQE backend for finite-action GenPQR fits."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

import numpy as np

from genpqr.exceptions import GenPQRConfigurationError, GenPQRMissingDependencyError
from genpqr.types import ActionSpaceSpec, Array, NormalizationPolicy
from genpqr.validation import as_1d_float, as_2d_float, optional_terminals, optional_weights


ActivationName = Literal["relu", "tanh", "silu", "gelu"]
LossName = Literal["squared", "huber"]


@dataclass(frozen=True)
class ActionHeadNeuralFQEConfig:
    """Configuration for discrete action-head neural FQE.

    The network has a pooled state trunk, a state-only baseline head, and
    action-specific residual heads. Penalizing residual heads interpolates
    between a fully pooled state-value model and fully stratified action heads.
    Balanced mini-batches keep sparse actions from being washed out by the
    dominant behavior action mix.

    Parameters
    ----------
    hidden_dims:
        Widths for the shared state trunk.
    head_hidden_dims:
        Widths for the residual head network before the action outputs.
    residual_l2:
        L2 penalty on centered action residuals. Larger values shrink toward a
        pooled state-only model; smaller values allow fully stratified actions.
    policy_log_prob_skip:
        Whether to add centered target-policy log probabilities as a fixed
        action skip connection. This is useful for GenPQR because the FQE
        pseudo-reward is itself a policy log probability.
    balanced_batches:
        Whether to sample actions approximately uniformly within each gradient
        batch when all actions have support.
    """

    hidden_dims: Sequence[int] = (256, 256, 128)
    head_hidden_dims: Sequence[int] = (128,)
    activation: ActivationName = "silu"
    learning_rate: float = 6e-4
    weight_decay: float = 1e-5
    batch_size: int = 256
    num_iterations: int = 180
    gradient_steps_per_iteration: int = 40
    target_update_tau: float = 0.30
    loss: LossName = "squared"
    huber_delta: float = 1.0
    validation_fraction: float = 0.15
    early_stopping: bool = True
    patience: int = 30
    min_improvement: float = 1e-6
    grad_clip_norm: float | None = 10.0
    residual_l2: float = 1e-3
    policy_log_prob_skip: bool = True
    policy_skip_scale: float = 1.0
    balanced_batches: bool = True
    standardize_states: bool = True
    device: str = "cpu"
    seed: int = 123

    def __post_init__(self) -> None:
        if not tuple(self.hidden_dims) or any(int(width) <= 0 for width in self.hidden_dims):
            raise ValueError("hidden_dims must contain positive widths.")
        if any(int(width) <= 0 for width in self.head_hidden_dims):
            raise ValueError("head_hidden_dims must contain positive widths.")
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
        if self.huber_delta <= 0.0:
            raise ValueError("huber_delta must be positive.")
        if not (0.0 < self.validation_fraction < 1.0):
            raise ValueError("validation_fraction must be in (0, 1).")
        if self.patience < 0:
            raise ValueError("patience must be nonnegative.")
        if self.min_improvement < 0.0:
            raise ValueError("min_improvement must be nonnegative.")
        if self.grad_clip_norm is not None and self.grad_clip_norm <= 0.0:
            raise ValueError("grad_clip_norm must be positive when supplied.")
        if self.residual_l2 < 0.0:
            raise ValueError("residual_l2 must be nonnegative.")
        if not np.isfinite(float(self.policy_skip_scale)):
            raise ValueError("policy_skip_scale must be finite.")

    @classmethod
    def stable_defaults(cls, **overrides: Any) -> "ActionHeadNeuralFQEConfig":
        """Construct balanced action-head defaults."""

        return cls(**dict(overrides))

    @classmethod
    def fast_defaults(cls, **overrides: Any) -> "ActionHeadNeuralFQEConfig":
        """Construct a faster smoke-test configuration."""

        params: dict[str, Any] = {
            "hidden_dims": (128, 128),
            "head_hidden_dims": (64,),
            "batch_size": 128,
            "num_iterations": 80,
            "gradient_steps_per_iteration": 24,
            "patience": 12,
            "target_update_tau": 0.35,
            "learning_rate": 8e-4,
            "residual_l2": 1e-3,
        }
        params.update(overrides)
        return cls(**params)


@dataclass
class ActionHeadNeuralFQEstimator:
    """Discrete action-head neural FQE adapter.

    The fitted model estimates all action values from a shared state trunk and
    action-specific residual heads. It is intended for finite-action pooled-FQE
    GenPQR fits where a generic ``[state, one_hot(action)]`` MLP may over-share
    across actions.
    """

    config: ActionHeadNeuralFQEConfig | None = None
    config_overrides: dict[str, Any] = field(default_factory=dict)
    n_next_action_samples: int = 8

    def preflight(self, **_: Any) -> None:
        """Validate adapter configuration before fitting."""

        if self.config is not None and self.config_overrides:
            raise GenPQRConfigurationError("Pass either config or config_overrides, not both.")

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
        policy: Any | None = None,
    ) -> "ActionHeadNeuralFQEFunction":
        """Fit a discrete action-head neural FQE model."""

        self.preflight()
        action_space = normalization_policy.action_space
        if action_space.kind != "discrete":
            raise GenPQRConfigurationError("ActionHeadNeuralFQEstimator requires a discrete action space.")
        cfg = self.config or ActionHeadNeuralFQEConfig.stable_defaults(**self.config_overrides)
        states_2d = as_2d_float(states, "states")
        next_states_2d = as_2d_float(next_states, "next_states", n_rows=states_2d.shape[0])
        if next_states_2d.shape[1] != states_2d.shape[1]:
            raise ValueError("next_states must have the same number of columns as states.")
        action_idx = action_space.action_indices(actions, n_rows=states_2d.shape[0])
        rewards = as_1d_float(pseudo_rewards, "pseudo_rewards", n_rows=states_2d.shape[0])
        terminals_1d = optional_terminals(terminals, states_2d.shape[0])
        weights = (
            np.ones(states_2d.shape[0], dtype=np.float64)
            if sample_weight is None
            else optional_weights(sample_weight, states_2d.shape[0])
        )
        next_probs = _next_action_probabilities(
            normalization_policy=normalization_policy,
            states=next_states_2d,
            n_actions=int(action_space.n_actions),
            n_samples=int(self.n_next_action_samples),
            seed=int(cfg.seed),
        )
        policy_log_probs = _policy_log_prob_matrix(
            policy=policy,
            states=states_2d,
            action_space=action_space,
            enabled=bool(cfg.policy_log_prob_skip),
        )
        next_policy_log_probs = _policy_log_prob_matrix(
            policy=policy,
            states=next_states_2d,
            action_space=action_space,
            enabled=bool(cfg.policy_log_prob_skip),
        )
        return _fit_action_head_fqe(
            states=states_2d,
            actions=action_idx,
            next_states=next_states_2d,
            next_probs=next_probs,
            policy_log_probs=policy_log_probs,
            next_policy_log_probs=next_policy_log_probs,
            policy=policy,
            rewards=rewards,
            gamma=float(gamma),
            terminals=terminals_1d,
            sample_weight=weights,
            action_space=action_space,
            config=cfg,
        )


@dataclass
class ActionHeadNeuralFQEFunction:
    """Fitted discrete action-head neural FQE function."""

    network: Any
    target_network: Any
    action_space: ActionSpaceSpec
    input_mean: Array
    input_std: Array
    config: ActionHeadNeuralFQEConfig
    diagnostics: dict[str, Any]
    policy: Any | None = None

    def __getstate__(self) -> dict[str, Any]:
        """Return a pickle-safe representation without nested Torch classes."""

        return {
            "network_state": _state_dict_to_numpy(self.network),
            "target_network_state": _state_dict_to_numpy(self.target_network),
            "action_space": self.action_space,
            "input_mean": np.asarray(self.input_mean, dtype=np.float64),
            "input_std": np.asarray(self.input_std, dtype=np.float64),
            "config": self.config,
            "diagnostics": dict(self.diagnostics),
            "policy": self.policy,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore a fitted action-head FQE function from pickle state."""

        config = state["config"]
        if not isinstance(config, ActionHeadNeuralFQEConfig):
            config = ActionHeadNeuralFQEConfig(**dict(config))
        network = _network_from_numpy_state(
            state["network_state"],
            action_space=state["action_space"],
            state_dim=int(np.asarray(state["input_mean"]).shape[0]),
            config=config,
        )
        target_state = state.get("target_network_state") or state["network_state"]
        target_network = _network_from_numpy_state(
            target_state,
            action_space=state["action_space"],
            state_dim=int(np.asarray(state["input_mean"]).shape[0]),
            config=config,
        )
        self.network = network
        self.target_network = target_network
        self.action_space = state["action_space"]
        self.input_mean = np.asarray(state["input_mean"], dtype=np.float64)
        self.input_std = np.asarray(state["input_std"], dtype=np.float64)
        self.config = config
        self.diagnostics = dict(state.get("diagnostics", {}))
        self.policy = state.get("policy")

    def predict_q(self, states: Array, actions: Array) -> Array:
        """Predict Q-values for state-action rows."""

        states_2d = as_2d_float(states, "states")
        idx = self.action_space.action_indices(actions, n_rows=states_2d.shape[0])
        q_matrix = self.predict_q_matrix(states_2d)
        return q_matrix[np.arange(states_2d.shape[0]), idx]

    def predict_q_matrix(self, states: Array) -> Array:
        """Predict all finite-action Q-values."""

        torch, _ = _require_torch()
        states_2d = as_2d_float(states, "states")
        self.network.eval()
        with torch.no_grad():
            x = _states_to_tensor(states_2d, self.input_mean, self.input_std, self.config.device)
            base_q = self.network(x)[0]
            policy_skip = _policy_log_prob_matrix(
                policy=self.policy,
                states=states_2d,
                action_space=self.action_space,
                enabled=bool(self.config.policy_log_prob_skip),
            )
            q = (
                base_q.detach().cpu().numpy().astype(np.float64)
                + float(self.config.policy_skip_scale) * policy_skip
            )
        if q.shape != (states_2d.shape[0], int(self.action_space.n_actions)) or not np.all(np.isfinite(q)):
            raise FloatingPointError("action-head FQE returned invalid Q predictions.")
        return q

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
        q_matrix = self.predict_q_matrix(states_2d)
        if hasattr(normalization_policy, "predict_proba"):
            probs = normalization_policy.predict_proba(states_2d)  # type: ignore[attr-defined]
            return np.sum(np.asarray(probs, dtype=np.float64) * q_matrix, axis=1)
        samples = normalization_policy.sample(states_2d, rng, int(n_action_samples))
        idx = self.action_space.encode_samples(samples, n_rows=states_2d.shape[0], name="normalization samples")
        if idx.ndim == 2:
            action_idx = self.action_space.action_indices(idx, n_rows=states_2d.shape[0])
            return q_matrix[np.arange(states_2d.shape[0]), action_idx]
        values = []
        for j in range(idx.shape[1]):
            action_idx = self.action_space.action_indices(idx[:, j, :], n_rows=states_2d.shape[0])
            values.append(q_matrix[np.arange(states_2d.shape[0]), action_idx])
        return np.mean(np.stack(values, axis=1), axis=1)


def _fit_action_head_fqe(
    *,
    states: Array,
    actions: Array,
    next_states: Array,
    next_probs: Array,
    policy_log_probs: Array,
    next_policy_log_probs: Array,
    policy: Any | None,
    rewards: Array,
    gamma: float,
    terminals: Array,
    sample_weight: Array,
    action_space: ActionSpaceSpec,
    config: ActionHeadNeuralFQEConfig,
) -> ActionHeadNeuralFQEFunction:
    if not (0.0 <= float(gamma) < 1.0):
        raise ValueError("gamma must be in [0, 1).")
    torch, nn = _require_torch()
    _seed_torch(torch, int(config.seed))
    if config.device == "cpu":
        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass

    n_rows = states.shape[0]
    train_idx, val_idx = _train_validation_indices(n_rows, float(config.validation_fraction), int(config.seed))
    mean, std = _state_normalizer(states[train_idx], bool(config.standardize_states))
    device = torch.device(config.device)
    x = _states_to_tensor(states, mean, std, str(device))
    x_next = _states_to_tensor(next_states, mean, std, str(device))
    actions_t = torch.as_tensor(actions, dtype=torch.long, device=device)
    next_probs_t = torch.as_tensor(next_probs, dtype=torch.float32, device=device)
    policy_log_probs_t = torch.as_tensor(policy_log_probs, dtype=torch.float32, device=device)
    next_policy_log_probs_t = torch.as_tensor(next_policy_log_probs, dtype=torch.float32, device=device)
    rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=device)
    terminals_t = torch.as_tensor(terminals, dtype=torch.float32, device=device)
    weights_t = torch.as_tensor(
        sample_weight / max(float(np.mean(sample_weight)), 1e-12),
        dtype=torch.float32,
        device=device,
    )
    train_idx_t = torch.as_tensor(train_idx, dtype=torch.long, device=device)
    val_idx_t = torch.as_tensor(val_idx, dtype=torch.long, device=device)
    network = _ActionHeadNetwork(
        nn=nn,
        state_dim=states.shape[1],
        n_actions=int(action_space.n_actions),
        hidden_dims=tuple(config.hidden_dims),
        head_hidden_dims=tuple(config.head_hidden_dims),
        activation=config.activation,
    ).to(device)
    target_network = deepcopy(network).to(device)
    optimizer = torch.optim.AdamW(
        network.parameters(),
        lr=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
    )
    action_rows = _action_row_index(train_idx, actions, int(action_space.n_actions))
    batch_size = min(int(config.batch_size), int(train_idx.shape[0]))
    rng = np.random.default_rng(int(config.seed) + 88_001)
    best_risk = np.inf
    best_state = deepcopy(network.state_dict())
    best_target_state = deepcopy(target_network.state_dict())
    stale = 0
    stopped_early = False
    stop_reason = ""
    history: list[dict[str, Any]] = []

    for iteration in range(int(config.num_iterations)):
        network.train()
        target_network.eval()
        with torch.no_grad():
            next_q, _ = target_network(x_next)
            next_q = next_q + float(config.policy_skip_scale) * next_policy_log_probs_t
            next_v = torch.sum(next_probs_t * next_q, dim=1)
            targets = rewards_t + float(gamma) * (1.0 - terminals_t) * next_v
        last_loss = float("nan")
        for _ in range(int(config.gradient_steps_per_iteration)):
            if config.balanced_batches:
                batch_np = _balanced_batch(action_rows=action_rows, batch_size=batch_size, rng=rng)
            else:
                batch_np = rng.choice(train_idx, size=batch_size, replace=True)
            batch = torch.as_tensor(batch_np, dtype=torch.long, device=device)
            q_all, residuals = network(x[batch])
            q_all = q_all + float(config.policy_skip_scale) * policy_log_probs_t[batch]
            pred = q_all.gather(1, actions_t[batch].reshape(-1, 1)).reshape(-1)
            fit_loss = _loss(pred, targets[batch], weights_t[batch], config, torch)
            reg = float(config.residual_l2) * torch.mean(residuals**2)
            loss = fit_loss + reg
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if config.grad_clip_norm is not None:
                nn.utils.clip_grad_norm_(network.parameters(), float(config.grad_clip_norm))
            optimizer.step()
            last_loss = float(loss.detach().cpu().item())
        _polyak_update(target_network, network, float(config.target_update_tau), torch)
        train_risk = _bellman_risk_torch(
            network=network,
            target_network=target_network,
            x=x,
            x_next=x_next,
            actions=actions_t,
            next_probs=next_probs_t,
            rewards=rewards_t,
            terminals=terminals_t,
            weights=weights_t,
            idx=train_idx_t,
            gamma=float(gamma),
            config=config,
            policy_log_probs=policy_log_probs_t,
            next_policy_log_probs=next_policy_log_probs_t,
        )
        val_risk = _bellman_risk_torch(
            network=network,
            target_network=target_network,
            x=x,
            x_next=x_next,
            actions=actions_t,
            next_probs=next_probs_t,
            rewards=rewards_t,
            terminals=terminals_t,
            weights=weights_t,
            idx=val_idx_t,
            gamma=float(gamma),
            config=config,
            policy_log_probs=policy_log_probs_t,
            next_policy_log_probs=next_policy_log_probs_t,
        )
        improved = (not config.early_stopping) or val_risk <= best_risk - float(config.min_improvement)
        if improved:
            best_risk = min(best_risk, val_risk)
            best_state = deepcopy(network.state_dict())
            best_target_state = deepcopy(target_network.state_dict())
            stale = 0
        else:
            stale += 1
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
        if config.early_stopping and config.patience > 0 and stale >= int(config.patience):
            stopped_early = True
            stop_reason = "patience"
            break

    network.load_state_dict(best_state)
    target_network.load_state_dict(best_target_state)
    network.eval()
    target_network.eval()
    counts = np.bincount(actions, minlength=int(action_space.n_actions))
    diagnostics = {
        "backend": "action_head_neural_fqe",
        "mode": "q",
        "gamma": float(gamma),
        "n_samples": int(n_rows),
        "n_train": int(train_idx.shape[0]),
        "n_validation": int(val_idx.shape[0]),
        "action_counts": counts.astype(int).tolist(),
        "min_action_count": int(np.min(counts)),
        "balanced_batches": bool(config.balanced_batches),
        "residual_l2": float(config.residual_l2),
        "policy_log_prob_skip": bool(config.policy_log_prob_skip),
        "policy_skip_scale": float(config.policy_skip_scale),
        "num_iterations_requested": int(config.num_iterations),
        "actual_iterations": int(len(history)),
        "accepted_iterations": int(sum(1 for row in history if row["accepted"])),
        "stopped_early": bool(stopped_early),
        "stop_reason": stop_reason,
        "best_validation_bellman_risk": float(best_risk),
        "final_train_bellman_risk": float(history[-1]["train_bellman_risk"]) if history else np.nan,
        "final_validation_bellman_risk": float(history[-1]["validation_bellman_risk"]) if history else np.nan,
        "device": str(device),
    }
    return ActionHeadNeuralFQEFunction(
        network=network,
        target_network=target_network,
        action_space=action_space,
        input_mean=mean,
        input_std=std,
        config=config,
        diagnostics=diagnostics,
        policy=policy,
    )


class _ActionHeadNetwork:
    def __new__(cls, *, nn: Any, **kwargs: Any) -> Any:
        class Network(nn.Module):  # type: ignore[misc, valid-type]
            def __init__(self, **inner_kwargs: Any) -> None:
                super().__init__()
                state_dim = int(inner_kwargs["state_dim"])
                n_actions = int(inner_kwargs["n_actions"])
                hidden_dims = tuple(inner_kwargs["hidden_dims"])
                head_hidden_dims = tuple(inner_kwargs["head_hidden_dims"])
                activation = str(inner_kwargs["activation"])
                self.trunk = _mlp(nn, state_dim, hidden_dims, hidden_dims[-1], activation)
                self.base_head = nn.Linear(hidden_dims[-1], 1)
                self.residual_head = _mlp(nn, hidden_dims[-1], head_hidden_dims, n_actions, activation)

            def forward(self, states_t: Any) -> tuple[Any, Any]:
                features = self.trunk(states_t)
                base = self.base_head(features)
                residuals = self.residual_head(features)
                centered = residuals - residuals.mean(dim=1, keepdim=True)
                return base + centered, centered

        return Network(**kwargs)


def _mlp(nn: Any, input_dim: int, hidden_dims: tuple[int, ...], output_dim: int, activation: str) -> Any:
    layers: list[Any] = []
    last = int(input_dim)
    for width in hidden_dims:
        layers.append(nn.Linear(last, int(width)))
        layers.append(_activation(nn, activation))
        last = int(width)
    layers.append(nn.Linear(last, int(output_dim)))
    return nn.Sequential(*layers)


def _activation(nn: Any, name: str) -> Any:
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "silu":
        return nn.SiLU()
    if name == "gelu":
        return nn.GELU()
    raise ValueError("Unknown activation.")


def _next_action_probabilities(
    *,
    normalization_policy: NormalizationPolicy,
    states: Array,
    n_actions: int,
    n_samples: int,
    seed: int,
) -> Array:
    states_2d = as_2d_float(states, "states")
    if hasattr(normalization_policy, "predict_proba"):
        probs = np.asarray(
            normalization_policy.predict_proba(states_2d),  # type: ignore[attr-defined]
            dtype=np.float64,
        )
        if probs.shape != (states_2d.shape[0], int(n_actions)):
            raise ValueError("normalization_policy probabilities have the wrong shape.")
        return probs / probs.sum(axis=1, keepdims=True)
    rng = np.random.default_rng(int(seed))
    samples = normalization_policy.sample(states_2d, rng, int(n_samples))
    action_space = ActionSpaceSpec.discrete(int(n_actions))
    encoded = action_space.encode_samples(samples, n_rows=states_2d.shape[0], name="next normalization samples")
    if encoded.ndim == 2:
        return encoded
    return np.mean(encoded, axis=1)


def _policy_log_prob_matrix(
    *,
    policy: Any | None,
    states: Array,
    action_space: ActionSpaceSpec,
    enabled: bool,
) -> Array:
    states_2d = as_2d_float(states, "states")
    n_actions = int(action_space.n_actions)
    if not enabled or policy is None or not hasattr(policy, "log_prob"):
        return np.zeros((states_2d.shape[0], n_actions), dtype=np.float64)
    cols = []
    for action in range(n_actions):
        action_vec = np.full(states_2d.shape[0], action, dtype=np.int64)
        cols.append(policy.log_prob(states_2d, action_vec))
    log_probs = np.stack(cols, axis=1).astype(np.float64)
    if log_probs.shape != (states_2d.shape[0], n_actions) or not np.all(np.isfinite(log_probs)):
        raise FloatingPointError("policy returned invalid log probabilities for action-head FQE.")
    return log_probs - np.mean(log_probs, axis=1, keepdims=True)


def _train_validation_indices(n_rows: int, validation_fraction: float, seed: int) -> tuple[Array, Array]:
    if int(n_rows) < 2:
        raise ValueError("at least two rows are required.")
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(int(n_rows))
    n_val = min(max(1, int(round(float(validation_fraction) * int(n_rows)))), int(n_rows) - 1)
    return np.sort(order[n_val:]), np.sort(order[:n_val])


def _state_normalizer(states: Array, standardize: bool) -> tuple[Array, Array]:
    if not standardize:
        return np.zeros(states.shape[1], dtype=np.float64), np.ones(states.shape[1], dtype=np.float64)
    mean = np.mean(states, axis=0).astype(np.float64)
    std = np.std(states, axis=0).astype(np.float64)
    std = np.where(std > 1e-8, std, 1.0)
    return mean, std


def _states_to_tensor(states: Array, mean: Array, std: Array, device: str) -> Any:
    torch, _ = _require_torch()
    arr = (np.asarray(states, dtype=np.float32) - mean.astype(np.float32)) / std.astype(np.float32)
    return torch.as_tensor(arr, dtype=torch.float32, device=torch.device(device))


def _action_row_index(train_idx: Array, actions: Array, n_actions: int) -> list[Array]:
    return [np.asarray(train_idx[actions[train_idx] == action], dtype=np.int64) for action in range(int(n_actions))]


def _balanced_batch(*, action_rows: list[Array], batch_size: int, rng: np.random.Generator) -> Array:
    nonempty = [rows for rows in action_rows if rows.shape[0] > 0]
    if not nonempty:
        raise ValueError("balanced batch requires at least one nonempty action.")
    per_action = max(1, int(np.ceil(int(batch_size) / len(nonempty))))
    pieces = [rng.choice(rows, size=per_action, replace=rows.shape[0] < per_action) for rows in nonempty]
    batch = np.concatenate(pieces)
    if batch.shape[0] > int(batch_size):
        batch = rng.choice(batch, size=int(batch_size), replace=False)
    rng.shuffle(batch)
    return batch.astype(np.int64, copy=False)


def _loss(pred: Any, target: Any, weight: Any, config: ActionHeadNeuralFQEConfig, torch: Any) -> Any:
    if config.loss == "squared":
        per_row = (pred - target) ** 2
    else:
        per_row = torch.nn.functional.huber_loss(
            pred,
            target,
            reduction="none",
            delta=float(config.huber_delta),
        )
    return torch.sum(per_row * weight) / torch.clamp(torch.sum(weight), min=1e-12)


def _bellman_risk_torch(
    *,
    network: Any,
    target_network: Any,
    x: Any,
    x_next: Any,
    actions: Any,
    next_probs: Any,
    rewards: Any,
    terminals: Any,
    weights: Any,
    idx: Any,
    gamma: float,
    config: ActionHeadNeuralFQEConfig,
    policy_log_probs: Any,
    next_policy_log_probs: Any,
) -> float:
    torch, _ = _require_torch()
    network.eval()
    target_network.eval()
    with torch.no_grad():
        q_all, _ = network(x[idx])
        q_all = q_all + float(config.policy_skip_scale) * policy_log_probs[idx]
        pred = q_all.gather(1, actions[idx].reshape(-1, 1)).reshape(-1)
        next_q, _ = target_network(x_next[idx])
        next_q = next_q + float(config.policy_skip_scale) * next_policy_log_probs[idx]
        next_v = torch.sum(next_probs[idx] * next_q, dim=1)
        target = rewards[idx] + float(gamma) * (1.0 - terminals[idx]) * next_v
        resid = (pred - target) ** 2
        risk = torch.sum(resid * weights[idx]) / torch.clamp(torch.sum(weights[idx]), min=1e-12)
    return float(risk.detach().cpu().item())


def _polyak_update(target: Any, online: Any, tau: float, torch: Any) -> None:
    with torch.no_grad():
        for target_param, online_param in zip(target.parameters(), online.parameters()):
            target_param.data.mul_(1.0 - float(tau)).add_(float(tau) * online_param.data)


def _state_dict_to_numpy(network: Any) -> list[tuple[str, Array]]:
    return [
        (str(key), value.detach().cpu().numpy())
        for key, value in network.state_dict().items()
    ]


def _network_from_numpy_state(
    state_items: list[tuple[str, Array]],
    *,
    action_space: ActionSpaceSpec,
    state_dim: int,
    config: ActionHeadNeuralFQEConfig,
) -> Any:
    torch, nn = _require_torch()
    device = torch.device(config.device)
    network = _ActionHeadNetwork(
        nn=nn,
        state_dim=int(state_dim),
        n_actions=int(action_space.n_actions),
        hidden_dims=tuple(config.hidden_dims),
        head_hidden_dims=tuple(config.head_hidden_dims),
        activation=config.activation,
    ).to(device)
    state_dict = {
        key: torch.as_tensor(value, device=device)
        for key, value in state_items
    }
    network.load_state_dict(state_dict)
    network.eval()
    return network


def _seed_torch(torch: Any, seed: int) -> None:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _require_torch() -> tuple[Any, Any]:
    try:
        import torch
        from torch import nn
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency.
        raise GenPQRMissingDependencyError("Install genpqr[torch] to use action-head neural FQE.") from exc
    return torch, nn


__all__ = [
    "ActionHeadNeuralFQEConfig",
    "ActionHeadNeuralFQEFunction",
    "ActionHeadNeuralFQEstimator",
]
