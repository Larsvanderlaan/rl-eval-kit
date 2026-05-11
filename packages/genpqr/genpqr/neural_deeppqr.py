"""Lazy Torch DeepPQR anchor-Q backend."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from genpqr.deeppqr import _anchor_probabilities, _normalization_log_ratio_shift
from genpqr.exceptions import GenPQRConfigurationError, GenPQRMissingDependencyError
from genpqr.types import ActionSpaceSpec, Array, EstimatedPolicy, NormalizationPolicy
from genpqr.validation import as_1d_float, as_2d_float, optional_terminals, optional_weights


@dataclass
class NeuralDeepPQRAnchorQEstimator:
    """Torch neural DeepPQR state-only anchor-Q estimator.

    The backend lazily imports Torch, fits ``W(s)=Q(s,a_anchor)`` on anchor rows
    only, and reconstructs action-stratified Q-values with policy log-ratios.
    """

    anchor_action: int | float | Array | Callable[[Array], Array] = 0
    anchor_selector: Callable[[Array, Array], Array] | None = None
    anchor_tolerance: float = 1e-8
    hidden_dims: tuple[int, ...] = (128, 128)
    lr: float = 1e-3
    batch_size: int = 256
    max_epochs: int = 500
    validation_fraction: float = 0.2
    patience: int = 20
    weight_decay: float = 1e-4
    seed: int = 123
    n_action_samples: int = 16
    weak_anchor_fraction: float = 0.05
    device: str = "cpu"
    target_update_interval: int = 5
    target_tau: float = 1.0
    gradient_clip_norm: float | None = 10.0
    min_anchor_count: int = 5
    validation_metric: str = "bellman_mse"
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
    ) -> "NeuralDeepPQRStratifiedQFunction":
        """Fit neural DeepPQR and return a stratified Q function."""

        if policy is None:
            raise GenPQRConfigurationError("NeuralDeepPQRAnchorQEstimator requires the fitted behavior policy.")
        action_space = normalization_policy.action_space
        states_2d = as_2d_float(states, "states")
        next_states_2d = as_2d_float(next_states, "next_states", n_rows=states_2d.shape[0])
        mask, anchor_for_states, anchor_for_next, anchor_diagnostics = _resolve_anchor_rows(
            action_space=action_space,
            states=states_2d,
            actions=actions,
            next_states=next_states_2d,
            anchor_action=self.anchor_action,
            anchor_selector=self.anchor_selector,
            anchor_tolerance=float(self.anchor_tolerance),
        )
        pseudo = as_1d_float(pseudo_rewards, "pseudo_rewards", n_rows=states_2d.shape[0])
        terminals_1d = optional_terminals(terminals, states_2d.shape[0])
        weights = np.ones(states_2d.shape[0], dtype=np.float64) if sample_weight is None else optional_weights(sample_weight, states_2d.shape[0])
        anchor_count = int(np.sum(mask))
        if anchor_count == 0:
            raise GenPQRConfigurationError("Neural DeepPQR has zero anchor-action rows.")
        if anchor_count < int(self.min_anchor_count):
            raise GenPQRConfigurationError(
                f"Neural DeepPQR requires at least min_anchor_count={int(self.min_anchor_count)} anchor rows."
            )
        if not np.any(weights[mask] > 0.0):
            raise GenPQRConfigurationError("Neural DeepPQR has zero positive-weight anchor-action rows.")
        try:
            import torch
            from torch import nn
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency.
            raise GenPQRMissingDependencyError("Install genpqr[torch] to use neural DeepPQR.") from exc

        torch.manual_seed(int(self.seed))
        if self.device == "cpu":
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:  # pragma: no cover - torch-version dependent.
                pass
        device = torch.device(self.device)

        mean = np.mean(states_2d, axis=0)
        std = np.where(np.std(states_2d, axis=0) < 1e-8, 1.0, np.std(states_2d, axis=0))
        x = ((states_2d[mask] - mean) / std).astype(np.float32)
        x_next = ((next_states_2d[mask] - mean) / std).astype(np.float32)
        rewards = pseudo[mask].astype(np.float32)
        done = terminals_1d[mask].astype(np.float32)
        w = weights[mask].astype(np.float32)
        shift_next = _normalization_log_ratio_shift(
            policy=policy,
            states=next_states_2d[mask],
            normalization_policy=normalization_policy,
            action_space=action_space,
            anchor_action=_slice_anchor_actions(anchor_for_next, mask, action_space),
            n_action_samples=int(self.n_action_samples),
        ).astype(np.float32)

        model = _build_mlp(nn, x.shape[1], self.hidden_dims).to(device)
        target_model = copy.deepcopy(model).to(device)
        target_model.eval()
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(self.lr), weight_decay=float(self.weight_decay))
        x_tensor = torch.as_tensor(x, dtype=torch.float32, device=device)
        x_next_tensor = torch.as_tensor(x_next, dtype=torch.float32, device=device)
        reward_tensor = torch.as_tensor(rewards, dtype=torch.float32, device=device)
        done_tensor = torch.as_tensor(done, dtype=torch.float32, device=device)
        weight_tensor = torch.as_tensor(w / max(float(np.mean(w)), 1e-8), dtype=torch.float32, device=device)
        shift_tensor = torch.as_tensor(shift_next, dtype=torch.float32, device=device)

        train_idx, val_idx = _train_val_split(anchor_count, float(self.validation_fraction), int(self.seed))
        best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        best_loss = np.inf
        last_train_loss = np.inf
        best_epoch = 0
        early_stop_reason = "max_epochs"
        stale = 0
        batch_size = max(1, min(int(self.batch_size), train_idx.shape[0]))
        rng = np.random.default_rng(int(self.seed))
        for epoch in range(int(self.max_epochs)):
            shuffled = np.array(train_idx, copy=True)
            rng.shuffle(shuffled)
            with torch.no_grad():
                next_values = target_model(x_next_tensor).reshape(-1)
                targets_all = reward_tensor + float(gamma) * (1.0 - done_tensor) * (next_values + shift_tensor)
            epoch_losses = []
            for start in range(0, shuffled.shape[0], batch_size):
                batch = shuffled[start : start + batch_size]
                pred = model(x_tensor[batch]).reshape(-1)
                loss_vec = (pred - targets_all[batch]) ** 2
                loss = torch.mean(loss_vec * weight_tensor[batch])
                optimizer.zero_grad()
                loss.backward()
                if self.gradient_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(self.gradient_clip_norm))
                optimizer.step()
                epoch_losses.append(float(loss.detach().cpu().item()))
            last_train_loss = float(np.mean(epoch_losses)) if epoch_losses else np.inf
            if (epoch + 1) % max(1, int(self.target_update_interval)) == 0:
                _soft_update(target_model, model, float(self.target_tau))
            with torch.no_grad():
                eval_idx = val_idx if val_idx.shape[0] else train_idx
                next_values = target_model(x_next_tensor).reshape(-1)
                targets_all = reward_tensor + float(gamma) * (1.0 - done_tensor) * (next_values + shift_tensor)
                val_pred = model(x_tensor[eval_idx]).reshape(-1)
                val_loss = torch.mean((val_pred - targets_all[eval_idx]) ** 2).item()
            if val_loss < best_loss - 1e-8:
                best_loss = float(val_loss)
                best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
                best_epoch = int(epoch + 1)
                stale = 0
            else:
                stale += 1
            if stale >= int(self.patience):
                early_stop_reason = "patience"
                break
        model.load_state_dict(best_state)
        model.eval()
        anchor_probs = _anchor_probabilities(policy, states_2d, action_space, anchor_for_states)
        diagnostics = {
            "backend": "neural_deep_pqr_anchor",
            "anchor_action": _diagnostic_anchor_action(anchor_for_states, action_space),
            "anchor_count": anchor_count,
            "weighted_anchor_count": float(np.sum(weights[mask])),
            "anchor_fraction": float(anchor_count / states_2d.shape[0]),
            "mean_anchor_policy_probability": float(np.mean(anchor_probs)) if action_space.kind == "discrete" else None,
            "mean_anchor_policy_density": float(np.mean(anchor_probs)) if action_space.kind == "continuous" else None,
            "weak_anchor_support": bool(anchor_count / states_2d.shape[0] < float(self.weak_anchor_fraction)),
            "anchor_tolerance": float(self.anchor_tolerance),
            "validation_loss": float(best_loss),
            "train_loss": float(last_train_loss),
            "best_epoch": int(best_epoch),
            "early_stop_reason": early_stop_reason,
            "epochs_trained": int(epoch + 1),
            "device": str(device),
            "deterministic_cpu": bool(str(device) == "cpu"),
            "target_update_interval": int(self.target_update_interval),
            "target_tau": float(self.target_tau),
            "gradient_clip_norm": None if self.gradient_clip_norm is None else float(self.gradient_clip_norm),
            "validation_metric": self.validation_metric,
            "normalization_log_ratio_shift_mean": float(np.mean(shift_next)),
            "normalization_log_ratio_shift_std": float(np.std(shift_next)),
            "normalization_log_ratio_shift_min": float(np.min(shift_next)),
            "normalization_log_ratio_shift_max": float(np.max(shift_next)),
        }
        diagnostics.update(anchor_diagnostics)
        self.diagnostics = diagnostics
        return NeuralDeepPQRStratifiedQFunction(
            model=model,
            input_mean=mean,
            input_std=std,
            policy=policy,
            action_space=action_space,
            anchor_action=self.anchor_action if action_space.kind == "continuous" else int(anchor_for_states),
            hidden_dims=tuple(int(width) for width in self.hidden_dims),
            diagnostics=diagnostics,
            torch_module=torch,
        )


@dataclass
class NeuralDeepPQRStratifiedQFunction:
    """Fitted neural DeepPQR stratified Q function."""

    model: Any
    input_mean: Array
    input_std: Array
    policy: EstimatedPolicy
    action_space: ActionSpaceSpec
    anchor_action: int | float | Array | Callable[[Array], Array]
    hidden_dims: tuple[int, ...] = (128, 128)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    torch_module: Any | None = None

    def predict_anchor_value(self, states: Array) -> Array:
        """Predict state-only anchor values."""

        torch = self.torch_module
        if torch is None:  # pragma: no cover - only for unusual unpickled states.
            import torch as torch_import

            torch = torch_import
        states_2d = as_2d_float(states, "states")
        x = ((states_2d - self.input_mean) / self.input_std).astype(np.float32)
        device = next(self.model.parameters()).device
        with torch.no_grad():
            values = self.model(torch.as_tensor(x, dtype=torch.float32, device=device)).detach().cpu().numpy().reshape(-1)
        return values.astype(np.float64)

    def predict_q(self, states: Array, actions: Array) -> Array:
        """Predict reconstructed DeepPQR Q-values."""

        states_2d = as_2d_float(states, "states")
        if self.action_space.kind == "discrete":
            encoded_actions = self.action_space.action_indices(actions, n_rows=states_2d.shape[0])
            anchor = np.full(states_2d.shape[0], int(self.anchor_action), dtype=np.int64)
        else:
            encoded_actions = self.action_space.action_matrix(actions, n_rows=states_2d.shape[0])
            anchor = _continuous_anchor_actions(
                self.anchor_action,
                states_2d,
                self.action_space,
                name="anchor_action",
            )
        logp_action = as_1d_float(
            self.policy.log_prob(states_2d, encoded_actions),
            "policy.log_prob",
            n_rows=states_2d.shape[0],
        )
        logp_anchor = as_1d_float(
            self.policy.log_prob(states_2d, anchor),
            "policy.log_prob",
            n_rows=states_2d.shape[0],
        )
        return self.predict_anchor_value(states_2d) + logp_action - logp_anchor

    def predict_q_matrix(self, states: Array) -> Array:
        """Predict Q-values for all finite actions."""

        states_2d = as_2d_float(states, "states")
        if self.action_space.kind != "discrete":
            raise GenPQRConfigurationError("predict_q_matrix is only available for discrete action spaces.")
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
        """Estimate ``E_mu[Q(s,A)]``."""

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


def _resolve_anchor_rows(
    *,
    action_space: ActionSpaceSpec,
    states: Array,
    actions: Array,
    next_states: Array,
    anchor_action: int | float | Array | Callable[[Array], Array],
    anchor_selector: Callable[[Array, Array], Array] | None,
    anchor_tolerance: float,
) -> tuple[Array, int | Array, int | Array, dict[str, Any]]:
    states_2d = as_2d_float(states, "states")
    next_states_2d = as_2d_float(next_states, "next_states", n_rows=states_2d.shape[0])
    if anchor_tolerance < 0.0:
        raise ValueError("anchor_tolerance must be nonnegative.")
    if action_space.kind == "discrete":
        anchor_idx = _discrete_anchor_index(anchor_action, action_space)
        action_idx = action_space.action_indices(actions, n_rows=states_2d.shape[0])
        mask = action_idx == anchor_idx
        diagnostics = {
            "anchor_kind": "discrete",
            "anchor_selector_used": False,
            "continuous_anchor_distance_mean": None,
            "continuous_anchor_distance_max": None,
        }
        return mask, anchor_idx, anchor_idx, diagnostics

    action_matrix = action_space.action_matrix(actions, n_rows=states_2d.shape[0])
    anchor_for_states = _continuous_anchor_actions(anchor_action, states_2d, action_space, name="anchor_action")
    anchor_for_next = _continuous_anchor_actions(anchor_action, next_states_2d, action_space, name="anchor_action")
    distances = np.max(np.abs(action_matrix - anchor_for_states), axis=1)
    selector_used = anchor_selector is not None
    if anchor_selector is None:
        mask = distances <= float(anchor_tolerance)
    else:
        raw_mask = np.asarray(anchor_selector(states_2d, action_matrix))
        if raw_mask.shape not in {(states_2d.shape[0],), (states_2d.shape[0], 1)}:
            raise ValueError("anchor_selector must return a boolean mask with one value per row.")
        if not np.all(np.isfinite(raw_mask.astype(np.float64))):
            raise ValueError("anchor_selector mask must contain finite boolean values.")
        mask = raw_mask.reshape(-1).astype(bool)
        if np.any(mask & (distances > float(anchor_tolerance))):
            raise GenPQRConfigurationError(
                "anchor_selector selected rows whose actions do not match anchor_action within anchor_tolerance. "
                "Use a callable anchor_action(states) that matches selected rows, or loosen anchor_tolerance."
            )
    selected_distances = distances[mask]
    diagnostics = {
        "anchor_kind": "continuous_selector" if selector_used else "continuous_fixed",
        "anchor_selector_used": bool(selector_used),
        "continuous_anchor_distance_mean": (
            None if selected_distances.size == 0 else float(np.mean(selected_distances))
        ),
        "continuous_anchor_distance_max": (
            None if selected_distances.size == 0 else float(np.max(selected_distances))
        ),
    }
    return mask, anchor_for_states, anchor_for_next, diagnostics


def _discrete_anchor_index(anchor_action: int | float | Array | Callable[[Array], Array], action_space: ActionSpaceSpec) -> int:
    if callable(anchor_action):
        raise GenPQRConfigurationError("Discrete neural DeepPQR requires an integer anchor_action, not a callable.")
    arr = np.asarray(anchor_action)
    if arr.ndim != 0:
        raise GenPQRConfigurationError("Discrete neural DeepPQR requires a scalar integer anchor_action.")
    anchor_idx = int(arr.item())
    if not np.isclose(float(anchor_idx), float(arr.item())):
        raise GenPQRConfigurationError("Discrete neural DeepPQR requires an integer anchor_action.")
    if anchor_idx < 0 or anchor_idx >= int(action_space.n_actions):
        raise ValueError("anchor_action is out of bounds.")
    return anchor_idx


def _continuous_anchor_actions(
    anchor_action: int | float | Array | Callable[[Array], Array],
    states: Array,
    action_space: ActionSpaceSpec,
    *,
    name: str,
) -> Array:
    states_2d = as_2d_float(states, "states")
    if action_space.kind != "continuous":
        raise GenPQRConfigurationError("_continuous_anchor_actions requires a continuous action space.")
    callable_anchor = callable(anchor_action)
    raw = anchor_action(states_2d) if callable_anchor else anchor_action
    arr = np.asarray(raw, dtype=np.float64)
    action_dim = int(action_space.action_dim)
    n_rows = states_2d.shape[0]
    if arr.ndim == 0:
        arr = np.full((n_rows, action_dim), float(arr), dtype=np.float64)
    elif arr.ndim == 1:
        if arr.shape[0] == action_dim:
            arr = np.tile(arr.reshape(1, action_dim), (n_rows, 1))
        elif action_dim == 1 and arr.shape[0] == n_rows:
            if not callable_anchor:
                raise ValueError(f"{name} cannot be a per-row array unless it is produced by a callable.")
            arr = arr.reshape(n_rows, 1)
        else:
            raise ValueError(f"{name} must be a scalar, an action_dim vector, or one action per state.")
    elif arr.ndim == 2:
        if arr.shape == (1, action_dim):
            arr = np.tile(arr, (n_rows, 1))
        elif arr.shape != (n_rows, action_dim):
            raise ValueError(f"{name} must have shape (n, action_dim).")
        elif not callable_anchor:
            raise ValueError(f"{name} cannot be a per-row array unless it is produced by a callable.")
    else:
        raise ValueError(f"{name} must be scalar, 1D, or 2D.")
    action_space.validate_actions(arr, n_rows=n_rows, name=name)
    return arr


def _slice_anchor_actions(anchor_actions: int | Array, mask: Array, action_space: ActionSpaceSpec) -> int | Array:
    if action_space.kind == "discrete":
        return int(anchor_actions)
    return np.asarray(anchor_actions, dtype=np.float64)[np.asarray(mask, dtype=bool)]


def _diagnostic_anchor_action(anchor_actions: int | Array, action_space: ActionSpaceSpec) -> int | list[float] | str:
    if action_space.kind == "discrete":
        return int(anchor_actions)
    arr = np.asarray(anchor_actions, dtype=np.float64)
    if arr.ndim == 2 and arr.shape[0] and np.allclose(arr, arr[[0]]):
        return [float(value) for value in arr[0]]
    return "state_dependent"


def _build_mlp(nn: Any, input_dim: int, hidden_dims: tuple[int, ...]) -> Any:
    layers = []
    last = int(input_dim)
    for width in hidden_dims:
        layers.append(nn.Linear(last, int(width)))
        layers.append(nn.ReLU())
        last = int(width)
    layers.append(nn.Linear(last, 1))
    return nn.Sequential(*layers)


def _train_val_split(n_rows: int, fraction: float, seed: int) -> tuple[Array, Array]:
    indices = np.arange(int(n_rows), dtype=np.int64)
    if n_rows < 3 or fraction <= 0.0:
        return indices, np.empty(0, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    rng.shuffle(indices)
    n_val = int(np.floor(n_rows * min(max(float(fraction), 0.0), 0.9)))
    n_val = max(1, min(n_val, n_rows - 1))
    return indices[n_val:], indices[:n_val]


def _soft_update(target_model: Any, model: Any, tau: float) -> None:
    tau_value = min(max(float(tau), 0.0), 1.0)
    for target_param, source_param in zip(target_model.parameters(), model.parameters()):
        target_param.data.mul_(1.0 - tau_value).add_(source_param.data, alpha=tau_value)
