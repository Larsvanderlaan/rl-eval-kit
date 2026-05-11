from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .utils import TransitionBatch, clip_normalize_weights, train_valid_split


@dataclass
class FQEConfig:
    """Configuration for regularized neural FQE."""

    gamma: float = 0.99
    hidden_dims: Sequence[int] = (128, 128)
    n_outer_iters: int = 40
    epochs_per_iter: int = 30
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip_norm: float | None = 5.0
    target_update_tau: float = 0.05
    valid_fraction: float = 0.1
    early_stopping_patience: int = 5
    min_improvement: float = 1e-5
    device: str = "cpu"


@dataclass
class FQEResult:
    """Neural FQE output."""

    model: nn.Module
    history: dict[str, list[float]]


class QNetwork(nn.Module):
    """Small MLP over one-hot state-action inputs."""

    def __init__(self, input_dim: int, hidden_dims: Iterable[int]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, int(hidden_dim)))
            layers.append(nn.ReLU())
            prev_dim = int(hidden_dim)
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _encode_state_action(
    states: np.ndarray,
    actions: np.ndarray,
    n_states: int,
    n_actions: int,
) -> np.ndarray:
    x = np.zeros((len(states), n_states + n_actions), dtype=np.float32)
    rows = np.arange(len(states))
    x[rows, np.asarray(states, dtype=np.int64)] = 1.0
    x[rows, n_states + np.asarray(actions, dtype=np.int64)] = 1.0
    return x


def _weighted_mse(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return torch.mean(weight * (pred - target) ** 2)


def fit_fqe_nn(
    batch: TransitionBatch,
    n_states: int,
    n_actions: int,
    weights: np.ndarray | None = None,
    state_action_features: np.ndarray | None = None,
    next_state_action_features: np.ndarray | None = None,
    config: FQEConfig | None = None,
    seed: int = 0,
) -> FQEResult:
    """
    Fit a neural FQE model with sample weights.

    The implementation is intentionally simple but uses standard production
    stabilizers for weighted Bellman regression:
    - Tikhonov regularization via AdamW weight decay,
    - gradient clipping,
    - a lagged target network,
    - a validation split with early stopping.
    """

    if config is None:
        config = FQEConfig()

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    device = torch.device(config.device)

    states = np.asarray(batch.states, dtype=np.int64)
    actions = np.asarray(batch.actions, dtype=np.int64)
    rewards = np.asarray(batch.rewards, dtype=np.float32)
    next_states = np.asarray(batch.next_states, dtype=np.int64)
    next_actions = np.asarray(batch.next_actions, dtype=np.int64)

    n = len(batch)
    if weights is None:
        sample_weights = np.ones(n, dtype=np.float32)
    else:
        sample_weights = clip_normalize_weights(np.asarray(weights, dtype=np.float64)).astype(np.float32)

    if state_action_features is None:
        x_sa = _encode_state_action(states, actions, n_states, n_actions)
    else:
        x_sa = np.asarray(state_action_features, dtype=np.float32)
    if next_state_action_features is None:
        x_next = _encode_state_action(next_states, next_actions, n_states, n_actions)
    else:
        x_next = np.asarray(next_state_action_features, dtype=np.float32)

    if x_sa.shape[0] != n or x_next.shape[0] != n:
        raise ValueError("Feature arrays must have one row per transition.")
    if x_sa.ndim != 2 or x_next.ndim != 2:
        raise ValueError("Feature arrays must be 2D.")
    if x_sa.shape[1] != x_next.shape[1]:
        raise ValueError("Current and next state-action features must have the same width.")

    x_sa_t = torch.tensor(x_sa, dtype=torch.float32, device=device)
    x_next_t = torch.tensor(x_next, dtype=torch.float32, device=device)
    rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)
    weights_t = torch.tensor(sample_weights, dtype=torch.float32, device=device)

    train_idx, valid_idx = train_valid_split(n, config.valid_fraction, rng)
    train_idx_t = torch.tensor(train_idx, dtype=torch.long, device=device)
    valid_idx_t = torch.tensor(valid_idx, dtype=torch.long, device=device)

    input_dim = x_sa.shape[1]
    model = QNetwork(input_dim=input_dim, hidden_dims=config.hidden_dims).to(device)
    target_model = QNetwork(input_dim=input_dim, hidden_dims=config.hidden_dims).to(device)
    target_model.load_state_dict(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    history: dict[str, list[float]] = {"train_loss": [], "valid_loss": []}

    for _ in range(config.n_outer_iters):
        with torch.no_grad():
            fixed_targets = rewards_t + config.gamma * target_model(x_next_t)

        train_dataset = TensorDataset(
            x_sa_t[train_idx_t],
            fixed_targets[train_idx_t],
            weights_t[train_idx_t],
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=min(config.batch_size, len(train_dataset)),
            shuffle=True,
        )

        best_state = None
        best_valid = float("inf")
        patience = 0

        for _epoch in range(config.epochs_per_iter):
            model.train()
            epoch_loss = 0.0
            epoch_weight = 0.0
            for x_batch, target_batch, weight_batch in train_loader:
                optimizer.zero_grad(set_to_none=True)
                pred = model(x_batch)
                loss = _weighted_mse(pred, target_batch, weight_batch)
                loss.backward()
                if config.grad_clip_norm is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.grad_clip_norm)
                optimizer.step()
                batch_weight = float(weight_batch.sum().item())
                epoch_loss += float(loss.item()) * batch_weight
                epoch_weight += batch_weight

            train_loss = epoch_loss / max(epoch_weight, 1e-8)

            model.eval()
            with torch.no_grad():
                if len(valid_idx) > 0:
                    pred_valid = model(x_sa_t[valid_idx_t])
                    valid_loss = float(
                        _weighted_mse(pred_valid, fixed_targets[valid_idx_t], weights_t[valid_idx_t]).item()
                    )
                else:
                    valid_loss = train_loss

            history["train_loss"].append(train_loss)
            history["valid_loss"].append(valid_loss)

            if valid_loss + config.min_improvement < best_valid:
                best_valid = valid_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1
                if patience >= config.early_stopping_patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        with torch.no_grad():
            for target_param, param in zip(target_model.parameters(), model.parameters()):
                target_param.data.mul_(1.0 - config.target_update_tau)
                target_param.data.add_(config.target_update_tau * param.data)

    return FQEResult(model=model, history=history)


def fit_weighted_fqe_nn(
    batch: TransitionBatch,
    n_states: int,
    n_actions: int,
    weights: np.ndarray | None = None,
    state_action_features: np.ndarray | None = None,
    next_state_action_features: np.ndarray | None = None,
    config: FQEConfig | None = None,
    seed: int = 0,
) -> FQEResult:
    """Thin alias emphasizing that weighted FQE takes per-sample weights directly."""

    return fit_fqe_nn(
        batch=batch,
        n_states=n_states,
        n_actions=n_actions,
        weights=weights,
        state_action_features=state_action_features,
        next_state_action_features=next_state_action_features,
        config=config,
        seed=seed,
    )


def predict_q_values(
    model: nn.Module,
    states: np.ndarray,
    actions: np.ndarray,
    n_states: int,
    n_actions: int,
    state_action_features: np.ndarray | None = None,
    device: str = "cpu",
) -> np.ndarray:
    """Predict Q-values for a batch of state-action pairs."""

    model.eval()
    if state_action_features is None:
        x = _encode_state_action(states, actions, n_states, n_actions)
    else:
        x = np.asarray(state_action_features, dtype=np.float32)
    with torch.no_grad():
        preds = model(torch.tensor(x, dtype=torch.float32, device=device)).cpu().numpy()
    return preds.astype(np.float64)
