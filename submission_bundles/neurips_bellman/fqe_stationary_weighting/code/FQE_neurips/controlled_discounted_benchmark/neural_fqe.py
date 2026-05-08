from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from FQE_neurips.utils import TransitionBatch

from .configs import NeuralFQEConfig
from .envs import LinearGaussianEnv
from .policies import GaussianLinearPolicy


class _QNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Sequence[int]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev, int(hidden_dim)))
            layers.append(nn.SiLU())
            prev = int(hidden_dim)
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


@dataclass
class NeuralFQEOutput:
    q_function: "NeuralQFunction"
    history: dict[str, list[float]]


class NeuralQFunction:
    def __init__(
        self,
        model: nn.Module,
        mean: np.ndarray,
        scale: np.ndarray,
        *,
        device: str,
        action_quadrature_order: int,
        state_quadrature_order: int,
    ) -> None:
        self.model = model
        self.mean = np.asarray(mean, dtype=np.float64).reshape(1, 3)
        self.scale = np.asarray(scale, dtype=np.float64).reshape(1, 3)
        self.device = device
        self.action_quadrature_order = int(action_quadrature_order)
        self.state_quadrature_order = int(state_quadrature_order)

    def _features(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        x = np.concatenate(
            [
                np.asarray(states, dtype=np.float64).reshape(-1, 2),
                np.asarray(actions, dtype=np.float64).reshape(-1, 1),
            ],
            axis=1,
        )
        return ((x - self.mean) / self.scale).astype(np.float32)

    def evaluate(self, states: np.ndarray, actions: np.ndarray, batch_size: int = 8192) -> np.ndarray:
        x = self._features(states, actions)
        preds = []
        self.model.eval()
        with torch.no_grad():
            for start in range(0, x.shape[0], batch_size):
                xb = torch.as_tensor(x[start : start + batch_size], dtype=torch.float32, device=self.device)
                preds.append(self.model(xb).detach().cpu().numpy())
        return np.concatenate(preds).astype(np.float64)

    def to_state_value(self, policy: GaussianLinearPolicy) -> "NeuralStateValueFunction":
        return NeuralStateValueFunction(self, policy)


class NeuralStateValueFunction:
    def __init__(self, q_function: NeuralQFunction, policy: GaussianLinearPolicy) -> None:
        self.q_function = q_function
        self.policy = policy
        nodes, weights = np.polynomial.hermite.hermgauss(q_function.action_quadrature_order)
        self.action_nodes = nodes.astype(np.float64)
        self.action_weights = (weights / np.sqrt(np.pi)).astype(np.float64)
        state_nodes, state_weights = np.polynomial.hermite.hermgauss(q_function.state_quadrature_order)
        self.state_nodes = state_nodes.astype(np.float64)
        self.state_weights = (state_weights / np.sqrt(np.pi)).astype(np.float64)

    def evaluate(self, states: np.ndarray) -> np.ndarray:
        states_arr = np.asarray(states, dtype=np.float64).reshape(-1, 2)
        action_mean = self.policy.mean_action(states_arr)
        out = np.zeros(states_arr.shape[0], dtype=np.float64)
        for node, weight in zip(self.action_nodes, self.action_weights):
            actions = action_mean + np.sqrt(2.0) * self.policy.action_sd * node
            out += weight * self.q_function.evaluate(states_arr, actions)
        return out

    def expectation_under_gaussian(self, mean: np.ndarray, cov: np.ndarray) -> float:
        mean_arr = np.asarray(mean, dtype=np.float64).reshape(2)
        cov_arr = np.asarray(cov, dtype=np.float64).reshape(2, 2)
        chol = np.linalg.cholesky(cov_arr + 1e-12 * np.eye(2, dtype=np.float64))
        total = 0.0
        for i, j in product(range(self.state_nodes.size), repeat=2):
            z = np.array([self.state_nodes[i], self.state_nodes[j]], dtype=np.float64)
            state = mean_arr + np.sqrt(2.0) * (chol @ z)
            weight = self.state_weights[i] * self.state_weights[j]
            total += weight * float(self.evaluate(state.reshape(1, 2))[0])
        return float(total)

    def expectation_under_transition(self, means: np.ndarray, transition_cov: np.ndarray) -> np.ndarray:
        means_arr = np.asarray(means, dtype=np.float64).reshape(-1, 2)
        cov_arr = np.asarray(transition_cov, dtype=np.float64).reshape(2, 2)
        chol = np.linalg.cholesky(cov_arr + 1e-12 * np.eye(2, dtype=np.float64))
        out = np.zeros(means_arr.shape[0], dtype=np.float64)
        for i, j in product(range(self.state_nodes.size), repeat=2):
            z = np.array([self.state_nodes[i], self.state_nodes[j]], dtype=np.float64)
            states = means_arr + np.sqrt(2.0) * (z @ chol.T)
            weight = self.state_weights[i] * self.state_weights[j]
            out += weight * self.evaluate(states)
        return out


def _normalize_weights(weights: np.ndarray | None, n: int) -> np.ndarray:
    if weights is None:
        return np.ones(n, dtype=np.float32)
    w = np.maximum(np.asarray(weights, dtype=np.float64).reshape(-1), 1e-12)
    w = w / np.maximum(np.mean(w), 1e-12)
    return w.astype(np.float32)


def _split_indices(n: int, valid_fraction: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    idx = np.arange(n)
    rng.shuffle(idx)
    n_valid = int(round(valid_fraction * n))
    return idx[n_valid:], idx[:n_valid]


def _expected_target_q(
    model: nn.Module,
    next_states: np.ndarray,
    target_policy: GaussianLinearPolicy,
    mean: np.ndarray,
    scale: np.ndarray,
    *,
    order: int,
    device: str,
    batch_size: int,
) -> np.ndarray:
    nodes, weights = np.polynomial.hermite.hermgauss(order)
    weights = weights / np.sqrt(np.pi)
    states_arr = np.asarray(next_states, dtype=np.float64).reshape(-1, 2)
    action_mean = target_policy.mean_action(states_arr)
    out = np.zeros(states_arr.shape[0], dtype=np.float64)
    model.eval()
    with torch.no_grad():
        for node, weight in zip(nodes, weights):
            actions = action_mean + np.sqrt(2.0) * target_policy.action_sd * float(node)
            x = np.concatenate([states_arr, actions], axis=1)
            x = ((x - mean.reshape(1, 3)) / scale.reshape(1, 3)).astype(np.float32)
            preds = []
            for start in range(0, x.shape[0], batch_size):
                xb = torch.as_tensor(x[start : start + batch_size], dtype=torch.float32, device=device)
                preds.append(model(xb).detach().cpu().numpy())
            out += float(weight) * np.concatenate(preds).astype(np.float64)
    return out


def fit_neural_fqe(
    batch: TransitionBatch,
    *,
    env: LinearGaussianEnv,
    target_policy: GaussianLinearPolicy,
    value_gamma: float,
    sample_weights: np.ndarray | None,
    config: NeuralFQEConfig,
    seed: int,
) -> NeuralFQEOutput:
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    device = torch.device(config.device)

    states = np.asarray(batch.states, dtype=np.float64).reshape(-1, 2)
    actions = np.asarray(batch.actions, dtype=np.float64).reshape(-1, 1)
    next_states = np.asarray(batch.next_states, dtype=np.float64).reshape(-1, 2)
    rewards = np.asarray(batch.rewards, dtype=np.float32).reshape(-1)
    n = states.shape[0]
    weights = _normalize_weights(sample_weights, n)
    train_idx, valid_idx = _split_indices(n, config.valid_fraction, rng)
    if train_idx.size == 0:
        train_idx = np.arange(n)
        valid_idx = np.arange(0)

    raw_x = np.concatenate([states, actions], axis=1)
    mean = raw_x[train_idx].mean(axis=0)
    scale = raw_x[train_idx].std(axis=0)
    scale = np.where(scale < 1e-8, 1.0, scale)
    x = ((raw_x - mean.reshape(1, 3)) / scale.reshape(1, 3)).astype(np.float32)

    x_t = torch.as_tensor(x, dtype=torch.float32, device=device)
    rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=device)
    weights_t = torch.as_tensor(weights, dtype=torch.float32, device=device)
    train_idx_t = torch.as_tensor(train_idx, dtype=torch.long, device=device)
    valid_idx_t = torch.as_tensor(valid_idx, dtype=torch.long, device=device)

    model = _QNetwork(input_dim=3, hidden_dims=config.hidden_dims).to(device)
    target_model = _QNetwork(input_dim=3, hidden_dims=config.hidden_dims).to(device)
    target_model.load_state_dict(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    history: dict[str, list[float]] = {"train_loss": [], "valid_loss": []}

    for _outer in range(config.n_outer_iters):
        next_values = _expected_target_q(
            target_model,
            next_states,
            target_policy,
            mean,
            scale,
            order=config.action_quadrature_order,
            device=config.device,
            batch_size=max(config.batch_size * 4, 1024),
        ).astype(np.float32)
        fixed_targets_t = rewards_t + float(value_gamma) * torch.as_tensor(next_values, dtype=torch.float32, device=device)

        train_dataset = TensorDataset(x_t[train_idx_t], fixed_targets_t[train_idx_t], weights_t[train_idx_t])
        train_loader = DataLoader(
            train_dataset,
            batch_size=min(config.batch_size, len(train_dataset)),
            shuffle=True,
        )
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        best_valid = float("inf")
        patience = 0
        last_train = float("nan")
        for _epoch in range(config.epochs_per_iter):
            model.train()
            weighted_loss_sum = 0.0
            weight_sum = 0.0
            for xb, yb, wb in train_loader:
                optimizer.zero_grad(set_to_none=True)
                pred = model(xb)
                loss = torch.mean(wb * (pred - yb) ** 2)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.grad_clip_norm)
                optimizer.step()
                weighted_loss_sum += float(loss.item()) * float(wb.sum().item())
                weight_sum += float(wb.sum().item())
            last_train = weighted_loss_sum / max(weight_sum, 1e-8)
            model.eval()
            with torch.no_grad():
                if valid_idx.size > 0:
                    valid_pred = model(x_t[valid_idx_t])
                    valid_loss = torch.mean(
                        weights_t[valid_idx_t] * (valid_pred - fixed_targets_t[valid_idx_t]) ** 2
                    )
                    valid_score = float(valid_loss.item())
                else:
                    valid_score = last_train
            if valid_score + config.min_improvement < best_valid:
                best_valid = valid_score
                patience = 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience += 1
                if patience >= config.early_stopping_patience:
                    break
        model.load_state_dict(best_state)
        history["train_loss"].append(float(last_train))
        history["valid_loss"].append(float(best_valid))
        with torch.no_grad():
            for target_param, param in zip(target_model.parameters(), model.parameters()):
                target_param.data.mul_(1.0 - config.target_update_tau)
                target_param.data.add_(config.target_update_tau * param.data)

    model.eval()
    q_function = NeuralQFunction(
        model=model,
        mean=mean,
        scale=scale,
        device=config.device,
        action_quadrature_order=config.action_quadrature_order,
        state_quadrature_order=config.state_quadrature_order,
    )
    return NeuralFQEOutput(q_function=q_function, history=history)
