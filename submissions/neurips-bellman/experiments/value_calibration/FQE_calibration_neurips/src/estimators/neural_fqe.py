from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ..data import TransitionBatch
from ..policies import SoftmaxPolicy
from .random_feature_fqe import state_action_matrix


Array = np.ndarray


@dataclass
class NeuralFQEConfig:
    gamma: float = 0.95
    hidden_dims: tuple[int, ...] = (64, 64)
    n_iters: int = 18
    epochs_per_iter: int = 8
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4
    target_tau: float = 0.2
    bootstrap_on_first_iter: bool = True
    device: str = "cpu"


class QNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: tuple[int, ...]):
        super().__init__()
        layers: list[nn.Module] = []
        last = input_dim
        for width in hidden_dims:
            layers.extend([nn.Linear(last, int(width)), nn.ReLU()])
            last = int(width)
        layers.append(nn.Linear(last, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class NeuralFQEModel:
    def __init__(self, model: QNet, n_actions: int, device: str, diagnostics: dict[str, float | str] | None = None):
        self.model = model
        self.n_actions = int(n_actions)
        self.device = str(device)
        self.diagnostics = diagnostics or {}

    def _features(self, states: Array, actions: Array) -> Array:
        return state_action_matrix(states, actions, self.n_actions).astype(np.float32)

    def predict_q(self, states: Array, actions: Array) -> Array:
        self.model.eval()
        x = torch.tensor(self._features(states, actions), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            return self.model(x).detach().cpu().numpy().astype(float)

    def value(self, states: Array, policy: SoftmaxPolicy) -> Array:
        probs = policy.action_probabilities(states)
        vals = np.column_stack([
            self.predict_q(states, np.full(states.shape[0], a, dtype=int))
            for a in range(self.n_actions)
        ])
        return np.sum(probs * vals, axis=1)


def fit_neural_fqe(
    batch: TransitionBatch,
    n_actions: int,
    policy: SoftmaxPolicy,
    config: NeuralFQEConfig,
    seed: int,
    initial_model: NeuralFQEModel | None = None,
) -> NeuralFQEModel:
    torch.manual_seed(int(seed))
    device = torch.device(config.device)
    x = torch.tensor(state_action_matrix(batch.states, batch.actions, n_actions), dtype=torch.float32, device=device)
    x_next = torch.tensor(
        state_action_matrix(batch.next_states, batch.next_actions, n_actions), dtype=torch.float32, device=device
    )
    rewards = torch.tensor(batch.rewards, dtype=torch.float32, device=device)
    if initial_model is None:
        model = QNet(x.shape[1], tuple(config.hidden_dims)).to(device)
    else:
        model = QNet(x.shape[1], tuple(config.hidden_dims)).to(device)
        model.load_state_dict(initial_model.model.state_dict())
    target_model = QNet(x.shape[1], tuple(config.hidden_dims)).to(device)
    target_model.load_state_dict(model.state_dict())
    opt = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    loader = DataLoader(TensorDataset(x, rewards), batch_size=min(config.batch_size, len(batch)), shuffle=True)
    final_loss = float("nan")
    for bellman_iter in range(int(config.n_iters)):
        with torch.no_grad():
            if bellman_iter == 0 and not bool(config.bootstrap_on_first_iter):
                targets = rewards
            else:
                targets = rewards + float(config.gamma) * target_model(x_next)
        for _epoch in range(int(config.epochs_per_iter)):
            for xb, _ in loader:
                idx = torch.randint(0, x.shape[0], (xb.shape[0],), device=device)
                opt.zero_grad(set_to_none=True)
                pred = model(x[idx])
                loss = torch.mean((pred - targets[idx]) ** 2)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                final_loss = float(loss.detach().cpu().item())
        with torch.no_grad():
            for tp, p in zip(target_model.parameters(), model.parameters()):
                tp.data.mul_(1.0 - float(config.target_tau)).add_(float(config.target_tau) * p.data)
    with torch.no_grad():
        q_train = model(x).detach().cpu().numpy().astype(float)
    diagnostics = {
        "actual_bellman_iterations": float(config.n_iters),
        "actual_epochs_per_iter": float(config.epochs_per_iter),
        "bootstrap_on_first_iter": float(bool(config.bootstrap_on_first_iter)),
        "neural_train_loss_last": float(final_loss),
        "q_train_min": float(np.nanmin(q_train)) if q_train.size else float("nan"),
        "q_train_max": float(np.nanmax(q_train)) if q_train.size else float("nan"),
        "q_train_std": float(np.nanstd(q_train)) if q_train.size else float("nan"),
    }
    return NeuralFQEModel(model, n_actions, config.device, diagnostics=diagnostics)
