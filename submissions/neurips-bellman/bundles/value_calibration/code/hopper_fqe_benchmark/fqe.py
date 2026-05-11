from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from FQE_neurips.utils import stabilize_weights

from .data import HopperTrajectoryDataset
from .policies import HopperPicklePolicy


def _soft_update(net: nn.Module, target_net: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.mul_(1.0 - tau)
            target_param.data.add_(tau * param.data)


class CriticNet(nn.Module):
    def __init__(self, state_dim: int, action_dim: int) -> None:
        super().__init__()
        self.critic = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        x = torch.cat([states, actions], dim=-1)
        return self.critic(x).squeeze(-1)


@dataclass
class QFitterConfig:
    gamma: float = 0.995
    critic_lr: float = 3e-4
    weight_decay: float = 1e-5
    tau: float = 0.005
    batch_size: int = 256
    num_updates: int = 20_000
    log_interval: int = 1_000
    device: str = "cpu"


@dataclass
class QFitterResult:
    model: CriticNet
    target_model: CriticNet
    loss_history: list[float]
    training_metadata: dict[str, float]


class QFitter:
    def __init__(self, state_dim: int, action_dim: int, config: QFitterConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)
        self.critic = CriticNet(state_dim, action_dim).to(self.device)
        self.critic_target = CriticNet(state_dim, action_dim).to(self.device)
        _soft_update(self.critic, self.critic_target, tau=1.0)
        self.optimizer = torch.optim.AdamW(
            self.critic.parameters(),
            lr=config.critic_lr,
            weight_decay=config.weight_decay,
        )

    def __call__(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.critic_target(states, actions)

    def update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        next_states: torch.Tensor,
        next_actions: torch.Tensor,
        rewards: torch.Tensor,
        masks: torch.Tensor,
        weights: torch.Tensor,
        min_reward: float,
        max_reward: float,
    ) -> float:
        discount = self.config.gamma
        next_q = self.critic_target(next_states, next_actions) / (1.0 - discount)
        target_q = rewards + discount * masks * next_q
        target_q = torch.clamp(
            target_q,
            min=min_reward / (1.0 - discount),
            max=max_reward / (1.0 - discount),
        )

        q = self.critic(states, actions) / (1.0 - discount)
        loss = torch.sum((target_q - q) ** 2 * weights) / torch.sum(weights)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        _soft_update(self.critic, self.critic_target, tau=self.config.tau)
        return float(loss.item())

    def estimate_returns(
        self,
        initial_states: torch.Tensor,
        initial_actions: torch.Tensor,
        initial_weights: torch.Tensor,
    ) -> float:
        preds = self(initial_states, initial_actions)
        value = torch.sum(preds * initial_weights) / torch.sum(initial_weights)
        return float(value.item())


def train_q_fitter(
    dataset: HopperTrajectoryDataset,
    policy: HopperPicklePolicy,
    *,
    sample_weights: np.ndarray | None = None,
    config: QFitterConfig | None = None,
    seed: int = 0,
) -> QFitterResult:
    if config is None:
        config = QFitterConfig()

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    fitter = QFitter(dataset.observation_dim, dataset.action_dim, config=config)
    device = fitter.device

    if sample_weights is None:
        train_weights = np.ones(len(dataset), dtype=np.float32)
    else:
        train_weights, _ = stabilize_weights(np.asarray(sample_weights, dtype=np.float64))
        train_weights = train_weights.astype(np.float32)

    states = torch.as_tensor(dataset.observations, dtype=torch.float32, device=device)
    actions = torch.as_tensor(dataset.actions, dtype=torch.float32, device=device)
    next_states = torch.as_tensor(dataset.next_observations, dtype=torch.float32, device=device)
    rewards = torch.as_tensor(dataset.rewards, dtype=torch.float32, device=device)
    masks = torch.as_tensor(dataset.masks, dtype=torch.float32, device=device)
    weights = torch.as_tensor(train_weights, dtype=torch.float32, device=device)

    min_reward = float(dataset.rewards.min())
    max_reward = float(dataset.rewards.max())
    batch_size = min(int(config.batch_size), len(dataset))

    loss_history: list[float] = []
    for step in range(config.num_updates):
        indices = rng.integers(0, len(dataset), size=batch_size)
        next_actions_np = policy.sample_actions(dataset.next_observations_raw[indices], rng=rng, deterministic=False)
        next_actions = torch.as_tensor(next_actions_np, dtype=torch.float32, device=device)
        loss = fitter.update(
            states=states[indices],
            actions=actions[indices],
            next_states=next_states[indices],
            next_actions=next_actions,
            rewards=rewards[indices],
            masks=masks[indices],
            weights=weights[indices],
            min_reward=min_reward,
            max_reward=max_reward,
        )
        if step % config.log_interval == 0 or step == config.num_updates - 1:
            loss_history.append(loss)

    return QFitterResult(
        model=fitter.critic,
        target_model=fitter.critic_target,
        loss_history=loss_history,
        training_metadata={
            "num_updates": float(config.num_updates),
            "batch_size": float(batch_size),
            "gamma": float(config.gamma),
        },
    )


def estimate_policy_return(
    fitter_result: QFitterResult,
    dataset: HopperTrajectoryDataset,
    policy: HopperPicklePolicy,
    *,
    gamma: float,
    seed: int = 0,
    device: str = "cpu",
) -> float:
    rng = np.random.default_rng(seed)
    target_model = fitter_result.target_model
    target_model.eval()
    initial_actions = policy.sample_actions(dataset.initial_observations_raw, rng=rng, deterministic=False)
    with torch.no_grad():
        value_scaled = (
            target_model(
                torch.as_tensor(dataset.normalize_states(dataset.initial_observations_raw), dtype=torch.float32, device=device),
                torch.as_tensor(initial_actions, dtype=torch.float32, device=device),
            )
            .mul(torch.as_tensor(dataset.initial_weights, dtype=torch.float32, device=device))
            .sum()
            .div(torch.as_tensor(dataset.initial_weights, dtype=torch.float32, device=device).sum())
            .item()
        )
    value_scaled = float(dataset.unnormalize_rewards(np.array(value_scaled, dtype=np.float32)))
    return value_scaled / max(1.0 - gamma, 1e-8)
