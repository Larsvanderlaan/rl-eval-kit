from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from FQE_neurips.utils import stabilize_weights

from .data import HopperTrajectoryDataset
from .fqe import CriticNet
from .policies import HopperPicklePolicy


@dataclass
class DualDICEConfig:
    gamma: float = 0.995
    weight_decay: float = 1e-5
    nu_lr: float = 1e-4
    zeta_lr: float = 1e-3
    batch_size: int = 256
    num_updates: int = 20_000
    log_interval: int = 1_000
    device: str = "cpu"


@dataclass
class DualDICEResult:
    nu: CriticNet
    zeta: CriticNet
    loss_history: list[float]
    pred_ratio: float
    training_metadata: dict[str, float]


class DualDICE:
    def __init__(self, state_dim: int, action_dim: int, config: DualDICEConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)
        self.nu = CriticNet(state_dim, action_dim).to(self.device)
        self.zeta = CriticNet(state_dim, action_dim).to(self.device)
        self.nu_optimizer = torch.optim.AdamW(
            self.nu.parameters(),
            lr=config.nu_lr,
            betas=(0.0, 0.99),
            weight_decay=config.weight_decay,
        )
        self.zeta_optimizer = torch.optim.AdamW(
            self.zeta.parameters(),
            lr=config.zeta_lr,
            betas=(0.0, 0.99),
            weight_decay=config.weight_decay,
        )

    def update(
        self,
        initial_states: torch.Tensor,
        initial_actions: torch.Tensor,
        initial_weights: torch.Tensor,
        states: torch.Tensor,
        actions: torch.Tensor,
        next_states: torch.Tensor,
        next_actions: torch.Tensor,
        masks: torch.Tensor,
        weights: torch.Tensor,
    ) -> float:
        discount = self.config.gamma

        nu = self.nu(states, actions)
        nu_next = self.nu(next_states, next_actions)
        nu_0 = self.nu(initial_states, initial_actions)
        zeta = self.zeta(states, actions)

        nu_loss = (
            torch.sum(weights * ((nu - discount * masks * nu_next) * zeta - 0.5 * zeta.pow(2))) / torch.sum(weights)
            - torch.sum(initial_weights * (1.0 - discount) * nu_0) / torch.sum(initial_weights)
        )
        zeta_loss = -nu_loss

        self.nu_optimizer.zero_grad(set_to_none=True)
        self.zeta_optimizer.zero_grad(set_to_none=True)

        nu_params = tuple(self.nu.parameters())
        zeta_params = tuple(self.zeta.parameters())
        nu_grads = torch.autograd.grad(nu_loss, nu_params, retain_graph=True, allow_unused=False)
        zeta_grads = torch.autograd.grad(zeta_loss, zeta_params, retain_graph=False, allow_unused=False)

        for param, grad in zip(nu_params, nu_grads):
            param.grad = grad
        for param, grad in zip(zeta_params, zeta_grads):
            param.grad = grad

        self.nu_optimizer.step()
        self.zeta_optimizer.step()
        return float(nu_loss.item())

    def estimate_returns(
        self,
        dataset: HopperTrajectoryDataset,
        *,
        num_samples: int = 100,
        seed: int = 0,
    ) -> tuple[float, float]:
        rng = np.random.default_rng(seed)
        batch_size = min(int(self.config.batch_size), len(dataset))
        states = torch.as_tensor(dataset.observations, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(dataset.actions, dtype=torch.float32, device=self.device)
        rewards = torch.as_tensor(dataset.rewards, dtype=torch.float32, device=self.device)
        weights = torch.ones(len(dataset), dtype=torch.float32, device=self.device)

        pred_returns = 0.0
        pred_ratio = 0.0
        with torch.no_grad():
            for _ in range(num_samples):
                indices = rng.integers(0, len(dataset), size=batch_size)
                zeta = self.zeta(states[indices], actions[indices])
                batch_weights = weights[indices]
                batch_rewards = rewards[indices]
                pred_ratio += float(torch.sum(batch_weights * zeta).item() / torch.sum(batch_weights).item())
                pred_returns += float(torch.sum(batch_weights * zeta * batch_rewards).item() / torch.sum(batch_weights).item())
        return pred_returns / num_samples, pred_ratio / num_samples

    def predict_weights(self, dataset: HopperTrajectoryDataset) -> np.ndarray:
        with torch.no_grad():
            weights = (
                self.zeta(
                    torch.as_tensor(dataset.observations, dtype=torch.float32, device=self.device),
                    torch.as_tensor(dataset.actions, dtype=torch.float32, device=self.device),
                )
                .cpu()
                .numpy()
            )
        stabilized, _ = stabilize_weights(weights, min_weight=1e-4, max_weight=20.0)
        return stabilized.astype(np.float32)


def train_dual_dice(
    dataset: HopperTrajectoryDataset,
    policy: HopperPicklePolicy,
    *,
    config: DualDICEConfig | None = None,
    seed: int = 0,
) -> DualDICEResult:
    if config is None:
        config = DualDICEConfig()

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    model = DualDICE(dataset.observation_dim, dataset.action_dim, config=config)
    device = model.device

    states = torch.as_tensor(dataset.observations, dtype=torch.float32, device=device)
    actions = torch.as_tensor(dataset.actions, dtype=torch.float32, device=device)
    next_states = torch.as_tensor(dataset.next_observations, dtype=torch.float32, device=device)
    masks = torch.as_tensor(dataset.masks, dtype=torch.float32, device=device)
    weights = torch.ones(len(dataset), dtype=torch.float32, device=device)
    initial_states = torch.as_tensor(
        dataset.normalize_states(dataset.initial_observations_raw),
        dtype=torch.float32,
        device=device,
    )
    initial_weights = torch.as_tensor(dataset.initial_weights, dtype=torch.float32, device=device)

    batch_size = min(int(config.batch_size), len(dataset))
    loss_history: list[float] = []
    for step in range(config.num_updates):
        indices = rng.integers(0, len(dataset), size=batch_size)
        next_actions = policy.sample_actions(dataset.next_observations_raw[indices], rng=rng, deterministic=False)
        initial_actions = policy.sample_actions(dataset.initial_observations_raw, rng=rng, deterministic=False)
        loss = model.update(
            initial_states=initial_states,
            initial_actions=torch.as_tensor(initial_actions, dtype=torch.float32, device=device),
            initial_weights=initial_weights,
            states=states[indices],
            actions=actions[indices],
            next_states=next_states[indices],
            next_actions=torch.as_tensor(next_actions, dtype=torch.float32, device=device),
            masks=masks[indices],
            weights=weights[indices],
        )
        if step % config.log_interval == 0 or step == config.num_updates - 1:
            loss_history.append(loss)

    pred_scaled, pred_ratio = model.estimate_returns(dataset, num_samples=100, seed=seed + 17)
    return DualDICEResult(
        nu=model.nu,
        zeta=model.zeta,
        loss_history=loss_history,
        pred_ratio=float(pred_ratio),
        training_metadata={
            "num_updates": float(config.num_updates),
            "batch_size": float(batch_size),
            "gamma": float(config.gamma),
            "pred_scaled": float(pred_scaled),
        },
    )


def estimate_dual_dice_return(
    result: DualDICEResult,
    dataset: HopperTrajectoryDataset,
    *,
    gamma: float,
    seed: int = 0,
    batch_size: int = 256,
    num_samples: int = 100,
    device: str = "cpu",
) -> tuple[float, float]:
    model = DualDICE(dataset.observation_dim, dataset.action_dim, config=DualDICEConfig(gamma=gamma, device=device))
    model.nu.load_state_dict(result.nu.state_dict())
    model.zeta.load_state_dict(result.zeta.state_dict())
    pred_scaled, pred_ratio = model.estimate_returns(dataset, num_samples=num_samples, seed=seed)
    pred_scaled = float(dataset.unnormalize_rewards(np.array(pred_scaled, dtype=np.float32)))
    return pred_scaled / max(1.0 - gamma, 1e-8), float(pred_ratio)


def extract_dual_dice_weights(
    result: DualDICEResult,
    dataset: HopperTrajectoryDataset,
    *,
    gamma: float,
    device: str = "cpu",
) -> np.ndarray:
    model = DualDICE(dataset.observation_dim, dataset.action_dim, config=DualDICEConfig(gamma=gamma, device=device))
    model.nu.load_state_dict(result.nu.state_dict())
    model.zeta.load_state_dict(result.zeta.state_dict())
    return model.predict_weights(dataset)
