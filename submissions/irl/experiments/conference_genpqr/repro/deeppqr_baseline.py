"""DeepPQR-style baseline implementation."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mpl_cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(__file__).resolve().parent / ".cache"))

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from policy_estimation import EstimatedPolicy, fit_airl_policy
from utils import EPS, one_hot, standardize_fit


@dataclass
class DeepPQRResult:
    """Container for the DeepPQR baseline outputs."""

    behavior_policy: EstimatedPolicy
    anchor_q: object
    full_q: object
    reward_network: object


class AnchorQEstimate:
    """Torch-backed anchor-action Q estimate."""

    def __init__(self, model, transform, n_actions: int):
        self.model = model
        self.transform = transform
        self.n_actions = n_actions

    def predict_anchor(self, states_np: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x = torch.as_tensor(self.transform(states_np), dtype=torch.float32)
            return self.model(x).cpu().numpy()

    def predict_q(self, states_np: np.ndarray, actions_np: np.ndarray) -> np.ndarray:
        del actions_np
        return self.predict_anchor(states_np)

    def predict_all_actions(self, states_np: np.ndarray) -> np.ndarray:
        anchor = self.predict_anchor(states_np)
        return np.tile(anchor[:, None], (1, self.n_actions))


class DeepPQRFullQEstimate:
    """DeepPQR full-Q recovery from anchor Q and policy log ratios."""

    def __init__(
        self,
        anchor_q: AnchorQEstimate,
        policy: EstimatedPolicy,
        anchor_action: int,
        alpha: float,
        min_policy_prob: float,
    ):
        self.anchor_q = anchor_q
        self.policy = policy
        self.anchor_action = anchor_action
        self.alpha = alpha
        self.min_policy_prob = min_policy_prob

    def predict_all_actions(self, states_np: np.ndarray) -> np.ndarray:
        pi = np.clip(self.policy.predict_proba(states_np), self.min_policy_prob, 1.0)
        q_anchor = self.anchor_q.predict_anchor(states_np)
        return q_anchor[:, None] + self.alpha * (np.log(pi) - np.log(pi[:, [self.anchor_action]]))

    def predict_q(self, states_np: np.ndarray, actions_np: np.ndarray) -> np.ndarray:
        q_matrix = self.predict_all_actions(states_np)
        actions_np = np.asarray(actions_np, dtype=int).reshape(-1)
        return q_matrix[np.arange(q_matrix.shape[0]), actions_np]


class RewardNetworkEstimate:
    """Reward network trained from DeepPQR Algorithm 1 targets."""

    def __init__(self, model, transform, n_actions: int):
        self.model = model
        self.transform = transform
        self.n_actions = n_actions

    def _features(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        return np.concatenate([self.transform(states), one_hot(actions, self.n_actions)], axis=1)

    def predict(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        features = self._features(states, actions)
        with torch.no_grad():
            return self.model(torch.as_tensor(features, dtype=torch.float32)).cpu().numpy().reshape(-1)


def fit_anchor_q_network(
    states: np.ndarray,
    next_states: np.ndarray,
    dones: np.ndarray,
    policy: EstimatedPolicy,
    anchor_action: int,
    gamma: float,
    g_values: np.ndarray | None = None,
    alpha: float = 1.0,
    min_policy_prob: float = 1e-2,
    hidden_sizes: Sequence[int] = (128, 128),
    n_iters: int = 80,
    epochs_per_iter: int = 3,
    learning_rate: float = 1e-3,
    target_clip: float = 50.0,
) -> AnchorQEstimate:
    """Estimate the anchor-action Q function using DeepPQR's FQI-I target."""
    mean, std = standardize_fit(states)

    def transform(x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, dtype=float) - mean) / std

    class AnchorNet(nn.Module):
        def __init__(self, input_dim: int, hidden: Sequence[int]) -> None:
            super().__init__()
            layers = []
            prev = input_dim
            for h in hidden:
                layers.append(nn.Linear(prev, h))
                layers.append(nn.ReLU())
                prev = h
            layers.append(nn.Linear(prev, 1))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x).squeeze(-1)

    x = torch.as_tensor(transform(states), dtype=torch.float32)
    x_next = torch.as_tensor(transform(next_states), dtype=torch.float32)
    done_t = torch.as_tensor(dones, dtype=torch.float32)
    g_t = torch.as_tensor(
        np.zeros(states.shape[0], dtype=float) if g_values is None else np.asarray(g_values, dtype=float).reshape(-1),
        dtype=torch.float32,
    )
    net = AnchorNet(states.shape[1], hidden_sizes)
    optimizer = torch.optim.Adagrad(net.parameters(), lr=learning_rate)

    for _ in range(n_iters):
        with torch.no_grad():
            q_next_anchor = net(x_next)
            pi_next = np.clip(policy.predict_proba(next_states)[:, anchor_action], min_policy_prob, 1.0)
            log_pi_next_anchor = torch.as_tensor(np.log(pi_next), dtype=torch.float32)
            targets = g_t + gamma * (1.0 - done_t) * (-alpha * log_pi_next_anchor + q_next_anchor)
            targets = torch.clamp(targets, -target_clip, target_clip)
        ds = TensorDataset(x, targets)
        dl = DataLoader(ds, batch_size=256, shuffle=True)
        for _ in range(epochs_per_iter):
            for xb, yb in dl:
                pred = net(xb)
                loss = ((pred - yb) ** 2).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    return AnchorQEstimate(net, transform, n_actions=policy.n_actions)


def fit_reward_regression_from_q(
    states: np.ndarray,
    actions: np.ndarray,
    next_states: np.ndarray,
    dones: np.ndarray,
    full_q,
    policy: EstimatedPolicy,
    anchor_action: int,
    gamma: float,
    alpha: float = 1.0,
    min_policy_prob: float = 1e-2,
    hidden_sizes: Sequence[int] = (128, 128),
    n_epochs: int = 200,
    learning_rate: float = 1e-3,
    target_clip: float = 50.0,
) -> RewardNetworkEstimate:
    """Fit DeepPQR Algorithm 1's auxiliary regression `h(s,a)`."""
    mean, std = standardize_fit(states)

    def transform(x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, dtype=float) - mean) / std

    class HNet(nn.Module):
        def __init__(self, input_dim: int, hidden: Sequence[int]) -> None:
            super().__init__()
            layers = []
            prev = input_dim
            for h in hidden:
                layers.append(nn.Linear(prev, h))
                layers.append(nn.ReLU())
                prev = h
            layers.append(nn.Linear(prev, 1))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x).squeeze(-1)

    dones = np.asarray(dones, dtype=float).reshape(-1)
    q_next_anchor = full_q.predict_q(next_states, np.full(next_states.shape[0], anchor_action, dtype=int))
    pi_next_anchor = np.clip(policy.predict_proba(next_states)[:, anchor_action], min_policy_prob, 1.0)
    targets = (1.0 - dones) * (-alpha * np.log(pi_next_anchor) + q_next_anchor)
    targets = np.clip(targets, -target_clip, target_clip)

    features = np.concatenate([transform(states), one_hot(actions, policy.n_actions)], axis=1)
    x = torch.as_tensor(features, dtype=torch.float32)
    y = torch.as_tensor(targets, dtype=torch.float32)
    model = HNet(features.shape[1], hidden_sizes)
    optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate)
    ds = TensorDataset(x, y)
    dl = DataLoader(ds, batch_size=256, shuffle=True)
    for _ in range(n_epochs):
        for xb, yb in dl:
            pred = model(xb)
            loss = ((pred - yb) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return RewardNetworkEstimate(model, transform, n_actions=policy.n_actions)


def fit_deeppqr_baseline(
    states: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    next_states: np.ndarray,
    dones: np.ndarray,
    n_actions: int,
    gamma: float = 0.95,
    anchor_action: int = 0,
    g_values: np.ndarray | None = None,
    alpha: float = 1.0,
    min_policy_prob: float = 1e-2,
    behavior_policy: EstimatedPolicy | None = None,
) -> DeepPQRResult:
    """Implement a neural DeepPQR baseline in the anchor-action setting."""
    del rewards
    mask = np.asarray(actions) == anchor_action
    if behavior_policy is None:
        behavior_policy = fit_airl_policy(
            states=states,
            actions=actions,
            n_actions=n_actions,
            next_states=next_states,
            dones=dones,
            gamma=gamma,
            n_iters=80,
        )
    anchor_estimate = fit_anchor_q_network(
        states=np.asarray(states)[mask],
        next_states=np.asarray(next_states)[mask],
        dones=np.asarray(dones)[mask],
        policy=behavior_policy,
        anchor_action=anchor_action,
        gamma=gamma,
        g_values=None if g_values is None else np.asarray(g_values)[mask],
        alpha=alpha,
        min_policy_prob=min_policy_prob,
    )
    full_q = DeepPQRFullQEstimate(
        anchor_q=anchor_estimate,
        policy=behavior_policy,
        anchor_action=anchor_action,
        alpha=alpha,
        min_policy_prob=min_policy_prob,
    )
    reward_network = fit_reward_regression_from_q(
        states=np.asarray(states),
        actions=np.asarray(actions),
        next_states=np.asarray(next_states),
        dones=np.asarray(dones),
        full_q=full_q,
        policy=behavior_policy,
        anchor_action=anchor_action,
        gamma=gamma,
        alpha=alpha,
        min_policy_prob=min_policy_prob,
    )
    return DeepPQRResult(
        behavior_policy=behavior_policy,
        anchor_q=anchor_estimate,
        full_q=full_q,
        reward_network=reward_network,
    )
