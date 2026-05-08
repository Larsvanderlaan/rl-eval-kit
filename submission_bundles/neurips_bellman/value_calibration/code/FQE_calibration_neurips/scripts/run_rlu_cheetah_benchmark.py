#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from FQE_calibration_neurips.src.calibration.calibrators import fit_calibrator  # noqa: E402


@dataclass
class ContinuousBatch:
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    discounts: np.ndarray
    next_states: np.ndarray
    episode_ids: np.ndarray
    initial_states: np.ndarray

    def __len__(self) -> int:
        return int(self.rewards.shape[0])

    def subset(self, mask: np.ndarray) -> "ContinuousBatch":
        mask = np.asarray(mask, dtype=bool)
        initial_ids = sorted(set(np.asarray(self.episode_ids[mask], dtype=int).tolist()))
        first = []
        for eid in initial_ids:
            idx = np.flatnonzero(self.episode_ids == eid)
            if idx.size:
                first.append(self.states[int(idx[0])])
        return ContinuousBatch(
            states=self.states[mask],
            actions=self.actions[mask],
            rewards=self.rewards[mask],
            discounts=self.discounts[mask],
            next_states=self.next_states[mask],
            episode_ids=self.episode_ids[mask],
            initial_states=np.asarray(first, dtype=np.float32) if first else self.initial_states[:0],
        )


@dataclass
class Standardizer:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray) -> "Standardizer":
        mean = np.asarray(x, dtype=np.float64).mean(axis=0)
        scale = np.asarray(x, dtype=np.float64).std(axis=0)
        scale = np.where(scale < 1e-6, 1.0, scale)
        return cls(mean=mean, scale=scale)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return ((np.asarray(x, dtype=np.float64) - self.mean) / self.scale).astype(np.float32)


class DeepOPESavedModelPolicy:
    """Thin adapter around official Deep OPE RL Unplugged SavedModel policies."""

    def __init__(
        self,
        policy_id: str,
        policy_root: str | Path | None = None,
        policy_cache_dir: str | Path | None = None,
    ) -> None:
        import tensorflow as tf

        self.policy_id = str(policy_id)
        if policy_cache_dir is not None:
            uri = _ensure_cached_policy(self.policy_id, Path(policy_cache_dir))
        elif policy_root is not None:
            uri = str(Path(policy_root) / self.policy_id)
        else:
            uri = f"gs://gresearch/deep-ope/rlunplugged/{self.policy_id}"
        self.uri = uri
        self.model = tf.saved_model.load(uri)

    def sample_actions(self, states: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
        import tensorflow as tf

        x = np.asarray(states, dtype=np.float32)
        obs = {
            "position": tf.convert_to_tensor(x[:, :8], dtype=tf.float32),
            "velocity": tf.convert_to_tensor(x[:, 8:17], dtype=tf.float32),
        }
        try:
            if hasattr(self.model, "initial_state"):
                out = self.model(obs, ((),))[0]
            else:
                out = self.model(obs)
            actions = _extract_saved_model_action(out)
            if actions.shape == (x.shape[0], 6) and np.isfinite(actions).all():
                return actions.astype(np.float32)
        except Exception:
            pass
        candidates = []
        if hasattr(self.model, "signatures"):
            candidates.extend(self.model.signatures.values())
        if callable(self.model):
            candidates.append(self.model)
        last_error: Exception | None = None
        for fn in candidates:
            payloads = (
                obs,
                (obs, ()),
                (obs, ((),)),
                {"observation": obs},
                ({"observation": obs}, ()),
                ({"observation": obs}, ((),)),
                tf.convert_to_tensor(x, dtype=tf.float32),
            )
            for payload in payloads:
                try:
                    out = fn(*payload) if isinstance(payload, tuple) else fn(payload)
                    return _extract_saved_model_action(out)
                except Exception as exc:  # pragma: no cover - depends on SavedModel signature.
                    last_error = exc
        raise RuntimeError(f"Could not call policy {self.policy_id} from {self.uri}: {last_error}")


def _extract_saved_model_action(out: object) -> np.ndarray:
    if isinstance(out, dict):
        for key in ("action", "actions", "sample", "output_0"):
            if key in out:
                return np.asarray(out[key], dtype=np.float32)
        return np.asarray(next(iter(out.values())), dtype=np.float32)
    if isinstance(out, (tuple, list)):
        return np.asarray(out[0], dtype=np.float32)
    return np.asarray(out, dtype=np.float32)


def _ensure_cached_policy(policy_id: str, policy_cache_dir: Path) -> str:
    local_dir = policy_cache_dir / policy_id
    saved_model = local_dir / "saved_model.pb"
    if saved_model.exists():
        return str(local_dir)
    source = f"gs://gresearch/deep-ope/rlunplugged/{policy_id}"
    try:
        _copy_tf_gfile_tree(source, local_dir)
    except Exception as exc:
        raise RuntimeError(
            f"Policy {policy_id} is not cached at {local_dir} and could not be downloaded from {source}. "
            "Run once with network access or provide --policy_root pointing to local Deep OPE policies."
        ) from exc
    if not saved_model.exists():
        raise RuntimeError(f"Downloaded policy {policy_id} to {local_dir}, but saved_model.pb is missing.")
    return str(local_dir)


def _copy_tf_gfile_tree(source: str, destination: Path) -> None:
    import tensorflow as tf

    destination.mkdir(parents=True, exist_ok=True)
    for entry in tf.io.gfile.listdir(source):
        src = f"{source.rstrip('/')}/{entry}"
        dst = destination / entry
        if tf.io.gfile.isdir(src):
            _copy_tf_gfile_tree(src, dst)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            tf.io.gfile.copy(src, str(dst), overwrite=True)


class SyntheticLinearPolicy:
    def __init__(self, action_dim: int, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.w = rng.normal(scale=0.2, size=(action_dim, 17))
        self.b = rng.normal(scale=0.05, size=action_dim)

    def sample_actions(self, states: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
        x = np.asarray(states, dtype=np.float32)[:, : self.w.shape[1]]
        return np.tanh(x @ self.w.T + self.b).astype(np.float32)


class LinearQuadraticFeatures:
    def __init__(self, x: np.ndarray) -> None:
        self.standardizer = Standardizer.fit(x)

    def transform(self, x: np.ndarray) -> np.ndarray:
        z = self.standardizer.transform(x)
        return np.concatenate([np.ones((z.shape[0], 1), dtype=np.float32), z, z * z], axis=1)


class RandomFourierFeatures:
    def __init__(self, x: np.ndarray, n_components: int = 512, seed: int = 0) -> None:
        self.standardizer = Standardizer.fit(x)
        z = self.standardizer.transform(x)
        rng = np.random.default_rng(seed)
        bandwidth = _median_bandwidth(z, seed=seed)
        self.w = rng.normal(scale=1.0 / max(bandwidth, 1e-6), size=(z.shape[1], int(n_components))).astype(np.float32)
        self.b = rng.uniform(0.0, 2.0 * np.pi, size=int(n_components)).astype(np.float32)

    def transform(self, x: np.ndarray) -> np.ndarray:
        z = self.standardizer.transform(x)
        return (np.sqrt(2.0 / self.w.shape[1]) * np.cos(z @ self.w + self.b)).astype(np.float32)


class NeuralQ(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Sequence[int] = (256, 256)) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for hidden in hidden_dims:
            layers.extend([nn.Linear(prev, int(hidden)), nn.ReLU()])
            prev = int(hidden)
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


@dataclass
class Predictor:
    learner: str
    feature_map: object | None
    theta: np.ndarray | None = None
    model: nn.Module | None = None
    state_action_standardizer: Standardizer | None = None
    prediction_divisor: float = 1.0
    diagnostics: dict[str, float | str] | None = None

    def predict_q(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        x = np.concatenate([states, actions], axis=1)
        if self.model is not None:
            assert self.state_action_standardizer is not None
            z = self.state_action_standardizer.transform(x)
            self.model.eval()
            with torch.no_grad():
                raw = self.model(torch.as_tensor(z, dtype=torch.float32)).cpu().numpy().astype(np.float64)
                return raw / max(float(self.prediction_divisor), 1e-8)
        assert self.feature_map is not None and self.theta is not None
        return np.asarray(self.feature_map.transform(x) @ self.theta, dtype=np.float64)

    def predict_v(self, states: np.ndarray, policy: object, *, action_samples: int, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        preds = []
        for _ in range(max(1, int(action_samples))):
            actions = policy.sample_actions(states, rng)
            preds.append(self.predict_q(states, actions))
        return np.mean(np.vstack(preds), axis=0)


def _median_bandwidth(x: np.ndarray, seed: int = 0, max_points: int = 2000) -> float:
    rng = np.random.default_rng(seed)
    z = np.asarray(x, dtype=np.float64)
    if z.shape[0] > max_points:
        z = z[rng.choice(z.shape[0], size=max_points, replace=False)]
    if z.shape[0] < 2:
        return 1.0
    diffs = z[:, None, :] - z[None, :, :]
    dist = np.sqrt(np.sum(diffs * diffs, axis=-1))
    tri = dist[np.triu_indices(dist.shape[0], k=1)]
    tri = tri[tri > 0]
    return float(np.median(tri)) if tri.size else 1.0


def _ridge_fit(phi: np.ndarray, y: np.ndarray, ridge: float) -> np.ndarray:
    gram = phi.T @ phi + float(ridge) * np.eye(phi.shape[1])
    rhs = phi.T @ y
    try:
        return np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(gram, rhs, rcond=None)[0]


def fit_linear_or_rf_fqe(
    batch: ContinuousBatch,
    policy: object,
    learner: str,
    *,
    gamma: float,
    ridge: float,
    n_iters: int,
    n_components: int,
    action_samples: int,
    seed: int,
    solver: str = "iterated",
) -> Predictor:
    x = np.concatenate([batch.states, batch.actions], axis=1)
    fmap = LinearQuadraticFeatures(x) if learner == "linear_fqe" else RandomFourierFeatures(x, n_components, seed)
    phi = fmap.transform(x)
    rng = np.random.default_rng(seed)
    if solver == "fixed_point":
        next_phi = np.zeros_like(phi, dtype=np.float64)
        for _sample in range(max(1, int(action_samples))):
            next_actions = policy.sample_actions(batch.next_states, rng)
            next_phi += fmap.transform(np.concatenate([batch.next_states, next_actions], axis=1))
        next_phi /= max(1, int(action_samples))
        delta = phi.astype(np.float64) - float(gamma) * batch.discounts[:, None].astype(np.float64) * next_phi
        lhs = phi.T @ delta + float(ridge) * np.eye(phi.shape[1])
        rhs = phi.T @ batch.rewards
        try:
            theta = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            theta = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
        return Predictor(
            learner=learner,
            feature_map=fmap,
            theta=theta,
            diagnostics={"n_iters": float(n_iters), "linear_solver": "fixed_point"},
        )
    theta = np.zeros(phi.shape[1], dtype=np.float64)
    for _ in range(int(n_iters)):
        next_v = np.zeros(len(batch), dtype=np.float64)
        for _sample in range(max(1, int(action_samples))):
            next_actions = policy.sample_actions(batch.next_states, rng)
            next_phi = fmap.transform(np.concatenate([batch.next_states, next_actions], axis=1))
            next_v += next_phi @ theta
        next_v /= max(1, int(action_samples))
        target = batch.rewards + float(gamma) * batch.discounts * next_v
        theta = _ridge_fit(phi, target, ridge)
    return Predictor(learner=learner, feature_map=fmap, theta=theta, diagnostics={"n_iters": float(n_iters), "linear_solver": "iterated"})


def fit_neural_fqe(
    batch: ContinuousBatch,
    policy: object,
    *,
    gamma: float,
    n_outer_iters: int,
    epochs_per_iter: int,
    action_samples: int,
    seed: int,
    num_updates: int | None = None,
    hidden_dims: Sequence[int] = (256, 256),
    batch_size: int = 256,
    lr: float = 3e-4,
    weight_decay: float = 1e-5,
    target_tau: float = 0.005,
    scaled_outputs: bool = True,
    device: str = "cpu",
) -> Predictor:
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    x = np.concatenate([batch.states, batch.actions], axis=1)
    standardizer = Standardizer.fit(x)
    z = standardizer.transform(x)
    model = NeuralQ(z.shape[1], hidden_dims=hidden_dims).to(device)
    target_model = NeuralQ(z.shape[1], hidden_dims=hidden_dims).to(device)
    target_model.load_state_dict(model.state_dict())
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    z_t = torch.as_tensor(z, dtype=torch.float32, device=device)
    rewards_t = torch.as_tensor(batch.rewards, dtype=torch.float32, device=device)
    discounts_t = torch.as_tensor(batch.discounts, dtype=torch.float32, device=device)
    output_scale = max(1.0 - float(gamma), 1e-8) if scaled_outputs and float(gamma) < 1.0 else 1.0
    min_target = float(output_scale) * float(np.min(batch.rewards)) / max(1.0 - float(gamma), 1e-8)
    max_target = float(output_scale) * float(np.max(batch.rewards)) / max(1.0 - float(gamma), 1e-8)
    n = len(batch.rewards)
    if num_updates is None:
        num_updates = max(1, int(n_outer_iters)) * max(1, int(epochs_per_iter)) * max(1, int(np.ceil(n / int(batch_size))))
    losses: list[float] = []
    batch_size = min(int(batch_size), n)
    for step in range(int(num_updates)):
        idx = rng.integers(0, n, size=batch_size)
        idx_t = torch.as_tensor(idx, dtype=torch.long, device=device)
        with torch.no_grad():
            next_v = torch.zeros(batch_size, dtype=torch.float32, device=device)
            for _sample in range(max(1, int(action_samples))):
                next_actions = policy.sample_actions(batch.next_states[idx], rng)
                next_z = standardizer.transform(np.concatenate([batch.next_states[idx], next_actions], axis=1))
                next_v += target_model(torch.as_tensor(next_z, dtype=torch.float32, device=device))
            next_v = next_v / max(1, int(action_samples))
            targets = float(output_scale) * rewards_t[idx_t] + float(gamma) * discounts_t[idx_t] * next_v
            targets = torch.clamp(targets, min=min_target, max=max_target)
        pred = model(z_t[idx_t])
        loss = torch.mean((pred - targets) ** 2)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        with torch.no_grad():
            for tp, p in zip(target_model.parameters(), model.parameters()):
                tp.data.mul_(1.0 - float(target_tau))
                tp.data.add_(float(target_tau) * p.data)
        if step == 0 or step == int(num_updates) - 1 or (step + 1) % max(1, int(num_updates) // 10) == 0:
            losses.append(float(loss.item()))
    return Predictor(
        learner="neural_fqe",
        feature_map=None,
        model=target_model.cpu(),
        state_action_standardizer=standardizer,
        prediction_divisor=float(output_scale),
        diagnostics={
            "n_iters": float(n_outer_iters),
            "neural_num_updates": float(num_updates),
            "neural_target_tau": float(target_tau),
            "neural_output_scale": float(output_scale),
            "neural_loss_first": float(losses[0]) if losses else float("nan"),
            "neural_loss_last": float(losses[-1]) if losses else float("nan"),
            "neural_value_scale": float(1.0 / max(float(output_scale), 1e-8)),
        },
    )


def fit_rkhs_minimax_fqe(
    batch: ContinuousBatch,
    policy: object,
    *,
    gamma: float,
    ridge: float,
    n_components: int,
    n_iters: int,
    action_samples: int,
    seed: int,
    critic_anchors: int = 512,
    critic_bandwidth_scale: float = 1.0,
) -> Predictor:
    # Fixed RF value model with an RKHS/Nystrom critic penalty. This is a stable
    # least-squares minimax Bellman-error estimator: min_theta ||K^{1/2}(Phi theta
    # - r - gamma Phi' theta)||^2 + ridge ||theta||^2.
    x = np.concatenate([batch.states, batch.actions], axis=1)
    fmap = RandomFourierFeatures(x, n_components=n_components, seed=seed)
    phi = fmap.transform(x).astype(np.float64)
    rng = np.random.default_rng(seed)
    next_phi_avg = np.zeros_like(phi)
    for _sample in range(max(1, int(action_samples))):
        next_actions = policy.sample_actions(batch.next_states, rng)
        next_phi_avg += fmap.transform(np.concatenate([batch.next_states, next_actions], axis=1))
    next_phi_avg /= max(1, int(action_samples))
    delta = phi - float(gamma) * batch.discounts[:, None] * next_phi_avg
    anchors = _select_anchor_rows(delta, max_anchors=min(int(critic_anchors), max(8, delta.shape[0] // 2)), seed=seed)
    bw = _median_bandwidth(delta, seed=seed) * float(critic_bandwidth_scale)
    k = np.exp(-_sqdist(delta, anchors) / (2.0 * bw * bw + 1e-8))
    lhs = delta.T @ k @ k.T @ delta + float(ridge) * np.eye(phi.shape[1])
    rhs = delta.T @ k @ k.T @ batch.rewards
    try:
        theta = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        theta = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
    return Predictor(
        learner="rkhs_minimax_fqe",
        feature_map=fmap,
        theta=theta,
        diagnostics={
            "n_iters": float(n_iters),
            "rkhs_anchors": float(anchors.shape[0]),
            "rkhs_bandwidth": float(bw),
            "rkhs_bandwidth_scale": float(critic_bandwidth_scale),
        },
    )


def _sqdist(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.sum(x * x, axis=1, keepdims=True) + np.sum(y * y, axis=1)[None, :] - 2.0 * x @ y.T


def _select_anchor_rows(x: np.ndarray, max_anchors: int, seed: int) -> np.ndarray:
    if x.shape[0] <= max_anchors:
        return x
    rng = np.random.default_rng(seed)
    return x[rng.choice(x.shape[0], size=int(max_anchors), replace=False)]


def load_rlu_control_suite_npz(cache_path: str | Path) -> ContinuousBatch:
    data = np.load(cache_path)
    return ContinuousBatch(
        states=data["states"].astype(np.float32),
        actions=data["actions"].astype(np.float32),
        rewards=data["rewards"].astype(np.float32),
        discounts=data["discounts"].astype(np.float32),
        next_states=data["next_states"].astype(np.float32),
        episode_ids=data["episode_ids"].astype(np.int64),
        initial_states=data["initial_states"].astype(np.float32),
    )


def append_time_feature(batch: ContinuousBatch, horizon: int = 1000) -> ContinuousBatch:
    time = np.zeros(len(batch), dtype=np.float32)
    next_time = np.zeros(len(batch), dtype=np.float32)
    for eid in np.unique(batch.episode_ids):
        idx = np.flatnonzero(batch.episode_ids == eid)
        order = np.arange(idx.size, dtype=np.float32)
        time[idx] = order / float(horizon)
        next_time[idx] = np.minimum(order + 1.0, float(horizon)) / float(horizon)
    initial_time = np.zeros((batch.initial_states.shape[0], 1), dtype=np.float32)
    return ContinuousBatch(
        states=np.concatenate([batch.states, time[:, None]], axis=1).astype(np.float32),
        actions=batch.actions,
        rewards=batch.rewards,
        discounts=batch.discounts,
        next_states=np.concatenate([batch.next_states, next_time[:, None]], axis=1).astype(np.float32),
        episode_ids=batch.episode_ids,
        initial_states=np.concatenate([batch.initial_states, initial_time], axis=1).astype(np.float32),
    )


def build_rlu_control_suite_cache(
    task: str,
    cache_path: str | Path,
    *,
    max_episodes: int | None = None,
    tfds_data_dir: str | Path | None = None,
) -> Path:
    import tensorflow_datasets as tfds

    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    ds = tfds.load(
        f"rlu_control_suite/{task}",
        split="train",
        shuffle_files=False,
        data_dir=str(tfds_data_dir) if tfds_data_dir is not None else None,
    )
    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    rewards: list[float] = []
    discounts: list[float] = []
    next_states: list[np.ndarray] = []
    episode_ids: list[int] = []
    initial_states: list[np.ndarray] = []
    for ep_idx, episode in enumerate(tfds.as_numpy(ds)):
        if max_episodes is not None and ep_idx >= int(max_episodes):
            break
        steps = list(episode["steps"])
        if len(steps) < 2:
            continue
        initial_states.append(_flatten_obs(steps[0]["observation"]))
        for t in range(len(steps) - 1):
            s = _flatten_obs(steps[t]["observation"])
            sp = _flatten_obs(steps[t + 1]["observation"])
            states.append(s)
            actions.append(np.asarray(steps[t]["action"], dtype=np.float32))
            rewards.append(float(steps[t]["reward"]))
            discounts.append(float(steps[t]["discount"]))
            next_states.append(sp)
            episode_ids.append(int(ep_idx))
    np.savez_compressed(
        cache_path,
        states=np.asarray(states, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.float32),
        rewards=np.asarray(rewards, dtype=np.float32),
        discounts=np.asarray(discounts, dtype=np.float32),
        next_states=np.asarray(next_states, dtype=np.float32),
        episode_ids=np.asarray(episode_ids, dtype=np.int64),
        initial_states=np.asarray(initial_states, dtype=np.float32),
    )
    return cache_path


def _flatten_obs(obs: dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate([np.asarray(obs["position"], dtype=np.float32), np.asarray(obs["velocity"], dtype=np.float32)])


def make_synthetic_batch(n_episodes: int = 12, horizon: int = 20, seed: int = 0) -> ContinuousBatch:
    rng = np.random.default_rng(seed)
    states = []
    actions = []
    rewards = []
    discounts = []
    next_states = []
    episode_ids = []
    initial_states = []
    for ep in range(n_episodes):
        s = rng.normal(size=17).astype(np.float32)
        initial_states.append(s.copy())
        for _ in range(horizon):
            a = np.tanh(0.15 * s[:6] + rng.normal(scale=0.2, size=6)).astype(np.float32)
            sp = (0.95 * s + 0.03 * rng.normal(size=17)).astype(np.float32)
            r = float(0.3 * s[0] + 0.2 * s[8] - 0.05 * np.sum(a * a))
            states.append(s.copy())
            actions.append(a.copy())
            rewards.append(r)
            discounts.append(0.99)
            next_states.append(sp.copy())
            episode_ids.append(ep)
            s = sp
    return ContinuousBatch(
        states=np.asarray(states, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.float32),
        rewards=np.asarray(rewards, dtype=np.float32),
        discounts=np.asarray(discounts, dtype=np.float32),
        next_states=np.asarray(next_states, dtype=np.float32),
        episode_ids=np.asarray(episode_ids, dtype=np.int64),
        initial_states=np.asarray(initial_states, dtype=np.float32),
    )


def split_by_episode(batch: ContinuousBatch, seed: int, fractions: tuple[float, float, float] = (0.6, 0.2, 0.2)) -> tuple[ContinuousBatch, ContinuousBatch, ContinuousBatch]:
    rng = np.random.default_rng(seed)
    ids = np.unique(batch.episode_ids)
    rng.shuffle(ids)
    n_train = max(1, int(round(fractions[0] * len(ids))))
    n_cal = max(1, int(round(fractions[1] * len(ids))))
    train_ids = set(ids[:n_train])
    cal_ids = set(ids[n_train : n_train + n_cal])
    diag_ids = set(ids[n_train + n_cal :])
    if not diag_ids:
        diag_ids = set(ids[-1:])
        train_ids.discard(ids[-1])
    return (
        batch.subset(np.isin(batch.episode_ids, list(train_ids))),
        batch.subset(np.isin(batch.episode_ids, list(cal_ids))),
        batch.subset(np.isin(batch.episode_ids, list(diag_ids))),
    )


def subsample_transitions(batch: ContinuousBatch, max_transitions: int | None, seed: int) -> ContinuousBatch:
    if max_transitions is None or len(batch) <= int(max_transitions):
        return batch
    rng = np.random.default_rng(seed)
    keep = rng.choice(len(batch), size=int(max_transitions), replace=False)
    mask = np.zeros(len(batch), dtype=bool)
    mask[keep] = True
    return batch.subset(mask)


def bellman_calibrate(predictor: Predictor, calibrator: str, batch: ContinuousBatch, policy: object, gamma: float, seed: int, action_samples: int):
    raw = predictor.predict_v(batch.states, policy, action_samples=action_samples, seed=seed)
    nxt = predictor.predict_v(batch.next_states, policy, action_samples=action_samples, seed=seed + 1)
    target = batch.rewards + float(gamma) * batch.discounts * nxt
    return fit_calibrator(calibrator, raw, target)


def evaluate_prediction(
    predictor: Predictor,
    calibrator_obj: object | None,
    batch: ContinuousBatch,
    policy: object,
    *,
    gamma: float,
    seed: int,
    action_samples: int,
) -> dict[str, float]:
    pred = predictor.predict_v(batch.states, policy, action_samples=action_samples, seed=seed)
    nxt = predictor.predict_v(batch.next_states, policy, action_samples=action_samples, seed=seed + 1)
    if calibrator_obj is not None:
        pred = calibrator_obj.predict(pred)
        nxt = calibrator_obj.predict(nxt)
    outcome = batch.rewards + float(gamma) * batch.discounts * nxt
    initial_values = predictor.predict_v(batch.initial_states, policy, action_samples=action_samples, seed=seed + 2)
    if calibrator_obj is not None:
        initial_values = calibrator_obj.predict(initial_values)
    return {
        "bellman_outcome_mse": float(np.mean((pred - outcome) ** 2)),
        "bellman_calibration_error": _calibration_error(pred, outcome),
        "initial_value_estimate": float(np.mean(initial_values)),
    }


def coverage_diagnostics(
    train: ContinuousBatch,
    batch: ContinuousBatch,
    policy: object,
    *,
    seed: int,
    max_train: int = 2000,
    max_eval: int = 1000,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    target_actions = policy.sample_actions(batch.states, rng)
    train_x = np.concatenate([train.states, train.actions], axis=1)
    eval_x = np.concatenate([batch.states, target_actions], axis=1)
    if train_x.shape[0] > max_train:
        train_x = train_x[rng.choice(train_x.shape[0], size=max_train, replace=False)]
    if eval_x.shape[0] > max_eval:
        eval_x = eval_x[rng.choice(eval_x.shape[0], size=max_eval, replace=False)]
    standardizer = Standardizer.fit(train_x)
    train_z = standardizer.transform(train_x).astype(np.float64)
    eval_z = standardizer.transform(eval_x).astype(np.float64)
    min_d2 = []
    for start in range(0, eval_z.shape[0], 256):
        block = eval_z[start : start + 256]
        min_d2.append(np.min(_sqdist(block, train_z), axis=1))
    d = np.sqrt(np.maximum(np.concatenate(min_d2), 0.0))
    bw = _median_bandwidth(train_z, seed=seed, max_points=min(1000, train_z.shape[0]))
    weights = np.exp(-(d * d) / (2.0 * bw * bw + 1e-8))
    ess = (weights.sum() ** 2) / (np.sum(weights * weights) + 1e-12)
    normalized_ess = ess / max(1, weights.size)
    return {
        "mean_target_action_distance": float(np.mean(d)),
        "p90_target_action_distance": float(np.quantile(d, 0.9)),
        "normalized_coverage_ess": float(normalized_ess),
    }


def _calibration_error(pred: np.ndarray, outcome: np.ndarray, n_bins: int = 20) -> float:
    pred = np.asarray(pred, dtype=float)
    outcome = np.asarray(outcome, dtype=float)
    if pred.size < 2:
        return float("nan")
    edges = np.unique(np.quantile(pred, np.linspace(0, 1, min(n_bins, max(2, pred.size // 10)) + 1)))
    if edges.size < 3:
        return float(np.mean((pred - outcome) ** 2))
    bins = np.searchsorted(edges[1:-1], pred, side="right")
    total = 0.0
    for b in np.unique(bins):
        idx = bins == b
        total += idx.mean() * float((pred[idx].mean() - outcome[idx].mean()) ** 2)
    return float(total)


def _write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    keys: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _load_policy_ids(benchmark_dir: Path, task: str, stage: str, policy_indices: Sequence[int] | None = None) -> list[str]:
    policies = pickle.load((benchmark_dir / "rlunplugged_policys.pkl").open("rb"))
    ids = list(policies[task])
    if policy_indices:
        return [ids[int(i)] for i in policy_indices]
    if stage == "screen":
        idx = np.linspace(0, len(ids) - 1, min(12, len(ids)), dtype=int)
        return [ids[int(i)] for i in idx]
    if stage == "smoke":
        return [ids[0], ids[min(len(ids) - 1, 8)]]
    return ids


def _load_gt(benchmark_dir: Path) -> dict[str, float]:
    gt = pickle.load((benchmark_dir / "rlunplugged_gt.pkl").open("rb"))
    return {k: float(v[0]) for k, v in gt.items()}


def run_benchmark(args: argparse.Namespace) -> dict[str, Path]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache = Path(args.cache_path)
    if args.synthetic:
        full_batch = make_synthetic_batch(seed=args.seed)
    else:
        if not cache.exists():
            build_rlu_control_suite_cache(
                args.task,
                cache,
                max_episodes=args.max_cache_episodes,
                tfds_data_dir=args.tfds_data_dir,
            )
        full_batch = load_rlu_control_suite_npz(cache)
        if args.include_time_feature:
            full_batch = append_time_feature(full_batch, horizon=args.episode_horizon)
    policy_ids = _load_policy_ids(Path(args.benchmark_dir), args.task, args.stage, args.policy_indices)
    gt = _load_gt(Path(args.benchmark_dir))
    rows: list[dict[str, object]] = []
    learner_names = args.learners
    policies: dict[str, object] = {}
    for seed in args.seeds:
        train, cal, diag = split_by_episode(full_batch, seed=int(seed))
        train = subsample_transitions(train, args.max_transitions_per_split, int(seed) + 11)
        cal = subsample_transitions(cal, args.max_transitions_per_split, int(seed) + 12)
        diag = subsample_transitions(diag, args.max_transitions_per_split, int(seed) + 13)
        for policy_id in policy_ids:
            print(f"[rlu-cheetah] seed={seed} policy={policy_id}", flush=True)
            if args.synthetic:
                policy = SyntheticLinearPolicy(train.actions.shape[1], seed=int(seed))
            else:
                if policy_id not in policies:
                    policies[policy_id] = DeepOPESavedModelPolicy(policy_id, args.policy_root, args.policy_cache_dir)
                policy = policies[policy_id]
            coverage = coverage_diagnostics(train, diag, policy, seed=int(seed) + 3000)
            for learner in learner_names:
                print(f"[rlu-cheetah] fitting learner={learner}", flush=True)
                predictor = _fit_learner(learner, train, policy, args, int(seed))
                for calibrator in ["none", "linear", "isotonic"]:
                    cal_obj = None if calibrator == "none" else bellman_calibrate(
                        predictor, calibrator, cal, policy, args.gamma, int(seed) + 1000, args.action_samples
                    )
                    metrics = evaluate_prediction(
                        predictor, cal_obj, diag, policy, gamma=args.gamma, seed=int(seed) + 2000, action_samples=args.action_samples
                    )
                    estimate = metrics.pop("initial_value_estimate")
                    truth = gt.get(policy_id, float("nan"))
                    rows.append(
                        {
                            "task": args.task,
                            "stage": args.stage,
                            "seed": int(seed),
                            "policy_id": policy_id,
                            "learner": learner,
                            "calibrator": calibrator,
                            "method": f"{learner}_{calibrator}",
                            "n_train_transitions": len(train.rewards),
                            "n_calibration_transitions": len(cal.rewards),
                            "n_diagnostic_transitions": len(diag.rewards),
                            "train_data_provenance": f"rlu_control_suite/{args.task};episode_split_seed={seed};train",
                            "calibration_data_provenance": "none" if calibrator == "none" else f"rlu_control_suite/{args.task};episode_split_seed={seed};calibration",
                            "evaluation_data_provenance": f"rlu_control_suite/{args.task};episode_split_seed={seed};diagnostic",
                            "estimated_return": estimate,
                            "ground_truth_return": truth,
                            "absolute_ope_error": abs(estimate - truth) if math.isfinite(truth) else float("nan"),
                            **metrics,
                            **coverage,
                            **(predictor.diagnostics or {}),
                        }
                    )
    raw = output_dir / "rlu_cheetah_results.csv"
    summary = output_dir / "rlu_cheetah_summary.csv"
    audit = output_dir / "rlu_cheetah_audit.csv"
    config = output_dir / "rlu_cheetah_config.json"
    _write_csv(rows, raw)
    _write_csv(_summary(rows), summary)
    _write_csv(_audit(rows), audit)
    config.write_text(json.dumps(vars(args), indent=2, default=str))
    _plot(_summary(rows), output_dir)
    return {"raw": raw, "summary": summary, "audit": audit, "config": config}


def _fit_learner(learner: str, train: ContinuousBatch, policy: object, args: argparse.Namespace, seed: int) -> Predictor:
    if learner == "linear_fqe":
        return fit_linear_or_rf_fqe(train, policy, learner, gamma=args.gamma, ridge=1e-3, n_iters=args.fqe_iters, n_components=0, action_samples=args.action_samples, seed=seed, solver=args.linear_solver)
    if learner == "rf_fqe":
        return fit_linear_or_rf_fqe(train, policy, learner, gamma=args.gamma, ridge=1e-3, n_iters=args.fqe_iters, n_components=args.rf_components, action_samples=args.action_samples, seed=seed, solver=args.linear_solver)
    if learner == "neural_fqe":
        return fit_neural_fqe(
            train,
            policy,
            gamma=args.gamma,
            n_outer_iters=args.neural_iters,
            epochs_per_iter=args.neural_epochs,
            num_updates=args.neural_num_updates,
            action_samples=args.action_samples,
            seed=seed,
            target_tau=args.neural_target_tau,
            scaled_outputs=args.neural_scaled_outputs,
            device=args.device,
        )
    if learner == "rkhs_minimax_fqe":
        return fit_rkhs_minimax_fqe(
            train,
            policy,
            gamma=args.gamma,
            ridge=1e-4,
            n_components=args.rf_components,
            n_iters=args.fqe_iters,
            action_samples=args.action_samples,
            seed=seed,
            critic_anchors=args.rkhs_critic_anchors,
            critic_bandwidth_scale=args.rkhs_critic_bandwidth_scale,
        )
    raise ValueError(f"Unknown learner '{learner}'")


def _summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    df = pd.DataFrame(rows)
    if df.empty:
        return []
    out = []
    for (learner, calibrator), group in df.groupby(["learner", "calibrator"]):
        base = df[(df["learner"].eq(learner)) & (df["calibrator"].eq("none"))]
        cal = pd.to_numeric(group["bellman_calibration_error"], errors="coerce")
        ope = pd.to_numeric(group["absolute_ope_error"], errors="coerce")
        base_cal = pd.to_numeric(base["bellman_calibration_error"], errors="coerce")
        base_ope = pd.to_numeric(base["absolute_ope_error"], errors="coerce")
        out.append(
            {
                "learner": learner,
                "calibrator": calibrator,
                "n_rows": int(len(group)),
                "n_policies": int(group["policy_id"].nunique()),
                "n_seeds": int(group["seed"].nunique()),
                "mean_bellman_calibration_error": float(cal.mean()),
                "mean_absolute_ope_error": float(ope.mean()),
                "relative_bellman_calibration_error": float(cal.mean() / base_cal.mean()) if base_cal.mean() > 0 else float("nan"),
                "relative_absolute_ope_error": float(ope.mean() / base_ope.mean()) if base_ope.mean() > 0 else float("nan"),
                "bellman_calibration_win_rate": _matched_win_rate(df, learner, calibrator, "bellman_calibration_error"),
                "absolute_ope_win_rate": _matched_win_rate(df, learner, calibrator, "absolute_ope_error"),
                "raw_policy_spearman": _raw_policy_spearman(df, learner),
                "mean_normalized_coverage_ess": float(pd.to_numeric(group["normalized_coverage_ess"], errors="coerce").mean()),
                "mean_target_action_distance": float(pd.to_numeric(group["mean_target_action_distance"], errors="coerce").mean()),
            }
        )
    return out


def _matched_win_rate(df: pd.DataFrame, learner: str, calibrator: str, metric: str) -> float:
    if calibrator == "none":
        return float("nan")
    raw = df[(df["learner"].eq(learner)) & (df["calibrator"].eq("none"))]
    calibrated = df[(df["learner"].eq(learner)) & (df["calibrator"].eq(calibrator))]
    merged = calibrated.merge(
        raw[["seed", "policy_id", metric]],
        on=["seed", "policy_id"],
        suffixes=("", "_raw"),
        how="inner",
    )
    if merged.empty:
        return float("nan")
    left = pd.to_numeric(merged[metric], errors="coerce")
    right = pd.to_numeric(merged[f"{metric}_raw"], errors="coerce")
    finite = np.isfinite(left) & np.isfinite(right)
    if not finite.any():
        return float("nan")
    return float((left[finite] < right[finite]).mean())


def _safe_spearman(x: Iterable[float], y: Iterable[float]) -> float:
    frame = pd.DataFrame({"x": list(x), "y": list(y)}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(frame) < 2 or frame["x"].nunique() < 2 or frame["y"].nunique() < 2:
        return float("nan")
    return float(frame["x"].rank().corr(frame["y"].rank()))


def _raw_policy_spearman(df: pd.DataFrame, learner: str) -> float:
    raw = df[(df["learner"].eq(learner)) & (df["calibrator"].eq("none"))]
    vals = []
    for _seed, group in raw.groupby("seed"):
        vals.append(_safe_spearman(group["estimated_return"], group["ground_truth_return"]))
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def _audit(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summary = _summary(rows)
    out = []
    for row in summary:
        reasons = []
        if row["calibrator"] != "none":
            if row["relative_bellman_calibration_error"] >= 0.85:
                reasons.append("calibration_error_gate_failed")
            if row["relative_absolute_ope_error"] >= 0.90:
                reasons.append("ope_error_gate_failed")
            if row.get("bellman_calibration_win_rate", 0.0) < 0.60:
                reasons.append("calibration_win_rate_gate_failed")
            if row.get("absolute_ope_win_rate", 0.0) < 0.60:
                reasons.append("ope_win_rate_gate_failed")
            if not (row.get("raw_policy_spearman", float("nan")) > 0.3):
                reasons.append("raw_rank_gate_failed")
            if row.get("mean_normalized_coverage_ess", 0.0) < 0.02:
                reasons.append("coverage_ess_gate_failed")
            if row["n_policies"] < 3:
                reasons.append("too_few_policies")
            if row["n_seeds"] < 2:
                reasons.append("too_few_seeds")
            label = "promote_main" if not reasons else "not_promoted"
        else:
            label = "raw_baseline"
        out.append({**row, "audit_label": label, "failure_reasons": ";".join(reasons)})
    return out


def _plot(summary_rows: list[dict[str, object]], output_dir: Path) -> None:
    if not summary_rows:
        return
    df = pd.DataFrame(summary_rows)
    df = df[df["calibrator"].ne("none")]
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(7.0, 2.8), constrained_layout=True)
    labels = df["learner"] + " / " + df["calibrator"]
    ax.barh(labels, df["relative_bellman_calibration_error"])
    ax.axvline(1.0, color="0.35", linewidth=0.8)
    ax.set_xlabel("relative Bellman CAL")
    ax.set_title("RL Unplugged cheetah_run Bellman calibration", loc="left", fontweight="bold")
    fig.savefig(output_dir / "rlu_cheetah_calibration.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "rlu_cheetah_calibration.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RL Unplugged cheetah_run Bellman calibration benchmark.")
    parser.add_argument("--stage", choices=["smoke", "screen", "confirm", "final"], default="smoke")
    parser.add_argument("--task", default="cheetah_run")
    parser.add_argument("--benchmark_dir", default="hopper_fqe_benchmark/artifacts/benchmark/dope")
    parser.add_argument("--cache_path", default="FQE_calibration_neurips/results/rlu_cache/cheetah_run.npz")
    parser.add_argument("--tfds_data_dir", default="FQE_calibration_neurips/results/tfds")
    parser.add_argument("--policy_root", default=None)
    parser.add_argument("--policy_cache_dir", default=None)
    parser.add_argument("--output_dir", default="FQE_calibration_neurips/results/rlu_cheetah_smoke")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--max_cache_episodes", type=int, default=None)
    parser.add_argument("--max_transitions_per_split", type=int, default=None)
    parser.add_argument("--policy_indices", nargs="+", type=int, default=None)
    parser.add_argument("--include_time_feature", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--episode_horizon", type=int, default=1000)
    parser.add_argument("--learners", nargs="+", default=["linear_fqe", "rf_fqe", "neural_fqe"])
    parser.add_argument("--fqe_iters", type=int, default=50)
    parser.add_argument("--linear_solver", choices=["iterated", "fixed_point"], default="fixed_point")
    parser.add_argument("--rf_components", type=int, default=512)
    parser.add_argument("--rkhs_critic_anchors", type=int, default=96)
    parser.add_argument("--rkhs_critic_bandwidth_scale", type=float, default=1.5)
    parser.add_argument("--neural_iters", type=int, default=30)
    parser.add_argument("--neural_epochs", type=int, default=3)
    parser.add_argument("--neural_num_updates", type=int, default=None)
    parser.add_argument("--neural_target_tau", type=float, default=0.005)
    parser.add_argument("--neural_scaled_outputs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--action_samples", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    default_output_dir = parser.get_default("output_dir")
    args = parser.parse_args()
    if args.stage == "screen":
        if args.seeds is None:
            args.seeds = [0, 1, 2]
        if args.policy_indices is None:
            args.policy_indices = list(range(8))
        if args.neural_num_updates is None:
            args.neural_num_updates = 20_000
        args.gamma = 0.995 if args.gamma is None else args.gamma
        if args.output_dir == default_output_dir:
            args.output_dir = "FQE_calibration_neurips/results/rlu_cheetah_screen"
    elif args.stage == "confirm":
        if args.seeds is None:
            args.seeds = [10, 11, 12]
        args.learners = args.learners or ["neural_fqe", "rkhs_minimax_fqe"]
        if args.policy_indices is None:
            args.policy_indices = list(range(8))
        args.fqe_iters = max(args.fqe_iters, 100)
        if args.neural_num_updates is None:
            args.neural_num_updates = 100_000
        if args.output_dir == default_output_dir:
            args.output_dir = "FQE_calibration_neurips/results/rlu_cheetah_confirm"
    elif args.stage == "final":
        if args.seeds is None:
            args.seeds = [20, 21, 22, 23, 24]
        args.learners = args.learners or ["neural_fqe", "rkhs_minimax_fqe"]
        if args.policy_indices is None:
            args.policy_indices = list(range(8))
        args.fqe_iters = max(args.fqe_iters, 150)
        if args.neural_num_updates is None:
            args.neural_num_updates = 200_000
        if args.output_dir == default_output_dir:
            args.output_dir = "FQE_calibration_neurips/results/rlu_cheetah_final"
    if args.seeds is None:
        args.seeds = [0]
    return args


def main() -> None:
    for name, path in run_benchmark(parse_args()).items():
        print(f"Wrote {name}: {path}")


if __name__ == "__main__":
    main()
