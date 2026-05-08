from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import tensorflow as tf

from .data import HopperTrajectoryDataset
from .dice import DualDICEConfig
from .fqe import QFitterConfig
from .policies import HopperPicklePolicy


GOOGLE_RESEARCH_ROOT = Path("/tmp/google-research")
if str(GOOGLE_RESEARCH_ROOT) not in sys.path:
    sys.path.insert(0, str(GOOGLE_RESEARCH_ROOT))

from policy_eval.dual_dice import DualDICE as OfficialDualDICE  # noqa: E402
from policy_eval.q_fitter import QFitter as OfficialQFitter  # noqa: E402


@dataclass
class OfficialQFitterResult:
    model: OfficialQFitter
    loss_history: list[float]
    training_metadata: dict[str, float]


@dataclass
class OfficialDualDICEResult:
    model: OfficialDualDICE
    loss_history: list[float]
    pred_ratio: float
    training_metadata: dict[str, float]


class _RandomBatchIterator:
    def __init__(
        self,
        dataset: HopperTrajectoryDataset,
        sample_weights: np.ndarray | None,
        batch_size: int,
        seed: int,
    ) -> None:
        self.dataset = dataset
        self.batch_size = min(batch_size, len(dataset))
        self.rng = np.random.default_rng(seed)
        if sample_weights is None:
            self.weights_np = np.ones(len(dataset), dtype=np.float32)
        else:
            self.weights_np = np.asarray(sample_weights, dtype=np.float32).reshape(-1)
        self.states = tf.convert_to_tensor(dataset.observations, dtype=tf.float32)
        self.actions = tf.convert_to_tensor(dataset.actions, dtype=tf.float32)
        self.next_states = tf.convert_to_tensor(dataset.next_observations, dtype=tf.float32)
        self.rewards = tf.convert_to_tensor(dataset.rewards, dtype=tf.float32)
        self.masks = tf.convert_to_tensor(dataset.masks, dtype=tf.float32)
        self.weights = tf.convert_to_tensor(self.weights_np, dtype=tf.float32)
        self.steps = tf.convert_to_tensor(dataset.steps, dtype=tf.float32)

    def __iter__(self) -> _RandomBatchIterator:
        return self

    def __next__(self) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        indices = self.rng.integers(0, len(self.dataset), size=self.batch_size)
        return (
            tf.gather(self.states, indices),
            tf.gather(self.actions, indices),
            tf.gather(self.next_states, indices),
            tf.gather(self.rewards, indices),
            tf.gather(self.masks, indices),
            tf.gather(self.weights, indices),
            tf.gather(self.steps, indices),
        )


def _random_iterator(
    dataset: HopperTrajectoryDataset,
    sample_weights: np.ndarray | None,
    batch_size: int,
    seed: int,
) -> _RandomBatchIterator:
    return _RandomBatchIterator(dataset, sample_weights, batch_size, seed)


def _tf_dataset_tensors(
    dataset: HopperTrajectoryDataset,
    sample_weights: np.ndarray | None,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    if sample_weights is None:
        weights = np.ones(len(dataset), dtype=np.float32)
    else:
        weights = np.asarray(sample_weights, dtype=np.float32).reshape(-1)
    return (
        tf.convert_to_tensor(dataset.observations, dtype=tf.float32),
        tf.convert_to_tensor(dataset.actions, dtype=tf.float32),
        tf.convert_to_tensor(dataset.next_observations, dtype=tf.float32),
        tf.convert_to_tensor(dataset.rewards, dtype=tf.float32),
        tf.convert_to_tensor(dataset.masks, dtype=tf.float32),
        tf.convert_to_tensor(weights, dtype=tf.float32),
        tf.convert_to_tensor(dataset.steps, dtype=tf.float32),
    )


def _sample_actions(
    policy: HopperPicklePolicy,
    dataset: HopperTrajectoryDataset,
    normalized_states: tf.Tensor,
    *,
    seed: int,
) -> tf.Tensor:
    states_np = dataset.unnormalize_states(np.asarray(normalized_states.numpy(), dtype=np.float32))
    actions = policy.sample_actions(states_np, rng=np.random.default_rng(seed), deterministic=False)
    return tf.convert_to_tensor(actions, dtype=tf.float32)


def train_q_fitter(
    dataset: HopperTrajectoryDataset,
    policy: HopperPicklePolicy,
    *,
    sample_weights: np.ndarray | None = None,
    config: QFitterConfig | None = None,
    seed: int = 0,
) -> OfficialQFitterResult:
    if config is None:
        config = QFitterConfig()

    np.random.seed(seed)
    tf.random.set_seed(seed)
    model = OfficialQFitter(
        dataset.observation_dim,
        dataset.action_dim,
        config.critic_lr,
        config.weight_decay,
        config.tau,
    )

    iterator = _random_iterator(dataset, sample_weights, config.batch_size, seed)
    min_reward = tf.reduce_min(tf.convert_to_tensor(dataset.rewards, dtype=tf.float32))
    max_reward = tf.reduce_max(tf.convert_to_tensor(dataset.rewards, dtype=tf.float32))

    loss_history: list[float] = []
    for step in range(config.num_updates):
        states, actions, next_states, rewards, masks, weights, _ = next(iterator)
        next_actions = _sample_actions(policy, dataset, next_states, seed=seed + step)
        loss = model.update(
            states,
            actions,
            next_states,
            next_actions,
            rewards,
            masks,
            weights,
            config.gamma,
            min_reward,
            max_reward,
        )
        if step % config.log_interval == 0 or step == config.num_updates - 1:
            loss_history.append(float(loss.numpy()))

    return OfficialQFitterResult(
        model=model,
        loss_history=loss_history,
        training_metadata={
            "num_updates": float(config.num_updates),
            "batch_size": float(min(config.batch_size, len(dataset))),
            "gamma": float(config.gamma),
        },
    )


def estimate_policy_return(
    fitter_result: OfficialQFitterResult,
    dataset: HopperTrajectoryDataset,
    policy: HopperPicklePolicy,
    *,
    gamma: float,
    seed: int = 0,
) -> float:
    initial_states = tf.convert_to_tensor(dataset.normalize_states(dataset.initial_observations_raw), dtype=tf.float32)
    initial_actions = _sample_actions(policy, dataset, initial_states, seed=seed)
    initial_weights = tf.convert_to_tensor(dataset.initial_weights, dtype=tf.float32)
    preds = fitter_result.model(initial_states, initial_actions)
    value_scaled = tf.reduce_sum(preds * initial_weights) / tf.reduce_sum(initial_weights)
    value = float(dataset.unnormalize_rewards(value_scaled.numpy()))
    return value / max(1.0 - gamma, 1e-8)


def train_dual_dice(
    dataset: HopperTrajectoryDataset,
    policy: HopperPicklePolicy,
    *,
    config: DualDICEConfig | None = None,
    seed: int = 0,
) -> OfficialDualDICEResult:
    if config is None:
        config = DualDICEConfig()

    np.random.seed(seed)
    tf.random.set_seed(seed)
    model = OfficialDualDICE(dataset.observation_dim, dataset.action_dim, config.weight_decay)

    iterator = _random_iterator(dataset, None, config.batch_size, seed)
    initial_states = tf.convert_to_tensor(dataset.normalize_states(dataset.initial_observations_raw), dtype=tf.float32)
    initial_weights = tf.convert_to_tensor(dataset.initial_weights, dtype=tf.float32)

    loss_history: list[float] = []
    for step in range(config.num_updates):
        states, actions, next_states, _, masks, weights, _ = next(iterator)
        next_actions = _sample_actions(policy, dataset, next_states, seed=seed + 10_000 + step)
        initial_actions = _sample_actions(policy, dataset, initial_states, seed=seed + 20_000 + step)
        loss = model.update(
            initial_states,
            initial_actions,
            initial_weights,
            states,
            actions,
            next_states,
            next_actions,
            masks,
            weights,
            config.gamma,
        )
        if step % config.log_interval == 0 or step == config.num_updates - 1:
            loss_history.append(float(loss.numpy()))

    pred_scaled, pred_ratio = model.estimate_returns(_random_iterator(dataset, None, config.batch_size, seed + 1_000_000), num_samples=100)
    return OfficialDualDICEResult(
        model=model,
        loss_history=loss_history,
        pred_ratio=float(pred_ratio.numpy()),
        training_metadata={
            "num_updates": float(config.num_updates),
            "batch_size": float(min(config.batch_size, len(dataset))),
            "gamma": float(config.gamma),
            "pred_scaled": float(pred_scaled.numpy()),
        },
    )


def estimate_dual_dice_return(
    result: OfficialDualDICEResult,
    dataset: HopperTrajectoryDataset,
    *,
    gamma: float,
    seed: int = 0,
    batch_size: int = 256,
    num_samples: int = 100,
) -> tuple[float, float]:
    pred_scaled, pred_ratio = result.model.estimate_returns(
        _random_iterator(dataset, None, batch_size, seed),
        num_samples=num_samples,
    )
    value = float(dataset.unnormalize_rewards(pred_scaled.numpy()))
    return value / max(1.0 - gamma, 1e-8), float(pred_ratio.numpy())


def extract_dual_dice_weights(
    result: OfficialDualDICEResult,
    dataset: HopperTrajectoryDataset,
) -> np.ndarray:
    weights = result.model.zeta(
        tf.convert_to_tensor(dataset.observations, dtype=tf.float32),
        tf.convert_to_tensor(dataset.actions, dtype=tf.float32),
    )
    return np.asarray(weights.numpy(), dtype=np.float32)
