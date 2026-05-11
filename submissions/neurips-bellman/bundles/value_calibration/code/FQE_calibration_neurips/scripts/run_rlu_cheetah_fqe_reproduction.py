#!/usr/bin/env python
"""Train a Deep OPE-style FQE critic on RL Unplugged cheetah_run.

This runner is intentionally separate from the policy-level score calibration
screen. It asks whether a model trained with the public Google Research
`policy_eval` FQE architecture reproduces the official FQE-L2 scalar policy
values, then applies held-out Bellman calibration to the learned value function.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from FQE_calibration_neurips.scripts.run_rlu_cheetah_benchmark import (  # noqa: E402
    DeepOPESavedModelPolicy,
    ContinuousBatch,
    split_by_episode,
)
from FQE_calibration_neurips.src.calibration.calibrators import fit_calibrator  # noqa: E402


@dataclass
class OfficialRLDBatch:
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    discounts: np.ndarray
    next_states: np.ndarray
    episode_ids: np.ndarray
    initial_states: np.ndarray
    step_indices: np.ndarray
    is_first: np.ndarray
    is_last: np.ndarray
    is_terminal: np.ndarray
    next_is_last: np.ndarray
    current_step_rewards: np.ndarray
    next_step_rewards: np.ndarray
    episode_return_sums: np.ndarray
    transition_return_sums: np.ndarray
    metadata: dict[str, object]

    def to_continuous(self) -> ContinuousBatch:
        return ContinuousBatch(
            states=self.states,
            actions=self.actions,
            rewards=self.rewards,
            discounts=self.discounts,
            next_states=self.next_states,
            episode_ids=self.episode_ids,
            initial_states=self.initial_states,
        )


@dataclass
class Normalizer:
    state_mean: np.ndarray
    state_std: np.ndarray
    reward_mean: float
    reward_std: float
    normalize_rewards: bool = True

    @classmethod
    def fit(
        cls,
        states: np.ndarray,
        rewards: np.ndarray,
        discounts: np.ndarray,
        eps: float = 1e-5,
        normalize_rewards: bool = True,
    ) -> "Normalizer":
        state_std = np.std(states, axis=0)
        if normalize_rewards:
            reward_mean = float(np.mean(rewards))
            if np.min(discounts) == 0.0:
                reward_mean = 0.0
            reward_std = float(np.std(rewards))
        else:
            reward_mean = 0.0
            reward_std = 1.0
        return cls(
            state_mean=np.mean(states, axis=0).astype(np.float32),
            state_std=np.maximum(state_std, eps).astype(np.float32),
            reward_mean=reward_mean,
            reward_std=max(reward_std, eps),
            normalize_rewards=normalize_rewards,
        )

    def norm_states(self, states: np.ndarray) -> np.ndarray:
        return ((np.asarray(states, dtype=np.float32) - self.state_mean) / self.state_std).astype(np.float32)

    def norm_rewards(self, rewards: np.ndarray) -> np.ndarray:
        return ((np.asarray(rewards, dtype=np.float32) - self.reward_mean) / self.reward_std).astype(np.float32)

    def unnorm_scaled_return(self, value: np.ndarray, gamma: float) -> np.ndarray:
        raw_per_step = np.asarray(value, dtype=np.float64) * self.reward_std + self.reward_mean
        return raw_per_step / max(1.0 - float(gamma), 1e-8)

    def unnorm_policy_eval_score(self, value: np.ndarray) -> np.ndarray:
        return np.asarray(value, dtype=np.float64) * self.reward_std + self.reward_mean


def _load_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def _load_policy_ids(benchmark_dir: Path, task: str, indices: list[int]) -> list[str]:
    policies = _load_pickle(benchmark_dir / "rlunplugged_policys.pkl")
    ids = list(policies[task])
    return [ids[int(i)] for i in indices]


def _flatten_obs(obs: dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(obs["position"], dtype=np.float32).reshape(-1),
            np.asarray(obs["velocity"], dtype=np.float32).reshape(-1),
        ]
    ).astype(np.float32)


def _build_official_transitions_from_episodes(
    episodes: Iterable[dict[str, object]],
    *,
    task: str,
    max_episodes: int | None,
) -> OfficialRLDBatch:
    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    rewards: list[float] = []
    discounts: list[float] = []
    next_states: list[np.ndarray] = []
    episode_ids: list[int] = []
    initial_states: list[np.ndarray] = []
    step_indices: list[int] = []
    is_first: list[bool] = []
    is_last: list[bool] = []
    is_terminal: list[bool] = []
    next_is_last: list[bool] = []
    current_step_rewards: list[float] = []
    next_step_rewards: list[float] = []
    episode_return_sums: list[float] = []
    transition_return_sums: list[float] = []

    kept_episodes = 0
    for ep_idx, episode in enumerate(episodes):
        if max_episodes is not None and kept_episodes >= int(max_episodes):
            break
        steps = list(episode["steps"])  # type: ignore[index]
        if len(steps) < 2:
            continue
        initial_states.append(_flatten_obs(steps[0]["observation"]))
        episode_reward_sum = float(np.sum([float(step["reward"]) for step in steps]))
        transition_reward_sum = 0.0
        for t in range(len(steps) - 1):
            step = steps[t]
            next_step = steps[t + 1]
            reward = float(next_step["reward"])
            terminal_next = bool(next_step["is_last"]) or bool(next_step["is_terminal"])
            states.append(_flatten_obs(step["observation"]))
            actions.append(np.asarray(step["action"], dtype=np.float32).reshape(-1))
            rewards.append(reward)
            discounts.append(0.0 if terminal_next else float(next_step["discount"]))
            next_states.append(_flatten_obs(next_step["observation"]))
            episode_ids.append(int(ep_idx))
            step_indices.append(int(t))
            is_first.append(bool(step["is_first"]))
            is_last.append(bool(step["is_last"]))
            is_terminal.append(bool(step["is_terminal"]))
            next_is_last.append(terminal_next)
            current_step_rewards.append(float(step["reward"]))
            next_step_rewards.append(reward)
            transition_reward_sum += reward
        episode_return_sums.append(episode_reward_sum)
        transition_return_sums.append(float(transition_reward_sum))
        kept_episodes += 1

    if not states:
        raise ValueError(f"No usable episodes found for {task}.")
    metadata = {
        "task": task,
        "max_episodes": -1 if max_episodes is None else int(max_episodes),
        "n_episodes": int(kept_episodes),
        "transition_convention": "S_t,A_t,R_{t+1},S_{t+1};discount=0_if_next_is_last",
        "mean_episode_return_sum": float(np.mean(episode_return_sums)),
        "mean_transition_return_sum": float(np.mean(transition_return_sums)),
        "max_abs_return_sum_gap": float(np.max(np.abs(np.asarray(episode_return_sums) - np.asarray(transition_return_sums)))),
    }
    return OfficialRLDBatch(
        states=np.asarray(states, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.float32),
        rewards=np.asarray(rewards, dtype=np.float32),
        discounts=np.asarray(discounts, dtype=np.float32),
        next_states=np.asarray(next_states, dtype=np.float32),
        episode_ids=np.asarray(episode_ids, dtype=np.int64),
        initial_states=np.asarray(initial_states, dtype=np.float32),
        step_indices=np.asarray(step_indices, dtype=np.int32),
        is_first=np.asarray(is_first, dtype=bool),
        is_last=np.asarray(is_last, dtype=bool),
        is_terminal=np.asarray(is_terminal, dtype=bool),
        next_is_last=np.asarray(next_is_last, dtype=bool),
        current_step_rewards=np.asarray(current_step_rewards, dtype=np.float32),
        next_step_rewards=np.asarray(next_step_rewards, dtype=np.float32),
        episode_return_sums=np.asarray(episode_return_sums, dtype=np.float32),
        transition_return_sums=np.asarray(transition_return_sums, dtype=np.float32),
        metadata=metadata,
    )


def build_official_rlds_cache(
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
    batch = _build_official_transitions_from_episodes(tfds.as_numpy(ds), task=task, max_episodes=max_episodes)
    np.savez_compressed(
        cache_path,
        states=batch.states,
        actions=batch.actions,
        rewards=batch.rewards,
        discounts=batch.discounts,
        next_states=batch.next_states,
        episode_ids=batch.episode_ids,
        initial_states=batch.initial_states,
        step_indices=batch.step_indices,
        is_first=batch.is_first,
        is_last=batch.is_last,
        is_terminal=batch.is_terminal,
        next_is_last=batch.next_is_last,
        current_step_rewards=batch.current_step_rewards,
        next_step_rewards=batch.next_step_rewards,
        episode_return_sums=batch.episode_return_sums,
        transition_return_sums=batch.transition_return_sums,
        metadata_json=np.asarray(json.dumps(batch.metadata), dtype=object),
    )
    return cache_path


def load_official_rlds_npz(path: str | Path) -> OfficialRLDBatch:
    data = np.load(path, allow_pickle=True)
    required = {
        "states",
        "actions",
        "rewards",
        "discounts",
        "next_states",
        "episode_ids",
        "initial_states",
        "step_indices",
        "is_first",
        "is_last",
        "is_terminal",
        "next_is_last",
        "current_step_rewards",
        "next_step_rewards",
        "episode_return_sums",
        "transition_return_sums",
        "metadata_json",
    }
    missing = required - set(data.files)
    if missing:
        raise ValueError(f"Cache {path} is not an official RLDS FQE cache; missing {sorted(missing)}.")
    metadata = json.loads(str(data["metadata_json"].item()))
    return OfficialRLDBatch(
        states=data["states"].astype(np.float32),
        actions=data["actions"].astype(np.float32),
        rewards=data["rewards"].astype(np.float32),
        discounts=data["discounts"].astype(np.float32),
        next_states=data["next_states"].astype(np.float32),
        episode_ids=data["episode_ids"].astype(np.int64),
        initial_states=data["initial_states"].astype(np.float32),
        step_indices=data["step_indices"].astype(np.int32),
        is_first=data["is_first"].astype(bool),
        is_last=data["is_last"].astype(bool),
        is_terminal=data["is_terminal"].astype(bool),
        next_is_last=data["next_is_last"].astype(bool),
        current_step_rewards=data["current_step_rewards"].astype(np.float32),
        next_step_rewards=data["next_step_rewards"].astype(np.float32),
        episode_return_sums=data["episode_return_sums"].astype(np.float32),
        transition_return_sums=data["transition_return_sums"].astype(np.float32),
        metadata=metadata,
    )


def _cache_matches_request(path: Path, *, task: str, max_episodes: int | None) -> bool:
    if not path.exists():
        return False
    try:
        batch = load_official_rlds_npz(path)
    except Exception:
        return False
    expected = -1 if max_episodes is None else int(max_episodes)
    return str(batch.metadata.get("task")) == str(task) and int(batch.metadata.get("max_episodes", -2)) == expected


def validate_official_rlds_batch(batch: OfficialRLDBatch, *, max_return_gap: float = 1.0) -> dict[str, float]:
    if not np.isfinite(batch.states).all() or not np.isfinite(batch.next_states).all():
        raise ValueError("RLDS cache contains nonfinite states.")
    if not np.isfinite(batch.actions).all() or not np.isfinite(batch.rewards).all():
        raise ValueError("RLDS cache contains nonfinite actions or rewards.")
    if batch.actions.ndim != 2 or batch.actions.shape[1] != 6:
        raise ValueError(f"Expected cheetah_run actions with shape (n, 6), got {batch.actions.shape}.")
    if batch.states.ndim != 2 or batch.states.shape[1] != 17:
        raise ValueError(f"Expected cheetah_run states with shape (n, 17), got {batch.states.shape}.")
    if not np.isin(np.unique(batch.discounts), [0.0, 1.0]).all():
        raise ValueError("Expected transition masks/discounts to be 0/1 after terminal handling.")
    if not bool(np.any(batch.discounts == 0.0)):
        raise ValueError("No terminal transition masks found; transition construction is likely wrong.")
    gaps = np.abs(batch.episode_return_sums - batch.transition_return_sums)
    if float(np.nanmax(gaps)) > float(max_return_gap):
        raise ValueError(
            f"Transition rewards do not reconstruct episode rewards: max gap {float(np.nanmax(gaps)):.4f}."
        )
    return {
        "n_episodes": float(len(batch.episode_return_sums)),
        "n_transitions": float(len(batch.rewards)),
        "mean_episode_return_sum": float(np.mean(batch.episode_return_sums)),
        "mean_transition_return_sum": float(np.mean(batch.transition_return_sums)),
        "max_abs_return_sum_gap": float(np.max(gaps)),
        "terminal_mask_fraction": float(np.mean(batch.discounts == 0.0)),
    }


def _sample_policy_actions(
    policy: DeepOPESavedModelPolicy,
    states: np.ndarray,
    *,
    batch_size: int = 8192,
) -> np.ndarray:
    states = np.asarray(states, dtype=np.float32)
    chunks: list[np.ndarray] = []
    for start in range(0, states.shape[0], int(batch_size)):
        chunk = states[start : start + int(batch_size)]
        actions = policy.sample_actions(chunk, None)
        if actions.shape != (chunk.shape[0], 6):
            raise ValueError(
                f"Policy {policy.policy_id} returned actions with shape {actions.shape}; expected {(chunk.shape[0], 6)}."
            )
        if not np.isfinite(actions).all():
            raise ValueError(f"Policy {policy.policy_id} returned nonfinite actions.")
        chunks.append(actions.astype(np.float32))
    return np.concatenate(chunks, axis=0)


def _sample_actions(policy: DeepOPESavedModelPolicy, normalizer: Normalizer, normalized_states, seed: int):
    import tensorflow as tf

    states_np = np.asarray(normalized_states.numpy(), dtype=np.float32) * normalizer.state_std + normalizer.state_mean
    actions = _sample_policy_actions(policy, states_np, batch_size=states_np.shape[0])
    if actions.shape != (states_np.shape[0], 6):
        raise ValueError(f"Policy {policy.policy_id} returned actions with shape {actions.shape}; expected {(states_np.shape[0], 6)}.")
    if not np.isfinite(actions).all():
        raise ValueError(f"Policy {policy.policy_id} returned nonfinite actions.")
    return tf.convert_to_tensor(actions, dtype=tf.float32)


def _predict_q_raw(
    model,
    normalizer: Normalizer,
    states: np.ndarray,
    actions: np.ndarray,
    *,
    gamma: float,
    scale: str = "discounted_sum",
) -> np.ndarray:
    import tensorflow as tf

    z = tf.convert_to_tensor(normalizer.norm_states(states), dtype=tf.float32)
    a = tf.convert_to_tensor(np.asarray(actions, dtype=np.float32), dtype=tf.float32)
    scaled = np.asarray(model(z, a).numpy(), dtype=np.float64)
    if scale == "policy_eval":
        return normalizer.unnorm_policy_eval_score(scaled)
    if scale != "discounted_sum":
        raise ValueError(f"Unknown value scale: {scale}")
    return normalizer.unnorm_scaled_return(scaled, gamma)


def _predict_v_raw(
    model,
    normalizer: Normalizer,
    policy: DeepOPESavedModelPolicy,
    states: np.ndarray,
    *,
    gamma: float,
    seed: int,
    scale: str = "discounted_sum",
) -> np.ndarray:
    del seed
    actions = _sample_policy_actions(policy, states, batch_size=8192)
    return _predict_q_raw(model, normalizer, states, actions, gamma=gamma, scale=scale)


def _train_official_style_fqe(train, policy: DeepOPESavedModelPolicy, normalizer: Normalizer, args, seed: int):
    import tensorflow as tf

    google_root = Path(args.google_research_root)
    if str(google_root) not in sys.path:
        sys.path.insert(0, str(google_root))
    from policy_eval.q_fitter import QFitter  # type: ignore

    tf.random.set_seed(seed)
    np.random.seed(seed)
    model = QFitter(train.states.shape[1], train.actions.shape[1], args.lr, args.weight_decay, args.tau)

    states = tf.convert_to_tensor(normalizer.norm_states(train.states), dtype=tf.float32)
    actions = tf.convert_to_tensor(train.actions, dtype=tf.float32)
    next_states = tf.convert_to_tensor(normalizer.norm_states(train.next_states), dtype=tf.float32)
    rewards = tf.convert_to_tensor(normalizer.norm_rewards(train.rewards), dtype=tf.float32)
    masks = tf.convert_to_tensor(train.discounts, dtype=tf.float32)
    weights = tf.ones_like(rewards, dtype=tf.float32)
    min_reward = tf.reduce_min(rewards)
    max_reward = tf.reduce_max(rewards)
    next_actions_all = tf.convert_to_tensor(
        _sample_policy_actions(policy, train.next_states, batch_size=int(args.policy_action_batch_size)),
        dtype=tf.float32,
    )
    rng = np.random.default_rng(seed)
    n = len(train.rewards)
    batch_size = min(int(args.batch_size), n)
    losses: list[float] = []
    chunk_size = max(1, int(args.graph_train_chunk_size))
    if chunk_size > 1:

        @tf.function
        def train_chunk(offset, chunk_steps):
            loss = tf.constant(float("nan"), dtype=tf.float32)
            i = tf.constant(0, dtype=tf.int32)

            def cond(i, loss):
                del loss
                return i < chunk_steps

            def body(i, loss):
                idx = tf.random.stateless_uniform(
                    shape=(batch_size,),
                    seed=tf.stack(
                        [
                            tf.cast(seed % 2147483647, tf.int32),
                            tf.cast(offset + i, tf.int32),
                        ]
                    ),
                    minval=0,
                    maxval=n,
                    dtype=tf.int32,
                )
                ns = tf.gather(next_states, idx)
                loss = model.update(
                    tf.gather(states, idx),
                    tf.gather(actions, idx),
                    ns,
                    tf.gather(next_actions_all, idx),
                    tf.gather(rewards, idx),
                    tf.gather(masks, idx),
                    tf.gather(weights, idx),
                    float(args.gamma),
                    min_reward,
                    max_reward,
                )
                return i + 1, loss

            _, loss = tf.while_loop(cond, body, [i, loss])
            return loss

        total = int(args.num_updates)
        log_every = max(1, total // 10)
        offset = 0
        while offset < total:
            steps = min(chunk_size, total - offset)
            loss = train_chunk(tf.constant(offset, dtype=tf.int32), tf.constant(steps, dtype=tf.int32))
            if offset == 0 or offset + steps >= total or (offset + steps) % log_every < steps:
                losses.append(float(loss.numpy()))
            offset += steps
        return model, losses

    for step in range(int(args.num_updates)):
        idx = rng.integers(0, n, size=batch_size)
        idx_tf = tf.convert_to_tensor(idx, dtype=tf.int32)
        ns = tf.gather(next_states, idx_tf)
        loss = model.update(
            tf.gather(states, idx_tf),
            tf.gather(actions, idx_tf),
            ns,
            tf.gather(next_actions_all, idx_tf),
            tf.gather(rewards, idx_tf),
            tf.gather(masks, idx_tf),
            tf.gather(weights, idx_tf),
            float(args.gamma),
            min_reward,
            max_reward,
        )
        if step == 0 or step == int(args.num_updates) - 1 or (step + 1) % max(1, int(args.num_updates) // 10) == 0:
            losses.append(float(loss.numpy()))
    return model, losses


def _calibration_error(pred: np.ndarray, target: np.ndarray, n_bins: int = 20) -> float:
    pred = np.asarray(pred, dtype=float)
    target = np.asarray(target, dtype=float)
    edges = np.unique(np.quantile(pred, np.linspace(0, 1, min(n_bins, max(2, pred.size // 10)) + 1)))
    if edges.size < 3:
        return float(np.mean((pred - target) ** 2))
    bins = np.searchsorted(edges[1:-1], pred, side="right")
    return float(sum((bins == b).mean() * (pred[bins == b].mean() - target[bins == b].mean()) ** 2 for b in np.unique(bins)))


def _read_existing_rows(path: Path) -> list[dict[str, object]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    return pd.read_csv(path).to_dict("records")


def _completed_policy_indices(rows: list[dict[str, object]], args: argparse.Namespace) -> set[int]:
    if not rows:
        return set()
    df = pd.DataFrame(rows)
    required = {"policy_index", "calibrator", "task", "seed", "num_updates", "gamma"}
    if not required.issubset(df.columns):
        return set()
    matched = df[
        df["task"].astype(str).eq(str(args.task))
        & df["seed"].astype(int).eq(int(args.seed))
        & df["num_updates"].astype(int).eq(int(args.num_updates))
        & np.isclose(df["gamma"].astype(float), float(args.gamma))
    ]
    completed: set[int] = set()
    for policy_i, group in matched.groupby("policy_index"):
        if {"none", "linear", "isotonic"}.issubset(set(group["calibrator"].astype(str))):
            completed.add(int(policy_i))
    return completed


def _write_run_outputs(rows: list[dict[str, object]], out: Path, args: argparse.Namespace) -> dict[str, Path]:
    raw = out / "rlu_cheetah_fqe_reproduction_rows.csv"
    summary = out / "rlu_cheetah_fqe_reproduction_summary.csv"
    config = out / "rlu_cheetah_fqe_reproduction_config.json"
    _write_csv(rows, raw)
    _write_csv(_summary(rows), summary)
    config.write_text(json.dumps(vars(args), indent=2, default=str))
    return {"raw": raw, "summary": summary, "config": config}


def run(args: argparse.Namespace) -> dict[str, Path]:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    raw = out / "rlu_cheetah_fqe_reproduction_rows.csv"
    config = out / "rlu_cheetah_fqe_reproduction_config.json"
    config.write_text(json.dumps(vars(args), indent=2, default=str))
    cache = Path(args.cache_path)
    if bool(args.rebuild_cache) or not _cache_matches_request(cache, task=args.task, max_episodes=args.max_cache_episodes):
        build_official_rlds_cache(args.task, cache, max_episodes=args.max_cache_episodes, tfds_data_dir=args.tfds_data_dir)
    official_full = load_official_rlds_npz(cache)
    cache_diagnostics = validate_official_rlds_batch(official_full)
    full = official_full.to_continuous()
    train, cal, diag = split_by_episode(
        full,
        seed=int(args.seed),
        fractions=(args.train_fraction, args.cal_fraction, 1 - args.train_fraction - args.cal_fraction),
    )

    normalizer = Normalizer.fit(train.states, train.rewards, train.discounts, normalize_rewards=bool(args.normalize_rewards))
    bench = Path(args.benchmark_dir)
    official_fqe = _load_pickle(bench / "rlunplugged_fqel2.pkl")
    official_gt = _load_pickle(bench / "rlunplugged_gt.pkl")
    policy_ids = _load_policy_ids(bench, args.task, args.policy_indices)

    rows: list[dict[str, object]] = _read_existing_rows(raw) if bool(args.resume) else []
    completed = _completed_policy_indices(rows, args)
    for policy_i, policy_id in zip(args.policy_indices, policy_ids):
        if int(policy_i) in completed:
            print(f"[fqe-repro] skip completed policy_index={policy_i} policy={policy_id}", flush=True)
            continue
        print(f"[fqe-repro] policy_index={policy_i} policy={policy_id}", flush=True)
        policy = DeepOPESavedModelPolicy(policy_id, args.policy_root, args.policy_cache_dir)
        model, losses = _train_official_style_fqe(train, policy, normalizer, args, int(args.seed) + int(policy_i) * 997)

        cal_actions = _sample_policy_actions(policy, cal.states, batch_size=int(args.policy_action_batch_size))
        cal_next_actions = _sample_policy_actions(policy, cal.next_states, batch_size=int(args.policy_action_batch_size))
        diag_actions = _sample_policy_actions(policy, diag.states, batch_size=int(args.policy_action_batch_size))
        diag_next_actions = _sample_policy_actions(policy, diag.next_states, batch_size=int(args.policy_action_batch_size))
        init_actions = _sample_policy_actions(policy, full.initial_states, batch_size=int(args.policy_action_batch_size))

        cal_v = _predict_q_raw(model, normalizer, cal.states, cal_actions, gamma=args.gamma, scale="discounted_sum")
        cal_next = _predict_q_raw(model, normalizer, cal.next_states, cal_next_actions, gamma=args.gamma, scale="discounted_sum")
        cal_target = cal.rewards + float(args.gamma) * cal.discounts * cal_next
        diag_v = _predict_q_raw(model, normalizer, diag.states, diag_actions, gamma=args.gamma, scale="discounted_sum")
        diag_next = _predict_q_raw(model, normalizer, diag.next_states, diag_next_actions, gamma=args.gamma, scale="discounted_sum")
        diag_target = diag.rewards + float(args.gamma) * diag.discounts * diag_next
        init_states = full.initial_states
        init_v = _predict_q_raw(model, normalizer, init_states, init_actions, gamma=args.gamma, scale="discounted_sum")
        init_policy_eval = _predict_q_raw(model, normalizer, init_states, init_actions, gamma=args.gamma, scale="policy_eval")

        policy_rows: list[dict[str, object]] = []
        for calibrator in ["none", "linear", "isotonic"]:
            cal_obj = None if calibrator == "none" else fit_calibrator(calibrator, cal_v, cal_target)
            pred = diag_v if cal_obj is None else cal_obj.predict(diag_v)
            next_pred = diag_next if cal_obj is None else cal_obj.predict(diag_next)
            target = diag.rewards + float(args.gamma) * diag.discounts * next_pred
            init_pred = init_v if cal_obj is None else cal_obj.predict(init_v)
            estimate = float(np.mean(init_pred))
            policy_eval_score = estimate * max(1.0 - float(args.gamma), 1e-8)
            official_value = float(official_fqe[policy_id][0])
            official_return = float(official_gt[policy_id][0])
            policy_rows.append(
                {
                    "task": args.task,
                    "seed": int(args.seed),
                    "policy_index": int(policy_i),
                    "policy_id": policy_id,
                    "calibrator": calibrator,
                    "estimated_return": estimate,
                    "discounted_return_estimate": estimate,
                    "policy_eval_score": policy_eval_score,
                    "raw_policy_eval_score": float(np.mean(init_policy_eval)),
                    "official_fqe_l2": official_value,
                    "official_return": official_return,
                    "absolute_error_vs_official_fqe_l2": abs(estimate - official_value),
                    "absolute_ope_error_vs_official_return": abs(estimate - official_return),
                    "bellman_calibration_error": _calibration_error(pred, target),
                    "bellman_outcome_mse": float(np.mean((pred - target) ** 2)),
                    "n_train_transitions": len(train.rewards),
                    "n_calibration_transitions": len(cal.rewards),
                    "n_diagnostic_transitions": len(diag.rewards),
                    "n_cache_episodes": int(cache_diagnostics["n_episodes"]),
                    "n_cache_transitions": int(cache_diagnostics["n_transitions"]),
                    "cache_mean_episode_return_sum": float(cache_diagnostics["mean_episode_return_sum"]),
                    "cache_mean_transition_return_sum": float(cache_diagnostics["mean_transition_return_sum"]),
                    "cache_max_abs_return_sum_gap": float(cache_diagnostics["max_abs_return_sum_gap"]),
                    "terminal_mask_fraction": float(cache_diagnostics["terminal_mask_fraction"]),
                    "num_updates": int(args.num_updates),
                    "gamma": float(args.gamma),
                    "normalize_states": True,
                    "normalize_rewards": bool(args.normalize_rewards),
                    "value_scale": "discounted_sum",
                    "reward_mean": float(normalizer.reward_mean),
                    "reward_std": float(normalizer.reward_std),
                    "critic_architecture": "Deep OPE policy_eval.QFitter MLP(256,256)",
                    "target_policy_actions": "precomputed_fixed_savedmodel_actions",
                    "policy_action_batch_size": int(args.policy_action_batch_size),
                    "loss_first": losses[0] if losses else float("nan"),
                    "loss_last": losses[-1] if losses else float("nan"),
                    "loss_finite": bool(np.isfinite(losses).all()) if losses else False,
                    "diag_value_std": float(np.std(pred)),
                    "init_value_std": float(np.std(init_pred)),
                    "scalar_initial_state_source": "all_cached_episode_initial_states",
                }
            )
        rows.extend(policy_rows)
        paths = _write_run_outputs(rows, out, args)
        print(
            f"[fqe-repro] checkpoint policy_index={policy_i} raw={paths['raw']} summary={paths['summary']}",
            flush=True,
        )
    paths = _write_run_outputs(rows, out, args)
    print(f"Wrote raw: {paths['raw']}\nWrote summary: {paths['summary']}\nWrote config: {paths['config']}")
    return paths


def _summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    df = pd.DataFrame(rows)
    out = []
    for calibrator, group in df.groupby("calibrator"):
        raw = df[df["calibrator"].eq("none")]
        out.append(
            {
                "calibrator": calibrator,
                "n_policies": int(group["policy_id"].nunique()),
                "mean_abs_error_vs_official_fqe_l2": float(group["absolute_error_vs_official_fqe_l2"].mean()),
                "mean_abs_ope_error_vs_official_return": float(group["absolute_ope_error_vs_official_return"].mean()),
                "relative_bellman_calibration_error": float(group["bellman_calibration_error"].mean() / raw["bellman_calibration_error"].mean()),
                "relative_ope_error": float(group["absolute_ope_error_vs_official_return"].mean() / raw["absolute_ope_error_vs_official_return"].mean()),
                "relative_error_vs_official_fqe_l2": float(group["absolute_error_vs_official_fqe_l2"].mean() / raw["absolute_error_vs_official_fqe_l2"].mean()),
                "policy_spearman_vs_official_fqe_l2": float(group["discounted_return_estimate"].rank().corr(group["official_fqe_l2"].rank())),
                "policy_spearman_vs_official_return": float(group["discounted_return_estimate"].rank().corr(group["official_return"].rank())),
                "policy_eval_score_spearman_vs_official_fqe_l2": float(group["policy_eval_score"].rank().corr(group["official_fqe_l2"].rank())),
                "mean_diag_value_std": float(group["diag_value_std"].mean()),
                "all_losses_finite": bool(group["loss_finite"].all()),
                "raw_rank_gate_pass": bool(
                    calibrator == "none"
                    and (
                        group["discounted_return_estimate"].rank().corr(group["official_fqe_l2"].rank()) > 0.3
                        or group["discounted_return_estimate"].rank().corr(group["official_return"].rank()) > 0.3
                    )
                ),
            }
        )
    return out


def _write_csv(rows: list[dict[str, object]], path: Path) -> None:
    if not rows:
        path.write_text("")
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce Deep OPE cheetah_run FQE-L2 values with a trained FQE critic.")
    parser.add_argument("--task", default="cheetah_run")
    parser.add_argument("--benchmark_dir", default="hopper_fqe_benchmark/artifacts/benchmark/dope")
    parser.add_argument("--cache_path", default="FQE_calibration_neurips/results/rlu_cache/cheetah_run_official_rlds.npz")
    parser.add_argument("--tfds_data_dir", default="FQE_calibration_neurips/results/tfds")
    parser.add_argument("--google_research_root", default="/tmp/google-research")
    parser.add_argument("--policy_root", default=None)
    parser.add_argument("--policy_cache_dir", default="FQE_calibration_neurips/results/rlu_policy_cache")
    parser.add_argument("--output_dir", default="FQE_calibration_neurips/results/rlu_cheetah_fqe_reproduction")
    parser.add_argument("--policy_indices", nargs="+", type=int, default=list(range(8)))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--max_cache_episodes", type=int, default=None)
    parser.add_argument("--train_fraction", type=float, default=0.7)
    parser.add_argument("--cal_fraction", type=float, default=0.15)
    parser.add_argument("--num_updates", type=int, default=100000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--normalize_rewards", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--policy_action_batch_size", type=int, default=8192)
    parser.add_argument("--graph_train_chunk_size", type=int, default=1000)
    parser.add_argument("--value_scale", choices=["discounted_sum", "policy_eval"], default="discounted_sum")
    parser.add_argument("--rebuild_cache", action="store_true")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
