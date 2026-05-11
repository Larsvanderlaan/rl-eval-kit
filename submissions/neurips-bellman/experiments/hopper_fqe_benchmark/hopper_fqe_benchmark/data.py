from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen

import h5py
import numpy as np


HOPPER_DATASET_SPECS = {
    "hopper-medium-v0": (
        "hopper_medium-v0.hdf5",
        "https://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/hopper_medium.hdf5",
    ),
    "hopper-random-v0": (
        "hopper_random-v0.hdf5",
        "https://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/hopper_random.hdf5",
    ),
    "hopper-medium-replay-v0": (
        "hopper_medium_replay-v0.hdf5",
        "https://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/hopper_medium_replay.hdf5",
    ),
    "hopper-medium-expert-v0": (
        "hopper_medium_expert-v0.hdf5",
        "https://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/hopper_medium_expert.hdf5",
    ),
    "hopper-expert-v0": (
        "hopper_expert-v0.hdf5",
        "https://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/hopper_expert.hdf5",
    ),
    # Local artifact support for newer D4RL/Minari-style files. These are only
    # used when the file is already present in data_dir.
    "hopper-medium-v2": ("hopper_medium-v2.hdf5", None),
    "hopper-full-replay-v2": ("hopper_full_replay-v2.hdf5", None),
    "hopper-medium-replay-v2": ("hopper_medium_replay-v2.hdf5", None),
    "hopper-medium-expert-v2": ("hopper_medium_expert-v2.hdf5", None),
    "hopper-random-v2": ("hopper_random-v2.hdf5", None),
}

HOPPER_MEDIUM_V0_URL = HOPPER_DATASET_SPECS["hopper-medium-v0"][1]
HOPPER_MEDIUM_V0_FILENAME = HOPPER_DATASET_SPECS["hopper-medium-v0"][0]


@dataclass
class HopperTrajectoryDataset:
    observations_raw: np.ndarray
    actions: np.ndarray
    next_observations_raw: np.ndarray
    rewards_raw: np.ndarray
    masks: np.ndarray
    steps: np.ndarray
    initial_observations_raw: np.ndarray
    initial_weights: np.ndarray
    state_mean: np.ndarray
    state_std: np.ndarray
    reward_mean: float
    reward_std: float
    observations: np.ndarray
    next_observations: np.ndarray
    rewards: np.ndarray
    trajectory_count: int

    def __len__(self) -> int:
        return int(self.observations.shape[0])

    @property
    def observation_dim(self) -> int:
        return int(self.observations.shape[1])

    @property
    def action_dim(self) -> int:
        return int(self.actions.shape[1])

    def normalize_states(self, states: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        states_arr = np.asarray(states, dtype=np.float32)
        return (states_arr - self.state_mean) / np.maximum(self.state_std, eps)

    def unnormalize_states(self, states: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        states_arr = np.asarray(states, dtype=np.float32)
        return states_arr * np.maximum(self.state_std, eps) + self.state_mean

    def normalize_rewards(self, rewards: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        rewards_arr = np.asarray(rewards, dtype=np.float32)
        return (rewards_arr - self.reward_mean) / max(self.reward_std, eps)

    def unnormalize_rewards(self, rewards: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        rewards_arr = np.asarray(rewards, dtype=np.float32)
        return rewards_arr * max(self.reward_std, eps) + self.reward_mean


def _download_file(url: str, destination: Path, chunk_size: int = 1 << 20) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url) as response:
        with destination.open("wb") as output:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                output.write(chunk)
    return destination


def ensure_hopper_dataset(data_dir: str | Path, dataset_name: str = "hopper-medium-v0") -> Path:
    data_dir = Path(data_dir)
    key = str(dataset_name).replace("_", "-")
    if key not in HOPPER_DATASET_SPECS:
        raise KeyError(f"Unknown Hopper dataset '{dataset_name}'. Available: {sorted(HOPPER_DATASET_SPECS)}")
    filename, url = HOPPER_DATASET_SPECS[key]
    path = data_dir / filename
    if not path.exists():
        if url is None:
            raise FileNotFoundError(
                f"Dataset '{key}' is configured for local use but {path} does not exist."
            )
        _download_file(url, path)
    return path


def ensure_hopper_medium_v0(data_dir: str | Path) -> Path:
    return ensure_hopper_dataset(data_dir, "hopper-medium-v0")


def _load_array(handle: h5py.File, key: str, dtype: np.dtype) -> np.ndarray:
    if key not in handle:
        raise KeyError(f"Expected key '{key}' in dataset {handle.filename}.")
    return np.asarray(handle[key], dtype=dtype)


def _load_raw_hdf5(path: Path) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as handle:
        out = {
            "observations": _load_array(handle, "observations", np.float32),
            "actions": _load_array(handle, "actions", np.float32),
            "rewards": _load_array(handle, "rewards", np.float32).reshape(-1),
            "terminals": _load_array(handle, "terminals", np.float32).reshape(-1) > 0.5,
        }
        if "timeouts" in handle:
            out["timeouts"] = _load_array(handle, "timeouts", np.float32).reshape(-1) > 0.5
        else:
            out["timeouts"] = np.zeros_like(out["terminals"], dtype=bool)
        if "next_observations" in handle:
            out["next_observations"] = _load_array(handle, "next_observations", np.float32)
        return out


def _extract_trajectories(raw_dataset: dict[str, np.ndarray]) -> list[dict[str, np.ndarray]]:
    trajectories: list[dict[str, list[np.ndarray] | list[float]]] = []
    new_trajectory = True
    trajectory: dict[str, list[np.ndarray] | list[float]] | None = None

    dataset_length = int(raw_dataset["actions"].shape[0])
    for idx in range(dataset_length):
        if new_trajectory:
            trajectory = {
                "states": [],
                "actions": [],
                "next_states": [],
                "rewards": [],
                "masks": [],
            }

        assert trajectory is not None
        trajectory["states"].append(raw_dataset["observations"][idx])
        trajectory["actions"].append(raw_dataset["actions"][idx])
        trajectory["rewards"].append(raw_dataset["rewards"][idx])
        trajectory["masks"].append(np.float32(1.0 - raw_dataset["terminals"][idx]))
        if "next_observations" in raw_dataset:
            trajectory["next_states"].append(raw_dataset["next_observations"][idx])
        elif not new_trajectory:
            trajectory["next_states"].append(raw_dataset["observations"][idx])

        end_trajectory = bool(raw_dataset["terminals"][idx] or raw_dataset["timeouts"][idx])
        if end_trajectory:
            if "next_observations" not in raw_dataset:
                trajectory["next_states"].append(raw_dataset["observations"][idx])
            if raw_dataset["timeouts"][idx] and not raw_dataset["terminals"][idx]:
                for key in trajectory:
                    del trajectory[key][-1]
            if trajectory["actions"]:
                finalized = {
                    key: np.asarray(value, dtype=np.float32)
                    for key, value in trajectory.items()
                }
                trajectories.append(finalized)
        new_trajectory = end_trajectory

    return trajectories


def _select_trajectories(
    trajectories: list[dict[str, np.ndarray]],
    max_trajectories: int | None,
    max_transitions: int | None,
    seed: int,
) -> list[dict[str, np.ndarray]]:
    if max_trajectories is None and max_transitions is None:
        return trajectories

    rng = np.random.default_rng(seed)
    indices = np.arange(len(trajectories))
    rng.shuffle(indices)

    selected: list[dict[str, np.ndarray]] = []
    total_transitions = 0
    for idx in indices:
        trajectory = trajectories[int(idx)]
        if max_trajectories is not None and len(selected) >= max_trajectories:
            break
        next_total = total_transitions + int(trajectory["actions"].shape[0])
        if max_transitions is not None and selected and next_total > max_transitions:
            break
        selected.append(trajectory)
        total_transitions = next_total
        if max_transitions is not None and total_transitions >= max_transitions:
            break

    if not selected:
        raise ValueError("No trajectories selected; increase max_trajectories or max_transitions.")
    return selected


def _augment_trajectories(
    trajectories: list[dict[str, np.ndarray]],
    noise_scale: float,
) -> list[dict[str, np.ndarray]]:
    if noise_scale <= 0.0:
        return trajectories

    reward_std = float(np.std(np.concatenate([traj["rewards"] for traj in trajectories], axis=0)))
    augmented: list[dict[str, np.ndarray]] = []
    for trajectory in trajectories:
        base = {key: np.asarray(value, dtype=np.float32).copy() for key, value in trajectory.items()}
        plus = {key: np.asarray(value, dtype=np.float32).copy() for key, value in trajectory.items()}
        minus = {key: np.asarray(value, dtype=np.float32).copy() for key, value in trajectory.items()}
        plus["rewards"] = plus["rewards"] + reward_std * noise_scale
        minus["rewards"] = minus["rewards"] - reward_std * noise_scale
        augmented.extend((base, plus, minus))
    return augmented


def load_hopper_medium_v0(
    data_dir: str | Path,
    *,
    normalize_states: bool = True,
    normalize_rewards: bool = True,
    bootstrap: bool = False,
    noise_scale: float = 0.0,
    max_trajectories: int | None = None,
    max_transitions: int | None = None,
    seed: int = 0,
    eps: float = 1e-5,
) -> HopperTrajectoryDataset:
    path = ensure_hopper_medium_v0(data_dir)
    return load_hopper_dataset_from_path(
        path,
        normalize_states=normalize_states,
        normalize_rewards=normalize_rewards,
        bootstrap=bootstrap,
        noise_scale=noise_scale,
        max_trajectories=max_trajectories,
        max_transitions=max_transitions,
        seed=seed,
        eps=eps,
    )


def load_hopper_dataset(
    data_dir: str | Path,
    dataset_name: str = "hopper-medium-v0",
    *,
    normalize_states: bool = True,
    normalize_rewards: bool = True,
    bootstrap: bool = False,
    noise_scale: float = 0.0,
    max_trajectories: int | None = None,
    max_transitions: int | None = None,
    seed: int = 0,
    eps: float = 1e-5,
) -> HopperTrajectoryDataset:
    path = ensure_hopper_dataset(data_dir, dataset_name)
    return load_hopper_dataset_from_path(
        path,
        normalize_states=normalize_states,
        normalize_rewards=normalize_rewards,
        bootstrap=bootstrap,
        noise_scale=noise_scale,
        max_trajectories=max_trajectories,
        max_transitions=max_transitions,
        seed=seed,
        eps=eps,
    )


def load_hopper_dataset_from_path(
    path: str | Path,
    *,
    normalize_states: bool = True,
    normalize_rewards: bool = True,
    bootstrap: bool = False,
    noise_scale: float = 0.0,
    max_trajectories: int | None = None,
    max_transitions: int | None = None,
    seed: int = 0,
    eps: float = 1e-5,
) -> HopperTrajectoryDataset:
    path = Path(path)
    raw_dataset = _load_raw_hdf5(path)
    trajectories = _extract_trajectories(raw_dataset)
    trajectories = _select_trajectories(
        trajectories,
        max_trajectories=max_trajectories,
        max_transitions=max_transitions,
        seed=seed,
    )
    trajectories = _augment_trajectories(trajectories, noise_scale=noise_scale)

    observations_raw = np.concatenate([trajectory["states"] for trajectory in trajectories], axis=0).astype(np.float32)
    actions = np.concatenate([trajectory["actions"] for trajectory in trajectories], axis=0).astype(np.float32)
    next_observations_raw = np.concatenate([trajectory["next_states"] for trajectory in trajectories], axis=0).astype(np.float32)
    rewards_raw = np.concatenate([trajectory["rewards"] for trajectory in trajectories], axis=0).astype(np.float32)
    masks = np.concatenate([trajectory["masks"] for trajectory in trajectories], axis=0).astype(np.float32)
    steps = np.concatenate(
        [np.arange(trajectory["actions"].shape[0], dtype=np.float32) for trajectory in trajectories],
        axis=0,
    )
    initial_observations_raw = np.stack([trajectory["states"][0] for trajectory in trajectories], axis=0).astype(np.float32)

    num_trajectories = len(trajectories)
    if bootstrap:
        initial_weights = np.random.multinomial(
            num_trajectories,
            [1.0 / num_trajectories] * num_trajectories,
            1,
        ).astype(np.float32)[0]
    else:
        initial_weights = np.ones(num_trajectories, dtype=np.float32)

    if normalize_states:
        state_mean = observations_raw.mean(axis=0).astype(np.float32)
        state_std = observations_raw.std(axis=0).astype(np.float32)
        state_std = np.where(state_std < eps, 1.0, state_std).astype(np.float32)
        observations = ((observations_raw - state_mean) / state_std).astype(np.float32)
        next_observations = ((next_observations_raw - state_mean) / state_std).astype(np.float32)
    else:
        state_mean = np.zeros(observations_raw.shape[1], dtype=np.float32)
        state_std = np.ones(observations_raw.shape[1], dtype=np.float32)
        observations = observations_raw.copy()
        next_observations = next_observations_raw.copy()

    if normalize_rewards:
        reward_mean = float(rewards_raw.mean())
        if np.min(masks) == 0.0:
            reward_mean = 0.0
        reward_std = float(rewards_raw.std())
        reward_std = max(reward_std, eps)
        rewards = ((rewards_raw - reward_mean) / reward_std).astype(np.float32)
    else:
        reward_mean = 0.0
        reward_std = 1.0
        rewards = rewards_raw.copy()

    return HopperTrajectoryDataset(
        observations_raw=observations_raw,
        actions=actions,
        next_observations_raw=next_observations_raw,
        rewards_raw=rewards_raw,
        masks=masks,
        steps=steps,
        initial_observations_raw=initial_observations_raw,
        initial_weights=initial_weights,
        state_mean=state_mean,
        state_std=state_std,
        reward_mean=reward_mean,
        reward_std=reward_std,
        observations=observations,
        next_observations=next_observations,
        rewards=rewards,
        trajectory_count=num_trajectories,
    )
