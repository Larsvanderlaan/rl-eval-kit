"""Dataset containers and converters for GenPQR."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from genpqr.types import ActionSpaceSpec, Array
from genpqr.validation import as_1d_float, as_2d_float, optional_terminals, optional_weights


@dataclass(frozen=True)
class TransitionDataset:
    """Validated row-wise transition dataset.

    Parameters
    ----------
    states, actions, next_states:
        Row-wise transition arrays.
    terminals:
        Terminal indicators with one value per row.
    action_space:
        Explicit action-space contract.
    sample_weight:
        Optional nonnegative row weights.
    episode_ids:
        Optional episode identifier per row. Preserved for episode-aware folds
        and ordered-dataset adapter preflight.
    initial_states, initial_actions:
        Optional initial distribution payloads preserved for downstream
        algorithms that need them.
    metadata:
        Free-form user metadata. GenPQR does not interpret these values.
    """

    states: Array
    actions: Array
    next_states: Array
    terminals: Array
    action_space: ActionSpaceSpec
    sample_weight: Array | None = None
    episode_ids: Array | None = None
    initial_states: Array | None = None
    initial_actions: Array | None = None
    metadata: dict[str, Any] | None = None

    @classmethod
    def from_arrays(
        cls,
        *,
        states: Array,
        actions: Array,
        next_states: Array,
        terminals: Array | None = None,
        action_space: ActionSpaceSpec | None = None,
        sample_weight: Array | None = None,
        episode_ids: Array | None = None,
        initial_states: Array | None = None,
        initial_actions: Array | None = None,
        metadata: Mapping[str, Any] | None = None,
        strict_episodes: bool = False,
    ) -> "TransitionDataset":
        """Create a validated transition dataset from arrays."""

        states_2d = as_2d_float(states, "states")
        n_rows = states_2d.shape[0]
        if n_rows == 0:
            raise ValueError("states must be nonempty.")
        next_states_2d = as_2d_float(next_states, "next_states", n_rows=n_rows)
        if next_states_2d.shape[1] != states_2d.shape[1]:
            raise ValueError("next_states must have the same number of columns as states.")
        spec = ActionSpaceSpec.infer(actions) if action_space is None else action_space
        spec.validate_actions(actions, n_rows=n_rows)
        terminals_1d = optional_terminals(terminals, n_rows)
        weights = optional_weights(sample_weight, n_rows)
        episodes = None if episode_ids is None else np.asarray(episode_ids).reshape(-1)
        if episodes is not None and episodes.shape[0] != n_rows:
            raise ValueError("episode_ids must contain one value per row.")
        init_states = None if initial_states is None else as_2d_float(initial_states, "initial_states")
        init_actions = None
        if initial_actions is not None:
            if init_states is None:
                raise ValueError("initial_actions requires initial_states.")
            spec.validate_actions(initial_actions, n_rows=init_states.shape[0], name="initial_actions")
            init_actions = np.asarray(initial_actions)
        out = cls(
            states=states_2d,
            actions=np.asarray(actions),
            next_states=next_states_2d,
            terminals=terminals_1d,
            action_space=spec,
            sample_weight=weights,
            episode_ids=episodes,
            initial_states=init_states,
            initial_actions=init_actions,
            metadata=dict(metadata or {}),
        )
        if strict_episodes:
            out.validate_ordered_episodes(strict_terminal=True)
            out.metadata["ordered_episodes_validated"] = True
            out.metadata["d3rlpy_transition_compatible"] = True
        return out

    @classmethod
    def from_d3rlpy(
        cls,
        dataset: Mapping[str, Array] | object,
        *,
        next_states: Array | None = None,
        action_space: ActionSpaceSpec | None = None,
        metadata: Mapping[str, Any] | None = None,
        strict_episodes: bool = True,
    ) -> "TransitionDataset":
        """Create a dataset from d3rlpy-like arrays without importing d3rlpy."""

        observations = _get_mapping_or_attr(dataset, "observations")
        actions = _get_mapping_or_attr(dataset, "actions")
        terminals = _get_mapping_or_attr(dataset, "terminals", default=None)
        episode_ids = _get_mapping_or_attr(dataset, "episode_ids", default=None)
        next_obs = next_states
        if next_obs is None:
            next_obs = _get_mapping_or_attr(dataset, "next_observations", default=None)
        if next_obs is None:
            next_obs = _shift_next_states(observations, terminals=terminals, episode_ids=episode_ids)
        return cls.from_arrays(
            states=observations,
            actions=actions,
            next_states=next_obs,
            terminals=terminals,
            action_space=action_space,
            episode_ids=episode_ids,
            metadata={"source": "d3rlpy", **dict(metadata or {})},
            strict_episodes=strict_episodes and episode_ids is not None,
        )

    @classmethod
    def from_scope_rl(
        cls,
        logged_dataset: Mapping[str, Any],
        *,
        action_space: ActionSpaceSpec | None = None,
        metadata: Mapping[str, Any] | None = None,
        strict_episodes: bool = True,
    ) -> "TransitionDataset":
        """Create a dataset from a SCOPE-RL-style logged-dataset dictionary."""

        states, actions, terminals, episode_ids = _flatten_scope_rl_logged_dataset(logged_dataset)
        next_states = _shift_next_states(states, terminals=terminals, episode_ids=episode_ids)
        n_actions = logged_dataset.get("n_actions")
        spec = action_space or (ActionSpaceSpec.discrete(int(n_actions)) if n_actions is not None else None)
        return cls.from_arrays(
            states=states,
            actions=actions,
            next_states=next_states,
            terminals=terminals,
            action_space=spec,
            episode_ids=episode_ids,
            metadata={"source": "scope_rl", **dict(metadata or {})},
            strict_episodes=strict_episodes,
        )

    @property
    def n_rows(self) -> int:
        """Number of transition rows."""

        return int(self.states.shape[0])

    @property
    def encoded_actions(self) -> Array:
        """Actions encoded in the matrix representation used by generic Q backends."""

        return self.action_space.action_matrix(self.actions, n_rows=self.n_rows)

    @property
    def has_ordered_episodes(self) -> bool:
        """Whether episode identifiers are available."""

        return self.episode_ids is not None

    def validate_ordered_episodes(self, *, strict_terminal: bool = True) -> None:
        """Validate that episode ids form contiguous ordered trajectory blocks."""

        if self.episode_ids is None:
            raise ValueError("episode_ids are required for ordered episode validation.")
        ids = np.asarray(self.episode_ids)
        seen: set[Any] = set()
        last = ids[0]
        seen.add(last.item() if hasattr(last, "item") else last)
        starts = [0]
        for i, value in enumerate(ids[1:], start=1):
            key = value.item() if hasattr(value, "item") else value
            last_key = last.item() if hasattr(last, "item") else last
            if key != last_key:
                if key in seen:
                    raise ValueError("episode_ids must be contiguous ordered blocks.")
                seen.add(key)
                starts.append(i)
                last = value
        starts.append(ids.shape[0])
        if strict_terminal:
            for start, stop in zip(starts[:-1], starts[1:]):
                episode_terminals = self.terminals[start:stop]
                if episode_terminals.shape[0] > 1 and np.any(episode_terminals[:-1] > 0.0):
                    raise ValueError("terminal indicators may only appear at the final row of an episode.")
                if episode_terminals.shape[0] == 0 or episode_terminals[-1] <= 0.0:
                    raise ValueError("terminal indicators must appear at the final row of each episode.")
                if stop - start > 1:
                    shifted = self.states[start + 1 : stop]
                    supplied = self.next_states[start : stop - 1]
                    if not np.allclose(supplied, shifted):
                        raise ValueError("next_states must match the next row within each ordered episode.")

    def summary(self) -> dict[str, Any]:
        """Return JSON-safe dataset summary statistics."""

        episode_count = None if self.episode_ids is None else int(np.unique(self.episode_ids).shape[0])
        weights = self.sample_weight
        action_summary: dict[str, Any]
        if self.action_space.kind == "discrete":
            idx = self.action_space.action_indices(self.actions, n_rows=self.n_rows)
            counts = np.bincount(idx, minlength=int(self.action_space.n_actions))
            action_summary = {"counts": counts.astype(int).tolist()}
        else:
            matrix = self.action_space.action_matrix(self.actions, n_rows=self.n_rows)
            action_summary = {
                "mean": np.mean(matrix, axis=0).tolist(),
                "std": np.std(matrix, axis=0).tolist(),
            }
        return {
            "n_rows": self.n_rows,
            "n_episodes": episode_count,
            "action_space": self.action_space.kind,
            "terminal_rate": float(np.mean(self.terminals > 0.0)),
            "has_initial_states": self.initial_states is not None,
            "has_initial_actions": self.initial_actions is not None,
            "sample_weight_mean": None if weights is None else float(np.mean(weights)),
            "sample_weight_min": None if weights is None else float(np.min(weights)),
            "sample_weight_max": None if weights is None else float(np.max(weights)),
            "actions": action_summary,
        }

    def subset(self, indices: Array) -> "TransitionDataset":
        """Return a row subset preserving metadata and initial payloads."""

        idx = np.asarray(indices, dtype=np.int64)
        return TransitionDataset(
            states=self.states[idx],
            actions=self.actions[idx],
            next_states=self.next_states[idx],
            terminals=self.terminals[idx],
            action_space=self.action_space,
            sample_weight=None if self.sample_weight is None else self.sample_weight[idx],
            episode_ids=None if self.episode_ids is None else self.episode_ids[idx],
            initial_states=self.initial_states,
            initial_actions=self.initial_actions,
            metadata=dict(self.metadata or {}),
        )

    def to_d3rlpy_kwargs(self, *, rewards: Array | None = None) -> dict[str, Array]:
        """Return kwargs suitable for lazy d3rlpy ``MDPDataset`` construction."""

        acts = (
            self.action_space.action_indices(self.actions, n_rows=self.n_rows)
            if self.action_space.kind == "discrete"
            else self.action_space.action_matrix(self.actions, n_rows=self.n_rows)
        )
        rew = np.zeros(self.n_rows, dtype=np.float64) if rewards is None else as_1d_float(rewards, "rewards", n_rows=self.n_rows)
        if rew.shape[0] != self.n_rows:
            raise ValueError("rewards must contain one value per row.")
        return {
            "observations": self.states,
            "actions": acts,
            "rewards": rew,
            "terminals": self.terminals.astype(bool),
        }

    def to_scope_rl_logged_dataset(self, *, rewards: Array | None = None) -> dict[str, Any]:
        """Return a minimal SCOPE-RL-style logged dataset payload."""

        if self.action_space.kind != "discrete":
            raise ValueError("SCOPE-RL logged dataset conversion currently supports discrete actions.")
        idx = self.action_space.action_indices(self.actions, n_rows=self.n_rows)
        rew = np.zeros(self.n_rows, dtype=np.float64) if rewards is None else as_1d_float(rewards, "rewards", n_rows=self.n_rows)
        terminals = self.terminals.astype(bool)
        if self.episode_ids is None:
            n_trajectories = self.n_rows
            horizon = 1
            state = self.states[:, None, :]
            action = idx.reshape(-1, 1)
            reward = rew.reshape(-1, 1)
            done = terminals.reshape(-1, 1)
        else:
            self.validate_ordered_episodes(strict_terminal=True)
            slices = _episode_slices(self.episode_ids)
            lengths = [stop - start for start, stop in slices]
            if len(set(lengths)) != 1:
                raise ValueError("SCOPE-RL conversion requires equal-length ordered episodes.")
            n_trajectories = len(slices)
            horizon = int(lengths[0])
            state = np.stack([self.states[start:stop] for start, stop in slices], axis=0)
            action = np.stack([idx[start:stop] for start, stop in slices], axis=0)
            reward = np.stack([rew[start:stop] for start, stop in slices], axis=0)
            done = np.stack([terminals[start:stop] for start, stop in slices], axis=0)
        return {
            "size": int(n_trajectories),
            "n_trajectories": int(n_trajectories),
            "step_per_trajectory": int(horizon),
            "action_type": "discrete",
            "n_actions": int(self.action_space.n_actions),
            "state_dim": int(self.states.shape[1]),
            "state": state,
            "action": action,
            "reward": reward,
            "done": done,
            "terminal": done,
            "pscore": np.ones((int(n_trajectories), int(horizon)), dtype=np.float64),
            "behavior_policy": "genpqr_behavior",
            "dataset_id": 0,
        }

    def make_folds(self, *, n_folds: int, seed: int = 123, episode_respecting: bool = True) -> list[tuple[Array, Array]]:
        """Create deterministic train/test row splits."""

        if n_folds < 2:
            raise ValueError("n_folds must be at least 2.")
        if n_folds > self.n_rows:
            raise ValueError("n_folds cannot exceed the number of rows.")
        rng = np.random.default_rng(int(seed))
        if episode_respecting and self.episode_ids is not None:
            units = np.unique(self.episode_ids)
            if n_folds > units.shape[0]:
                raise ValueError("n_folds cannot exceed the number of episodes when episode_respecting=True.")
            shuffled_units = np.array(units, copy=True)
            rng.shuffle(shuffled_units)
            fold_units = np.array_split(shuffled_units, int(n_folds))
            folds = []
            all_idx = np.arange(self.n_rows)
            for held_out_units in fold_units:
                test_mask = np.isin(self.episode_ids, held_out_units)
                test_idx = all_idx[test_mask]
                train_idx = all_idx[~test_mask]
                if np.intersect1d(self.episode_ids[train_idx], self.episode_ids[test_idx]).size:
                    raise RuntimeError("episode-respecting folds leaked an episode across train/test.")
                folds.append((train_idx, test_idx))
            return folds
        indices = np.arange(self.n_rows)
        rng.shuffle(indices)
        fold_indices = np.array_split(indices, int(n_folds))
        folds = []
        for test_idx in fold_indices:
            train_idx = np.setdiff1d(indices, test_idx, assume_unique=False)
            folds.append((train_idx, np.asarray(test_idx, dtype=np.int64)))
        return folds


@dataclass(frozen=True)
class EpisodeDataset:
    """Trajectory-preserving dataset container."""

    episodes: tuple[TransitionDataset, ...]
    action_space: ActionSpaceSpec
    metadata: dict[str, Any] | None = None

    @classmethod
    def from_episodes(
        cls,
        episodes: Sequence[Mapping[str, Array]],
        *,
        action_space: ActionSpaceSpec | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "EpisodeDataset":
        """Create an episode dataset from dictionaries with transition arrays."""

        if not episodes:
            raise ValueError("episodes must be nonempty.")
        spec = action_space
        validated: list[TransitionDataset] = []
        for episode_id, episode in enumerate(episodes):
            if spec is None:
                spec = ActionSpaceSpec.infer(episode["actions"])
            dataset = TransitionDataset.from_arrays(
                states=episode["states"],
                actions=episode["actions"],
                next_states=episode["next_states"],
                terminals=episode.get("terminals"),
                action_space=spec,
                sample_weight=episode.get("sample_weight"),
                episode_ids=np.full(np.asarray(episode["states"]).shape[0], episode_id),
                initial_states=episode.get("initial_states"),
                initial_actions=episode.get("initial_actions"),
                metadata={"episode_index": episode_id},
                strict_episodes=True,
            )
            validated.append(dataset)
        if spec is None:
            raise ValueError("could not infer action space from episodes.")
        return cls(episodes=tuple(validated), action_space=spec, metadata=dict(metadata or {}))

    @property
    def n_rows(self) -> int:
        """Total number of transition rows."""

        return int(sum(ep.n_rows for ep in self.episodes))

    def to_transition_dataset(self) -> TransitionDataset:
        """Flatten episodes into a row-wise transition dataset with episode ids."""

        states = np.concatenate([ep.states for ep in self.episodes], axis=0)
        actions = np.concatenate([ep.actions for ep in self.episodes], axis=0)
        next_states = np.concatenate([ep.next_states for ep in self.episodes], axis=0)
        terminals = np.concatenate([ep.terminals for ep in self.episodes], axis=0)
        weights = [ep.sample_weight for ep in self.episodes]
        sample_weight = None if all(w is None for w in weights) else np.concatenate([np.ones(ep.n_rows) if ep.sample_weight is None else ep.sample_weight for ep in self.episodes])
        episode_ids = np.concatenate(
            [np.full(ep.n_rows, idx, dtype=np.int64) for idx, ep in enumerate(self.episodes)],
            axis=0,
        )
        initial_states = _collect_initial_states(self.episodes)
        initial_actions = _collect_initial_actions(self.episodes)
        return TransitionDataset.from_arrays(
            states=states,
            actions=actions,
            next_states=next_states,
            terminals=terminals,
            action_space=self.action_space,
            sample_weight=sample_weight,
            episode_ids=episode_ids,
            initial_states=initial_states,
            initial_actions=initial_actions,
            metadata=dict(self.metadata or {}),
            strict_episodes=True,
        )

    def to_d3rlpy_kwargs(self, *, rewards: Array | None = None) -> dict[str, Array]:
        """Return kwargs suitable for lazy d3rlpy ``MDPDataset`` construction."""

        return self.to_transition_dataset().to_d3rlpy_kwargs(rewards=rewards)


def ensure_transition_dataset(
    *,
    dataset: TransitionDataset | EpisodeDataset | None = None,
    states: Array | None = None,
    actions: Array | None = None,
    next_states: Array | None = None,
    terminals: Array | None = None,
    action_space: ActionSpaceSpec | None = None,
    sample_weight: Array | None = None,
    episode_ids: Array | None = None,
    initial_states: Array | None = None,
    initial_actions: Array | None = None,
) -> TransitionDataset:
    """Resolve either a dataset object or legacy array inputs."""

    if dataset is not None:
        conflicts = {
            "states": states,
            "actions": actions,
            "next_states": next_states,
            "terminals": terminals,
            "action_space": action_space,
            "sample_weight": sample_weight,
            "episode_ids": episode_ids,
            "initial_states": initial_states,
            "initial_actions": initial_actions,
        }
        supplied = [name for name, value in conflicts.items() if value is not None]
        if supplied:
            raise ValueError("Pass either dataset or row-wise inputs, not both: " + ", ".join(supplied))
        if isinstance(dataset, EpisodeDataset):
            return dataset.to_transition_dataset()
        if isinstance(dataset, TransitionDataset):
            return dataset
        raise TypeError("dataset must be a TransitionDataset or EpisodeDataset.")
    if states is None or actions is None or next_states is None:
        raise ValueError("states, actions, and next_states are required when dataset is omitted.")
    return TransitionDataset.from_arrays(
        states=states,
        actions=actions,
        next_states=next_states,
        terminals=terminals,
        action_space=action_space,
        sample_weight=sample_weight,
        episode_ids=episode_ids,
        initial_states=initial_states,
        initial_actions=initial_actions,
    )


def _get_mapping_or_attr(obj: Mapping[str, Any] | object, name: str, *, default: Any = ...):
    if isinstance(obj, Mapping):
        if name in obj:
            return obj[name]
        if default is not ...:
            return default
        raise KeyError(name)
    if hasattr(obj, name):
        return getattr(obj, name)
    if default is not ...:
        return default
    raise AttributeError(name)


def _shift_next_states(states: Array, *, terminals: Array | None, episode_ids: Array | None) -> Array:
    states_2d = as_2d_float(states, "states")
    next_states = np.array(states_2d, copy=True)
    if states_2d.shape[0] > 1:
        next_states[:-1] = states_2d[1:]
    if terminals is not None:
        done = np.asarray(terminals).reshape(-1).astype(bool)
        next_states[done] = states_2d[done]
    if episode_ids is not None:
        ids = np.asarray(episode_ids).reshape(-1)
        boundary = np.zeros(ids.shape[0], dtype=bool)
        boundary[:-1] = ids[1:] != ids[:-1]
        next_states[boundary] = states_2d[boundary]
    return next_states


def _episode_slices(episode_ids: Array) -> list[tuple[int, int]]:
    ids = np.asarray(episode_ids).reshape(-1)
    if ids.shape[0] == 0:
        return []
    starts = [0]
    for i in range(1, ids.shape[0]):
        if ids[i] != ids[i - 1]:
            starts.append(i)
    starts.append(ids.shape[0])
    return [(int(start), int(stop)) for start, stop in zip(starts[:-1], starts[1:])]


def _flatten_scope_rl_logged_dataset(logged_dataset: Mapping[str, Any]) -> tuple[Array, Array, Array, Array]:
    states = np.asarray(logged_dataset["state"], dtype=np.float64)
    actions = np.asarray(logged_dataset["action"])
    terminal_key = "done" if "done" in logged_dataset else "terminal"
    terminals = np.asarray(logged_dataset.get(terminal_key, np.zeros(actions.shape[:2])), dtype=np.float64)
    if states.ndim == 3:
        n_traj, horizon, state_dim = states.shape
        flat_states = states.reshape(n_traj * horizon, state_dim)
        flat_actions = actions.reshape(n_traj * horizon, *actions.shape[2:]).reshape(n_traj * horizon, -1)
        if flat_actions.shape[1] == 1:
            flat_actions = flat_actions.reshape(-1)
        flat_terminals = terminals.reshape(n_traj * horizon)
        episode_ids = np.repeat(np.arange(n_traj, dtype=np.int64), horizon)
        return flat_states, flat_actions, flat_terminals, episode_ids
    if states.ndim == 2:
        flat_actions = actions.reshape(states.shape[0], -1)
        if flat_actions.shape[1] == 1:
            flat_actions = flat_actions.reshape(-1)
        return states, flat_actions, terminals.reshape(-1), np.arange(states.shape[0], dtype=np.int64)
    raise ValueError("SCOPE-RL state must have shape (n, d) or (n_trajectories, horizon, d).")


def _collect_initial_states(episodes: Sequence[TransitionDataset]) -> Array | None:
    values = []
    for episode in episodes:
        if episode.initial_states is not None:
            values.append(episode.initial_states[0])
        else:
            values.append(episode.states[0])
    return np.asarray(values, dtype=np.float64)


def _collect_initial_actions(episodes: Sequence[TransitionDataset]) -> Array | None:
    values = []
    for episode in episodes:
        if episode.initial_actions is not None:
            values.append(np.asarray(episode.initial_actions)[0])
        else:
            values.append(np.asarray(episode.actions)[0])
    return np.asarray(values)
