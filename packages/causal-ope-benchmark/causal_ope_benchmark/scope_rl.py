from __future__ import annotations

from typing import Any

import numpy as np

from causal_ope_benchmark.types import LongitudinalDataset, assert_no_forbidden_public_keys
from causal_ope_benchmark.validation import validate_longitudinal_dataset, validate_scope_rl_logged_dataset


def to_scope_rl_logged_dataset(
    dataset: LongitudinalDataset,
    *,
    behavior_policy_name: str = "behavior",
    dataset_id: int = 0,
) -> dict[str, Any]:
    """Export a public dataset to a SCOPE-RL-style logged-dataset dict.

    Variable-length episodes are padded to a fixed trajectory length. Padded
    rewards are zero, and done flags stay true after the observed episode ends.
    """

    validate_longitudinal_dataset(dataset)
    units = np.unique(np.asarray(dataset.unit_id))
    row_groups = [np.flatnonzero(np.asarray(dataset.unit_id) == unit) for unit in units]
    step_per_trajectory = max((idx.size for idx in row_groups), default=0)
    if step_per_trajectory <= 0:
        raise ValueError("dataset contains no trajectory rows.")
    n_trajectories = len(row_groups)
    state_dim = int(dataset.state_dim)
    n_actions = int(dataset.action_dim)
    states = np.zeros((n_trajectories, step_per_trajectory, state_dim), dtype=np.float64)
    actions = np.zeros((n_trajectories, step_per_trajectory), dtype=np.int64)
    rewards = np.zeros((n_trajectories, step_per_trajectory), dtype=np.float64)
    done = np.ones((n_trajectories, step_per_trajectory), dtype=bool)
    terminal = np.zeros((n_trajectories, step_per_trajectory), dtype=bool)
    pscore = np.ones((n_trajectories, step_per_trajectory), dtype=np.float64)
    action_available = np.zeros((n_trajectories, step_per_trajectory, n_actions), dtype=np.float64)
    censoring = np.zeros((n_trajectories, step_per_trajectory), dtype=np.float64)
    target_propensity_observed_action = np.zeros((n_trajectories, step_per_trajectory), dtype=np.float64)
    target_action_probabilities = np.zeros((n_trajectories, step_per_trajectory, n_actions), dtype=np.float64)
    action_idx = np.argmax(np.asarray(dataset.actions), axis=1)
    for traj_id, idx in enumerate(row_groups):
        ordered = idx[np.argsort(np.asarray(dataset.time)[idx])]
        length = int(ordered.size)
        states[traj_id, :length] = np.asarray(dataset.states)[ordered]
        actions[traj_id, :length] = action_idx[ordered]
        rewards[traj_id, :length] = np.asarray(dataset.rewards)[ordered]
        terminal[traj_id, :length] = np.asarray(dataset.terminals)[ordered] > 0.5
        censoring[traj_id, :length] = np.asarray(dataset.censoring)[ordered]
        pscore[traj_id, :length] = np.clip(np.asarray(dataset.behavior_propensity)[ordered], 1e-12, np.inf)
        action_available[traj_id, :length] = np.asarray(dataset.action_available)[ordered]
        target_propensity_observed_action[traj_id, :length] = np.asarray(dataset.target_propensity_observed_action)[ordered]
        if dataset.target_action_probabilities is not None:
            target_action_probabilities[traj_id, :length] = np.asarray(dataset.target_action_probabilities)[ordered]
        observed_done = np.maximum(np.asarray(dataset.terminals)[ordered], np.asarray(dataset.censoring)[ordered]) > 0.5
        if np.any(observed_done):
            first_done = int(np.flatnonzero(observed_done)[0])
            done[traj_id, : first_done + 1] = False
            done[traj_id, first_done:] = True
        else:
            done[traj_id, :length] = False
            if length:
                done[traj_id, length - 1] = True
        if length < step_per_trajectory:
            done[traj_id, length:] = True
    info = {
        "action_available": action_available,
        "censoring": censoring,
        "target_propensity_observed_action": target_propensity_observed_action,
        "target_action_probabilities": target_action_probabilities,
        "panel_only": bool(dataset.family == "streamlift"),
        "metadata_public": _safe_metadata(dataset),
    }
    payload = {
        "size": int(n_trajectories * step_per_trajectory),
        "n_trajectories": int(n_trajectories),
        "step_per_trajectory": int(step_per_trajectory),
        "action_type": "discrete",
        "n_actions": int(n_actions),
        "state_dim": int(state_dim),
        "state": states,
        "action": actions,
        "reward": rewards,
        "done": done,
        "terminal": terminal,
        "pscore": pscore,
        "info": info,
        "behavior_policy": str(behavior_policy_name),
        "dataset_id": int(dataset_id),
    }
    validate_scope_rl_logged_dataset(payload)
    return payload


def _safe_metadata(dataset: LongitudinalDataset) -> dict[str, Any]:
    metadata = dict(dataset.metadata_public)
    assert_no_forbidden_public_keys(metadata)
    return metadata
