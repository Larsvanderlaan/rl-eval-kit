from __future__ import annotations

import numpy as np
import pytest

from genpqr import ActionSpaceSpec, EpisodeDataset, TransitionDataset


def test_episode_dataset_preserves_initial_payloads_and_summary() -> None:
    episodes = EpisodeDataset.from_episodes(
        [
            {
                "states": np.array([[0.0], [1.0]]),
                "actions": np.array([0, 1]),
                "next_states": np.array([[1.0], [1.0]]),
                "terminals": np.array([0.0, 1.0]),
                "initial_states": np.array([[0.0]]),
                "initial_actions": np.array([0]),
            },
            {
                "states": np.array([[2.0], [3.0]]),
                "actions": np.array([1, 0]),
                "next_states": np.array([[3.0], [3.0]]),
                "terminals": np.array([0.0, 1.0]),
            },
        ],
        action_space=ActionSpaceSpec.discrete(2),
    )
    flat = episodes.to_transition_dataset()
    assert flat.initial_states.shape == (2, 1)
    assert np.array_equal(flat.initial_actions, np.array([0, 1]))
    assert flat.metadata["ordered_episodes_validated"] is True
    summary = flat.summary()
    assert summary["n_rows"] == 4
    assert summary["n_episodes"] == 2
    assert summary["actions"]["counts"] == [2, 2]


def test_strict_episode_validation_rejects_noncontiguous_ids_and_early_terminal() -> None:
    with pytest.raises(ValueError, match="contiguous"):
        TransitionDataset.from_arrays(
            states=np.zeros((4, 1)),
            actions=np.array([0, 1, 0, 1]),
            next_states=np.zeros((4, 1)),
            terminals=np.array([0.0, 0.0, 0.0, 1.0]),
            action_space=ActionSpaceSpec.discrete(2),
            episode_ids=np.array([0, 1, 0, 1]),
            strict_episodes=True,
        )
    with pytest.raises(ValueError, match="final row"):
        TransitionDataset.from_arrays(
            states=np.zeros((3, 1)),
            actions=np.array([0, 1, 0]),
            next_states=np.zeros((3, 1)),
            terminals=np.array([1.0, 0.0, 0.0]),
            action_space=ActionSpaceSpec.discrete(2),
            episode_ids=np.array([0, 0, 0]),
            strict_episodes=True,
        )
    with pytest.raises(ValueError, match="final row"):
        TransitionDataset.from_arrays(
            states=np.array([[0.0], [1.0]]),
            actions=np.array([0, 1]),
            next_states=np.array([[1.0], [1.0]]),
            terminals=np.array([0.0, 0.0]),
            action_space=ActionSpaceSpec.discrete(2),
            episode_ids=np.array([0, 0]),
            strict_episodes=True,
        )
    with pytest.raises(ValueError, match="next_states"):
        TransitionDataset.from_arrays(
            states=np.array([[0.0], [1.0]]),
            actions=np.array([0, 1]),
            next_states=np.array([[0.0], [1.0]]),
            terminals=np.array([0.0, 1.0]),
            action_space=ActionSpaceSpec.discrete(2),
            episode_ids=np.array([0, 0]),
            strict_episodes=True,
        )


def test_dataset_converters_without_optional_imports() -> None:
    d3 = TransitionDataset.from_d3rlpy(
        {
            "observations": np.array([[0.0], [1.0], [2.0]]),
            "actions": np.array([0, 1, 0]),
            "terminals": np.array([0.0, 0.0, 1.0]),
            "episode_ids": np.array([0, 0, 0]),
        },
        action_space=ActionSpaceSpec.discrete(2),
    )
    assert d3.next_states[-1, 0] == 2.0
    assert d3.metadata["source"] == "d3rlpy"

    scope = TransitionDataset.from_scope_rl(
        {
            "state": np.array([[[0.0], [1.0]], [[2.0], [3.0]]]),
            "action": np.array([[0, 1], [1, 0]]),
            "done": np.array([[0.0, 1.0], [0.0, 1.0]]),
            "n_actions": 2,
        }
    )
    assert scope.n_rows == 4
    assert np.array_equal(scope.episode_ids, np.array([0, 0, 1, 1]))
    assert scope.to_d3rlpy_kwargs()["actions"].shape == (4,)
    payload = scope.to_scope_rl_logged_dataset()
    assert payload["state"].shape == (2, 2, 1)
    assert payload["n_trajectories"] == 2
    assert payload["step_per_trajectory"] == 2


def test_scope_rl_conversion_rejects_ragged_episodes() -> None:
    dataset = TransitionDataset.from_arrays(
        states=np.array([[0.0], [1.0], [2.0]]),
        actions=np.array([0, 1, 0]),
        next_states=np.array([[1.0], [1.0], [2.0]]),
        terminals=np.array([0.0, 1.0, 1.0]),
        action_space=ActionSpaceSpec.discrete(2),
        episode_ids=np.array([0, 0, 1]),
        strict_episodes=True,
    )
    with pytest.raises(ValueError, match="equal-length"):
        dataset.to_scope_rl_logged_dataset()


def test_dataset_object_rejects_conflicting_inputs() -> None:
    dataset = TransitionDataset.from_arrays(
        states=np.array([[0.0]]),
        actions=np.array([0]),
        next_states=np.array([[0.0]]),
        terminals=np.array([1.0]),
        action_space=ActionSpaceSpec.discrete(2),
    )
    from genpqr.datasets import ensure_transition_dataset

    with pytest.raises(ValueError, match="terminals"):
        ensure_transition_dataset(dataset=dataset, terminals=np.array([1.0]))
