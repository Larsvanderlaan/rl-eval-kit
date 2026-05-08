from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from occupancy_ratio_benchmark.data import BenchmarkDataset, one_hot


Array = np.ndarray


class OptionalDatasetUnavailable(RuntimeError):
    """Raised when an optional benchmark dataset cannot be loaded."""


@dataclass(frozen=True)
class TabularPolicies:
    behavior: Array
    target: Array


def make_openml_contextual_bandit_dataset(
    *,
    task_id: int,
    gamma: float,
    sample_size: int,
    seed: int,
) -> BenchmarkDataset:
    features, labels, dataset_name = _load_openml_task(int(task_id))
    return make_openml_contextual_bandit_from_arrays(
        features=features,
        labels=labels,
        gamma=gamma,
        sample_size=sample_size,
        seed=seed,
        task_id=task_id,
        dataset_name=dataset_name,
    )


def make_openml_finite_mdp_dataset(
    *,
    task_id: int,
    gamma: float,
    sample_size: int,
    seed: int,
    state_cap: int,
) -> BenchmarkDataset:
    features, labels, dataset_name = _load_openml_task(int(task_id))
    return make_openml_finite_mdp_from_arrays(
        features=features,
        labels=labels,
        gamma=gamma,
        sample_size=sample_size,
        seed=seed,
        task_id=task_id,
        dataset_name=dataset_name,
        state_cap=state_cap,
    )


def make_openml_contextual_bandit_from_arrays(
    *,
    features: Array,
    labels: Array,
    gamma: float,
    sample_size: int,
    seed: int,
    task_id: int | str = "fixture",
    dataset_name: str = "fixture",
) -> BenchmarkDataset:
    rng = np.random.default_rng(seed)
    x, y, n_actions = _prepare_tabular_arrays(features, labels)
    policies = _synthetic_policies(x, n_actions)
    n = x.shape[0]
    state_idx = rng.integers(0, n, size=int(sample_size))
    states = x[state_idx]
    behavior_probs = policies.behavior[state_idx]
    target_probs = policies.target[state_idx]
    actions_i = _sample_categorical(behavior_probs, rng)
    target_actions_i = _sample_categorical(target_probs, rng)
    next_idx = rng.integers(0, n, size=int(sample_size))
    next_states = x[next_idx]
    next_target_actions_i = _sample_categorical(policies.target[next_idx], rng)
    initial_idx = rng.integers(0, n, size=max(256, min(2_000, int(sample_size))))
    initial_actions_i = _sample_categorical(policies.target[initial_idx], rng)
    rewards = (actions_i == y[state_idx]).astype(np.float64)
    action_ratio = target_probs[np.arange(state_idx.shape[0]), actions_i] / np.maximum(
        behavior_probs[np.arange(state_idx.shape[0]), actions_i],
        1e-12,
    )
    return BenchmarkDataset(
        setting="openml_contextual_bandit",
        states=states,
        actions=one_hot(actions_i, n_actions),
        next_states=next_states,
        target_actions=one_hot(target_actions_i, n_actions),
        next_target_actions=one_hot(next_target_actions_i, n_actions),
        rewards=rewards,
        true_ratio=action_ratio,
        true_action_ratio=action_ratio,
        true_transition_ratio=np.ones(int(sample_size), dtype=np.float64),
        initial_states=x[initial_idx],
        initial_actions=one_hot(initial_actions_i, n_actions),
        initial_weights=np.ones(initial_idx.shape[0], dtype=np.float64),
        masks=np.ones(int(sample_size), dtype=np.float64),
        gamma=float(gamma),
        seed=int(seed),
        sample_size=int(sample_size),
        metadata={
            "dataset_variant": str(task_id),
            "openml_task_id": task_id,
            "openml_dataset_name": str(dataset_name),
            "truth_source": "synthetic_contextual_bandit_propensity",
            "reference_distribution": "empirical_context_distribution",
            "n_actions": int(n_actions),
            "state_dim": int(x.shape[1]),
        },
    )


def make_openml_finite_mdp_from_arrays(
    *,
    features: Array,
    labels: Array,
    gamma: float,
    sample_size: int,
    seed: int,
    task_id: int | str = "fixture",
    dataset_name: str = "fixture",
    state_cap: int = 256,
) -> BenchmarkDataset:
    rng = np.random.default_rng(seed)
    x, y, n_actions = _prepare_tabular_arrays(features, labels)
    state_cap = min(int(state_cap), x.shape[0])
    if state_cap <= 1:
        raise ValueError("state_cap must leave at least two tabular states.")
    selected = np.sort(rng.choice(x.shape[0], size=state_cap, replace=False))
    x_cap = x[selected]
    y_cap = y[selected]
    policies = _synthetic_policies(x_cap, n_actions)
    transition = build_knn_transition_matrix(x_cap, n_actions=n_actions)
    reference_state = np.ones(state_cap, dtype=np.float64) / state_cap
    target_state = solve_discounted_occupancy(transition, policies.target, reference_state, gamma)
    target_joint = target_state[:, None] * policies.target
    reference_joint = reference_state[:, None] * policies.behavior
    ratio_table = target_joint / np.maximum(reference_joint, 1e-12)

    state_idx = rng.integers(0, state_cap, size=int(sample_size))
    actions_i = _sample_categorical(policies.behavior[state_idx], rng)
    next_idx = _sample_categorical(transition[state_idx, actions_i], rng)
    target_actions_i = _sample_categorical(policies.target[state_idx], rng)
    next_target_actions_i = _sample_categorical(policies.target[next_idx], rng)
    initial_idx = rng.integers(0, state_cap, size=max(256, min(2_000, int(sample_size))))
    initial_actions_i = _sample_categorical(policies.target[initial_idx], rng)
    rewards = (actions_i == y_cap[state_idx]).astype(np.float64)
    true_ratio = ratio_table[state_idx, actions_i]
    true_action_ratio = policies.target[state_idx, actions_i] / np.maximum(policies.behavior[state_idx, actions_i], 1e-12)
    true_transition_ratio = transition[state_idx, actions_i, next_idx] / np.maximum(reference_state[next_idx], 1e-12)

    return BenchmarkDataset(
        setting="openml_finite_mdp",
        states=x_cap[state_idx],
        actions=one_hot(actions_i, n_actions),
        next_states=x_cap[next_idx],
        target_actions=one_hot(target_actions_i, n_actions),
        next_target_actions=one_hot(next_target_actions_i, n_actions),
        rewards=rewards,
        true_ratio=true_ratio,
        true_action_ratio=true_action_ratio,
        true_transition_ratio=true_transition_ratio,
        initial_states=x_cap[initial_idx],
        initial_actions=one_hot(initial_actions_i, n_actions),
        initial_weights=np.ones(initial_idx.shape[0], dtype=np.float64),
        masks=np.ones(int(sample_size), dtype=np.float64),
        gamma=float(gamma),
        seed=int(seed),
        sample_size=int(sample_size),
        metadata={
            "dataset_variant": str(task_id),
            "openml_task_id": task_id,
            "openml_dataset_name": str(dataset_name),
            "truth_source": "synthetic_finite_mdp_linear_solve",
            "reference_distribution": "empirical_uniform_tabular_states",
            "n_states": int(state_cap),
            "n_actions": int(n_actions),
            "state_dim": int(x_cap.shape[1]),
        },
    )


def make_obp_logged_bandit_dataset(
    *,
    campaign: str,
    gamma: float,
    sample_size: int,
    seed: int,
) -> BenchmarkDataset:
    try:
        from obp.dataset import OpenBanditDataset
    except Exception as exc:  # pragma: no cover - depends on optional package
        raise OptionalDatasetUnavailable("Install the tabular-benchmark extra to use OBP settings.") from exc
    try:
        dataset = OpenBanditDataset(behavior_policy="bts", campaign=str(campaign))
        feedback = dataset.obtain_batch_bandit_feedback()
    except Exception as exc:  # pragma: no cover - depends on local OBP data/cache
        raise OptionalDatasetUnavailable(f"OBP campaign '{campaign}' is unavailable: {exc}") from exc
    context = np.asarray(feedback["context"], dtype=np.float64)
    action = np.asarray(feedback["action"], dtype=np.int64).reshape(-1)
    reward = np.asarray(feedback["reward"], dtype=np.float64).reshape(-1)
    pscore = np.asarray(feedback["pscore"], dtype=np.float64).reshape(-1)
    n_actions = int(np.max(action)) + 1
    rng = np.random.default_rng(seed)
    idx = rng.choice(context.shape[0], size=int(sample_size), replace=context.shape[0] < int(sample_size))
    states = _standardize_features(context)[idx]
    actions_i = action[idx]
    rewards = reward[idx]
    target_policy = _synthetic_policies(states, n_actions).target
    target_actions_i = _sample_categorical(target_policy, rng)
    next_idx = rng.choice(context.shape[0], size=int(sample_size), replace=True)
    next_states = _standardize_features(context)[next_idx]
    next_target_actions_i = _sample_categorical(_synthetic_policies(next_states, n_actions).target, rng)
    initial_idx = rng.choice(context.shape[0], size=max(256, min(2_000, int(sample_size))), replace=True)
    initial_states = _standardize_features(context)[initial_idx]
    initial_actions_i = _sample_categorical(_synthetic_policies(initial_states, n_actions).target, rng)
    true_ratio = target_policy[np.arange(idx.shape[0]), actions_i] / np.maximum(pscore[idx], 1e-12)
    return BenchmarkDataset(
        setting="obp_logged_bandit",
        states=states,
        actions=one_hot(actions_i, n_actions),
        next_states=next_states,
        target_actions=one_hot(target_actions_i, n_actions),
        next_target_actions=one_hot(next_target_actions_i, n_actions),
        rewards=rewards,
        true_ratio=true_ratio,
        true_action_ratio=true_ratio,
        true_transition_ratio=np.ones(int(sample_size), dtype=np.float64),
        initial_states=initial_states,
        initial_actions=one_hot(initial_actions_i, n_actions),
        initial_weights=np.ones(initial_states.shape[0], dtype=np.float64),
        masks=np.ones(int(sample_size), dtype=np.float64),
        gamma=float(gamma),
        seed=int(seed),
        sample_size=int(sample_size),
        metadata={
            "dataset_variant": str(campaign),
            "obp_campaign": str(campaign),
            "truth_source": "obp_logged_pscore_synthetic_target",
            "reference_distribution": "logged_context_distribution",
            "n_actions": int(n_actions),
            "state_dim": int(states.shape[1]),
        },
    )


def make_minari_dataset(
    *,
    setting: str,
    dataset_id: str,
    gamma: float,
    sample_size: int,
    seed: int,
) -> BenchmarkDataset:
    try:
        import minari
    except Exception as exc:  # pragma: no cover - depends on optional package
        raise OptionalDatasetUnavailable("Install the tabular-benchmark extra to use Minari settings.") from exc
    try:
        dataset = minari.load_dataset(str(dataset_id))
    except Exception as exc:  # pragma: no cover - depends on local Minari data/cache
        raise OptionalDatasetUnavailable(f"Minari dataset '{dataset_id}' is unavailable: {exc}") from exc
    transitions = list(_iter_minari_transitions(dataset))
    if not transitions:
        raise OptionalDatasetUnavailable(f"Minari dataset '{dataset_id}' did not expose any transitions.")
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(transitions), size=int(sample_size), replace=len(transitions) < int(sample_size))
    obs = np.asarray([transitions[i][0] for i in idx], dtype=np.float64)
    act_raw = [transitions[i][1] for i in idx]
    rew = np.asarray([transitions[i][2] for i in idx], dtype=np.float64)
    next_obs = np.asarray([transitions[i][3] for i in idx], dtype=np.float64)
    done = np.asarray([transitions[i][4] for i in idx], dtype=bool)
    actions, target_actions, next_target_actions = _minari_action_arrays(act_raw, sample_size=int(sample_size), rng=rng)
    init_idx = rng.choice(len(transitions), size=max(256, min(2_000, int(sample_size))), replace=True)
    initial_states = np.asarray([transitions[i][0] for i in init_idx], dtype=np.float64)
    initial_actions, _, _ = _minari_action_arrays(
        [transitions[i][1] for i in init_idx],
        sample_size=init_idx.shape[0],
        rng=rng,
    )
    return BenchmarkDataset(
        setting=str(setting),
        states=_standardize_features(obs),
        actions=actions,
        next_states=_standardize_features(next_obs),
        target_actions=target_actions,
        next_target_actions=next_target_actions,
        rewards=rew,
        true_ratio=None,
        initial_states=_standardize_features(initial_states),
        initial_actions=initial_actions,
        initial_weights=np.ones(initial_states.shape[0], dtype=np.float64),
        masks=(~done).astype(np.float64),
        gamma=float(gamma),
        seed=int(seed),
        sample_size=int(sample_size),
        metadata={
            "dataset_variant": str(dataset_id),
            "minari_dataset_id": str(dataset_id),
            "truth_source": "unavailable_real_offline_rl",
            "reference_distribution": "logged_minari_transitions",
            "state_dim": int(obs.shape[1]),
            "action_dim": int(actions.reshape(actions.shape[0], -1).shape[1]),
        },
    )


def build_knn_transition_matrix(features: Array, *, n_actions: int, k: int = 5) -> Array:
    x = np.asarray(features, dtype=np.float64)
    n = x.shape[0]
    k_eff = max(1, min(int(k), n))
    transition = np.zeros((n, int(n_actions), n), dtype=np.float64)
    directions = _action_directions(x.shape[1], int(n_actions))
    for state in range(n):
        for action in range(int(n_actions)):
            anchor = x[state] + 0.35 * directions[action]
            distances = np.sum((x - anchor) ** 2, axis=1)
            nn = np.argpartition(distances, k_eff - 1)[:k_eff]
            weights = np.exp(-distances[nn] / max(float(np.median(distances[nn]) + 1e-8), 1e-8))
            weights = weights / np.sum(weights)
            transition[state, action, nn] = weights
    return transition


def solve_discounted_occupancy(transition: Array, target_policy: Array, reference_state: Array, gamma: float) -> Array:
    p_pi = np.einsum("sa,san->sn", np.asarray(target_policy, dtype=np.float64), np.asarray(transition, dtype=np.float64))
    system = np.eye(p_pi.shape[0], dtype=np.float64) - float(gamma) * p_pi.T
    rhs = (1.0 - float(gamma)) * np.asarray(reference_state, dtype=np.float64).reshape(-1)
    occupancy = np.linalg.solve(system, rhs)
    occupancy = np.maximum(occupancy, 0.0)
    total = float(np.sum(occupancy))
    return occupancy / total if total > 0.0 else np.ones_like(occupancy) / occupancy.shape[0]


def _load_openml_task(task_id: int) -> tuple[Array, Array, str]:
    try:
        import openml
    except Exception as exc:  # pragma: no cover - depends on optional package
        raise OptionalDatasetUnavailable("Install the tabular-benchmark extra to use OpenML settings.") from exc
    try:
        task = openml.tasks.get_task(int(task_id), download_data=True)
        dataset = task.get_dataset()
        target_name = task.target_name
        x, y, _, _ = dataset.get_data(target=target_name, dataset_format="dataframe")
        return _dataframe_to_numpy(x), np.asarray(y), str(dataset.name)
    except Exception as exc:  # pragma: no cover - depends on network/cache
        raise OptionalDatasetUnavailable(f"OpenML task {task_id} is unavailable: {exc}") from exc


def _prepare_tabular_arrays(features: Array, labels: Array) -> tuple[Array, Array, int]:
    x = _standardize_features(_dataframe_to_numpy(features))
    y_raw = np.asarray(labels).reshape(-1)
    if x.shape[0] != y_raw.shape[0]:
        raise ValueError("features and labels must have matching rows.")
    _, y = np.unique(y_raw.astype(str), return_inverse=True)
    n_actions = max(2, min(4, int(np.max(y)) + 1))
    return x, np.asarray(y % n_actions, dtype=np.int64), n_actions


def _dataframe_to_numpy(value: Any) -> Array:
    if hasattr(value, "select_dtypes"):
        numeric = value.select_dtypes(include=["number", "bool"])
        other = value.drop(columns=list(numeric.columns))
        if getattr(other, "shape", (0, 0))[1]:
            other = other.astype(str)
            try:
                import pandas as pd

                encoded = pd.get_dummies(other, dummy_na=True)
                value = np.concatenate([numeric.to_numpy(), encoded.to_numpy(dtype=np.float64)], axis=1)
            except Exception:
                value = numeric.to_numpy()
        else:
            value = numeric.to_numpy()
    arr = np.asarray(value)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        arr = arr.reshape(arr.shape[0], -1)
    return arr.astype(np.float64, copy=False)


def _standardize_features(features: Array) -> Array:
    x = np.asarray(features, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    mean = np.mean(x, axis=0, keepdims=True)
    sd = np.std(x, axis=0, keepdims=True)
    return (x - mean) / np.maximum(sd, 1e-6)


def _synthetic_policies(features: Array, n_actions: int) -> TabularPolicies:
    x = np.asarray(features, dtype=np.float64)
    directions = _action_directions(x.shape[1], int(n_actions)).T
    base_logits = 0.35 * x @ directions
    target_logits = base_logits + 0.70 * np.tanh(x[:, :1] @ np.linspace(-1.0, 1.0, int(n_actions)).reshape(1, -1))
    behavior = 0.90 * _softmax(base_logits) + 0.10 / int(n_actions)
    target = 0.90 * _softmax(target_logits) + 0.10 / int(n_actions)
    return TabularPolicies(behavior=behavior / behavior.sum(axis=1, keepdims=True), target=target / target.sum(axis=1, keepdims=True))


def _action_directions(dim: int, n_actions: int) -> Array:
    rng = np.random.default_rng(91_337 + int(dim) * 31 + int(n_actions))
    directions = rng.normal(size=(int(n_actions), int(dim)))
    directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-8)
    return directions


def _softmax(logits: Array) -> Array:
    z = np.asarray(logits, dtype=np.float64)
    z = z - np.max(z, axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / np.sum(exp, axis=1, keepdims=True)


def _sample_categorical(probs: Array, rng: np.random.Generator) -> Array:
    p = np.asarray(probs, dtype=np.float64)
    cdf = np.cumsum(p / np.sum(p, axis=1, keepdims=True), axis=1)
    return (rng.random(size=p.shape[0])[:, None] > cdf[:, :-1]).sum(axis=1).astype(np.int64)


def _iter_minari_transitions(dataset: Any) -> Iterable[tuple[Array, Any, float, Array, bool]]:
    episodes = dataset.iterate_episodes() if hasattr(dataset, "iterate_episodes") else dataset
    for episode in episodes:
        observations = getattr(episode, "observations", None)
        actions = getattr(episode, "actions", None)
        rewards = getattr(episode, "rewards", None)
        terminations = getattr(episode, "terminations", None)
        truncations = getattr(episode, "truncations", None)
        if observations is None and isinstance(episode, dict):
            observations = episode.get("observations")
            actions = episode.get("actions")
            rewards = episode.get("rewards")
            terminations = episode.get("terminations")
            truncations = episode.get("truncations")
        if observations is None or actions is None or rewards is None:
            continue
        obs_arr = [_flatten_observation(obs) for obs in observations]
        done = np.zeros(len(rewards), dtype=bool)
        if terminations is not None:
            done |= np.asarray(terminations, dtype=bool).reshape(-1)[: len(rewards)]
        if truncations is not None:
            done |= np.asarray(truncations, dtype=bool).reshape(-1)[: len(rewards)]
        for i in range(min(len(rewards), len(actions), len(obs_arr) - 1)):
            yield obs_arr[i], actions[i], float(rewards[i]), obs_arr[i + 1], bool(done[i])


def _flatten_observation(obs: Any) -> Array:
    if isinstance(obs, dict):
        parts = []
        for key in sorted(obs):
            try:
                arr = np.asarray(obs[key], dtype=np.float64).reshape(-1)
            except (TypeError, ValueError):
                continue
            parts.append(arr)
        return np.concatenate(parts) if parts else np.zeros(1, dtype=np.float64)
    return np.asarray(obs, dtype=np.float64).reshape(-1)


def _minari_action_arrays(
    actions: list[Any],
    *,
    sample_size: int,
    rng: np.random.Generator,
) -> tuple[Array, Array, Array]:
    arr = np.asarray(actions)
    if np.issubdtype(arr.dtype, np.integer) or (arr.ndim == 1 and np.all(np.equal(arr, np.round(arr)))):
        idx = np.asarray(arr, dtype=np.int64).reshape(-1)
        n_actions = int(np.max(idx)) + 1 if idx.size else 1
        target_idx = rng.integers(0, max(n_actions, 1), size=int(sample_size))
        next_target_idx = rng.integers(0, max(n_actions, 1), size=int(sample_size))
        return one_hot(idx, n_actions), one_hot(target_idx, n_actions), one_hot(next_target_idx, n_actions)
    continuous = np.asarray(arr, dtype=np.float64).reshape(int(sample_size), -1)
    scale = np.maximum(np.std(continuous, axis=0, keepdims=True), 1e-3)
    target = continuous + 0.10 * scale * rng.normal(size=continuous.shape)
    next_target = continuous + 0.10 * scale * rng.normal(size=continuous.shape)
    return continuous, target, next_target
