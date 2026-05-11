from __future__ import annotations

from dataclasses import dataclass
import importlib
from typing import Any, Callable

import numpy as np

from occupancy_ratio_benchmark.data import BenchmarkDataset, one_hot
from occupancy_ratio_benchmark.tabular import OptionalDatasetUnavailable


Array = np.ndarray


APPLICATION_SIMULATOR_SETTINGS = {
    "rtbgym_discrete": ("rtbgym", "RTBEnv-discrete-v0"),
    "recgym_recommender": ("recgym", "RECEnv-v0"),
}


@dataclass(frozen=True)
class ApplicationSoftmaxPolicy:
    """Simple reproducible policy for application-shaped discrete simulators."""

    weights: Array
    bias: Array
    temperature: float
    epsilon: float

    def probabilities(self, states: Array) -> Array:
        states_arr = _as_2d_float(states)
        logits = states_arr @ self.weights.T + self.bias.reshape(1, -1)
        logits = logits / max(float(self.temperature), 1e-6)
        logits = logits - np.max(logits, axis=1, keepdims=True)
        probs = np.exp(logits)
        probs = probs / np.maximum(np.sum(probs, axis=1, keepdims=True), 1e-12)
        if self.epsilon > 0.0:
            probs = (1.0 - float(self.epsilon)) * probs + float(self.epsilon) / probs.shape[1]
        return probs

    def sample(self, states: Array, rng: np.random.Generator) -> Array:
        probs = self.probabilities(states)
        return _sample_categorical(probs, rng)


def make_application_simulator_dataset(
    *,
    setting: str,
    gamma: float,
    sample_size: int,
    seed: int,
    target_value_rollouts: int,
) -> BenchmarkDataset:
    """Create application-shaped simulator datasets from RTBGym/RecGym.

    These benchmarks are OPE stress tests. They intentionally do not report
    occupancy-ratio truth because the simulator state distribution is generated
    through the environment rather than by an exact transition-matrix solve.
    """

    if setting not in APPLICATION_SIMULATOR_SETTINGS:
        raise ValueError(f"Unknown application simulator setting '{setting}'.")
    package, env_id = APPLICATION_SIMULATOR_SETTINGS[setting]
    try:
        importlib.import_module(package)
    except Exception as exc:  # pragma: no cover - optional package
        raise OptionalDatasetUnavailable(f"Install `{package}` or `scope-rl` to use setting '{setting}'.") from exc
    gym_modules = []
    try:
        import gymnasium as gymnasium_module

        gym_modules.append(gymnasium_module)
    except Exception:
        pass
    try:
        import gym as gym_module

        if gym_module not in gym_modules:
            gym_modules.append(gym_module)
    except Exception:
        pass
    if not gym_modules:
        raise OptionalDatasetUnavailable("Install Gymnasium or Gym to use application simulator settings.")

    def env_factory():
        errors = []
        for gym_module in gym_modules:
            try:
                return gym_module.make(env_id)
            except Exception as exc:  # pragma: no cover - optional package/env registration
                errors.append(str(exc))
        detail = "; ".join(errors)
        raise OptionalDatasetUnavailable(f"Could not create application simulator `{env_id}`: {detail}")

    return make_application_simulator_dataset_from_env_factory(
        setting=setting,
        env_id=env_id,
        env_factory=env_factory,
        gamma=gamma,
        sample_size=sample_size,
        seed=seed,
        target_value_rollouts=target_value_rollouts,
    )


def make_application_simulator_dataset_from_env_factory(
    *,
    setting: str,
    env_id: str,
    env_factory: Callable[[], Any],
    gamma: float,
    sample_size: int,
    seed: int,
    target_value_rollouts: int,
) -> BenchmarkDataset:
    rng = np.random.default_rng(int(seed))
    env = env_factory()
    try:
        n_actions = _discrete_action_count(env)
        initial_obs = _reset_env(env, rng)
        state_dim = int(initial_obs.shape[0])
        behavior_policy, target_policy = _make_application_policies(
            setting=setting,
            state_dim=state_dim,
            n_actions=n_actions,
        )
        max_steps = _max_episode_steps(env)
        candidates = _collect_discounted_candidates(
            env=env,
            behavior_policy=behavior_policy,
            gamma=float(gamma),
            sample_size=int(sample_size),
            rng=rng,
            max_steps=max_steps,
        )
        pick = _sample_candidate_indices(candidates["discount_weights"], sample_size=int(sample_size), rng=rng)
        states = candidates["states"][pick]
        actions_i = candidates["actions"][pick]
        next_states = candidates["next_states"][pick]
        rewards = candidates["rewards"][pick]
        masks = candidates["masks"][pick]
        target_actions_i = target_policy.sample(states, rng)
        next_target_actions_i = target_policy.sample(next_states, rng)
        initial_states = _sample_initial_states(env_factory, max(256, min(2_000, int(sample_size))), rng)
        initial_actions_i = target_policy.sample(initial_states, rng)
    finally:
        _close_env(env)

    target_value, target_value_se = _estimate_target_value(
        env_factory=env_factory,
        target_policy=target_policy,
        gamma=float(gamma),
        seed=int(seed + 121_001),
        rollouts=int(target_value_rollouts),
        max_steps=max_steps,
    )

    behavior_probs = behavior_policy.probabilities(states)
    target_probs = target_policy.probabilities(states)
    action_ratio = target_probs[np.arange(actions_i.shape[0]), actions_i] / np.maximum(
        behavior_probs[np.arange(actions_i.shape[0]), actions_i],
        1e-12,
    )
    return BenchmarkDataset(
        setting=setting,
        states=states,
        actions=one_hot(actions_i, n_actions),
        next_states=next_states,
        target_actions=one_hot(target_actions_i, n_actions),
        next_target_actions=one_hot(next_target_actions_i, n_actions),
        rewards=rewards,
        true_ratio=None,
        true_action_ratio=action_ratio,
        true_transition_ratio=None,
        initial_states=initial_states,
        initial_actions=one_hot(initial_actions_i, n_actions),
        initial_weights=np.ones(initial_states.shape[0], dtype=np.float64),
        masks=masks,
        gamma=float(gamma),
        seed=int(seed),
        sample_size=int(sample_size),
        metadata={
            "truth_source": "target_policy_mc_rollout",
            "reference_distribution": "behavior_discounted_application_simulator",
            "env_id": str(env_id),
            "target_policy_value": float(target_value),
            "target_policy_value_se": float(target_value_se),
            "target_value_rollouts": int(target_value_rollouts),
            "max_episode_steps": int(max_steps),
            "state_dim": int(state_dim),
            "n_actions": int(n_actions),
            "has_ratio_truth": 0.0,
        },
    )


def _make_application_policies(
    *,
    setting: str,
    state_dim: int,
    n_actions: int,
) -> tuple[ApplicationSoftmaxPolicy, ApplicationSoftmaxPolicy]:
    rng = np.random.default_rng(_stable_seed(setting))
    target_weights = rng.normal(scale=0.35, size=(int(n_actions), int(state_dim)))
    target_bias = rng.normal(scale=0.15, size=int(n_actions))
    behavior_weights = 0.65 * target_weights + rng.normal(scale=0.25, size=target_weights.shape)
    behavior_bias = target_bias + rng.normal(scale=0.25, size=int(n_actions))
    return (
        ApplicationSoftmaxPolicy(
            weights=behavior_weights,
            bias=behavior_bias,
            temperature=1.35,
            epsilon=0.20,
        ),
        ApplicationSoftmaxPolicy(
            weights=target_weights,
            bias=target_bias,
            temperature=0.85,
            epsilon=0.05,
        ),
    )


def _collect_discounted_candidates(
    *,
    env: Any,
    behavior_policy: ApplicationSoftmaxPolicy,
    gamma: float,
    sample_size: int,
    rng: np.random.Generator,
    max_steps: int,
) -> dict[str, Array]:
    target_count = max(int(sample_size) * 5, int(sample_size) + 256)
    states: list[Array] = []
    actions: list[int] = []
    next_states: list[Array] = []
    rewards: list[float] = []
    masks: list[float] = []
    discount_weights: list[float] = []
    attempts = 0
    while len(states) < target_count and attempts < max(20, target_count * 3):
        attempts += 1
        obs = _reset_env(env, rng)
        for t in range(int(max_steps)):
            action = int(behavior_policy.sample(obs.reshape(1, -1), rng)[0])
            next_obs, reward, done, truncated = _step_env(env, action)
            states.append(obs)
            actions.append(action)
            next_states.append(next_obs)
            rewards.append(float(reward))
            terminal = bool(done or truncated)
            masks.append(0.0 if terminal else 1.0)
            discount_weights.append(float(gamma) ** int(t))
            obs = next_obs
            if terminal:
                break
    if not states:
        raise OptionalDatasetUnavailable("Application simulator produced no transitions.")
    return {
        "states": np.asarray(states, dtype=np.float64),
        "actions": np.asarray(actions, dtype=np.int64),
        "next_states": np.asarray(next_states, dtype=np.float64),
        "rewards": np.asarray(rewards, dtype=np.float64),
        "masks": np.asarray(masks, dtype=np.float64),
        "discount_weights": np.asarray(discount_weights, dtype=np.float64),
    }


def _estimate_target_value(
    *,
    env_factory: Callable[[], Any],
    target_policy: ApplicationSoftmaxPolicy,
    gamma: float,
    seed: int,
    rollouts: int,
    max_steps: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(int(seed))
    values: list[float] = []
    for _ in range(int(rollouts)):
        env = env_factory()
        try:
            obs = _reset_env(env, rng)
            value = 0.0
            discount = 1.0
            for _t in range(int(max_steps)):
                action = int(target_policy.sample(obs.reshape(1, -1), rng)[0])
                obs, reward, done, truncated = _step_env(env, action)
                value += discount * float(reward)
                discount *= float(gamma)
                if done or truncated:
                    break
            values.append((1.0 - float(gamma)) * value)
        finally:
            _close_env(env)
    arr = np.asarray(values, dtype=np.float64)
    se = float(np.std(arr, ddof=1) / np.sqrt(arr.shape[0])) if arr.shape[0] > 1 else 0.0
    return float(np.mean(arr)), se


def _sample_initial_states(
    env_factory: Callable[[], Any],
    count: int,
    rng: np.random.Generator,
) -> Array:
    states = []
    for _ in range(int(count)):
        env = env_factory()
        try:
            states.append(_reset_env(env, rng))
        finally:
            _close_env(env)
    return np.asarray(states, dtype=np.float64)


def _reset_env(env: Any, rng: np.random.Generator) -> Array:
    seed = int(rng.integers(0, 2**31 - 1))
    try:
        out = env.reset(seed=seed)
    except TypeError:
        if hasattr(env, "seed"):
            env.seed(seed)
        out = env.reset()
    obs = out[0] if isinstance(out, tuple) else out
    return _flatten_observation(obs)


def _step_env(env: Any, action: int) -> tuple[Array, float, bool, bool]:
    out = env.step(int(action))
    if len(out) == 5:
        obs, reward, terminated, truncated, _info = out
    else:
        obs, reward, done, _info = out
        terminated, truncated = bool(done), False
    return _flatten_observation(obs), float(reward), bool(terminated), bool(truncated)


def _flatten_observation(obs: Any) -> Array:
    if isinstance(obs, dict):
        parts = [_flatten_observation(obs[key]) for key in sorted(obs)]
        return np.concatenate(parts).astype(np.float64)
    arr = np.asarray(obs, dtype=np.float64)
    return arr.reshape(-1)


def _discrete_action_count(env: Any) -> int:
    n_actions = getattr(getattr(env, "action_space", None), "n", None)
    if n_actions is None:
        raise OptionalDatasetUnavailable("Application simulator setting requires a discrete action space.")
    return int(n_actions)


def _max_episode_steps(env: Any) -> int:
    for attr in ("step_per_episode", "max_episode_steps"):
        value = getattr(env, attr, None)
        if value is not None:
            return int(value)
    spec = getattr(env, "spec", None)
    if getattr(spec, "max_episode_steps", None) is not None:
        return int(spec.max_episode_steps)
    return 100


def _sample_candidate_indices(weights: Array, *, sample_size: int, rng: np.random.Generator) -> Array:
    probs = np.asarray(weights, dtype=np.float64).reshape(-1)
    probs = probs / np.maximum(np.sum(probs), 1e-12)
    return rng.choice(probs.shape[0], size=int(sample_size), replace=probs.shape[0] < int(sample_size), p=probs)


def _sample_categorical(probs: Array, rng: np.random.Generator) -> Array:
    probs_arr = np.asarray(probs, dtype=np.float64)
    cdf = np.cumsum(probs_arr, axis=1)
    draws = rng.random(probs_arr.shape[0])
    return (cdf < draws[:, None]).sum(axis=1).astype(np.int64)


def _as_2d_float(values: Array) -> Array:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    return arr.reshape(arr.shape[0], -1)


def _stable_seed(text: str) -> int:
    return int(sum((idx + 1) * ord(char) for idx, char in enumerate(text)) % (2**32 - 1))


def _close_env(env: Any) -> None:
    close = getattr(env, "close", None)
    if callable(close):
        close()
