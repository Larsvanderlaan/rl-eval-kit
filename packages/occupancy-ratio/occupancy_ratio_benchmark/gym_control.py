from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from occupancy_ratio_benchmark.data import BenchmarkDataset


Array = np.ndarray


GYM_CONTROL_SETTINGS = {
    "gym_pendulum": "Pendulum-v1",
    "gym_mountain_car_continuous": "MountainCarContinuous-v0",
    "gym_halfcheetah": "HalfCheetah-v4",
    "gym_hopper": "Hopper-v4",
}


@dataclass(frozen=True)
class GymGaussianPolicy:
    """Simple reproducible continuous policy used for modern-control benchmarks."""

    setting: str
    action_low: Array
    action_high: Array
    gain: Array
    bias: Array
    noise_scale: float

    def mean_action(self, states: Array) -> Array:
        states_arr = np.asarray(states, dtype=np.float64).reshape(-1, self.gain.shape[1])
        action_center = 0.5 * (self.action_high + self.action_low)
        action_scale = 0.5 * (self.action_high - self.action_low)
        if self.setting == "gym_pendulum" and states_arr.shape[1] >= 3:
            raw = -1.8 * states_arr[:, [1]] - 0.25 * states_arr[:, [2]] + self.bias.reshape(1, -1)
        elif self.setting == "gym_mountain_car_continuous" and states_arr.shape[1] >= 2:
            raw = 3.5 * states_arr[:, [1]] + 1.5 * (states_arr[:, [0]] + 0.5) + self.bias.reshape(1, -1)
        else:
            raw = np.tanh(states_arr / 5.0) @ self.gain.T + self.bias.reshape(1, -1)
        return action_center.reshape(1, -1) + action_scale.reshape(1, -1) * np.tanh(raw)

    def sample(self, states: Array, rng: np.random.Generator) -> Array:
        mean = self.mean_action(states)
        action_scale = 0.5 * (self.action_high - self.action_low)
        noise = float(self.noise_scale) * action_scale.reshape(1, -1) * rng.normal(size=mean.shape)
        return np.clip(mean + noise, self.action_low.reshape(1, -1), self.action_high.reshape(1, -1))


def make_gym_control_dataset(
    *,
    setting: str,
    gamma: float,
    sample_size: int,
    seed: int,
    target_value_rollouts: int,
) -> BenchmarkDataset:
    """Create an offline continuous-control OPE benchmark from Gymnasium.

    Rows are sampled from behavior trajectories with probability proportional
    to gamma^t, so uniform averages over rows approximate a normalized
    discounted behavior occupancy. Ratio truth is intentionally unavailable;
    target-policy value is estimated by Monte Carlo rollouts.
    """

    if setting not in GYM_CONTROL_SETTINGS:
        raise ValueError(f"Unknown Gym control setting '{setting}'.")
    try:
        import gymnasium as gym
    except ImportError as exc:  # pragma: no cover - exercised only without extras
        raise RuntimeError("Install the benchmark extra with Gymnasium support to use Gym control settings.") from exc

    env_id = GYM_CONTROL_SETTINGS[setting]
    rng = np.random.default_rng(int(seed))
    env = gym.make(env_id)
    try:
        state_dim = int(np.asarray(env.observation_space.shape).prod())
        action_dim = int(np.asarray(env.action_space.shape).prod())
        action_low = np.asarray(env.action_space.low, dtype=np.float64).reshape(action_dim)
        action_high = np.asarray(env.action_space.high, dtype=np.float64).reshape(action_dim)
        action_low, action_high = _finite_action_bounds(action_low, action_high)
        behavior_policy, target_policy = _make_policies(
            setting=setting,
            state_dim=state_dim,
            action_low=action_low,
            action_high=action_high,
        )
        max_steps = int(getattr(env.spec, "max_episode_steps", None) or 1_000)
        candidates = _collect_discounted_behavior_candidates(
            env=env,
            behavior_policy=behavior_policy,
            gamma=float(gamma),
            sample_size=int(sample_size),
            seed=int(seed),
            rng=rng,
            max_steps=max_steps,
        )
        pick = _sample_candidate_indices(candidates["discount_weights"], sample_size=int(sample_size), rng=rng)
        states = candidates["states"][pick]
        actions = candidates["actions"][pick]
        next_states = candidates["next_states"][pick]
        rewards = candidates["rewards"][pick]
        masks = candidates["masks"][pick]
        target_actions = target_policy.sample(states, rng)
        next_target_actions = target_policy.sample(next_states, rng)
        initial_states = _sample_initial_states(env, max(256, min(2_000, int(sample_size))), rng)
        initial_actions = target_policy.sample(initial_states, rng)
    finally:
        env.close()

    target_value, target_value_se = _estimate_policy_value(
        env_id=env_id,
        policy=target_policy,
        gamma=float(gamma),
        seed=int(seed + 91_001),
        rollouts=int(target_value_rollouts),
        max_steps=max_steps,
    )

    return BenchmarkDataset(
        setting=setting,
        states=states,
        actions=actions,
        next_states=next_states,
        target_actions=target_actions,
        next_target_actions=next_target_actions,
        rewards=rewards,
        true_ratio=None,
        true_action_ratio=None,
        true_transition_ratio=None,
        initial_states=initial_states,
        initial_actions=initial_actions,
        initial_weights=np.ones(initial_states.shape[0], dtype=np.float64),
        masks=masks,
        gamma=float(gamma),
        seed=int(seed),
        sample_size=int(sample_size),
        metadata={
            "truth_source": "target_policy_mc_rollout",
            "reference_distribution": "behavior_discounted_occupancy_sampled",
            "env_id": env_id,
            "target_policy_value": float(target_value),
            "target_policy_value_se": float(target_value_se),
            "target_value_rollouts": int(target_value_rollouts),
            "max_episode_steps": int(max_steps),
            "state_dim": int(state_dim),
            "action_dim": int(action_dim),
            "has_ratio_truth": 0.0,
        },
    )


def _stable_seed(text: str) -> int:
    return int(sum((idx + 1) * ord(char) for idx, char in enumerate(text)) % (2**32 - 1))


def _finite_action_bounds(low: Array, high: Array) -> tuple[Array, Array]:
    lo = np.asarray(low, dtype=np.float64).copy()
    hi = np.asarray(high, dtype=np.float64).copy()
    lo[~np.isfinite(lo)] = -1.0
    hi[~np.isfinite(hi)] = 1.0
    same = hi <= lo
    lo[same] = -1.0
    hi[same] = 1.0
    return lo, hi


def _make_policies(
    *,
    setting: str,
    state_dim: int,
    action_low: Array,
    action_high: Array,
) -> tuple[GymGaussianPolicy, GymGaussianPolicy]:
    rng = np.random.default_rng(_stable_seed(setting))
    action_dim = int(action_low.shape[0])
    target_gain = rng.normal(scale=0.20, size=(action_dim, int(state_dim)))
    target_bias = rng.normal(scale=0.10, size=action_dim)
    behavior_gain = 0.70 * target_gain + rng.normal(scale=0.15, size=target_gain.shape)
    behavior_bias = target_bias + rng.normal(scale=0.20, size=action_dim)
    if setting in {"gym_pendulum", "gym_mountain_car_continuous"}:
        target_bias = np.zeros(action_dim, dtype=np.float64)
        behavior_bias = rng.normal(scale=0.25, size=action_dim)
    return (
        GymGaussianPolicy(
            setting=setting,
            action_low=action_low,
            action_high=action_high,
            gain=behavior_gain,
            bias=behavior_bias,
            noise_scale=0.35,
        ),
        GymGaussianPolicy(
            setting=setting,
            action_low=action_low,
            action_high=action_high,
            gain=target_gain,
            bias=target_bias,
            noise_scale=0.12,
        ),
    )


def _reset_env(env: Any, rng: np.random.Generator) -> Array:
    obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
    return np.asarray(obs, dtype=np.float64).reshape(-1)


def _collect_discounted_behavior_candidates(
    *,
    env: Any,
    behavior_policy: GymGaussianPolicy,
    gamma: float,
    sample_size: int,
    seed: int,
    rng: np.random.Generator,
    max_steps: int,
) -> dict[str, Array]:
    del seed
    target_candidates = max(int(sample_size) * 3, int(sample_size) + 512)
    states: list[Array] = []
    actions: list[Array] = []
    next_states: list[Array] = []
    rewards: list[float] = []
    masks: list[float] = []
    discount_weights: list[float] = []
    while len(states) < target_candidates:
        obs = _reset_env(env, rng)
        discount = 1.0
        for _ in range(int(max_steps)):
            action = behavior_policy.sample(obs.reshape(1, -1), rng).reshape(-1)
            step = env.step(action.astype(env.action_space.dtype, copy=False))
            next_obs, reward, terminated, truncated, _ = step
            done = bool(terminated or truncated)
            states.append(obs.copy())
            actions.append(action.astype(np.float64, copy=True))
            next_states.append(np.asarray(next_obs, dtype=np.float64).reshape(-1))
            rewards.append(float(reward))
            masks.append(0.0 if done else 1.0)
            discount_weights.append(float(discount))
            if done or len(states) >= target_candidates:
                break
            obs = np.asarray(next_obs, dtype=np.float64).reshape(-1)
            discount *= float(gamma)
    return {
        "states": np.asarray(states, dtype=np.float64),
        "actions": np.asarray(actions, dtype=np.float64),
        "next_states": np.asarray(next_states, dtype=np.float64),
        "rewards": np.asarray(rewards, dtype=np.float64),
        "masks": np.asarray(masks, dtype=np.float64),
        "discount_weights": np.asarray(discount_weights, dtype=np.float64),
    }


def _sample_candidate_indices(weights: Array, *, sample_size: int, rng: np.random.Generator) -> Array:
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    w = np.maximum(w, 0.0)
    if not np.isfinite(w).all() or float(np.sum(w)) <= 0.0:
        probs = None
    else:
        probs = w / float(np.sum(w))
    return rng.choice(w.shape[0], size=int(sample_size), replace=w.shape[0] < int(sample_size), p=probs)


def _sample_initial_states(env: Any, n: int, rng: np.random.Generator) -> Array:
    states = [_reset_env(env, rng) for _ in range(int(n))]
    return np.asarray(states, dtype=np.float64)


def _estimate_policy_value(
    *,
    env_id: str,
    policy: GymGaussianPolicy,
    gamma: float,
    seed: int,
    rollouts: int,
    max_steps: int,
) -> tuple[float, float]:
    import gymnasium as gym

    rng = np.random.default_rng(int(seed))
    env = gym.make(env_id)
    values = []
    try:
        for _ in range(max(int(rollouts), 1)):
            obs = _reset_env(env, rng)
            discount = 1.0
            total = 0.0
            for _ in range(int(max_steps)):
                action = policy.sample(obs.reshape(1, -1), rng).reshape(-1)
                next_obs, reward, terminated, truncated, _ = env.step(action.astype(env.action_space.dtype, copy=False))
                total += discount * float(reward)
                discount *= float(gamma)
                obs = np.asarray(next_obs, dtype=np.float64).reshape(-1)
                if bool(terminated or truncated):
                    break
            values.append((1.0 - float(gamma)) * total)
    finally:
        env.close()
    arr = np.asarray(values, dtype=np.float64)
    se = float(np.std(arr, ddof=1) / np.sqrt(arr.size)) if arr.size > 1 else 0.0
    return float(np.mean(arr)), se
