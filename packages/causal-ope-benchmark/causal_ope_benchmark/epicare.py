from __future__ import annotations

from typing import Any

import numpy as np

from causal_ope_benchmark.config import DomainScenario
from causal_ope_benchmark.contracts import contract_for_family
from causal_ope_benchmark.gym_envs import make_epicare_gym_env
from causal_ope_benchmark.policies import get_fixed_policy
from causal_ope_benchmark.simulators import _calibrated_behavior_probs, _dataset_from_rows, _discounted_sum, _public_scenario_label, _se, _truth_noise_floor
from causal_ope_benchmark.types import Array, BenchmarkProblem, TruthBundle, normalize_action_probs


def make_epicare_problem(
    *,
    sample_size: int,
    gamma: float,
    seed: int,
    scenario: DomainScenario,
    target_policy: str = "moderate",
    horizon: int = 24,
    mc_truth_rollouts: int = 96,
    env_id: str = "EpiCare-v0",
) -> BenchmarkProblem:
    """Collect an optional EpiCare logged dataset and sealed MC truth.

    The EpiCare simulator is loaded lazily and used as-is. This package does
    not vendor or reimplement EpiCare dynamics.
    """

    rng = np.random.default_rng(seed)
    env = make_epicare_gym_env(seed=seed, env_id=env_id)
    n_actions = _discrete_action_count(env)
    action_names = tuple(f"action_{i}" for i in range(n_actions))
    policy = get_fixed_policy("epicare", target_policy)
    rows: list[dict[str, object]] = []
    initial_states = []
    initial_target_action_idx = []
    for unit in range(int(sample_size)):
        obs, _ = _reset_env(env, seed + unit)
        state = _flatten_observation(obs)
        initial_states.append(state)
        availability = np.ones(n_actions, dtype=np.float64)
        initial_probs = policy.probabilities(
            state.reshape(1, -1),
            availability.reshape(1, -1),
            np.asarray([0], dtype=np.int64),
        )[0]
        initial_target_action_idx.append(int(rng.choice(n_actions, p=initial_probs)))
        for t in range(int(horizon)):
            availability = np.ones(n_actions, dtype=np.float64)
            target_probs = policy.probabilities(
                state.reshape(1, -1),
                availability.reshape(1, -1),
                np.asarray([t], dtype=np.int64),
            )[0]
            nominal = np.full(n_actions, 1.0 / n_actions, dtype=np.float64)
            behavior_probs = _calibrated_behavior_probs(
                target_probs,
                nominal,
                availability,
                desired_tv=float(scenario.target_policy_distance),
            )
            action = int(rng.choice(n_actions, p=behavior_probs))
            target_action = int(rng.choice(n_actions, p=target_probs))
            step = _step_env(env, action)
            next_state = _flatten_observation(step.observation)
            next_target_probs = policy.probabilities(
                next_state.reshape(1, -1),
                availability.reshape(1, -1),
                np.asarray([t + 1], dtype=np.int64),
            )[0]
            next_target_action = int(rng.choice(n_actions, p=next_target_probs))
            rows.append(
                {
                    "unit": unit,
                    "time": t,
                    "state": state,
                    "action": action,
                    "reward": float(step.reward),
                    "next_state": next_state,
                    "terminal": float(step.terminated),
                    "availability": availability,
                    "behavior_p": float(behavior_probs[action]),
                    "behavior_probs": behavior_probs,
                    "target_probs": target_probs,
                    "target_action": target_action,
                    "next_target_action": next_target_action,
                    "next_target_probs": next_target_probs,
                    "censoring": float(step.truncated and not step.terminated),
                    "components": {"reward": float(step.reward)},
                }
            )
            state = next_state
            if step.terminated or step.truncated:
                break
    scenario_label = _public_scenario_label(scenario.name)
    state_dim = len(initial_states[0]) if initial_states else 0
    dataset = _dataset_from_rows(
        rows=rows,
        name=f"epicare_{scenario_label}_{target_policy}_seed{seed}",
        family="epicare",
        scenario=scenario,
        gamma=gamma,
        seed=seed,
        n_actions=n_actions,
        initial_states=np.vstack(initial_states).astype(np.float64),
        initial_action_indices=np.asarray(initial_target_action_idx, dtype=np.int64),
        metadata_public={
            "family": "epicare",
            "scenario": scenario_label,
            "sample_size": int(sample_size),
            "gamma": float(gamma),
            "trajectory_horizon": int(horizon),
            "target_policy": str(target_policy),
            "leaderboard_eligible": bool(scenario.leaderboard_eligible and scenario.confounding != "latent"),
            "state_features": "|".join(f"state_{i}" for i in range(state_dim)),
            "action_names": "|".join(action_names),
            "outcome_names": "reward",
            "external_benchmark": "EpiCare",
            "gym_env_id": env_id,
            "gym_api": _gym_api_name(env),
            "action_constraints": "external_env",
            "target_policy_distance": _mean_tv_from_rows(rows),
        },
    )
    dataset.information_contract = _contract_with_epicare_metadata(dataset.metadata_public)
    truth = _epicare_truth(
        env_id=env_id,
        dataset_name=dataset.name,
        gamma=gamma,
        seed=seed + 71_000,
        target_policy=target_policy,
        horizon=int(horizon),
        n_actions=n_actions,
        mc_rollouts=int(mc_truth_rollouts),
        scenario=scenario,
    )
    truth.oracle_ratios["row_target_over_behavior"] = np.asarray(dataset.target_propensity_observed_action) / np.asarray(dataset.behavior_propensity)
    return BenchmarkProblem(dataset=dataset, truth=truth)


class _StepResult:
    def __init__(self, observation: Any, reward: float, terminated: bool, truncated: bool, info: dict[str, Any]) -> None:
        self.observation = observation
        self.reward = float(reward)
        self.terminated = bool(terminated)
        self.truncated = bool(truncated)
        self.info = info


def _epicare_truth(
    *,
    env_id: str,
    dataset_name: str,
    gamma: float,
    seed: int,
    target_policy: str,
    horizon: int,
    n_actions: int,
    mc_rollouts: int,
    scenario: DomainScenario,
) -> TruthBundle:
    rng = np.random.default_rng(seed)
    policy = get_fixed_policy("epicare", target_policy)
    values = []
    terminal_survival = []
    for rollout in range(int(mc_rollouts)):
        env = make_epicare_gym_env(seed=seed + rollout, env_id=env_id)
        obs, _ = _reset_env(env, seed + rollout)
        state = _flatten_observation(obs)
        rewards = []
        alive = True
        for t in range(int(horizon)):
            availability = np.ones(n_actions, dtype=np.float64)
            probs = policy.probabilities(
                state.reshape(1, -1),
                availability.reshape(1, -1),
                np.asarray([t], dtype=np.int64),
            )[0]
            action = int(rng.choice(n_actions, p=probs))
            step = _step_env(env, action)
            state = _flatten_observation(step.observation)
            rewards.append(float(step.reward))
            if step.terminated:
                alive = False
                break
            if step.truncated:
                break
        values.append(_discounted_sum(np.asarray(rewards, dtype=np.float64), gamma))
        terminal_survival.append(float(alive))
    policy_value = float(np.mean(values)) if values else 0.0
    survival_horizon = float(np.mean(terminal_survival)) if terminal_survival else 0.0
    truth_values = {"policy_value": policy_value, "survival_horizon": survival_horizon}
    return TruthBundle(
        dataset_name=dataset_name,
        family="epicare",
        values=truth_values,
        target_mc_values=truth_values.copy(),
        target_standard_errors={
            "policy_value": _se(values),
            "survival_horizon": _se(terminal_survival),
        },
        truth_noise_floor=_truth_noise_floor(truth_values),
        mc_rollouts=int(mc_rollouts),
        private_metadata={
            "external_benchmark": "EpiCare",
            "gym_env_id": env_id,
            "scenario_private_name": scenario.name,
            "target_policy": target_policy,
        },
        leaderboard_eligible=bool(scenario.leaderboard_eligible and scenario.confounding != "latent"),
    )


def _reset_env(env: Any, seed: int) -> tuple[Any, dict[str, Any]]:
    try:
        out = env.reset(seed=int(seed))
    except TypeError:
        if hasattr(env, "seed"):
            env.seed(int(seed))
        out = env.reset()
    if isinstance(out, tuple) and len(out) == 2:
        return out[0], dict(out[1] or {})
    return out, {}


def _step_env(env: Any, action: int) -> _StepResult:
    out = env.step(int(action))
    if isinstance(out, tuple) and len(out) == 5:
        obs, reward, terminated, truncated, info = out
        return _StepResult(obs, float(reward), bool(terminated), bool(truncated), dict(info or {}))
    if isinstance(out, tuple) and len(out) == 4:
        obs, reward, done, info = out
        return _StepResult(obs, float(reward), bool(done), False, dict(info or {}))
    raise RuntimeError("EpiCare env.step returned an unsupported shape.")


def _flatten_observation(obs: Any) -> Array:
    if isinstance(obs, dict):
        pieces = [_flatten_observation(obs[key]) for key in sorted(obs)]
        return np.concatenate(pieces).astype(np.float64)
    if isinstance(obs, (tuple, list)) and not _looks_numeric_sequence(obs):
        pieces = [_flatten_observation(value) for value in obs]
        return np.concatenate(pieces).astype(np.float64)
    arr = np.asarray(obs, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        raise RuntimeError("EpiCare observation could not be flattened.")
    return np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float64)


def _looks_numeric_sequence(value: Any) -> bool:
    try:
        np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return True


def _discrete_action_count(env: Any) -> int:
    action_space = getattr(env, "action_space", None)
    n = getattr(action_space, "n", None)
    if n is None:
        raise RuntimeError("EpiCare integration currently supports only discrete action spaces.")
    return int(n)


def _mean_tv_from_rows(rows: list[dict[str, object]]) -> float:
    if not rows:
        return 0.0
    behavior = np.vstack([row["behavior_probs"] for row in rows]).astype(np.float64)
    target = np.vstack([row["target_probs"] for row in rows]).astype(np.float64)
    return float(np.mean(0.5 * np.sum(np.abs(normalize_action_probs(behavior, np.ones_like(behavior)) - target), axis=1)))


def _contract_with_epicare_metadata(metadata_public: dict[str, object]):
    base = contract_for_family("epicare")
    return type(base)(
        family=base.family,
        visible_arrays=base.visible_arrays,
        visible_metadata=tuple(dict.fromkeys((*base.visible_metadata, *metadata_public.keys()))),
        allows_behavior_propensity=base.allows_behavior_propensity,
        allows_target_propensity=base.allows_target_propensity,
        allows_censoring=base.allows_censoring,
        notes=base.notes,
    )


def _gym_api_name(env: Any) -> str:
    module = type(env).__module__.split(".", maxsplit=1)[0]
    if module in {"gym", "gymnasium"}:
        return module
    return "gym"
