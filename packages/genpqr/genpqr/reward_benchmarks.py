"""Reward-recovery benchmarks for GenPQR and DeepGenPQR."""

from __future__ import annotations

from dataclasses import dataclass, field
import importlib.util
import os
import tempfile
from typing import Any

import numpy as np

from genpqr.datasets import TransitionDataset
from genpqr.deepgenpqr import DeepGenPQRConfig, DeepGenPQRResult, fit_deep_genpqr
from genpqr.normalization import DiscreteNormalizationPolicy
from genpqr.types import ActionSpaceSpec, Array
from genpqr.validation import as_2d_float


@dataclass(frozen=True)
class SoftGridOraclePolicy:
    """Tabular soft policy defined on the SoftGrid state lattice.

    Parameters
    ----------
    probabilities:
        State-by-action probability matrix.
    grid_states:
        Continuous coordinates for each grid state.
    action_space:
        Discrete action-space contract.
    name:
        Diagnostic name for the policy.
    """

    probabilities: Array
    grid_states: Array
    action_space: ActionSpaceSpec
    name: str = "soft_grid_oracle"
    lookup_decimals: int = 12
    _lookup: dict[tuple[float, ...], int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        grid = as_2d_float(self.grid_states, "grid_states")
        probs = np.asarray(self.probabilities, dtype=np.float64)
        if self.action_space.kind != "discrete":
            raise ValueError("SoftGridOraclePolicy requires a discrete action space.")
        expected = (grid.shape[0], int(self.action_space.n_actions))
        if probs.shape != expected:
            raise ValueError(f"probabilities must have shape {expected}.")
        if not np.all(np.isfinite(probs)) or np.any(probs <= 0.0):
            raise ValueError("probabilities must be finite and strictly positive.")
        row_sums = probs.sum(axis=1, keepdims=True)
        if np.any(row_sums <= 0.0):
            raise ValueError("policy probability rows must have positive mass.")
        object.__setattr__(self, "grid_states", grid)
        object.__setattr__(self, "probabilities", probs / row_sums)
        lookup = {
            tuple(np.round(row, int(self.lookup_decimals))): i
            for i, row in enumerate(grid)
        }
        object.__setattr__(self, "_lookup", lookup)

    def predict_proba(self, states: Array) -> Array:
        """Return action probabilities for each state row."""

        idx = self.state_indices(states)
        return np.array(self.probabilities[idx], copy=True)

    def log_prob(self, states: Array, actions: Array) -> Array:
        """Return log probabilities for state-action rows."""

        states_2d = as_2d_float(states, "states")
        idx = self.state_indices(states_2d)
        action_idx = self.action_space.action_indices(actions, n_rows=states_2d.shape[0])
        probs = self.probabilities[idx, action_idx]
        return np.log(np.clip(probs, 1e-300, None))

    def sample(self, states: Array, rng: np.random.Generator, n_samples: int = 1) -> Array:
        """Sample finite actions from the policy."""

        if int(n_samples) <= 0:
            raise ValueError("n_samples must be positive.")
        probs = self.predict_proba(states)
        choices = np.arange(int(self.action_space.n_actions), dtype=np.int64)
        draws = np.empty((probs.shape[0], int(n_samples)), dtype=np.int64)
        for i, row in enumerate(probs):
            draws[i] = rng.choice(choices, size=int(n_samples), p=row)
        return draws.reshape(-1) if int(n_samples) == 1 else draws

    def state_indices(self, states: Array) -> Array:
        """Map continuous grid coordinates to nearest lattice indices."""

        states_2d = as_2d_float(states, "states")
        if states_2d.shape[1] != self.grid_states.shape[1]:
            raise ValueError("states have the wrong feature dimension for this policy.")
        out = np.empty(states_2d.shape[0], dtype=np.int64)
        unknown: list[int] = []
        for i, row in enumerate(states_2d):
            key = tuple(np.round(row, int(self.lookup_decimals)))
            mapped = self._lookup.get(key)
            if mapped is None:
                unknown.append(i)
            else:
                out[i] = int(mapped)
        if unknown:
            grid = self.grid_states
            for i in unknown:
                distances = np.sum((grid - states_2d[i]) ** 2, axis=1)
                out[i] = int(np.argmin(distances))
        return out


@dataclass(frozen=True)
class SoftGridRewardRecoveryProblem:
    """Complete SoftGrid normalized-reward benchmark instance."""

    dataset: TransitionDataset
    action_space: ActionSpaceSpec
    grid_states: Array
    transition: Array
    raw_rewards: Array
    true_rewards: Array
    oracle_q: Array
    oracle_values: Array
    oracle_policy: SoftGridOraclePolicy
    behavior_policy: SoftGridOraclePolicy
    state_distribution: Array
    gamma: float
    soft_temperature: float
    anchor_action: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_states(self) -> int:
        """Number of grid states."""

        return int(self.grid_states.shape[0])

    @property
    def n_actions(self) -> int:
        """Number of finite actions."""

        return int(self.action_space.n_actions)


def make_soft_grid_reward_recovery(
    *,
    n_x: int = 15,
    n_y: int = 15,
    sample_size: int = 3_000,
    seed: int = 123,
    gamma: float = 0.95,
    soft_temperature: float = 1.0,
    anchor_action: int = 0,
    behavior_mixture: float = 0.45,
    behavior_floor: float = 0.02,
    anchor_behavior_scale: float = 1.0,
    min_anchor_fraction: float = 0.08,
    low: float = -1.0,
    high: float = 1.0,
    action_scale: float = 0.23,
    drift_scale: float = 0.13,
    interaction_scale: float = 0.055,
    process_noise: float = 0.075,
    teleport_prob: float = 0.002,
    goal: tuple[float, float] = (0.62, 0.62),
    decoy: tuple[float, float] = (-0.55, 0.45),
    goal_width: float = 0.20,
    decoy_width: float = 0.23,
    barrier_width: float = 0.22,
    action_cost: float = 0.035,
) -> SoftGridRewardRecoveryProblem:
    """Create a SoftGrid normalized-reward recovery benchmark.

    The benchmark uses a five-action nonlinear grid MDP with action ``0`` as
    the no-op normalization anchor. The true reward is the raw reward after
    state-wise action-anchor subtraction, so ``r_true(s, 0) = 0`` exactly. The
    oracle policy is the entropy-regularized soft-optimal policy for this
    normalized reward.

    Parameters
    ----------
    n_x, n_y:
        Grid dimensions.
    sample_size:
        Number of row-wise logged transitions to sample from the behavior
        policy.
    seed:
        Random seed controlling dataset generation.
    gamma:
        Discount factor for soft value iteration and downstream GenPQR fits.
    soft_temperature:
        Temperature used by soft value iteration. The default ``1.0`` is
        aligned with the current DeepPQR ``alpha=1`` convention.
    anchor_action:
        Finite action whose normalized reward is fixed to zero.
    anchor_behavior_scale:
        Multiplicative behavior-policy odds scale for the anchor action before
        renormalization. Values below one create rare-anchor stress screens.
    min_anchor_fraction:
        Minimum empirical anchor-action fraction enforced in the sampled logged
        dataset. Set to ``0`` for natural rare-anchor screens.

    Returns
    -------
    SoftGridRewardRecoveryProblem
        Dataset, oracle policy, transition kernel, and exact reward truth.
    """

    if int(n_x) < 2 or int(n_y) < 2:
        raise ValueError("n_x and n_y must both be at least 2.")
    if int(sample_size) <= 0:
        raise ValueError("sample_size must be positive.")
    if not (0.0 <= float(gamma) < 1.0):
        raise ValueError("gamma must be in [0, 1).")
    if float(soft_temperature) <= 0.0:
        raise ValueError("soft_temperature must be positive.")
    n_actions = 5
    if int(anchor_action) < 0 or int(anchor_action) >= n_actions:
        raise ValueError("anchor_action is out of bounds.")
    if not (0.0 <= float(behavior_mixture) <= 1.0):
        raise ValueError("behavior_mixture must lie in [0, 1].")
    if float(behavior_floor) < 0.0 or float(behavior_floor) * n_actions >= 1.0:
        raise ValueError("behavior_floor must be nonnegative and leave positive remaining mass.")
    if float(anchor_behavior_scale) < 0.0:
        raise ValueError("anchor_behavior_scale must be nonnegative.")
    if not (0.0 <= float(min_anchor_fraction) <= 1.0):
        raise ValueError("min_anchor_fraction must lie in [0, 1].")

    rng = np.random.default_rng(int(seed))
    action_space = ActionSpaceSpec.discrete(n_actions)
    grid_states = _make_grid(int(n_x), int(n_y), float(low), float(high))
    transition_means = _transition_means(
        grid_states=grid_states,
        low=float(low),
        high=float(high),
        action_scale=float(action_scale),
        drift_scale=float(drift_scale),
        interaction_scale=float(interaction_scale),
    )
    transition = _transition_kernel(
        grid_states=grid_states,
        transition_means=transition_means,
        process_noise=float(process_noise),
        teleport_prob=float(teleport_prob),
    )
    raw_rewards = _raw_reward_matrix(
        grid_states=grid_states,
        transition_means=transition_means,
        action_scale=float(action_scale),
        goal=np.asarray(goal, dtype=np.float64),
        decoy=np.asarray(decoy, dtype=np.float64),
        goal_width=float(goal_width),
        decoy_width=float(decoy_width),
        barrier_width=float(barrier_width),
        action_cost=float(action_cost),
    )
    true_rewards = raw_rewards - raw_rewards[:, [int(anchor_action)]]
    oracle_q, oracle_values = _soft_value_iteration(
        rewards=true_rewards,
        transition=transition,
        gamma=float(gamma),
        temperature=float(soft_temperature),
    )
    oracle_probs = _floor_probabilities(_softmax(oracle_q / float(soft_temperature), axis=1), floor=1e-12)
    oracle_policy = SoftGridOraclePolicy(
        probabilities=oracle_probs,
        grid_states=grid_states,
        action_space=action_space,
        name="soft_grid_soft_optimal_oracle",
    )
    decoy_probs = _decoy_policy(
        transition_means=transition_means,
        decoy=np.asarray(decoy, dtype=np.float64),
        decoy_width=float(decoy_width),
        temperature=max(float(soft_temperature), 0.25),
    )
    behavior_probs = (1.0 - float(behavior_mixture)) * oracle_probs + float(behavior_mixture) * decoy_probs
    behavior_probs = (1.0 - n_actions * float(behavior_floor)) * behavior_probs + float(behavior_floor)
    behavior_probs[:, int(anchor_action)] *= float(anchor_behavior_scale)
    behavior_probs = behavior_probs / behavior_probs.sum(axis=1, keepdims=True)
    behavior_policy = SoftGridOraclePolicy(
        probabilities=behavior_probs,
        grid_states=grid_states,
        action_space=action_space,
        name="soft_grid_decoy_mixture_behavior",
    )
    state_distribution = _stationary_distribution(transition, behavior_probs)
    dataset = _sample_soft_grid_dataset(
        rng=rng,
        grid_states=grid_states,
        transition=transition,
        behavior_probs=behavior_probs,
        state_distribution=state_distribution,
        action_space=action_space,
        sample_size=int(sample_size),
        anchor_action=int(anchor_action),
        min_anchor_fraction=float(min_anchor_fraction),
    )
    dataset.metadata.update(
        {
            "benchmark": "soft_grid_reward_recovery",
            "n_x": int(n_x),
            "n_y": int(n_y),
            "gamma": float(gamma),
            "soft_temperature": float(soft_temperature),
            "anchor_action": int(anchor_action),
            "oracle_policy": oracle_policy.name,
            "behavior_policy": behavior_policy.name,
        }
    )
    metadata = {
        "benchmark": "soft_grid_reward_recovery",
        "n_x": int(n_x),
        "n_y": int(n_y),
        "n_states": int(grid_states.shape[0]),
        "n_actions": n_actions,
        "sample_size": int(sample_size),
        "seed": int(seed),
        "gamma": float(gamma),
        "soft_temperature": float(soft_temperature),
        "anchor_action": int(anchor_action),
        "behavior_mixture": float(behavior_mixture),
        "behavior_floor": float(behavior_floor),
        "anchor_behavior_scale": float(anchor_behavior_scale),
        "min_anchor_fraction": float(min_anchor_fraction),
        "truth": "r_raw(s,a) - r_raw(s,anchor_action)",
    }
    return SoftGridRewardRecoveryProblem(
        dataset=dataset,
        action_space=action_space,
        grid_states=grid_states,
        transition=transition,
        raw_rewards=raw_rewards,
        true_rewards=true_rewards,
        oracle_q=oracle_q,
        oracle_values=oracle_values,
        oracle_policy=oracle_policy,
        behavior_policy=behavior_policy,
        state_distribution=state_distribution,
        gamma=float(gamma),
        soft_temperature=float(soft_temperature),
        anchor_action=int(anchor_action),
        metadata=metadata,
    )


def run_reward_recovery_benchmark(
    *,
    problem: SoftGridRewardRecoveryProblem | None = None,
    n_x: int = 15,
    n_y: int = 15,
    sample_size: int = 3_000,
    seed: int = 123,
    gamma: float = 0.95,
    soft_temperature: float = 1.0,
    include_pooled_fqe: bool = True,
    include_native_bc: bool = True,
    include_neural_deeppqr: bool | None = None,
    neural_fast: bool = True,
) -> dict[str, Any]:
    """Run the SoftGrid normalized-reward benchmark.

    Parameters
    ----------
    problem:
        Optional prebuilt SoftGrid problem. When omitted, one is generated from
        the supplied grid and sample parameters.
    include_pooled_fqe:
        Whether to try the oracle-policy pooled neural FQE row.
    include_native_bc:
        Whether to try the learned native behavior-cloning plus pooled
        action-head neural FQE row.
    include_neural_deeppqr:
        Whether to try the Torch neural DeepPQR anchor row. ``None`` means
        include it only when Torch is importable.
    neural_fast:
        Use smaller neural configs for smoke screens.

    Returns
    -------
    dict
        JSON-safe benchmark report with baseline and fitted-method rows.
    """

    if problem is None:
        problem = make_soft_grid_reward_recovery(
            n_x=int(n_x),
            n_y=int(n_y),
            sample_size=int(sample_size),
            seed=int(seed),
            gamma=float(gamma),
            soft_temperature=float(soft_temperature),
        )
    if include_neural_deeppqr is None:
        include_neural_deeppqr = importlib.util.find_spec("torch") is not None

    rows: list[dict[str, Any]] = []
    true_rewards = problem.true_rewards
    constant_value = float(np.mean(true_rewards[:, 1:])) if true_rewards.shape[1] > 1 else float(np.mean(true_rewards))
    rows.append(_baseline_row("constant_mean_reward", np.full_like(true_rewards, constant_value), problem))
    rng = np.random.default_rng(int(seed) + 10_001)
    scale = float(np.std(true_rewards))
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = 1.0
    rows.append(_baseline_row("random_reward", rng.normal(scale=scale, size=true_rewards.shape), problem))

    mu = DiscreteNormalizationPolicy.anchor(problem.n_actions, problem.anchor_action)
    rows.append(
        _fit_deepgenpqr_row(
            name="oracle_policy_anchor_deeppqr",
            problem=problem,
            normalization_policy=mu,
            config=DeepGenPQRConfig(
                policy=problem.oracle_policy,
                q_mode="anchor_deeppqr",
                anchor_backend="deeppqr_linear",
                anchor_action=problem.anchor_action,
                min_anchor_count=1,
                seed=int(seed),
                n_action_samples=problem.n_actions,
                q_config={"ridge": 1e-6, "n_iterations": 120},
            ),
        )
    )
    if include_pooled_fqe:
        rows.append(
            _fit_deepgenpqr_row(
                name="oracle_policy_pooled_action_head_fqe",
                problem=problem,
                normalization_policy=mu,
                config=DeepGenPQRConfig(
                    policy=problem.oracle_policy,
                    q_mode="pooled_fqe",
                    q_backend="fqe_action_head_neural",
                    seed=int(seed),
                    n_action_samples=problem.n_actions,
                    q_config=_action_head_fqe_config(neural_fast=bool(neural_fast), seed=int(seed)),
                ),
            )
        )
    if include_native_bc:
        rows.append(
            _fit_deepgenpqr_row(
                name="native_bc_pooled_action_head_fqe",
                problem=problem,
                normalization_policy=mu,
                config=DeepGenPQRConfig(
                    policy="behavior_cloning_native",
                    q_mode="pooled_fqe",
                    q_backend="fqe_action_head_neural",
                    seed=int(seed),
                    n_action_samples=problem.n_actions,
                    policy_config={"n_epochs": 250, "learning_rate": 0.04, "l2": 1e-3},
                    q_config=_action_head_fqe_config(neural_fast=bool(neural_fast), seed=int(seed) + 17),
                ),
            )
        )
    if include_neural_deeppqr:
        rows.append(
            _fit_deepgenpqr_row(
                name="oracle_policy_neural_deeppqr",
                problem=problem,
                normalization_policy=mu,
                config=DeepGenPQRConfig(
                    policy=problem.oracle_policy,
                    q_mode="anchor_deeppqr",
                    anchor_backend="neural_deeppqr",
                    anchor_action=problem.anchor_action,
                    min_anchor_count=5,
                    seed=int(seed),
                    n_action_samples=problem.n_actions,
                    q_config=_neural_deeppqr_config(neural_fast=bool(neural_fast)),
                ),
            )
        )

    return _json_safe(
        {
            "benchmark": "soft_grid_reward_recovery",
            "problem": {
                **problem.metadata,
                "dataset": problem.dataset.summary(),
                "reward_scale": {
                    "mean": float(np.mean(problem.true_rewards)),
                    "std": float(np.std(problem.true_rewards)),
                    "min": float(np.min(problem.true_rewards)),
                    "max": float(np.max(problem.true_rewards)),
                },
            },
            "rows": rows,
        }
    )


def evaluate_reward_matrix(
    predicted_rewards: Array,
    problem: SoftGridRewardRecoveryProblem,
) -> dict[str, Any]:
    """Compute normalized-reward recovery metrics for a full reward matrix."""

    pred = np.asarray(predicted_rewards, dtype=np.float64)
    true = np.asarray(problem.true_rewards, dtype=np.float64)
    if pred.shape != true.shape:
        raise ValueError(f"predicted_rewards must have shape {true.shape}.")
    if not np.all(np.isfinite(pred)):
        raise FloatingPointError("predicted_rewards must be finite.")
    err = pred - true
    true_flat = true.reshape(-1)
    pred_flat = pred.reshape(-1)
    true_std = float(np.std(true_flat))
    pred_std = float(np.std(pred_flat))
    corr = np.nan
    if true_std > 1e-12 and pred_std > 1e-12:
        corr = float(np.corrcoef(true_flat, pred_flat)[0, 1])
    residual = pred[:, int(problem.anchor_action)]
    action_counts = np.bincount(
        problem.action_space.action_indices(problem.dataset.actions, n_rows=problem.dataset.n_rows),
        minlength=problem.n_actions,
    )
    per_action_rmse = np.sqrt(np.mean(err**2, axis=0))
    per_action_mae = np.mean(np.abs(err), axis=0)
    rare_action = int(np.argmin(action_counts))
    common_action = int(np.argmax(action_counts))
    nonanchor_actions = np.asarray(
        [action for action in range(problem.n_actions) if action != int(problem.anchor_action)],
        dtype=np.int64,
    )
    rare_nonanchor = int(nonanchor_actions[np.argmin(action_counts[nonanchor_actions])])
    common_nonanchor = int(nonanchor_actions[np.argmax(action_counts[nonanchor_actions])])
    return {
        "reward_rmse": float(np.sqrt(np.mean(err**2))),
        "reward_mae": float(np.mean(np.abs(err))),
        "reward_correlation": corr,
        "anchor_rmse": float(np.sqrt(np.mean(residual**2))),
        "normalization_residual_abs_mean": float(np.mean(np.abs(residual))),
        "normalization_residual_max_abs": float(np.max(np.abs(residual))),
        "action_ranking_accuracy": float(np.mean(np.argmax(pred, axis=1) == np.argmax(true, axis=1))),
        "anchor_support_count": int(action_counts[int(problem.anchor_action)]),
        "anchor_support_fraction": float(action_counts[int(problem.anchor_action)] / max(problem.dataset.n_rows, 1)),
        "min_behavior_probability": float(np.min(problem.behavior_policy.probabilities)),
        "mean_behavior_anchor_probability": float(
            np.mean(problem.behavior_policy.probabilities[:, int(problem.anchor_action)])
        ),
        "observed_action_counts": action_counts.astype(int).tolist(),
        "per_action_rmse": per_action_rmse.astype(float).tolist(),
        "per_action_mae": per_action_mae.astype(float).tolist(),
        "rare_action_index": rare_action,
        "rare_action_count": int(action_counts[rare_action]),
        "rare_action_rmse": float(per_action_rmse[rare_action]),
        "common_action_index": common_action,
        "common_action_count": int(action_counts[common_action]),
        "common_action_rmse": float(per_action_rmse[common_action]),
        "mean_nonanchor_rmse": float(np.mean(per_action_rmse[nonanchor_actions])),
        "max_nonanchor_rmse": float(np.max(per_action_rmse[nonanchor_actions])),
        "rare_nonanchor_action_index": rare_nonanchor,
        "rare_nonanchor_action_count": int(action_counts[rare_nonanchor]),
        "rare_nonanchor_action_rmse": float(per_action_rmse[rare_nonanchor]),
        "common_nonanchor_action_index": common_nonanchor,
        "common_nonanchor_action_count": int(action_counts[common_nonanchor]),
        "common_nonanchor_action_rmse": float(per_action_rmse[common_nonanchor]),
    }


def _make_grid(n_x: int, n_y: int, low: float, high: float) -> Array:
    x = np.linspace(float(low), float(high), int(n_x))
    y = np.linspace(float(low), float(high), int(n_y))
    xx, yy = np.meshgrid(x, y, indexing="xy")
    return np.column_stack([xx.reshape(-1), yy.reshape(-1)]).astype(np.float64)


def _action_vectors(action_scale: float) -> Array:
    return float(action_scale) * np.asarray(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.0, 1.0],
            [0.0, -1.0],
        ],
        dtype=np.float64,
    )


def _transition_means(
    *,
    grid_states: Array,
    low: float,
    high: float,
    action_scale: float,
    drift_scale: float,
    interaction_scale: float,
) -> Array:
    actions = _action_vectors(float(action_scale))
    x = grid_states[:, 0]
    y = grid_states[:, 1]
    drift = np.column_stack(
        [
            float(drift_scale) * np.sin(np.pi * y) + float(interaction_scale) * x * (1.0 - y**2),
            float(drift_scale) * np.cos(np.pi * x) - float(interaction_scale) * y * (1.0 - x**2),
        ]
    )
    action_phase = np.arange(actions.shape[0], dtype=np.float64).reshape(1, -1)
    swirl = float(interaction_scale) * np.stack(
        [
            np.sin(2.0 * y[:, None] + action_phase),
            np.cos(2.0 * x[:, None] - action_phase),
        ],
        axis=2,
    )
    means = grid_states[:, None, :] + actions[None, :, :] + drift[:, None, :] + swirl
    return np.clip(means, float(low), float(high))


def _transition_kernel(
    *,
    grid_states: Array,
    transition_means: Array,
    process_noise: float,
    teleport_prob: float,
) -> Array:
    noise = max(float(process_noise), 1e-4)
    diff = transition_means[:, :, None, :] - grid_states[None, None, :, :]
    scores = -np.sum(diff**2, axis=3) / (2.0 * noise**2)
    scores = scores - np.max(scores, axis=2, keepdims=True)
    kernel = np.exp(scores)
    kernel = kernel / kernel.sum(axis=2, keepdims=True)
    if float(teleport_prob) > 0.0:
        n_states = grid_states.shape[0]
        teleport = min(max(float(teleport_prob), 0.0), 1.0)
        kernel = (1.0 - teleport) * kernel + teleport / float(n_states)
    return kernel


def _raw_reward_matrix(
    *,
    grid_states: Array,
    transition_means: Array,
    action_scale: float,
    goal: Array,
    decoy: Array,
    goal_width: float,
    decoy_width: float,
    barrier_width: float,
    action_cost: float,
) -> Array:
    state_x = grid_states[:, 0]
    state_y = grid_states[:, 1]
    state_baseline = 0.22 * np.sin(2.0 * np.pi * state_x) + 0.13 * np.cos(3.0 * np.pi * state_y)
    goal_bonus = 1.75 * _bump(transition_means, goal.reshape(1, 1, 2), float(goal_width))
    decoy_penalty = 1.05 * _bump(transition_means, decoy.reshape(1, 1, 2), float(decoy_width))
    barrier_coordinate = transition_means[:, :, 0] + 0.25 * transition_means[:, :, 1]
    ridge = 0.45 * np.exp(-(barrier_coordinate**2) / (2.0 * float(barrier_width) ** 2))
    actions = _action_vectors(float(action_scale))
    effort = (
        float(action_cost)
        * np.sum(actions**2, axis=1)
        / max(float(action_scale) ** 2, 1e-12)
    )
    direction_bonus = 0.08 * np.sin(3.0 * transition_means[:, :, 0] - 2.0 * transition_means[:, :, 1])
    return state_baseline[:, None] + goal_bonus - decoy_penalty - ridge + direction_bonus - effort[None, :]


def _bump(points: Array, center: Array, width: float) -> Array:
    return np.exp(-np.sum((points - center) ** 2, axis=-1) / (2.0 * max(float(width), 1e-8) ** 2))


def _soft_value_iteration(
    *,
    rewards: Array,
    transition: Array,
    gamma: float,
    temperature: float,
    max_iter: int = 10_000,
    tol: float = 1e-10,
) -> tuple[Array, Array]:
    values = np.zeros(rewards.shape[0], dtype=np.float64)
    q = np.array(rewards, dtype=np.float64, copy=True)
    for _ in range(int(max_iter)):
        q_new = rewards + float(gamma) * np.einsum("sak,k->sa", transition, values)
        values_new = _soft_value(q_new, temperature=float(temperature))
        if float(np.max(np.abs(values_new - values))) < float(tol):
            return q_new, values_new
        q = q_new
        values = values_new
    return q, values


def _soft_value(q: Array, *, temperature: float) -> Array:
    z = q / float(temperature)
    zmax = np.max(z, axis=1, keepdims=True)
    return float(temperature) * (zmax[:, 0] + np.log(np.sum(np.exp(z - zmax), axis=1)))


def _softmax(x: Array, *, axis: int) -> Array:
    x_shift = x - np.max(x, axis=axis, keepdims=True)
    out = np.exp(x_shift)
    return out / np.sum(out, axis=axis, keepdims=True)


def _floor_probabilities(probs: Array, *, floor: float) -> Array:
    clipped = np.maximum(np.asarray(probs, dtype=np.float64), float(floor))
    return clipped / np.sum(clipped, axis=1, keepdims=True)


def _decoy_policy(*, transition_means: Array, decoy: Array, decoy_width: float, temperature: float) -> Array:
    scores = 1.4 * _bump(transition_means, decoy.reshape(1, 1, 2), float(decoy_width))
    scores -= 0.02 * np.arange(transition_means.shape[1], dtype=np.float64).reshape(1, -1)
    return _softmax(scores / float(temperature), axis=1)


def _stationary_distribution(
    transition: Array,
    policy_probs: Array,
    *,
    max_iter: int = 20_000,
    tol: float = 1e-13,
) -> Array:
    state_kernel = np.einsum("sa,sak->sk", policy_probs, transition)
    dist = np.full(state_kernel.shape[0], 1.0 / state_kernel.shape[0], dtype=np.float64)
    for _ in range(int(max_iter)):
        new_dist = dist @ state_kernel
        if np.max(np.abs(new_dist - dist)) < float(tol):
            return new_dist / np.sum(new_dist)
        dist = new_dist
    return dist / np.sum(dist)


def _sample_soft_grid_dataset(
    *,
    rng: np.random.Generator,
    grid_states: Array,
    transition: Array,
    behavior_probs: Array,
    state_distribution: Array,
    action_space: ActionSpaceSpec,
    sample_size: int,
    anchor_action: int,
    min_anchor_fraction: float,
) -> TransitionDataset:
    n_states, n_actions = behavior_probs.shape
    state_idx = rng.choice(np.arange(n_states, dtype=np.int64), size=int(sample_size), p=state_distribution)
    actions = np.empty(int(sample_size), dtype=np.int64)
    choices = np.arange(n_actions, dtype=np.int64)
    for i, state in enumerate(state_idx):
        actions[i] = rng.choice(choices, p=behavior_probs[state])
    _ensure_anchor_rows(
        actions,
        anchor_action=int(anchor_action),
        min_anchor_fraction=float(min_anchor_fraction),
        rng=rng,
    )
    next_idx = np.empty(int(sample_size), dtype=np.int64)
    for i, (state, action) in enumerate(zip(state_idx, actions)):
        next_idx[i] = rng.choice(np.arange(n_states, dtype=np.int64), p=transition[state, action])
    init_count = min(max(32, n_states), max(int(sample_size), 1))
    initial_idx = rng.choice(np.arange(n_states, dtype=np.int64), size=init_count, p=state_distribution)
    initial_actions = np.empty(init_count, dtype=np.int64)
    for i, state in enumerate(initial_idx):
        initial_actions[i] = rng.choice(choices, p=behavior_probs[state])
    return TransitionDataset.from_arrays(
        states=grid_states[state_idx],
        actions=actions,
        next_states=grid_states[next_idx],
        terminals=np.zeros(int(sample_size), dtype=np.float64),
        action_space=action_space,
        episode_ids=np.arange(int(sample_size), dtype=np.int64),
        initial_states=grid_states[initial_idx],
        initial_actions=initial_actions,
        metadata={"source": "soft_grid_reward_recovery"},
    )


def _ensure_anchor_rows(
    actions: Array,
    *,
    anchor_action: int,
    min_anchor_fraction: float,
    rng: np.random.Generator,
) -> None:
    n_rows = int(actions.shape[0])
    if float(min_anchor_fraction) <= 0.0:
        return
    target = min(n_rows, int(np.ceil(float(min_anchor_fraction) * n_rows)))
    current = np.flatnonzero(actions == int(anchor_action))
    if current.shape[0] >= target:
        return
    candidates = np.flatnonzero(actions != int(anchor_action))
    if candidates.shape[0] == 0:
        return
    chosen = rng.choice(candidates, size=min(target - current.shape[0], candidates.shape[0]), replace=False)
    actions[chosen] = int(anchor_action)


def _baseline_row(name: str, rewards: Array, problem: SoftGridRewardRecoveryProblem) -> dict[str, Any]:
    metrics = evaluate_reward_matrix(rewards, problem)
    return {
        "name": name,
        "status": "ok",
        "method": "baseline",
        "metrics": metrics,
        "summary": {
            "benchmark_reward_rmse": metrics["reward_rmse"],
            "benchmark_anchor_rmse": metrics["anchor_rmse"],
            "normalization_residual_abs_mean": metrics["normalization_residual_abs_mean"],
        },
    }


def _fit_deepgenpqr_row(
    *,
    name: str,
    problem: SoftGridRewardRecoveryProblem,
    normalization_policy: DiscreteNormalizationPolicy,
    config: DeepGenPQRConfig,
) -> dict[str, Any]:
    try:
        _prepare_optional_backend_runtime(config)
        result = fit_deep_genpqr(
            dataset=problem.dataset,
            gamma=problem.gamma,
            normalization_policy=normalization_policy,
            anchor_function=0.0,
            config=config,
        )
        pred = result.reward_function.predict_reward_matrix(problem.grid_states)
        metrics = evaluate_reward_matrix(pred, problem)
        _attach_benchmark_metrics(result, metrics)
        return {
            "name": name,
            "status": "ok",
            "method": "deepgenpqr",
            "metrics": metrics,
            "summary": result.summary(),
            "diagnostics": {
                "deepgenpqr_mode": result.diagnostics.get("deepgenpqr_mode"),
                "policy_backend": result.diagnostics.get("policy_backend"),
                "q_backend": result.diagnostics.get("q_backend"),
                "anchor_support": result.diagnostics.get("anchor_support"),
            },
        }
    except Exception as exc:  # pragma: no cover - exercised when optional stacks are absent.
        return {
            "name": name,
            "status": "skipped_or_error",
            "method": "deepgenpqr",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def _prepare_optional_backend_runtime(config: DeepGenPQRConfig) -> None:
    names = [
        getattr(config, "q_backend", None),
        getattr(config, "anchor_backend", None),
    ]
    lowered = {str(name).lower() for name in names if isinstance(name, str)}
    neural_fqe_names = {
        "fqe_neural",
        "neural_fqe",
        "fqe_action_head_neural",
        "action_head_neural_fqe",
        "stratified_neural_fqe",
    }
    if lowered & neural_fqe_names:
        os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
    if not lowered & (neural_fqe_names | {"neural_deeppqr", "deeppqr_neural"}):
        return
    try:
        import torch
    except ModuleNotFoundError:
        return
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def _attach_benchmark_metrics(result: DeepGenPQRResult, metrics: dict[str, Any]) -> None:
    payload = {
        "benchmark_reward_rmse": metrics["reward_rmse"],
        "benchmark_reward_mae": metrics["reward_mae"],
        "benchmark_reward_correlation": metrics["reward_correlation"],
        "benchmark_anchor_rmse": metrics["anchor_rmse"],
        "benchmark_action_ranking_accuracy": metrics["action_ranking_accuracy"],
        "benchmark_anchor_support_fraction": metrics["anchor_support_fraction"],
    }
    result.diagnostics.update(payload)
    result.genpqr_result.diagnostics.update(payload)
    if result.genpqr_result.diagnostics_report is not None:
        result.genpqr_result.diagnostics_report.extra.update(payload)


def _action_head_fqe_config(*, neural_fast: bool, seed: int) -> dict[str, Any]:
    if neural_fast:
        overrides = {
            "hidden_dims": (128, 128),
            "head_hidden_dims": (64,),
            "batch_size": 128,
            "num_iterations": 80,
            "gradient_steps_per_iteration": 24,
            "patience": 12,
            "target_update_tau": 0.35,
            "seed": int(seed),
        }
    else:
        overrides = {"seed": int(seed)}
    return {"config_overrides": overrides, "n_next_action_samples": 8}


def _neural_deeppqr_config(*, neural_fast: bool) -> dict[str, Any]:
    if neural_fast:
        return {
            "hidden_dims": (64, 64),
            "batch_size": 128,
            "max_epochs": 80,
            "patience": 8,
            "validation_fraction": 0.2,
        }
    return {}


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


__all__ = [
    "SoftGridOraclePolicy",
    "SoftGridRewardRecoveryProblem",
    "evaluate_reward_matrix",
    "make_soft_grid_reward_recovery",
    "run_reward_recovery_benchmark",
]
