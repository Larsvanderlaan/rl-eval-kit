"""Tiny benchmark datasets for GenPQR examples and smoke tests."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import numpy as np

from genpqr.datasets import TransitionDataset
from genpqr.q_estimators import ConstantFittedQFunction
from genpqr.reward_benchmarks import (
    SoftGridOraclePolicy,
    SoftGridRewardRecoveryProblem,
    evaluate_reward_matrix,
    make_soft_grid_reward_recovery,
    run_reward_recovery_benchmark,
)
from genpqr.types import ActionSpaceSpec

__all__ = [
    "SoftGridOraclePolicy",
    "SoftGridRewardRecoveryProblem",
    "evaluate_reward_matrix",
    "make_gridworld",
    "make_linear_gaussian",
    "make_soft_grid_reward_recovery",
    "make_tabular_chain",
    "run_reward_recovery_benchmark",
    "run_tiny_production_validation",
]


def make_tabular_chain(n_steps: int = 100, *, seed: int = 123) -> TransitionDataset:
    """Create a small two-action chain dataset."""

    rng = np.random.default_rng(int(seed))
    state_idx = rng.integers(0, 5, size=int(n_steps))
    actions = rng.integers(0, 2, size=int(n_steps))
    next_idx = np.clip(state_idx + np.where(actions == 1, 1, -1), 0, 4)
    terminals = (next_idx == 4).astype(float)
    return TransitionDataset.from_arrays(
        states=state_idx.reshape(-1, 1),
        actions=actions,
        next_states=next_idx.reshape(-1, 1),
        terminals=terminals,
        action_space=ActionSpaceSpec.discrete(2),
        episode_ids=np.arange(int(n_steps)),
    )


def make_gridworld(n_steps: int = 100, *, seed: int = 123) -> TransitionDataset:
    """Create a tiny four-action gridworld transition dataset."""

    rng = np.random.default_rng(int(seed))
    states = rng.integers(0, 4, size=(int(n_steps), 2))
    actions = rng.integers(0, 4, size=int(n_steps))
    moves = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
    next_states = np.clip(states + moves[actions], 0, 3)
    terminals = np.all(next_states == 3, axis=1).astype(float)
    return TransitionDataset.from_arrays(
        states=states,
        actions=actions,
        next_states=next_states,
        terminals=terminals,
        action_space=ActionSpaceSpec.discrete(4),
        episode_ids=np.arange(int(n_steps)),
    )


def make_linear_gaussian(n_steps: int = 100, *, seed: int = 123) -> TransitionDataset:
    """Create a continuous-action linear-Gaussian transition dataset."""

    rng = np.random.default_rng(int(seed))
    states = rng.normal(size=(int(n_steps), 2))
    actions = states[:, [0]] + rng.normal(scale=0.5, size=(int(n_steps), 1))
    next_states = 0.8 * states + np.concatenate([actions, -actions], axis=1) * 0.1
    terminals = np.zeros(int(n_steps), dtype=float)
    return TransitionDataset.from_arrays(
        states=states,
        actions=actions,
        next_states=next_states,
        terminals=terminals,
        action_space=ActionSpaceSpec.continuous(1),
        episode_ids=np.arange(int(n_steps)),
    )


def run_tiny_production_validation(*, n_steps: int = 32, seed: int = 123) -> dict[str, Any]:
    """Run a CI-friendly GenPQR/DeepGenPQR validation suite.

    The suite intentionally uses dependency-light native policies and portable
    constant-Q pooled backends. It validates package wiring, diagnostics shape,
    reward finiteness, anchor support reporting, and cross-fit instability
    without requiring optional AIRL/FQE stacks.
    """

    from genpqr.api import GenPQRConfig
    from genpqr.crossfit import fit_genpqr_crossfit
    from genpqr.normalization import DiscreteNormalizationPolicy

    chain = make_tabular_chain(n_steps=n_steps, seed=seed)
    grid = make_gridworld(n_steps=n_steps, seed=seed + 1)
    gaussian = make_linear_gaussian(n_steps=n_steps, seed=seed + 2)
    report: dict[str, Any] = {
        "tabular_chain": _run_discrete_validation(chain, gamma=0.0, n_actions=2),
        "gridworld": _run_discrete_validation(grid, gamma=0.0, n_actions=4),
        "linear_gaussian": _run_continuous_validation(gaussian, gamma=0.0),
    }
    crossfit = fit_genpqr_crossfit(
        dataset=chain,
        gamma=0.0,
        n_folds=2,
        normalization_policy=DiscreteNormalizationPolicy.uniform(2),
        config=GenPQRConfig(
            policy="behavior_cloning_native",
            q=_ConstantQEstimator(value=0.0, backend="benchmark_constant_q"),
            policy_config={"n_epochs": 10},
        ),
    )
    report["tabular_chain"]["crossfit"] = {
        "fold_reward_mean_spread": crossfit.diagnostics["fold_reward_mean_spread"],
        "fold_reward_std_spread": crossfit.diagnostics["fold_reward_std_spread"],
        "final_refit_reward_correlation": crossfit.diagnostics["final_refit_reward_correlation"],
    }
    return _json_safe(report)


def _run_discrete_validation(dataset: TransitionDataset, *, gamma: float, n_actions: int) -> dict[str, Any]:
    from genpqr.api import GenPQRConfig, fit_genpqr
    from genpqr.deepgenpqr import DeepGenPQRConfig, fit_deep_genpqr
    from genpqr.normalization import DiscreteNormalizationPolicy

    mu = DiscreteNormalizationPolicy.uniform(n_actions)
    common_policy = {"n_epochs": 10}
    return {
        "fit_genpqr": _fit_summary(
            "fit_genpqr",
            lambda: fit_genpqr(
                dataset=dataset,
                gamma=gamma,
                normalization_policy=mu,
                config=GenPQRConfig(
                    policy="behavior_cloning_native",
                    q=_ConstantQEstimator(value=0.0, backend="benchmark_constant_q"),
                    policy_config=common_policy,
                ),
            ),
            dataset,
        ),
        "deepgenpqr_pooled_fqe": _fit_summary(
            "deepgenpqr_pooled_fqe",
            lambda: fit_deep_genpqr(
                dataset=dataset,
                gamma=gamma,
                normalization_policy=mu,
                config=DeepGenPQRConfig(
                    policy="behavior_cloning_native",
                    q_backend=_ConstantQEstimator(value=0.0, backend="benchmark_pooled_q"),
                    policy_config=common_policy,
                ),
            ),
            dataset,
        ),
        "deepgenpqr_anchor_deeppqr": _fit_summary(
            "deepgenpqr_anchor_deeppqr",
            lambda: fit_deep_genpqr(
                dataset=dataset,
                gamma=gamma,
                normalization_policy=mu,
                config=DeepGenPQRConfig(
                    policy="behavior_cloning_native",
                    q_mode="anchor_deeppqr",
                    anchor_backend="deeppqr_linear",
                    anchor_action=0,
                    policy_config=common_policy,
                    q_config={"n_iterations": 5},
                ),
            ),
            dataset,
        ),
    }


def _run_continuous_validation(dataset: TransitionDataset, *, gamma: float) -> dict[str, Any]:
    from genpqr.deepgenpqr import DeepGenPQRConfig, fit_deep_genpqr
    from genpqr.normalization import ContinuousNormalizationPolicy

    mu = ContinuousNormalizationPolicy(
        action_dim=int(dataset.action_space.action_dim),
        sampler=lambda states, rng, n: rng.normal(size=(states.shape[0], int(n), int(dataset.action_space.action_dim))),
    )
    return {
        "deepgenpqr_pooled_fqe": _fit_summary(
            "deepgenpqr_pooled_fqe",
            lambda: fit_deep_genpqr(
                dataset=dataset,
                gamma=gamma,
                normalization_policy=mu,
                config=DeepGenPQRConfig(
                    policy="behavior_cloning_native",
                    q_backend=_ConstantQEstimator(value=0.0, backend="benchmark_continuous_q"),
                    policy_config={"n_epochs": 10},
                    n_action_samples=4,
                ),
            ),
            dataset,
        )
    }


def _fit_summary(name: str, fit: Any, dataset: TransitionDataset) -> dict[str, Any]:
    start = time.perf_counter()
    result = fit()
    runtime = time.perf_counter() - start
    rewards = result.predict_reward(dataset.states, dataset.actions)
    diagnostics = dict(result.diagnostics)
    anchor_support = diagnostics.get("anchor_support")
    if anchor_support is None:
        anchor_support = {
            "anchor_count": diagnostics.get("q_anchor_count"),
            "anchor_fraction": diagnostics.get("q_anchor_fraction"),
            "weak_anchor_support": diagnostics.get("q_weak_anchor_support"),
        }
    return {
        "name": name,
        "runtime_seconds": float(runtime),
        "reward_finite_fraction": float(np.mean(np.isfinite(rewards))),
        "reward_mean": float(np.mean(rewards[np.isfinite(rewards)])) if np.any(np.isfinite(rewards)) else None,
        "normalization_residual_abs_mean": diagnostics.get("normalization_residual_abs_mean"),
        "q_backend": diagnostics.get("q_backend"),
        "anchor_support": anchor_support,
    }


@dataclass
class _ConstantQEstimator:
    value: float = 0.0
    backend: str = "benchmark_constant_q"

    def fit(self, *, normalization_policy: Any, **_: Any) -> ConstantFittedQFunction:
        return ConstantFittedQFunction(
            action_space=normalization_policy.action_space,
            value=float(self.value),
            backend=self.backend,
        )


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(val) for val in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)
