"""Run the 20-action complementary-source GenPQR experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from functools import partial
from multiprocessing import get_context
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence

for thread_variable in [
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
]:
    os.environ.setdefault(thread_variable, "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).resolve().parent
REPRO_ROOT = Path(__file__).resolve().parents[1]
CACHE_ROOT = Path(os.environ.get("TMPDIR", "/tmp")) / "irl_data_fusion_cache"
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_ROOT / "mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_ROOT / "xdg"))
if str(REPRO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPRO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_fusion import (
    DataFusionWorld,
    OutcomeSource,
    fit_state_reward_regressor,
    generate_data_fusion_world,
    recover_reward_matrix,
)
from data_generation import SimulationConfig, make_anchor_mu
from deeppqr_baseline import fit_deeppqr_baseline
from policy_estimation import fit_behavior_cloning_policy
from q_evaluation import fit_fqe_neural
from utils import EPS


FUSION_METHOD = "GenPQR fusion (estimated nuisances)"
ANCHOR_GEN_METHOD = "GenPQR anchor (behavior only)"
DEEP_METHOD = "DeepPQR anchor (behavior only)"
DEEP_PROJECTED_METHOD = "DeepPQR anchor-projected (sensitivity)"
ORACLE_METHOD = "GenPQR fusion (oracle nuisances)"


@dataclass(frozen=True)
class ExperimentConfig:
    """Immutable configuration for one experiment stage."""

    stage: str
    seeds: Sequence[int]
    outcome_trajectories: Sequence[int]
    behavior_trajectories: int
    test_trajectories: int
    horizon: int = 10
    state_dim: int = 5
    n_actions: int = 20
    gamma: float = 0.9
    anchor_logit_shift: float = 0.0
    bc_epochs: int = 40
    fqe_iters: int = 8
    fqe_epochs_per_iter: int = 4
    g_estimators: int = 150

    def validate(self) -> None:
        """Validate the run configuration."""
        if not self.seeds:
            raise ValueError("At least one seed is required.")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("Seeds must be unique.")
        if self.n_actions < 2:
            raise ValueError("n_actions must be at least two.")
        if self.behavior_trajectories <= 0 or self.test_trajectories <= 0:
            raise ValueError("Behavior and test trajectory counts must be positive.")
        if not self.outcome_trajectories or min(self.outcome_trajectories) <= 0:
            raise ValueError("Outcome trajectory counts must be positive.")
        if len(set(self.outcome_trajectories)) != len(self.outcome_trajectories):
            raise ValueError("Outcome trajectory counts must be unique.")
        if self.horizon <= 0 or self.state_dim <= 0:
            raise ValueError("horizon and state_dim must be positive.")
        if not 0.0 <= self.gamma < 1.0:
            raise ValueError("gamma must lie in [0, 1).")
        if min(self.bc_epochs, self.fqe_iters, self.fqe_epochs_per_iter, self.g_estimators) <= 0:
            raise ValueError("All estimator budgets must be positive.")


class PolicyAdapter:
    """Minimal policy interface used by FQE."""

    def __init__(self, predict_fn: Callable[[np.ndarray], np.ndarray], n_actions: int):
        self._predict_fn = predict_fn
        self.n_actions = n_actions

    def predict_proba(self, states: np.ndarray) -> np.ndarray:
        probabilities = np.asarray(self._predict_fn(np.asarray(states, dtype=float)), dtype=float)
        if probabilities.shape != (states.shape[0], self.n_actions):
            raise ValueError(f"Policy returned invalid shape {probabilities.shape}.")
        return probabilities

    def sample_actions(self, states: np.ndarray, seed: int | None = None) -> np.ndarray:
        rng = np.random.default_rng(0 if seed is None else seed)
        probabilities = self.predict_proba(states)
        return np.asarray([rng.choice(self.n_actions, p=row) for row in probabilities], dtype=int)


def _component_seed(seed: int, component: str, outcome_trajectories: int = 0) -> int:
    payload = f"{seed}:{component}:{outcome_trajectories}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "little") % (2**31 - 1)


def _configure_worker_threads() -> None:
    try:
        import torch

        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
    except ImportError:  # pragma: no cover - torch is a repro dependency
        pass


def _safe_correlation(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    if x.shape[0] < 2 or np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return math.nan
    return float(np.corrcoef(x, y)[0, 1])


def _common_metrics(
    *,
    reward_matrix: np.ndarray,
    world: DataFusionWorld,
    estimated_policy,
    plugin_policy: Callable[[np.ndarray], np.ndarray],
    plugin_g: Callable[[np.ndarray], np.ndarray],
    g_prediction: np.ndarray | None,
) -> Dict[str, float]:
    test = world.test
    reward_matrix = np.asarray(reward_matrix, dtype=float)
    if reward_matrix.shape != test.true_reward_matrix.shape:
        raise ValueError(
            f"Recovered reward has shape {reward_matrix.shape}, expected {test.true_reward_matrix.shape}."
        )
    if not np.all(np.isfinite(reward_matrix)):
        raise ValueError("Recovered reward contains nonfinite values.")

    error = reward_matrix - test.true_reward_matrix
    rows = np.arange(test.states.shape[0])
    sampled_error = error[rows, test.actions]
    true_policy = test.true_policy_probs
    estimated_policy_probs = np.asarray(estimated_policy.predict_proba(test.states), dtype=float)
    estimated_policy_probs = np.clip(estimated_policy_probs, EPS, 1.0)
    plugin_probs = np.asarray(plugin_policy(test.states), dtype=float)
    plugin_g_values = np.asarray(plugin_g(test.states), dtype=float).reshape(-1)
    plugin_residual = np.sum(plugin_probs * reward_matrix, axis=1) - plugin_g_values
    oracle_residual = np.sum(true_policy * reward_matrix, axis=1) - test.true_g
    true_probs_clipped = np.clip(true_policy, EPS, 1.0)

    metrics = {
        "reward_mse_behavior": float(np.mean(np.sum(true_policy * error**2, axis=1))),
        "reward_mae_behavior": float(np.mean(np.sum(true_policy * np.abs(error), axis=1))),
        "reward_mse_uniform": float(np.mean(error**2)),
        "reward_mae_uniform": float(np.mean(np.abs(error))),
        "reward_mse_sampled": float(np.mean(sampled_error**2)),
        "reward_correlation_uniform": _safe_correlation(
            reward_matrix.reshape(-1), test.true_reward_matrix.reshape(-1)
        ),
        "reward_correlation_sampled": _safe_correlation(
            reward_matrix[rows, test.actions],
            test.true_reward_matrix[rows, test.actions],
        ),
        "anchor_rmse": float(np.sqrt(np.mean(reward_matrix[:, 0] ** 2))),
        "normalization_identity_rmse": float(np.sqrt(np.mean(plugin_residual**2))),
        "oracle_normalization_rmse": float(np.sqrt(np.mean(oracle_residual**2))),
        "policy_nll": float(
            -np.mean(np.log(estimated_policy_probs[rows, test.actions]))
        ),
        "policy_oracle_kl": float(
            np.mean(
                np.sum(
                    true_policy
                    * (np.log(true_probs_clipped) - np.log(estimated_policy_probs)),
                    axis=1,
                )
            )
        ),
        "policy_min_probability": float(np.min(estimated_policy_probs)),
        "reward_prediction_max_abs": float(np.max(np.abs(reward_matrix))),
        "oracle_g_rmse": math.nan,
    }
    if g_prediction is not None:
        g_prediction = np.asarray(g_prediction, dtype=float).reshape(-1)
        if g_prediction.shape != test.true_g.shape:
            raise ValueError("g(s) prediction has an invalid shape.")
        metrics["oracle_g_rmse"] = float(
            np.sqrt(np.mean((g_prediction - test.true_g) ** 2))
        )
    return metrics


def _fit_genpqr(
    *,
    world: DataFusionWorld,
    config: ExperimentConfig,
    estimated_policy,
    log_policy,
    normalization_policy,
    normalization_function,
    seed: int,
) -> tuple[np.ndarray, float, float]:
    behavior = world.behavior
    policy_probs = np.clip(log_policy.predict_proba(behavior.states), EPS, 1.0)
    g_values = np.asarray(normalization_function(behavior.states), dtype=float).reshape(-1)
    pseudo_rewards = np.log(policy_probs[np.arange(behavior.actions.shape[0]), behavior.actions])
    pseudo_rewards = pseudo_rewards - g_values
    start = time.perf_counter()
    q_estimate = fit_fqe_neural(
        states=behavior.states,
        actions=behavior.actions,
        rewards=pseudo_rewards,
        next_states=behavior.next_states,
        dones=behavior.dones,
        policy=normalization_policy,
        n_actions=config.n_actions,
        gamma=config.gamma,
        n_fqe_iters=config.fqe_iters,
        epochs_per_iter=config.fqe_epochs_per_iter,
        seed=seed,
    )
    reward_matrix = recover_reward_matrix(
        q_estimate=q_estimate,
        normalization_policy=normalization_policy.predict_proba,
        normalization_function=normalization_function,
        states=world.test.states,
    )

    q_observed = q_estimate.predict_q(behavior.states, behavior.actions)
    q_next = q_estimate.predict_all_actions(behavior.next_states)
    next_value = np.sum(normalization_policy.predict_proba(behavior.next_states) * q_next, axis=1)
    bellman_target = pseudo_rewards + config.gamma * (1.0 - behavior.dones) * next_value
    bellman_rmse = float(np.sqrt(np.mean((q_observed - bellman_target) ** 2)))
    return reward_matrix, time.perf_counter() - start, bellman_rmse


def _fit_deeppqr(
    *,
    world: DataFusionWorld,
    config: ExperimentConfig,
    estimated_policy,
    seed: int,
) -> tuple[np.ndarray, float]:
    behavior = world.behavior
    start = time.perf_counter()
    estimate = fit_deeppqr_baseline(
        states=behavior.states,
        actions=behavior.actions,
        rewards=np.zeros(behavior.actions.shape[0], dtype=float),
        next_states=behavior.next_states,
        dones=behavior.dones,
        n_actions=config.n_actions,
        gamma=config.gamma,
        anchor_action=0,
        g_values=np.zeros(behavior.actions.shape[0], dtype=float),
        behavior_policy=estimated_policy,
        seed=seed,
    )
    states = world.test.states
    q_matrix = estimate.full_q.predict_all_actions(states)
    continuation_columns = []
    for action in range(config.n_actions):
        actions = np.full(states.shape[0], action, dtype=int)
        continuation_columns.append(estimate.reward_network.predict(states, actions))
    continuation_matrix = np.stack(continuation_columns, axis=1)
    reward_matrix = q_matrix - config.gamma * continuation_matrix
    return reward_matrix, time.perf_counter() - start


def _base_row(
    *,
    seed: int,
    method: str,
    outcome_trajectories: int,
    world: DataFusionWorld,
    config: ExperimentConfig,
    generation_seconds: float,
    policy_seconds: float,
) -> Dict[str, object]:
    behavior = world.behavior
    action_counts = np.bincount(behavior.actions, minlength=config.n_actions)
    return {
        "stage": config.stage,
        "seed": int(seed),
        "method": method,
        "outcome_trajectories": int(outcome_trajectories),
        "outcome_observations": int(
            0 if outcome_trajectories <= 0 else outcome_trajectories * config.horizon
        ),
        "behavior_trajectories": int(config.behavior_trajectories),
        "behavior_transitions": int(behavior.actions.shape[0]),
        "test_trajectories": int(config.test_trajectories),
        "n_actions": int(config.n_actions),
        "anchor_count": int(action_counts[0]),
        "anchor_fraction": float(action_counts[0] / behavior.actions.shape[0]),
        "minimum_action_count": int(np.min(action_counts)),
        "world_generation_seconds": float(generation_seconds),
        "policy_fit_seconds": float(policy_seconds),
        "g_fit_seconds": 0.0,
        "second_stage_seconds": math.nan,
        "fit_seconds_total": math.nan,
        "training_bellman_rmse": math.nan,
        "status": "ok",
        "error": "",
    }


def _failure_row(
    *,
    seed: int,
    method: str,
    outcome_trajectories: int,
    world: DataFusionWorld,
    config: ExperimentConfig,
    generation_seconds: float,
    policy_seconds: float,
    exc: BaseException,
) -> Dict[str, object]:
    row = _base_row(
        seed=seed,
        method=method,
        outcome_trajectories=outcome_trajectories,
        world=world,
        config=config,
        generation_seconds=generation_seconds,
        policy_seconds=policy_seconds,
    )
    row["status"] = "error"
    row["error"] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-2000:]}"
    return row


def _run_seed(seed: int, config: ExperimentConfig) -> Dict[str, object]:
    _configure_worker_threads()
    seed_start = time.perf_counter()
    simulation_config = SimulationConfig(
        state_dim=config.state_dim,
        n_actions=config.n_actions,
        horizon=config.horizon,
        gamma=config.gamma,
        anchor_logit_shift=config.anchor_logit_shift,
        seed=seed,
    )
    generation_start = time.perf_counter()
    world = generate_data_fusion_world(
        seed=seed,
        config=simulation_config,
        behavior_trajectories=config.behavior_trajectories,
        outcome_trajectories=max(config.outcome_trajectories),
        test_trajectories=config.test_trajectories,
    )
    generation_seconds = time.perf_counter() - generation_start
    action_counts = np.bincount(world.behavior.actions, minlength=config.n_actions)
    if np.any(action_counts == 0):
        raise ValueError(f"Behavior source omitted actions {np.flatnonzero(action_counts == 0).tolist()}.")

    policy_start = time.perf_counter()
    estimated_policy = fit_behavior_cloning_policy(
        world.behavior.states,
        world.behavior.actions,
        n_actions=config.n_actions,
        n_epochs=config.bc_epochs,
        seed=_component_seed(seed, "behavior_policy"),
    )
    policy_seconds = time.perf_counter() - policy_start
    rows: List[Dict[str, object]] = []

    estimated_policy_adapter = PolicyAdapter(estimated_policy.predict_proba, config.n_actions)
    anchor_function = make_anchor_mu(0, config.n_actions)
    anchor_policy = PolicyAdapter(anchor_function, config.n_actions)
    zero_g = lambda states: np.zeros(np.asarray(states).shape[0], dtype=float)
    true_policy = PolicyAdapter(world.oracle.policy_probs, config.n_actions)

    try:
        reward_matrix, method_seconds, bellman_rmse = _fit_genpqr(
            world=world,
            config=config,
            estimated_policy=estimated_policy,
            log_policy=estimated_policy,
            normalization_policy=anchor_policy,
            normalization_function=zero_g,
            seed=_component_seed(seed, "genpqr_fqe"),
        )
        row = _base_row(
            seed=seed,
            method=ANCHOR_GEN_METHOD,
            outcome_trajectories=0,
            world=world,
            config=config,
            generation_seconds=generation_seconds,
            policy_seconds=policy_seconds,
        )
        row.update(
            _common_metrics(
                reward_matrix=reward_matrix,
                world=world,
                estimated_policy=estimated_policy,
                plugin_policy=anchor_policy.predict_proba,
                plugin_g=zero_g,
                g_prediction=None,
            )
        )
        row["second_stage_seconds"] = method_seconds
        row["training_bellman_rmse"] = bellman_rmse
        row["fit_seconds_total"] = policy_seconds + method_seconds
        rows.append(row)
    except Exception as exc:
        rows.append(
            _failure_row(
                seed=seed,
                method=ANCHOR_GEN_METHOD,
                outcome_trajectories=0,
                world=world,
                config=config,
                generation_seconds=generation_seconds,
                policy_seconds=policy_seconds,
                exc=exc,
            )
        )
    try:
        reward_matrix, method_seconds = _fit_deeppqr(
            world=world,
            config=config,
            estimated_policy=estimated_policy,
            seed=_component_seed(seed, "deeppqr"),
        )
        row = _base_row(
            seed=seed,
            method=DEEP_METHOD,
            outcome_trajectories=0,
            world=world,
            config=config,
            generation_seconds=generation_seconds,
            policy_seconds=policy_seconds,
        )
        row.update(
            _common_metrics(
                reward_matrix=reward_matrix,
                world=world,
                estimated_policy=estimated_policy,
                plugin_policy=anchor_policy.predict_proba,
                plugin_g=zero_g,
                g_prediction=None,
            )
        )
        row["second_stage_seconds"] = method_seconds
        row["fit_seconds_total"] = policy_seconds + method_seconds
        rows.append(row)

        projected_reward_matrix = reward_matrix.copy()
        projected_reward_matrix[:, 0] = 0.0
        projected_row = _base_row(
            seed=seed,
            method=DEEP_PROJECTED_METHOD,
            outcome_trajectories=0,
            world=world,
            config=config,
            generation_seconds=generation_seconds,
            policy_seconds=policy_seconds,
        )
        projected_row.update(
            _common_metrics(
                reward_matrix=projected_reward_matrix,
                world=world,
                estimated_policy=estimated_policy,
                plugin_policy=anchor_policy.predict_proba,
                plugin_g=zero_g,
                g_prediction=None,
            )
        )
        projected_row["second_stage_seconds"] = method_seconds
        projected_row["fit_seconds_total"] = policy_seconds + method_seconds
        rows.append(projected_row)
    except Exception as exc:
        rows.append(
            _failure_row(
                seed=seed,
                method=DEEP_METHOD,
                outcome_trajectories=0,
                world=world,
                config=config,
                generation_seconds=generation_seconds,
                policy_seconds=policy_seconds,
                exc=exc,
            )
        )
        rows.append(
            _failure_row(
                seed=seed,
                method=DEEP_PROJECTED_METHOD,
                outcome_trajectories=0,
                world=world,
                config=config,
                generation_seconds=generation_seconds,
                policy_seconds=policy_seconds,
                exc=exc,
            )
        )

    try:
        reward_matrix, method_seconds, bellman_rmse = _fit_genpqr(
            world=world,
            config=config,
            estimated_policy=estimated_policy,
            log_policy=true_policy,
            normalization_policy=true_policy,
            normalization_function=world.oracle.g,
            seed=_component_seed(seed, "genpqr_fqe"),
        )
        row = _base_row(
            seed=seed,
            method=ORACLE_METHOD,
            outcome_trajectories=0,
            world=world,
            config=config,
            generation_seconds=generation_seconds,
            policy_seconds=0.0,
        )
        row.update(
            _common_metrics(
                reward_matrix=reward_matrix,
                world=world,
                estimated_policy=true_policy,
                plugin_policy=true_policy.predict_proba,
                plugin_g=world.oracle.g,
                g_prediction=world.test.true_g,
            )
        )
        row["second_stage_seconds"] = method_seconds
        row["training_bellman_rmse"] = bellman_rmse
        row["fit_seconds_total"] = method_seconds
        rows.append(row)
    except Exception as exc:
        rows.append(
            _failure_row(
                seed=seed,
                method=ORACLE_METHOD,
                outcome_trajectories=0,
                world=world,
                config=config,
                generation_seconds=generation_seconds,
                policy_seconds=0.0,
                exc=exc,
            )
        )

    for outcome_trajectories in sorted(config.outcome_trajectories):
        try:
            outcome_source: OutcomeSource = world.outcome.subset_trajectories(outcome_trajectories)
            g_start = time.perf_counter()
            g_estimate = fit_state_reward_regressor(
                outcome_source,
                seed=_component_seed(seed, "g_regression", outcome_trajectories),
                n_estimators=config.g_estimators,
            )
            g_seconds = time.perf_counter() - g_start
            reward_matrix, method_seconds, bellman_rmse = _fit_genpqr(
                world=world,
                config=config,
                estimated_policy=estimated_policy,
                log_policy=estimated_policy,
                normalization_policy=estimated_policy_adapter,
                normalization_function=g_estimate.predict,
                seed=_component_seed(seed, "genpqr_fqe"),
            )
            g_test_prediction = g_estimate.predict(world.test.states)
            row = _base_row(
                seed=seed,
                method=FUSION_METHOD,
                outcome_trajectories=outcome_trajectories,
                world=world,
                config=config,
                generation_seconds=generation_seconds,
                policy_seconds=policy_seconds,
            )
            row.update(
                _common_metrics(
                    reward_matrix=reward_matrix,
                    world=world,
                    estimated_policy=estimated_policy,
                    plugin_policy=estimated_policy_adapter.predict_proba,
                    plugin_g=g_estimate.predict,
                    g_prediction=g_test_prediction,
                )
            )
            row["g_fit_seconds"] = g_seconds
            row["second_stage_seconds"] = method_seconds
            row["training_bellman_rmse"] = bellman_rmse
            row["fit_seconds_total"] = policy_seconds + g_seconds + method_seconds
            rows.append(row)
        except Exception as exc:
            rows.append(
                _failure_row(
                    seed=seed,
                    method=FUSION_METHOD,
                    outcome_trajectories=outcome_trajectories,
                    world=world,
                    config=config,
                    generation_seconds=generation_seconds,
                    policy_seconds=policy_seconds,
                    exc=exc,
                )
            )

    return {
        "seed": seed,
        "rows": rows,
        "worker_seconds": time.perf_counter() - seed_start,
    }


SUMMARY_METRICS = [
    "reward_mse_behavior",
    "reward_mae_behavior",
    "reward_mse_uniform",
    "reward_mae_uniform",
    "reward_mse_sampled",
    "reward_correlation_uniform",
    "reward_correlation_sampled",
    "anchor_rmse",
    "normalization_identity_rmse",
    "oracle_normalization_rmse",
    "policy_nll",
    "policy_oracle_kl",
    "oracle_g_rmse",
    "training_bellman_rmse",
    "anchor_count",
    "anchor_fraction",
    "minimum_action_count",
    "world_generation_seconds",
    "policy_fit_seconds",
    "g_fit_seconds",
    "second_stage_seconds",
    "fit_seconds_total",
]


def _summarize(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (method, outcome_trajectories), group in results.groupby(
        ["method", "outcome_trajectories"], dropna=False, sort=False
    ):
        successful = group[group["status"] == "ok"]
        row: Dict[str, object] = {
            "method": method,
            "outcome_trajectories": int(outcome_trajectories),
            "n_total": int(group.shape[0]),
            "n_success": int(successful.shape[0]),
            "failure_rate": float(1.0 - successful.shape[0] / max(group.shape[0], 1)),
        }
        for metric in SUMMARY_METRICS:
            values = pd.to_numeric(successful.get(metric), errors="coerce").dropna()
            row[f"{metric}_mean"] = float(values.mean()) if not values.empty else math.nan
            row[f"{metric}_std"] = float(values.std(ddof=1)) if values.shape[0] > 1 else math.nan
            row[f"{metric}_ci95"] = (
                float(1.96 * values.std(ddof=1) / math.sqrt(values.shape[0]))
                if values.shape[0] > 1
                else math.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _paired_contrasts(results: pd.DataFrame) -> pd.DataFrame:
    successful = results[results["status"] == "ok"]
    fusion = successful[successful["method"] == FUSION_METHOD]
    rows = []
    for comparator in [
        DEEP_METHOD,
        DEEP_PROJECTED_METHOD,
        ANCHOR_GEN_METHOD,
        ORACLE_METHOD,
    ]:
        baseline = successful[successful["method"] == comparator][
            ["seed", "reward_mse_behavior", "reward_mse_uniform"]
        ]
        for outcome_trajectories, group in fusion.groupby("outcome_trajectories"):
            merged = group.merge(baseline, on="seed", suffixes=("_fusion", "_comparator"))
            for metric in ["reward_mse_behavior", "reward_mse_uniform"]:
                difference = (
                    merged[f"{metric}_fusion"] - merged[f"{metric}_comparator"]
                )
                rows.append(
                    {
                        "outcome_trajectories": int(outcome_trajectories),
                        "comparator": comparator,
                        "metric": metric,
                        "n": int(difference.shape[0]),
                        "fusion_minus_comparator_mean": float(difference.mean()),
                        "fusion_minus_comparator_ci95": (
                            float(1.96 * difference.std(ddof=1) / math.sqrt(difference.shape[0]))
                            if difference.shape[0] > 1
                            else math.nan
                        ),
                        "fusion_win_rate": float(np.mean(difference < 0.0)),
                    }
                )
    return pd.DataFrame(rows)


def _write_markdown_table(table: pd.DataFrame, path: Path) -> None:
    headers = list(table.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for values in table.itertuples(index=False, name=None):
        formatted = []
        for value in values:
            if isinstance(value, float):
                formatted.append("NA" if not np.isfinite(value) else f"{value:.4f}")
            else:
                formatted.append(str(value))
        lines.append("| " + " | ".join(formatted) + " |")
    path.write_text("\n".join(lines) + "\n")


def _plot_results(summary: pd.DataFrame, output_dir: Path) -> None:
    fusion = summary[summary["method"] == FUSION_METHOD].sort_values("outcome_trajectories")
    if fusion.empty:
        return
    x = fusion["outcome_trajectories"].to_numpy(dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.3))
    for ax, metric, ylabel in [
        (axes[0], "reward_mse_behavior", "Behavior-weighted reward MSE"),
        (axes[1], "reward_mse_uniform", "Uniform-action reward MSE"),
    ]:
        mean = fusion[f"{metric}_mean"].to_numpy(dtype=float)
        ci = fusion[f"{metric}_ci95"].fillna(0.0).to_numpy(dtype=float)
        ax.errorbar(x, mean, yerr=ci, marker="o", capsize=3, linewidth=2, label=FUSION_METHOD)
        for method, linestyle in [
            (DEEP_METHOD, "--"),
            (DEEP_PROJECTED_METHOD, (0, (3, 2, 1, 2))),
            (ANCHOR_GEN_METHOD, "-."),
            (ORACLE_METHOD, ":"),
        ]:
            row = summary[summary["method"] == method]
            if row.empty:
                continue
            y = float(row.iloc[0][f"{metric}_mean"])
            ax.axhline(y, linestyle=linestyle, linewidth=2, label=method)
        ax.set_xscale("log")
        ax.set_xlabel("Auxiliary outcome trajectories")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.2)
    axes[1].legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "reward_mse_vs_outcome_sample.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.4, 4.3))
    mean = fusion["oracle_g_rmse_mean"].to_numpy(dtype=float)
    ci = fusion["oracle_g_rmse_ci95"].fillna(0.0).to_numpy(dtype=float)
    ax.errorbar(x, mean, yerr=ci, marker="o", capsize=3, linewidth=2)
    ax.set_xscale("log")
    ax.set_xlabel("Auxiliary outcome trajectories")
    ax.set_ylabel(r"Oracle diagnostic: $g$ RMSE")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "g_rmse_vs_outcome_sample.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.0, 5.5))
    behavior_mean = fusion["reward_mse_behavior_mean"].to_numpy(dtype=float)
    behavior_ci = fusion["reward_mse_behavior_ci95"].fillna(0.0).to_numpy(dtype=float)
    anchor = summary[summary["method"] == ANCHOR_GEN_METHOD]
    oracle = summary[summary["method"] == ORACLE_METHOD]
    if not anchor.empty:
        ax.axhline(
            float(anchor.iloc[0]["reward_mse_behavior_mean"]),
            color="#D55E00",
            linestyle="--",
            linewidth=3,
            label="Anchor GenPQR",
        )
    if not oracle.empty:
        ax.axhline(
            float(oracle.iloc[0]["reward_mse_behavior_mean"]),
            color="0.5",
            linestyle=":",
            linewidth=3,
            label="Oracle nuisances",
        )
    ax.errorbar(
        x,
        behavior_mean,
        yerr=behavior_ci,
        color="#0072B2",
        marker="o",
        markersize=8,
        capsize=4,
        linewidth=3,
        label="Fusion GenPQR",
    )
    ax.set_xscale("log")
    ax.set_xlabel("Auxiliary outcome trajectories", fontsize=18)
    ax.set_ylabel("Behavior-weighted reward MSE", fontsize=18)
    ax.tick_params(axis="both", labelsize=16)
    ax.grid(alpha=0.2)
    ax.legend(fontsize=16, loc="best")
    fig.tight_layout()
    fig.savefig(
        output_dir / "data_fusion_behavior_mse_compact.png",
        dpi=180,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close(fig)


def _atomic_write_part(result: Dict[str, object], parts_dir: Path) -> None:
    seed = int(result["seed"])
    final_path = parts_dir / f"seed_{seed}.csv"
    temporary_path = parts_dir / f".seed_{seed}.tmp"
    pd.DataFrame(result["rows"]).to_csv(temporary_path, index=False)
    os.replace(temporary_path, final_path)


def _load_parts(parts_dir: Path) -> pd.DataFrame:
    paths = sorted(parts_dir.glob("seed_*.csv"))
    if not paths:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)


def _git_sha() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPRO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unknown"


def _source_provenance() -> Dict[str, object]:
    relative_paths = [
        "data_generation.py",
        "data_fusion.py",
        "deeppqr_baseline.py",
        "policy_estimation.py",
        "q_evaluation.py",
        "utils.py",
        "experiments/data_fusion_large_action_runner.py",
        "tests/test_data_fusion.py",
    ]
    source_hashes = {}
    for relative_path in relative_paths:
        path = REPRO_ROOT / relative_path
        source_hashes[relative_path] = hashlib.sha256(path.read_bytes()).hexdigest()
    serialized = json.dumps(source_hashes, sort_keys=True).encode("utf-8")
    return {
        "source_hashes": source_hashes,
        "code_fingerprint": hashlib.sha256(serialized).hexdigest(),
    }


def run_experiment(config: ExperimentConfig, output_dir: Path, jobs: int) -> pd.DataFrame:
    """Run or resume a checkpointed experiment."""
    config.validate()
    if jobs <= 0:
        raise ValueError("jobs must be positive.")
    output_dir.mkdir(parents=True, exist_ok=True)
    parts_dir = output_dir / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    provenance = _source_provenance()
    manifest = {
        "config": asdict(config),
        "git_sha": _git_sha(),
        "code_fingerprint": provenance["code_fingerprint"],
        "source_hashes": provenance["source_hashes"],
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        if existing.get("config") != manifest["config"]:
            raise ValueError(
                f"Output directory {output_dir} contains a different configuration."
            )
        if existing.get("code_fingerprint") != manifest["code_fingerprint"]:
            raise ValueError(
                f"Output directory {output_dir} was produced by different source code; "
                "refusing to mix checkpoint parts."
            )
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    completed_seeds = {
        int(path.stem.removeprefix("seed_")) for path in parts_dir.glob("seed_*.csv")
    }
    pending = [seed for seed in config.seeds if seed not in completed_seeds]
    print(
        f"stage={config.stage} completed={len(completed_seeds)} pending={len(pending)} jobs={jobs}",
        flush=True,
    )

    if jobs == 1:
        iterator: Iterable[Dict[str, object]] = (_run_seed(seed, config) for seed in pending)
        for result in iterator:
            _atomic_write_part(result, parts_dir)
            print(
                f"completed seed={result['seed']} worker_seconds={result['worker_seconds']:.1f}",
                flush=True,
            )
    elif pending:
        with get_context("spawn").Pool(processes=jobs) as pool:
            worker = partial(_run_seed, config=config)
            for result in pool.imap_unordered(worker, pending):
                _atomic_write_part(result, parts_dir)
                print(
                    f"completed seed={result['seed']} worker_seconds={result['worker_seconds']:.1f}",
                    flush=True,
                )

    results = _load_parts(parts_dir)
    if results.empty:
        raise RuntimeError("No result parts were produced.")
    results = results.sort_values(["seed", "method", "outcome_trajectories"]).reset_index(drop=True)
    results.to_csv(output_dir / "results.csv", index=False)
    summary = _summarize(results)
    summary.to_csv(output_dir / "summary.csv", index=False)
    contrasts = _paired_contrasts(results)
    contrasts.to_csv(output_dir / "paired_contrasts.csv", index=False)
    table_columns = [
        "method",
        "outcome_trajectories",
        "n_success",
        "failure_rate",
        "reward_mse_behavior_mean",
        "reward_mse_behavior_ci95",
        "reward_mse_uniform_mean",
        "reward_mse_uniform_ci95",
        "reward_correlation_uniform_mean",
        "oracle_g_rmse_mean",
        "anchor_count_mean",
        "fit_seconds_total_mean",
    ]
    table = summary[table_columns].sort_values(["method", "outcome_trajectories"])
    table.to_csv(output_dir / "table.csv", index=False)
    _write_markdown_table(table, output_dir / "table.md")
    _plot_results(summary, output_dir)
    return results


def _stage_defaults(stage: str) -> Dict[str, object]:
    if stage == "debug":
        return {
            "replicates": 1,
            "seed_start": 890_000,
            "outcome_trajectories": [20, 50],
            "behavior_trajectories": 100,
            "test_trajectories": 50,
            "horizon": 5,
            "bc_epochs": 5,
            "fqe_iters": 2,
            "fqe_epochs_per_iter": 1,
            "g_estimators": 20,
        }
    if stage == "smoke":
        return {
            "replicates": 2,
            "seed_start": 900_000,
            "outcome_trajectories": [100, 5000],
            "behavior_trajectories": 2500,
            "test_trajectories": 300,
            "horizon": 10,
            "bc_epochs": 40,
            "fqe_iters": 8,
            "fqe_epochs_per_iter": 4,
            "g_estimators": 150,
        }
    if stage == "calibration":
        return {
            "replicates": 10,
            "seed_start": 910_000,
            "outcome_trajectories": [100, 250, 500, 1000, 2500, 5000],
            "behavior_trajectories": 2500,
            "test_trajectories": 300,
            "horizon": 10,
            "bc_epochs": 40,
            "fqe_iters": 8,
            "fqe_epochs_per_iter": 4,
            "g_estimators": 150,
        }
    if stage == "confirmatory":
        return {
            "replicates": 100,
            "seed_start": 920_000,
            "outcome_trajectories": [100, 250, 500, 1000, 2500, 5000],
            "behavior_trajectories": 2500,
            "test_trajectories": 300,
            "horizon": 10,
            "bc_epochs": 40,
            "fqe_iters": 8,
            "fqe_epochs_per_iter": 4,
            "g_estimators": 150,
        }
    if stage == "custom":
        return _stage_defaults("calibration")
    raise ValueError(f"Unknown stage {stage}.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=["debug", "smoke", "calibration", "confirmatory", "custom"],
        default="debug",
    )
    parser.add_argument("--replicates", type=int)
    parser.add_argument("--seed-start", type=int)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--outcome-trajectories", type=int, nargs="+")
    parser.add_argument("--behavior-trajectories", type=int)
    parser.add_argument("--test-trajectories", type=int)
    parser.add_argument("--horizon", type=int)
    parser.add_argument("--bc-epochs", type=int)
    parser.add_argument("--fqe-iters", type=int)
    parser.add_argument("--fqe-epochs-per-iter", type=int)
    parser.add_argument("--g-estimators", type=int)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    defaults = _stage_defaults(args.stage)
    for key in list(defaults):
        value = getattr(args, key, None)
        if value is not None:
            defaults[key] = value
    seeds = list(
        range(
            int(defaults["seed_start"]),
            int(defaults["seed_start"]) + int(defaults["replicates"]),
        )
    )
    config = ExperimentConfig(
        stage=args.stage,
        seeds=seeds,
        outcome_trajectories=list(defaults["outcome_trajectories"]),
        behavior_trajectories=int(defaults["behavior_trajectories"]),
        test_trajectories=int(defaults["test_trajectories"]),
        horizon=int(defaults["horizon"]),
        bc_epochs=int(defaults["bc_epochs"]),
        fqe_iters=int(defaults["fqe_iters"]),
        fqe_epochs_per_iter=int(defaults["fqe_epochs_per_iter"]),
        g_estimators=int(defaults["g_estimators"]),
    )
    output_dir = args.output_dir or ROOT / "outputs" / f"data_fusion_{args.stage}"
    results = run_experiment(config, output_dir=output_dir, jobs=args.jobs)
    failures = int(np.sum(results["status"] != "ok"))
    print(
        f"completed stage={args.stage} rows={results.shape[0]} failures={failures} output={output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
