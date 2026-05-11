#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
import sys
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

import matplotlib.pyplot as plt  # noqa: E402

from FQE_calibration_neurips.scripts.run_hopper_calibration_benchmark import (  # noqa: E402
    bellman_metrics,
    clipped_normalized_density_weights,
    fit_behavior_density,
    target_policy_log_prob,
)
from FQE_calibration_neurips.scripts.run_rlu_cheetah_benchmark import (  # noqa: E402
    ContinuousBatch,
    Predictor,
    fit_linear_or_rf_fqe,
)
from FQE_calibration_neurips.src.calibration.calibrators import (  # noqa: E402
    BaseCalibrator,
    IdentityCalibrator,
    fit_calibrator,
)
from hopper_fqe_benchmark.data import HopperTrajectoryDataset, load_hopper_dataset  # noqa: E402
from hopper_fqe_benchmark.fqe import QFitterConfig, QFitterResult, train_q_fitter  # noqa: E402
from hopper_fqe_benchmark.policies import HOPPER_MEDIUM_POLICY_SPECS, load_policy  # noqa: E402


DEFAULT_CALIBRATORS = ["none", "linear", "isotonic", "histogram", "isotonic_histogram"]
DEFAULT_CRITIC_FAMILIES = ["linear_fqe", "rf_fqe", "neural_fqe"]


@dataclass
class QBellmanCalibrator:
    base: BaseCalibrator
    method: str
    n_iterations: int
    diagnostics: dict[str, float | str]

    def predict(self, q_values: np.ndarray) -> np.ndarray:
        return self.base.predict(q_values)


def _replace_args(args: argparse.Namespace, **updates: object) -> argparse.Namespace:
    values = vars(args).copy()
    values.update(updates)
    return argparse.Namespace(**values)


def _fmt_float(value: float) -> str:
    text = f"{float(value):.0e}" if abs(float(value)) < 1e-2 else f"{float(value):g}"
    return text.replace("-", "m").replace("+", "").replace(".", "p")


def _safe_id(value: object) -> str:
    return str(value).replace("/", "_").replace(" ", "_").replace(":", "_")


def _parse_floats(raw: str) -> list[float]:
    return [float(x) for x in str(raw).replace(",", " ").split() if x]


def _parse_ints(raw: str) -> list[int]:
    return [int(x) for x in str(raw).replace(",", " ").split() if x]


def _critic_complexity(args: argparse.Namespace) -> float:
    if args.critic_family == "linear_fqe":
        return float(args.fit_action_samples)
    if args.critic_family == "rf_fqe":
        return float(args.rf_components) * float(args.fit_action_samples)
    return float(args.fqe_updates) * float(args.batch_size)


def _calibrator_complexity(args: argparse.Namespace) -> float:
    return float(args.n_bins) * float(args.calibration_iterations)


def _critic_config_id(args: argparse.Namespace) -> str:
    explicit = getattr(args, "critic_config_id", None)
    if explicit:
        return str(explicit)
    if args.critic_family == "linear_fqe":
        return f"linear_solver{args.linear_solver}_ridge{_fmt_float(args.ridge)}_a{args.fit_action_samples}"
    if args.critic_family == "rf_fqe":
        return (
            f"rf_c{args.rf_components}_solver{args.linear_solver}_"
            f"ridge{_fmt_float(args.ridge)}_a{args.fit_action_samples}"
        )
    return (
        f"neural_u{args.fqe_updates}_lr{_fmt_float(args.critic_lr)}_"
        f"tau{_fmt_float(args.target_tau)}_bs{args.batch_size}"
    )


def _calibrator_config_id(args: argparse.Namespace) -> str:
    explicit = getattr(args, "calibrator_config_id", None)
    if explicit:
        return str(explicit)
    return f"cal_bins{args.n_bins}_min{args.min_bin_size}_iter{args.calibration_iterations}"


def _unit_id(args: argparse.Namespace, seed: int, policy_id: str) -> str:
    return "__".join(
        [
            _safe_id(args.critic_family),
            _safe_id(_critic_config_id(args)),
            _safe_id(_calibrator_config_id(args)),
            f"seed{int(seed):04d}",
            _safe_id(policy_id),
        ]
    )


def _unit_path(output_dir: Path, args: argparse.Namespace, seed: int, policy_id: str) -> Path:
    return output_dir / "units" / f"{_unit_id(args, seed, policy_id)}.csv"


def _trajectory_ids_from_steps(steps: np.ndarray) -> np.ndarray:
    starts = np.asarray(steps).reshape(-1) == 0
    return np.cumsum(starts).astype(int) - 1


def _subset_dataset_by_trajectories(dataset: HopperTrajectoryDataset, trajectory_ids: Iterable[int]) -> HopperTrajectoryDataset:
    selected = np.asarray(sorted(set(int(i) for i in trajectory_ids)), dtype=int)
    transition_traj_ids = _trajectory_ids_from_steps(dataset.steps)
    mask = np.isin(transition_traj_ids, selected)
    return HopperTrajectoryDataset(
        observations_raw=dataset.observations_raw[mask],
        actions=dataset.actions[mask],
        next_observations_raw=dataset.next_observations_raw[mask],
        rewards_raw=dataset.rewards_raw[mask],
        masks=dataset.masks[mask],
        steps=dataset.steps[mask],
        initial_observations_raw=dataset.initial_observations_raw[selected],
        initial_weights=dataset.initial_weights[selected],
        state_mean=dataset.state_mean,
        state_std=dataset.state_std,
        reward_mean=dataset.reward_mean,
        reward_std=dataset.reward_std,
        observations=dataset.observations[mask],
        next_observations=dataset.next_observations[mask],
        rewards=dataset.rewards[mask],
        trajectory_count=int(selected.size),
    )


def _split_trajectories(
    n_trajectories: int,
    seed: int,
    fractions: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    ids = np.arange(int(n_trajectories))
    rng.shuffle(ids)
    n_train = max(1, int(round(float(fractions[0]) * len(ids))))
    n_cal = max(1, int(round(float(fractions[1]) * len(ids))))
    train = ids[:n_train]
    cal = ids[n_train : n_train + n_cal]
    diag = ids[n_train + n_cal :]
    if diag.size == 0:
        diag = cal[-1:]
        cal = cal[:-1]
    if cal.size == 0:
        cal = train[-1:]
        train = train[:-1]
    return train.astype(int), cal.astype(int), diag.astype(int)


def _load_ground_truth(benchmark_dir: Path) -> dict[str, float]:
    with (benchmark_dir / "d4rl_gt.pkl").open("rb") as handle:
        raw = pickle.load(handle)
    return {str(k): float(v[0]) for k, v in raw.items()}


def _as_continuous_batch(dataset: HopperTrajectoryDataset) -> ContinuousBatch:
    transition_traj_ids = _trajectory_ids_from_steps(dataset.steps)
    return ContinuousBatch(
        states=dataset.observations_raw.astype(np.float32),
        actions=dataset.actions.astype(np.float32),
        rewards=dataset.rewards.astype(np.float32),
        discounts=dataset.masks.astype(np.float32),
        next_states=dataset.next_observations_raw.astype(np.float32),
        episode_ids=transition_traj_ids.astype(np.int64),
        initial_states=dataset.initial_observations_raw.astype(np.float32),
    )


def _fit_frozen_critic(
    train_dataset: HopperTrajectoryDataset,
    policy: object,
    *,
    seed: int,
    args: argparse.Namespace,
) -> QFitterResult | Predictor:
    if args.critic_family == "neural_fqe":
        q_cfg = QFitterConfig(
            gamma=float(args.gamma),
            critic_lr=float(args.critic_lr),
            num_updates=int(args.fqe_updates),
            log_interval=max(1, int(args.fqe_updates) // 8),
            batch_size=int(args.batch_size),
            device=str(args.device),
            tau=float(args.target_tau),
        )
        return train_q_fitter(train_dataset, policy, config=q_cfg, seed=seed)
    return fit_linear_or_rf_fqe(
        _as_continuous_batch(train_dataset),
        policy,
        str(args.critic_family),
        gamma=float(args.gamma),
        ridge=float(args.ridge),
        n_iters=int(args.fqe_updates),
        n_components=int(args.rf_components),
        action_samples=int(args.fit_action_samples),
        seed=seed,
        solver=str(args.linear_solver),
    )


def _critic_diagnostics(result: QFitterResult | Predictor) -> dict[str, float | str]:
    if isinstance(result, QFitterResult):
        return {
            "fqe_loss_first": float(result.loss_history[0]) if result.loss_history else float("nan"),
            "fqe_loss_last": float(result.loss_history[-1]) if result.loss_history else float("nan"),
        }
    return {
        "fqe_loss_first": float("nan"),
        "fqe_loss_last": float("nan"),
        **(result.diagnostics or {}),
    }


def _predict_q_return(
    result: QFitterResult | Predictor,
    dataset: HopperTrajectoryDataset,
    states_raw: np.ndarray,
    actions: np.ndarray,
    *,
    gamma: float,
    device: str,
    chunk_size: int = 8192,
) -> np.ndarray:
    if isinstance(result, Predictor):
        return result.predict_q(np.asarray(states_raw, dtype=np.float32), np.asarray(actions, dtype=np.float32))
    model = result.target_model.to(device)
    model.eval()
    states_raw = np.asarray(states_raw, dtype=np.float32)
    actions = np.asarray(actions, dtype=np.float32)
    out = np.zeros(states_raw.shape[0], dtype=np.float64)
    divisor = max(1.0 - float(gamma), 1e-8)
    with torch.no_grad():
        for start in range(0, states_raw.shape[0], chunk_size):
            stop = min(start + chunk_size, states_raw.shape[0])
            pred = model(
                torch.as_tensor(dataset.normalize_states(states_raw[start:stop]), dtype=torch.float32, device=device),
                torch.as_tensor(actions[start:stop], dtype=torch.float32, device=device),
            )
            out[start:stop] = pred.cpu().numpy().astype(np.float64) / divisor
    return out


def _sampled_next_q_matrix(
    result: QFitterResult | Predictor,
    dataset: HopperTrajectoryDataset,
    policy: object,
    next_states_raw: np.ndarray,
    *,
    gamma: float,
    seed: int,
    action_samples: int,
    device: str,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    next_states_raw = np.asarray(next_states_raw, dtype=np.float32)
    out = np.zeros((next_states_raw.shape[0], int(action_samples)), dtype=np.float64)
    for sample_id in range(int(action_samples)):
        actions = policy.sample_actions(next_states_raw, rng=rng, deterministic=False)
        out[:, sample_id] = _predict_q_return(
            result,
            dataset,
            next_states_raw,
            actions,
            gamma=gamma,
            device=device,
        )
    return out


def _sampled_initial_q_matrix(
    result: QFitterResult | Predictor,
    dataset: HopperTrajectoryDataset,
    policy: object,
    *,
    gamma: float,
    seed: int,
    action_samples: int,
    device: str,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    states = dataset.initial_observations_raw
    out = np.zeros((states.shape[0], int(action_samples)), dtype=np.float64)
    for sample_id in range(int(action_samples)):
        actions = policy.sample_actions(states, rng=rng, deterministic=False)
        out[:, sample_id] = _predict_q_return(
            result,
            dataset,
            states,
            actions,
            gamma=gamma,
            device=device,
        )
    return out


def _weighted_average(x: np.ndarray, weights: np.ndarray | None = None) -> float:
    arr = np.asarray(x, dtype=float)
    if weights is None:
        return float(np.mean(arr))
    w = np.asarray(weights, dtype=float)
    return float(np.sum(w * arr) / max(float(np.sum(w)), 1e-12))


def _weighted_mean_square(x: np.ndarray, weights: np.ndarray | None = None) -> float:
    arr = np.asarray(x, dtype=float)
    if weights is None:
        return float(np.mean(arr * arr))
    w = np.asarray(weights, dtype=float)
    return float(np.sum(w * arr * arr) / max(float(np.sum(w)), 1e-12))


def _raw_return_from_normalized_return(values: np.ndarray, dataset: HopperTrajectoryDataset, gamma: float) -> float:
    return float(dataset.reward_std * np.mean(values) + dataset.reward_mean / max(1.0 - float(gamma), 1e-8))


def _fit_q_bellman_calibrator(
    method: str,
    q: np.ndarray,
    next_q_samples: np.ndarray,
    rewards: np.ndarray,
    masks: np.ndarray,
    *,
    gamma: float,
    sample_weight: np.ndarray | None,
    n_iterations: int,
    n_bins: int,
    min_bin_size: int,
) -> QBellmanCalibrator:
    if method in {"none", "identity"}:
        return QBellmanCalibrator(
            base=IdentityCalibrator(),
            method="none",
            n_iterations=0,
            diagnostics={
                "q_calibration_object": "q_function",
                "q_calibrator_method": "none",
                "q_calibration_iterations": 0.0,
            },
        )

    base: BaseCalibrator = IdentityCalibrator()
    losses: list[float] = []
    for _ in range(max(1, int(n_iterations))):
        next_cal = base.predict(next_q_samples.reshape(-1)).reshape(next_q_samples.shape)
        target = np.asarray(rewards, dtype=float) + float(gamma) * np.asarray(masks, dtype=float) * np.mean(next_cal, axis=1)
        base = fit_calibrator(
            method,
            q,
            target,
            n_bins=int(n_bins),
            bin_strategy="quantile",
            min_bin_size=int(min_bin_size),
            sample_weight=sample_weight,
        )
        pred = base.predict(q)
        losses.append(_weighted_mean_square(pred - target, sample_weight))
    return QBellmanCalibrator(
        base=base,
        method=method,
        n_iterations=max(1, int(n_iterations)),
        diagnostics={
            "q_calibration_object": "q_function",
            "q_calibrator_method": method,
            "q_calibration_iterations": float(max(1, int(n_iterations))),
            "q_calibration_loss_first": float(losses[0]) if losses else float("nan"),
            "q_calibration_loss_last": float(losses[-1]) if losses else float("nan"),
            "raw_q_min": float(np.nanmin(q)),
            "raw_q_max": float(np.nanmax(q)),
            "calibrated_q_min": float(np.nanmin(base.predict(q))),
            "calibrated_q_max": float(np.nanmax(base.predict(q))),
        },
    )


def _evaluate_q_calibrator(
    calibrator: QBellmanCalibrator,
    q: np.ndarray,
    next_q_samples: np.ndarray,
    rewards: np.ndarray,
    masks: np.ndarray,
    weights: np.ndarray,
    *,
    gamma: float,
    n_bins: int,
    min_bin_size: int,
    n_folds: int,
) -> dict[str, float]:
    pred = calibrator.predict(q)
    next_cal = calibrator.predict(next_q_samples.reshape(-1)).reshape(next_q_samples.shape)
    outcome = np.asarray(rewards, dtype=float) + float(gamma) * np.asarray(masks, dtype=float) * np.mean(next_cal, axis=1)
    metrics = bellman_metrics(
        pred,
        outcome,
        weights,
        n_bins=int(n_bins),
        min_bin_size=int(min_bin_size),
        n_folds=int(n_folds),
    )
    metrics["q_bellman_mse"] = _weighted_mean_square(pred - outcome, weights)
    metrics["q_bellman_bias"] = _weighted_average(pred - outcome, weights)
    metrics["q_prediction_mean"] = _weighted_average(pred, weights)
    metrics["q_target_mean"] = _weighted_average(outcome, weights)
    return metrics


def _estimate_policy_value(
    result: QFitterResult | Predictor,
    train_dataset: HopperTrajectoryDataset,
    initial_dataset: HopperTrajectoryDataset,
    policy: object,
    calibrator: QBellmanCalibrator,
    *,
    gamma: float,
    seed: int,
    action_samples: int,
    device: str,
) -> float:
    q = _sampled_initial_q_matrix(
        result,
        train_dataset,
        policy,
        gamma=gamma,
        seed=seed,
        action_samples=action_samples,
        device=device,
    )
    calibrated = calibrator.predict(q.reshape(-1)).reshape(q.shape)
    values = np.mean(calibrated, axis=1)
    if initial_dataset.initial_weights.shape[0] == values.shape[0]:
        values = values * initial_dataset.initial_weights / max(float(np.mean(initial_dataset.initial_weights)), 1e-12)
    return _raw_return_from_normalized_return(values, train_dataset, gamma)


def _importance_weights(
    train_dataset: HopperTrajectoryDataset,
    eval_dataset: HopperTrajectoryDataset,
    policy: object,
    *,
    clip: float,
    behavior_ridge: float,
) -> tuple[np.ndarray, dict[str, float]]:
    behavior = fit_behavior_density(train_dataset, ridge=float(behavior_ridge))
    weights, diag = clipped_normalized_density_weights(
        target_policy_log_prob(policy, eval_dataset.observations_raw, eval_dataset.actions),
        behavior.log_prob(eval_dataset.observations_raw, eval_dataset.actions),
        clip=float(clip),
    )
    return weights.astype(float), diag


def _run_policy_seed(
    full_dataset: HopperTrajectoryDataset,
    *,
    policy_id: str,
    seed: int,
    truth: float,
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    rng = np.random.default_rng(seed)
    train_ids, cal_ids, diag_ids = _split_trajectories(
        full_dataset.trajectory_count,
        seed=seed,
        fractions=(float(args.train_fraction), float(args.calibration_fraction), float(args.diagnostic_fraction)),
    )
    train_dataset = _subset_dataset_by_trajectories(full_dataset, train_ids)
    cal_dataset = _subset_dataset_by_trajectories(full_dataset, cal_ids)
    diag_dataset = _subset_dataset_by_trajectories(full_dataset, diag_ids)
    policy = load_policy(policy_id, args.artifact_dir)

    result = _fit_frozen_critic(train_dataset, policy, seed=seed, args=args)
    critic_diag = _critic_diagnostics(result)

    cal_q = _predict_q_return(
        result,
        train_dataset,
        cal_dataset.observations_raw,
        cal_dataset.actions,
        gamma=float(args.gamma),
        device=str(args.device),
    )
    diag_q = _predict_q_return(
        result,
        train_dataset,
        diag_dataset.observations_raw,
        diag_dataset.actions,
        gamma=float(args.gamma),
        device=str(args.device),
    )
    cal_next_q = _sampled_next_q_matrix(
        result,
        train_dataset,
        policy,
        cal_dataset.next_observations_raw,
        gamma=float(args.gamma),
        seed=seed + 10_001,
        action_samples=int(args.action_samples),
        device=str(args.device),
    )
    diag_next_q = _sampled_next_q_matrix(
        result,
        train_dataset,
        policy,
        diag_dataset.next_observations_raw,
        gamma=float(args.gamma),
        seed=seed + 20_001,
        action_samples=int(args.action_samples),
        device=str(args.device),
    )

    if args.weighting == "action_ratio":
        cal_weights, cal_weight_diag = _importance_weights(
            train_dataset,
            cal_dataset,
            policy,
            clip=float(args.importance_weight_clip),
            behavior_ridge=float(args.behavior_ridge),
        )
        diag_weights, diag_weight_diag = _importance_weights(
            train_dataset,
            diag_dataset,
            policy,
            clip=float(args.importance_weight_clip),
            behavior_ridge=float(args.behavior_ridge),
        )
    else:
        cal_weights = np.ones(len(cal_dataset), dtype=float)
        diag_weights = np.ones(len(diag_dataset), dtype=float)
        cal_weight_diag = {
            "importance_weight_ess": float(len(cal_dataset)),
            "importance_weight_ess_fraction": 1.0,
            "importance_weight_max": 1.0,
            "importance_weight_raw_max": 1.0,
            "importance_weight_clip": float("nan"),
        }
        diag_weight_diag = {
            "importance_weight_ess": float(len(diag_dataset)),
            "importance_weight_ess_fraction": 1.0,
            "importance_weight_max": 1.0,
            "importance_weight_raw_max": 1.0,
            "importance_weight_clip": float("nan"),
        }

    rows: list[dict[str, object]] = []
    for method in args.calibrators:
        calibrator = _fit_q_bellman_calibrator(
            str(method),
            cal_q,
            cal_next_q,
            cal_dataset.rewards,
            cal_dataset.masks,
            gamma=float(args.gamma),
            sample_weight=cal_weights,
            n_iterations=int(args.calibration_iterations),
            n_bins=int(args.n_bins),
            min_bin_size=int(args.min_bin_size),
        )
        metrics = _evaluate_q_calibrator(
            calibrator,
            diag_q,
            diag_next_q,
            diag_dataset.rewards,
            diag_dataset.masks,
            diag_weights,
            gamma=float(args.gamma),
            n_bins=int(args.metric_bins),
            min_bin_size=int(args.metric_min_bin_size),
            n_folds=int(args.metric_folds),
        )
        estimate = _estimate_policy_value(
            result,
            train_dataset,
            diag_dataset,
            policy,
            calibrator,
            gamma=float(args.gamma),
            seed=seed + 30_001,
            action_samples=int(args.initial_action_samples),
            device=str(args.device),
        )
        rows.append(
            {
                "dataset_name": args.dataset_name,
                "policy_id": policy_id,
                "seed": int(seed),
                "method": str(method),
                "weighting": str(args.weighting),
                "unit_id": _unit_id(args, seed, policy_id),
                "critic_family": str(args.critic_family),
                "critic_config_id": _critic_config_id(args),
                "calibrator_config_id": _calibrator_config_id(args),
                "ground_truth_return": float(truth),
                "estimated_return": float(estimate),
                "absolute_ope_error": abs(float(estimate) - float(truth)),
                "n_train_transitions": len(train_dataset),
                "n_calibration_transitions": len(cal_dataset),
                "n_diagnostic_transitions": len(diag_dataset),
                "n_train_trajectories": int(train_dataset.trajectory_count),
                "n_calibration_trajectories": int(cal_dataset.trajectory_count),
                "n_diagnostic_trajectories": int(diag_dataset.trajectory_count),
                "fqe_updates": int(args.fqe_updates),
                "critic_lr": float(args.critic_lr),
                "target_tau": float(args.target_tau),
                "batch_size": int(args.batch_size),
                "ridge": float(args.ridge),
                "rf_components": int(args.rf_components),
                "fit_action_samples": int(args.fit_action_samples),
                "action_samples": int(args.action_samples),
                "n_bins": int(args.n_bins),
                "min_bin_size": int(args.min_bin_size),
                "critic_complexity": _critic_complexity(args),
                "calibrator_complexity": _calibrator_complexity(args),
                "calibration_importance_weight_ess_fraction": float(
                    cal_weight_diag.get("importance_weight_ess_fraction", float("nan"))
                ),
                **{f"diagnostic_{k}": v for k, v in diag_weight_diag.items()},
                **critic_diag,
                **metrics,
                **calibrator.diagnostics,
            }
        )
    return rows


def _write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _paired_ratio_ci(
    merged: pd.DataFrame,
    metric: str,
    raw_metric: str,
    *,
    reps: int,
    seed: int,
) -> tuple[float, float]:
    if reps <= 0 or merged.empty:
        return float("nan"), float("nan")
    frame = merged[["seed", "policy_id", metric, raw_metric]].copy()
    frame[metric] = pd.to_numeric(frame[metric], errors="coerce")
    frame[raw_metric] = pd.to_numeric(frame[raw_metric], errors="coerce")
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    frame = frame[frame[raw_metric] > 0]
    if frame.empty:
        return float("nan"), float("nan")
    grouped = frame.groupby(["seed", "policy_id"], as_index=False)[[metric, raw_metric]].mean()
    cal = grouped[metric].to_numpy(dtype=float)
    raw = grouped[raw_metric].to_numpy(dtype=float)
    if cal.size < 2 or float(np.mean(raw)) <= 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    ratios = np.empty(int(reps), dtype=float)
    for rep in range(int(reps)):
        idx = rng.integers(0, cal.size, size=cal.size)
        ratios[rep] = float(np.mean(cal[idx]) / max(float(np.mean(raw[idx])), 1e-12))
    return float(np.quantile(ratios, 0.025)), float(np.quantile(ratios, 0.975))


def _summary(rows: list[dict[str, object]], bootstrap_reps: int = 0, bootstrap_seed: int = 0) -> list[dict[str, object]]:
    df = pd.DataFrame(rows)
    if df.empty:
        return []
    for col, default in [
        ("critic_family", "unknown"),
        ("critic_config_id", "manual"),
        ("calibrator_config_id", "manual"),
        ("critic_complexity", float("nan")),
        ("calibrator_complexity", float("nan")),
    ]:
        if col not in df:
            df[col] = default
    out: list[dict[str, object]] = []
    raw = df[df["method"].astype(str).eq("none")].copy()
    group_cols = ["critic_family", "critic_config_id", "calibrator_config_id", "method", "weighting"]
    for key, group in df.groupby(group_cols, dropna=False):
        critic_family, critic_config_id, calibrator_config_id, method, weighting = key
        merged = group.merge(
            raw[
                [
                    "critic_family",
                    "critic_config_id",
                    "calibrator_config_id",
                    "policy_id",
                    "seed",
                    "weighting",
                    "absolute_ope_error",
                    "q_bellman_mse",
                    "bellman_calibration_error",
                    "bellman_calibration_error_plugin",
                ]
            ],
            on=["critic_family", "critic_config_id", "calibrator_config_id", "policy_id", "seed", "weighting"],
            suffixes=("", "_raw"),
            how="left",
        )
        cal = pd.to_numeric(group["bellman_calibration_error"], errors="coerce")
        raw_cal = pd.to_numeric(merged["bellman_calibration_error_raw"], errors="coerce")
        mse = pd.to_numeric(group["q_bellman_mse"], errors="coerce")
        raw_mse = pd.to_numeric(merged["q_bellman_mse_raw"], errors="coerce")
        ope = pd.to_numeric(group["absolute_ope_error"], errors="coerce")
        raw_ope = pd.to_numeric(merged["absolute_ope_error_raw"], errors="coerce")
        plugin = pd.to_numeric(group["bellman_calibration_error_plugin"], errors="coerce")
        raw_plugin = pd.to_numeric(merged["bellman_calibration_error_plugin_raw"], errors="coerce")
        cal_cmp = cal.to_numpy(dtype=float)
        raw_cal_cmp = raw_cal.to_numpy(dtype=float)
        mse_cmp = mse.to_numpy(dtype=float)
        raw_mse_cmp = raw_mse.to_numpy(dtype=float)
        ope_cmp = ope.to_numpy(dtype=float)
        raw_ope_cmp = raw_ope.to_numpy(dtype=float)
        rel_cal_lo, rel_cal_hi = _paired_ratio_ci(
            merged,
            "bellman_calibration_error",
            "bellman_calibration_error_raw",
            reps=bootstrap_reps,
            seed=bootstrap_seed,
        )
        rel_plugin_lo, rel_plugin_hi = _paired_ratio_ci(
            merged,
            "bellman_calibration_error_plugin",
            "bellman_calibration_error_plugin_raw",
            reps=bootstrap_reps,
            seed=bootstrap_seed + 1,
        )
        rel_mse_lo, rel_mse_hi = _paired_ratio_ci(
            merged,
            "q_bellman_mse",
            "q_bellman_mse_raw",
            reps=bootstrap_reps,
            seed=bootstrap_seed + 2,
        )
        rel_ope_lo, rel_ope_hi = _paired_ratio_ci(
            merged,
            "absolute_ope_error",
            "absolute_ope_error_raw",
            reps=bootstrap_reps,
            seed=bootstrap_seed + 3,
        )
        out.append(
            {
                "critic_family": critic_family,
                "critic_config_id": critic_config_id,
                "calibrator_config_id": calibrator_config_id,
                "method": method,
                "weighting": weighting,
                "n_rows": int(len(group)),
                "n_policies": int(group["policy_id"].nunique()),
                "n_seeds": int(group["seed"].nunique()),
                "mean_bellman_calibration_error": float(cal.mean()),
                "mean_q_bellman_mse": float(mse.mean()),
                "mean_bellman_calibration_error_plugin": float(plugin.mean()),
                "mean_absolute_ope_error": float(ope.mean()),
                "relative_bellman_calibration_error": float(cal.mean() / raw_cal.mean()) if raw_cal.mean() > 0 else float("nan"),
                "relative_q_bellman_mse": float(mse.mean() / raw_mse.mean()) if raw_mse.mean() > 0 else float("nan"),
                "relative_bellman_calibration_error_plugin": float(plugin.mean() / raw_plugin.mean())
                if raw_plugin.mean() > 0
                else float("nan"),
                "relative_absolute_ope_error": float(ope.mean() / raw_ope.mean()) if raw_ope.mean() > 0 else float("nan"),
                "relative_bellman_calibration_error_ci_low": rel_cal_lo,
                "relative_bellman_calibration_error_ci_high": rel_cal_hi,
                "relative_bellman_calibration_error_plugin_ci_low": rel_plugin_lo,
                "relative_bellman_calibration_error_plugin_ci_high": rel_plugin_hi,
                "relative_q_bellman_mse_ci_low": rel_mse_lo,
                "relative_q_bellman_mse_ci_high": rel_mse_hi,
                "relative_absolute_ope_error_ci_low": rel_ope_lo,
                "relative_absolute_ope_error_ci_high": rel_ope_hi,
                "bellman_calibration_win_rate": float(np.nanmean(cal_cmp < raw_cal_cmp)) if str(method) != "none" else float("nan"),
                "q_bellman_mse_win_rate": float(np.nanmean(mse_cmp < raw_mse_cmp)) if str(method) != "none" else float("nan"),
                "absolute_ope_win_rate": float(np.nanmean(ope_cmp < raw_ope_cmp)) if str(method) != "none" else float("nan"),
                "mean_diag_ess_fraction": float(pd.to_numeric(group["diagnostic_importance_weight_ess_fraction"], errors="coerce").mean()),
                "mean_diag_max_weight": float(pd.to_numeric(group["diagnostic_importance_weight_max"], errors="coerce").mean()),
                "mean_fqe_loss_first": float(pd.to_numeric(group["fqe_loss_first"], errors="coerce").mean()),
                "mean_fqe_loss_last": float(pd.to_numeric(group["fqe_loss_last"], errors="coerce").mean()),
                "critic_complexity": float(pd.to_numeric(group["critic_complexity"], errors="coerce").mean()),
                "calibrator_complexity": float(pd.to_numeric(group["calibrator_complexity"], errors="coerce").mean()),
                "fqe_updates": float(pd.to_numeric(group["fqe_updates"], errors="coerce").median()),
                "critic_lr": float(pd.to_numeric(group.get("critic_lr", pd.Series(dtype=float)), errors="coerce").median()),
                "target_tau": float(pd.to_numeric(group.get("target_tau", pd.Series(dtype=float)), errors="coerce").median()),
                "ridge": float(pd.to_numeric(group.get("ridge", pd.Series(dtype=float)), errors="coerce").median()),
                "rf_components": float(pd.to_numeric(group.get("rf_components", pd.Series(dtype=float)), errors="coerce").median()),
                "fit_action_samples": float(pd.to_numeric(group.get("fit_action_samples", pd.Series(dtype=float)), errors="coerce").median()),
                "n_bins": float(pd.to_numeric(group.get("n_bins", pd.Series(dtype=float)), errors="coerce").median()),
                "min_bin_size": float(pd.to_numeric(group.get("min_bin_size", pd.Series(dtype=float)), errors="coerce").median()),
                "calibration_iterations": float(pd.to_numeric(group.get("q_calibration_iterations", pd.Series(dtype=float)), errors="coerce").max()),
            }
        )
    return out


def _audit(summary_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    for row in summary_rows:
        reasons: list[str] = []
        if row["method"] == "none":
            label = "raw_baseline"
        else:
            if not (float(row["relative_bellman_calibration_error"]) < 0.90 or float(row["relative_bellman_calibration_error_plugin"]) < 0.90):
                reasons.append("calibration_error_not_improved")
            if not (float(row["relative_q_bellman_mse"]) < 1.05):
                reasons.append("bellman_mse_regression")
            if not (float(row["bellman_calibration_win_rate"]) >= 0.50 or float(row["q_bellman_mse_win_rate"]) >= 0.50):
                reasons.append("low_transition_metric_win_rate")
            if float(row["mean_diag_ess_fraction"]) < 0.05:
                reasons.append("low_weight_ess")
            label = "promising_q_calibration" if not reasons else "mixed_or_negative"
        rows.append({**row, "audit_label": label, "failure_reasons": ";".join(reasons)})
    return rows


def _plot(summary_rows: list[dict[str, object]], output_dir: Path) -> None:
    df = pd.DataFrame(summary_rows)
    if df.empty:
        return
    df = df[df["method"].astype(str).ne("none")].copy()
    if df.empty:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 2.4), constrained_layout=True)
    labels = df["critic_family"].astype(str) + " / " + df["method"].astype(str)
    panels = [
        ("relative_bellman_calibration_error", "Debiased CAL"),
        ("relative_q_bellman_mse", "Bellman MSE"),
        ("relative_absolute_ope_error", "OPE abs. err."),
    ]
    for ax, (col, title) in zip(axes, panels):
        ax.barh(labels, pd.to_numeric(df[col], errors="coerce"), color="#4C78A8")
        ax.axvline(1.0, color="0.25", linewidth=0.8, linestyle=(0, (3, 2)))
        ax.set_title(title, loc="left", fontsize=8, fontweight="bold")
        ax.tick_params(axis="both", labelsize=7)
        ax.grid(axis="x", color="#D9D9D9", linewidth=0.5)
        ax.set_axisbelow(True)
    fig.savefig(output_dir / "hopper_q_calibration_summary.png", dpi=220, bbox_inches="tight")
    fig.savefig(output_dir / "hopper_q_calibration_summary.pdf", bbox_inches="tight")
    plt.close(fig)


def _critic_grid_specs(args: argparse.Namespace) -> list[argparse.Namespace]:
    specs: list[argparse.Namespace] = []
    ridges = _parse_floats(args.tune_ridges)
    fit_samples = _parse_ints(args.tune_fit_action_samples)
    rf_components = _parse_ints(args.tune_rf_components)
    neural_updates = _parse_ints(args.tune_neural_updates)
    neural_lrs = _parse_floats(args.tune_neural_lrs)
    neural_taus = _parse_floats(args.tune_neural_taus)

    for family in args.critic_families:
        if family == "linear_fqe":
            for ridge, fit_action_samples in product(ridges, fit_samples):
                spec = _replace_args(
                    args,
                    critic_family=family,
                    ridge=ridge,
                    fit_action_samples=fit_action_samples,
                    rf_components=0,
                    fqe_updates=1,
                    linear_solver="fixed_point",
                )
                spec.critic_config_id = _critic_config_id(spec)
                specs.append(spec)
        elif family == "rf_fqe":
            for components, ridge, fit_action_samples in product(rf_components, ridges, fit_samples):
                spec = _replace_args(
                    args,
                    critic_family=family,
                    rf_components=components,
                    ridge=ridge,
                    fit_action_samples=fit_action_samples,
                    fqe_updates=1,
                    linear_solver="fixed_point",
                )
                spec.critic_config_id = _critic_config_id(spec)
                specs.append(spec)
        elif family == "neural_fqe":
            for updates, lr, tau in product(neural_updates, neural_lrs, neural_taus):
                spec = _replace_args(
                    args,
                    critic_family=family,
                    fqe_updates=updates,
                    critic_lr=lr,
                    target_tau=tau,
                    batch_size=256,
                )
                spec.critic_config_id = _critic_config_id(spec)
                specs.append(spec)
        else:
            raise ValueError(f"Unknown critic family '{family}'.")
    return specs


def _calibrator_grid_specs(args: argparse.Namespace) -> list[dict[str, object]]:
    specs: list[dict[str, object]] = []
    for n_bins, min_bin_size, n_iter in product(
        _parse_ints(args.tune_calibrator_bins),
        _parse_ints(args.tune_calibrator_min_bin_sizes),
        _parse_ints(args.tune_calibration_iterations),
    ):
        calib_args = _replace_args(args, n_bins=n_bins, min_bin_size=min_bin_size, calibration_iterations=n_iter)
        specs.append(
            {
                "n_bins": n_bins,
                "min_bin_size": min_bin_size,
                "calibration_iterations": n_iter,
                "calibrator_config_id": _calibrator_config_id(calib_args),
            }
        )
    return specs


def _manual_specs(args: argparse.Namespace) -> list[argparse.Namespace]:
    specs = []
    for family in args.critic_families:
        spec = _replace_args(args, critic_family=family)
        spec.critic_config_id = _critic_config_id(spec)
        spec.calibrator_config_id = _calibrator_config_id(spec)
        specs.append(spec)
    return specs


def _load_selected_configs(path: str | Path | None) -> dict[str, dict[str, object]]:
    if not path:
        return {}
    config_path = Path(path)
    if not config_path.exists():
        return {}
    payload = json.loads(config_path.read_text())
    selected = payload.get("selected", payload)
    if not isinstance(selected, dict):
        return {}
    return {str(k): dict(v) for k, v in selected.items()}


def _selected_specs(args: argparse.Namespace) -> list[argparse.Namespace]:
    selected = _load_selected_configs(args.tuned_config_path)
    if not selected:
        if args.stage == "final" and not args.allow_default_final_config:
            raise FileNotFoundError(
                "Final stage requires --tuned_config_path with selected configs, "
                "or pass --allow_default_final_config to use CLI/default hyperparameters."
            )
        return _manual_specs(args)

    specs: list[argparse.Namespace] = []
    for family in args.critic_families:
        if family not in selected:
            if args.stage == "final" and not args.allow_default_final_config:
                raise KeyError(f"Tuned config file lacks selected config for critic family '{family}'.")
            specs.extend(_manual_specs(_replace_args(args, critic_families=[family])))
            continue
        config = selected[family]
        updates = {
            key: value
            for key, value in config.items()
            if key
            in {
                "critic_family",
                "critic_config_id",
                "calibrator_config_id",
                "fqe_updates",
                "critic_lr",
                "target_tau",
                "batch_size",
                "ridge",
                "rf_components",
                "fit_action_samples",
                "linear_solver",
                "n_bins",
                "min_bin_size",
                "calibration_iterations",
            }
        }
        updates["critic_family"] = family
        spec = _replace_args(args, **updates)
        spec.critic_config_id = str(config.get("critic_config_id", _critic_config_id(spec)))
        spec.calibrator_config_id = str(config.get("calibrator_config_id", _calibrator_config_id(spec)))
        specs.append(spec)
    return specs


def _experiment_specs(args: argparse.Namespace) -> list[argparse.Namespace]:
    if args.stage in {"tune", "tuning"}:
        if args.tuning_mode == "calibrator":
            base_specs = _selected_specs(args)
        else:
            base_specs = _critic_grid_specs(args)

        if args.tuning_mode == "base":
            calibrator_specs = [
                {
                    "n_bins": args.n_bins,
                    "min_bin_size": args.min_bin_size,
                    "calibration_iterations": args.calibration_iterations,
                    "calibrator_config_id": _calibrator_config_id(args),
                }
            ]
        else:
            calibrator_specs = _calibrator_grid_specs(args)

        out: list[argparse.Namespace] = []
        for base_spec, calibrator_spec in product(base_specs, calibrator_specs):
            spec = _replace_args(base_spec, **calibrator_spec)
            spec.calibrator_config_id = str(calibrator_spec["calibrator_config_id"])
            out.append(spec)
        return out
    return _selected_specs(args)


def _read_csv_rows(paths: list[Path]) -> list[dict[str, object]]:
    frames = []
    for path in paths:
        if path.exists():
            frames.append(pd.read_csv(path))
    if not frames:
        return []
    return pd.concat(frames, ignore_index=True, sort=False).to_dict("records")


def _finite_float(row: dict[str, object], key: str, default: float) -> float:
    try:
        value = float(row.get(key, default))
    except (TypeError, ValueError):
        return float(default)
    return value if np.isfinite(value) else float(default)


def _finite_int(row: dict[str, object], key: str, default: int) -> int:
    return int(round(_finite_float(row, key, float(default))))


def _rank_by_columns(frame: pd.DataFrame, rank_cols: list[str]) -> pd.Series:
    rank_parts = []
    for col in rank_cols:
        values = pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        fill = float(values.max()) if values.notna().any() else 1e12
        rank_parts.append(values.fillna(fill + 1.0).rank(method="average"))
    return pd.concat(rank_parts, axis=1).mean(axis=1)


def _selected_config_from_summary_row(
    row: dict[str, object],
    *,
    family: str,
    selection_mode: str,
    rank_cols: list[str],
    source_tuning_mode: str,
) -> dict[str, object]:
    return {
        "critic_family": str(family),
        "critic_config_id": str(row["critic_config_id"]),
        "calibrator_config_id": str(row["calibrator_config_id"]),
        "selected_by_method": str(row["method"]),
        "selection_mode": selection_mode,
        "selection_metrics": rank_cols,
        "source_tuning_mode": source_tuning_mode,
        "mean_validation_rank": float(row["mean_validation_rank"]),
        "fqe_updates": _finite_int(row, "fqe_updates", 1),
        "critic_lr": _finite_float(row, "critic_lr", 3e-4),
        "target_tau": _finite_float(row, "target_tau", 0.005),
        "batch_size": 256,
        "ridge": _finite_float(row, "ridge", 1e-3),
        "rf_components": _finite_int(row, "rf_components", 0),
        "fit_action_samples": _finite_int(row, "fit_action_samples", 4),
        "linear_solver": "fixed_point",
        "n_bins": _finite_int(row, "n_bins", 20),
        "min_bin_size": _finite_int(row, "min_bin_size", 100),
        "calibration_iterations": _finite_int(row, "calibration_iterations", 4),
    }


def _select_tuned_configs(summary_rows: list[dict[str, object]], tuning_mode: str = "calibrator") -> dict[str, object]:
    df = pd.DataFrame(summary_rows)
    if df.empty:
        return {
            "selected": {},
            "selection_rows": [],
            "selection_mode": tuning_mode,
            "selection_metrics": [],
            "source_tuning_mode": tuning_mode,
        }

    if tuning_mode == "base":
        candidates = df[df["method"].astype(str).eq("none")].copy()
        rank_cols = [
            "mean_q_bellman_mse",
            "mean_bellman_calibration_error",
            "mean_bellman_calibration_error_plugin",
        ]
        selection_mode = "raw_critic_heldout_bellman"
    else:
        candidates = df[df["method"].astype(str).ne("none")].copy()
        rank_cols = [
            "mean_bellman_calibration_error_plugin",
            "mean_bellman_calibration_error",
            "mean_q_bellman_mse",
            "relative_bellman_calibration_error_plugin",
            "relative_bellman_calibration_error",
            "relative_q_bellman_mse",
        ]
        selection_mode = "posthoc_calibrator_heldout_bellman"

    if candidates.empty:
        return {
            "selected": {},
            "selection_rows": [],
            "selection_mode": selection_mode,
            "selection_metrics": rank_cols,
            "source_tuning_mode": tuning_mode,
        }

    selected: dict[str, dict[str, object]] = {}
    selection_rows: list[dict[str, object]] = []
    for family, group in candidates.groupby("critic_family", dropna=False):
        scored = group.copy()
        scored["mean_validation_rank"] = _rank_by_columns(scored, rank_cols)
        scored = scored.sort_values(
            ["mean_validation_rank", "critic_complexity", "calibrator_complexity", "critic_config_id", "calibrator_config_id"],
            kind="mergesort",
        )
        row = scored.iloc[0].to_dict()
        selected[str(family)] = _selected_config_from_summary_row(
            row,
            family=str(family),
            selection_mode=selection_mode,
            rank_cols=rank_cols,
            source_tuning_mode=tuning_mode,
        )
        selection_rows.append(row)
    return {
        "selected": selected,
        "selection_rows": selection_rows,
        "selection_mode": selection_mode,
        "selection_metrics": rank_cols,
        "source_tuning_mode": tuning_mode,
    }


def run(args: argparse.Namespace) -> dict[str, Path]:
    if int(args.torch_threads) > 0:
        torch.set_num_threads(int(args.torch_threads))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    unit_dir = output_dir / "units"
    unit_dir.mkdir(parents=True, exist_ok=True)
    full_dataset = load_hopper_dataset(
        args.data_dir,
        args.dataset_name,
        max_trajectories=args.max_trajectories,
        max_transitions=args.max_transitions,
        seed=args.dataset_seed,
    )
    gt = _load_ground_truth(Path(args.benchmark_dir))
    specs = list(HOPPER_MEDIUM_POLICY_SPECS)
    policy_ids = [specs[int(i)].policy_id for i in args.policy_indices]
    experiment_specs = _experiment_specs(args)
    expected_unit_paths: list[Path] = []
    completed_now = 0
    skipped_existing = 0
    failed_units: list[dict[str, object]] = []

    for spec in experiment_specs:
        for seed in args.seeds:
            for policy_id in policy_ids:
                unit_path = _unit_path(output_dir, spec, int(seed), policy_id)
                expected_unit_paths.append(unit_path)
                if unit_path.exists() and args.resume_units and not args.overwrite_units:
                    skipped_existing += 1
                    continue
                print(
                    "[hopper-q-cal] "
                    f"critic={spec.critic_family} critic_cfg={_critic_config_id(spec)} "
                    f"cal_cfg={_calibrator_config_id(spec)} seed={seed} policy={policy_id}",
                    flush=True,
                )
                try:
                    unit_rows = _run_policy_seed(
                        full_dataset,
                        policy_id=policy_id,
                        seed=int(seed),
                        truth=gt.get(policy_id, float("nan")),
                        args=spec,
                    )
                    _write_csv(unit_rows, unit_path)
                    completed_now += 1
                except Exception as exc:
                    failed_units.append(
                        {
                            "critic_family": spec.critic_family,
                            "critic_config_id": _critic_config_id(spec),
                            "calibrator_config_id": _calibrator_config_id(spec),
                            "seed": int(seed),
                            "policy_id": policy_id,
                            "error": repr(exc),
                        }
                    )
                    if not args.continue_on_error:
                        raise

    rows = _read_csv_rows(expected_unit_paths)
    summary_rows = _summary(rows, bootstrap_reps=int(args.bootstrap_reps), bootstrap_seed=int(args.bootstrap_seed))
    audit_rows = _audit(summary_rows)
    result_path = output_dir / "hopper_q_calibration_results.csv"
    summary_path = output_dir / "hopper_q_calibration_summary.csv"
    audit_path = output_dir / "hopper_q_calibration_audit.csv"
    config_path = output_dir / "hopper_q_calibration_config.json"
    manifest_path = output_dir / "hopper_q_calibration_manifest.json"
    _write_csv(rows, result_path)
    _write_csv(summary_rows, summary_path)
    _write_csv(audit_rows, audit_path)
    config_path.write_text(json.dumps(vars(args), indent=2, default=str))
    manifest = {
        "n_expected_units": len(expected_unit_paths),
        "n_completed_unit_files": sum(1 for path in expected_unit_paths if path.exists()),
        "n_completed_now": completed_now,
        "n_skipped_existing": skipped_existing,
        "n_failed_units": len(failed_units),
        "failed_units": failed_units,
        "n_result_rows": len(rows),
        "n_experiment_specs": len(experiment_specs),
        "critic_families": args.critic_families,
        "policy_indices": args.policy_indices,
        "seeds": args.seeds,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
    _plot(summary_rows, output_dir)
    outputs = {
        "results": result_path,
        "summary": summary_path,
        "audit": audit_path,
        "config": config_path,
        "manifest": manifest_path,
    }
    if args.stage in {"tune", "tuning"}:
        tuned_path = output_dir / "tuned_configs.json"
        tuned = _select_tuned_configs(summary_rows, tuning_mode=str(args.tuning_mode))
        tuned_path.write_text(json.dumps(tuned, indent=2, default=str))
        outputs["tuned_configs"] = tuned_path
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run frozen-Q Bellman calibration on the Hopper Deep OPE benchmark.")
    parser.add_argument("--data_dir", default="hopper_fqe_benchmark/artifacts")
    parser.add_argument("--artifact_dir", default="hopper_fqe_benchmark/artifacts")
    parser.add_argument("--benchmark_dir", default="hopper_fqe_benchmark/artifacts/benchmark/dope")
    parser.add_argument("--dataset_name", default="hopper-medium-v0")
    parser.add_argument("--output_dir", default="FQE_calibration_neurips/results/hopper_q_calibration")
    parser.add_argument("--stage", choices=["smoke", "pilot", "tune", "tuning", "final", "paper", "custom"], default="pilot")
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--seed_start", type=int, default=None)
    parser.add_argument("--seed_stop", type=int, default=None)
    parser.add_argument("--policy_indices", nargs="+", type=int, default=None)
    parser.add_argument("--max_trajectories", type=int, default=None)
    parser.add_argument("--max_transitions", type=int, default=None)
    parser.add_argument("--dataset_seed", type=int, default=0)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--critic_family", choices=["linear_fqe", "rf_fqe", "neural_fqe"], default="rf_fqe")
    parser.add_argument("--critic_families", nargs="+", choices=["linear_fqe", "rf_fqe", "neural_fqe"], default=None)
    parser.add_argument("--critic_config_id", default=None)
    parser.add_argument("--calibrator_config_id", default=None)
    parser.add_argument("--fqe_updates", type=int, default=3000)
    parser.add_argument("--critic_lr", type=float, default=3e-4)
    parser.add_argument("--rf_components", type=int, default=128)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--linear_solver", choices=["iterated", "fixed_point"], default="fixed_point")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--target_tau", type=float, default=0.005)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch_threads", type=int, default=1)
    parser.add_argument("--calibrators", nargs="+", default=DEFAULT_CALIBRATORS)
    parser.add_argument("--calibration_iterations", type=int, default=4)
    parser.add_argument("--n_bins", type=int, default=20)
    parser.add_argument("--min_bin_size", type=int, default=100)
    parser.add_argument("--metric_bins", type=int, default=30)
    parser.add_argument("--metric_min_bin_size", type=int, default=300)
    parser.add_argument("--metric_folds", type=int, default=5)
    parser.add_argument("--action_samples", type=int, default=4)
    parser.add_argument("--fit_action_samples", type=int, default=4)
    parser.add_argument("--initial_action_samples", type=int, default=16)
    parser.add_argument("--weighting", choices=["none", "action_ratio"], default="none")
    parser.add_argument("--importance_weight_clip", type=float, default=10.0)
    parser.add_argument("--behavior_ridge", type=float, default=1e-3)
    parser.add_argument("--train_fraction", type=float, default=0.60)
    parser.add_argument("--calibration_fraction", type=float, default=0.20)
    parser.add_argument("--diagnostic_fraction", type=float, default=0.20)
    parser.add_argument("--resume_units", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite_units", action="store_true")
    parser.add_argument("--continue_on_error", action="store_true")
    parser.add_argument("--bootstrap_reps", type=int, default=1000)
    parser.add_argument("--bootstrap_seed", type=int, default=9713)
    parser.add_argument("--tuned_config_path", default=None)
    parser.add_argument("--allow_default_final_config", action="store_true")
    parser.add_argument("--tuning_mode", choices=["base", "calibrator", "full"], default="base")
    parser.add_argument("--tune_ridges", default="1e-4 1e-3 1e-2")
    parser.add_argument("--tune_fit_action_samples", default="4 8")
    parser.add_argument("--tune_rf_components", default="128 256 512")
    parser.add_argument("--tune_neural_updates", default="5000 20000 50000")
    parser.add_argument("--tune_neural_lrs", default="1e-4 3e-4")
    parser.add_argument("--tune_neural_taus", default="0.005 0.02")
    parser.add_argument("--tune_calibrator_bins", default="10 20 40")
    parser.add_argument("--tune_calibrator_min_bin_sizes", default="100 300")
    parser.add_argument("--tune_calibration_iterations", default="1 4")
    args = parser.parse_args()

    if args.stage == "smoke":
        if args.critic_families is None:
            args.critic_families = DEFAULT_CRITIC_FAMILIES.copy()
        if args.seeds is None and args.seed_start is None:
            args.seeds = [0]
        if args.policy_indices is None:
            args.policy_indices = [0, 1]
        args.max_trajectories = args.max_trajectories or 96
        args.fqe_updates = min(int(args.fqe_updates), 25)
        args.action_samples = min(int(args.action_samples), 2)
        args.fit_action_samples = min(int(args.fit_action_samples), 2)
        args.initial_action_samples = min(int(args.initial_action_samples), 4)
        args.bootstrap_reps = min(int(args.bootstrap_reps), 200)
        if args.output_dir == "FQE_calibration_neurips/results/hopper_q_calibration":
            args.output_dir = "FQE_calibration_neurips/results/hopper_q_calibration_smoke"
    elif args.stage == "pilot":
        if args.critic_families is None:
            args.critic_families = ["linear_fqe", "rf_fqe"]
        if args.seeds is None and args.seed_start is None:
            args.seeds = [0, 1]
        if args.policy_indices is None:
            args.policy_indices = [0, 1, 2, 3]
        args.max_trajectories = args.max_trajectories or 192
        args.fqe_updates = min(int(args.fqe_updates), 50)
        if args.output_dir == "FQE_calibration_neurips/results/hopper_q_calibration":
            args.output_dir = "FQE_calibration_neurips/results/hopper_q_calibration_pilot"
    elif args.stage in {"tune", "tuning"}:
        if args.critic_families is None:
            args.critic_families = DEFAULT_CRITIC_FAMILIES.copy()
        if args.seeds is None and args.seed_start is None:
            args.seeds = list(range(1000, 1010))
        if args.policy_indices is None:
            args.policy_indices = [0, 3, 6, 10]
        if args.output_dir == "FQE_calibration_neurips/results/hopper_q_calibration":
            args.output_dir = "FQE_calibration_neurips/results/hopper_q_calibration_tuning"
    elif args.stage in {"final", "paper"}:
        if args.critic_families is None:
            args.critic_families = DEFAULT_CRITIC_FAMILIES.copy()
        if args.seeds is None and args.seed_start is None:
            args.seed_start = 0
            args.seed_stop = 100
        if args.policy_indices is None:
            args.policy_indices = list(range(11))
        args.max_trajectories = args.max_trajectories or None
        args.weighting = "none"
        if args.output_dir == "FQE_calibration_neurips/results/hopper_q_calibration":
            args.output_dir = "FQE_calibration_neurips/results/hopper_q_calibration_final"
    else:
        if args.critic_families is None:
            args.critic_families = [args.critic_family]
        if args.seeds is None and args.seed_start is None:
            args.seeds = [0]
        if args.policy_indices is None:
            args.policy_indices = [0]
    if args.critic_families is None:
        args.critic_families = [args.critic_family]
    if args.seeds is None:
        if args.seed_start is None or args.seed_stop is None:
            raise ValueError("Provide --seeds or both --seed_start and --seed_stop.")
        args.seeds = list(range(int(args.seed_start), int(args.seed_stop)))
    args.critic_family = args.critic_families[0]
    return args


def main() -> None:
    for name, path in run(parse_args()).items():
        print(f"Wrote {name}: {path}", flush=True)


if __name__ == "__main__":
    main()
