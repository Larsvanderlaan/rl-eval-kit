#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from FQE_calibration_neurips.src.calibration.calibrators import BaseCalibrator, IdentityCalibrator, fit_calibrator  # noqa: E402
from hopper_fqe_benchmark.data import HOPPER_DATASET_SPECS, HopperTrajectoryDataset, load_hopper_dataset  # noqa: E402
from hopper_fqe_benchmark.fqe import QFitterConfig, QFitterResult, train_q_fitter  # noqa: E402
from hopper_fqe_benchmark.policies import HOPPER_MEDIUM_POLICY_SPECS, POLICY_SPECS, load_policy  # noqa: E402


@dataclass
class HopperValueCalibrator:
    base: BaseCalibrator
    method: str
    n_iterations: int
    diagnostics: dict[str, float | str]

    def predict(self, values: np.ndarray) -> np.ndarray:
        return self.base.predict(values)


@dataclass
class BehaviorGaussianDensity:
    model: Ridge
    std: np.ndarray
    tanh_actions: bool = True

    def mean(self, observations: np.ndarray) -> np.ndarray:
        return self.model.predict(np.asarray(observations, dtype=np.float64))

    def log_prob(self, observations: np.ndarray, actions: np.ndarray) -> np.ndarray:
        z = _atanh_clipped(actions) if self.tanh_actions else np.asarray(actions, dtype=np.float64)
        mean = self.mean(observations)
        return _diag_normal_log_prob(z, mean, self.std) - _tanh_log_det(actions) if self.tanh_actions else _diag_normal_log_prob(z, mean, self.std)


def _trajectory_ids_from_steps(steps: np.ndarray) -> np.ndarray:
    starts = np.asarray(steps).reshape(-1) == 0
    return np.cumsum(starts).astype(int) - 1


def _trajectory_folds(n_trajectories: int, n_folds: int, seed: int) -> list[np.ndarray]:
    ids = np.arange(int(n_trajectories))
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    return [fold.astype(int) for fold in np.array_split(ids, max(2, int(n_folds)))]


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


def _atanh_clipped(actions: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    a = np.clip(np.asarray(actions, dtype=np.float64), -1.0 + eps, 1.0 - eps)
    return 0.5 * (np.log1p(a) - np.log1p(-a))


def _tanh_log_det(actions: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    a = np.clip(np.asarray(actions, dtype=np.float64), -1.0 + eps, 1.0 - eps)
    return np.sum(np.log(np.maximum(1.0 - a * a, eps)), axis=1)


def _diag_normal_log_prob(z: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    mean = np.asarray(mean, dtype=np.float64)
    std = np.maximum(np.asarray(std, dtype=np.float64), 1e-4)
    return -0.5 * np.sum(((z - mean) / std) ** 2 + 2.0 * np.log(std) + math.log(2.0 * math.pi), axis=1)


def target_policy_log_prob(policy: object, observations_raw: np.ndarray, actions: np.ndarray) -> np.ndarray:
    mean, log_std = policy._forward(np.asarray(observations_raw, dtype=np.float32))  # noqa: SLF001
    log_std = np.clip(np.asarray(log_std, dtype=np.float64), -5.0, 2.0)
    z = _atanh_clipped(actions) if getattr(policy, "output_distribution", "") == "tanh_gaussian" else np.asarray(actions, dtype=np.float64)
    base = _diag_normal_log_prob(z, np.asarray(mean, dtype=np.float64), np.exp(log_std))
    if getattr(policy, "output_distribution", "") == "tanh_gaussian":
        base = base - _tanh_log_det(actions)
    return base


def fit_behavior_density(dataset: HopperTrajectoryDataset, ridge: float = 1e-3) -> BehaviorGaussianDensity:
    z_actions = _atanh_clipped(dataset.actions)
    model = Ridge(alpha=float(ridge), fit_intercept=True).fit(dataset.observations_raw, z_actions)
    resid = z_actions - model.predict(dataset.observations_raw)
    std = np.maximum(np.std(resid, axis=0), 0.05)
    return BehaviorGaussianDensity(model=model, std=std)


def clipped_normalized_density_weights(
    target_log_prob: np.ndarray,
    behavior_log_prob: np.ndarray,
    *,
    clip: float,
    normalize: bool = True,
) -> tuple[np.ndarray, dict[str, float]]:
    raw = np.exp(np.clip(np.asarray(target_log_prob) - np.asarray(behavior_log_prob), -50.0, 50.0))
    clipped = np.minimum(raw, float(clip))
    weights = clipped / max(float(np.mean(clipped)), 1e-12) if normalize else clipped
    ess = float((np.sum(weights) ** 2) / max(np.sum(weights**2), 1e-12))
    return weights.astype(float), {
        "importance_weight_ess": ess,
        "importance_weight_ess_fraction": float(ess / max(weights.size, 1)),
        "importance_weight_max": float(np.max(weights)) if weights.size else float("nan"),
        "importance_weight_raw_max": float(np.max(raw)) if raw.size else float("nan"),
        "importance_weight_clip": float(clip),
    }


def predict_value_normalized_return(
    result: QFitterResult,
    dataset: HopperTrajectoryDataset,
    policy: object,
    observations_raw: np.ndarray,
    *,
    gamma: float,
    seed: int,
    n_action_samples: int,
    device: str,
    chunk_size: int = 4096,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    states_raw = np.asarray(observations_raw, dtype=np.float32)
    out = np.zeros(states_raw.shape[0], dtype=np.float64)
    target_model = result.target_model.to(device)
    target_model.eval()
    for start in range(0, states_raw.shape[0], chunk_size):
        stop = min(start + chunk_size, states_raw.shape[0])
        chunk = states_raw[start:stop]
        repeated_states = np.repeat(chunk, int(n_action_samples), axis=0)
        actions = policy.sample_actions(repeated_states, rng=rng, deterministic=False)
        norm_states = dataset.normalize_states(repeated_states)
        with torch.no_grad():
            q_scaled = target_model(
                torch.as_tensor(norm_states, dtype=torch.float32, device=device),
                torch.as_tensor(actions, dtype=torch.float32, device=device),
            ).cpu().numpy()
        q_return = q_scaled.reshape(chunk.shape[0], int(n_action_samples)) / max(1.0 - float(gamma), 1e-8)
        out[start:stop] = np.mean(q_return, axis=1)
    return out


def _raw_return_from_normalized_return(values: np.ndarray, dataset: HopperTrajectoryDataset, gamma: float) -> float:
    return float(dataset.reward_std * np.mean(values) + dataset.reward_mean / max(1.0 - float(gamma), 1e-8))


def fit_hopper_value_calibrator(
    method: str,
    values: np.ndarray,
    next_values: np.ndarray,
    rewards: np.ndarray,
    masks: np.ndarray,
    weights: np.ndarray,
    gamma: float,
    *,
    n_iterations: int,
    n_bins: int,
    min_bin_size: int,
) -> HopperValueCalibrator:
    current: BaseCalibrator = IdentityCalibrator()
    losses: list[float] = []
    for _ in range(max(1, int(n_iterations))):
        target = np.asarray(rewards, dtype=float) + float(gamma) * np.asarray(masks, dtype=float) * current.predict(next_values)
        current = fit_calibrator(
            method,
            values,
            target,
            n_bins=int(n_bins),
            bin_strategy="quantile",
            min_bin_size=int(min_bin_size),
            sample_weight=weights,
        )
        pred = current.predict(values)
        losses.append(float(np.average((pred - target) ** 2, weights=weights)))
    return HopperValueCalibrator(
        base=current,
        method=method,
        n_iterations=max(1, int(n_iterations)),
        diagnostics={
            "value_calibration_loss_first": losses[0],
            "value_calibration_loss_last": losses[-1],
            "value_calibration_iterations": float(max(1, int(n_iterations))),
        },
    )


def _weighted_mean_square(x: np.ndarray, weights: np.ndarray) -> float:
    return float(np.average(np.asarray(x, dtype=float) ** 2, weights=np.asarray(weights, dtype=float)))


def _safe_spearman(x: Iterable[float], y: Iterable[float]) -> float:
    import pandas as pd

    frame = pd.DataFrame({"x": list(x), "y": list(y)}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(frame) < 2 or frame["x"].nunique() < 2 or frame["y"].nunique() < 2:
        return float("nan")
    return float(frame["x"].rank().corr(frame["y"].rank()))


def _safe_pearson(x: Iterable[float], y: Iterable[float]) -> float:
    import pandas as pd

    frame = pd.DataFrame({"x": list(x), "y": list(y)}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(frame) < 2 or frame["x"].nunique() < 2 or frame["y"].nunique() < 2:
        return float("nan")
    return float(frame["x"].corr(frame["y"]))


def bellman_metrics(
    pred: np.ndarray,
    outcome: np.ndarray,
    weights: np.ndarray,
    *,
    n_bins: int = 50,
    min_bin_size: int = 1000,
    n_folds: int = 5,
) -> dict[str, float]:
    pred = np.asarray(pred, dtype=float)
    outcome = np.asarray(outcome, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = np.isfinite(pred) & np.isfinite(outcome) & np.isfinite(weights) & (weights >= 0)
    pred, outcome, weights = pred[mask], outcome[mask], weights[mask]
    if pred.size < 2:
        return {
            "bellman_brier_score": float("nan"),
            "bellman_calibration_error": float("nan"),
            "bellman_calibration_error_plugin": float("nan"),
            "bellman_calibration_error_debiased_raw": float("nan"),
            "bellman_calibration_bins_used": 0,
            "bellman_calibration_mean_bin_size": float("nan"),
        }
    brier = _weighted_mean_square(pred - outcome, weights)
    n_bins_requested = int(n_bins)
    n_bins = max(2, min(n_bins_requested, pred.size // max(1, int(min_bin_size))))
    edges = np.unique(np.quantile(pred, np.linspace(0.0, 1.0, n_bins + 1)))
    if edges.size < 3:
        edges = np.linspace(float(np.min(pred)) - 1e-8, float(np.max(pred)) + 1e-8, 3)
    bin_ids = np.searchsorted(edges[1:-1], pred, side="right")
    plugin = 0.0
    total_w = float(np.sum(weights))
    for bin_id in range(edges.size - 1):
        idx = bin_ids == bin_id
        if not np.any(idx):
            continue
        w_bin = weights[idx]
        diff = float(np.average(pred[idx], weights=w_bin) - np.average(outcome[idx], weights=w_bin))
        plugin += float(np.sum(w_bin)) * diff * diff
    plugin = float(plugin / max(total_w, 1e-12))
    folds = min(max(int(n_folds), 2), pred.size)
    gamma_hat = np.full(pred.size, np.nan, dtype=float)
    rng = np.random.default_rng(91037)
    order = rng.permutation(pred.size)
    for hold_idx in np.array_split(order, folds):
        train_mask = np.ones(pred.size, dtype=bool)
        train_mask[hold_idx] = False
        train_pred = pred[train_mask]
        train_outcome = outcome[train_mask]
        train_w = weights[train_mask]
        train_edges = np.unique(np.quantile(train_pred, np.linspace(0.0, 1.0, n_bins + 1)))
        if train_edges.size < 3:
            train_edges = np.linspace(float(np.min(train_pred)) - 1e-8, float(np.max(train_pred)) + 1e-8, 3)
        train_bin_ids = np.searchsorted(train_edges[1:-1], train_pred, side="right")
        fallback = float(np.average(train_outcome, weights=train_w))
        means = np.full(train_edges.size - 1, fallback, dtype=float)
        for bin_id in range(train_edges.size - 1):
            idx = train_bin_ids == bin_id
            if np.any(idx):
                means[bin_id] = float(np.average(train_outcome[idx], weights=train_w[idx]))
        hold_bins = np.searchsorted(train_edges[1:-1], pred[hold_idx], side="right")
        gamma_hat[hold_idx] = means[np.clip(hold_bins, 0, means.size - 1)]
    score = (outcome - pred) * (gamma_hat - pred)
    raw = float(np.nansum(weights * score) / max(float(np.nansum(weights)), 1e-12))
    return {
        "bellman_brier_score": brier,
        "bellman_calibration_error": float(max(raw, 0.0)) if np.isfinite(raw) else float("nan"),
        "bellman_calibration_error_plugin": plugin,
        "bellman_calibration_error_debiased_raw": raw,
        "bellman_calibration_bins_used": int(edges.size - 1),
        "bellman_calibration_bins_requested": int(n_bins_requested),
        "bellman_calibration_mean_bin_size": float(pred.size / max(edges.size - 1, 1)),
    }


def _write_csv(rows: list[dict[str, object]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _load_ground_truth(benchmark_dir: str | Path) -> dict[str, tuple[float, float]]:
    with (Path(benchmark_dir) / "d4rl_gt.pkl").open("rb") as handle:
        return pickle.load(handle)


def _run_policy_seed(
    *,
    full_dataset: HopperTrajectoryDataset,
    dataset_name: str,
    policy_id: str,
    ground_truth_return: float,
    seed: int,
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    rng = np.random.default_rng(seed)
    policy = load_policy(policy_id, args.artifact_dir)
    traj_ids = np.arange(full_dataset.trajectory_count)
    rng.shuffle(traj_ids)
    n_diag = max(1, int(round(full_dataset.trajectory_count * float(args.diagnostic_fraction))))
    diag_ids = traj_ids[:n_diag]
    train_ids = traj_ids[n_diag:]
    train_dataset = _subset_dataset_by_trajectories(full_dataset, train_ids)
    diag_dataset = _subset_dataset_by_trajectories(full_dataset, diag_ids)

    q_cfg = QFitterConfig(
        gamma=float(args.gamma),
        num_updates=int(args.fqe_updates),
        log_interval=max(int(args.fqe_updates) // 10, 1),
        batch_size=int(args.batch_size),
        device=str(args.device),
    )
    full_result = train_q_fitter(train_dataset, policy, config=q_cfg, seed=seed)
    full_initial_v = predict_value_normalized_return(
        full_result,
        train_dataset,
        policy,
        diag_dataset.initial_observations_raw,
        gamma=float(args.gamma),
        seed=seed + 1001,
        n_action_samples=int(args.value_action_samples),
        device=str(args.device),
    )
    behavior_diag = fit_behavior_density(train_dataset, ridge=float(args.behavior_ridge))
    diag_weights, diag_weight_diag = clipped_normalized_density_weights(
        target_policy_log_prob(policy, diag_dataset.observations_raw, diag_dataset.actions),
        behavior_diag.log_prob(diag_dataset.observations_raw, diag_dataset.actions),
        clip=float(args.importance_weight_clip),
    )
    full_diag_v = predict_value_normalized_return(
        full_result,
        train_dataset,
        policy,
        diag_dataset.observations_raw,
        gamma=float(args.gamma),
        seed=seed + 1002,
        n_action_samples=int(args.value_action_samples),
        device=str(args.device),
    )
    full_diag_next_v = predict_value_normalized_return(
        full_result,
        train_dataset,
        policy,
        diag_dataset.next_observations_raw,
        gamma=float(args.gamma),
        seed=seed + 1003,
        n_action_samples=int(args.value_action_samples),
        device=str(args.device),
    )
    full_outcome = diag_dataset.rewards + float(args.gamma) * diag_dataset.masks * full_diag_next_v
    full_metrics = bellman_metrics(
        full_diag_v,
        full_outcome,
        diag_weights,
        n_bins=int(args.calibration_bins),
        min_bin_size=int(args.calibration_min_bin_size),
        n_folds=int(args.calibration_error_folds),
    )

    rows: list[dict[str, object]] = [
        {
            "method": "uncalibrated_all_data_neural_fqe",
            "dataset_name": dataset_name,
            "seed": int(seed),
            "policy_id": policy_id,
            "policy_label": POLICY_SPECS[policy_id].label,
            "ground_truth_return": float(ground_truth_return),
            "estimated_return": _raw_return_from_normalized_return(full_initial_v, train_dataset, float(args.gamma)),
            "absolute_error": abs(_raw_return_from_normalized_return(full_initial_v, train_dataset, float(args.gamma)) - float(ground_truth_return)),
            "calibration_protocol": "uncalibrated_all_data",
            "calibrator": "none",
            "n_train_transitions": len(train_dataset),
            "n_diagnostic_transitions": len(diag_dataset),
            "n_train_trajectories": int(train_dataset.trajectory_count),
            "n_diagnostic_trajectories": int(diag_dataset.trajectory_count),
            "train_data_provenance": f"hopper_dataset={dataset_name};train_trajectories_seed={seed};not_diagnostic_or_oracle",
            "calibration_data_provenance": "none",
            "evaluation_data_provenance": f"hopper_dataset={dataset_name};diagnostic_trajectories_seed={seed};not_train_or_calibration",
            **full_metrics,
            **diag_weight_diag,
            "diagnostic_only": False,
        }
    ]

    fold_models: list[tuple[QFitterResult, HopperTrajectoryDataset]] = []
    cal_values: list[np.ndarray] = []
    cal_next_values: list[np.ndarray] = []
    cal_rewards: list[np.ndarray] = []
    cal_masks: list[np.ndarray] = []
    cal_weights: list[np.ndarray] = []
    for fold_id, hold_ids in enumerate(_trajectory_folds(train_dataset.trajectory_count, int(args.cross_folds), seed + 11)):
        fold_train_ids = np.setdiff1d(np.arange(train_dataset.trajectory_count), hold_ids)
        fold_train = _subset_dataset_by_trajectories(train_dataset, fold_train_ids)
        fold_hold = _subset_dataset_by_trajectories(train_dataset, hold_ids)
        fold_result = train_q_fitter(fold_train, policy, config=q_cfg, seed=seed + 101 * (fold_id + 1))
        fold_models.append((fold_result, fold_train))
        behavior = fit_behavior_density(fold_train, ridge=float(args.behavior_ridge))
        weights, _ = clipped_normalized_density_weights(
            target_policy_log_prob(policy, fold_hold.observations_raw, fold_hold.actions),
            behavior.log_prob(fold_hold.observations_raw, fold_hold.actions),
            clip=float(args.importance_weight_clip),
            normalize=False,
        )
        cal_values.append(
            predict_value_normalized_return(
                fold_result,
                fold_train,
                policy,
                fold_hold.observations_raw,
                gamma=float(args.gamma),
                seed=seed + 2001 + fold_id,
                n_action_samples=int(args.value_action_samples),
                device=str(args.device),
            )
        )
        cal_next_values.append(
            predict_value_normalized_return(
                fold_result,
                fold_train,
                policy,
                fold_hold.next_observations_raw,
                gamma=float(args.gamma),
                seed=seed + 3001 + fold_id,
                n_action_samples=int(args.value_action_samples),
                device=str(args.device),
            )
        )
        cal_rewards.append(fold_hold.rewards)
        cal_masks.append(fold_hold.masks)
        cal_weights.append(weights)

    values = np.concatenate(cal_values)
    next_values = np.concatenate(cal_next_values)
    rewards = np.concatenate(cal_rewards)
    masks = np.concatenate(cal_masks)
    weights = np.concatenate(cal_weights)
    weights = weights / max(float(np.mean(weights)), 1e-12)
    weight_ess = float((np.sum(weights) ** 2) / max(np.sum(weights**2), 1e-12))
    diagnostic_only = bool(weight_ess / max(weights.size, 1) < float(args.min_ess_fraction))

    def fold_median_values(raw_states: np.ndarray, calibrator: HopperValueCalibrator | None, base_seed: int) -> np.ndarray:
        fold_preds = []
        for fold_id, (result, fold_dataset) in enumerate(fold_models):
            pred = predict_value_normalized_return(
                result,
                fold_dataset,
                policy,
                raw_states,
                gamma=float(args.gamma),
                seed=base_seed + fold_id,
                n_action_samples=int(args.value_action_samples),
                device=str(args.device),
            )
            if calibrator is not None:
                pred = calibrator.predict(pred)
            fold_preds.append(pred)
        return np.nanmedian(np.vstack(fold_preds), axis=0)

    for method in args.calibrators:
        calibrator = fit_hopper_value_calibrator(
            str(method),
            values,
            next_values,
            rewards,
            masks,
            weights,
            float(args.gamma),
            n_iterations=int(args.value_calibration_iterations),
            n_bins=int(args.n_bins),
            min_bin_size=int(args.min_bin_size),
        )
        init_v = fold_median_values(diag_dataset.initial_observations_raw, calibrator, seed + 4001)
        diag_v = fold_median_values(diag_dataset.observations_raw, calibrator, seed + 5001)
        diag_next_v = fold_median_values(diag_dataset.next_observations_raw, calibrator, seed + 6001)
        outcome = diag_dataset.rewards + float(args.gamma) * diag_dataset.masks * diag_next_v
        metrics = bellman_metrics(
            diag_v,
            outcome,
            diag_weights,
            n_bins=int(args.calibration_bins),
            min_bin_size=int(args.calibration_min_bin_size),
            n_folds=int(args.calibration_error_folds),
        )
        estimate = _raw_return_from_normalized_return(init_v, train_dataset, float(args.gamma))
        rows.append(
            {
                "method": f"strict_cross_{method}",
                "dataset_name": dataset_name,
                "seed": int(seed),
                "policy_id": policy_id,
                "policy_label": POLICY_SPECS[policy_id].label,
                "ground_truth_return": float(ground_truth_return),
                "estimated_return": estimate,
                "absolute_error": abs(estimate - float(ground_truth_return)),
                "calibration_protocol": "cross",
                "calibrator": str(method),
                "n_train_transitions": len(train_dataset),
                "n_diagnostic_transitions": len(diag_dataset),
                "n_train_trajectories": int(train_dataset.trajectory_count),
                "n_diagnostic_trajectories": int(diag_dataset.trajectory_count),
                "train_data_provenance": f"hopper_dataset={dataset_name};train_trajectories_seed={seed};not_diagnostic_or_oracle",
                "calibration_data_provenance": (
                    f"hopper_dataset={dataset_name};cross_fit_training_trajectory_folds_seed={seed + 11};"
                    "not_diagnostic_or_oracle"
                ),
                "evaluation_data_provenance": f"hopper_dataset={dataset_name};diagnostic_trajectories_seed={seed};not_train_or_calibration",
                **metrics,
                **diag_weight_diag,
                "calibration_importance_weight_ess": weight_ess,
                "calibration_importance_weight_ess_fraction": float(weight_ess / max(weights.size, 1)),
                "calibration_importance_weight_max": float(np.max(weights)),
                "diagnostic_only": diagnostic_only,
                **calibrator.diagnostics,
            }
        )
    return rows


def _summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    import pandas as pd

    df = pd.DataFrame(rows)
    out = []
    def numeric(group: pd.DataFrame, column: str) -> pd.Series:
        if column not in group:
            return pd.Series([float("nan")] * len(group), index=group.index)
        return pd.to_numeric(group[column], errors="coerce")

    group_cols = ["dataset_name", "method"] if "dataset_name" in df.columns else ["method"]
    for key, group in df.groupby(group_cols, dropna=False):
        if isinstance(key, tuple) and len(key) == 2:
            dataset_name, method = key
        else:
            dataset_name = "hopper-medium-v0"
            method = key[0] if isinstance(key, tuple) else key
        err = numeric(group, "absolute_error")
        raw = df[
            df["method"].astype(str).eq("uncalibrated_all_data_neural_fqe")
            & df.get("dataset_name", pd.Series("hopper-medium-v0", index=df.index)).astype(str).eq(str(dataset_name))
        ]
        raw_err_mean = float(numeric(raw, "absolute_error").mean()) if not raw.empty else float("nan")
        raw_cal_mean = float(numeric(raw, "bellman_calibration_error").mean()) if not raw.empty else float("nan")
        rel_err = float(err.mean() / raw_err_mean) if raw_err_mean and np.isfinite(raw_err_mean) else float("nan")
        cal_mean = float(numeric(group, "bellman_calibration_error").mean())
        rel_cal = float(cal_mean / raw_cal_mean) if raw_cal_mean and np.isfinite(raw_cal_mean) else float("nan")
        out.append(
            {
                "dataset_name": dataset_name,
                "method": method,
                "n_rows": int(len(group)),
                "n_policies": int(group["policy_id"].nunique()) if "policy_id" in group else 0,
                "n_seeds": int(group["seed"].nunique()) if "seed" in group else 0,
                "mean_absolute_error": float(err.mean()),
                "se_absolute_error": float(err.std(ddof=1) / math.sqrt(max(err.notna().sum(), 1))) if err.notna().sum() > 1 else float("nan"),
                "mean_bellman_brier_score": float(numeric(group, "bellman_brier_score").mean()),
                "mean_bellman_calibration_error": cal_mean,
                "mean_bellman_calibration_error_plugin": float(numeric(group, "bellman_calibration_error_plugin").mean()),
                "mean_bellman_calibration_error_debiased_raw": float(numeric(group, "bellman_calibration_error_debiased_raw").mean()),
                "mean_bellman_calibration_bins_used": float(numeric(group, "bellman_calibration_bins_used").mean()),
                "mean_bellman_calibration_mean_bin_size": float(numeric(group, "bellman_calibration_mean_bin_size").mean()),
                "mean_importance_weight_ess_fraction": float(numeric(group, "importance_weight_ess_fraction").mean()),
                "diagnostic_only_rate": float(group["diagnostic_only"].astype(bool).mean()),
                "relative_absolute_error_vs_raw": rel_err,
                "relative_bellman_calibration_error_vs_raw": rel_cal,
            }
        )
    return out


def _promotion_audit(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    import pandas as pd

    df = pd.DataFrame(rows)
    if df.empty:
        return []
    if "dataset_name" not in df:
        df["dataset_name"] = "hopper-medium-v0"
    numeric_cols = ["absolute_error", "bellman_calibration_error", "importance_weight_ess_fraction"]
    for col in numeric_cols:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    raw = df[df["method"].astype(str).eq("uncalibrated_all_data_neural_fqe")].copy()
    raw_corr_rows: list[dict[str, object]] = []
    for (dataset_name, seed), group in raw.groupby(["dataset_name", "seed"], dropna=False):
        raw_corr_rows.append(
            {
                "dataset_name": dataset_name,
                "seed": seed,
                "raw_policy_spearman": _safe_spearman(group["estimated_return"], group["ground_truth_return"]),
                "raw_policy_pearson": _safe_pearson(group["estimated_return"], group["ground_truth_return"]),
            }
        )
    raw_corr = pd.DataFrame(raw_corr_rows)
    audit_rows: list[dict[str, object]] = []
    for (dataset_name, method), group in df.groupby(["dataset_name", "method"], dropna=False):
        baseline = raw[raw["dataset_name"].astype(str).eq(str(dataset_name))]
        if baseline.empty:
            continue
        raw_cal_mean = float(baseline["bellman_calibration_error"].mean())
        raw_err_mean = float(baseline["absolute_error"].mean())
        cal_mean = float(group["bellman_calibration_error"].mean())
        err_mean = float(group["absolute_error"].mean())
        rel_cal = cal_mean / raw_cal_mean if np.isfinite(raw_cal_mean) and raw_cal_mean > 0 else float("nan")
        rel_err = err_mean / raw_err_mean if np.isfinite(raw_err_mean) and raw_err_mean > 0 else float("nan")
        merged = group.merge(
            baseline[["dataset_name", "seed", "policy_id", "absolute_error", "bellman_calibration_error"]],
            on=["dataset_name", "seed", "policy_id"],
            suffixes=("", "_raw"),
            how="left",
        )
        ope_win = float((merged["absolute_error"] < merged["absolute_error_raw"]).mean()) if not merged.empty else float("nan")
        cal_win = (
            float((merged["bellman_calibration_error"] < merged["bellman_calibration_error_raw"]).mean())
            if not merged.empty
            else float("nan")
        )
        corr_group = raw_corr[raw_corr["dataset_name"].astype(str).eq(str(dataset_name))]
        raw_spearman = float(pd.to_numeric(corr_group.get("raw_policy_spearman"), errors="coerce").mean()) if not corr_group.empty else float("nan")
        raw_pearson = float(pd.to_numeric(corr_group.get("raw_policy_pearson"), errors="coerce").mean()) if not corr_group.empty else float("nan")
        diagnostic_rate = float(group["diagnostic_only"].astype(bool).mean()) if "diagnostic_only" in group else 1.0
        n_policies = int(group["policy_id"].nunique()) if "policy_id" in group else 0
        n_seeds = int(group["seed"].nunique()) if "seed" in group else 0
        finite_metrics = bool(
            np.isfinite([raw_spearman, raw_pearson, raw_cal_mean, raw_err_mean, rel_cal, rel_err, ope_win, cal_win]).all()
        )
        reasons: list[str] = []
        if str(method) == "uncalibrated_all_data_neural_fqe":
            label = "raw_baseline"
        else:
            if n_policies < 3:
                reasons.append("too_few_target_policies")
            if n_seeds < 2:
                reasons.append("too_few_seeds")
            if diagnostic_rate > 0:
                reasons.append("diagnostic_only_rows")
            if not finite_metrics:
                reasons.append("nonfinite_metrics")
            if not (raw_spearman > 0 and raw_pearson > 0):
                reasons.append("raw_fqe_not_informative")
            if not (raw_cal_mean > 1e-8):
                reasons.append("raw_not_miscalibrated")
            if not (rel_cal < 0.85):
                reasons.append("calibration_error_gate_failed")
            if not (rel_err < 0.90):
                reasons.append("ope_error_gate_failed")
            if not (ope_win >= 0.60 and cal_win >= 0.60):
                reasons.append("win_rate_gate_failed")
            label = "promote_main" if not reasons else "not_promoted"
        audit_rows.append(
            {
                "dataset_name": dataset_name,
                "method": method,
                "audit_label": label,
                "n_rows": int(len(group)),
                "n_policies": n_policies,
                "n_seeds": n_seeds,
                "raw_policy_spearman": raw_spearman,
                "raw_policy_pearson": raw_pearson,
                "relative_absolute_error_vs_raw": rel_err,
                "relative_bellman_calibration_error_vs_raw": rel_cal,
                "absolute_error_win_rate_vs_raw": ope_win,
                "calibration_error_win_rate_vs_raw": cal_win,
                "diagnostic_only_rate": diagnostic_rate,
                "mean_importance_weight_ess_fraction": float(group["importance_weight_ess_fraction"].mean())
                if "importance_weight_ess_fraction" in group
                else float("nan"),
                "failure_reasons": ";".join(reasons),
            }
        )
    return audit_rows


def _write_readout(audit_rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    promoted = [row for row in audit_rows if row.get("audit_label") == "promote_main"]
    lines = [
        "# Hopper Calibration Benchmark Readout",
        "",
        f"Promoted calibrated rows: {len(promoted)}",
        "",
    ]
    if promoted:
        lines.append("| Dataset | Method | Rel. OPE error | Rel. Bellman cal. | OPE win | Cal. win |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for row in promoted:
            lines.append(
                "| {dataset_name} | {method} | {relative_absolute_error_vs_raw:.3f} | "
                "{relative_bellman_calibration_error_vs_raw:.3f} | {absolute_error_win_rate_vs_raw:.2f} | "
                "{calibration_error_win_rate_vs_raw:.2f} |".format(**row)
            )
    else:
        lines.extend(
            [
                "No calibrated Hopper row passed the predeclared gates.",
                "Do not make a main-text Hopper claim unless a later final run changes this label.",
            ]
        )
    path.write_text("\n".join(lines) + "\n")


def apply_stage_defaults(args: argparse.Namespace) -> argparse.Namespace:
    stage = str(getattr(args, "stage", "custom"))
    if stage == "custom":
        return args
    presets = {
        "smoke": {
            "seeds": [0],
            "dataset_names": ["hopper-medium-v0"],
            "target_policies": ["hopper-medium_00", "hopper-medium_05", "hopper-medium_10"],
            "max_trajectories": 16,
            "diagnostic_fraction": 0.25,
            "cross_folds": 2,
            "fqe_updates": 20,
            "value_action_samples": 2,
            "calibrators": ["linear", "isotonic"],
            "value_calibration_iterations": 2,
            "n_bins": 4,
            "min_bin_size": 2,
            "calibration_bins": 4,
            "calibration_min_bin_size": 2,
            "calibration_error_folds": 2,
            "output_dir": "FQE_calibration_neurips/results/hopper_calibration_smoke",
        },
        "pilot": {
            "seeds": [0, 1, 2],
            "dataset_names": ["hopper-medium-v0"],
            "target_policies": ["hopper-medium_00", "hopper-medium_05", "hopper-medium_10"],
            "max_trajectories": 64,
            "diagnostic_fraction": 0.2,
            "cross_folds": 2,
            "fqe_updates": 60,
            "value_action_samples": 2,
            "calibrators": ["linear", "isotonic"],
            "value_calibration_iterations": 3,
            "n_bins": 8,
            "min_bin_size": 20,
            "calibration_bins": 12,
            "calibration_min_bin_size": 50,
            "calibration_error_folds": 3,
            "output_dir": "FQE_calibration_neurips/results/hopper_calibration_pilot",
        },
        "pilot_all_policies": {
            "seeds": [0, 1, 2],
            "dataset_names": ["hopper-medium-v0"],
            "target_policies": None,
            "max_trajectories": 64,
            "diagnostic_fraction": 0.2,
            "cross_folds": 2,
            "fqe_updates": 60,
            "value_action_samples": 2,
            "calibrators": ["linear", "isotonic"],
            "value_calibration_iterations": 3,
            "n_bins": 8,
            "min_bin_size": 20,
            "calibration_bins": 12,
            "calibration_min_bin_size": 50,
            "calibration_error_folds": 3,
            "output_dir": "FQE_calibration_neurips/results/hopper_calibration_pilot_all_policies",
        },
        "expansion": {
            "seeds": [0, 1],
            "dataset_names": sorted(HOPPER_DATASET_SPECS),
            "target_policies": ["hopper-medium_00", "hopper-medium_05", "hopper-medium_10"],
            "max_trajectories": 64,
            "diagnostic_fraction": 0.2,
            "cross_folds": 2,
            "fqe_updates": 120,
            "value_action_samples": 2,
            "calibrators": ["linear", "isotonic"],
            "value_calibration_iterations": 3,
            "n_bins": 8,
            "min_bin_size": 20,
            "calibration_bins": 12,
            "calibration_min_bin_size": 50,
            "calibration_error_folds": 3,
            "skip_missing_datasets": True,
            "output_dir": "FQE_calibration_neurips/results/hopper_calibration_expansion",
        },
        "final": {
            "seeds": [10, 11, 12, 13, 14],
            "dataset_names": ["hopper-medium-v0"],
            "target_policies": None,
            "max_trajectories": 128,
            "diagnostic_fraction": 0.2,
            "cross_folds": 5,
            "fqe_updates": 3000,
            "value_action_samples": 8,
            "calibrators": ["linear", "isotonic"],
            "value_calibration_iterations": 4,
            "n_bins": 20,
            "min_bin_size": 20,
            "calibration_bins": 50,
            "calibration_min_bin_size": 500,
            "calibration_error_folds": 5,
            "output_dir": "FQE_calibration_neurips/results/hopper_calibration_final",
        },
    }
    if stage not in presets:
        raise ValueError(f"Unknown Hopper calibration stage '{stage}'.")
    for key, value in presets[stage].items():
        setattr(args, key, value)
    return args


def _plot(rows: list[dict[str, object]], output_dir: Path) -> None:
    import pandas as pd

    df = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), constrained_layout=True)
    for method, group in df.groupby("method", sort=False):
        axes[0].scatter(group["ground_truth_return"], group["estimated_return"], s=16, alpha=0.7, label=method)
    lo = min(float(df["ground_truth_return"].min()), float(df["estimated_return"].min()))
    hi = max(float(df["ground_truth_return"].max()), float(df["estimated_return"].max()))
    axes[0].plot([lo, hi], [lo, hi], color="0.3", linewidth=0.8)
    axes[0].set_xlabel("ground truth return")
    axes[0].set_ylabel("estimated return")
    axes[0].set_title("Hopper policy values", loc="left", fontweight="bold")
    summary = pd.DataFrame(_summary(rows))
    labels = summary["method"].astype(str)
    if "dataset_name" in summary and summary["dataset_name"].nunique() > 1:
        labels = summary["dataset_name"].astype(str) + " / " + labels
    axes[1].barh(labels, summary["mean_absolute_error"])
    axes[1].set_xlabel("mean absolute error")
    axes[1].set_title("Return error", loc="left", fontweight="bold")
    axes[0].legend(frameon=False, fontsize=6)
    fig.savefig(output_dir / "hopper_calibration_appendix.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "hopper_calibration_appendix.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def run_hopper_calibration_benchmark(args: argparse.Namespace) -> dict[str, Path]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ground_truth = _load_ground_truth(args.benchmark_dir)
    dataset_names = list(getattr(args, "dataset_names", None) or ["hopper-medium-v0"])
    policy_ids = list(args.target_policies or [spec.policy_id for spec in HOPPER_MEDIUM_POLICY_SPECS])
    rows: list[dict[str, object]] = []
    skipped_datasets: list[dict[str, str]] = []
    for dataset_name in dataset_names:
        try:
            full_dataset = load_hopper_dataset(
                args.data_dir,
                dataset_name=dataset_name,
                normalize_states=True,
                normalize_rewards=True,
                bootstrap=False,
                noise_scale=float(args.noise_scale),
                max_trajectories=args.max_trajectories,
                max_transitions=args.max_transitions,
                seed=0,
            )
        except Exception as exc:
            if bool(getattr(args, "skip_missing_datasets", False)):
                skipped_datasets.append({"dataset_name": str(dataset_name), "reason": str(exc)})
                continue
            raise
        for seed in args.seeds:
            for policy_id in policy_ids:
                rows.extend(
                    _run_policy_seed(
                        full_dataset=full_dataset,
                        dataset_name=str(dataset_name),
                        policy_id=policy_id,
                        ground_truth_return=float(ground_truth[policy_id][0]),
                        seed=int(seed),
                        args=args,
                    )
                )
    raw_path = output_dir / "hopper_calibration_results.csv"
    summary_path = output_dir / "hopper_calibration_summary.csv"
    audit_path = output_dir / "hopper_calibration_audit.csv"
    skipped_path = output_dir / "hopper_skipped_datasets.csv"
    readout_path = output_dir / "hopper_calibration_readout.md"
    config_path = output_dir / "hopper_calibration_config.json"
    _write_csv(rows, raw_path)
    _write_csv(_summary(rows), summary_path)
    audit_rows = _promotion_audit(rows)
    _write_csv(audit_rows, audit_path)
    _write_csv(skipped_datasets, skipped_path)
    _write_readout(audit_rows, readout_path)
    config = {key: value for key, value in vars(args).items() if key != "target_policies"}
    config["target_policies"] = policy_ids
    config["dataset_names"] = dataset_names
    config["available_dataset_specs"] = sorted(HOPPER_DATASET_SPECS)
    config_path.write_text(json.dumps(config, indent=2))
    if rows:
        _plot(rows, output_dir)
    return {"raw": raw_path, "summary": summary_path, "audit": audit_path, "readout": readout_path, "config": config_path}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run strict value-space calibration on the Hopper medium benchmark.")
    parser.add_argument(
        "--stage",
        choices=["custom", "smoke", "pilot", "pilot_all_policies", "expansion", "final"],
        default="custom",
    )
    parser.add_argument("--data_dir", default="hopper_fqe_benchmark/artifacts")
    parser.add_argument("--artifact_dir", default="hopper_fqe_benchmark/artifacts")
    parser.add_argument("--benchmark_dir", default="hopper_fqe_benchmark/artifacts/benchmark/dope")
    parser.add_argument("--output_dir", default="FQE_calibration_neurips/results/hopper_calibration_appendix")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--dataset_names", nargs="+", default=["hopper-medium-v0"])
    parser.add_argument("--target_policies", nargs="*", default=None)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--noise_scale", type=float, default=0.0)
    parser.add_argument("--max_trajectories", type=int, default=64)
    parser.add_argument("--max_transitions", type=int, default=None)
    parser.add_argument("--diagnostic_fraction", type=float, default=0.2)
    parser.add_argument("--cross_folds", type=int, default=5)
    parser.add_argument("--fqe_updates", type=int, default=3000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--value_action_samples", type=int, default=8)
    parser.add_argument("--importance_weight_clip", type=float, default=20.0)
    parser.add_argument("--behavior_ridge", type=float, default=1e-3)
    parser.add_argument("--min_ess_fraction", type=float, default=0.05)
    parser.add_argument("--calibrators", nargs="+", default=["isotonic", "isotonic_histogram", "histogram", "linear"])
    parser.add_argument("--value_calibration_iterations", type=int, default=4)
    parser.add_argument("--n_bins", type=int, default=20)
    parser.add_argument("--min_bin_size", type=int, default=20)
    parser.add_argument("--calibration_bins", type=int, default=50)
    parser.add_argument("--calibration_min_bin_size", type=int, default=1000)
    parser.add_argument("--calibration_error_folds", type=int, default=5)
    parser.add_argument("--skip_missing_datasets", action="store_true")
    args = parser.parse_args()
    args = apply_stage_defaults(args)
    outputs = run_hopper_calibration_benchmark(args)
    for name, path in outputs.items():
        print(f"Wrote {name}: {path}")


if __name__ == "__main__":
    main()
