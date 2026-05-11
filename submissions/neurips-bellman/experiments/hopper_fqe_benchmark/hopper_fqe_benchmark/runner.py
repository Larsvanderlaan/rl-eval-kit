from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import json
import os
from pathlib import Path
import pickle
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", str(Path("hopper_fqe_benchmark/artifacts/mplconfig").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(Path("hopper_fqe_benchmark/artifacts/cache").resolve()))

import matplotlib.pyplot as plt
import numpy as np

from FQE_neurips.neural_rkhs_weights import KernelConfig, NeuralRKHSWeightsConfig
from FQE_neurips.ratio_estimation import NeuralRatioConfig
from FQE_neurips.sw_fqe import resolve_sample_weights
from FQE_neurips.utils import TransitionBatch

from .data import HopperTrajectoryDataset, load_hopper_medium_v0
from .dice import DualDICEConfig
from .features import ContinuousFeatureEncoder
from .fqe import QFitterConfig
from .official_tf_baselines import (
    estimate_dual_dice_return,
    estimate_policy_return,
    extract_dual_dice_weights,
    train_dual_dice,
    train_q_fitter,
)
from .policies import HOPPER_MEDIUM_POLICY_SPECS, POLICY_SPECS, load_policy


OFFICIAL_BASELINE_FILES = {
    "official_fqe_l2": "d4rl_fqel2.pkl",
    "official_dice": "d4rl_dice.pkl",
}


@dataclass
class BenchmarkConfig:
    data_dir: str = "hopper_fqe_benchmark/artifacts"
    artifact_dir: str = "hopper_fqe_benchmark/artifacts"
    benchmark_dir: str = "hopper_fqe_benchmark/artifacts/benchmark/dope"
    output_dir: str = "hopper_fqe_benchmark/outputs"
    task_name: str = "hopper-medium"
    gamma_eval: float = 0.99
    gamma_ratio: float = 0.99
    noise_scale: float = 0.25
    max_trajectories: int | None = None
    max_transitions: int | None = None
    target_policies: tuple[str, ...] = tuple(spec.policy_id for spec in HOPPER_MEDIUM_POLICY_SPECS)
    methods: tuple[str, ...] = (
        "standard_fqe",
        "weighted_dual_dice",
        "weighted_linear",
    )
    seeds: tuple[int, ...] = (0, 1, 2)
    min_weight: float = 1e-4
    max_weight: float = 20.0
    clip_quantile: float = 0.995
    uniform_mix: float = 0.05
    target_ess_fraction: float = 0.4
    ratio_feature_quadratic: bool = False
    ratio_feature_cross_terms: bool = False
    fqe_num_updates: int = 20_000
    dice_num_updates: int = 20_000
    saddle_max_steps: int = 5_000
    rkhs_max_steps: int = 3_000
    saddle_normalization_penalty: float = 2.0
    saddle_uniform_mix: float = 0.0
    saddle_target_ess_fraction: float | None = 0.2
    saddle_max_weight: float = 100.0
    rkhs_normalization_penalty: float = 2.0
    rkhs_uniform_mix: float = 0.0
    rkhs_target_ess_fraction: float | None = 0.2
    rkhs_max_weight: float = 100.0
    device: str = "cpu"


def default_q_fitter_config(cfg: BenchmarkConfig) -> QFitterConfig:
    return QFitterConfig(
        gamma=cfg.gamma_eval,
        critic_lr=3e-4,
        weight_decay=1e-5,
        tau=0.005,
        batch_size=256,
        num_updates=cfg.fqe_num_updates,
        log_interval=max(cfg.fqe_num_updates // 20, 1),
        device=cfg.device,
    )


def default_dual_dice_config(cfg: BenchmarkConfig) -> DualDICEConfig:
    return DualDICEConfig(
        gamma=cfg.gamma_eval,
        weight_decay=1e-5,
        nu_lr=1e-4,
        zeta_lr=1e-3,
        batch_size=256,
        num_updates=cfg.dice_num_updates,
        log_interval=max(cfg.dice_num_updates // 20, 1),
        device=cfg.device,
    )


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _load_pickle(path: Path) -> dict:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _load_benchmark_truth_and_refs(cfg: BenchmarkConfig) -> tuple[dict[str, tuple[float, float]], dict[str, dict[str, tuple[float, float]]]]:
    benchmark_dir = Path(cfg.benchmark_dir)
    ground_truth = _load_pickle(benchmark_dir / "d4rl_gt.pkl")
    references = {
        method: _load_pickle(benchmark_dir / filename)
        for method, filename in OFFICIAL_BASELINE_FILES.items()
    }
    return ground_truth, references


def _sample_actions_full(
    policy,
    observations: np.ndarray,
    *,
    seed: int,
    chunk_size: int = 65_536,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    observations_arr = np.asarray(observations, dtype=np.float32)
    actions = np.empty((observations_arr.shape[0], policy.action_dim), dtype=np.float32)
    for start in range(0, observations_arr.shape[0], chunk_size):
        stop = min(start + chunk_size, observations_arr.shape[0])
        actions[start:stop] = policy.sample_actions(observations_arr[start:stop], rng=rng, deterministic=False)
    return actions


def _ratio_action_cache_path(cfg: BenchmarkConfig, policy_id: str, dataset: HopperTrajectoryDataset, seed: int) -> Path:
    cache_dir = Path(cfg.artifact_dir) / "ratio_action_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    name = f"{policy_id}_n{len(dataset)}_traj{dataset.trajectory_count}_seed{seed}.npy"
    return cache_dir / name


def _estimate_action_mse(dataset: HopperTrajectoryDataset, policy, seed: int, n_samples: int = 20_000) -> float:
    rng = np.random.default_rng(seed)
    sample_size = min(int(n_samples), len(dataset))
    indices = rng.choice(len(dataset), size=sample_size, replace=False)
    policy_actions = policy.sample_actions(dataset.observations_raw[indices], rng=rng, deterministic=False)
    residual = policy_actions - dataset.actions[indices]
    return float(np.mean(np.sum(residual**2, axis=1)))


def _build_ratio_batch(dataset: HopperTrajectoryDataset, policy_id: str, policy, cfg: BenchmarkConfig, seed: int) -> TransitionBatch:
    cache_path = _ratio_action_cache_path(cfg, policy_id=policy_id, dataset=dataset, seed=seed)
    if cache_path.exists():
        next_actions = np.load(cache_path)
    else:
        next_actions = _sample_actions_full(policy, dataset.next_observations_raw, seed=seed)
        np.save(cache_path, next_actions)
    return TransitionBatch(
        states=np.asarray(dataset.observations, dtype=np.float32),
        actions=np.asarray(dataset.actions, dtype=np.float32),
        rewards=np.asarray(dataset.rewards, dtype=np.float32),
        next_states=np.asarray(dataset.next_observations, dtype=np.float32),
        next_actions=next_actions,
    )


def _policy_return_row(
    *,
    method: str,
    seed: int,
    policy_id: str,
    ground_truth_return: float,
    estimated_return: float,
    action_mse: float,
    n_transitions: int,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    row = {
        "method": method,
        "seed": int(seed),
        "policy_id": policy_id,
        "policy_label": POLICY_SPECS[policy_id].label,
        "ground_truth_return": float(ground_truth_return),
        "estimated_return": float(estimated_return),
        "absolute_error": float(abs(estimated_return - ground_truth_return)),
        "action_mse_vs_dataset": float(action_mse),
        "n_transitions": int(n_transitions),
    }
    if extra:
        row.update(extra)
    return row


def _compact_diagnostics(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _compact_diagnostics(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        seq = list(value)
        if len(seq) > 8 and all(isinstance(item, (int, float, np.floating)) for item in seq):
            return {
                "n": len(seq),
                "first": float(seq[0]),
                "last": float(seq[-1]),
                "min": float(np.min(seq)),
                "max": float(np.max(seq)),
            }
        return [_compact_diagnostics(item) for item in seq]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _spearman_rank_correlation(prediction: np.ndarray, ground_truth: np.ndarray) -> float:
    def average_ranks(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        order = np.argsort(values, kind="mergesort")
        sorted_vals = values[order]
        ranks = np.empty_like(sorted_vals, dtype=np.float64)
        n = len(sorted_vals)
        start = 0
        while start < n:
            end = start + 1
            while end < n and sorted_vals[end] == sorted_vals[start]:
                end += 1
            avg_rank = 0.5 * (start + end - 1)
            ranks[start:end] = avg_rank
            start = end
        out = np.empty_like(ranks)
        out[order] = ranks
        return out

    pred = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(ground_truth, dtype=np.float64)
    pred_rank = average_ranks(pred)
    truth_rank = average_ranks(truth)
    pred_centered = pred_rank - pred_rank.mean()
    truth_centered = truth_rank - truth_rank.mean()
    denom = np.sqrt(np.sum(pred_centered**2) * np.sum(truth_centered**2))
    if denom <= 1e-12:
        return 0.0
    return float(np.sum(pred_centered * truth_centered) / denom)


def _normalized_regret(prediction: np.ndarray, ground_truth: np.ndarray, top_k: int = 1) -> float:
    pred = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(ground_truth, dtype=np.float64)
    max_val = float(np.max(truth))
    top_indices = np.argsort(pred)[-top_k:]
    top_truth = float(np.max(truth[top_indices]))
    return float((max_val - top_truth) / max(max_val, 1e-8))


def _normalized_values(prediction: np.ndarray, ground_truth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    truth = np.asarray(ground_truth, dtype=np.float64)
    pred = np.asarray(prediction, dtype=np.float64)
    worst = float(np.min(truth))
    best = float(np.max(truth))
    scale = max(best - worst, 1e-8)
    truth_norm = (truth - worst) / scale
    pred_norm = np.minimum((pred - worst) / scale, 2.0)
    return pred_norm, truth_norm


def _metric_row(method: str, seed: int, prediction: np.ndarray, ground_truth: np.ndarray) -> dict[str, object]:
    pred_norm, truth_norm = _normalized_values(prediction, ground_truth)
    return {
        "method": method,
        "seed": int(seed),
        "absolute_error_mean": float(np.mean(np.abs(pred_norm - truth_norm))),
        "rank_correlation": _spearman_rank_correlation(pred_norm, truth_norm),
        "regret_at_1": _normalized_regret(pred_norm, truth_norm, top_k=1),
    }


def _write_csv(rows: list[dict[str, object]], path: Path) -> None:
    if not rows:
        return
    _ensure_parent(path)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _aggregate_metrics(metric_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in metric_rows:
        grouped.setdefault(str(row["method"]), []).append(row)

    summary_rows: list[dict[str, object]] = []
    for method, rows in sorted(grouped.items()):
        summary_rows.append(
            {
                "method": method,
                "n_runs": int(len(rows)),
                "absolute_error_mean": float(np.mean([float(row["absolute_error_mean"]) for row in rows])),
                "absolute_error_std": float(np.std([float(row["absolute_error_mean"]) for row in rows], ddof=0)),
                "rank_correlation_mean": float(np.mean([float(row["rank_correlation"]) for row in rows])),
                "rank_correlation_std": float(np.std([float(row["rank_correlation"]) for row in rows], ddof=0)),
                "regret_at_1_mean": float(np.mean([float(row["regret_at_1"]) for row in rows])),
                "regret_at_1_std": float(np.std([float(row["regret_at_1"]) for row in rows], ddof=0)),
            }
        )
    return summary_rows


def _plot_metrics(summary_rows: Iterable[dict[str, object]], path: Path) -> None:
    rows = list(summary_rows)
    if not rows:
        return
    methods = [str(row["method"]) for row in rows]
    metrics = [
        ("absolute_error_mean", "Norm. Absolute Error"),
        ("rank_correlation_mean", "Rank Corr."),
        ("regret_at_1_mean", "Regret@1"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    x = np.arange(len(methods), dtype=np.float64)
    for ax, (metric_key, title) in zip(axes, metrics):
        values = [float(row[metric_key]) for row in rows]
        errors = [float(row[metric_key.replace("_mean", "_std")]) for row in rows]
        ax.bar(x, values, yerr=errors)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=35, ha="right")
        ax.set_title(title)
    fig.tight_layout()
    _ensure_parent(path)
    fig.savefig(path)
    plt.close(fig)


def _reference_metric_rows(
    cfg: BenchmarkConfig,
    ground_truth: dict[str, tuple[float, float]],
    references: dict[str, dict[str, tuple[float, float]]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    policy_ids = list(cfg.target_policies)
    truth = np.asarray([float(ground_truth[policy_id][0]) for policy_id in policy_ids], dtype=np.float64)
    rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    for method, benchmark_predictions in references.items():
        preds = np.asarray([float(benchmark_predictions[policy_id][0]) for policy_id in policy_ids], dtype=np.float64)
        for policy_id, pred in zip(policy_ids, preds):
            rows.append(
                _policy_return_row(
                    method=method,
                    seed=-1,
                    policy_id=policy_id,
                    ground_truth_return=float(ground_truth[policy_id][0]),
                    estimated_return=float(pred),
                    action_mse=float("nan"),
                    n_transitions=0,
                )
            )
        metric_rows.append(_metric_row(method=method, seed=-1, prediction=preds, ground_truth=truth))
    return rows, metric_rows


def _train_methods_for_policy(
    dataset: HopperTrajectoryDataset,
    policy_id: str,
    ground_truth_return: float,
    cfg: BenchmarkConfig,
    seed: int,
) -> list[dict[str, object]]:
    policy = load_policy(policy_id, cfg.artifact_dir)
    action_mse = _estimate_action_mse(dataset, policy, seed=seed + 101)
    rows: list[dict[str, object]] = []

    if any(method in cfg.methods for method in ("weighted_linear", "weighted_neural_saddle", "weighted_rkhs")):
        encoder = ContinuousFeatureEncoder.fit(
            dataset.observations,
            dataset.actions,
            include_quadratic=cfg.ratio_feature_quadratic,
            include_cross_terms=cfg.ratio_feature_cross_terms,
        )
        batch = _build_ratio_batch(dataset, policy_id=policy_id, policy=policy, cfg=cfg, seed=seed + 211)
    else:
        encoder = None
        batch = None

    if "standard_fqe" in cfg.methods:
        result = train_q_fitter(dataset, policy, config=default_q_fitter_config(cfg), seed=seed)
        estimate = estimate_policy_return(
            result,
            dataset,
            policy,
            gamma=cfg.gamma_eval,
            seed=seed + 307,
        )
        rows.append(
            _policy_return_row(
                method="standard_fqe",
                seed=seed,
                policy_id=policy_id,
                ground_truth_return=ground_truth_return,
                estimated_return=estimate,
                action_mse=action_mse,
                n_transitions=len(dataset),
                extra={"diagnostics_json": json.dumps(_compact_diagnostics({"loss_history": result.loss_history}))},
            )
        )

    dual_dice_result = None
    if any(method in cfg.methods for method in ("dual_dice", "weighted_dual_dice")):
        dual_dice_result = train_dual_dice(dataset, policy, config=default_dual_dice_config(cfg), seed=seed)

    if "dual_dice" in cfg.methods and dual_dice_result is not None:
        estimate, pred_ratio = estimate_dual_dice_return(
            dual_dice_result,
            dataset,
            gamma=cfg.gamma_eval,
            seed=seed + 401,
        )
        rows.append(
            _policy_return_row(
                method="dual_dice",
                seed=seed,
                policy_id=policy_id,
                ground_truth_return=ground_truth_return,
                estimated_return=estimate,
                action_mse=action_mse,
                n_transitions=len(dataset),
                extra={
                    "pred_ratio": float(pred_ratio),
                    "diagnostics_json": json.dumps(_compact_diagnostics({"loss_history": dual_dice_result.loss_history})),
                },
            )
        )

    if "weighted_dual_dice" in cfg.methods and dual_dice_result is not None:
        zeta_weights = extract_dual_dice_weights(
            dual_dice_result,
            dataset,
        )
        result = train_q_fitter(
            dataset,
            policy,
            sample_weights=zeta_weights,
            config=default_q_fitter_config(cfg),
            seed=seed + 509,
        )
        estimate = estimate_policy_return(
            result,
            dataset,
            policy,
            gamma=cfg.gamma_eval,
            seed=seed + 607,
        )
        rows.append(
            _policy_return_row(
                method="weighted_dual_dice",
                seed=seed,
                policy_id=policy_id,
                ground_truth_return=ground_truth_return,
                estimated_return=estimate,
                action_mse=action_mse,
                n_transitions=len(dataset),
                extra={
                    "weight_mean": float(np.mean(zeta_weights)),
                    "weight_min": float(np.min(zeta_weights)),
                    "weight_max": float(np.max(zeta_weights)),
                    "ess_fraction": float((zeta_weights.sum() ** 2) / max(np.sum(zeta_weights**2), 1e-8) / len(zeta_weights)),
                    "diagnostics_json": json.dumps(_compact_diagnostics({"loss_history": result.loss_history})),
                },
            )
        )

    if "weighted_linear" in cfg.methods and batch is not None and encoder is not None:
        weights, ratio_result, metadata = resolve_sample_weights(
            batch=batch,
            n_states=1,
            n_actions=1,
            ratio_model="linear",
            ratio_solver="closed_form",
            ratio_feature_map=lambda s, a: encoder.transform(s, a),
            gamma_ratio=cfg.gamma_ratio,
            min_weight=cfg.min_weight,
            max_weight=cfg.max_weight,
            ratio_kwargs={
                "ridge_primal": 1e-4,
                "ridge_dual": 1e-4,
                "normalization_penalty": 10.0,
                "clip_quantile": cfg.clip_quantile,
                "uniform_mix": cfg.uniform_mix,
                "target_ess_fraction": cfg.target_ess_fraction,
            },
        )
        result = train_q_fitter(
            dataset,
            policy,
            sample_weights=weights,
            config=default_q_fitter_config(cfg),
            seed=seed + 701,
        )
        estimate = estimate_policy_return(
            result,
            dataset,
            policy,
            gamma=cfg.gamma_eval,
            seed=seed + 809,
        )
        rows.append(
            _policy_return_row(
                method="weighted_linear",
                seed=seed,
                policy_id=policy_id,
                ground_truth_return=ground_truth_return,
                estimated_return=estimate,
                action_mse=action_mse,
                n_transitions=len(dataset),
                extra={
                    "weight_source": metadata["weight_source"],
                    "weight_mean": float(np.mean(weights)),
                    "weight_min": float(np.min(weights)),
                    "weight_max": float(np.max(weights)),
                    "ess_fraction": float((weights.sum() ** 2) / max(np.sum(weights**2), 1e-8) / len(weights)),
                    "diagnostics_json": json.dumps(_compact_diagnostics(ratio_result.diagnostics if ratio_result is not None else {})),
                },
            )
        )

    if "weighted_neural_saddle" in cfg.methods and batch is not None and encoder is not None:
        saddle_config = NeuralRatioConfig(
            hidden_dims_weight=(128, 128),
            hidden_dims_critic=(128, 128),
            activation="relu",
            max_steps=cfg.saddle_max_steps,
            batch_size=512,
            step_size=1e-3,
            ridge_weight=1e-5,
            ridge_critic=1e-5,
            normalization_penalty=cfg.saddle_normalization_penalty,
            positivity="softplus",
            grad_clip_norm=5.0,
            log_every=max(cfg.saddle_max_steps // 10, 50),
            valid_fraction=0.05,
            early_stopping_patience=20,
            min_improvement=1e-5,
            use_ema=True,
            ema_decay=0.995,
            clip_quantile=cfg.clip_quantile,
            uniform_mix=cfg.saddle_uniform_mix,
            target_ess_fraction=cfg.saddle_target_ess_fraction,
            max_uniform_mix=0.3,
            device=cfg.device,
            seed=seed,
        )
        weights, ratio_result, metadata = resolve_sample_weights(
            batch=batch,
            n_states=1,
            n_actions=1,
            ratio_model="neural",
            ratio_feature_map=lambda s, a: encoder.transform(s, a),
            gamma_ratio=cfg.gamma_ratio,
            min_weight=cfg.min_weight,
            max_weight=cfg.saddle_max_weight,
            ratio_kwargs={"config": saddle_config},
        )
        result = train_q_fitter(
            dataset,
            policy,
            sample_weights=weights,
            config=default_q_fitter_config(cfg),
            seed=seed + 857,
        )
        estimate = estimate_policy_return(
            result,
            dataset,
            policy,
            gamma=cfg.gamma_eval,
            seed=seed + 907,
        )
        rows.append(
            _policy_return_row(
                method="weighted_neural_saddle",
                seed=seed,
                policy_id=policy_id,
                ground_truth_return=ground_truth_return,
                estimated_return=estimate,
                action_mse=action_mse,
                n_transitions=len(dataset),
                extra={
                    "weight_source": metadata["weight_source"],
                    "weight_mean": float(np.mean(weights)),
                    "weight_min": float(np.min(weights)),
                    "weight_max": float(np.max(weights)),
                    "ess_fraction": float((weights.sum() ** 2) / max(np.sum(weights**2), 1e-8) / len(weights)),
                    "diagnostics_json": json.dumps(_compact_diagnostics(ratio_result.diagnostics if ratio_result is not None else {})),
                },
            )
        )

    if "weighted_rkhs" in cfg.methods and batch is not None and encoder is not None:
        rkhs_config = NeuralRKHSWeightsConfig(
            hidden_dims_weight=(128, 128),
            learning_rate=1e-3,
            weight_decay=1e-5,
            critic_ridge=1e-5,
            normalization_penalty=cfg.rkhs_normalization_penalty,
            max_steps=cfg.rkhs_max_steps,
            valid_fraction=0.05,
            early_stopping_patience=20,
            clip_quantile=cfg.clip_quantile,
            max_weight=cfg.rkhs_max_weight,
            min_weight=cfg.min_weight,
            uniform_mix=cfg.rkhs_uniform_mix,
            target_ess_fraction=cfg.rkhs_target_ess_fraction,
            device=cfg.device,
            seed=seed,
            kernel=KernelConfig(kernel="rbf", bandwidth="median", max_anchors=512),
        )
        weights, ratio_result, metadata = resolve_sample_weights(
            batch=batch,
            n_states=1,
            n_actions=1,
            ratio_model="neural_rkhs",
            ratio_feature_map=lambda s, a: encoder.transform(s, a),
            gamma_ratio=cfg.gamma_ratio,
            min_weight=cfg.min_weight,
            max_weight=cfg.rkhs_max_weight,
            ratio_kwargs={"config": rkhs_config},
        )
        result = train_q_fitter(
            dataset,
            policy,
            sample_weights=weights,
            config=default_q_fitter_config(cfg),
            seed=seed + 907,
        )
        estimate = estimate_policy_return(
            result,
            dataset,
            policy,
            gamma=cfg.gamma_eval,
            seed=seed + 1_009,
        )
        rows.append(
            _policy_return_row(
                method="weighted_rkhs",
                seed=seed,
                policy_id=policy_id,
                ground_truth_return=ground_truth_return,
                estimated_return=estimate,
                action_mse=action_mse,
                n_transitions=len(dataset),
                extra={
                    "weight_source": metadata["weight_source"],
                    "weight_mean": float(np.mean(weights)),
                    "weight_min": float(np.min(weights)),
                    "weight_max": float(np.max(weights)),
                    "ess_fraction": float((weights.sum() ** 2) / max(np.sum(weights**2), 1e-8) / len(weights)),
                    "diagnostics_json": json.dumps(_compact_diagnostics(ratio_result.diagnostics if ratio_result is not None else {})),
                },
            )
        )

    return rows


def run_benchmark(cfg: BenchmarkConfig) -> dict[str, object]:
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ground_truth, references = _load_benchmark_truth_and_refs(cfg)
    dataset = load_hopper_medium_v0(
        cfg.data_dir,
        normalize_states=True,
        normalize_rewards=True,
        bootstrap=False,
        noise_scale=cfg.noise_scale,
        max_trajectories=cfg.max_trajectories,
        max_transitions=cfg.max_transitions,
        seed=0,
    )

    policy_ids = list(cfg.target_policies)
    all_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []

    for seed in cfg.seeds:
        seed_rows: list[dict[str, object]] = []
        for policy_id in policy_ids:
            seed_rows.extend(
                _train_methods_for_policy(
                    dataset=dataset,
                    policy_id=policy_id,
                    ground_truth_return=float(ground_truth[policy_id][0]),
                    cfg=cfg,
                    seed=seed,
                )
            )
        all_rows.extend(seed_rows)
        for method in cfg.methods:
            method_rows = [row for row in seed_rows if row["method"] == method]
            if not method_rows:
                continue
            preds = np.asarray([float(row["estimated_return"]) for row in method_rows], dtype=np.float64)
            truth = np.asarray([float(row["ground_truth_return"]) for row in method_rows], dtype=np.float64)
            metric_rows.append(_metric_row(method=method, seed=seed, prediction=preds, ground_truth=truth))

    reference_rows, reference_metric_rows = _reference_metric_rows(cfg, ground_truth=ground_truth, references=references)
    all_rows.extend(reference_rows)
    metric_rows.extend(reference_metric_rows)

    summary_rows = _aggregate_metrics(metric_rows)

    raw_path = output_dir / "hopper_medium_results.csv"
    metric_path = output_dir / "hopper_medium_metrics.csv"
    summary_path = output_dir / "hopper_medium_summary.csv"
    config_path = output_dir / "hopper_medium_config.json"
    figure_path = output_dir / "hopper_medium_metrics.png"

    _write_csv(all_rows, raw_path)
    _write_csv(metric_rows, metric_path)
    _write_csv(summary_rows, summary_path)
    _plot_metrics(summary_rows, figure_path)
    _ensure_parent(config_path)
    config_path.write_text(json.dumps(asdict(cfg), indent=2))

    return {
        "config": asdict(cfg),
        "results_csv": str(raw_path),
        "metrics_csv": str(metric_path),
        "summary_csv": str(summary_path),
        "figure_path": str(figure_path),
        "n_policy_rows": len(all_rows),
        "n_metric_rows": len(metric_rows),
    }
