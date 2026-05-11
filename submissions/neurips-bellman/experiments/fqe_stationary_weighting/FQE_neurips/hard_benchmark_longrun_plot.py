from __future__ import annotations

import argparse
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib-codex"))
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .fqe_linear import LinearFQEConfig, fit_weighted_linear_fqe, predict_linear_q_values
from .hard_benchmark_experiment import _estimate_screening_weights
from .hub_spoke_latent_benchmark import HubSpokeLatentConfig, build_hub_spoke_latent_benchmark
from .latent_garnet_benchmark import simulate_behavior_trajectory
from .utils import evaluate_policy_tabular, stationary_state_action_distribution


@dataclass
class HardLongRunPlotConfig:
    behavior_solid_prob: float = 0.40
    dataset_size: int = 2500
    gamma_eval: float = 0.95
    ridge: float = 1e-2
    n_outer_iters: int = 100
    seeds: int = 50
    output_dir: str = "FQE_neurips/outputs/paper_figures"


def _compute_curves(config: HardLongRunPlotConfig) -> dict[str, list[list[float]]]:
    methods = ["unweighted", "oracle", "weighted_linear_basic"]
    all_curves = {method: [] for method in methods}

    for seed in range(config.seeds):
        benchmark = build_hub_spoke_latent_benchmark(
            HubSpokeLatentConfig(
                behavior_solid_prob=config.behavior_solid_prob,
                dataset_size=config.dataset_size,
                reward_gamma=config.gamma_eval,
                seed=seed,
            )
        )
        batch = simulate_behavior_trajectory(benchmark, seed=seed)
        x = benchmark.featurize_state_actions(batch.states, batch.actions, feature_set="linear_q")
        x_next = benchmark.featurize_state_actions(batch.next_states, batch.next_actions, feature_set="linear_q")
        weights = _estimate_screening_weights(benchmark, batch, gamma_ratio=1.0, seed=seed, include_rkhs=False)
        q_star = evaluate_policy_tabular(benchmark.mdp, gamma=config.gamma_eval).reshape(-1)
        mu_pi = stationary_state_action_distribution(benchmark.mdp, benchmark.mdp.target_policy)
        grid_states = np.repeat(np.arange(benchmark.config.n_states), benchmark.config.n_actions)
        grid_actions = np.tile(np.arange(benchmark.config.n_actions), benchmark.config.n_states)
        grid_feat = benchmark.featurize_state_actions(grid_states, grid_actions, feature_set="linear_q")

        weight_map = {
            "unweighted": None,
            "oracle": weights["oracle_weights"],
            "weighted_linear_basic": weights["linear_basic"].weights,
        }
        fqe_config = LinearFQEConfig(
            solver="iterative",
            gamma=config.gamma_eval,
            ridge=config.ridge,
            n_outer_iters=config.n_outer_iters,
            target_update_tau=1.0,
            valid_fraction=0.0,
            early_stopping_patience=None,
            min_improvement=1e-7,
            tol=1e-12,
            use_averaging=False,
            averaging_start_iter=5,
            initial_theta_mode="zero",
            selection_mode="last_iter",
            track_iterates=True,
            reduce_rank=True,
        )
        for method, sample_weights in weight_map.items():
            result = fit_weighted_linear_fqe(
                batch=batch,
                state_action_features=x,
                next_state_action_features=x_next,
                weights=sample_weights,
                config=fqe_config,
                seed=seed,
            )
            curve: list[float] = []
            for theta in result.theta_iterates:
                preds = predict_linear_q_values(theta, grid_feat)
                curve.append(float(np.sqrt(np.sum(mu_pi * (preds - q_star) ** 2))))
            all_curves[method].append(curve)
    return all_curves


def _summarize(arr: np.ndarray) -> dict[str, float]:
    return {
        "iter10": float(np.median(arr[:, 9])),
        "iter25": float(np.median(arr[:, 24])),
        "iter50": float(np.median(arr[:, 49])),
        "final": float(np.median(arr[:, -1])),
        "best": float(np.median(np.min(arr, axis=1))),
        "best_iter": float(np.median(np.argmin(arr, axis=1) + 1)),
    }


def _plot_band(
    curves: dict[str, np.ndarray],
    output_path: Path,
    title: str,
    lower_q: float,
    upper_q: float,
) -> dict[str, dict[str, float]]:
    style = {
        "unweighted": ("Unweighted", "#444444"),
        "oracle": ("Oracle stationary", "#1f77b4"),
        "weighted_linear_basic": ("Learned stationary", "#2ca02c"),
    }
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    summary: dict[str, dict[str, float]] = {}
    for method, arr in curves.items():
        label, color = style[method]
        med = np.median(arr, axis=0)
        lo = np.quantile(arr, lower_q, axis=0)
        hi = np.quantile(arr, upper_q, axis=0)
        xs = np.arange(1, len(med) + 1)
        ax.plot(xs, med, label=label, color=color, linewidth=2.4)
        ax.fill_between(xs, lo, hi, color=color, alpha=0.12)
        summary[method] = _summarize(arr)
    ax.set_xlabel("Linear FQE iteration")
    ax.set_ylabel("Target RMSE")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return summary


def generate_hard_longrun_plots(config: HardLongRunPlotConfig | None = None) -> dict[str, object]:
    if config is None:
        config = HardLongRunPlotConfig()

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_curves = _compute_curves(config)
    curves = {method: np.asarray(values, dtype=np.float64) for method, values in raw_curves.items()}

    iqr_path = output_dir / "hard_longrun_ridge001_50seeds_iqr.png"
    ci95_path = output_dir / "hard_longrun_ridge001_50seeds_95.png"
    iqr_summary = _plot_band(
        curves=curves,
        output_path=iqr_path,
        title="Hard benchmark: long-run undamped linear FQE (median + IQR)",
        lower_q=0.25,
        upper_q=0.75,
    )
    ci95_summary = _plot_band(
        curves=curves,
        output_path=ci95_path,
        title="Hard benchmark: long-run undamped linear FQE (median + 95% band)",
        lower_q=0.025,
        upper_q=0.975,
    )

    summary = {
        "config": {
            "behavior_solid_prob": config.behavior_solid_prob,
            "dataset_size": config.dataset_size,
            "gamma_eval": config.gamma_eval,
            "ridge": config.ridge,
            "n_outer_iters": config.n_outer_iters,
            "seeds": config.seeds,
        },
        "iqr_plot_path": str(iqr_path.resolve()),
        "ci95_plot_path": str(ci95_path.resolve()),
        "iqr_summary": iqr_summary,
        "ci95_summary": ci95_summary,
    }
    json_path = output_dir / "hard_longrun_ridge001_50seeds_summary.json"
    json_path.write_text(json.dumps(summary, indent=2))
    summary["json_path"] = str(json_path.resolve())
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate long-run hard-benchmark linear-FQE plots with many seeds.")
    parser.add_argument("--behavior-solid-prob", type=float, default=0.40)
    parser.add_argument("--dataset-size", type=int, default=2500)
    parser.add_argument("--gamma-eval", type=float, default=0.95)
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument("--n-outer-iters", type=int, default=100)
    parser.add_argument("--seeds", type=int, default=50)
    parser.add_argument("--output-dir", type=str, default="FQE_neurips/outputs/paper_figures")
    args = parser.parse_args()

    output = generate_hard_longrun_plots(
        HardLongRunPlotConfig(
            behavior_solid_prob=args.behavior_solid_prob,
            dataset_size=args.dataset_size,
            gamma_eval=args.gamma_eval,
            ridge=args.ridge,
            n_outer_iters=args.n_outer_iters,
            seeds=args.seeds,
            output_dir=args.output_dir,
        )
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
