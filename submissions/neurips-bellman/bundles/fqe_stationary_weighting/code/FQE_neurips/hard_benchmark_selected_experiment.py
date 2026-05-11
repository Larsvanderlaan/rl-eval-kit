from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .fqe import FQEConfig, fit_weighted_fqe_nn, predict_q_values
from .fqe_linear import LinearFQEConfig, fit_weighted_linear_fqe, predict_linear_q_values
from .hard_benchmark_experiment import _estimate_screening_weights
from .hub_spoke_latent_benchmark import HubSpokeLatentConfig, build_hub_spoke_latent_benchmark
from .latent_garnet_benchmark import simulate_behavior_trajectory
from .utils import evaluate_policy_tabular, stationary_state_action_distribution


@dataclass
class SelectedHardExperimentConfig:
    behavior_solid_prob: float = 0.40
    dataset_size: int = 2500
    gamma_eval: float = 0.95
    linear_ridge: float = 1e-2
    linear_outer_iters: int = 100
    seeds: int = 10
    quick_neural: bool = False
    include_rkhs: bool = True
    output_dir: str = "FQE_neurips/outputs/paper_figures"


def _linear_config(config: SelectedHardExperimentConfig) -> LinearFQEConfig:
    return LinearFQEConfig(
        solver="iterative",
        gamma=config.gamma_eval,
        ridge=config.linear_ridge,
        n_outer_iters=config.linear_outer_iters,
        target_update_tau=1.0,
        valid_fraction=0.0,
        early_stopping_patience=None,
        min_improvement=1e-7,
        tol=1e-12,
        use_averaging=False,
        averaging_start_iter=5,
        initial_theta_mode="zero",
        selection_mode="last_iter",
        track_iterates=False,
        reduce_rank=True,
    )


def _small_neural_config(gamma_eval: float, quick: bool) -> FQEConfig:
    return FQEConfig(
        gamma=gamma_eval,
        hidden_dims=(64, 64),
        n_outer_iters=12 if quick else 30,
        epochs_per_iter=10 if quick else 20,
        batch_size=256,
        learning_rate=5e-4,
        weight_decay=5e-4,
        grad_clip_norm=5.0,
        target_update_tau=0.1,
        valid_fraction=0.1,
        early_stopping_patience=5,
        min_improvement=1e-5,
        device="cpu",
    )


def _flex_neural_config(gamma_eval: float, quick: bool) -> FQEConfig:
    return FQEConfig(
        gamma=gamma_eval,
        hidden_dims=(256, 256),
        n_outer_iters=15 if quick else 35,
        epochs_per_iter=12 if quick else 25,
        batch_size=256,
        learning_rate=5e-4,
        weight_decay=1e-4,
        grad_clip_norm=5.0,
        target_update_tau=0.1,
        valid_fraction=0.1,
        early_stopping_patience=6,
        min_improvement=1e-5,
        device="cpu",
    )


def _median(values: list[float]) -> float:
    return float(np.median(np.asarray(values, dtype=np.float64)))


def run_selected_hard_experiment(config: SelectedHardExperimentConfig | None = None) -> dict[str, object]:
    if config is None:
        config = SelectedHardExperimentConfig()

    per_seed: list[dict[str, object]] = []
    for seed in range(config.seeds):
        print(f"[selected-hard] seed {seed + 1}/{config.seeds}", flush=True)
        benchmark = build_hub_spoke_latent_benchmark(
            HubSpokeLatentConfig(
                behavior_solid_prob=config.behavior_solid_prob,
                dataset_size=config.dataset_size,
                reward_gamma=config.gamma_eval,
                seed=seed,
            )
        )
        batch = simulate_behavior_trajectory(benchmark, seed=seed)
        weights = _estimate_screening_weights(
            benchmark,
            batch,
            gamma_ratio=1.0,
            seed=seed,
            include_rkhs=config.include_rkhs,
        )

        x_linear = benchmark.featurize_state_actions(batch.states, batch.actions, feature_set="linear_q")
        x_linear_next = benchmark.featurize_state_actions(batch.next_states, batch.next_actions, feature_set="linear_q")
        x_raw = benchmark.featurize_state_actions(batch.states, batch.actions, feature_set="neural_structured")
        x_raw_next = benchmark.featurize_state_actions(batch.next_states, batch.next_actions, feature_set="neural_structured")

        grid_states = np.repeat(np.arange(benchmark.config.n_states), benchmark.config.n_actions)
        grid_actions = np.tile(np.arange(benchmark.config.n_actions), benchmark.config.n_states)
        grid_linear = benchmark.featurize_state_actions(grid_states, grid_actions, feature_set="linear_q")
        grid_raw = benchmark.featurize_state_actions(grid_states, grid_actions, feature_set="neural_structured")

        q_star = evaluate_policy_tabular(benchmark.mdp, gamma=config.gamma_eval).reshape(-1)
        mu_pi = stationary_state_action_distribution(benchmark.mdp, benchmark.mdp.target_policy)

        weight_map = {
            "unweighted": None,
            "policy_ratio": weights["policy_ratio_weights"],
            "oracle": weights["oracle_weights"],
            "learned_linear_basic": weights["linear_basic"].weights,
            "learned_linear_flexible": weights["linear_flexible"].weights,
        }
        if config.include_rkhs and "neural_rkhs" in weights:
            weight_map["learned_neural_rkhs"] = weights["neural_rkhs"].weights

        linear_results = {}
        for name, sample_weights in weight_map.items():
            result = fit_weighted_linear_fqe(
                batch=batch,
                state_action_features=x_linear,
                next_state_action_features=x_linear_next,
                weights=sample_weights,
                config=_linear_config(config),
                seed=seed,
            )
            preds = predict_linear_q_values(result.theta, grid_linear)
            linear_results[name] = float(np.sqrt(np.sum(mu_pi * (preds - q_star) ** 2)))

        small_neural_results = {}
        small_config = _small_neural_config(config.gamma_eval, quick=config.quick_neural)
        for name, sample_weights in weight_map.items():
            result = fit_weighted_fqe_nn(
                batch=batch,
                n_states=benchmark.config.n_states,
                n_actions=benchmark.config.n_actions,
                weights=sample_weights,
                state_action_features=x_raw,
                next_state_action_features=x_raw_next,
                config=small_config,
                seed=seed,
            )
            preds = predict_q_values(
                result.model,
                grid_states,
                grid_actions,
                benchmark.config.n_states,
                benchmark.config.n_actions,
                state_action_features=grid_raw,
                device="cpu",
            )
            small_neural_results[name] = float(np.sqrt(np.sum(mu_pi * (preds - q_star) ** 2)))

        flex_neural_results = {}
        flex_config = _flex_neural_config(config.gamma_eval, quick=config.quick_neural)
        for name, sample_weights in weight_map.items():
            result = fit_weighted_fqe_nn(
                batch=batch,
                n_states=benchmark.config.n_states,
                n_actions=benchmark.config.n_actions,
                weights=sample_weights,
                state_action_features=x_raw,
                next_state_action_features=x_raw_next,
                config=flex_config,
                seed=seed,
            )
            preds = predict_q_values(
                result.model,
                grid_states,
                grid_actions,
                benchmark.config.n_states,
                benchmark.config.n_actions,
                state_action_features=grid_raw,
                device="cpu",
            )
            flex_neural_results[name] = float(np.sqrt(np.sum(mu_pi * (preds - q_star) ** 2)))

        per_seed.append(
            {
                "seed": seed,
                "linear_target_rmse": linear_results,
                "small_neural_target_rmse": small_neural_results,
                "flex_neural_target_rmse": flex_neural_results,
            }
        )

    def summarize(key: str) -> dict[str, float]:
        methods = per_seed[0][key].keys()
        return {method: _median([row[key][method] for row in per_seed]) for method in methods}

    output = {
        "config": {
            "behavior_solid_prob": config.behavior_solid_prob,
            "dataset_size": config.dataset_size,
            "gamma_eval": config.gamma_eval,
            "linear_ridge": config.linear_ridge,
            "linear_outer_iters": config.linear_outer_iters,
            "seeds": config.seeds,
            "quick_neural": config.quick_neural,
            "include_rkhs": config.include_rkhs,
        },
        "median_linear_target_rmse": summarize("linear_target_rmse"),
        "median_small_neural_target_rmse": summarize("small_neural_target_rmse"),
        "median_flex_neural_target_rmse": summarize("flex_neural_target_rmse"),
        "per_seed": per_seed,
    }

    outdir = Path(config.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / "hard_selected_experiment.json"
    outpath.write_text(json.dumps(output, indent=2))
    output["json_path"] = str(outpath.resolve())
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the selected hard-benchmark experiment with linear and neural FQE.")
    parser.add_argument("--behavior-solid-prob", type=float, default=0.40)
    parser.add_argument("--dataset-size", type=int, default=2500)
    parser.add_argument("--gamma-eval", type=float, default=0.95)
    parser.add_argument("--linear-ridge", type=float, default=1e-2)
    parser.add_argument("--linear-outer-iters", type=int, default=100)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--quick-neural", action="store_true")
    parser.add_argument("--no-rkhs", action="store_true")
    parser.add_argument("--output-dir", type=str, default="FQE_neurips/outputs/paper_figures")
    args = parser.parse_args()

    output = run_selected_hard_experiment(
        SelectedHardExperimentConfig(
            behavior_solid_prob=args.behavior_solid_prob,
            dataset_size=args.dataset_size,
            gamma_eval=args.gamma_eval,
            linear_ridge=args.linear_ridge,
            linear_outer_iters=args.linear_outer_iters,
            seeds=args.seeds,
            quick_neural=args.quick_neural,
            include_rkhs=not args.no_rkhs,
            output_dir=args.output_dir,
        )
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
