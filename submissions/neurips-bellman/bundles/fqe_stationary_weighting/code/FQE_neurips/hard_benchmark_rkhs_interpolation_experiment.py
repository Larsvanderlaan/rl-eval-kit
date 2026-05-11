from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

import numpy as np

from .fqe import FQEConfig, fit_weighted_fqe_nn, predict_q_values
from .hub_spoke_latent_benchmark import HubSpokeLatentConfig, build_hub_spoke_latent_benchmark
from .latent_garnet_benchmark import simulate_behavior_trajectory
from .neural_rkhs_weights import KernelConfig, NeuralRKHSWeightsConfig, estimate_ratio_neural_rkhs
from .utils import evaluate_policy_tabular, interpolate_with_uniform, stabilize_weights, stationary_state_action_distribution


@dataclass
class HardBenchmarkRKHSInterpolationConfig:
    behavior_solid_prob: float = 0.30
    dataset_size: int = 1000
    gamma_eval: float = 0.97
    gamma_ratio: float = 1.0
    lambdas: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
    seeds: int = 5


def _flexible_neural_config(gamma_eval: float) -> FQEConfig:
    return FQEConfig(
        gamma=gamma_eval,
        hidden_dims=(256, 256),
        n_outer_iters=30,
        epochs_per_iter=20,
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


def _hard_rkhs_config(seed: int) -> NeuralRKHSWeightsConfig:
    """
    Tuned RKHS-weight configuration for the hard benchmark.

    The goal here is not maximal smoothing but a reasonably strong lambda=1
    endpoint so the interpolation path is informative.
    """

    return NeuralRKHSWeightsConfig(
        max_steps=600,
        learning_rate=1e-3,
        weight_decay=1e-4,
        critic_ridge=1e-4,
        normalization_penalty=10.0,
        valid_fraction=0.1,
        early_stopping_patience=8,
        uniform_mix=0.02,
        target_ess_fraction=0.4,
        max_uniform_mix=0.5,
        clip_quantile=0.995,
        max_weight=20.0,
        seed=seed,
        kernel=KernelConfig(kernel="rbf", bandwidth="median", max_anchors=128),
    )


def run_hard_benchmark_rkhs_interpolation(
    config: HardBenchmarkRKHSInterpolationConfig | None = None,
) -> dict[str, object]:
    if config is None:
        config = HardBenchmarkRKHSInterpolationConfig()

    summary_rows: list[dict[str, object]] = []
    for lam in config.lambdas:
        target_errors = []
        initial_errors = []
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
            x_raw = benchmark.featurize_state_actions(batch.states, batch.actions, feature_set="raw")
            x_raw_next = benchmark.featurize_state_actions(batch.next_states, batch.next_actions, feature_set="raw")
            rkhs = estimate_ratio_neural_rkhs(
                weight_features=x_raw,
                critic_features=x_raw,
                next_critic_features=x_raw_next,
                gamma_ratio=config.gamma_ratio,
                config=_hard_rkhs_config(seed),
            )
            weights = interpolate_with_uniform(rkhs.weights, lam)
            weights, _ = stabilize_weights(
                weights,
                uniform_mix=0.0,
                target_ess_fraction=None,
                clip_quantile=None,
                max_weight=None,
                return_metadata=True,
            )

            result = fit_weighted_fqe_nn(
                batch=batch,
                n_states=benchmark.config.n_states,
                n_actions=benchmark.config.n_actions,
                weights=weights,
                state_action_features=x_raw,
                next_state_action_features=x_raw_next,
                config=_flexible_neural_config(config.gamma_eval),
                seed=seed,
            )

            grid_states = np.repeat(np.arange(benchmark.config.n_states), benchmark.config.n_actions)
            grid_actions = np.tile(np.arange(benchmark.config.n_actions), benchmark.config.n_states)
            grid_features = benchmark.featurize_state_actions(grid_states, grid_actions, feature_set="raw")
            q_star = evaluate_policy_tabular(benchmark.mdp, gamma=config.gamma_eval).reshape(-1)
            mu_pi = stationary_state_action_distribution(benchmark.mdp, benchmark.mdp.target_policy)
            q_rms = float(np.sqrt(np.sum(mu_pi * q_star**2)))
            preds = predict_q_values(
                result.model,
                grid_states,
                grid_actions,
                benchmark.config.n_states,
                benchmark.config.n_actions,
                state_action_features=grid_features,
                device="cpu",
            )
            target_errors.append(float(np.sqrt(np.sum(mu_pi * (preds - q_star) ** 2)) / max(q_rms, 1e-12)))

            initial_dist = np.asarray(benchmark.initial_state_distribution, dtype=np.float64)
            q_hat = preds.reshape(benchmark.config.n_states, benchmark.config.n_actions)
            v_hat = (benchmark.mdp.target_policy * q_hat).sum(axis=1)
            v_star = (benchmark.mdp.target_policy * q_star.reshape(benchmark.config.n_states, benchmark.config.n_actions)).sum(axis=1)
            initial_errors.append(float(np.sqrt(initial_dist @ ((v_hat - v_star) ** 2))))

        summary_rows.append(
            {
                "lambda": float(lam),
                "median_relative_target_rmse": float(np.median(target_errors)),
                "median_initial_state_value_rmse": float(np.median(initial_errors)),
            }
        )

    return {
        "config": {
            "behavior_solid_prob": config.behavior_solid_prob,
            "dataset_size": config.dataset_size,
            "gamma_eval": config.gamma_eval,
            "gamma_ratio": config.gamma_ratio,
            "lambdas": list(config.lambdas),
            "seeds": config.seeds,
        },
        "flexible_neural_rkhs_interpolation": summary_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Interpolate RKHS weights for flexible neural FQE on the hard benchmark.")
    parser.add_argument("--behavior-solid-prob", type=float, default=0.30)
    parser.add_argument("--dataset-size", type=int, default=1000)
    parser.add_argument("--gamma-eval", type=float, default=0.97)
    parser.add_argument("--gamma-ratio", type=float, default=1.0)
    parser.add_argument("--lambdas", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--seeds", type=int, default=5)
    args = parser.parse_args()

    output = run_hard_benchmark_rkhs_interpolation(
        HardBenchmarkRKHSInterpolationConfig(
            behavior_solid_prob=args.behavior_solid_prob,
            dataset_size=args.dataset_size,
            gamma_eval=args.gamma_eval,
            gamma_ratio=args.gamma_ratio,
            lambdas=tuple(args.lambdas),
            seeds=args.seeds,
        )
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
