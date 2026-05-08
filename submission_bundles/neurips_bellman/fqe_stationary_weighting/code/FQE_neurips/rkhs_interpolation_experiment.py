from __future__ import annotations

import argparse
import json

import numpy as np

from .fqe import FQEConfig, fit_weighted_fqe_nn, predict_q_values
from .fqe_linear import LinearFQEConfig, fit_weighted_linear_fqe, predict_linear_q_values
from .latent_garnet_benchmark import (
    LatentGarnetBenchmark,
    LatentGarnetConfig,
    build_latent_garnet_benchmark,
    estimate_weight_methods_on_benchmark,
)
from .utils import evaluate_policy_tabular, interpolate_with_uniform, stationary_state_action_distribution


def _default_small_fqe(gamma_eval: float) -> FQEConfig:
    return FQEConfig(
        gamma=gamma_eval,
        hidden_dims=(64, 64),
        n_outer_iters=30,
        epochs_per_iter=20,
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


def _default_flexible_fqe(gamma_eval: float) -> FQEConfig:
    return FQEConfig(
        gamma=gamma_eval,
        hidden_dims=(256, 256),
        n_outer_iters=35,
        epochs_per_iter=25,
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


def _default_linear_fqe(gamma_eval: float) -> LinearFQEConfig:
    return LinearFQEConfig(
        gamma=gamma_eval,
        ridge=5e-3,
        n_outer_iters=60,
        target_update_tau=0.35,
        valid_fraction=0.1,
        early_stopping_patience=8,
        min_improvement=1e-6,
        tol=1e-8,
        use_averaging=True,
        averaging_start_iter=5,
    )


def _summarize_nn(
    benchmark: LatentGarnetBenchmark,
    result,
    q_star: np.ndarray,
    mu_pi: np.ndarray,
    nu_b: np.ndarray,
    initial_state_dist: np.ndarray,
    state_action_features: np.ndarray,
    device: str,
) -> dict[str, float]:
    grid_states = np.repeat(np.arange(benchmark.config.n_states), benchmark.config.n_actions)
    grid_actions = np.tile(np.arange(benchmark.config.n_actions), benchmark.config.n_states)
    preds = predict_q_values(
        result.model,
        grid_states,
        grid_actions,
        benchmark.config.n_states,
        benchmark.config.n_actions,
        state_action_features=state_action_features,
        device=device,
    )
    err = preds - q_star
    q_hat = preds.reshape(benchmark.config.n_states, benchmark.config.n_actions)
    v_hat = (benchmark.mdp.target_policy * q_hat).sum(axis=1)
    v_star = (benchmark.mdp.target_policy * q_star.reshape(benchmark.config.n_states, benchmark.config.n_actions)).sum(axis=1)
    initial_value_estimate = float(initial_state_dist @ v_hat)
    initial_value_true = float(initial_state_dist @ v_star)
    return {
        "target_policy_rmse": float(np.sqrt(np.sum(mu_pi * err**2))),
        "behavior_policy_rmse": float(np.sqrt(np.sum(nu_b * err**2))),
        "initial_policy_value_abs_error": float(abs(initial_value_estimate - initial_value_true)),
        "initial_policy_value_error": float(initial_value_estimate - initial_value_true),
    }


def _summarize_linear(
    benchmark: LatentGarnetBenchmark,
    result,
    q_star: np.ndarray,
    mu_pi: np.ndarray,
    nu_b: np.ndarray,
    initial_state_dist: np.ndarray,
    state_action_features: np.ndarray,
) -> dict[str, float]:
    preds = predict_linear_q_values(result.theta, state_action_features)
    err = preds - q_star
    q_hat = preds.reshape(benchmark.config.n_states, benchmark.config.n_actions)
    v_hat = (benchmark.mdp.target_policy * q_hat).sum(axis=1)
    v_star = (benchmark.mdp.target_policy * q_star.reshape(benchmark.config.n_states, benchmark.config.n_actions)).sum(axis=1)
    initial_value_estimate = float(initial_state_dist @ v_hat)
    initial_value_true = float(initial_state_dist @ v_star)
    return {
        "target_policy_rmse": float(np.sqrt(np.sum(mu_pi * err**2))),
        "behavior_policy_rmse": float(np.sqrt(np.sum(nu_b * err**2))),
        "initial_policy_value_abs_error": float(abs(initial_value_estimate - initial_value_true)),
        "initial_policy_value_error": float(initial_value_estimate - initial_value_true),
    }


def run_rkhs_interpolation_experiment(
    benchmark: LatentGarnetBenchmark,
    lambdas: list[float],
    gamma_eval: float = 0.95,
    seed: int | None = None,
) -> dict[str, object]:
    used_seed = benchmark.config.seed if seed is None else seed
    artifacts = estimate_weight_methods_on_benchmark(benchmark, gamma_ratio=1.0, seed=used_seed)
    batch = artifacts["batch"]
    rkhs_weights = np.asarray(artifacts["neural_rkhs"].weights, dtype=np.float64)

    small_cfg = _default_small_fqe(gamma_eval)
    flex_cfg = _default_flexible_fqe(gamma_eval)
    linear_cfg = _default_linear_fqe(gamma_eval)

    x_raw = artifacts["state_action_features_raw"]
    x_next_raw = artifacts["next_state_action_features_raw"]
    x_linear = benchmark.featurize_state_actions(batch.states, batch.actions, feature_set="linear_q")
    x_linear_next = benchmark.featurize_state_actions(batch.next_states, batch.next_actions, feature_set="linear_q")

    grid_states = np.repeat(np.arange(benchmark.config.n_states), benchmark.config.n_actions)
    grid_actions = np.tile(np.arange(benchmark.config.n_actions), benchmark.config.n_states)
    grid_raw = benchmark.featurize_state_actions(grid_states, grid_actions, feature_set="raw")
    grid_linear = benchmark.featurize_state_actions(grid_states, grid_actions, feature_set="linear_q")

    q_star = evaluate_policy_tabular(benchmark.mdp, gamma=gamma_eval).reshape(-1)
    mu_pi = stationary_state_action_distribution(benchmark.mdp, benchmark.mdp.target_policy)
    nu_b = stationary_state_action_distribution(benchmark.mdp, benchmark.mdp.behavior_policy)
    initial_state_dist = np.asarray(benchmark.initial_state_distribution, dtype=np.float64)

    results: list[dict[str, object]] = []
    for lam in lambdas:
        weights = interpolate_with_uniform(rkhs_weights, lam)
        linear_result = fit_weighted_linear_fqe(
            batch=batch,
            state_action_features=x_linear,
            next_state_action_features=x_linear_next,
            weights=weights,
            config=linear_cfg,
            seed=used_seed,
        )
        nn_result = fit_weighted_fqe_nn(
            batch=batch,
            n_states=benchmark.config.n_states,
            n_actions=benchmark.config.n_actions,
            weights=weights,
            state_action_features=x_raw,
            next_state_action_features=x_next_raw,
            config=small_cfg,
            seed=used_seed,
        )
        flex_result = fit_weighted_fqe_nn(
            batch=batch,
            n_states=benchmark.config.n_states,
            n_actions=benchmark.config.n_actions,
            weights=weights,
            state_action_features=x_raw,
            next_state_action_features=x_next_raw,
            config=flex_cfg,
            seed=used_seed,
        )
        results.append(
            {
                "lambda": float(lam),
                "weight_summary": {
                    "min": float(weights.min()),
                    "max": float(weights.max()),
                    "std": float(weights.std()),
                    "ess_fraction": float((weights.sum() ** 2) / max(np.sum(weights**2), 1e-12) / len(weights)),
                },
                "linear_fqe": _summarize_linear(
                    benchmark, linear_result, q_star, mu_pi, nu_b, initial_state_dist, grid_linear
                ),
                "neural_fqe": _summarize_nn(
                    benchmark, nn_result, q_star, mu_pi, nu_b, initial_state_dist, grid_raw, small_cfg.device
                ),
                "neural_fqe_flexible": _summarize_nn(
                    benchmark, flex_result, q_star, mu_pi, nu_b, initial_state_dist, grid_raw, flex_cfg.device
                ),
            }
        )

    return {
        "config": {
            "n_states": benchmark.config.n_states,
            "n_actions": benchmark.config.n_actions,
            "dataset_size": benchmark.config.dataset_size,
            "behavior_coverage": benchmark.config.behavior_coverage,
            "gamma_eval": gamma_eval,
            "seed": used_seed,
        },
        "overlap_metrics": benchmark.stationary_overlap_metrics(),
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep interpolation between unweighted and RKHS stationary-ratio weighting.")
    parser.add_argument("--lambdas", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--n-states", type=int, default=50)
    parser.add_argument("--n-actions", type=int, default=4)
    parser.add_argument("--latent-dim", type=int, default=3)
    parser.add_argument("--branching-factor", type=int, default=5)
    parser.add_argument("--dataset-size", type=int, default=800)
    parser.add_argument("--behavior-coverage", type=float, default=0.5)
    parser.add_argument("--observation-mode", type=str, default="compact_nonlinear")
    parser.add_argument("--gamma-eval", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    benchmark = build_latent_garnet_benchmark(
        LatentGarnetConfig(
            n_states=args.n_states,
            n_actions=args.n_actions,
            latent_dim=args.latent_dim,
            branching_factor=args.branching_factor,
            dataset_size=args.dataset_size,
            behavior_coverage=args.behavior_coverage,
            observation_mode=args.observation_mode,
            seed=args.seed,
        )
    )
    output = run_rkhs_interpolation_experiment(
        benchmark=benchmark,
        lambdas=list(args.lambdas),
        gamma_eval=args.gamma_eval,
        seed=args.seed,
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
