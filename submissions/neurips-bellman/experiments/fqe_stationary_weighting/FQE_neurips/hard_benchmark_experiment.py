from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from itertools import product

import numpy as np

from .fqe import FQEConfig, fit_weighted_fqe_nn, predict_q_values
from .fqe_linear import LinearFQEConfig, fit_weighted_linear_fqe, predict_linear_q_values
from .hub_spoke_latent_benchmark import HubSpokeLatentConfig, build_hub_spoke_latent_benchmark
from .latent_garnet_benchmark import simulate_behavior_trajectory
from .neural_rkhs_weights import KernelConfig, NeuralRKHSWeightsConfig, estimate_ratio_neural_rkhs
from .ratio_estimation import estimate_ratio_closed_form_linear
from .utils import (
    TransitionBatch,
    evaluate_policy_tabular,
    sample_actions,
    sample_next_states,
    set_random_seed,
    stabilize_weights,
    stationary_state_action_distribution,
)


@dataclass
class HardBenchmarkScreenConfig:
    """Screening configuration for the realistic-but-hard hub-spoke benchmark."""

    behavior_solid_grid: tuple[float, ...] = (0.30, 0.25, 0.20)
    dataset_size_grid: tuple[int, ...] = (1000, 1500)
    gamma_eval_grid: tuple[float, ...] = (0.97, 0.99)
    n_outer_iters_grid: tuple[int, ...] = (5, 6, 8)
    screen_seeds: int = 5
    final_seeds: int = 20
    min_realizability_rmse: float = 1e-8
    min_bellman_incompleteness_rmse: float = 0.02
    oracle_improvement_threshold: float = 0.20
    learned_fraction_threshold: float = 0.40
    neighbor_oracle_threshold: float = 0.15
    neighbor_learned_fraction_threshold: float = 0.30
    max_relative_target_rmse: float = 3.0
    max_relative_initial_value_error: float = 5.0
    max_curve_blowup_factor: float = 5.0
    max_final_over_best_curve_ratio: float = 1.25
    weighted_curve_advantage_threshold: float = 0.05
    oracle_primary_margin: float = 0.02
    seed_offset: int = 0
    include_rkhs_in_screen: bool = False


def _primary_linear_config(gamma_eval: float, feature_dim: int, n_outer_iters: int) -> LinearFQEConfig:
    return LinearFQEConfig(
        solver="iterative",
        gamma=gamma_eval,
        ridge=3e-4,
        n_outer_iters=n_outer_iters,
        target_update_tau=1.0,
        valid_fraction=0.1,
        early_stopping_patience=None,
        min_improvement=1e-7,
        tol=1e-10,
        use_averaging=False,
        averaging_start_iter=5,
        initial_theta_mode="zero",
        initial_theta=np.zeros(feature_dim, dtype=np.float64),
        selection_mode="last_iter",
        track_iterates=True,
    )


def _bad_init_linear_config(gamma_eval: float, feature_dim: int) -> LinearFQEConfig:
    theta0 = np.zeros(feature_dim, dtype=np.float64)
    theta0[-1] = 8.0
    return LinearFQEConfig(
        solver="iterative",
        gamma=gamma_eval,
        ridge=1e-4,
        n_outer_iters=120 if gamma_eval <= 0.97 else 180,
        target_update_tau=1.0,
        valid_fraction=0.1,
        early_stopping_patience=None,
        min_improvement=1e-7,
        tol=1e-10,
        use_averaging=False,
        averaging_start_iter=5,
        initial_theta_mode="zero",
        initial_theta=theta0,
        selection_mode="last_iter",
        track_iterates=True,
    )


def _primary_neural_config(gamma_eval: float) -> FQEConfig:
    return FQEConfig(
        gamma=gamma_eval,
        hidden_dims=(64, 64),
        n_outer_iters=25,
        epochs_per_iter=15,
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


def _estimate_screening_weights(benchmark, batch, gamma_ratio: float, seed: int, include_rkhs: bool) -> dict[str, object]:
    phi_basic = benchmark.featurize_state_actions(batch.states, batch.actions, feature_set="linear_basic")
    phi_next_basic = benchmark.featurize_state_actions(batch.next_states, batch.next_actions, feature_set="linear_basic")
    phi_flex = benchmark.featurize_state_actions(batch.states, batch.actions, feature_set="flexible_linear")
    phi_next_flex = benchmark.featurize_state_actions(batch.next_states, batch.next_actions, feature_set="flexible_linear")
    phi_raw = benchmark.featurize_state_actions(batch.states, batch.actions, feature_set="raw")
    phi_next_raw = benchmark.featurize_state_actions(batch.next_states, batch.next_actions, feature_set="raw")

    exact_ratio = benchmark.exact_state_action_ratio(gamma_ratio=gamma_ratio)
    batch_indices = batch.states * benchmark.config.n_actions + batch.actions
    oracle_weights, oracle_meta = stabilize_weights(exact_ratio[batch_indices], return_metadata=True)

    raw_policy_ratio = (
        benchmark.mdp.target_policy[batch.states, batch.actions]
        / np.maximum(benchmark.mdp.behavior_policy[batch.states, batch.actions], 1e-12)
    )
    policy_ratio_weights, policy_ratio_meta = stabilize_weights(raw_policy_ratio, return_metadata=True)

    linear_basic = estimate_ratio_closed_form_linear(
        weight_features=phi_basic,
        critic_features=phi_basic,
        next_critic_features=phi_next_basic,
        gamma_ratio=gamma_ratio,
        ridge_primal=1e-5,
        ridge_dual=1e-5,
        normalization_penalty=10.0,
    )
    linear_flexible = estimate_ratio_closed_form_linear(
        weight_features=phi_flex,
        critic_features=phi_flex,
        next_critic_features=phi_next_flex,
        gamma_ratio=gamma_ratio,
        ridge_primal=1e-5,
        ridge_dual=1e-5,
        normalization_penalty=10.0,
    )
    output = {
        "exact_ratio": exact_ratio,
        "oracle_weights": oracle_weights,
        "oracle_meta": oracle_meta,
        "policy_ratio_weights": policy_ratio_weights,
        "policy_ratio_meta": policy_ratio_meta,
        "linear_basic": linear_basic,
        "linear_flexible": linear_flexible,
    }
    if include_rkhs:
        output["neural_rkhs"] = estimate_ratio_neural_rkhs(
            weight_features=phi_raw,
            critic_features=phi_raw,
            next_critic_features=phi_next_raw,
            gamma_ratio=gamma_ratio,
            config=NeuralRKHSWeightsConfig(
                max_steps=900,
                learning_rate=1e-3,
                weight_decay=1e-4,
                critic_ridge=1e-4,
                normalization_penalty=10.0,
                valid_fraction=0.1,
                early_stopping_patience=8,
                uniform_mix=0.02,
                seed=seed,
                kernel=KernelConfig(kernel="rbf", bandwidth="median", max_anchors=256),
            ),
        )
    return output


def _simulate_stationary_iid_batch(benchmark, dataset_size: int, seed: int) -> TransitionBatch:
    rng = set_random_seed(seed)
    nu_b = stationary_state_action_distribution(benchmark.mdp, benchmark.mdp.behavior_policy)
    flat_index = rng.choice(nu_b.shape[0], size=dataset_size, p=nu_b)
    states = flat_index // benchmark.config.n_actions
    actions = flat_index % benchmark.config.n_actions
    rewards = benchmark.mdp.rewards[states, actions]
    next_states = sample_next_states(benchmark.mdp.transition_prob, states, actions, rng)
    next_actions = sample_actions(benchmark.mdp.target_policy, next_states, rng)
    return TransitionBatch(
        states=states,
        actions=actions,
        rewards=rewards,
        next_states=next_states,
        next_actions=next_actions,
    )


def _curve_rmse(theta_iterates: np.ndarray | None, grid_features: np.ndarray, q_star: np.ndarray, measure: np.ndarray) -> list[float]:
    if theta_iterates is None:
        return []
    curves = []
    for theta in theta_iterates:
        err = predict_linear_q_values(theta, grid_features) - q_star
        curves.append(float(np.sqrt(np.sum(measure * err**2))))
    return curves


def _curve_initial_abs_error(
    theta_iterates: np.ndarray | None,
    grid_features: np.ndarray,
    n_states: int,
    n_actions: int,
    target_policy: np.ndarray,
    initial_dist: np.ndarray,
    initial_value_true: float,
) -> list[float]:
    if theta_iterates is None:
        return []
    curves = []
    for theta in theta_iterates:
        preds = predict_linear_q_values(theta, grid_features).reshape(n_states, n_actions)
        v_hat = (target_policy * preds).sum(axis=1)
        curves.append(float(abs(initial_dist @ v_hat - initial_value_true)))
    return curves


def _curve_diagnostics(curve: list[float]) -> dict[str, float]:
    if not curve:
        return {
            "initial": float("nan"),
            "best": float("nan"),
            "final": float("nan"),
            "final_over_initial": float("nan"),
            "final_over_best": float("nan"),
            "best_improvement_fraction": float("nan"),
        }
    initial = float(curve[0])
    best = float(np.min(curve))
    final = float(curve[-1])
    return {
        "initial": initial,
        "best": best,
        "final": final,
        "final_over_initial": float(final / max(initial, 1e-12)),
        "final_over_best": float(final / max(best, 1e-12)),
        "best_improvement_fraction": float(max(initial - best, 0.0) / max(initial, 1e-12)),
    }


def _summarize_linear_result(
    result,
    grid_features: np.ndarray,
    q_star: np.ndarray,
    mu_pi: np.ndarray,
    nu_b: np.ndarray,
    initial_dist: np.ndarray,
    target_policy: np.ndarray,
    n_states: int,
    n_actions: int,
    initial_value_true: float,
    q_star_rms: float,
) -> dict[str, object]:
    preds = predict_linear_q_values(result.theta, grid_features)
    err = preds - q_star
    q_hat = preds.reshape(n_states, n_actions)
    v_hat = (target_policy * q_hat).sum(axis=1)
    mu_state = mu_pi.reshape(n_states, n_actions).sum(axis=1)
    nu_state = nu_b.reshape(n_states, n_actions).sum(axis=1)
    v_star = (target_policy * q_star.reshape(n_states, n_actions)).sum(axis=1)
    v_err = v_hat - v_star
    target_rmse = float(np.sqrt(np.sum(mu_pi * err**2)))
    behavior_rmse = float(np.sqrt(np.sum(nu_b * err**2)))
    initial_abs_error = float(abs(initial_dist @ v_hat - initial_value_true))
    target_curve = _curve_rmse(result.theta_iterates, grid_features, q_star, mu_pi)
    initial_curve = _curve_initial_abs_error(
        result.theta_iterates,
        grid_features,
        n_states,
        n_actions,
        target_policy,
        initial_dist,
        initial_value_true,
    )
    return {
        "stationary_q_rmse": target_rmse,
        "target_policy_rmse": target_rmse,
        "target_policy_relative_rmse": float(target_rmse / max(q_star_rms, 1e-12)),
        "behavior_q_rmse": behavior_rmse,
        "behavior_policy_rmse": behavior_rmse,
        "stationary_v_rmse": float(np.sqrt(np.sum(mu_state * v_err**2))),
        "behavior_v_rmse": float(np.sqrt(np.sum(nu_state * v_err**2))),
        "uniform_rmse": float(np.sqrt(np.mean(err**2))),
        "initial_policy_value_estimate": float(initial_dist @ v_hat),
        "initial_policy_value_true": float(initial_value_true),
        "initial_policy_value_abs_error": initial_abs_error,
        "initial_policy_value_relative_abs_error": float(initial_abs_error / max(abs(initial_value_true), 1e-12)),
        "selected_iteration": int(result.selected_iteration),
        "n_iterates": int(len(result.history["train_loss"])),
        "target_rmse_curve": target_curve,
        "initial_abs_error_curve": initial_curve,
        "target_curve_diagnostics": _curve_diagnostics(target_curve),
        "initial_curve_diagnostics": _curve_diagnostics(initial_curve),
    }


def _summarize_neural_result(
    result,
    grid_states: np.ndarray,
    grid_actions: np.ndarray,
    grid_features: np.ndarray,
    q_star: np.ndarray,
    mu_pi: np.ndarray,
    nu_b: np.ndarray,
    initial_dist: np.ndarray,
    target_policy: np.ndarray,
    n_states: int,
    n_actions: int,
    initial_value_true: float,
    q_star_rms: float,
    device: str,
) -> dict[str, object]:
    preds = predict_q_values(
        result.model,
        grid_states,
        grid_actions,
        n_states,
        n_actions,
        state_action_features=grid_features,
        device=device,
    )
    err = preds - q_star
    q_hat = preds.reshape(n_states, n_actions)
    v_hat = (target_policy * q_hat).sum(axis=1)
    mu_state = mu_pi.reshape(n_states, n_actions).sum(axis=1)
    nu_state = nu_b.reshape(n_states, n_actions).sum(axis=1)
    v_star = (target_policy * q_star.reshape(n_states, n_actions)).sum(axis=1)
    v_err = v_hat - v_star
    target_rmse = float(np.sqrt(np.sum(mu_pi * err**2)))
    behavior_rmse = float(np.sqrt(np.sum(nu_b * err**2)))
    initial_abs_error = float(abs(initial_dist @ v_hat - initial_value_true))
    return {
        "stationary_q_rmse": target_rmse,
        "target_policy_rmse": target_rmse,
        "target_policy_relative_rmse": float(target_rmse / max(q_star_rms, 1e-12)),
        "behavior_q_rmse": behavior_rmse,
        "behavior_policy_rmse": behavior_rmse,
        "stationary_v_rmse": float(np.sqrt(np.sum(mu_state * v_err**2))),
        "behavior_v_rmse": float(np.sqrt(np.sum(nu_state * v_err**2))),
        "uniform_rmse": float(np.sqrt(np.mean(err**2))),
        "initial_policy_value_estimate": float(initial_dist @ v_hat),
        "initial_policy_value_true": float(initial_value_true),
        "initial_policy_value_abs_error": initial_abs_error,
        "initial_policy_value_relative_abs_error": float(initial_abs_error / max(abs(initial_value_true), 1e-12)),
        "history_length": int(len(result.history["train_loss"])),
    }


def evaluate_hard_benchmark_setting(
    behavior_solid_prob: float,
    dataset_size: int,
    gamma_eval: float,
    n_outer_iters: int,
    seed: int,
    include_secondary: bool = False,
    include_rkhs: bool = False,
    linear_solver: str = "iterative",
    data_mode: str = "trajectory",
    include_neural: bool = False,
) -> dict[str, object]:
    benchmark = build_hub_spoke_latent_benchmark(
        HubSpokeLatentConfig(
            behavior_solid_prob=behavior_solid_prob,
            dataset_size=dataset_size,
            reward_gamma=gamma_eval,
            seed=seed,
        )
    )
    if data_mode == "trajectory":
        batch = simulate_behavior_trajectory(benchmark, seed=seed)
    elif data_mode == "stationary_iid":
        batch = _simulate_stationary_iid_batch(benchmark, dataset_size=dataset_size, seed=seed)
    else:
        raise ValueError(f"Unsupported hard-benchmark data_mode '{data_mode}'.")
    weights = _estimate_screening_weights(benchmark, batch, gamma_ratio=1.0, seed=seed, include_rkhs=include_rkhs)

    x_linear = benchmark.featurize_state_actions(batch.states, batch.actions, feature_set="linear_q")
    x_linear_next = benchmark.featurize_state_actions(batch.next_states, batch.next_actions, feature_set="linear_q")

    sample_weight_map = {
        "unweighted": None,
        "oracle": weights["oracle_weights"],
        "weighted_policy_ratio": weights["policy_ratio_weights"],
        "weighted_linear_basic": weights["linear_basic"].weights,
        "weighted_linear_flexible": weights["linear_flexible"].weights,
    }
    if include_rkhs:
        sample_weight_map["weighted_neural_rkhs"] = weights["neural_rkhs"].weights

    linear_config = _primary_linear_config(gamma_eval, x_linear.shape[1], n_outer_iters)
    linear_config.solver = linear_solver
    linear_results = {
        name: fit_weighted_linear_fqe(
            batch=batch,
            state_action_features=x_linear,
            next_state_action_features=x_linear_next,
            weights=sample_weights,
            config=linear_config,
            seed=seed,
        )
        for name, sample_weights in sample_weight_map.items()
    }
    neural_results = {}
    flexible_neural_results = {}
    if include_neural:
        x_neural = benchmark.featurize_state_actions(batch.states, batch.actions, feature_set="neural_structured")
        x_neural_next = benchmark.featurize_state_actions(batch.next_states, batch.next_actions, feature_set="neural_structured")
        primary_neural_config = _primary_neural_config(gamma_eval)
        flexible_neural_config = _flexible_neural_config(gamma_eval)
        neural_results = {
            name: fit_weighted_fqe_nn(
                batch=batch,
                n_states=benchmark.config.n_states,
                n_actions=benchmark.config.n_actions,
                weights=sample_weights,
                state_action_features=x_neural,
                next_state_action_features=x_neural_next,
                config=primary_neural_config,
                seed=seed,
            )
            for name, sample_weights in sample_weight_map.items()
        }
        flexible_neural_results = {
            name: fit_weighted_fqe_nn(
                batch=batch,
                n_states=benchmark.config.n_states,
                n_actions=benchmark.config.n_actions,
                weights=sample_weights,
                state_action_features=x_neural,
                next_state_action_features=x_neural_next,
                config=flexible_neural_config,
                seed=seed,
            )
            for name, sample_weights in sample_weight_map.items()
        }

    n_states = benchmark.config.n_states
    n_actions = benchmark.config.n_actions
    grid_states = np.repeat(np.arange(n_states), n_actions)
    grid_actions = np.tile(np.arange(n_actions), n_states)
    grid_features = benchmark.featurize_state_actions(grid_states, grid_actions, feature_set="linear_q")
    grid_features_neural = benchmark.featurize_state_actions(grid_states, grid_actions, feature_set="neural_structured")
    q_star = evaluate_policy_tabular(benchmark.mdp, gamma=gamma_eval).reshape(-1)
    q_star_rms = float(np.sqrt(np.sum(stationary_state_action_distribution(benchmark.mdp, benchmark.mdp.target_policy) * q_star**2)))
    mu_pi = stationary_state_action_distribution(benchmark.mdp, benchmark.mdp.target_policy)
    nu_b = stationary_state_action_distribution(benchmark.mdp, benchmark.mdp.behavior_policy)
    initial_dist = np.asarray(benchmark.initial_state_distribution, dtype=np.float64)
    v_star = (benchmark.mdp.target_policy * q_star.reshape(n_states, n_actions)).sum(axis=1)
    initial_value_true = float(initial_dist @ v_star)

    linear_metrics = {
        name: _summarize_linear_result(
            result=result,
            grid_features=grid_features,
            q_star=q_star,
            mu_pi=mu_pi,
            nu_b=nu_b,
            initial_dist=initial_dist,
            target_policy=benchmark.mdp.target_policy,
            n_states=n_states,
            n_actions=n_actions,
            initial_value_true=initial_value_true,
            q_star_rms=q_star_rms,
        )
        for name, result in linear_results.items()
    }

    learned_methods = ["weighted_linear_basic", "weighted_linear_flexible"]
    if include_rkhs:
        learned_methods.append("weighted_neural_rkhs")
    best_learned_method = min(learned_methods, key=lambda name: linear_metrics[name]["target_policy_rmse"])

    output = {
        "setting": {
            "behavior_solid_prob": behavior_solid_prob,
            "dataset_size": dataset_size,
            "gamma_eval": gamma_eval,
            "n_outer_iters": n_outer_iters,
            "seed": seed,
        },
        "overlap_metrics": benchmark.stationary_overlap_metrics(),
        "value_scale": {
            "target_q_rms": q_star_rms,
            "target_q_abs_mean": float(np.sum(mu_pi * np.abs(q_star))),
            "initial_policy_value_true": initial_value_true,
        },
        "linear_metrics": linear_metrics,
        "weight_diagnostics": {
            "oracle": weights["oracle_meta"],
            "policy_ratio": weights["policy_ratio_meta"],
            "linear_basic": weights["linear_basic"].diagnostics,
            "linear_flexible": weights["linear_flexible"].diagnostics,
        },
        "best_learned_method": best_learned_method,
    }
    if include_neural:
        output["neural_metrics"] = {
            name: _summarize_neural_result(
                result=result,
                grid_states=grid_states,
                grid_actions=grid_actions,
                grid_features=grid_features_neural,
                q_star=q_star,
                mu_pi=mu_pi,
                nu_b=nu_b,
                initial_dist=initial_dist,
                target_policy=benchmark.mdp.target_policy,
                n_states=n_states,
                n_actions=n_actions,
                initial_value_true=initial_value_true,
                q_star_rms=q_star_rms,
                device=_primary_neural_config(gamma_eval).device,
            )
            for name, result in neural_results.items()
        }
        output["neural_flexible_metrics"] = {
            name: _summarize_neural_result(
                result=result,
                grid_states=grid_states,
                grid_actions=grid_actions,
                grid_features=grid_features_neural,
                q_star=q_star,
                mu_pi=mu_pi,
                nu_b=nu_b,
                initial_dist=initial_dist,
                target_policy=benchmark.mdp.target_policy,
                n_states=n_states,
                n_actions=n_actions,
                initial_value_true=initial_value_true,
                q_star_rms=q_star_rms,
                device=_flexible_neural_config(gamma_eval).device,
            )
            for name, result in flexible_neural_results.items()
        }
    if include_rkhs:
        output["weight_diagnostics"]["neural_rkhs"] = weights["neural_rkhs"].diagnostics

    if include_secondary:
        bad_init_config = _bad_init_linear_config(gamma_eval, x_linear.shape[1])
        secondary_methods = ["unweighted", "oracle", "weighted_policy_ratio", best_learned_method]
        secondary_results = {
            name: fit_weighted_linear_fqe(
                batch=batch,
                state_action_features=x_linear,
                next_state_action_features=x_linear_next,
                weights=sample_weight_map[name],
                config=bad_init_config,
                seed=seed,
            )
            for name in secondary_methods
        }
        discounted_weights = _estimate_screening_weights(
            benchmark,
            batch,
            gamma_ratio=gamma_eval,
            seed=seed,
            include_rkhs=include_rkhs,
        )
        discounted_map = {
            "oracle_discounted": discounted_weights["oracle_weights"],
            "weighted_linear_basic_discounted": discounted_weights["linear_basic"].weights,
            "weighted_linear_flexible_discounted": discounted_weights["linear_flexible"].weights,
        }
        if include_rkhs:
            discounted_map["weighted_neural_rkhs_discounted"] = discounted_weights["neural_rkhs"].weights
        discounted_results = {
            name: fit_weighted_linear_fqe(
                batch=batch,
                state_action_features=x_linear,
                next_state_action_features=x_linear_next,
                weights=sample_weights,
                config=linear_config,
                seed=seed,
            )
            for name, sample_weights in discounted_map.items()
        }
        output["secondary"] = {
            "bad_init_linear_metrics": {
                name: _summarize_linear_result(
                    result=result,
                    grid_features=grid_features,
                    q_star=q_star,
                    mu_pi=mu_pi,
                    nu_b=nu_b,
                    initial_dist=initial_dist,
                    target_policy=benchmark.mdp.target_policy,
                    n_states=n_states,
                    n_actions=n_actions,
                    initial_value_true=initial_value_true,
                    q_star_rms=q_star_rms,
                )
                for name, result in secondary_results.items()
            },
            "discounted_ratio_linear_metrics": {
                name: _summarize_linear_result(
                    result=result,
                    grid_features=grid_features,
                    q_star=q_star,
                    mu_pi=mu_pi,
                    nu_b=nu_b,
                    initial_dist=initial_dist,
                    target_policy=benchmark.mdp.target_policy,
                    n_states=n_states,
                    n_actions=n_actions,
                    initial_value_true=initial_value_true,
                    q_star_rms=q_star_rms,
                )
                for name, result in discounted_results.items()
            },
        }

    return output


def _median(values: list[float]) -> float:
    return float(np.median(np.asarray(values, dtype=np.float64)))


def _aggregate_setting_results(
    setting_key: tuple[float, int, float, int],
    per_seed_results: list[dict[str, object]],
    config: HardBenchmarkScreenConfig,
) -> dict[str, object]:
    methods = list(per_seed_results[0]["linear_metrics"].keys())
    median_target = {
        method: _median([result["linear_metrics"][method]["target_policy_rmse"] for result in per_seed_results])
        for method in methods
    }
    median_relative_target = {
        method: _median([result["linear_metrics"][method]["target_policy_relative_rmse"] for result in per_seed_results])
        for method in methods
    }
    median_initial_abs = {
        method: _median([result["linear_metrics"][method]["initial_policy_value_abs_error"] for result in per_seed_results])
        for method in methods
    }
    median_relative_initial = {
        method: _median([result["linear_metrics"][method]["initial_policy_value_relative_abs_error"] for result in per_seed_results])
        for method in methods
    }
    best_learned_method = min(
        ["weighted_linear_basic", "weighted_linear_flexible"] + (["weighted_neural_rkhs"] if config.include_rkhs_in_screen else []),
        key=lambda method: median_relative_target[method],
    )
    unweighted = median_relative_target["unweighted"]
    oracle = median_relative_target["oracle"]
    best_learned = median_relative_target[best_learned_method]
    policy_ratio = median_relative_target["weighted_policy_ratio"]
    oracle_improvement = max(unweighted - oracle, 0.0) / max(unweighted, 1e-12)
    learned_improvement = max(unweighted - best_learned, 0.0) / max(unweighted, 1e-12)
    learned_fraction = learned_improvement / max(oracle_improvement, 1e-12)
    overlap_median = {
        key: _median([result["overlap_metrics"][key] for result in per_seed_results])
        for key in per_seed_results[0]["overlap_metrics"].keys()
    }
    value_scale_median = {
        key: _median([result["value_scale"][key] for result in per_seed_results])
        for key in per_seed_results[0]["value_scale"].keys()
    }
    oracle_curve = {
        key: _median([result["linear_metrics"]["oracle"]["target_curve_diagnostics"][key] for result in per_seed_results])
        for key in per_seed_results[0]["linear_metrics"]["oracle"]["target_curve_diagnostics"].keys()
    }
    best_learned_curve = {
        key: _median([result["linear_metrics"][best_learned_method]["target_curve_diagnostics"][key] for result in per_seed_results])
        for key in per_seed_results[0]["linear_metrics"][best_learned_method]["target_curve_diagnostics"].keys()
    }
    unweighted_curve = {
        key: _median([result["linear_metrics"]["unweighted"]["target_curve_diagnostics"][key] for result in per_seed_results])
        for key in per_seed_results[0]["linear_metrics"]["unweighted"]["target_curve_diagnostics"].keys()
    }
    oracle_stable = (
        oracle_curve["final_over_best"] <= config.max_final_over_best_curve_ratio
        and oracle_curve["final_over_initial"] <= config.max_curve_blowup_factor
    )
    learned_stable = (
        best_learned_curve["final_over_best"] <= config.max_final_over_best_curve_ratio
        and best_learned_curve["final_over_initial"] <= config.max_curve_blowup_factor
    )
    weighted_curve_advantage = (
        oracle_curve["best_improvement_fraction"] - unweighted_curve["best_improvement_fraction"]
    )
    oracle_is_best_stationary = oracle <= min(
        median_relative_target["weighted_policy_ratio"],
        min(median_relative_target[m] for m in ["weighted_linear_basic", "weighted_linear_flexible"] + (["weighted_neural_rkhs"] if config.include_rkhs_in_screen else [])),
    ) + config.oracle_primary_margin
    accepted = (
        overlap_median["linear_realizability_rmse"] <= config.min_realizability_rmse
        and overlap_median["bellman_incompleteness_rmse"] >= config.min_bellman_incompleteness_rmse
        and oracle_improvement >= config.oracle_improvement_threshold
        and learned_fraction >= config.learned_fraction_threshold
        and policy_ratio > oracle
        and median_relative_target["oracle"] <= config.max_relative_target_rmse
        and median_relative_target[best_learned_method] <= config.max_relative_target_rmse
        and median_relative_initial["oracle"] <= config.max_relative_initial_value_error
        and median_relative_initial[best_learned_method] <= config.max_relative_initial_value_error
        and oracle_stable
        and learned_stable
        and weighted_curve_advantage >= config.weighted_curve_advantage_threshold
    )

    key_methods = ["unweighted", "oracle", "weighted_policy_ratio", best_learned_method]
    curve_summary = {}
    for method in key_methods:
        target_curves = [result["linear_metrics"][method]["target_rmse_curve"] for result in per_seed_results]
        initial_curves = [result["linear_metrics"][method]["initial_abs_error_curve"] for result in per_seed_results]
        min_len = min((len(curve) for curve in target_curves), default=0)
        if min_len == 0:
            continue
        curve_summary[method] = {
            "target_rmse_curve_median": np.median(np.asarray([curve[:min_len] for curve in target_curves]), axis=0).tolist(),
            "initial_abs_error_curve_median": np.median(np.asarray([curve[:min_len] for curve in initial_curves]), axis=0).tolist(),
        }

    return {
        "setting": {
            "behavior_solid_prob": float(setting_key[0]),
            "dataset_size": int(setting_key[1]),
            "gamma_eval": float(setting_key[2]),
            "n_outer_iters": int(setting_key[3]),
        },
        "screen_seeds": len(per_seed_results),
        "median_overlap_metrics": overlap_median,
        "median_value_scale": value_scale_median,
        "median_linear_target_rmse": median_target,
        "median_linear_relative_target_rmse": median_relative_target,
        "median_linear_initial_abs_error": median_initial_abs,
        "median_linear_relative_initial_abs_error": median_relative_initial,
        "oracle_improvement_fraction": float(oracle_improvement),
        "best_learned_method": best_learned_method,
        "best_learned_improvement_fraction": float(learned_improvement),
        "learned_fraction_of_oracle_gain": float(learned_fraction),
        "policy_ratio_weaker_than_stationary": bool(policy_ratio > oracle),
        "oracle_primary_tuning_pass": bool(oracle_is_best_stationary),
        "oracle_curve_stable": bool(oracle_stable),
        "best_learned_curve_stable": bool(learned_stable),
        "weighted_curve_advantage": float(weighted_curve_advantage),
        "accepted_primary": bool(accepted),
        "curve_summary": curve_summary,
        "per_seed_results": per_seed_results,
    }


def _neighbor_indices(items: list[dict[str, object]], idx: int) -> list[int]:
    current = items[idx]["setting"]
    neighbors: list[int] = []
    for j, item in enumerate(items):
        if j == idx:
            continue
        setting = item["setting"]
        same_gamma = setting["gamma_eval"] == current["gamma_eval"]
        same_iters = setting["n_outer_iters"] == current["n_outer_iters"]
        adjacent_behavior = setting["dataset_size"] == current["dataset_size"] and abs(
            setting["behavior_solid_prob"] - current["behavior_solid_prob"]
        ) < 0.11
        adjacent_dataset = setting["behavior_solid_prob"] == current["behavior_solid_prob"] and abs(
            setting["dataset_size"] - current["dataset_size"]
        ) <= 500
        if same_gamma and same_iters and (adjacent_behavior or adjacent_dataset):
            neighbors.append(j)
    return neighbors


def screen_hard_benchmark(config: HardBenchmarkScreenConfig | None = None) -> dict[str, object]:
    if config is None:
        config = HardBenchmarkScreenConfig()

    aggregated = []
    for behavior_solid_prob, dataset_size, gamma_eval, n_outer_iters in product(
        config.behavior_solid_grid,
        config.dataset_size_grid,
        config.gamma_eval_grid,
        config.n_outer_iters_grid,
    ):
        per_seed_results = []
        for offset in range(config.screen_seeds):
            seed = config.seed_offset + offset
            per_seed_results.append(
                evaluate_hard_benchmark_setting(
                    behavior_solid_prob=behavior_solid_prob,
                    dataset_size=dataset_size,
                    gamma_eval=gamma_eval,
                    n_outer_iters=n_outer_iters,
                    seed=seed,
                    include_secondary=False,
                    include_rkhs=config.include_rkhs_in_screen,
                )
            )
        aggregated.append(
            _aggregate_setting_results(
                setting_key=(behavior_solid_prob, dataset_size, gamma_eval, n_outer_iters),
                per_seed_results=per_seed_results,
                config=config,
            )
        )

    for idx, item in enumerate(aggregated):
        neighbors = _neighbor_indices(aggregated, idx)
        neighbor_supported = any(
            aggregated[n]["oracle_improvement_fraction"] >= config.neighbor_oracle_threshold
            and aggregated[n]["learned_fraction_of_oracle_gain"] >= config.neighbor_learned_fraction_threshold
            and aggregated[n]["policy_ratio_weaker_than_stationary"]
            and aggregated[n]["oracle_curve_stable"]
            and aggregated[n]["best_learned_curve_stable"]
            for n in neighbors
        )
        item["neighbor_supported"] = bool(neighbor_supported)
        item["accepted_with_neighbor_check"] = bool(item["accepted_primary"] and neighbor_supported)

    selected = next((item for item in aggregated if item["accepted_with_neighbor_check"]), None)
    if selected is None:
        selected = next((item for item in aggregated if item["accepted_primary"]), None)
    if selected is None:
        selected = max(
            aggregated,
            key=lambda item: (
                item["oracle_improvement_fraction"],
                item["accepted_primary"],
                item["learned_fraction_of_oracle_gain"],
                -item["median_linear_relative_target_rmse"]["oracle"],
            ),
        )

    selected_setting = selected["setting"]
    selected_secondary = evaluate_hard_benchmark_setting(
        behavior_solid_prob=selected_setting["behavior_solid_prob"],
        dataset_size=selected_setting["dataset_size"],
        gamma_eval=selected_setting["gamma_eval"],
        n_outer_iters=selected_setting["n_outer_iters"],
        seed=config.seed_offset,
        include_secondary=True,
        include_rkhs=config.include_rkhs_in_screen,
    )

    return {
        "config": {
            "behavior_solid_grid": list(config.behavior_solid_grid),
            "dataset_size_grid": list(config.dataset_size_grid),
            "gamma_eval_grid": list(config.gamma_eval_grid),
            "n_outer_iters_grid": list(config.n_outer_iters_grid),
            "screen_seeds": config.screen_seeds,
            "final_seeds": config.final_seeds,
            "min_realizability_rmse": config.min_realizability_rmse,
            "min_bellman_incompleteness_rmse": config.min_bellman_incompleteness_rmse,
            "oracle_improvement_threshold": config.oracle_improvement_threshold,
            "learned_fraction_threshold": config.learned_fraction_threshold,
            "max_relative_target_rmse": config.max_relative_target_rmse,
            "max_relative_initial_value_error": config.max_relative_initial_value_error,
            "max_curve_blowup_factor": config.max_curve_blowup_factor,
            "max_final_over_best_curve_ratio": config.max_final_over_best_curve_ratio,
            "weighted_curve_advantage_threshold": config.weighted_curve_advantage_threshold,
            "oracle_primary_margin": config.oracle_primary_margin,
            "include_rkhs_in_screen": config.include_rkhs_in_screen,
        },
        "screened_settings": [
            {
                key: value
                for key, value in item.items()
                if key != "per_seed_results"
            }
            for item in aggregated
        ],
        "selected_setting": {
            key: value
            for key, value in selected.items()
            if key != "per_seed_results"
        },
        "selected_secondary_diagnostics": selected_secondary["secondary"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Screen and summarize the realistic hard hub-spoke benchmark.")
    parser.add_argument("--behavior-solid-grid", type=float, nargs="+", default=[0.30, 0.25, 0.20])
    parser.add_argument("--dataset-size-grid", type=int, nargs="+", default=[1000, 1500])
    parser.add_argument("--gamma-eval-grid", type=float, nargs="+", default=[0.97, 0.99])
    parser.add_argument("--n-outer-iters-grid", type=int, nargs="+", default=[5, 6, 8])
    parser.add_argument("--screen-seeds", type=int, default=5)
    parser.add_argument("--final-seeds", type=int, default=20)
    parser.add_argument("--seed-offset", type=int, default=0)
    args = parser.parse_args()

    output = screen_hard_benchmark(
        HardBenchmarkScreenConfig(
            behavior_solid_grid=tuple(args.behavior_solid_grid),
            dataset_size_grid=tuple(args.dataset_size_grid),
            gamma_eval_grid=tuple(args.gamma_eval_grid),
            n_outer_iters_grid=tuple(args.n_outer_iters_grid),
            screen_seeds=args.screen_seeds,
            final_seeds=args.final_seeds,
            seed_offset=args.seed_offset,
        )
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
