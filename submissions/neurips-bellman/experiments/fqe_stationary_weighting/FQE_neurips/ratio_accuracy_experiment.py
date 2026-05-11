from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass

import numpy as np
import torch

from .neural_rkhs_weights import KernelConfig, NeuralRKHSWeightsConfig, estimate_ratio_neural_rkhs
from .ratio_estimation import (
    NeuralRatioConfig,
    estimate_ratio_closed_form_linear,
    positive_linear_ratio_weights,
    estimate_ratio_saddle_neural,
)
from .utils import (
    DiscreteMDP,
    exact_ratio_against_behavior,
    sample_actions,
    sample_next_states,
    set_random_seed,
    stationary_state_action_distribution,
    state_action_one_hot,
)


@dataclass
class RatioAccuracyMetrics:
    estimator: str
    gamma_ratio: float
    weighted_rmse: float
    unweighted_rmse: float
    max_abs_error: float
    corr: float


def build_random_ergodic_mdp(
    n_states: int,
    n_actions: int,
    seed: int,
    behavior_mix: float = 0.15,
    teleport: float = 0.05,
) -> DiscreteMDP:
    """Random small ergodic MDP with full-support target and behavior policies."""

    rng = set_random_seed(seed)
    transition_prob = np.zeros((n_states, n_actions, n_states), dtype=np.float64)
    for s in range(n_states):
        for a in range(n_actions):
            probs = rng.dirichlet(0.7 * np.ones(n_states))
            transition_prob[s, a] = (1.0 - teleport) * probs + teleport / n_states

    target_policy = np.vstack([rng.dirichlet(0.8 * np.ones(n_actions)) for _ in range(n_states)])
    behavior_raw = np.vstack([rng.dirichlet(0.8 * np.ones(n_actions)) for _ in range(n_states)])
    behavior_policy = behavior_mix * target_policy + (1.0 - behavior_mix) * behavior_raw
    behavior_policy = behavior_policy / behavior_policy.sum(axis=1, keepdims=True)

    rewards = np.zeros((n_states, n_actions), dtype=np.float64)
    return DiscreteMDP(
        transition_prob=transition_prob,
        rewards=rewards,
        target_policy=target_policy,
        behavior_policy=behavior_policy,
    )


def simulate_behavior_data(
    mdp: DiscreteMDP,
    n_samples: int,
    seed: int,
    burn_in: int = 1000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample a behavior-policy chain and attach target next-actions."""

    rng = set_random_seed(seed)
    state = int(rng.integers(mdp.n_states))
    states = []
    actions = []
    next_states = []

    for t in range(burn_in + n_samples):
        action = int(sample_actions(mdp.behavior_policy, np.array([state]), rng)[0])
        next_state = int(sample_next_states(mdp.transition_prob, np.array([state]), np.array([action]), rng)[0])
        if t >= burn_in:
            states.append(state)
            actions.append(action)
            next_states.append(next_state)
        state = next_state

    states = np.asarray(states, dtype=np.int64)
    actions = np.asarray(actions, dtype=np.int64)
    next_states = np.asarray(next_states, dtype=np.int64)
    next_actions = sample_actions(mdp.target_policy, next_states, rng)
    return states, actions, next_states, next_actions


def _evaluate_grid_weights(
    estimator_name: str,
    exact_ratio: np.ndarray,
    nu_b: np.ndarray,
    estimated_ratio: np.ndarray,
) -> RatioAccuracyMetrics:
    error = estimated_ratio - exact_ratio
    weighted_rmse = float(np.sqrt(np.sum(nu_b * error**2)))
    unweighted_rmse = float(np.sqrt(np.mean(error**2)))
    max_abs_error = float(np.max(np.abs(error)))
    corr = float(np.corrcoef(exact_ratio, estimated_ratio)[0, 1]) if np.std(estimated_ratio) > 0 else 0.0
    return RatioAccuracyMetrics(
        estimator=estimator_name,
        gamma_ratio=np.nan,
        weighted_rmse=weighted_rmse,
        unweighted_rmse=unweighted_rmse,
        max_abs_error=max_abs_error,
        corr=corr,
    )


def run_ratio_accuracy_experiment(
    n_states: int = 12,
    n_actions: int = 3,
    n_samples: int = 5_000,
    gamma_ratio: float = 0.99,
    seed: int = 0,
) -> dict:
    mdp = build_random_ergodic_mdp(n_states=n_states, n_actions=n_actions, seed=seed)
    states, actions, next_states, next_actions = simulate_behavior_data(mdp, n_samples=n_samples, seed=seed)

    phi = state_action_one_hot(states, actions, n_states=n_states, n_actions=n_actions)
    phi_next = state_action_one_hot(next_states, next_actions, n_states=n_states, n_actions=n_actions)

    exact_ratio = exact_ratio_against_behavior(mdp, gamma_ratio=gamma_ratio)
    nu_b = stationary_state_action_distribution(mdp, mdp.behavior_policy)

    linear_result = estimate_ratio_closed_form_linear(
        weight_features=phi,
        critic_features=phi,
        next_critic_features=phi_next,
        gamma_ratio=gamma_ratio,
        ridge_primal=1e-5,
        ridge_dual=1e-5,
        normalization_penalty=10.0,
    )

    neural_result = estimate_ratio_saddle_neural(
        weight_features=phi,
        critic_features=phi,
        next_critic_features=phi_next,
        gamma_ratio=gamma_ratio,
        config=NeuralRatioConfig(
            max_steps=2_000,
            batch_size=512,
            step_size=1e-3,
            ridge_weight=1e-4,
            ridge_critic=1e-4,
            normalization_penalty=10.0,
            valid_fraction=0.1,
            early_stopping_patience=15,
            log_every=100,
            uniform_mix=0.02,
        ),
    )
    neural_rkhs_result = estimate_ratio_neural_rkhs(
        weight_features=phi,
        critic_features=phi,
        next_critic_features=phi_next,
        gamma_ratio=gamma_ratio,
        config=NeuralRKHSWeightsConfig(
            max_steps=1500,
            learning_rate=1e-3,
            weight_decay=1e-4,
            critic_ridge=1e-4,
            normalization_penalty=10.0,
            valid_fraction=0.1,
            early_stopping_patience=10,
            log_every=100,
            uniform_mix=0.02,
            kernel=KernelConfig(kernel="rbf", bandwidth="median", max_anchors=256),
        ),
    )

    linear_grid_ratio = positive_linear_ratio_weights(
        alpha=linear_result.alpha,
        features=state_action_one_hot(
            np.repeat(np.arange(n_states), n_actions),
            np.tile(np.arange(n_actions), n_states),
            n_states=n_states,
            n_actions=n_actions,
        ),
    )
    with torch.no_grad():
        grid_phi = torch.tensor(
            state_action_one_hot(
                np.repeat(np.arange(n_states), n_actions),
                np.tile(np.arange(n_actions), n_states),
                n_states=n_states,
                n_actions=n_actions,
            ),
            dtype=torch.float32,
        )
        neural_grid_raw = neural_result.weight_model(grid_phi).cpu().numpy().astype(np.float64)
        neural_grid_ratio = neural_grid_raw / np.sum(neural_grid_raw * nu_b)
        neural_rkhs_grid_raw = neural_rkhs_result.weight_model(grid_phi).cpu().numpy().astype(np.float64)
        neural_rkhs_grid_ratio = neural_rkhs_grid_raw / np.sum(neural_rkhs_grid_raw * nu_b)

    linear_metrics = _evaluate_grid_weights("linear_closed_form", exact_ratio, nu_b, linear_grid_ratio)
    linear_metrics.gamma_ratio = gamma_ratio
    neural_metrics = _evaluate_grid_weights("neural_saddle", exact_ratio, nu_b, neural_grid_ratio)
    neural_metrics.gamma_ratio = gamma_ratio
    neural_rkhs_metrics = _evaluate_grid_weights("neural_rkhs", exact_ratio, nu_b, neural_rkhs_grid_ratio)
    neural_rkhs_metrics.gamma_ratio = gamma_ratio

    return {
        "config": {
            "n_states": n_states,
            "n_actions": n_actions,
            "n_samples": n_samples,
            "gamma_ratio": gamma_ratio,
            "seed": seed,
        },
        "linear_closed_form": asdict(linear_metrics),
        "neural_saddle": asdict(neural_metrics),
        "neural_rkhs": asdict(neural_rkhs_metrics),
        "linear_diagnostics": linear_result.diagnostics,
        "neural_diagnostics": neural_result.diagnostics,
        "neural_rkhs_diagnostics": neural_rkhs_result.diagnostics,
        "exact_ratio_summary": {
            "min": float(exact_ratio.min()),
            "max": float(exact_ratio.max()),
            "mean_under_behavior": float(np.sum(exact_ratio * nu_b)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ratio-estimation accuracy against exact discrete-MDP ratios.")
    parser.add_argument("--n-states", type=int, default=12)
    parser.add_argument("--n-actions", type=int, default=3)
    parser.add_argument("--n-samples", type=int, default=5000)
    parser.add_argument("--gamma-ratio", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    results = run_ratio_accuracy_experiment(
        n_states=args.n_states,
        n_actions=args.n_actions,
        n_samples=args.n_samples,
        gamma_ratio=args.gamma_ratio,
        seed=args.seed,
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
