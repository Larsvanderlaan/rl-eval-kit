from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass

import numpy as np

from .fqe import FQEConfig, fit_fqe_nn, predict_q_values
from .sw_fqe import fit_stationary_weighted_fqe
from .utils import (
    DiscreteMDP,
    TransitionBatch,
    evaluate_policy_tabular,
    induced_state_transition,
    sample_actions,
    sample_next_states,
    set_random_seed,
    state_action_one_hot,
    stationary_distribution,
)


@dataclass
class ExperimentSummary:
    gamma_eval: float
    gamma_ratio: float
    n_transitions: int
    rmse_unweighted: float
    rmse_closed_form: float
    rmse_extragradient: float
    closed_form_ratio_stats: dict
    extragradient_ratio_stats: dict


def build_baird_style_mdp(kappa: float, reward_hub: float = 1.0) -> DiscreteMDP:
    """A simple 7-state, 2-action Baird-style MDP used for the smoke experiment."""

    n_states, n_actions = 7, 2
    solid, dashed = 0, 1

    transition_prob = np.zeros((n_states, n_actions, n_states), dtype=np.float64)
    transition_prob[:, solid, 0] = 1.0
    transition_prob[:, dashed, 1:] = 1.0 / 6.0

    rewards = np.zeros((n_states, n_actions), dtype=np.float64)
    rewards[0, solid] = reward_hub

    target_policy = np.zeros((n_states, n_actions), dtype=np.float64)
    target_policy[:, solid] = 1.0

    behavior_policy = np.zeros((n_states, n_actions), dtype=np.float64)
    behavior_policy[:, solid] = kappa
    behavior_policy[:, dashed] = 1.0 - kappa

    return DiscreteMDP(
        transition_prob=transition_prob,
        rewards=rewards,
        target_policy=target_policy,
        behavior_policy=behavior_policy,
    )


def simulate_offline_batch(
    mdp: DiscreteMDP,
    n_transitions: int,
    seed: int,
    burn_in: int = 100,
) -> TransitionBatch:
    """Generate a behavior-policy trajectory and attach target next-actions."""

    rng = set_random_seed(seed)
    state = int(rng.integers(mdp.n_states))

    states = []
    actions = []
    rewards = []
    next_states = []

    total_steps = burn_in + n_transitions
    for t in range(total_steps):
        action = int(sample_actions(mdp.behavior_policy, np.array([state]), rng)[0])
        next_state = int(sample_next_states(mdp.transition_prob, np.array([state]), np.array([action]), rng)[0])
        reward = float(mdp.rewards[state, action])

        if t >= burn_in:
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)

        state = next_state

    states_arr = np.asarray(states, dtype=np.int64)
    next_states_arr = np.asarray(next_states, dtype=np.int64)
    next_actions_arr = sample_actions(mdp.target_policy, next_states_arr, rng)

    return TransitionBatch(
        states=states_arr,
        actions=np.asarray(actions, dtype=np.int64),
        rewards=np.asarray(rewards, dtype=np.float64),
        next_states=next_states_arr,
        next_actions=next_actions_arr,
    )


def _rmse(pred: np.ndarray, truth: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - truth) ** 2)))


def run_experiment(
    gamma_eval: float = 0.99,
    gamma_ratio: float = 0.995,
    n_transitions: int = 4_000,
    kappa: float = 0.1,
    seed: int = 0,
) -> ExperimentSummary:
    mdp = build_baird_style_mdp(kappa=kappa)
    batch = simulate_offline_batch(mdp=mdp, n_transitions=n_transitions, seed=seed)

    fqe_config = FQEConfig(
        gamma=gamma_eval,
        hidden_dims=(64, 64),
        n_outer_iters=25,
        epochs_per_iter=20,
        batch_size=256,
        learning_rate=5e-4,
        weight_decay=1e-4,
        grad_clip_norm=5.0,
        target_update_tau=0.1,
    )

    unweighted_fqe = fit_fqe_nn(batch=batch, n_states=mdp.n_states, n_actions=mdp.n_actions, weights=None, config=fqe_config, seed=seed)
    closed_form_sw = fit_stationary_weighted_fqe(
        batch=batch,
        n_states=mdp.n_states,
        n_actions=mdp.n_actions,
        ratio_model="linear",
        ratio_solver="closed_form",
        ratio_feature_map=lambda s, a: state_action_one_hot(s, a, mdp.n_states, mdp.n_actions),
        gamma_ratio=gamma_ratio,
        fqe_config=fqe_config,
        ratio_kwargs={
            "ridge_primal": 1e-4,
            "ridge_dual": 1e-4,
            "normalization_penalty": 10.0,
        },
        seed=seed,
    )
    extragrad_sw = fit_stationary_weighted_fqe(
        batch=batch,
        n_states=mdp.n_states,
        n_actions=mdp.n_actions,
        ratio_model="linear",
        ratio_solver="saddle",
        ratio_feature_map=lambda s, a: state_action_one_hot(s, a, mdp.n_states, mdp.n_actions),
        gamma_ratio=gamma_ratio,
        fqe_config=fqe_config,
        ratio_kwargs={
            "ridge_primal": 1e-4,
            "ridge_dual": 1e-4,
            "normalization_penalty": 10.0,
            "max_iters": 3_000,
        },
        seed=seed,
    )

    q_star = evaluate_policy_tabular(mdp, gamma=gamma_eval)
    grid_states = np.repeat(np.arange(mdp.n_states), mdp.n_actions)
    grid_actions = np.tile(np.arange(mdp.n_actions), mdp.n_states)
    q_star_flat = q_star.reshape(-1)

    q_hat_unweighted = predict_q_values(unweighted_fqe.model, grid_states, grid_actions, mdp.n_states, mdp.n_actions)
    q_hat_closed = predict_q_values(closed_form_sw.fqe_result.model, grid_states, grid_actions, mdp.n_states, mdp.n_actions)
    q_hat_extra = predict_q_values(extragrad_sw.fqe_result.model, grid_states, grid_actions, mdp.n_states, mdp.n_actions)

    return ExperimentSummary(
        gamma_eval=gamma_eval,
        gamma_ratio=gamma_ratio,
        n_transitions=n_transitions,
        rmse_unweighted=_rmse(q_hat_unweighted, q_star_flat),
        rmse_closed_form=_rmse(q_hat_closed, q_star_flat),
        rmse_extragradient=_rmse(q_hat_extra, q_star_flat),
        closed_form_ratio_stats=closed_form_sw.ratio_result.diagnostics if closed_form_sw.ratio_result is not None else {},
        extragradient_ratio_stats=extragrad_sw.ratio_result.diagnostics if extragrad_sw.ratio_result is not None else {},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a minimal stationary-weighted FQE experiment.")
    parser.add_argument("--gamma-eval", type=float, default=0.99)
    parser.add_argument("--gamma-ratio", type=float, default=0.995)
    parser.add_argument("--n-transitions", type=int, default=4000)
    parser.add_argument("--kappa", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    summary = run_experiment(
        gamma_eval=args.gamma_eval,
        gamma_ratio=args.gamma_ratio,
        n_transitions=args.n_transitions,
        kappa=args.kappa,
        seed=args.seed,
    )

    mdp = build_baird_style_mdp(kappa=args.kappa)
    behavior_chain = induced_state_transition(mdp, mdp.behavior_policy)
    target_chain = induced_state_transition(mdp, mdp.target_policy)
    behavior_stationary = stationary_distribution(behavior_chain)
    target_stationary = stationary_distribution(target_chain)

    output = {
        "summary": asdict(summary),
        "behavior_stationary_state_dist": behavior_stationary.tolist(),
        "target_stationary_state_dist": target_stationary.tolist(),
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
