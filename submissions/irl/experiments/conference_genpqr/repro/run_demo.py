"""Example script for reproducing and extending the simulation pipeline."""

from __future__ import annotations

import os
import numpy as np

os.environ.setdefault("MPLCONFIGDIR", str((__import__("pathlib").Path(__file__).resolve().parent / ".mpl_cache")))
os.environ.setdefault("XDG_CACHE_HOME", str((__import__("pathlib").Path(__file__).resolve().parent / ".cache")))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from data_generation import (
    SimulationConfig,
    deterministic_transition_mean,
    generate_deeppqr_style_data,
    make_anchor_mu,
    make_constant_g,
    make_zero_g,
)
from policy_estimation import (
    fit_airl_policy,
    fit_behavior_cloning_policy,
    fit_maxent_irl_policy,
)
from q_evaluation import fit_fqe_boosted, fit_fqe_neural
from reward_recovery import recover_reward_and_continuation


def main() -> None:
    """Run one compact end-to-end experiment."""
    config = SimulationConfig(seed=123, horizon=15, n_actions=4, state_dim=4)

    # Default to the DeepPQR comparison setting: anchor-action normalization with g(s) = 0.
    mu = make_anchor_mu(anchor_action=0, n_actions=config.n_actions)
    g = make_zero_g()
    # Example generalized alternative:
    # mu = make_anchor_mu(anchor_action=0, n_actions=config.n_actions)
    # g = make_constant_g(1.0)

    data = generate_deeppqr_style_data(
        n_trajectories=120,
        config=config,
        mu=mu,
        g=g,
    )

    states = data["states"]
    actions = data["actions"]
    rewards = data["rewards"]
    next_states = data["next_states"]
    dones = data["dones"]

    maxent_policy = fit_maxent_irl_policy(states, actions, n_actions=config.n_actions)
    transition_model = lambda states, actions: deterministic_transition_mean(
        states=states,
        actions=actions,
        params=data["simulation_parameters"],
    )
    airl_policy = fit_airl_policy(
        states,
        actions,
        n_actions=config.n_actions,
        next_states=next_states,
        dones=dones,
        gamma=config.gamma,
        transition_model=transition_model,
        n_iters=10,
    )
    bc_policy = fit_behavior_cloning_policy(states, actions, n_actions=config.n_actions, n_epochs=25)

    normalization_policy = type(
        "NormalizationPolicy",
        (),
        {
            "predict_proba": staticmethod(mu),
            "sample_actions": staticmethod(lambda states, seed=0: np.zeros(states.shape[0], dtype=int)),
        },
    )()
    log_bc = np.log(np.clip(bc_policy.predict_proba(states)[np.arange(states.shape[0]), actions], 1e-8, 1.0))
    pseudo_rewards = log_bc - data["g_values"]

    neural_q = fit_fqe_neural(
        states=states,
        actions=actions,
        rewards=pseudo_rewards,
        next_states=next_states,
        dones=dones,
        policy=normalization_policy,
        n_actions=config.n_actions,
        gamma=config.gamma,
        n_fqe_iters=8,
        epochs_per_iter=4,
    )
    boosted_q = fit_fqe_boosted(
        states=states,
        actions=actions,
        rewards=pseudo_rewards,
        next_states=next_states,
        dones=dones,
        policy=normalization_policy,
        n_actions=config.n_actions,
        gamma=config.gamma,
        n_fqe_iters=4,
        n_estimators=20,
    )

    recovered = recover_reward_and_continuation(
        policy_estimate=bc_policy,
        q_estimate=neural_q,
        normalization_policy=mu,
        normalization_function=g,
        states=states,
        actions=actions,
        gamma=config.gamma,
    )

    print("Dataset size:", states.shape[0])
    print("MaxEnt policy avg entropy:", np.mean(-np.sum(maxent_policy.predict_proba(states) * np.log(np.clip(maxent_policy.predict_proba(states), 1e-8, None)), axis=1)))
    print("AIRL policy avg entropy:", np.mean(-np.sum(airl_policy.predict_proba(states) * np.log(np.clip(airl_policy.predict_proba(states), 1e-8, None)), axis=1)))
    print("BC policy avg entropy:", np.mean(-np.sum(bc_policy.predict_proba(states) * np.log(np.clip(bc_policy.predict_proba(states), 1e-8, None)), axis=1)))
    print("Recovered reward mean:", float(np.mean(recovered["reward"])))
    print("Recovered continuation mean:", float(np.mean(recovered["continuation_value"])))
    print("Boosted Q mean on observed actions:", float(np.mean(boosted_q.predict_q(states, actions))))


if __name__ == "__main__":
    main()
