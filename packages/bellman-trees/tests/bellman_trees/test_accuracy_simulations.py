from __future__ import annotations

import numpy as np
import pytest
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

from bellman_trees import BellmanLeafEnsembleRegressor, solve_projected_bellman
from bellman_trees._features import leaf_assignments_to_csr


def _discrete_chain(seed: int = 0):
    rng = np.random.default_rng(seed)
    gamma = 0.9
    n_states = 5
    n_actions = 2
    transition = np.zeros((n_states, n_actions, n_states), dtype=np.float64)
    for state in range(n_states):
        for action in range(n_actions):
            destination = min(n_states - 1, max(0, state + (1 if action == 1 else -1)))
            transition[state, action, destination] += 0.85
            transition[state, action, state] += 0.15

    rewards = np.array(
        [
            [(1.0 if state == n_states - 1 else -0.1 * abs(state - 3)) - 0.05 * action for action in range(n_actions)]
            for state in range(n_states)
        ],
        dtype=np.float64,
    )
    policy = np.tile(np.array([0.3, 0.7], dtype=np.float64), (n_states, 1))
    n_sa = n_states * n_actions
    transition_pi = np.zeros((n_sa, n_sa), dtype=np.float64)
    for state in range(n_states):
        for action in range(n_actions):
            row = state * n_actions + action
            for next_state in range(n_states):
                for next_action in range(n_actions):
                    col = next_state * n_actions + next_action
                    transition_pi[row, col] += transition[state, action, next_state] * policy[next_state, next_action]
    q_true = np.linalg.solve(np.eye(n_sa) - gamma * transition_pi, rewards.reshape(-1))
    v_true_by_state = np.sum(policy * q_true.reshape(n_states, n_actions), axis=1)

    n = 10_000
    states = rng.integers(0, n_states, size=n)
    actions = rng.integers(0, n_actions, size=n)
    next_states = np.array([rng.choice(n_states, p=transition[s, a]) for s, a in zip(states, actions)])
    next_actions = np.array([rng.choice(n_actions, p=policy[sp]) for sp in next_states])
    X = np.column_stack([states, actions]).astype(np.float64)
    X_next = np.column_stack([next_states, next_actions]).astype(np.float64)
    reward = rewards[states, actions]
    X_all = np.array([[s, a] for s in range(n_states) for a in range(n_actions)], dtype=np.float64)
    return gamma, policy, q_true, float(np.mean(v_true_by_state)), X, reward, X_next, X_all


def _gaussian_one_action(seed: int = 8):
    rng = np.random.default_rng(seed)
    gamma = 0.75
    rho = 0.65
    sigma = 0.25
    v2 = -1.0 / (1.0 - gamma * rho**2)
    v0 = (1.0 + gamma * v2 * sigma**2) / (1.0 - gamma)

    def q_true(states: np.ndarray) -> np.ndarray:
        s = np.asarray(states, dtype=np.float64).reshape(-1)
        return v0 + v2 * s**2

    n = 4_000
    states = rng.normal(0.0, 1.3, size=n)
    next_states = rho * states + sigma * rng.normal(size=n)
    reward = 1.0 - states**2
    X = states.reshape(-1, 1)
    X_next = next_states.reshape(-1, 1)
    eval_states = rng.normal(0.0, 1.3, size=2_000)
    initial_states = rng.normal(0.0, 1.0, size=2_000)
    return gamma, q_true, X, reward, X_next, eval_states.reshape(-1, 1), initial_states.reshape(-1, 1)


@pytest.mark.slow
def test_discrete_state_mdp_recovers_q_and_policy_value() -> None:
    gamma, policy, q_true, value_true, X, reward, X_next, X_all = _discrete_chain()
    current_ids = (X[:, 0].astype(int) * policy.shape[1] + X[:, 1].astype(int))
    next_ids = (X_next[:, 0].astype(int) * policy.shape[1] + X_next[:, 1].astype(int))
    tabular = solve_projected_bellman(
        leaf_assignments_to_csr(current_ids, n_features=q_true.size),
        leaf_assignments_to_csr(next_ids, n_features=q_true.size),
        reward,
        gamma=gamma,
        ridge=1e-8,
    )
    assert np.sqrt(np.mean((tabular.theta - q_true) ** 2)) < 0.10

    estimators = [
        (
            "tree",
            BellmanLeafEnsembleRegressor(
                DecisionTreeRegressor(max_leaf_nodes=10, min_samples_leaf=20, random_state=1),
                gamma=gamma,
                ridge=1e-6,
            ),
            0.08,
            0.05,
        ),
        (
            "random_forest",
            BellmanLeafEnsembleRegressor(
                RandomForestRegressor(n_estimators=30, max_depth=4, min_samples_leaf=20, random_state=2, n_jobs=1),
                gamma=gamma,
                ridge=1e-6,
            ),
            0.08,
            0.06,
        ),
        (
            "gradient_boosting",
            BellmanLeafEnsembleRegressor(
                GradientBoostingRegressor(n_estimators=30, max_depth=2, random_state=3),
                gamma=gamma,
                ridge=1e-6,
            ),
            0.35,
            0.25,
        ),
    ]
    for name, estimator, q_tol, value_tol in estimators:
        fitted = estimator.fit(X, reward, X_next)
        q_hat = fitted.predict(X_all)
        value_hat = float(np.mean(np.sum(policy * q_hat.reshape(policy.shape), axis=1)))
        assert np.sqrt(np.mean((q_hat - q_true) ** 2)) < q_tol, name
        assert abs(value_hat - value_true) < value_tol, name


@pytest.mark.slow
def test_continuous_gaussian_mdp_recovers_quadratic_q_and_value() -> None:
    gamma, q_true, X, reward, X_next, X_eval, X_initial = _gaussian_one_action()
    q_eval = q_true(X_eval)
    q_scale = float(np.std(q_eval))
    value_true = float(np.mean(q_true(X_initial)))
    estimators = [
        BellmanLeafEnsembleRegressor(
            DecisionTreeRegressor(max_leaf_nodes=64, min_samples_leaf=50, random_state=1),
            gamma=gamma,
            ridge=1e-5,
        ),
        BellmanLeafEnsembleRegressor(
            RandomForestRegressor(n_estimators=25, max_depth=6, min_samples_leaf=40, random_state=2, n_jobs=1),
            gamma=gamma,
            ridge=1e-5,
        ),
        BellmanLeafEnsembleRegressor(
            GradientBoostingRegressor(n_estimators=30, max_depth=2, min_samples_leaf=40, random_state=3),
            gamma=gamma,
            ridge=1e-5,
        ),
    ]
    for estimator in estimators:
        fitted = estimator.fit(X, reward, X_next)
        q_rmse = float(np.sqrt(np.mean((fitted.predict(X_eval) - q_eval) ** 2)))
        value_error = abs(float(np.mean(fitted.predict(X_initial))) - value_true)
        assert q_rmse / q_scale < 0.26
        assert value_error < 0.20
