from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from bellman_trees import BellmanHistogramGradientBoostingRegressor
from bellman_trees._hist_gbt import (
    _find_best_split,
    _fit_quantile_binner,
    _transform_bins,
    _xgb_split_gain,
)
from bellman_trees._hist_gbt_fast import HAS_NUMBA


def _toy_data(n: int = 400, p: int = 4, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    X_next = 0.6 * X + 0.1 * rng.normal(size=(n, p))
    reward = 1.0 + 0.5 * X[:, 0] - 0.2 * X[:, 1] ** 2 + 0.3 * (X[:, 2] > 0.0)
    return X, reward, X_next


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
    value_true = float(np.mean(np.sum(policy * q_true.reshape(n_states, n_actions), axis=1)))
    n = 10_000
    states = rng.integers(0, n_states, size=n)
    actions = rng.integers(0, n_actions, size=n)
    next_states = np.array([rng.choice(n_states, p=transition[s, a]) for s, a in zip(states, actions)])
    next_actions = np.array([rng.choice(n_actions, p=policy[sp]) for sp in next_states])
    X = np.column_stack([states, actions]).astype(np.float64)
    X_next = np.column_stack([next_states, next_actions]).astype(np.float64)
    reward = rewards[states, actions]
    X_all = np.array([[s, a] for s in range(n_states) for a in range(n_actions)], dtype=np.float64)
    return gamma, policy, q_true, value_true, X, reward, X_next, X_all


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


def test_quantile_binning_is_deterministic_with_missing_and_constant_columns() -> None:
    X = np.array(
        [
            [1.0, np.nan, 5.0],
            [1.0, 0.0, 5.0],
            [2.0, 0.0, 5.0],
            [3.0, 1.0, 5.0],
            [3.0, np.nan, 5.0],
        ]
    )
    binner = _fit_quantile_binner(X, max_bins=4)
    bins_a, missing_a = _transform_bins(X, binner)
    bins_b, missing_b = _transform_bins(X, binner)
    assert bins_a.dtype == np.uint16
    assert np.array_equal(bins_a, bins_b)
    assert np.array_equal(missing_a, missing_b)
    assert int(np.sum(missing_a[:, 1])) == 2
    assert binner.thresholds[2].size == 0
    assert np.all(bins_a[:, 2] == 0)


def test_histogram_split_gain_matches_bruteforce() -> None:
    X = np.array([0.0, 1.0, 2.0, 3.0]).reshape(-1, 1)
    binner = _fit_quantile_binner(X, max_bins=4)
    bins, missing = _transform_bins(X, binner)
    grad = np.array([-2.0, -1.0, 1.0, 2.0])
    hess = np.ones(4)
    split = _find_best_split(
        bins=bins,
        missing=missing,
        binner=binner,
        grad=grad,
        hess=hess,
        rows=np.arange(4, dtype=np.int64),
        feature_indices=np.array([0], dtype=np.int64),
        min_samples_leaf=1,
        min_child_weight=1.0,
        l2_leaf_reg=1.0,
        split_gamma=0.0,
    )
    assert split is not None
    manual = []
    for threshold in range(int(binner.n_bins_per_feature[0]) - 1):
        left = bins[:, 0] <= threshold
        manual.append(
            float(
                _xgb_split_gain(
                    grad[left].sum(),
                    hess[left].sum(),
                    grad[~left].sum(),
                    hess[~left].sum(),
                    grad.sum(),
                    hess.sum(),
                    l2_leaf_reg=1.0,
                    split_gamma=0.0,
                )
            )
        )
    assert np.isclose(split.gain, max(manual))


def test_missing_value_default_direction_chooses_higher_gain_route() -> None:
    X = np.array([0.0, 1.0, 2.0, 3.0, np.nan, np.nan]).reshape(-1, 1)
    binner = _fit_quantile_binner(X, max_bins=4)
    bins, missing = _transform_bins(X, binner)
    grad = np.array([-3.0, -3.0, 3.0, 3.0, -3.0, -3.0])
    hess = np.ones(6)
    split = _find_best_split(
        bins=bins,
        missing=missing,
        binner=binner,
        grad=grad,
        hess=hess,
        rows=np.arange(6, dtype=np.int64),
        feature_indices=np.array([0], dtype=np.int64),
        min_samples_leaf=1,
        min_child_weight=1.0,
        l2_leaf_reg=1.0,
        split_gamma=0.0,
    )
    assert split is not None
    assert split.default_left is True


def test_hist_gbt_apply_transform_and_persist(tmp_path) -> None:
    X, reward, X_next = _toy_data(n=500, p=5, seed=2)
    model = BellmanHistogramGradientBoostingRegressor(
        gamma=0.7,
        n_estimators=12,
        learning_rate=0.08,
        max_depth=2,
        max_bins=16,
        min_samples_leaf=20,
        early_stopping_rounds=3,
        validation_fraction=0.2,
        random_state=3,
    ).fit(X, reward, X_next)
    leaves = model.apply(X[:25])
    features = model.transform(X[:25])
    assert leaves.shape == (25, model.feature_info_["n_trees"])
    assert sparse.issparse(features)
    assert features.nnz == 25 * model.feature_info_["n_trees"]
    assert np.all(np.isfinite(model.predict(X[:25])))
    path = tmp_path / "hist_gbt.pkl"
    model.save(path)
    loaded = BellmanHistogramGradientBoostingRegressor.load(path)
    assert np.array_equal(leaves, loaded.apply(X[:25]))
    assert np.allclose(model.predict(X[:25]), loaded.predict(X[:25]))


def test_hist_gbt_iterative_solver_matches_direct_on_fixed_leaf_space() -> None:
    X, reward, X_next = _toy_data(n=450, p=4, seed=4)
    common = dict(
        gamma=0.5,
        n_estimators=10,
        learning_rate=0.1,
        max_depth=2,
        max_bins=16,
        min_samples_leaf=15,
        ridge=1e-6,
        random_state=5,
    )
    direct = BellmanHistogramGradientBoostingRegressor(**common, solver_method="direct").fit(X, reward, X_next)
    iterative = BellmanHistogramGradientBoostingRegressor(
        **common,
        solver_method="iterative",
        solver_max_iter=300,
        solver_tol=1e-10,
    ).fit(X, reward, X_next)
    assert iterative.solver_info_["converged"] is True
    assert np.allclose(direct.apply(X), iterative.apply(X))
    assert np.allclose(direct.predict(X[:80]), iterative.predict(X[:80]), atol=1e-5)


def test_hist_gbt_x_next_multiple_samples_are_averaged() -> None:
    X, reward, X_next = _toy_data(n=350, p=3, seed=6)
    X_next_multi = np.stack([X_next, X_next], axis=1)
    common = dict(
        gamma=0.6,
        n_estimators=8,
        learning_rate=0.1,
        max_depth=2,
        max_bins=12,
        min_samples_leaf=15,
        random_state=7,
    )
    two_dim = BellmanHistogramGradientBoostingRegressor(**common).fit(X, reward, X_next)
    three_dim = BellmanHistogramGradientBoostingRegressor(**common).fit(X, reward, X_next_multi)
    assert np.allclose(two_dim.predict(X[:50]), three_dim.predict(X[:50]))


def test_streaming_direct_matches_csr_direct_solution() -> None:
    X, reward, X_next = _toy_data(n=520, p=4, seed=12)
    common = dict(
        gamma=0.6,
        n_estimators=8,
        learning_rate=0.1,
        max_depth=2,
        max_bins=16,
        min_samples_leaf=20,
        ridge=1e-6,
        random_state=13,
    )
    csr = BellmanHistogramGradientBoostingRegressor(
        **common,
        feature_storage="csr",
        solver_method="direct",
    ).fit(X, reward, X_next)
    streaming = BellmanHistogramGradientBoostingRegressor(
        **common,
        feature_storage="streaming",
        solver_method="streaming_direct",
        batch_size=128,
    ).fit(X, reward, X_next)
    assert streaming.feature_info_["feature_storage"] == "streaming"
    assert streaming.solver_info_["method"] == "streaming_direct"
    assert np.allclose(csr.predict(X[:100]), streaming.predict(X[:100]), atol=1e-8)


def test_streaming_direct_averages_multiple_next_samples() -> None:
    X, reward, X_next = _toy_data(n=360, p=3, seed=14)
    X_next_multi = np.stack([X_next, X_next], axis=1)
    common = dict(
        gamma=0.6,
        n_estimators=8,
        learning_rate=0.1,
        max_depth=2,
        max_bins=12,
        min_samples_leaf=15,
        feature_storage="streaming",
        solver_method="streaming_direct",
        batch_size=96,
        random_state=15,
    )
    two_dim = BellmanHistogramGradientBoostingRegressor(**common).fit(X, reward, X_next)
    three_dim = BellmanHistogramGradientBoostingRegressor(**common).fit(X, reward, X_next_multi)
    assert np.allclose(two_dim.predict(X[:50]), three_dim.predict(X[:50]), atol=1e-8)


def test_hashed_streaming_features_are_deterministic_after_save_load(tmp_path) -> None:
    X, reward, X_next = _toy_data(n=420, p=5, seed=16)
    model = BellmanHistogramGradientBoostingRegressor(
        gamma=0.6,
        n_estimators=10,
        learning_rate=0.08,
        max_depth=2,
        max_bins=16,
        min_samples_leaf=15,
        feature_storage="hashed",
        hash_dim=128,
        solver_method="streaming_fqe",
        solver_max_iter=40,
        solver_tol=1e-3,
        ridge=1e-4,
        batch_size=128,
        random_state=17,
    ).fit(X, reward, X_next)
    assert model.feature_info_["feature_storage"] == "hashed"
    assert model.feature_info_["n_features_solver"] == 128
    assert np.all(np.isfinite(model.predict(X[:50])))
    path = tmp_path / "hashed_hist_gbt.pkl"
    model.save(path)
    loaded = BellmanHistogramGradientBoostingRegressor.load(path)
    assert np.array_equal(model.apply(X[:40]), loaded.apply(X[:40]))
    assert np.allclose(model.predict(X[:40]), loaded.predict(X[:40]))


@pytest.mark.skipif(not HAS_NUMBA, reason="numba optional extra is not installed")
def test_numba_backend_matches_numpy_backend_predictions() -> None:
    X, reward, X_next = _toy_data(n=360, p=4, seed=18)
    common = dict(
        gamma=0.5,
        n_estimators=6,
        learning_rate=0.1,
        max_depth=2,
        max_bins=12,
        min_samples_leaf=15,
        feature_storage="streaming",
        solver_method="streaming_direct",
        batch_size=96,
        random_state=19,
    )
    numpy_model = BellmanHistogramGradientBoostingRegressor(**common, backend="numpy").fit(X, reward, X_next)
    numba_model = BellmanHistogramGradientBoostingRegressor(**common, backend="numba").fit(X, reward, X_next)
    assert np.array_equal(numpy_model.apply(X[:50]), numba_model.apply(X[:50]))
    assert np.allclose(numpy_model.predict(X[:50]), numba_model.predict(X[:50]))


@pytest.mark.slow
def test_hist_gbt_discrete_mdp_recovers_q_and_policy_value() -> None:
    gamma, policy, q_true, value_true, X, reward, X_next, X_all = _discrete_chain()
    model = BellmanHistogramGradientBoostingRegressor(
        gamma=gamma,
        n_estimators=5,
        learning_rate=0.3,
        max_depth=4,
        max_leaves=10,
        max_bins=5,
        min_samples_leaf=30,
        min_child_weight=1.0,
        ridge=1e-6,
        random_state=8,
    ).fit(X, reward, X_next)
    q_hat = model.predict(X_all)
    value_hat = float(np.mean(np.sum(policy * q_hat.reshape(policy.shape), axis=1)))
    assert np.sqrt(np.mean((q_hat - q_true) ** 2)) < 0.20
    assert abs(value_hat - value_true) < 0.12


@pytest.mark.slow
def test_hist_gbt_continuous_gaussian_mdp_recovers_quadratic_value() -> None:
    gamma, q_true, X, reward, X_next, X_eval, X_initial = _gaussian_one_action()
    q_eval = q_true(X_eval)
    model = BellmanHistogramGradientBoostingRegressor(
        gamma=gamma,
        n_estimators=50,
        learning_rate=0.06,
        max_depth=2,
        max_bins=64,
        min_samples_leaf=40,
        min_child_weight=10.0,
        ridge=1e-5,
        random_state=9,
    ).fit(X, reward, X_next)
    q_rmse = float(np.sqrt(np.mean((model.predict(X_eval) - q_eval) ** 2)))
    value_true = float(np.mean(q_true(X_initial)))
    value_error = abs(float(np.mean(model.predict(X_initial))) - value_true)
    assert q_rmse / float(np.std(q_eval)) < 0.26
    assert value_error < 0.20


@pytest.mark.slow
def test_hist_gbt_runtime_smoke_uses_sparse_leaf_features() -> None:
    X, reward, X_next = _toy_data(n=10_000, p=8, seed=10)
    model = BellmanHistogramGradientBoostingRegressor(
        gamma=0.7,
        n_estimators=100,
        learning_rate=0.05,
        max_depth=1,
        max_bins=16,
        min_samples_leaf=100,
        min_child_weight=50.0,
        subsample=0.5,
        colsample_bytree=0.8,
        solver_method="auto",
        solver_max_iter=100,
        solver_tol=1e-6,
        random_state=11,
    ).fit(X, reward, X_next)
    features = model.transform(X[:200])
    assert model.feature_info_["n_trees"] == 100
    assert sparse.issparse(features)
    assert features.nnz == 200 * 100
    assert np.all(np.isfinite(model.predict(X[:200])))
