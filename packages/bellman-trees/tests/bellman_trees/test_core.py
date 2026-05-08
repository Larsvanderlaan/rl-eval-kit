from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from bellman_trees import (
    BellmanAggregationForest,
    BellmanAggregationTree,
    BellmanLeafEnsembleRegressor,
    solve_projected_bellman,
    stabilize_weights,
)
from bellman_trees._features import leaf_assignments_to_csr


def _toy_data(n: int = 300, p: int = 3, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, size=(n, p))
    X_next = X.copy()
    reward = (X[:, 0] > 0.0).astype(float) + 0.1 * X[:, 1]
    weights = np.linspace(0.5, 2.0, n)
    return X, reward, X_next, weights


def _weighted_bellman_residual(est, X: np.ndarray, reward: np.ndarray, X_next: np.ndarray, weights: np.ndarray, gamma: float) -> float:
    pred = est.predict(X)
    next_pred = est.predict(X_next)
    residual = pred - (reward + gamma * next_pred)
    return float(np.sum(weights * residual**2) / np.sum(weights))


def test_weight_stabilization_normalizes_and_improves_ess() -> None:
    raw = np.array([1.0, 1.0, 1.0, 100.0])
    before = (raw.sum() ** 2) / np.sum(raw**2) / raw.size
    out = stabilize_weights(raw, max_weight=10.0, target_ess_fraction=0.8)
    after = (out.values.sum() ** 2) / np.sum(out.values**2) / out.values.size
    assert np.isclose(out.values.mean(), 1.0)
    assert after >= before
    assert out.diagnostics["ess_fraction_after_mix"] >= out.diagnostics["ess_fraction_before_mix"]


def test_projected_bellman_solver_recovers_tabular_value() -> None:
    phi = sparse.csr_matrix(np.eye(2))
    phi_next = sparse.csr_matrix(np.array([[0.0, 1.0], [0.0, 1.0]]))
    result = solve_projected_bellman(phi, phi_next, np.array([1.0, 2.0]), gamma=0.5, ridge=0.0)
    assert np.allclose(result.theta, np.array([3.0, 4.0]))


def test_iterative_fixed_feature_solver_matches_direct_solution() -> None:
    phi = sparse.csr_matrix(np.eye(2))
    phi_next = sparse.csr_matrix(np.array([[0.0, 1.0], [0.0, 1.0]]))
    direct = solve_projected_bellman(phi, phi_next, np.array([1.0, 2.0]), gamma=0.5, ridge=0.0)
    iterative = solve_projected_bellman(
        phi,
        phi_next,
        np.array([1.0, 2.0]),
        gamma=0.5,
        ridge=0.0,
        method="iterative",
        max_iter=200,
        tol=1e-10,
    )
    assert iterative.diagnostics["converged"] is True
    assert np.allclose(iterative.theta, direct.theta, atol=1e-8)


def test_rank_deficient_solver_preserves_duplicate_feature_predictions() -> None:
    phi_base = sparse.csr_matrix(np.eye(2))
    phi = sparse.hstack([phi_base, phi_base], format="csr")
    phi_next_base = sparse.csr_matrix(np.array([[0.0, 1.0], [0.0, 1.0]]))
    phi_next = sparse.hstack([phi_next_base, phi_next_base], format="csr")
    result = solve_projected_bellman(phi, phi_next, np.array([1.0, 2.0]), gamma=0.5, ridge=0.0)
    assert result.diagnostics["rank_deficient"] is True
    assert np.allclose(np.asarray(phi @ result.theta).reshape(-1), np.array([3.0, 4.0]))


def test_bat_leaf_aggregation_matches_explicit_solve() -> None:
    X, reward, X_next, weights = _toy_data(n=240)
    tree = BellmanAggregationTree(
        gamma=0.5,
        max_depth=1,
        max_leaves=2,
        max_bins=8,
        min_samples_leaf=20,
        min_leaf_ess=10,
        honest=False,
        random_state=7,
        ridge=0.0,
    ).fit(X, reward, X_next, weights)
    explicit = solve_projected_bellman(tree.transform(X), tree.transform_next(X_next), reward, weights, gamma=0.5, ridge=0.0)
    assert np.allclose(tree.theta_, explicit.theta)
    assert np.all(np.isfinite(tree.predict(X[:10])))


def test_x_next_multiple_samples_are_averaged_in_feature_space() -> None:
    X, reward, X_next, weights = _toy_data(n=200)
    X_next_multi = np.stack([X_next, X_next], axis=1)
    tree_2d = BellmanAggregationTree(
        gamma=0.7,
        max_depth=1,
        max_leaves=2,
        min_samples_leaf=20,
        min_leaf_ess=10,
        honest=False,
        random_state=3,
    ).fit(X, reward, X_next, weights)
    tree_3d = BellmanAggregationTree(
        gamma=0.7,
        max_depth=1,
        max_leaves=2,
        min_samples_leaf=20,
        min_leaf_ess=10,
        honest=False,
        random_state=3,
    ).fit(X, reward, X_next_multi, weights)
    assert np.allclose(tree_2d.predict(X[:20]), tree_3d.predict(X[:20]))


def test_tree_and_forest_fit_predict_and_persist(tmp_path) -> None:
    X, reward, X_next, weights = _toy_data(n=320, p=4, seed=11)
    tree = BellmanAggregationTree(
        gamma=0.8,
        max_depth=2,
        max_leaves=4,
        min_samples_leaf=20,
        min_leaf_ess=10,
        random_state=12,
    ).fit(X, reward, X_next, weights)
    forest = BellmanAggregationForest(
        gamma=0.8,
        n_estimators=5,
        max_depth=2,
        max_leaves=4,
        min_samples_leaf=20,
        min_leaf_ess=10,
        max_samples=0.7,
        random_state=13,
    ).fit(X, reward, X_next, weights)
    assert np.all(np.isfinite(tree.predict(X[:15])))
    assert np.all(np.isfinite(forest.predict(X[:15])))
    assert forest.feature_info_["joint_solve"] is True
    path = tmp_path / "tree.pkl"
    tree.save(path)
    loaded = BellmanAggregationTree.load(path)
    assert np.allclose(tree.predict(X[:10]), loaded.predict(X[:10]))


def test_baf_joint_solve_is_not_bagged_tree_average() -> None:
    X, reward, X_next, weights = _toy_data(n=420, p=5, seed=21)
    forest = BellmanAggregationForest(
        gamma=0.85,
        n_estimators=6,
        max_depth=2,
        max_leaves=4,
        min_samples_leaf=20,
        min_leaf_ess=10,
        max_samples=0.8,
        random_state=22,
    ).fit(X, reward, X_next, weights)
    joint = forest.predict(X[:50])
    bagged = np.mean(
        [
            tree.predict(X[:50, feat_idx])
            for tree, feat_idx in zip(forest.trees_, forest.tree_feature_indices_)
        ],
        axis=0,
    )
    assert not np.allclose(joint, bagged)


def test_bellman_aware_tree_beats_random_partition_residual() -> None:
    X, reward, X_next, weights = _toy_data(n=500, p=2, seed=31)
    gamma = 0.5
    tree = BellmanAggregationTree(
        gamma=gamma,
        max_depth=1,
        max_leaves=2,
        max_bins=16,
        min_samples_leaf=30,
        min_leaf_ess=20,
        honest=False,
        random_state=32,
    ).fit(X, reward, X_next, weights)
    tree_loss = _weighted_bellman_residual(tree, X, reward, X_next, weights, gamma)
    rng = np.random.default_rng(33)
    labels = rng.integers(0, 2, size=X.shape[0])
    phi = leaf_assignments_to_csr(labels, n_features=2)
    phi_next = phi.copy()
    random_solve = solve_projected_bellman(phi, phi_next, reward, weights, gamma=gamma)
    random_residual = np.asarray(phi @ random_solve.theta).reshape(-1) - (
        reward + gamma * np.asarray(phi_next @ random_solve.theta).reshape(-1)
    )
    random_loss = float(np.sum(weights * random_residual**2) / np.sum(weights))
    assert tree_loss <= random_loss


@pytest.mark.slow
def test_moderate_forest_runtime_smoke() -> None:
    X, reward, X_next, weights = _toy_data(n=10_000, p=8, seed=41)
    forest = BellmanAggregationForest(
        gamma=0.7,
        n_estimators=100,
        max_depth=1,
        max_leaves=2,
        max_bins=2,
        min_samples_leaf=100,
        min_leaf_ess=50,
        max_samples=0.2,
        max_features="sqrt",
        random_state=42,
    ).fit(X, reward, X_next, weights)
    assert forest.feature_info_["n_trees"] == 100
    assert np.all(np.isfinite(forest.predict(X[:25])))
