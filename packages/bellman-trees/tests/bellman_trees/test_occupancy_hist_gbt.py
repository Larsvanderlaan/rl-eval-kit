from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from bellman_trees import (
    DiscountedOccupancyHistogramGradientBoostingRatioEstimator,
    solve_discounted_occupancy_ratio,
)
from bellman_trees._occupancy_hist_gbt import _build_signed_flow_events


def test_signed_flow_events_match_discounted_balance_masses_for_2d_and_3d_next() -> None:
    X = np.array([[0.0], [1.0]])
    X_next = np.array([[1.0], [1.0]])
    X_next_multi = np.stack([X_next, X_next], axis=1)
    X_initial = np.array([[0.0]])
    ratio = np.array([2.0, 1.0])
    weights = np.array([1.0, 3.0])
    initial_weight = np.array([5.0])

    two_dim = _build_signed_flow_events(
        X,
        X_next,
        X_initial,
        ratio,
        weights,
        initial_weight,
        gamma=0.5,
    )
    three_dim = _build_signed_flow_events(
        X,
        X_next_multi,
        X_initial,
        ratio,
        weights,
        initial_weight,
        gamma=0.5,
    )

    assert two_dim.X.shape == (5, 1)
    assert np.allclose(two_dim.signed_mass[:2], [0.5, 0.75])
    assert np.allclose(two_dim.signed_mass[2:4], [-0.25, -0.375])
    assert np.allclose(two_dim.signed_mass[4:], [-0.5])
    assert three_dim.X.shape == (7, 1)
    assert np.isclose(np.sum(three_dim.signed_mass[:2]), np.sum(two_dim.signed_mass[:2]))
    assert np.isclose(np.sum(three_dim.signed_mass[2:6]), np.sum(two_dim.signed_mass[2:4]))
    assert np.isclose(np.sum(three_dim.signed_mass[6:]), np.sum(two_dim.signed_mass[4:]))
    assert np.allclose(three_dim.signed_mass[2:6], [-0.125, -0.125, -0.1875, -0.1875])


def test_boosted_occupancy_ratio_recovers_tabular_chain() -> None:
    gamma = 0.5
    counts = np.array([600, 300, 100])
    states = np.repeat(np.arange(3), counts)
    X = states.reshape(-1, 1).astype(np.float64)
    X_next = np.minimum(states + 1, 2).reshape(-1, 1).astype(np.float64)
    X_initial = np.zeros((500, 1), dtype=np.float64)
    behavior_mass = counts / counts.sum()
    true_mass = np.array([1.0 - gamma, (1.0 - gamma) * gamma, gamma**2])
    true_ratio = true_mass / behavior_mass

    model = DiscountedOccupancyHistogramGradientBoostingRatioEstimator(
        gamma=gamma,
        n_estimators=3,
        learning_rate=0.3,
        max_depth=2,
        max_leaves=3,
        max_bins=3,
        min_samples_leaf=20,
        min_child_weight=1e-8,
        l2_leaf_reg=1.0,
        hessian_floor=1e-8,
        ridge=1e-10,
        solver="lsq_linear",
        weight_clip_quantile=None,
        random_state=1,
    ).fit(X, X_next, X_initial)

    ratio = model.predict_ratio(np.array([[0.0], [1.0], [2.0]]))
    assert np.all(ratio >= -1e-10)
    assert np.allclose(ratio, true_ratio, atol=2e-3)
    assert model.solver_info_["nonnegative"] is True


def test_boosted_splits_select_flow_structure_not_balanced_nuisance() -> None:
    gamma = 0.5
    nuisance = np.linspace(-1.0, 1.0, 12)
    per_level_counts = np.array([180, 90, 30])
    rows = []
    next_rows = []
    initial_rows = []
    for value in nuisance:
        states = np.repeat(np.arange(3), per_level_counts)
        rows.append(np.column_stack([states, np.full(states.size, value)]))
        next_rows.append(np.column_stack([np.minimum(states + 1, 2), np.full(states.size, value)]))
        initial_rows.append(np.column_stack([np.zeros(80), np.full(80, value)]))
    X = np.concatenate(rows, axis=0).astype(np.float64)
    X_next = np.concatenate(next_rows, axis=0).astype(np.float64)
    X_initial = np.concatenate(initial_rows, axis=0).astype(np.float64)

    model = DiscountedOccupancyHistogramGradientBoostingRatioEstimator(
        gamma=gamma,
        n_estimators=1,
        learning_rate=0.3,
        max_depth=2,
        max_leaves=3,
        max_bins=8,
        min_samples_leaf=50,
        min_child_weight=1e-8,
        l2_leaf_reg=1.0,
        ridge=1e-10,
        solver="lsq_linear",
        weight_clip_quantile=None,
        random_state=4,
    ).fit(X, X_next, X_initial)

    first_tree_splits = model.split_history_[0]["splits"]
    assert len(first_tree_splits) == 2
    assert [row["feature"] for row in first_tree_splits] == [0, 0]


def test_transform_and_final_solve_agree_with_direct_sparse_leaf_solve() -> None:
    gamma = 0.6
    rng = np.random.default_rng(7)
    states = rng.choice(np.arange(4), size=900, p=np.array([0.45, 0.25, 0.2, 0.1]))
    X = states.reshape(-1, 1).astype(np.float64)
    X_next = np.minimum(states + 1, 3).reshape(-1, 1).astype(np.float64)
    X_initial = np.zeros((300, 1), dtype=np.float64)

    model = DiscountedOccupancyHistogramGradientBoostingRatioEstimator(
        gamma=gamma,
        n_estimators=4,
        learning_rate=0.2,
        max_depth=2,
        max_leaves=4,
        max_bins=4,
        min_samples_leaf=25,
        min_child_weight=1e-8,
        l2_leaf_reg=1.0,
        ridge=1e-8,
        solver="lsq_linear",
        weight_clip_quantile=None,
        random_state=8,
    ).fit(X, X_next, X_initial)
    phi = model.transform(X)
    phi_next = model.transform_next(X_next)
    phi_initial = model.transform_initial(X_initial)
    direct = solve_discounted_occupancy_ratio(
        phi,
        phi_next,
        phi_initial,
        gamma=gamma,
        ridge=1e-8,
        solver="lsq_linear",
        nonnegative=True,
    )

    assert sparse.issparse(phi)
    assert phi.nnz == X.shape[0] * model.feature_info_["n_trees"]
    assert np.allclose(model.beta_, direct.beta, atol=1e-8)
    assert np.allclose(model.predict_ratio(X[:100]), np.asarray(phi[:100] @ direct.beta).reshape(-1), atol=1e-8)


def test_streaming_direct_matches_csr_direct_solution() -> None:
    rng = np.random.default_rng(9)
    n = 360
    p = 3
    X = rng.normal(size=(n, p))
    X_next = 0.7 * X + 0.1 * rng.normal(size=(n, p))
    X_initial = rng.normal(size=(120, p))
    common = dict(
        gamma=0.6,
        n_estimators=5,
        learning_rate=0.1,
        max_depth=2,
        max_bins=10,
        min_samples_leaf=20,
        min_child_weight=1e-8,
        ridge=1e-6,
        max_event_rows=1_000,
        random_state=10,
    )
    csr = DiscountedOccupancyHistogramGradientBoostingRatioEstimator(
        **common,
        feature_storage="csr",
        solver="lsq_linear",
    ).fit(X, X_next, X_initial)
    streaming = DiscountedOccupancyHistogramGradientBoostingRatioEstimator(
        **common,
        feature_storage="streaming",
        solver="streaming_direct",
        batch_size=96,
    ).fit(X, X_next, X_initial)

    assert streaming.feature_info_["feature_storage"] == "streaming"
    assert streaming.solver_info_["method"] == "streaming_direct"
    assert np.allclose(csr.apply(X[:100]), streaming.apply(X[:100]))
    assert np.allclose(csr.predict_ratio(X[:100]), streaming.predict_ratio(X[:100]), atol=1e-8)


def test_hashed_streaming_occupancy_features_are_deterministic_after_save_load(tmp_path) -> None:
    rng = np.random.default_rng(20)
    n = 420
    p = 5
    X = rng.normal(size=(n, p))
    X_next = 0.75 * X + 0.15 * rng.normal(size=(n, p))
    X_initial = rng.normal(size=(140, p))

    model = DiscountedOccupancyHistogramGradientBoostingRatioEstimator(
        gamma=0.65,
        n_estimators=8,
        learning_rate=0.08,
        max_depth=2,
        max_bins=12,
        min_samples_leaf=18,
        min_child_weight=1e-8,
        feature_storage="hashed",
        hash_dim=128,
        solver="streaming_fista",
        inner_solver_max_iter=8,
        final_solver_max_iter=20,
        final_solver_tol=1e-3,
        ridge=1e-5,
        batch_size=96,
        max_event_rows=1_200,
        random_state=21,
    ).fit(X, X_next, X_initial)
    path = tmp_path / "hashed_occupancy_hist_gbt.pkl"
    model.save(path)
    loaded = DiscountedOccupancyHistogramGradientBoostingRatioEstimator.load(path)

    assert model.feature_info_["feature_storage"] == "hashed"
    assert model.feature_info_["n_features_solver"] == 128
    assert np.array_equal(model.apply(X[:40]), loaded.apply(X[:40]))
    assert np.allclose(model.predict_ratio(X[:40]), loaded.predict_ratio(X[:40]))
    assert np.all(model.predict_ratio(X[:40]) >= -1e-10)


def test_auto_storage_uses_streaming_for_large_row_count() -> None:
    rng = np.random.default_rng(22)
    n = 1_200
    p = 3
    X = rng.normal(size=(n, p))
    X_next = 0.8 * X + 0.1 * rng.normal(size=(n, p))
    X_initial = rng.normal(size=(200, p))

    model = DiscountedOccupancyHistogramGradientBoostingRatioEstimator(
        gamma=0.7,
        n_estimators=3,
        max_depth=1,
        max_bins=8,
        min_samples_leaf=30,
        min_child_weight=1e-8,
        feature_storage="auto",
        max_event_rows=800,
        batch_size=128,
        random_state=23,
    )
    model.fit(X, X_next, X_initial)
    assert model.feature_info_["feature_storage"] == "csr"

    model.feature_storage = "streaming"
    model.fit(X, X_next, X_initial)
    assert model.feature_info_["feature_storage"] == "streaming"
    assert model.solver_info_["method"] == "streaming_direct"


def test_missing_value_routing_early_stopping_and_persistence(tmp_path) -> None:
    rng = np.random.default_rng(10)
    n = 600
    states = rng.choice(np.arange(3), size=n, p=np.array([0.55, 0.3, 0.15]))
    noise = rng.normal(size=n)
    X = np.column_stack([states, noise]).astype(np.float64)
    X_next = np.column_stack([np.minimum(states + 1, 2), noise]).astype(np.float64)
    X[::17, 1] = np.nan
    X_next[::19, 1] = np.nan
    X_initial = np.column_stack([np.zeros(200), rng.normal(size=200)]).astype(np.float64)
    X_initial[::23, 1] = np.nan

    model = DiscountedOccupancyHistogramGradientBoostingRatioEstimator(
        gamma=0.5,
        n_estimators=12,
        learning_rate=0.1,
        max_depth=2,
        max_leaves=3,
        max_bins=8,
        min_samples_leaf=20,
        min_child_weight=1e-8,
        l2_leaf_reg=1.0,
        early_stopping_rounds=2,
        validation_fraction=0.2,
        ridge=1e-8,
        solver="lsq_linear",
        weight_clip_quantile=None,
        random_state=11,
    ).fit(X, X_next, X_initial)

    pred = model.predict_ratio(X[:50])
    path = tmp_path / "occupancy_hist_gbt.pkl"
    model.save(path)
    loaded = DiscountedOccupancyHistogramGradientBoostingRatioEstimator.load(path)

    assert len(model.trees_) <= 12
    assert np.all(np.isfinite(pred))
    assert np.array_equal(model.apply(X[:50]), loaded.apply(X[:50]))
    assert np.allclose(pred, loaded.predict_ratio(X[:50]))
    assert model.flow_validation_loss_


@pytest.mark.slow
def test_runtime_smoke_uses_sparse_leaf_features_and_auto_fista() -> None:
    rng = np.random.default_rng(12)
    n = 10_000
    p = 6
    X = rng.normal(size=(n, p))
    X_next = 0.8 * X + 0.2 * rng.normal(size=(n, p))
    X_initial = rng.normal(size=(1_000, p))

    model = DiscountedOccupancyHistogramGradientBoostingRatioEstimator(
        gamma=0.8,
        n_estimators=12,
        learning_rate=0.05,
        max_depth=2,
        max_leaves=4,
        max_bins=16,
        min_samples_leaf=100,
        min_child_weight=1e-8,
        l2_leaf_reg=1.0,
        hessian_floor=1e-6,
        subsample=0.5,
        max_event_rows=3_000,
        colsample_bytree=0.8,
        solver="auto",
        dense_threshold=8,
        inner_solver_tol=1e-3,
        inner_solver_max_iter=10,
        final_solver_tol=1e-4,
        final_solver_max_iter=50,
        random_state=13,
    ).fit(X, X_next, X_initial)

    features = model.transform(X[:200])
    assert model.solver_info_["solver"] == "fista"
    assert sparse.issparse(features)
    assert features.nnz == 200 * model.feature_info_["n_trees"]
    assert np.all(model.predict_ratio(X[:200]) >= -1e-10)
