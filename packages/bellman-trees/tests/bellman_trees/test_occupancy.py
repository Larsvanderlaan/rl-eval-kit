from __future__ import annotations

import numpy as np
from scipy import sparse

from bellman_trees import (
    DiscountedOccupancyRatioTree,
    discounted_flow_moment,
    solve_discounted_occupancy_ratio,
)


def _one_hot(ids: np.ndarray, width: int) -> sparse.csr_matrix:
    idx = np.asarray(ids, dtype=np.int64).reshape(-1)
    rows = np.arange(idx.size, dtype=np.int64)
    return sparse.csr_matrix((np.ones(idx.size), (rows, idx)), shape=(idx.size, width))


def test_discounted_occupancy_solver_recovers_tabular_ratio() -> None:
    phi = sparse.csr_matrix(np.eye(2))
    phi_next = sparse.csr_matrix(np.array([[0.0, 1.0], [0.0, 1.0]]))
    phi_initial = sparse.csr_matrix(np.array([[1.0, 0.0]]))
    behavior_mass = np.array([0.8, 0.2])

    result = solve_discounted_occupancy_ratio(
        phi,
        phi_next,
        phi_initial,
        sample_weight=behavior_mass,
        gamma=0.5,
        ridge=0.0,
        nonnegative=True,
    )

    assert result.diagnostics["nonnegative"] is True
    assert np.allclose(result.beta, np.array([0.625, 2.5]), atol=1e-6)
    assert np.isclose(np.sum(behavior_mass * result.beta), 1.0)
    assert np.linalg.norm(discounted_flow_moment(phi, phi_next, phi_initial, result.beta, behavior_mass, gamma=0.5)) < 1e-8


def test_discounted_occupancy_solver_estimates_sampled_tabular_chain() -> None:
    gamma = 0.5
    counts = np.array([600, 300, 100])
    states = np.repeat(np.arange(3), counts)
    next_states = np.minimum(states + 1, 2)
    initial_states = np.zeros(500, dtype=np.int64)
    true_discounted_mass = np.array([1.0 - gamma, (1.0 - gamma) * gamma, gamma**2])
    true_ratio = true_discounted_mass / (counts / counts.sum())

    result = solve_discounted_occupancy_ratio(
        _one_hot(states, 3),
        _one_hot(next_states, 3),
        _one_hot(initial_states, 3),
        gamma=gamma,
        ridge=1e-10,
        nonnegative=True,
    )

    assert np.allclose(result.beta, true_ratio, atol=1e-5)
    assert np.all(result.beta >= -1e-10)
    assert np.isclose(np.average(result.beta, weights=counts), 1.0)
    assert result.diagnostics["moment_violation_l2"] < 1e-5


def test_fista_solver_matches_small_exact_nnls_solution() -> None:
    gamma = 0.6
    counts = np.array([400, 250, 150])
    states = np.repeat(np.arange(3), counts)
    next_states = np.minimum(states + 1, 2)
    initial_states = np.zeros(500, dtype=np.int64)
    phi = _one_hot(states, 3)
    phi_next = _one_hot(next_states, 3)
    phi_initial = _one_hot(initial_states, 3)

    exact = solve_discounted_occupancy_ratio(
        phi,
        phi_next,
        phi_initial,
        gamma=gamma,
        ridge=1e-8,
        solver="lsq_linear",
        nonnegative=True,
    )
    fista = solve_discounted_occupancy_ratio(
        phi,
        phi_next,
        phi_initial,
        gamma=gamma,
        ridge=1e-8,
        solver="fista",
        nonnegative=True,
        tol=1e-8,
        max_iter=5000,
    )

    assert fista.diagnostics["linear_solve"] == "projected_fista"
    assert np.allclose(fista.beta, exact.beta, atol=2e-3)
    assert fista.diagnostics["objective_decreased"] is True
    assert np.isfinite(fista.diagnostics["projected_gradient_norm"])


def test_auto_solver_uses_matrix_free_fista_for_large_sparse_tabular_chain() -> None:
    gamma = 0.75
    n_states = 40
    repeats = 100
    states = np.repeat(np.arange(n_states), repeats)
    next_states = np.minimum(states + 1, n_states - 1)
    initial_states = np.zeros(2_000, dtype=np.int64)
    true_mass = (1.0 - gamma) * gamma ** np.arange(n_states)
    true_mass[-1] = gamma ** (n_states - 1)
    behavior_mass = np.full(n_states, 1.0 / n_states)
    true_ratio = true_mass / behavior_mass

    result = solve_discounted_occupancy_ratio(
        _one_hot(states, n_states),
        _one_hot(next_states, n_states),
        _one_hot(initial_states, n_states),
        gamma=gamma,
        ridge=1e-8,
        nonnegative=True,
        solver="auto",
        dense_threshold=8,
        tol=1e-7,
        max_iter=6000,
    )

    assert result.diagnostics["solver"] == "fista"
    assert result.diagnostics["linear_solve"] == "projected_fista"
    assert result.diagnostics["iterations"] <= 6000
    assert result.diagnostics["objective"] <= result.diagnostics["initial_objective"]
    assert np.all(result.beta >= -1e-10)
    assert np.isclose(np.mean(result.beta), 1.0, atol=1e-8)
    assert np.sqrt(np.mean((result.beta - true_ratio) ** 2)) < 0.05


def test_discounted_occupancy_solver_regularizes_and_preserves_nonnegativity() -> None:
    phi = sparse.csr_matrix(np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]]))
    phi_next = sparse.csr_matrix(np.array([[1.0, 1.0], [1.0, 2.0], [1.0, 2.0]]))
    phi_initial = sparse.csr_matrix(np.array([[1.0, 0.0]]))

    result = solve_discounted_occupancy_ratio(
        phi,
        phi_next,
        phi_initial,
        gamma=0.8,
        ridge=1e-2,
        nonnegative=True,
    )
    ratio = np.asarray(phi @ result.beta).reshape(-1)

    assert result.diagnostics["linear_solve"] == "lsq_linear"
    assert np.all(ratio >= -1e-10)
    assert np.isclose(np.mean(ratio), 1.0)
    assert np.isfinite(result.diagnostics["moment_violation_l2"])


def test_flow_balancing_tree_splits_on_discounted_flow_imbalance() -> None:
    X = np.concatenate([np.full((320, 1), -1.0), np.full((80, 1), 1.0)], axis=0)
    X_next = np.full_like(X, 1.0)
    X_initial = np.full((200, 1), -1.0)

    tree = DiscountedOccupancyRatioTree(
        gamma=0.5,
        max_depth=1,
        max_leaves=2,
        max_bins=2,
        min_samples_leaf=30,
        min_leaf_ess=20,
        min_improvement=1e-8,
        ridge=1e-8,
        nonnegative=True,
        honest=False,
        weight_clip_quantile=None,
        random_state=17,
    ).fit(X, X_next, X_initial)

    low_ratio, high_ratio = tree.predict_ratio(np.array([[-1.0], [1.0]]))
    fitted_ratio = tree.predict_ratio(X)

    assert tree.diagnostics_["n_splits"] == 1
    assert high_ratio > low_ratio
    assert np.all(fitted_ratio >= -1e-10)
    assert np.isclose(np.mean(fitted_ratio), 1.0)


def test_flow_balancing_tree_finds_tabular_structure_not_balanced_nuisance() -> None:
    gamma = 0.5
    nuisance_levels = np.linspace(-1.0, 1.0, 20)
    per_level_counts = np.array([300, 150, 50])
    rows = []
    next_rows = []
    for nuisance in nuisance_levels:
        states = np.repeat(np.arange(3), per_level_counts)
        rows.append(np.column_stack([states, np.full(states.size, nuisance)]))
        next_rows.append(np.column_stack([np.minimum(states + 1, 2), np.full(states.size, nuisance)]))
    X = np.concatenate(rows, axis=0).astype(float)
    X_next = np.concatenate(next_rows, axis=0).astype(float)
    X_initial = np.concatenate(
        [np.column_stack([np.zeros(100), np.full(100, nuisance)]) for nuisance in nuisance_levels],
        axis=0,
    ).astype(float)
    behavior_mass = per_level_counts / per_level_counts.sum()
    true_ratio = np.array([1.0 - gamma, (1.0 - gamma) * gamma, gamma**2]) / behavior_mass

    tree = DiscountedOccupancyRatioTree(
        gamma=gamma,
        max_depth=3,
        max_leaves=4,
        max_bins=8,
        min_samples_leaf=100,
        min_leaf_ess=80,
        min_improvement=1e-4,
        ridge=1e-8,
        nonnegative=True,
        honest=False,
        weight_clip_quantile=None,
        random_state=29,
    ).fit(X, X_next, X_initial)
    ratios = tree.predict_ratio(np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]))
    accepted = [row for row in tree.split_history_ if row["accepted"]]

    assert len(accepted) == 2
    assert [row["feature_index"] for row in accepted] == [0, 0]
    assert np.allclose(sorted(row["threshold"] for row in accepted), [0.5, 1.5])
    assert np.allclose(ratios, true_ratio, atol=5e-2)
    assert ratios[2] > 2.0 * ratios[0]
    assert np.all(tree.predict_ratio(X) >= -1e-10)


def test_aggregated_tree_scoring_matches_sparse_tree_scoring() -> None:
    gamma = 0.5
    nuisance_levels = np.linspace(-1.0, 1.0, 10)
    per_level_counts = np.array([180, 90, 30])
    rows = []
    next_rows = []
    for nuisance in nuisance_levels:
        states = np.repeat(np.arange(3), per_level_counts)
        rows.append(np.column_stack([states, np.full(states.size, nuisance)]))
        next_rows.append(np.column_stack([np.minimum(states + 1, 2), np.full(states.size, nuisance)]))
    X = np.concatenate(rows, axis=0).astype(float)
    X_next = np.concatenate(next_rows, axis=0).astype(float)
    X_initial = np.concatenate(
        [np.column_stack([np.zeros(80), np.full(80, nuisance)]) for nuisance in nuisance_levels],
        axis=0,
    ).astype(float)
    common_kwargs = dict(
        gamma=gamma,
        max_depth=3,
        max_leaves=4,
        max_bins=8,
        min_samples_leaf=80,
        min_leaf_ess=60,
        min_improvement=1e-4,
        ridge=1e-8,
        nonnegative=True,
        honest=False,
        weight_clip_quantile=None,
        random_state=31,
    )

    aggregated = DiscountedOccupancyRatioTree(split_score_mode="aggregated_flow", **common_kwargs).fit(X, X_next, X_initial)
    sparse_tree = DiscountedOccupancyRatioTree(split_score_mode="sparse_flow", **common_kwargs).fit(X, X_next, X_initial)
    aggregated_splits = [(row["feature_index"], row["threshold"]) for row in aggregated.split_history_ if row["accepted"]]
    sparse_splits = [(row["feature_index"], row["threshold"]) for row in sparse_tree.split_history_ if row["accepted"]]

    assert aggregated_splits == sparse_splits
    assert len(aggregated_splits) >= 1
    assert all(feature == 0 for feature, _ in aggregated_splits)
    assert np.allclose(aggregated.predict_ratio(X[:200]), sparse_tree.predict_ratio(X[:200]), atol=1e-5)
