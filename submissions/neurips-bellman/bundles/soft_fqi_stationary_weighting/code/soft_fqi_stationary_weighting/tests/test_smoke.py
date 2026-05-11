from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.algorithms import FQIConfig, _weighted_ridge_solution, run_minimax_soft_q
from src.data import sample_transition_batch
from src.env import EnvConfig, GridConfig, build_grid_mdp
from src.experiment import _method_specs
from src.features import QFeatureMap, RatioFeatureMap, linear_q_features
from src.soft_dp import soft_value_iteration, softmax_policy, state_action_distribution, stationary_state_distribution
from src.weights import estimate_moment_weights, oracle_sample_weights


def _tiny_context():
    mdp = build_grid_mdp(GridConfig(n_x=9, n_y=9), EnvConfig(process_noise=0.09, teleport_prob=0.004))
    vi = soft_value_iteration(mdp.transition, mdp.reward, gamma=0.95, tau=0.01, tol=1e-8, max_iter=4000)
    pi = softmax_policy(vi.q, tau=0.01)
    d_state, residual, converged = stationary_state_distribution(mdp.transition, pi)
    d_sa = state_action_distribution(d_state, pi)
    return mdp, vi, pi, d_state, d_sa, residual, converged


def test_transition_rows_sum_to_one() -> None:
    mdp, *_ = _tiny_context()
    row_sums = mdp.transition.sum(axis=2)
    assert np.max(np.abs(row_sums - 1.0)) < 1e-10
    assert np.all(mdp.transition >= 0.0)


def test_soft_value_iteration_and_stationary_distribution() -> None:
    _mdp, vi, _pi, _d_state, _d_sa, residual, converged = _tiny_context()
    assert vi.converged
    assert vi.sup_delta < 1e-7
    assert converged
    assert residual < 1e-9


def test_oracle_ratio_normalizes_under_behavior() -> None:
    mdp, _vi, pi, _d_state, target_sa, _residual, _converged = _tiny_context()
    uniform = np.ones_like(pi) / pi.shape[1]
    behavior = 0.8 * pi + 0.2 * uniform
    behavior_state, _resid, _conv = stationary_state_distribution(mdp.transition, behavior)
    behavior_sa = state_action_distribution(behavior_state, behavior)
    ratio = target_sa / np.maximum(behavior_sa, 1e-12)
    assert abs(float(np.sum(behavior_sa * ratio)) - 1.0) < 1e-8
    batch = sample_transition_batch(mdp, behavior_state, behavior, n_samples=300, seed=7)
    weights = oracle_sample_weights(batch, target_sa, behavior_sa)
    assert np.isfinite(weights).all()
    assert abs(float(np.mean(weights)) - 1.0) < 1e-10


def test_closed_form_ratio_estimator_is_finite() -> None:
    mdp, _vi, pi, _d_state, _target_sa, _residual, _converged = _tiny_context()
    behavior = 0.7 * pi + 0.3 / pi.shape[1]
    behavior_state, _resid, _conv = stationary_state_distribution(mdp.transition, behavior)
    batch = sample_transition_batch(mdp, behavior_state, behavior, n_samples=400, seed=11)
    ratio_features = RatioFeatureMap.from_grid(mdp.states, mdp.actions, n_state_centers=4)
    estimate = estimate_moment_weights(
        batch,
        states_grid=mdp.states,
        transition=mdp.transition,
        target_policy=pi,
        behavior_state_dist=behavior_state,
        ratio_features=ratio_features,
        gamma_weight=0.95,
        max_weight=20.0,
    )
    assert np.isfinite(estimate.weights).all()
    assert abs(float(np.mean(estimate.weights)) - 1.0) < 1e-10
    assert estimate.diagnostics["effective_sample_size_fraction"] > 0.1


def test_cv_ridge_ratio_estimator_selects_candidate() -> None:
    mdp, _vi, pi, _d_state, _target_sa, _residual, _converged = _tiny_context()
    behavior = 0.7 * pi + 0.3 / pi.shape[1]
    behavior_state, _resid, _conv = stationary_state_distribution(mdp.transition, behavior)
    batch = sample_transition_batch(mdp, behavior_state, behavior, n_samples=400, seed=13)
    ratio_features = RatioFeatureMap.from_grid(mdp.states, mdp.actions, n_state_centers=4)
    estimate = estimate_moment_weights(
        batch,
        states_grid=mdp.states,
        transition=mdp.transition,
        target_policy=pi,
        behavior_state_dist=behavior_state,
        ratio_features=ratio_features,
        gamma_weight=0.95,
        cv_ridge=True,
        cv_ridge_grid=(1e-6, 1e-4),
        cv_folds=2,
        max_weight=20.0,
    )
    assert np.isfinite(estimate.weights).all()
    assert estimate.diagnostics["cv_ridge_selected"] == 1.0
    assert estimate.diagnostics["ridge_primal"] in (1e-6, 1e-4)


def test_rich_q_feature_map_is_finite_and_stable() -> None:
    mdp, *_ = _tiny_context()
    q_features = QFeatureMap.from_grid(
        "rich_rbf",
        mdp.states,
        mdp.actions,
        n_state_centers=4,
        bandwidth_scale=0.8,
    )
    states = np.array([0, 1, 2, 3], dtype=np.int64)
    actions = np.array([0, 1, 2, 3], dtype=np.int64)
    phi = q_features.transform(mdp.states[states], actions)
    assert phi.shape == (4, q_features.dimension)
    assert q_features.dimension == 12 + 4 * mdp.n_actions
    assert np.isfinite(phi).all()


def test_method_specs_accept_explicit_estimated_gamma() -> None:
    specs = _method_specs(["unweighted", "estimated_g0p95", "minimax"], [0.5])
    assert specs[0][0] == "unweighted"
    assert np.isnan(specs[0][1])
    assert specs[1] == ("estimated_g0p95", 0.95)
    assert specs[2][0] == "minimax"
    assert np.isnan(specs[2][1])


def test_minimax_soft_q_returns_finite_metrics() -> None:
    mdp, vi, pi, _d_state, target_sa, _residual, _converged = _tiny_context()
    behavior = 0.7 * pi + 0.3 / pi.shape[1]
    behavior_state, _resid, _conv = stationary_state_distribution(mdp.transition, behavior)
    behavior_sa = state_action_distribution(behavior_state, behavior)
    batch = sample_transition_batch(mdp, behavior_state, behavior, n_samples=240, seed=17)
    ratio_features = RatioFeatureMap.from_grid(mdp.states, mdp.actions, n_state_centers=4)
    q_features = QFeatureMap.from_grid("linear", mdp.states, mdp.actions)
    rows = run_minimax_soft_q(
        mdp=mdp,
        batch=batch,
        q_star=vi.q,
        target_sa_dist=target_sa,
        behavior_sa_dist=behavior_sa,
        schedule="direct",
        fqi_config=FQIConfig(gamma=0.95, tau_final=0.01, n_iters=3, ridge=1e-4, metrics_stride=1),
        minimax_config={"q_ridge": 1e-4, "critic_ridge": 1e-4, "damping": 0.5},
        ratio_features=ratio_features,
        q_feature_map=q_features,
        reference_value=0.0,
        rho0=np.ones(mdp.n_states) / mdp.n_states,
        seed=17,
        base_meta={"failed": 0, "method": "minimax"},
    )
    assert rows
    final = rows[-1]
    assert final["is_final"] == 1
    assert np.isfinite(final["stationary_q_rmse"])
    assert np.isfinite(final["minimax_residual_norm"])


def test_weighted_ridge_matches_distribution_projection_scaling() -> None:
    mdp, vi, _pi, _d_state, target_sa, _residual, _converged = _tiny_context()
    state_ids, action_ids = np.meshgrid(np.arange(mdp.n_states), np.arange(mdp.n_actions), indexing="ij")
    state_ids = state_ids.reshape(-1)
    action_ids = action_ids.reshape(-1)
    phi = linear_q_features(mdp.states[state_ids], action_ids, mdp.actions)
    y = vi.q.reshape(-1)
    theta_dist = _weighted_ridge_solution(phi, y, target_sa.reshape(-1), ridge=1e-5)
    theta_scaled = _weighted_ridge_solution(phi, y, 17.0 * target_sa.reshape(-1), ridge=1e-5)
    assert np.max(np.abs(theta_dist - theta_scaled)) < 1e-6
