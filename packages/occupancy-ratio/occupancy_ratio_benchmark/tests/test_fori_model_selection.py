from __future__ import annotations

import numpy as np
import pytest

from occupancy_ratio.configs import SourceStateRatioConfig
from occupancy_ratio.fori_model_selection import (
    DirectMultiOutputAdjointBackupRegressor,
    FORICandidateSpec,
    FORITwoStageCV,
    FORITwoStageCVConfig,
    FirstStageDensityRatioCV,
    LowRankAdjointBackupRegressor,
    LowRankAdjointBellmanCV,
    adjoint_bellman_residual,
    compute_candidate_ratio_matrix,
    kfold_by_episode_ids,
    sample_target_successor_actions,
    split_by_episode_ids,
)
from occupancy_ratio.fori_model_selection import (
    _bootstrap_score_ses,
    _build_data_adapter,
    _candidate_diagnostics,
    _final_scores,
    _one_se_selection,
)


class ConstantRatio:
    def __init__(self, value: float):
        self.value = float(value)

    def predict_state_action_ratio(self, states, actions, *, clip=True):
        del actions, clip
        return np.full(np.asarray(states).shape[0], self.value, dtype=np.float64)


class LinearRatio:
    def __init__(self, coef: float, offset: float = 1.0):
        self.coef = float(coef)
        self.offset = float(offset)

    def predict_state_action_ratio(self, states, actions, *, clip=True):
        del actions, clip
        s = np.asarray(states, dtype=np.float64).reshape(-1)
        return self.offset + self.coef * s


def _tiny_data(n: int = 50):
    states = np.linspace(-1.0, 1.0, n).reshape(-1, 1)
    actions = np.zeros((n, 1))
    next_states = states.copy()
    target_actions = actions.copy()
    target_next_actions = actions.copy()
    episode_ids = np.arange(n)
    return states, actions, next_states, target_actions, target_next_actions, episode_ids


def _fast_config(**overrides):
    cfg = dict(
        backup_regressor_backend="ridge",
        svd_backend="numpy",
        low_rank_ranks=(1, 2, 4),
        first_stage_cv_folds=2,
        n_bootstrap=25,
        seed=7,
        first_stage_density_ratio_configs=[
            SourceStateRatioConfig(
                num_boost_round=2,
                early_stopping_rounds=0,
                validation_fraction=0.2,
                show_progress=False,
                lgb_params={"verbose": -1, "num_threads": 1, "min_data_in_leaf": 1},
            )
        ],
    )
    cfg.update(overrides)
    return FORITwoStageCVConfig(**cfg)


def test_episode_split_and_kfold_keep_episodes_disjoint():
    episode_ids = np.repeat(np.arange(10), 3)
    split_idx, split_eps = split_by_episode_ids(episode_ids, (0.3, 0.2, 0.2, 0.1, 0.2), seed=0)
    assert set(split_idx) == {"nuisance", "fori", "backup_train", "backup_val", "score"}
    seen = set()
    for eps in split_eps.values():
        eps_set = set(eps.tolist())
        assert seen.isdisjoint(eps_set)
        seen.update(eps_set)
    folds = kfold_by_episode_ids(episode_ids, 3, seed=1)
    fold_episode_sets = [set(episode_ids[idx].tolist()) for idx in folds]
    for i, left in enumerate(fold_episode_sets):
        for right in fold_episode_sets[i + 1 :]:
            assert left.isdisjoint(right)


def test_stochastic_target_policy_sampling_is_seeded():
    class Policy:
        def sample_action(self, obs, rng):
            return rng.integers(0, 3, size=(np.asarray(obs).shape[0], 1))

    obs = np.zeros((8, 2))
    a = sample_target_successor_actions(obs, Policy(), action_space=3, seed=10)
    b = sample_target_successor_actions(obs, Policy(), action_space=3, seed=10)
    c = sample_target_successor_actions(obs, Policy(), action_space=3, seed=11)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_live_only_terminal_continuation_uses_terminated_and_timeouts():
    states, actions, next_states, target_actions, target_next_actions, episode_ids = _tiny_data(4)
    data = _build_data_adapter(
        states=states,
        actions=actions,
        next_states=next_states,
        target_actions=target_actions,
        target_next_actions=target_next_actions,
        target_policy=None,
        action_space=None,
        n_action_samples=1,
        gamma=0.9,
        episode_ids=episode_ids,
        rewards=None,
        initial_states=None,
        initial_actions=None,
        initial_weights=None,
        initial_episode_ids=None,
        terminated=None,
        truncated=np.array([0, 1, 0, 1]),
        done=np.array([0, 1, 1, 1]),
        config=_fast_config(terminal_mode="live_only_submarkov", treat_timeouts_as_nonterminal=True),
    )
    np.testing.assert_allclose(data.continuation, np.array([1.0, 1.0, 0.0, 1.0]))


def test_live_only_cpi_preserves_continuation_mass_scale():
    states, actions, next_states, target_actions, target_next_actions, episode_ids = _tiny_data(12)
    config = _fast_config(terminal_mode="live_only_submarkov")
    data = _build_data_adapter(
        states=states,
        actions=actions,
        next_states=next_states,
        target_actions=target_actions,
        target_next_actions=target_next_actions,
        target_policy=None,
        action_space=None,
        n_action_samples=1,
        gamma=0.9,
        episode_ids=episode_ids,
        rewards=None,
        initial_states=None,
        initial_actions=None,
        initial_weights=None,
        initial_episode_ids=None,
        terminated=np.array([0, 1] * 6),
        truncated=None,
        done=None,
        config=config,
    )
    cfg = config.first_stage_density_ratio_configs[0]
    model = FirstStageDensityRatioCV(config)._fit_cpi_model(data=data, config=cfg, train_idx=np.arange(12), seed=4)
    assert model.scale == pytest.approx(0.5)


def test_adjoint_bellman_residual_algebra():
    w = np.array([[1.0, 2.0], [3.0, 4.0]])
    omega0 = np.array([0.5, 1.5])
    cpi = np.array([2.0, 3.0])
    m_hat = np.array([[1.0, 0.5], [2.0, 1.0]])
    resid = adjoint_bellman_residual(w_score=w, omega0_score=omega0, cpi_score=cpi, m_hat_score=m_hat, gamma=0.25)
    expected = w - (0.75 * omega0[:, None] + 0.25 * cpi[:, None] * m_hat)
    np.testing.assert_allclose(resid, expected)


def test_low_rank_pca_reconstructs_synthetic_low_rank_matrix():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(80, 3))
    z = np.column_stack([x[:, 0], x[:, 1]])
    loadings = rng.normal(size=(2, 6))
    w = z @ loadings + 1.0
    reg = LowRankAdjointBackupRegressor(rank=2, backend="ridge", svd_backend="numpy", ridge_alpha=1e-10)
    reg.fit(x, w)
    pred = reg.predict(x)
    assert np.mean((pred - w) ** 2) < 1e-8


def test_low_rank_adjoint_backup_scores_current_x_not_x_plus():
    x_current = np.linspace(-1, 1, 60).reshape(-1, 1)
    x_plus = x_current + 10.0
    w = np.column_stack([1.0 + x_current[:, 0], 2.0 - x_current[:, 0]])
    cv = LowRankAdjointBellmanCV(_fast_config(low_rank_ranks=(2,), n_bootstrap=0))
    cv.fit(x_plus_train=x_plus, w_train=w, x_plus_val=x_plus, w_val=w)
    # If scoring incorrectly evaluated the backup at X_plus, the ridge model
    # would predict the training W exactly. Current-X evaluation must differ.
    payload = cv.score(
        x_current_score=x_current,
        w_score=w,
        omega0_score=np.zeros(x_current.shape[0]),
        cpi_score=np.ones(x_current.shape[0]),
        gamma=1.0,
    )
    assert np.mean(payload["residual"] ** 2) > 1.0


def test_candidate_matrix_chunking_matches_full():
    states, actions, next_states, target_actions, target_next_actions, episode_ids = _tiny_data(17)
    data = _build_data_adapter(
        states=states,
        actions=actions,
        next_states=next_states,
        target_actions=target_actions,
        target_next_actions=target_next_actions,
        target_policy=None,
        action_space=None,
        n_action_samples=1,
        gamma=0.0,
        episode_ids=episode_ids,
        rewards=None,
        initial_states=None,
        initial_actions=None,
        initial_weights=None,
        initial_episode_ids=None,
        terminated=None,
        truncated=None,
        done=None,
        config=_fast_config(terminal_mode="absorbing_state"),
    )
    candidates = [
        FORICandidateSpec("c0", model=ConstantRatio(1.0)),
        FORICandidateSpec("c1", model=LinearRatio(0.5)),
    ]
    idx = np.arange(17)
    full = compute_candidate_ratio_matrix(candidates, data, idx, split_name="score", candidate_block_size=10, transition_batch_size=20)
    chunked = compute_candidate_ratio_matrix(candidates, data, idx, split_name="score", candidate_block_size=1, transition_batch_size=3)
    np.testing.assert_allclose(full, chunked)


def test_direct_multioutput_matches_full_rank_low_rank_with_ridge():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(60, 4))
    w = rng.normal(size=(60, 5))
    low = LowRankAdjointBackupRegressor(rank=5, backend="ridge", ridge_alpha=1e-8, svd_backend="numpy").fit(x, w)
    direct = DirectMultiOutputAdjointBackupRegressor(backend="ridge", ridge_alpha=1e-8).fit(x, w)
    probe = rng.normal(size=(10, 4))
    np.testing.assert_allclose(low.predict(probe), direct.predict(probe), atol=1e-6)


def test_score_leakage_errors_by_default():
    states, actions, next_states, target_actions, target_next_actions, episode_ids = _tiny_data(30)
    candidates = [FORICandidateSpec("leaky", model=ConstantRatio(1.0), trained_on_episode_ids=episode_ids)]
    with pytest.raises(ValueError, match="overlaps D_score"):
        FORITwoStageCV(_fast_config()).fit(
            states=states,
            actions=actions,
            next_states=next_states,
            target_actions=target_actions,
            target_next_actions=target_next_actions,
            gamma=0.0,
            episode_ids=episode_ids,
            candidates=candidates,
        )


def test_naive_internal_residual_warns_and_reports_diagnostic():
    states, actions, next_states, target_actions, target_next_actions, episode_ids = _tiny_data(40)
    candidate = FORICandidateSpec(
        "true_one",
        model=ConstantRatio(1.0),
        metadata={"internal_backup_model": lambda x: np.ones(np.asarray(x).shape[0])},
    )
    with pytest.warns(RuntimeWarning, match="tautological"):
        result = FORITwoStageCV(_fast_config(low_rank_ranks=(1,), n_bootstrap=0)).fit(
            states=states,
            actions=actions,
            next_states=next_states,
            target_actions=target_actions,
            target_next_actions=target_next_actions,
            gamma=0.0,
            episode_ids=episode_ids,
            candidates=[candidate],
        )
    assert np.isfinite(result.candidate_rows()[0]["naive_internal_ABE"])


def test_two_stage_selector_selects_true_constant_ratio_at_gamma_zero():
    states, actions, next_states, target_actions, target_next_actions, episode_ids = _tiny_data(40)
    result = FORITwoStageCV(_fast_config(low_rank_ranks=(1,))).fit(
        states=states,
        actions=actions,
        next_states=next_states,
        target_actions=target_actions,
        target_next_actions=target_next_actions,
        gamma=0.0,
        episode_ids=episode_ids,
        candidates=[
            FORICandidateSpec("true_one", model=ConstantRatio(1.0), complexity_order_key=(0,)),
            FORICandidateSpec("biased_two", model=ConstantRatio(2.0), complexity_order_key=(1,)),
        ],
    )
    assert result.selected_candidate_id == "true_one"
    rows = {row["candidate_id"]: row for row in result.candidate_rows()}
    assert rows["true_one"]["ABE_score"] < rows["biased_two"]["ABE_score"]


def test_default_selected_candidate_uses_one_se_recommendation():
    states, actions, next_states, target_actions, target_next_actions, episode_ids = _tiny_data(60)
    config = _fast_config(
        split_fractions=(0.20, 0.20, 0.20, 0.10, 0.30),
        low_rank_ranks=(1,),
        n_bootstrap=200,
    )
    split_idx, _ = split_by_episode_ids(episode_ids, config.split_fractions, config.seed)
    simple_cache = {name: np.full(idx.shape[0], 1.028) for name, idx in split_idx.items()}
    complex_cache = {name: np.ones(idx.shape[0]) for name, idx in split_idx.items()}
    complex_cache["score"] = complex_cache["score"].copy()
    complex_cache["score"][0] = 1.10

    result = FORITwoStageCV(config).fit(
        states=states,
        actions=actions,
        next_states=next_states,
        target_actions=target_actions,
        target_next_actions=target_next_actions,
        gamma=0.0,
        episode_ids=episode_ids,
        candidates=[
            FORICandidateSpec("simple_one_se", cached_predictions=simple_cache, complexity_order_key=(0,)),
            FORICandidateSpec("complex_min", cached_predictions=complex_cache, complexity_order_key=(1,)),
        ],
    )

    assert result.selection_rule == "one_se"
    assert result.selected_min_score_candidate_id == "complex_min"
    assert result.selected_one_se_candidate_id == "simple_one_se"
    assert result.selected_one_se_marginal_candidate_id == "simple_one_se"
    assert result.selected_one_se_paired_candidate_id is not None
    assert result.selected_candidate_id == "simple_one_se"


def test_paired_one_se_can_reject_marginal_underfit_candidate():
    candidates = [
        FORICandidateSpec("simple_underfit", complexity_order_key=(0,)),
        FORICandidateSpec("complex_min", complexity_order_key=(1,)),
    ]
    final_score = np.array([1.05, 1.00])
    final_se = np.array([0.20, 0.10])
    paired_diff_se = np.array([0.01, 0.0])

    marginal = _one_se_selection(final_score, final_se, candidates, 1, method="marginal")
    paired = _one_se_selection(
        final_score,
        final_se,
        candidates,
        1,
        method="paired",
        paired_diff_se=paired_diff_se,
    )

    assert candidates[marginal].candidate_id == "simple_underfit"
    assert candidates[paired].candidate_id == "complex_min"


def test_bootstrap_returns_paired_difference_se_over_score_episodes():
    residual = np.array(
        [
            [1.0, 1.1],
            [1.0, 1.1],
            [2.0, 2.1],
            [2.0, 2.1],
        ]
    )
    w_score = np.ones_like(residual)
    config = _fast_config(n_bootstrap=100, seed=13)
    abe_se, final_se, paired_diff_se, diagnostic_se = _bootstrap_score_ses(
        residual=residual,
        w_score=w_score,
        episode_ids=np.array([0, 0, 1, 1]),
        final_score_components={},
        config=config,
        selected_idx=0,
    )

    assert abe_se.shape == (2,)
    assert final_se.shape == (2,)
    assert paired_diff_se.shape == (2,)
    assert paired_diff_se[0] == pytest.approx(0.0)
    assert np.isfinite(paired_diff_se[1])
    assert "mean_ratio_se" in diagnostic_se


def test_abe_val_rank_selection_reports_validation_bellman_metric():
    rng = np.random.default_rng(21)
    x = rng.normal(size=(80, 3))
    w = np.column_stack([x[:, 0], x[:, 1] ** 2 + 0.1 * x[:, 2], x[:, 0] - x[:, 1]])
    config = _fast_config(
        low_rank_ranks=(1, 2, 3),
        backup_rank_selection_metric="abe_val",
        n_bootstrap=0,
    )
    cv = LowRankAdjointBellmanCV(config).fit(
        x_plus_train=x[:50],
        w_train=w[:50],
        x_plus_val=x[50:],
        w_val=w[50:],
        x_current_val=x[50:],
        omega0_val=np.zeros(30),
        cpi_val=np.ones(30),
        gamma=1.0,
    )

    assert cv.rank_selection_metric_ == "abe_val"
    assert np.isfinite(cv.backup_abe_val_score_)
    assert np.isfinite(cv.rank_selection_score_)
    assert {int(row["rank"]) for row in cv.rank_selection_table_} == {1, 2, 3}


def test_direct_agreement_rank_selection_reports_direct_metric():
    rng = np.random.default_rng(22)
    x = rng.normal(size=(50, 2))
    w = np.column_stack([x[:, 0], x[:, 1], x[:, 0] + x[:, 1]])
    cv = LowRankAdjointBellmanCV(
        _fast_config(
            low_rank_ranks=(1, 2, 3),
            backup_rank_selection_metric="direct_agreement",
            direct_multioutput_max_candidates=10,
            n_bootstrap=0,
        )
    ).fit(
        x_plus_train=x[:30],
        w_train=w[:30],
        x_plus_val=x[30:],
        w_val=w[30:],
        x_current_val=x[30:],
    )

    assert cv.rank_selection_metric_ == "direct_agreement"
    assert np.isfinite(cv.direct_agreement_val_mse_)
    assert np.isfinite(cv.rank_selection_score_)


def test_near_uniform_collapse_is_policy_shift_diagnostic_only():
    w = np.ones((20, 2))
    config = _fast_config(
        collapse_diagnostic_ess_fraction=0.95,
        collapse_diagnostic_weight_cv=0.01,
        collapse_diagnostic_min_action_shift=0.1,
    )
    shifted = _candidate_diagnostics(
        w,
        action_shift={"policy_action_shift_l2": 0.5, "policy_action_shift_mean_abs": 0.5},
        config=config,
    )
    unshifted = _candidate_diagnostics(
        w,
        action_shift={"policy_action_shift_l2": 0.0, "policy_action_shift_mean_abs": 0.0},
        config=config,
    )

    np.testing.assert_allclose(shifted["near_uniform_collapse"], np.ones(2))
    np.testing.assert_allclose(unshifted["near_uniform_collapse"], np.zeros(2))
    scores = _final_scores(np.array([1.0, 2.0]), shifted, config)
    np.testing.assert_allclose(scores, np.array([1.0, 2.0]))
