from __future__ import annotations

import numpy as np

from occupancy_ratio.fit_occupancy_ratio import (
    ActionRatioConfig,
    OccupancyRegressionConfig,
    SourceStateRatioConfig,
    TransitionRatioConfig,
)
from occupancy_ratio_benchmark.discrete import make_discrete_dataset
from occupancy_ratio_benchmark.fori_cv import (
    FORICVCandidate,
    fit_fori_cv_candidate,
    run_fori_cv_benchmark,
    score_value_grouped_moment_balance,
    summarize_cv_results,
)


def test_fori_cv_smoke_fits_each_fold_without_leakage() -> None:
    dataset = make_discrete_dataset(setting="discrete_chain", gamma=0.9, sample_size=80, seed=0)
    lgb_params = {"verbose": -1, "num_threads": 1, "min_data_in_leaf": 2}
    candidate = FORICVCandidate(
        name="boosted_tiny",
        family="boosted",
        occupancy=OccupancyRegressionConfig(
            num_iterations=2,
            trees_per_iteration=1,
            mcmc_samples=4,
            batch_size=64,
            validation_fraction=0.2,
            patience=1,
            lgb_params=lgb_params,
        ),
        action_ratio=ActionRatioConfig(
            num_boost_round=2,
            validation_fraction=0.2,
            early_stopping_rounds=1,
            lgb_params=lgb_params,
        ),
        source_state_ratio=SourceStateRatioConfig(
            num_boost_round=2,
            validation_fraction=0.2,
            early_stopping_rounds=1,
            lgb_params=lgb_params,
        ),
        transition_ratio=TransitionRatioConfig(
            num_boost_round=2,
            permutation_samples=2,
            validation_fraction=0.2,
            early_stopping_rounds=1,
            lgb_params=lgb_params,
        ),
    )

    result = run_fori_cv_benchmark(
        dataset,
        [candidate],
        k_folds=2,
        seed=1,
        rff_features=2,
        keep_fold_weights=False,
    )

    assert result.selected_candidate == "boosted_tiny"
    assert len(result.rows) == 2
    assert len(result.summary) == 1
    assert all(row["candidate"] == "boosted_tiny" for row in result.rows)
    assert all(np.isfinite(row["mb"]) for row in result.rows)
    assert all(np.isfinite(row["fp"]) for row in result.rows)
    assert all(row["mb_reward_features"] > 0 for row in result.rows)


def test_fori_cv_selection_uses_mb_not_fixed_point_residual() -> None:
    rows = [
        {
            "candidate": "low_fp_bad_mb",
            "fold": 0,
            "mb": 10.0,
            "fp": 0.001,
            "ess_fraction": 0.9,
            "mean_ratio": 1.0,
            "q95_ratio": 1.1,
            "q99_ratio": 1.2,
            "max_ratio": 1.3,
            "clipping_fraction": 0.0,
            "runtime_sec": 1.0,
            "invalid": False,
            "stabilization_strength": 1.0,
        },
        {
            "candidate": "higher_fp_good_mb",
            "fold": 0,
            "mb": 1.0,
            "fp": 0.5,
            "ess_fraction": 0.3,
            "mean_ratio": 1.0,
            "q95_ratio": 4.0,
            "q99_ratio": 8.0,
            "max_ratio": 10.0,
            "clipping_fraction": 0.0,
            "runtime_sec": 1.0,
            "invalid": False,
            "stabilization_strength": 0.2,
        },
    ]

    summary = summarize_cv_results(rows)

    selected = [row["candidate"] for row in summary if row["selected_by_mb"]]
    assert selected == ["higher_fp_good_mb"]


def test_value_grouped_moment_balance_is_leakage_safe_and_finite() -> None:
    dataset = make_discrete_dataset(setting="discrete_chain", gamma=0.9, sample_size=72, seed=1)
    train_idx = np.arange(48, dtype=np.int64)
    valid_idx = np.arange(48, 72, dtype=np.int64)
    lgb_params = {"verbose": -1, "num_threads": 1, "min_data_in_leaf": 2}
    candidate = FORICVCandidate(
        name="boosted_tiny",
        family="boosted",
        occupancy=OccupancyRegressionConfig(
            num_iterations=2,
            trees_per_iteration=1,
            mcmc_samples=4,
            batch_size=64,
            validation_fraction=0.2,
            patience=1,
            lgb_params=lgb_params,
        ),
        action_ratio=ActionRatioConfig(
            num_boost_round=2,
            validation_fraction=0.2,
            early_stopping_rounds=1,
            lgb_params=lgb_params,
        ),
        source_state_ratio=SourceStateRatioConfig(
            num_boost_round=2,
            validation_fraction=0.2,
            early_stopping_rounds=1,
            lgb_params=lgb_params,
        ),
        transition_ratio=TransitionRatioConfig(
            num_boost_round=2,
            permutation_samples=2,
            validation_fraction=0.2,
            early_stopping_rounds=1,
            lgb_params=lgb_params,
        ),
    )

    fit = fit_fori_cv_candidate(dataset, candidate, train_idx, fold=0, seed=5)
    scores = score_value_grouped_moment_balance(
        fit,
        dataset,
        valid_idx,
        seed=5,
        reward_max_steps=8,
        reward_patience=1,
        reward_feature_cap=4,
        fqe_iterations=2,
        fqe_patience=1,
        rff_features=2,
        geometry_features=2,
    )

    assert np.isfinite(scores["mb_value_grouped"])
    assert scores["mb_value_grouped_groups"] >= 3
    assert scores["mb_value_grouped_available"]
