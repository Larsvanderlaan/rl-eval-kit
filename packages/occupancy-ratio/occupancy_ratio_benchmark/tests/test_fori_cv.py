from __future__ import annotations

import numpy as np
import pytest

from occupancy_ratio.fit_occupancy_ratio import (
    ActionRatioConfig,
    OccupancyRegressionConfig,
    SourceStateRatioConfig,
    TransitionRatioConfig,
)
from occupancy_ratio.tuning import CandidateResult, FoldResult, OccupancyTuningConfig
from occupancy_ratio._tuning_staged import StagedCVCandidateRow, monotone_one_se_prune, run_staged_bootstrap_cv
import occupancy_ratio._tuning_impl as tuning_impl
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


def test_staged_bootstrap_cv_prunes_high_loss_candidate() -> None:
    good = _candidate_with_losses("boosted_000", [0.10, 0.12, 0.11], complexity_rank=2)
    bad = _candidate_with_losses("boosted_001", [2.0, 2.2, 1.9], complexity_rank=1)

    result = run_staged_bootstrap_cv(
        [good, bad],
        OccupancyTuningConfig(
            families=("boosted",),
            staged_bootstrap_cv=True,
            staged_cv_iterations=3,
            staged_cv_n_bootstrap=50,
        ),
        seed=5,
    )

    rows = {row["candidate_id"]: row for row in result.candidate_dicts()}
    assert rows["boosted_000"]["kept"]
    assert rows["boosted_001"]["pruned"]
    assert good.metrics["staged_cv_kept"] == pytest.approx(1.0)
    assert bad.metrics["staged_cv_pruned"] == pytest.approx(1.0)


def test_staged_bootstrap_cv_protects_larger_and_incomparable_candidates() -> None:
    rows = [
        StagedCVCandidateRow("small", "small", "neural", "staged_1", 4.0, 0.0, 0, False, True, False, 0.0),
        StagedCVCandidateRow("medium", "medium", "neural", "staged_1", 0.0, 0.0, 0, False, True, False, 0.0),
        StagedCVCandidateRow("large", "large", "neural", "staged_1", 9.0, 0.0, 0, False, True, False, 0.0),
        StagedCVCandidateRow("custom", "custom", "neural", "staged_1", 16.0, 0.0, 0, False, True, False, 0.0),
    ]
    complexity = {
        "small": {"group": "neural:params", "rank": (25.0,), "rank_repr": "25", "source": "explicit"},
        "medium": {"group": "neural:params", "rank": (100.0,), "rank_repr": "100", "source": "explicit"},
        "large": {"group": "neural:params", "rank": (400.0,), "rank_repr": "400", "source": "explicit"},
        "custom": {"group": "neural:custom", "rank": (1.0,), "rank_repr": "1", "source": "explicit"},
    }

    kept, best_id, _ = monotone_one_se_prune(
        rows,
        {"small", "medium", "large", "custom"},
        complexity,
        one_se_multiplier=1.0,
        min_survivors=1,
    )

    assert best_id == "medium"
    assert kept == {"medium", "large", "custom"}
    by_id = {row.candidate_id: row for row in rows}
    assert by_id["small"].prune_reason == "outside_one_se_simpler"
    assert by_id["large"].prune_reason == "protected_larger_or_equal"
    assert by_id["custom"].prune_reason == "protected_incomparable"


def test_product_staged_cv_keeps_larger_neural_prefix_loser_until_final(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeModel:
        def __init__(self, loss: float) -> None:
            self.history = [{"loss": float(loss)}]

        def predict_state_action_ratio(self, states, actions, *, clip=True):
            del actions, clip
            return np.ones(np.asarray(states).shape[0], dtype=np.float64)

    def fake_fit_family(**kwargs):
        occupancy = kwargs["configs"]["occupancy"]
        dims = tuple(int(width) for width in occupancy.hidden_dims)
        iteration = int(occupancy.num_iterations)
        if dims == (4,):
            loss = 5.0
        elif dims == (8,):
            loss = 0.0 if iteration <= 1 else 3.0
        else:
            loss = 10.0 if iteration <= 1 else 0.0
        return FakeModel(loss)

    def fake_moment_metrics(**kwargs):
        del kwargs
        return {
            "moment_balance": 0.1,
            "moment_balance_max_group": 0.1,
            "selection_risk": 0.1,
            "selection_risk_raw": 0.1,
            "selection_effective_dim": 1.0,
            "selection_complexity_penalty": 0.0,
        }

    monkeypatch.setattr(tuning_impl, "_fit_family", fake_fit_family)
    monkeypatch.setattr(tuning_impl, "_heldout_moment_balance_metrics", fake_moment_metrics)
    S = np.zeros((12, 2), dtype=np.float64)
    A = np.zeros((12, 1), dtype=np.float64)
    folds = [np.arange(0, 6), np.arange(6, 12)]
    candidates = [
        {
            "candidate_id": "neural_000",
            "candidate_label": "small",
            "family": "neural",
            "overrides": {
                "occupancy": {"hidden_dims": (4,)},
                "_meta": {"complexity_group": "neural_size_ladder", "complexity_rank": 1},
            },
        },
        {
            "candidate_id": "neural_001",
            "candidate_label": "medium",
            "family": "neural",
            "overrides": {
                "occupancy": {"hidden_dims": (8,)},
                "_meta": {"complexity_group": "neural_size_ladder", "complexity_rank": 2},
            },
        },
        {
            "candidate_id": "neural_002",
            "candidate_label": "large",
            "family": "neural",
            "overrides": {
                "occupancy": {"hidden_dims": (16,)},
                "_meta": {"complexity_group": "neural_size_ladder", "complexity_rank": 3},
            },
        },
    ]

    result, _, final_losses = tuning_impl._run_occupancy_staged_cv(
        candidates=candidates,
        folds=folds,
        S=S,
        A=A,
        S_next=S,
        A_pi=A,
        gamma=0.9,
        S_initial=None,
        A_initial=None,
        initial_weights=None,
        A_pi_next=A,
        rewards=None,
        space=tuning_impl.OccupancySearchSpace(),
        cfg=OccupancyTuningConfig(
            families=("neural",),
            staged_bootstrap_cv=True,
            staged_cv_iterations=2,
            staged_cv_n_bootstrap=0,
            staged_cv_loss_metric="validation_loss",
        ),
        seed=23,
        initial_ratio_mode="auto",
        one_step_ratio_mode="auto",
        first_stage_by_family={"neural": None},
        moment_block_cache=None,
    )

    rows = {(row.candidate_id, row.stage): row for row in result.candidate_rows}
    assert rows[("neural_000", 1)].pruned
    assert rows[("neural_000", 1)].prune_reason == "outside_one_se_simpler"
    assert not rows[("neural_002", 1)].pruned
    assert rows[("neural_002", 1)].prune_reason == "protected_larger_or_equal"
    assert result.selected_candidate_id == "neural_002"
    assert final_losses == {"neural_002": pytest.approx(0.0)}


def test_occupancy_neural_complexity_infers_parameter_count_for_occupancy_only() -> None:
    space = tuning_impl.OccupancySearchSpace()
    candidates = [
        {
            "candidate_id": "shallow",
            "family": "neural",
            "overrides": {"occupancy": {"hidden_dims": (128,)}, "action_ratio": {"hidden_dims": (512,)}},
        },
        {
            "candidate_id": "deep",
            "family": "neural",
            "overrides": {"occupancy": {"hidden_dims": (64, 64)}, "action_ratio": {"hidden_dims": (8,)}},
        },
    ]

    complexity = tuning_impl._occupancy_candidate_complexity_map(candidates, space, input_dim=4)

    assert complexity["shallow"]["group"] == complexity["deep"]["group"]
    assert complexity["deep"]["rank"][0] > complexity["shallow"]["rank"][0]


def test_staged_evaluation_preserves_explicit_source_and_one_step_modes(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, str]] = []

    class FakeModel:
        history = [{"validation_loss": 0.25}]

        def predict_state_action_ratio(self, states, actions, *, clip=True):
            del actions, clip
            return np.ones(np.asarray(states).shape[0], dtype=np.float64)

    def fake_fit_family(**kwargs):
        calls.append((kwargs["initial_ratio_mode"], kwargs["one_step_ratio_mode"]))
        return FakeModel()

    def fake_moment_metrics(**kwargs):
        del kwargs
        return {
            "moment_balance": 0.1,
            "moment_balance_max_group": 0.1,
            "selection_risk": 0.1,
            "selection_risk_raw": 0.1,
            "selection_effective_dim": 1.0,
            "selection_complexity_penalty": 0.0,
        }

    monkeypatch.setattr(tuning_impl, "_fit_family", fake_fit_family)
    monkeypatch.setattr(tuning_impl, "_heldout_moment_balance_metrics", fake_moment_metrics)
    S = np.arange(8, dtype=np.float64).reshape(-1, 1)
    A = np.zeros((8, 1), dtype=np.float64)
    folds = [np.arange(0, 4), np.arange(4, 8)]
    first_stage = {
        "selected_initial_ratio_mode": "joint",
        "selected_one_step_ratio_mode": "direct",
        "selected_configs": {},
        "fold_bundles": [{}, {}],
    }

    tuning_impl._evaluate_candidate(
        candidate={
            "candidate_id": "boosted_modes",
            "candidate_label": "boosted_modes",
            "family": "boosted",
            "overrides": {"modes": {"initial_ratio_mode": "factored", "one_step_ratio_mode": "factored"}},
        },
        budget_stage="full",
        screen_fraction=1.0,
        folds=folds,
        S=S,
        A=A,
        S_next=S,
        A_pi=A,
        gamma=0.9,
        S_initial=S,
        A_initial=A,
        initial_weights=None,
        A_pi_next=A,
        rewards=None,
        space=tuning_impl.OccupancySearchSpace(),
        cfg=OccupancyTuningConfig(families=("boosted",), staged_bootstrap_cv=True),
        seed=11,
        initial_ratio_mode="auto",
        one_step_ratio_mode="auto",
        first_stage=first_stage,
    )

    assert calls
    assert set(calls) == {("factored", "factored")}


def _candidate_with_losses(candidate_id: str, losses: list[float], complexity_rank: int | None = None) -> CandidateResult:
    folds = [
        FoldResult(
            candidate_id=candidate_id,
            family="boosted",
            budget_stage="full",
            fold=idx,
            runtime_sec=0.01,
            moment_balance=loss,
            moment_balance_max_group=loss,
            validation_loss=loss,
            norm_error=0.0,
            ess_fraction=1.0,
            p99=1.0,
            max_weight=1.0,
            clipped_fraction=0.0,
        )
        for idx, loss in enumerate(losses)
    ]
    return CandidateResult(
        candidate_id=candidate_id,
        candidate_label=candidate_id,
        family="boosted",
        budget_stage="full",
        overrides={}
        if complexity_rank is None
        else {"_meta": {"complexity_group": "boosted_size_ladder", "complexity_rank": complexity_rank}},
        fold_results=folds,
        metrics={},
    )
