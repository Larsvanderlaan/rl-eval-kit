from __future__ import annotations

import inspect

import numpy as np
import pytest

from fqe import (
    BoostedFQEConfig,
    FQEFoldResult,
    FQESearchSpace,
    FQEStagedCVFoldTelemetry,
    FQETargetValidationResult,
    FQETuningConfig,
    FQETuningResult,
    NeuralFQEConfig,
    tune_fqe,
    tune_fqe_auto,
    tune_fqe_with_target_validation,
)


def _constant_space() -> FQESearchSpace:
    base = BoostedFQEConfig.stable_defaults(
        num_iterations=4,
        validation_fraction=0.25,
        patience=2,
        refit_on_all_data=True,
        infer_value_bounds=False,
        seed=7,
    )
    return FQESearchSpace(
        boosted=base,
        boosted_candidates=(
            {},
            {"loss": "squared"},
            {"huber_delta_scale": 2.0},
        ),
    )


def test_tune_fqe_auto_value_mode_smoke_and_telemetry() -> None:
    states = np.zeros((24, 1))
    rewards = np.ones(24)
    result = tune_fqe_auto(
        states=states,
        next_states=states,
        rewards=rewards,
        gamma=0.5,
        initial_states=np.zeros((5, 1)),
        search_space=_constant_space(),
        config=FQETuningConfig(budget="fast", cv_folds=3, max_candidates=3, promotion_candidates=1),
    )
    assert isinstance(result, FQETuningResult)
    assert result.model is not None
    assert result.selected_family == "boosted"
    assert result.selected_candidate_id == "boosted_000"
    assert result.candidate_rows()
    assert result.fold_rows()
    assert any(row["promoted"] for row in result.candidate_rows() if row["candidate_id"] == "boosted_000")
    assert np.isfinite(result.model.estimate_policy_value(np.zeros((3, 1))))


def test_tune_fqe_q_mode_and_grouped_folds_keep_groups_intact() -> None:
    states = np.zeros((30, 1))
    actions = np.zeros((30, 1))
    next_actions = np.zeros((30, 1))
    rewards = np.linspace(0.0, 1.0, 30)
    groups = np.repeat(np.arange(10), 3)
    result = tune_fqe(
        states=states,
        actions=actions,
        next_states=states,
        next_actions=next_actions,
        rewards=rewards,
        gamma=0.0,
        groups=groups,
        initial_states=states[:6],
        initial_actions=actions[:6],
        search_space=_constant_space(),
        config=FQETuningConfig(budget="fast", cv_folds=2, max_candidates=2, promotion_candidates=1),
    )
    assert result.model is not None
    for fold in result.fold_rows():
        if fold["budget_stage"] != "full":
            continue
        assert fold["n_train"] % 3 == 0
        assert fold["n_validation"] % 3 == 0


def test_grouped_cv_rejects_too_few_groups() -> None:
    states = np.zeros((9, 1))
    with pytest.raises(ValueError, match="distinct group"):
        tune_fqe(
            states=states,
            next_states=states,
            rewards=np.zeros(9),
            gamma=0.0,
            groups=np.repeat(np.arange(2), [4, 5]),
            search_space=_constant_space(),
            config=FQETuningConfig(budget="fast", cv_folds=3, max_candidates=1, promotion_candidates=1),
        )


def test_fold_metrics_are_weighted_by_validation_mass() -> None:
    import fqe.tuning as tuning

    folds = [
        FQEFoldResult("c", "boosted", "full", 0, 0.0, bellman_risk=1.0, calibration_error=10.0, validation_weight_sum=1.0),
        FQEFoldResult("c", "boosted", "full", 1, 0.0, bellman_risk=9.0, calibration_error=2.0, validation_weight_sum=3.0),
    ]
    metrics = tuning._aggregate_fold_metrics(folds, runtime_sec=0.5)
    assert metrics["bellman_risk"] == pytest.approx(7.0)
    assert metrics["calibration_error"] == pytest.approx(4.0)


def test_tune_fqe_uses_sample_weights_in_refit() -> None:
    states = np.zeros((20, 1))
    next_states = np.zeros((20, 1))
    rewards = np.concatenate([np.zeros(10), np.full(10, 10.0)])
    light_weights = np.ones(20)
    heavy_tail_weights = np.concatenate([np.ones(10), np.full(10, 20.0)])
    cfg = FQETuningConfig(budget="fast", cv_folds=2, max_candidates=1, promotion_candidates=1)
    unweighted = tune_fqe(
        states=states,
        next_states=next_states,
        rewards=rewards,
        gamma=0.0,
        sample_weight=light_weights,
        search_space=_constant_space(),
        config=cfg,
    )
    weighted = tune_fqe(
        states=states,
        next_states=next_states,
        rewards=rewards,
        gamma=0.0,
        sample_weight=heavy_tail_weights,
        search_space=_constant_space(),
        config=cfg,
    )
    assert unweighted.model is not None
    assert weighted.model is not None
    pred_unweighted = unweighted.model.predict_value(np.zeros((1, 1)))[0]
    pred_weighted = weighted.model.predict_value(np.zeros((1, 1)))[0]
    assert pred_weighted > pred_unweighted + 3.0


def test_tune_fqe_rejects_mismatched_q_inputs() -> None:
    states = np.zeros((8, 1))
    rewards = np.ones(8)
    try:
        tune_fqe(states=states, actions=np.zeros((8, 1)), next_states=states, rewards=rewards, gamma=0.0)
    except ValueError as exc:
        assert "actions and next_actions" in str(exc)
    else:
        raise AssertionError("expected mismatched Q inputs to fail")


def test_tune_fqe_staged_bootstrap_cv_selects_final_stage_and_forces_baseline(monkeypatch) -> None:
    import fqe.tuning as tuning

    calls = []

    def fake_fit_family(**kwargs):
        cfg = kwargs["config"]
        calls.append((cfg.loss, cfg.num_iterations))
        if cfg.loss == "squared":
            return _FakeFQEModel(0.0, mode="value")
        if cfg.huber_delta_scale == 2.0:
            return _FakeFQEModel(2.0, mode="value")
        return _FakeFQEModel(10.0, mode="value")

    monkeypatch.setattr(tuning, "_fit_family", fake_fit_family)
    states = np.zeros((12, 1))
    result = tune_fqe(
        states=states,
        next_states=states,
        rewards=np.zeros(12),
        gamma=0.0,
        initial_states=np.zeros((3, 1)),
        search_space=_constant_space(),
        config=FQETuningConfig(
            families=("boosted",),
            max_candidates=3,
            cv_folds=2,
            staged_bootstrap_cv=True,
            staged_cv_iterations=(1, 2),
            staged_cv_bootstrap_samples=20,
            seed=11,
        ),
    )
    assert result.selected_candidate_id == "boosted_001"
    assert result.model is not None
    assert isinstance(FQEStagedCVFoldTelemetry("x", "boosted", 0, 1, 0, 0.0, 0.0, 0.0, 0.0, 1, 1), FQEStagedCVFoldTelemetry)
    assert any(row["budget_stage"] == "staged_1" for row in result.fold_rows())
    baseline = next(candidate for candidate in result.candidates if candidate.candidate_id == "boosted_000")
    assert "baseline_forced" in str(baseline.metrics["staged_cv_candidate_telemetry"])
    assert any(iteration == 2 for _, iteration in calls)


def test_fqe_monotone_pruning_keeps_larger_early_loser() -> None:
    from fqe.staged_cv import FQEStagedCVStageTelemetry, monotone_one_se_prune

    rows = [
        FQEStagedCVStageTelemetry("small", "neural", 1, 1, 9.0, 0.0),
        FQEStagedCVStageTelemetry("medium", "neural", 1, 1, 0.0, 0.0),
        FQEStagedCVStageTelemetry("large", "neural", 1, 1, 16.0, 0.0),
        FQEStagedCVStageTelemetry("custom", "neural", 1, 1, 25.0, 0.0),
    ]
    complexity = {
        "small": {"group": "neural:params", "rank": (25.0,), "rank_repr": "25", "source": "explicit"},
        "medium": {"group": "neural:params", "rank": (100.0,), "rank_repr": "100", "source": "explicit"},
        "large": {"group": "neural:params", "rank": (400.0,), "rank_repr": "400", "source": "explicit"},
        "custom": {"group": "neural:custom", "rank": (1.0,), "rank_repr": "1", "source": "explicit"},
    }

    kept, best_id, threshold = monotone_one_se_prune(
        rows,
        {"small", "medium", "large", "custom"},
        complexity,
        one_se_multiplier=1.0,
        min_survivors=1,
    )

    assert best_id == "medium"
    assert threshold == pytest.approx(0.0)
    assert kept == {"medium", "large", "custom"}
    by_id = {row.candidate_id: row for row in rows}
    assert by_id["small"].pruned
    assert by_id["small"].prune_reason == "outside_one_se_simpler"
    assert not by_id["large"].pruned
    assert by_id["large"].prune_reason == "protected_larger_or_equal"
    assert not by_id["custom"].pruned
    assert by_id["custom"].prune_reason == "protected_incomparable"


def test_tune_fqe_staged_bootstrap_cv_neural_large_survives_and_wins(monkeypatch) -> None:
    import fqe.tuning as tuning

    calls = []

    def fake_fit_family(**kwargs):
        cfg = kwargs["config"]
        dims = tuple(int(width) for width in cfg.hidden_dims)
        iteration = int(cfg.num_iterations)
        calls.append((dims, iteration))
        if dims == (4,):
            value = 5.0
        elif dims == (8,):
            value = 0.0 if iteration <= 1 else 3.0
        else:
            value = 10.0 if iteration <= 1 else 0.0
        return _FakeFQEModel(value, mode="value")

    monkeypatch.setattr(tuning, "_fit_family", fake_fit_family)
    states = np.zeros((12, 2))
    space = FQESearchSpace(
        neural=NeuralFQEConfig.stable_defaults(
            hidden_dims=(8,),
            num_iterations=2,
            gradient_steps_per_iteration=1,
            patience=1,
            validation_fraction=0.25,
        ),
        neural_candidates=(
            {"hidden_dims": (4,), "_meta": {"complexity_group": "neural_size_ladder", "complexity_rank": 1}},
            {"hidden_dims": (8,), "_meta": {"complexity_group": "neural_size_ladder", "complexity_rank": 2}},
            {"hidden_dims": (16,), "_meta": {"complexity_group": "neural_size_ladder", "complexity_rank": 3}},
        ),
    )

    result = tune_fqe(
        states=states,
        next_states=states,
        rewards=np.zeros(12),
        gamma=0.0,
        search_space=space,
        config=FQETuningConfig(
            families=("neural",),
            max_candidates=3,
            cv_folds=2,
            staged_bootstrap_cv=True,
            staged_cv_iterations=(1, 2),
            staged_cv_bootstrap_samples=0,
            seed=17,
        ),
    )

    assert result.selected_candidate_id == "neural_002"
    stage_rows = [row for row in result.staged_cv_rows() if row["row_type"] == "candidate_stage"]
    stage1 = {(row["candidate_id"], row["stage"]): row for row in stage_rows}
    assert stage1[("neural_000", 1)]["pruned"]
    assert stage1[("neural_000", 1)]["prune_reason"] == "outside_one_se_simpler"
    assert not stage1[("neural_002", 1)]["pruned"]
    assert stage1[("neural_002", 1)]["prune_reason"] == "protected_larger_or_equal"
    assert stage1[("neural_002", 2)]["selected"]
    assert any(dims == (16,) and iteration == 2 for dims, iteration in calls)


def test_fqe_neural_complexity_infers_parameter_count_across_depth() -> None:
    from fqe.staged_cv import _candidate_complexity_map

    space = FQESearchSpace(
        neural=NeuralFQEConfig.stable_defaults(hidden_dims=(16,), num_iterations=2),
        neural_candidates=(
            {"hidden_dims": (128,)},
            {"hidden_dims": (64, 64)},
        ),
    )
    candidates = [
        {"candidate_id": "shallow", "family": "neural", "overrides": {"hidden_dims": (128,)}},
        {"candidate_id": "deep", "family": "neural", "overrides": {"hidden_dims": (64, 64)}},
    ]

    complexity = _candidate_complexity_map(candidates, space, input_dim=4)

    assert complexity["shallow"]["group"] == complexity["deep"]["group"]
    assert complexity["deep"]["rank"][0] > complexity["shallow"]["rank"][0]


class _FakeFQEModel:
    def __init__(self, value: float, mode: str = "q") -> None:
        self.value = float(value)
        self.mode = mode

    def predict_q(self, states, actions):
        return np.full(np.asarray(states).shape[0], self.value, dtype=np.float64)

    def predict_value(self, states):
        return np.full(np.asarray(states).shape[0], self.value, dtype=np.float64)

    def estimate_policy_value(self, initial_states, initial_actions=None, initial_weights=None):
        values = np.full(np.asarray(initial_states).shape[0], self.value, dtype=np.float64)
        if initial_weights is None:
            return float(np.mean(values))
        return float(np.average(values, weights=np.asarray(initial_weights, dtype=np.float64).reshape(-1)))


def test_tune_fqe_target_validation_n_step_selects_lower_residual(monkeypatch) -> None:
    import fqe.tuning as tuning

    def fake_fit_family(**kwargs):
        cfg = kwargs["config"]
        return _FakeFQEModel(1.25 if cfg.loss == "squared" else 0.0, mode="q")

    monkeypatch.setattr(tuning, "_fit_family", fake_fit_family)
    states = np.zeros((6, 1))
    actions = np.zeros((6, 1))
    result = tune_fqe_with_target_validation(
        states=states,
        actions=actions,
        next_states=states,
        next_actions=actions,
        rewards=np.zeros(6),
        gamma=0.5,
        search_space=_constant_space(),
        config=FQETuningConfig(families=("boosted",), max_candidates=2, seed=5),
        validation_states=np.zeros((2, 1)),
        validation_actions=np.zeros((2, 1)),
        validation_rewards=np.ones(2),
        validation_next_states=np.zeros((2, 1)),
        validation_episode_ids=np.array([0, 0]),
        validation_timestep=np.array([0, 1]),
        validation_terminals=np.array([0, 1]),
        validation_tail_actions=np.zeros((2, 1)),
    )
    assert isinstance(result, FQETargetValidationResult)
    assert result.selected_candidate_id == "boosted_001"
    rows = {row["candidate_id"]: row for row in result.candidate_rows()}
    assert rows["boosted_001"]["score"] < rows["boosted_000"]["score"]


def test_tune_fqe_target_validation_scalar_value_and_truncation_diagnostics(monkeypatch) -> None:
    import fqe.tuning as tuning

    def fake_fit_family(**kwargs):
        cfg = kwargs["config"]
        return _FakeFQEModel(3.0 if cfg.loss == "squared" else 0.0, mode="value")

    monkeypatch.setattr(tuning, "_fit_family", fake_fit_family)
    states = np.zeros((6, 1))
    result = tune_fqe_with_target_validation(
        states=states,
        next_states=states,
        rewards=np.zeros(6),
        gamma=0.5,
        initial_states=np.zeros((3, 1)),
        search_space=_constant_space(),
        config=FQETuningConfig(families=("boosted",), max_candidates=2, seed=6),
        score_mode="scalar_value",
        target_value=3.1,
        target_value_se=10.0,
    )
    assert result.selected_candidate_id == "boosted_001"
    assert result.selection_rule == "min_score"
    assert result.validation_diagnostics["validation_selection_rule"] == "min_score"
    assert result.validation_diagnostics["selected_min_score_candidate_id"] == "boosted_001"
    assert result.validation_diagnostics["selected_one_se_candidate_id"] == "boosted_000"
    assert result.validation_diagnostics["validation_label_scope"] == "scalar_value_only"

    one_se_result = tune_fqe_with_target_validation(
        states=states,
        next_states=states,
        rewards=np.zeros(6),
        gamma=0.5,
        initial_states=np.zeros((3, 1)),
        search_space=_constant_space(),
        config=FQETuningConfig(families=("boosted",), max_candidates=2, seed=6),
        score_mode="scalar_value",
        selection_rule="one_se",
        target_value=3.1,
        target_value_se=10.0,
    )
    assert one_se_result.selected_candidate_id == "boosted_000"
    assert one_se_result.selection_rule == "one_se"
    assert one_se_result.validation_diagnostics["validation_selection_rule"] == "one_se"

    td_result = tune_fqe_with_target_validation(
        states=states,
        next_states=states,
        rewards=np.zeros(6),
        gamma=0.5,
        search_space=_constant_space(),
        config=FQETuningConfig(families=("boosted",), max_candidates=1, seed=7),
        validation_states=np.zeros((2, 1)),
        validation_rewards=np.ones(2),
        validation_next_states=np.zeros((2, 1)),
        validation_episode_ids=np.array([0, 0]),
        validation_timestep=np.array([0, 1]),
        validation_continuation=np.array([1, 1]),
    )
    assert float(td_result.validation_diagnostics["truncation_tail_mass_mean"]) > 0.0


def test_tune_fqe_target_validation_rejects_invalid_selection_rule() -> None:
    states = np.zeros((4, 1))
    with pytest.raises(ValueError, match="selection_rule"):
        tune_fqe_with_target_validation(
            states=states,
            next_states=states,
            rewards=np.zeros(4),
            gamma=0.5,
            search_space=_constant_space(),
            config=FQETuningConfig(families=("boosted",), max_candidates=1, seed=8),
            selection_rule="stable",  # type: ignore[arg-type]
            validation_states=np.zeros((2, 1)),
            validation_rewards=np.ones(2),
            validation_next_states=np.zeros((2, 1)),
            validation_episode_ids=np.array([0, 0]),
            validation_timestep=np.array([0, 1]),
            validation_terminals=np.array([0, 1]),
        )


def test_tune_fqe_target_validation_signature_default_selection_rule() -> None:
    import fqe
    import fqe.tuning as tuning

    signature = inspect.signature(tune_fqe_with_target_validation)
    assert signature.parameters["selection_rule"].default == "min_score"
    assert inspect.signature(fqe.tune_fqe_with_target_validation).parameters["selection_rule"].default == "min_score"
    assert inspect.signature(tuning.tune_fqe_with_target_validation).parameters["selection_rule"].default == "min_score"
