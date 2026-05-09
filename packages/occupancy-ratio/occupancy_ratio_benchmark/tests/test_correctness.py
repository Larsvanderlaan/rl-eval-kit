from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

from occupancy_ratio_benchmark.config import OccupancyRatioBenchmarkConfig
from occupancy_ratio_benchmark.conservatism_audit import (
    build_conservatism_audit_rows,
    render_conservatism_report,
    write_conservatism_audit,
)
from occupancy_ratio_benchmark.diagnostics import summarize_weights
from occupancy_ratio_benchmark.defaults_report import generate_defaults_report
from occupancy_ratio_benchmark.neural_default_ablation import (
    CANDIDATES,
    _tag_rows,
    render_ablation_report,
    summarize_ablation_rows,
)
from occupancy_ratio_benchmark.moment_evaluator_ablation import (
    paired_selection_delta_rows,
    render_evaluator_report,
    summarize_evaluator_rows,
)

from occupancy_ratio.fit_importance_and_transition_ratios import (
    fit_importance_ratio_lgbm,
    fit_state_density_ratio_lgbm,
    fit_transition_ratio_lgbm,
    importance_ratio_objective,
    transition_ratio_objective,
)
from occupancy_ratio.fit_occupancy_ratio import (
    ActionRatioConfig,
    DiscountedOccupancyRatioModel,
    OccupancyRegressionConfig,
    SourceStateRatioConfig,
    TransitionRatioConfig,
    _clip_pseudo_outcomes,
    _damped_update,
    _make_fold_indices,
    _make_occupancy_sample_weights,
    _make_occupancy_objective,
    _occupancy_loss_value,
    _project_nonnegative_normalized,
    fit_discounted_occupancy_ratio,
    make_direct_adjoint_occupancy_dataset,
    make_forward_occupancy_dataset,
    tune_discounted_occupancy_ratio_cv,
)
from occupancy_ratio.fit_occupancy_ratio_neural import (
    NeuralActionRatioConfig,
    NeuralDiscountedOccupancyRatioModel,
    NeuralOccupancyRegressionConfig,
    NeuralSourceStateRatioConfig,
    NeuralTransitionRatioConfig,
    _NeuralTargetBuilder,
    _RatioPredictor,
    _weighted_binary_ratio_loss,
    _weighted_source_lsif_loss,
    fit_action_ratio_neural,
    fit_discounted_occupancy_ratio_neural,
    fit_source_state_ratio_neural,
    fit_transition_ratio_neural,
    tune_discounted_occupancy_ratio_neural_cv,
)
from occupancy_ratio.tuning import (
    CandidateResult,
    OccupancySearchSpace,
    OccupancyTuningConfig,
    _FoldFeatureBuilder,
    _should_fallback_to_baseline,
    tune_occupancy_ratio_auto,
)
from occupancy_ratio_benchmark.external_baselines import preflight_google_dualdice
from occupancy_ratio_benchmark.discrete import exact_ratio_table, make_chain_mdp, make_discrete_dataset
from occupancy_ratio_benchmark.estimators import estimate_boosted_tree, estimate_neural_network, estimate_oracle
from occupancy_ratio_benchmark.gym_control import make_gym_control_dataset
from occupancy_ratio_benchmark.io import write_csv
from occupancy_ratio_benchmark import runner as runner_module
from occupancy_ratio_benchmark.runner import (
    _expanded_estimators,
    _high_stakes_diagnostic_status,
    _policy_shifts_for_setting,
    make_high_stakes_recommendations,
    make_winner_table,
    run_benchmark,
)
from occupancy_ratio_benchmark.run import load_config_file, parse_args as parse_benchmark_args
from occupancy_ratio_benchmark.tabular import (
    OptionalDatasetUnavailable,
    make_openml_contextual_bandit_from_arrays,
    make_openml_finite_mdp_from_arrays,
    solve_discounted_occupancy,
)


class _ConstantBooster:
    def __init__(self, value: float) -> None:
        self.value = float(value)

    def predict(self, x, **kwargs):
        return np.full(np.asarray(x).shape[0], self.value, dtype=np.float64)


class _FakeRatioModel:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=np.float64)
        self.state_action_calls = 0
        self.state_ratio_calls = 0

    def predict_state_action_ratio(self, states, actions, *, clip=True):
        del actions, clip
        self.state_action_calls += 1
        return self.values[: np.asarray(states).shape[0]]

    def predict_state_ratio(self, states, actions, *, clip=True):
        del states, actions, clip
        self.state_ratio_calls += 1
        raise AssertionError("comparison must not use state-ratio predictions")


def test_boosted_public_prediction_matches_training_state_with_damping() -> None:
    dataset = make_discrete_dataset(setting="discrete_chain", gamma=0.5, sample_size=60, seed=123)
    occupancy = OccupancyRegressionConfig(
        num_iterations=3,
        trees_per_iteration=1,
        mcmc_samples=2,
        batch_size=64,
        fixed_point_damping=0.5,
        early_stopping=False,
        show_progress=False,
        lgb_params={"min_data_in_leaf": 5, "num_leaves": 7, "verbose": -1},
    )
    action = ActionRatioConfig(
        num_boost_round=2,
        early_stopping_rounds=0,
        show_progress=False,
        lgb_params={"min_data_in_leaf": 5, "num_leaves": 7, "verbose": -1},
    )
    source = SourceStateRatioConfig(
        num_boost_round=2,
        early_stopping_rounds=0,
        show_progress=False,
        lgb_params={"min_data_in_leaf": 5, "num_leaves": 7, "verbose": -1},
    )
    transition = TransitionRatioConfig(
        num_boost_round=2,
        permutation_samples=1,
        early_stopping_rounds=0,
        show_progress=False,
        lgb_params={"min_data_in_leaf": 5, "num_leaves": 7, "verbose": -1},
    )
    model = fit_discounted_occupancy_ratio(
        states=dataset.states,
        actions=dataset.actions,
        next_states=dataset.next_states,
        target_actions=dataset.target_actions,
        gamma=dataset.gamma,
        occupancy=occupancy,
        action_ratio=action,
        source_state_ratio=source,
        transition_ratio=transition,
    )
    public = model.predict_state_action_ratio(dataset.states, dataset.actions)
    training_state = model.legacy_result["pred_state_action_ratio_beh"]
    assert np.allclose(public, training_state, atol=1e-7, rtol=1e-7)


def test_neural_public_prediction_matches_training_state_with_damping() -> None:
    pytest.importorskip("torch")
    dataset = make_discrete_dataset(setting="discrete_chain", gamma=0.5, sample_size=30, seed=321)
    occupancy = NeuralOccupancyRegressionConfig(
        num_iterations=2,
        gradient_steps_per_iteration=1,
        mcmc_samples=1,
        batch_size=16,
        hidden_dims=(8,),
        fixed_point_damping=0.5,
        early_stopping=False,
        seed=1,
    )
    model = fit_discounted_occupancy_ratio_neural(
        states=dataset.states,
        actions=dataset.actions,
        next_states=dataset.next_states,
        target_actions=dataset.target_actions,
        gamma=float(dataset.gamma),
        occupancy=occupancy,
        action_ratio=NeuralActionRatioConfig(max_steps=1, batch_size=16, hidden_dims=(8,), seed=2),
        source_state_ratio=NeuralSourceStateRatioConfig(max_steps=1, batch_size=16, hidden_dims=(8,), seed=3),
        transition_ratio=NeuralTransitionRatioConfig(
            max_steps=1,
            batch_size=16,
            permutation_samples=1,
            hidden_dims=(8,),
            seed=4,
        ),
    )
    public = model.predict_state_action_ratio(dataset.states, dataset.actions)
    training_state = model.legacy_result["pred_state_action_ratio_beh"]
    assert np.allclose(public, training_state, atol=1e-7, rtol=1e-7)


def test_config_presets_are_ordered_and_overrideable() -> None:
    stable = OccupancyRegressionConfig.stable_defaults()
    balanced = OccupancyRegressionConfig.balanced_defaults()
    dice = OccupancyRegressionConfig.dualdice_comparable_defaults(occupancy_ratio_max=999.0)
    assert stable.fixed_point_damping == 0.5
    assert balanced.fixed_point_damping == 1.0
    assert balanced.direct_adjoint_num_boost_round >= 128
    assert dice.normalize_occupancy is False
    assert dice.occupancy_ratio_max == 999.0
    assert ActionRatioConfig.balanced_defaults().prediction_max == 200.0
    assert SourceStateRatioConfig.dualdice_comparable_defaults().prediction_max is None
    assert TransitionRatioConfig.balanced_defaults().permutation_samples == 50
    assert NeuralOccupancyRegressionConfig.balanced_defaults().direct_adjoint_steps == 128
    assert NeuralActionRatioConfig.dualdice_comparable_defaults().normalization_penalty == 0.0
    assert NeuralTransitionRatioConfig.balanced_defaults().prediction_max == 200.0


def test_direct_adjoint_hyperparameters_are_reported() -> None:
    x_next = np.arange(12, dtype=np.float64).reshape(6, 2)
    x_query = np.arange(16, dtype=np.float64).reshape(8, 2) / 10.0
    builder = make_direct_adjoint_occupancy_dataset(
        X_sa_successor=x_next,
        X_sa_query=x_query,
        c_ratio_query=np.full(8, 2.0),
        w_source_query=np.full(8, 3.0),
        gamma=0.5,
        seed=10,
        num_boost_round=7,
        lgb_params={"min_data_in_leaf": 1, "num_leaves": 3, "verbose": -1},
        loss="squared",
        sample_weight_mode="sqrt_target",
        sample_weight_max=5.0,
    )
    out = builder(w_beh=np.linspace(1.0, 2.0, 6), clip_y_min=None)
    assert out["diag"]["direct_adjoint_num_boost_round"] == 7.0
    assert out["diag"]["direct_adjoint_loss"] == "squared"
    assert out["diag"]["direct_adjoint_sample_weight_mode"] == "sqrt_target"


def test_comparison_uses_state_action_ratio_and_matched_postprocessing() -> None:
    import json

    from occupancy_ratio.comparison import compare_fori_to_google_dualdice

    states = np.zeros((4, 1))
    actions = np.zeros((4, 1))
    fori = _FakeRatioModel([0.0, 2.0, 10.0, np.inf])
    google = _FakeRatioModel([1.0, 4.0, 8.0, 16.0])
    out = compare_fori_to_google_dualdice(fori, google, states, actions, cap=5.0, normalize=True, rewards=np.arange(4.0))
    assert fori.state_action_calls == 1
    assert google.state_action_calls == 1
    assert fori.state_ratio_calls == 0
    assert google.state_ratio_calls == 0
    assert out["object"] == "state_action_ratio"
    assert out["fori"]["max"] <= 5.0
    assert out["google"]["max"] <= 5.0
    assert "reward_value_fori" in out
    json.dumps(out, allow_nan=True)


def test_diagnostics_weight_summary_postprocess_and_report_are_json_friendly() -> None:
    import json

    from occupancy_ratio.diagnostics import postprocess_weights, regularization_path_report, weight_summary

    raw = np.array([0.0, 2.0, np.inf, np.nan, -1.0])
    processed = postprocess_weights(raw, cap=3.0, normalize=True)
    summary = weight_summary(processed, cap=3.0)
    assert summary["n"] == 5
    assert np.all(np.isfinite(processed))
    result = {
        "loss": "huber",
        "fixed_point_damping": 0.5,
        "normalize_occupancy": True,
        "occupancy_ratio_max": 3.0,
        "clip_pseudo_outcomes": True,
        "pred_iw": np.ones(4),
        "pred_state_action_ratio_beh": processed[:4],
        "pred_state_action_ratio_beh_raw": raw[:4],
        "history": [{"target_raw_max": 9.0, "target_projected_max": 3.0}],
    }
    report = regularization_path_report(result)
    assert "occupancy_stabilized_behavior" in report["summaries"]
    json.dumps(report, allow_nan=True)


def test_action_ratio_no_shift_stays_near_one() -> None:
    rng = np.random.default_rng(123)
    states = rng.normal(size=(250, 2))
    actions = rng.normal(size=(250, 1))
    fit = fit_importance_ratio_lgbm(
        S=states,
        A=actions,
        A_pi=actions.copy(),
        num_boost_round=5,
        early_stopping_rounds=0,
        refit_on_all_data=False,
        show_tqdm=False,
        lgb_params={"min_data_in_leaf": 10, "num_leaves": 7, "verbose": -1},
    )
    assert np.max(np.abs(fit["w_hat"] - 1.0)) < 1e-8


def test_logistic_action_ratio_no_shift_stays_near_one() -> None:
    rng = np.random.default_rng(123)
    states = rng.normal(size=(180, 2))
    actions = rng.normal(size=(180, 1))
    fit = fit_importance_ratio_lgbm(
        S=states,
        A=actions,
        A_pi=actions.copy(),
        density_ratio_loss="logistic",
        logistic_logit_clip=8.0,
        num_boost_round=5,
        early_stopping_rounds=0,
        refit_on_all_data=False,
        show_tqdm=False,
        lgb_params={"min_data_in_leaf": 10, "num_leaves": 7, "verbose": -1},
    )
    assert fit["density_ratio_loss"] == "logistic"
    assert np.isfinite(fit["prior_correction"])
    assert np.isclose(np.mean(fit["w_hat"]), 1.0, atol=0.05)


def test_logistic_transition_ratio_is_finite_and_uses_initial_state_reference() -> None:
    rng = np.random.default_rng(124)
    states = rng.normal(size=(90, 2))
    actions = rng.normal(size=(90, 1))
    next_states = states + rng.normal(loc=2.0, scale=0.1, size=states.shape)
    fit = fit_transition_ratio_lgbm(
        S=states,
        A=actions,
        S_next=next_states,
        density_ratio_loss="logistic",
        logistic_logit_clip=8.0,
        K_perm=2,
        num_boost_round=3,
        early_stopping_rounds=0,
        refit_on_all_data=False,
        show_tqdm=False,
        lgb_params={"min_data_in_leaf": 10, "num_leaves": 7, "verbose": -1},
    )
    assert fit["density_ratio_loss"] == "logistic"
    assert np.all(np.isfinite(fit["k_hat"]))
    assert np.all(fit["k_hat"] >= 0.0)
    assert np.allclose(fit["S_feat"], states.astype(np.float32))
    assert not np.allclose(fit["S_feat"], next_states.astype(np.float32))


def test_source_state_ratio_no_shift_is_finite_and_near_one() -> None:
    rng = np.random.default_rng(125)
    states = rng.normal(size=(140, 2))
    fit = fit_state_density_ratio_lgbm(
        S_ref=states,
        S_num=states.copy(),
        num_boost_round=3,
        early_stopping_rounds=0,
        refit_on_all_data=False,
        show_tqdm=False,
        lgb_params={"min_data_in_leaf": 10, "num_leaves": 7, "verbose": -1},
    )
    assert fit["density_ratio_loss"] == "lsif"
    assert np.all(np.isfinite(fit["source_hat"]))
    assert np.isclose(np.mean(fit["source_hat"]), 1.0, atol=0.05)


def test_neural_source_state_ratio_no_shift_is_finite_and_near_one() -> None:
    rng = np.random.default_rng(126)
    states = rng.normal(size=(80, 2))
    fit = fit_source_state_ratio_neural(
        states,
        states.copy(),
        NeuralSourceStateRatioConfig(max_steps=2, batch_size=32, hidden_dims=(8,), seed=3),
    )
    assert fit["density_ratio_loss"] == "lsif"
    assert np.all(np.isfinite(fit["source_hat"]))
    assert np.isclose(np.mean(fit["source_hat"]), 1.0, atol=0.05)


def test_forward_builder_source_state_ratio_multiplies_action_ratio() -> None:
    x_sa = np.zeros((2, 2), dtype=np.float64)
    x_s = np.zeros((2, 1), dtype=np.float64)
    builder = make_forward_occupancy_dataset(
        bst_k=_ConstantBooster(1.0),
        bst_iw=_ConstantBooster(2.0),
        k_prediction_offset=0.0,
        iw_prediction_offset=0.0,
        X_sa_kernel=x_sa,
        X_s_query=x_s,
        X_sa_iw=x_sa,
        X_sa_query_iw=x_sa,
        gamma=0.0,
        mcmc_samples=1,
        seed=1,
        batch_query=2,
        clip_w_query_max=None,
        clip_k_max=None,
        source_state_ratio_query=np.full(2, 3.0),
    )
    out = builder(w_beh=np.ones(2), w_old_query=np.ones(2), clip_y_min=None)
    assert np.allclose(out["w_query"], 2.0)
    assert np.allclose(out["y"], 6.0)
    assert out["diag"]["source_state_ratio_enabled"]
    assert out["diag"]["source_state_ratio_mean"] == 3.0

    default_builder = make_forward_occupancy_dataset(
        bst_k=_ConstantBooster(1.0),
        bst_iw=_ConstantBooster(2.0),
        k_prediction_offset=0.0,
        iw_prediction_offset=0.0,
        X_sa_kernel=x_sa,
        X_s_query=x_s,
        X_sa_iw=x_sa,
        X_sa_query_iw=x_sa,
        gamma=0.0,
        mcmc_samples=1,
        seed=1,
        batch_query=2,
        clip_w_query_max=None,
        clip_k_max=None,
    )
    default_out = default_builder(w_beh=np.ones(2), w_old_query=np.ones(2), clip_y_min=None)
    assert np.allclose(default_out["y"], 2.0)
    assert not default_out["diag"]["source_state_ratio_enabled"]


def test_forward_builder_respects_terminal_and_known_ratio_overrides() -> None:
    x_sa = np.zeros((3, 2), dtype=np.float64)
    x_s = np.zeros((3, 1), dtype=np.float64)
    builder = make_forward_occupancy_dataset(
        bst_k=_ConstantBooster(1.0),
        bst_iw=None,
        X_sa_kernel=x_sa,
        X_s_query=x_s,
        X_sa_iw=x_sa,
        X_sa_query_iw=x_sa,
        gamma=0.5,
        mcmc_samples=1,
        seed=1,
        batch_query=3,
        clip_k_max=None,
        w_source_query=np.ones(3),
        continuation_query=np.array([1.0, 0.0, 1.0]),
        w_query_override=np.full(3, 2.0),
    )
    out = builder(w_beh=np.ones(3), w_old_query=np.ones(3), clip_y_min=None)
    assert np.allclose(out["w_query"], 2.0)
    assert np.allclose(out["y"], np.array([1.5, 0.5, 1.5]))
    assert out["diag"]["continuation_min"] == 0.0


def test_boosted_terminal_timeout_known_ratio_multi_sample_and_serialization(tmp_path: Path) -> None:
    dataset = make_discrete_dataset(setting="discrete_chain", gamma=0.5, sample_size=36, seed=812)
    target_actions = np.stack([dataset.target_actions, dataset.target_actions], axis=1)
    action_values = np.concatenate(
        [
            np.ones(target_actions.reshape(-1, dataset.actions.shape[1]).shape[0], dtype=np.float64),
            np.full(dataset.actions.shape[0], 1.25, dtype=np.float64),
        ]
    )
    occupancy = OccupancyRegressionConfig(
        num_iterations=2,
        trees_per_iteration=1,
        mcmc_samples=1,
        batch_size=32,
        early_stopping=False,
        show_progress=False,
        lgb_params={"min_data_in_leaf": 5, "num_leaves": 7, "verbose": -1},
    )
    model = fit_discounted_occupancy_ratio(
        states=dataset.states,
        actions=dataset.actions,
        next_states=dataset.next_states,
        target_actions=target_actions,
        gamma=dataset.gamma,
        terminals=np.zeros(dataset.states.shape[0]),
        timeouts=np.r_[1.0, np.zeros(dataset.states.shape[0] - 1)],
        handle_timeouts="terminal",
        action_ratio_values=action_values,
        occupancy=occupancy,
        action_ratio=ActionRatioConfig(num_boost_round=1, early_stopping_rounds=0, show_progress=False),
        transition_ratio_booster=_ConstantBooster(1.0),
    )
    assert model.action_ratio_booster is None
    assert model.diagnostics["known_action_ratio"]
    assert model.diagnostics["num_target_action_samples"] == 2
    assert model.diagnostics["continuation_min"] == 0.0
    assert np.allclose(model.predict_action_ratio(dataset.states, dataset.actions), 1.25)

    path = tmp_path / "boosted.pkl"
    model.save(path)
    loaded = DiscountedOccupancyRatioModel.load(path)
    assert np.allclose(
        loaded.predict_state_action_ratio(dataset.states, dataset.actions),
        model.predict_state_action_ratio(dataset.states, dataset.actions),
    )
    assert np.allclose(loaded.predict_action_ratio(dataset.states, dataset.actions), 1.25)


def test_boosted_timeout_error_mode_rejects_timeouts() -> None:
    dataset = make_discrete_dataset(setting="discrete_chain", gamma=0.5, sample_size=12, seed=11)
    with pytest.raises(ValueError, match="timeouts"):
        fit_discounted_occupancy_ratio(
            states=dataset.states,
            actions=dataset.actions,
            next_states=dataset.next_states,
            target_actions=dataset.target_actions,
            gamma=dataset.gamma,
            timeouts=np.r_[1.0, np.zeros(dataset.states.shape[0] - 1)],
            handle_timeouts="error",
            occupancy=OccupancyRegressionConfig(num_iterations=1, mcmc_samples=1, show_progress=False),
            action_ratio=ActionRatioConfig(num_boost_round=1, early_stopping_rounds=0, show_progress=False),
            transition_ratio=TransitionRatioConfig(
                num_boost_round=1,
                permutation_samples=1,
                early_stopping_rounds=0,
                show_progress=False,
            ),
        )


def test_neural_known_action_ratio_terminal_and_serialization(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    dataset = make_discrete_dataset(setting="discrete_chain", gamma=0.5, sample_size=18, seed=91)
    action_values = np.concatenate(
        [
            np.ones(dataset.states.shape[0], dtype=np.float64),
            np.full(dataset.states.shape[0], 1.5, dtype=np.float64),
        ]
    )
    occupancy = NeuralOccupancyRegressionConfig(
        num_iterations=1,
        gradient_steps_per_iteration=1,
        mcmc_samples=1,
        batch_size=16,
        hidden_dims=(8,),
        early_stopping=False,
        seed=10,
    )
    model = fit_discounted_occupancy_ratio_neural(
        states=dataset.states,
        actions=dataset.actions,
        next_states=dataset.next_states,
        target_actions=dataset.target_actions,
        gamma=float(dataset.gamma),
        terminals=np.r_[1.0, np.zeros(dataset.states.shape[0] - 1)],
        action_ratio_values=action_values,
        occupancy=occupancy,
        action_ratio=NeuralActionRatioConfig(max_steps=1, batch_size=16, hidden_dims=(8,), seed=11),
        source_state_ratio=NeuralSourceStateRatioConfig(max_steps=1, batch_size=16, hidden_dims=(8,), seed=12),
        transition_ratio=NeuralTransitionRatioConfig(
            max_steps=1,
            batch_size=16,
            permutation_samples=1,
            hidden_dims=(8,),
            seed=13,
        ),
    )
    assert model.action_ratio_predictor is None
    assert model.diagnostics["known_action_ratio"]
    assert model.diagnostics["continuation_min"] == 0.0
    assert np.allclose(model.predict_action_ratio(dataset.states, dataset.actions), 1.5)

    path = tmp_path / "neural.pkl"
    model.save(path)
    loaded = NeuralDiscountedOccupancyRatioModel.load(path)
    assert np.allclose(
        loaded.predict_state_action_ratio(dataset.states, dataset.actions),
        model.predict_state_action_ratio(dataset.states, dataset.actions),
    )
    assert np.allclose(loaded.predict_action_ratio(dataset.states, dataset.actions), 1.5)


def test_direct_adjoint_builder_uses_joint_source_and_one_step_ratio() -> None:
    x_next = np.zeros((3, 2), dtype=np.float64)
    x_query = np.zeros((4, 2), dtype=np.float64)
    builder = make_direct_adjoint_occupancy_dataset(
        X_sa_successor=x_next,
        X_sa_query=x_query,
        c_ratio_query=np.full(4, 2.0),
        w_source_query=np.full(4, 3.0),
        gamma=0.0,
        seed=10,
        num_boost_round=1,
        lgb_params={"min_data_in_leaf": 1, "num_leaves": 2, "verbose": -1},
    )
    out = builder(w_beh=np.ones(3), clip_y_min=None)
    assert np.allclose(out["y"], 3.0)
    assert np.allclose(out["w_query"], 2.0)
    assert out["diag"]["one_step_direct_ratio_enabled"]


def test_boosted_joint_and_direct_modes_resolve_and_validate() -> None:
    dataset = make_discrete_dataset(setting="discrete_chain", gamma=0.5, sample_size=45, seed=14)
    occupancy = OccupancyRegressionConfig(
        num_iterations=1,
        trees_per_iteration=1,
        mcmc_samples=1,
        batch_size=64,
        show_progress=False,
        early_stopping=False,
        lgb_params={"min_data_in_leaf": 5, "num_leaves": 7, "verbose": -1},
    )
    action = ActionRatioConfig(
        num_boost_round=1,
        early_stopping_rounds=0,
        refit_on_all_data=False,
        show_progress=False,
        lgb_params={"min_data_in_leaf": 5, "num_leaves": 7, "verbose": -1},
    )
    source = SourceStateRatioConfig(
        num_boost_round=1,
        early_stopping_rounds=0,
        refit_on_all_data=False,
        show_progress=False,
        lgb_params={"min_data_in_leaf": 5, "num_leaves": 7, "verbose": -1},
    )
    transition = TransitionRatioConfig(
        num_boost_round=1,
        permutation_samples=1,
        early_stopping_rounds=0,
        refit_on_all_data=False,
        show_progress=False,
        lgb_params={"min_data_in_leaf": 5, "num_leaves": 7, "verbose": -1},
    )
    model = fit_discounted_occupancy_ratio(
        states=dataset.states,
        actions=dataset.actions,
        next_states=dataset.next_states,
        target_actions=dataset.target_actions,
        target_next_actions=dataset.next_target_actions,
        gamma=dataset.gamma,
        initial_states=dataset.initial_states,
        initial_actions=dataset.initial_actions,
        initial_weights=dataset.initial_weights,
        one_step_ratio_mode="direct",
        occupancy=occupancy,
        action_ratio=action,
        source_state_ratio=source,
        transition_ratio=transition,
    )
    assert model.diagnostics["initial_ratio_mode"] == "joint"
    assert model.diagnostics["one_step_ratio_mode"] == "direct"
    assert model.diagnostics["source_state_ratio_enabled"] is False
    assert model.diagnostics["initial_joint_ratio_enabled"] is True
    assert model.diagnostics["one_step_direct_ratio_enabled"] is True
    assert model.history
    assert np.isclose(
        model.history[0]["target_damped_std"],
        model.history[0]["target_projected_std"],
    )
    assert np.all(np.isfinite(model.predict_state_action_ratio(dataset.states, dataset.actions)))

    factored_model = fit_discounted_occupancy_ratio(
        states=dataset.states,
        actions=dataset.actions,
        next_states=dataset.next_states,
        target_actions=dataset.target_actions,
        target_next_actions=dataset.next_target_actions,
        gamma=dataset.gamma,
        initial_states=dataset.initial_states,
        initial_actions=dataset.initial_actions,
        initial_weights=dataset.initial_weights,
        initial_ratio_mode="factored",
        one_step_ratio_mode="factored",
        occupancy=occupancy,
        action_ratio=action,
        source_state_ratio=source,
        transition_ratio=transition,
    )
    assert factored_model.diagnostics["initial_ratio_mode"] == "factored"
    assert factored_model.diagnostics["one_step_ratio_mode"] == "factored"
    assert factored_model.diagnostics["source_state_ratio_enabled"] is True
    assert factored_model.diagnostics["initial_joint_ratio_enabled"] is False
    assert factored_model.diagnostics["one_step_direct_ratio_enabled"] is False
    assert np.all(np.isfinite(factored_model.predict_state_action_ratio(dataset.states, dataset.actions)))

    with pytest.raises(ValueError, match="initial_ratio_mode='joint'"):
        fit_discounted_occupancy_ratio(
            states=dataset.states,
            actions=dataset.actions,
            next_states=dataset.next_states,
            target_actions=dataset.target_actions,
            gamma=dataset.gamma,
            initial_states=dataset.initial_states,
            initial_ratio_mode="joint",
            occupancy=occupancy,
            action_ratio=action,
            source_state_ratio=source,
            transition_ratio=transition,
        )
    with pytest.raises(ValueError, match="one_step_ratio_mode='direct'"):
        fit_discounted_occupancy_ratio(
            states=dataset.states,
            actions=dataset.actions,
            next_states=dataset.next_states,
            target_actions=dataset.target_actions,
            gamma=dataset.gamma,
            one_step_ratio_mode="direct",
            occupancy=occupancy,
            action_ratio=action,
            source_state_ratio=source,
            transition_ratio=transition,
        )


def test_neural_target_builder_source_state_ratio_matches_boosted_algebra() -> None:
    x_sa = np.zeros((2, 2), dtype=np.float32)
    x_s = np.zeros((2, 1), dtype=np.float32)
    action = _RatioPredictor.constant(2.0, prediction_max=None, prediction_power=1.0, normalize_predictions=False)
    transition = _RatioPredictor.constant(1.0, prediction_max=None, prediction_power=1.0, normalize_predictions=False)
    builder = _NeuralTargetBuilder(
        transition_predictor=transition,
        action_predictor=action,
        X_sa_kernel=x_sa,
        X_s_query=x_s,
        X_sa_query_iw=x_sa,
        gamma=0.0,
        mcmc_samples=1,
        seed=1,
        batch_query=2,
        normalize_transition_cache=False,
        transition_cache_norm_eps=1e-12,
        source_state_ratio_query=np.full(2, 3.0),
    )
    out = builder(w_beh=np.ones(2))
    assert np.allclose(out["w_query"], 2.0)
    assert np.allclose(out["y"], 6.0)
    assert out["diag"]["source_state_ratio_enabled"]

    default_builder = _NeuralTargetBuilder(
        transition_predictor=transition,
        action_predictor=action,
        X_sa_kernel=x_sa,
        X_s_query=x_s,
        X_sa_query_iw=x_sa,
        gamma=0.0,
        mcmc_samples=1,
        seed=1,
        batch_query=2,
        normalize_transition_cache=False,
        transition_cache_norm_eps=1e-12,
    )
    default_out = default_builder(w_beh=np.ones(2))
    assert np.allclose(default_out["y"], 2.0)
    assert not default_out["diag"]["source_state_ratio_enabled"]


def test_boosted_discounted_source_state_correction_diagnostics() -> None:
    dataset = make_discrete_dataset(setting="discrete_chain", gamma=0.5, sample_size=70, seed=4)
    occupancy = OccupancyRegressionConfig(
        num_iterations=1,
        mcmc_samples=2,
        batch_size=64,
        show_progress=False,
        lgb_params={"min_data_in_leaf": 10, "num_leaves": 7, "verbose": -1},
    )
    action = ActionRatioConfig(
        num_boost_round=2,
        early_stopping_rounds=0,
        refit_on_all_data=False,
        show_progress=False,
        lgb_params={"min_data_in_leaf": 10, "num_leaves": 7, "verbose": -1},
    )
    transition = TransitionRatioConfig(
        num_boost_round=2,
        permutation_samples=1,
        early_stopping_rounds=0,
        refit_on_all_data=False,
        show_progress=False,
        lgb_params={"min_data_in_leaf": 10, "num_leaves": 7, "verbose": -1},
    )
    source = SourceStateRatioConfig(
        num_boost_round=2,
        early_stopping_rounds=0,
        refit_on_all_data=False,
        show_progress=False,
        lgb_params={"min_data_in_leaf": 10, "num_leaves": 7, "verbose": -1},
    )
    no_source = fit_discounted_occupancy_ratio(
        states=dataset.states,
        actions=dataset.actions,
        next_states=dataset.next_states,
        target_actions=dataset.target_actions,
        gamma=dataset.gamma,
        occupancy=occupancy,
        action_ratio=action,
        transition_ratio=transition,
    )
    assert no_source.diagnostics["source_state_ratio_enabled"] is False
    assert no_source.diagnostics["source_state_ratio_mean"] == 1.0

    with_source = fit_discounted_occupancy_ratio(
        states=dataset.states,
        actions=dataset.actions,
        next_states=dataset.next_states,
        target_actions=dataset.target_actions,
        gamma=dataset.gamma,
        initial_states=dataset.states,
        initial_weights=np.ones(dataset.n, dtype=np.float64),
        occupancy=occupancy,
        action_ratio=action,
        source_state_ratio=source,
        transition_ratio=transition,
    )
    weights = with_source.predict_state_action_ratio(dataset.states, dataset.actions)
    assert with_source.diagnostics["source_state_ratio_enabled"] is True
    assert np.isclose(with_source.diagnostics["source_state_ratio_mean"], 1.0, atol=0.15)
    assert np.isfinite(with_source.diagnostics["source_state_ratio_ess_fraction"])
    assert np.all(np.isfinite(weights))


def test_neural_discounted_source_state_correction_diagnostics() -> None:
    dataset = make_discrete_dataset(setting="discrete_chain", gamma=0.5, sample_size=50, seed=5)
    occupancy = NeuralOccupancyRegressionConfig(
        num_iterations=1,
        gradient_steps_per_iteration=1,
        mcmc_samples=1,
        batch_size=32,
        hidden_dims=(8,),
        seed=50,
    )
    action = NeuralActionRatioConfig(max_steps=1, batch_size=32, hidden_dims=(8,), seed=51)
    transition = NeuralTransitionRatioConfig(
        max_steps=1,
        batch_size=32,
        permutation_samples=1,
        hidden_dims=(8,),
        seed=52,
    )
    no_source = fit_discounted_occupancy_ratio_neural(
        states=dataset.states,
        actions=dataset.actions,
        next_states=dataset.next_states,
        target_actions=dataset.target_actions,
        gamma=float(dataset.gamma),
        occupancy=occupancy,
        action_ratio=action,
        transition_ratio=transition,
    )
    assert no_source.diagnostics["source_state_ratio_enabled"] is False

    with_source = fit_discounted_occupancy_ratio_neural(
        states=dataset.states,
        actions=dataset.actions,
        next_states=dataset.next_states,
        target_actions=dataset.target_actions,
        gamma=float(dataset.gamma),
        initial_states=dataset.states,
        occupancy=occupancy,
        action_ratio=action,
        source_state_ratio=NeuralSourceStateRatioConfig(max_steps=1, batch_size=32, hidden_dims=(8,), seed=53),
        transition_ratio=transition,
    )
    weights = with_source.predict_state_action_ratio(dataset.states, dataset.actions)
    assert with_source.diagnostics["source_state_ratio_enabled"] is True
    assert np.isclose(with_source.diagnostics["source_state_ratio_mean"], 1.0, atol=1e-8)
    assert np.all(np.isfinite(weights))


def test_neural_joint_and_direct_modes_resolve() -> None:
    dataset = make_discrete_dataset(setting="discrete_chain", gamma=0.5, sample_size=32, seed=15)
    occupancy = NeuralOccupancyRegressionConfig(
        num_iterations=1,
        gradient_steps_per_iteration=1,
        mcmc_samples=1,
        batch_size=16,
        hidden_dims=(8,),
        seed=150,
        early_stopping=False,
    )
    action = NeuralActionRatioConfig(max_steps=1, batch_size=16, hidden_dims=(8,), seed=151)
    source = NeuralSourceStateRatioConfig(max_steps=1, batch_size=16, hidden_dims=(8,), seed=152)
    transition = NeuralTransitionRatioConfig(
        max_steps=1,
        batch_size=16,
        permutation_samples=1,
        hidden_dims=(8,),
        seed=153,
    )
    model = fit_discounted_occupancy_ratio_neural(
        states=dataset.states,
        actions=dataset.actions,
        next_states=dataset.next_states,
        target_actions=dataset.target_actions,
        target_next_actions=dataset.next_target_actions,
        gamma=float(dataset.gamma),
        initial_states=dataset.initial_states,
        initial_actions=dataset.initial_actions,
        initial_weights=dataset.initial_weights,
        one_step_ratio_mode="direct",
        occupancy=occupancy,
        action_ratio=action,
        source_state_ratio=source,
        transition_ratio=transition,
    )
    weights = model.predict_state_action_ratio(dataset.states, dataset.actions)
    assert model.diagnostics["initial_ratio_mode"] == "joint"
    assert model.diagnostics["one_step_ratio_mode"] == "direct"
    assert model.diagnostics["source_state_ratio_enabled"] is False
    assert model.diagnostics["initial_joint_ratio_enabled"] is True
    assert model.diagnostics["one_step_direct_ratio_enabled"] is True
    assert model.diagnostics["initial_joint_ratio_density_ratio_loss"] == "lsif"
    assert model.diagnostics["one_step_direct_ratio_density_ratio_loss"] == "lsif"
    assert model.diagnostics["initial_joint_ratio_updates"] > 0
    assert model.diagnostics["one_step_direct_ratio_updates"] > 0
    assert np.all(np.isfinite(weights))

    factored_model = fit_discounted_occupancy_ratio_neural(
        states=dataset.states,
        actions=dataset.actions,
        next_states=dataset.next_states,
        target_actions=dataset.target_actions,
        target_next_actions=dataset.next_target_actions,
        gamma=float(dataset.gamma),
        initial_states=dataset.initial_states,
        initial_actions=dataset.initial_actions,
        initial_weights=dataset.initial_weights,
        initial_ratio_mode="factored",
        one_step_ratio_mode="factored",
        occupancy=occupancy,
        action_ratio=action,
        source_state_ratio=source,
        transition_ratio=transition,
    )
    factored_weights = factored_model.predict_state_action_ratio(dataset.states, dataset.actions)
    assert factored_model.diagnostics["initial_ratio_mode"] == "factored"
    assert factored_model.diagnostics["one_step_ratio_mode"] == "factored"
    assert factored_model.diagnostics["source_state_ratio_enabled"] is True
    assert factored_model.diagnostics["initial_joint_ratio_enabled"] is False
    assert factored_model.diagnostics["one_step_direct_ratio_enabled"] is False
    assert factored_model.diagnostics["source_state_ratio_updates"] > 0
    assert factored_model.diagnostics["initial_joint_ratio_updates"] == 0.0
    assert factored_model.diagnostics["one_step_direct_ratio_updates"] == 0.0
    assert np.all(np.isfinite(factored_weights))


def test_gamma_zero_exact_tabular_ratio_equals_action_ratio() -> None:
    mdp = make_chain_mdp()
    table = exact_ratio_table(mdp, gamma=0.0)
    action_ratio = mdp.target_policy / mdp.behavior_policy
    assert np.allclose(table, action_ratio)


def test_two_state_tabular_discounted_ratio_linear_solve() -> None:
    mdp = make_chain_mdp(n_states=2, policy_shift=0.4)
    table = exact_ratio_table(mdp, gamma=0.7)
    reference_joint = mdp.reference_state_dist[:, None] * mdp.behavior_policy
    assert np.isclose(np.sum(reference_joint * table), 1.0)
    assert np.all(table > 0.0)


def test_prediction_offsets_applied_once() -> None:
    model = DiscountedOccupancyRatioModel(
        occupancy_booster=_ConstantBooster(0.5),
        action_ratio_booster=_ConstantBooster(0.25),
        transition_ratio_booster=_ConstantBooster(0.75),
        occupancy_initial_ratio=1.0,
        action_ratio_offset=1.0,
        transition_ratio_offset=1.0,
        gamma=0.9,
        state_dim=2,
        action_dim=1,
        history=[],
        diagnostics={},
        legacy_result={"bst_w": None, "bst_iw": None, "bst_k": None},
    )
    states = np.zeros((3, 2))
    actions = np.zeros((3, 1))
    assert np.allclose(model.predict_state_action_ratio(states, actions, clip=False), 1.5)
    assert np.allclose(model.predict_action_ratio(states, actions, clip=False), 1.25)


def test_boosted_nuisance_objective_gradients_and_hessians_match_losses() -> None:
    class _Dataset:
        def __init__(self, labels: np.ndarray, weights: np.ndarray) -> None:
            self._labels = labels
            self._weights = weights

        def get_label(self) -> np.ndarray:
            return self._labels

        def get_weight(self) -> np.ndarray:
            return self._weights

    def finite_grad(loss_fn, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        out = np.zeros_like(x, dtype=np.float64)
        for idx in range(x.size):
            xp = x.copy()
            xm = x.copy()
            xp[idx] += eps
            xm[idx] -= eps
            out[idx] = (loss_fn(xp) - loss_fn(xm)) / (2.0 * eps)
        return out

    labels = np.array([1, 1, 0, 0], dtype=np.int32)
    weights = np.array([1.0, 0.5, 2.0, 1.5], dtype=np.float64)
    preds = np.array([0.7, 1.3, 0.2, 1.1], dtype=np.float64)
    dataset = _Dataset(labels, weights)
    grad, hess = importance_ratio_objective(preds, dataset, eps=0.25)

    def importance_loss(x: np.ndarray) -> float:
        return float(
            np.sum(weights[labels == 1] * x[labels == 1] ** 2)
            - 2.0 * np.sum(weights[labels == 0] * x[labels == 0])
        )

    assert np.allclose(grad, finite_grad(importance_loss, preds), atol=1e-6)
    assert np.allclose(hess[labels == 1], 2.0 * weights[labels == 1])
    assert np.allclose(hess[labels == 0], 2.0 * 0.25 * weights[labels == 0])

    t_labels = np.array([1, 1, 0, 0, 0], dtype=np.int32)
    t_weights = np.array([1.0, 0.75, 0.5, 0.5, 0.75], dtype=np.float64)
    t_preds = np.array([0.4, 1.2, 0.5, 1.0, 1.6], dtype=np.float64)
    t_dataset = _Dataset(t_labels, t_weights)
    t_grad, t_hess = transition_ratio_objective(t_preds, t_dataset, eps=0.1, lam_norm=1.0)

    def transition_loss(x: np.ndarray) -> float:
        is_beh = t_labels == 1
        is_ref = ~is_beh
        z_ref = float(np.sum(t_weights[is_ref]))
        mean_ref = float(np.sum(t_weights[is_ref] * x[is_ref]) / z_ref)
        return float(
            np.sum(t_weights[is_ref] * x[is_ref] ** 2)
            - 2.0 * np.sum(t_weights[is_beh] * x[is_beh])
            + z_ref * (mean_ref - 1.0) ** 2
        )

    assert np.allclose(t_grad, finite_grad(transition_loss, t_preds), atol=1e-6)
    is_ref = t_labels == 0
    z_ref = float(np.sum(t_weights[is_ref]))
    assert np.allclose(t_hess[is_ref], 2.0 * t_weights[is_ref] + 2.0 * (t_weights[is_ref] ** 2) / z_ref)
    assert np.allclose(t_hess[~is_ref], 0.1 * t_weights[~is_ref])


def test_neural_nuisance_losses_have_expected_gradients() -> None:
    torch = pytest.importorskip("torch")

    w_ref = torch.tensor([0.5, 1.5, 2.0], dtype=torch.float64, requires_grad=True)
    w_num = torch.tensor([0.25, 1.25], dtype=torch.float64, requires_grad=True)
    num_weight = torch.tensor([1.0, 3.0], dtype=torch.float64)
    loss = _weighted_source_lsif_loss(w_ref, w_num, num_weight, normalization_penalty=2.0)
    loss.backward()
    ref_mean = float(torch.mean(w_ref.detach()).item())
    expected_ref_grad = 2.0 * w_ref.detach().numpy() / 3.0 + 2.0 * 2.0 * (ref_mean - 1.0) / 3.0
    expected_num_grad = -2.0 * num_weight.numpy() / float(torch.sum(num_weight).item())
    assert np.allclose(w_ref.grad.detach().numpy(), expected_ref_grad)
    assert np.allclose(w_num.grad.detach().numpy(), expected_num_grad)

    logits_den = torch.tensor([-0.5, 0.75], dtype=torch.float64, requires_grad=True)
    logits_num = torch.tensor([0.2, -1.0], dtype=torch.float64, requires_grad=True)
    class_weight = torch.tensor([1.0, 2.0], dtype=torch.float64)
    logistic = _weighted_binary_ratio_loss(logits_den, logits_num, class_weight)
    logistic.backward()
    expected_den_grad = 0.5 * torch.sigmoid(logits_den.detach()).numpy() / logits_den.numel()
    expected_num_grad = -0.5 * (
        class_weight.numpy() * torch.sigmoid(-logits_num.detach()).numpy() / float(torch.sum(class_weight).item())
    )
    assert np.allclose(logits_den.grad.detach().numpy(), expected_den_grad)
    assert np.allclose(logits_num.grad.detach().numpy(), expected_num_grad)


def test_occupancy_loss_values_match_lightgbm_objectives() -> None:
    class _Labels:
        def __init__(self, labels: np.ndarray) -> None:
            self._labels = labels

        def get_label(self) -> np.ndarray:
            return self._labels

    labels = np.array([0.0, 1.0, -1.0])
    preds = np.array([1.0, 0.0, -0.5])
    resid = preds - labels
    squared_obj = _make_occupancy_objective(loss="squared", huber_delta=None, huber_hessian_floor=0.0)
    grad, hess = squared_obj(preds, _Labels(labels))
    assert np.allclose(grad, resid)
    assert np.allclose(hess, np.ones_like(resid))
    assert _occupancy_loss_value(preds, labels, loss="squared", huber_delta=None) == pytest.approx(
        0.5 * float(np.mean(resid**2))
    )

    huber_obj = _make_occupancy_objective(loss="huber", huber_delta=0.75, huber_hessian_floor=0.1)
    grad, hess = huber_obj(preds, _Labels(labels))
    assert np.allclose(grad, np.clip(resid, -0.75, 0.75))
    assert np.all(hess >= 0.1)
    assert _occupancy_loss_value(preds, labels, loss="huber", huber_delta=0.75) > 0.0


def test_invalid_occupancy_config_raises() -> None:
    assert OccupancyRegressionConfig().huber_delta_scale == pytest.approx(1.345)
    assert SourceStateRatioConfig().density_ratio_loss == "lsif"
    assert NeuralSourceStateRatioConfig().density_ratio_loss == "lsif"
    assert NeuralSourceStateRatioConfig().prediction_max == pytest.approx(50.0)
    assert NeuralOccupancyRegressionConfig().direct_one_step_density_ratio_loss == "lsif"

    with pytest.raises(ValueError, match="loss"):
        OccupancyRegressionConfig(loss="not_a_loss")
    with pytest.raises(ValueError, match="fixed_point_damping"):
        OccupancyRegressionConfig(fixed_point_damping=0.0)
    with pytest.raises(ValueError, match="pseudo_outcome_upper_quantile"):
        OccupancyRegressionConfig(pseudo_outcome_upper_quantile=1.0)
    with pytest.raises(ValueError, match="occupancy_sample_weight_mode"):
        OccupancyRegressionConfig(occupancy_sample_weight_mode="bad")
    with pytest.raises(ValueError, match="prediction_power"):
        ActionRatioConfig(prediction_power=1.5)
    with pytest.raises(ValueError, match="density_ratio_loss"):
        ActionRatioConfig(density_ratio_loss="bad")
    with pytest.raises(ValueError, match="logistic_logit_clip"):
        ActionRatioConfig(density_ratio_loss="logistic", logistic_logit_clip=0.0)
    with pytest.raises(ValueError, match="prediction_max"):
        TransitionRatioConfig(prediction_max=0.0)
    with pytest.raises(ValueError, match="density_ratio_loss"):
        TransitionRatioConfig(density_ratio_loss="bad")
    with pytest.raises(ValueError, match="logistic_logit_clip"):
        TransitionRatioConfig(density_ratio_loss="logistic", logistic_logit_clip=-1.0)
    with pytest.raises(ValueError, match="crossfit_folds"):
        ActionRatioConfig(crossfit_folds=0)
    with pytest.raises(ValueError, match="moment_calibration"):
        TransitionRatioConfig(moment_calibration="bad")
    with pytest.raises(ValueError, match="hidden_dims"):
        NeuralOccupancyRegressionConfig(hidden_dims=())
    with pytest.raises(ValueError, match="validation_warmup_iterations"):
        NeuralOccupancyRegressionConfig(validation_warmup_iterations=-1)
    with pytest.raises(ValueError, match="learning_rate"):
        NeuralActionRatioConfig(learning_rate=0.0)
    with pytest.raises(ValueError, match="permutation_samples"):
        NeuralTransitionRatioConfig(permutation_samples=0)
    with pytest.raises(ValueError, match="crossfit_folds"):
        NeuralActionRatioConfig(crossfit_folds=0)
    with pytest.raises(ValueError, match="moment_calibration"):
        NeuralTransitionRatioConfig(moment_calibration="bad")
    with pytest.raises(ValueError, match="density_ratio_loss"):
        NeuralActionRatioConfig(density_ratio_loss="bad")
    with pytest.raises(ValueError, match="logistic_logit_clip"):
        NeuralTransitionRatioConfig(density_ratio_loss="logistic", logistic_logit_clip=0.0)
    with pytest.raises(ValueError, match="neural_density_ratio_loss"):
        OccupancyRatioBenchmarkConfig(neural_density_ratio_loss="bad")
    with pytest.raises(ValueError, match="neural_logistic_logit_clip"):
        OccupancyRatioBenchmarkConfig(neural_logistic_logit_clip=0.0)
    with pytest.raises(ValueError, match="neural_validation_warmup_iterations"):
        OccupancyRatioBenchmarkConfig(neural_validation_warmup_iterations=-1)
    with pytest.raises(ValueError, match="neural_source_steps"):
        OccupancyRatioBenchmarkConfig(neural_source_steps=0)
    with pytest.raises(ValueError, match="neural_direct_one_step_steps"):
        OccupancyRatioBenchmarkConfig(neural_direct_one_step_steps=0)
    with pytest.raises(ValueError, match="neural_direct_adjoint_steps"):
        OccupancyRatioBenchmarkConfig(neural_direct_adjoint_steps=0)
    with pytest.raises(ValueError, match="neural_direct_adjoint_learning_rate"):
        OccupancyRatioBenchmarkConfig(neural_direct_adjoint_learning_rate=0.0)
    with pytest.raises(ValueError, match="neural_direct_adjoint_weight_decay"):
        OccupancyRatioBenchmarkConfig(neural_direct_adjoint_weight_decay=-1.0)
    with pytest.raises(ValueError, match="neural_action_hidden_dims"):
        OccupancyRatioBenchmarkConfig(neural_action_hidden_dims=(0,))
    with pytest.raises(ValueError, match="direct_one_step_hidden_dims"):
        NeuralOccupancyRegressionConfig(direct_one_step_hidden_dims=(0,))
    custom_nuisance_steps = OccupancyRatioBenchmarkConfig(
        neural_action_steps=11,
        neural_source_steps=13,
        neural_transition_steps=17,
        neural_direct_one_step_steps=19,
        neural_action_hidden_dims=(17, 17),
        neural_source_hidden_dims=(19, 19),
        neural_transition_hidden_dims=(23, 23),
        neural_direct_one_step_hidden_dims=(29, 29),
        neural_direct_adjoint_steps=23,
        neural_direct_adjoint_learning_rate=2e-3,
        neural_direct_adjoint_weight_decay=1e-6,
    )
    assert custom_nuisance_steps.neural_source_steps == 13
    assert custom_nuisance_steps.neural_direct_one_step_steps == 19
    assert custom_nuisance_steps.neural_direct_adjoint_steps == 23
    from occupancy_ratio_benchmark.estimators import _neural_direct_one_step_steps, _neural_source_steps

    assert _neural_source_steps(custom_nuisance_steps) == 13
    assert _neural_direct_one_step_steps(custom_nuisance_steps) == 19
    from occupancy_ratio_benchmark.estimators import _neural_stage_hidden_dims

    assert _neural_stage_hidden_dims(custom_nuisance_steps.neural_source_hidden_dims, fallback=(64, 64)) == (19, 19)
    assert _neural_stage_hidden_dims(None, fallback=(64, 64)) == (64, 64)
    legacy_nuisance_steps = OccupancyRatioBenchmarkConfig(neural_action_steps=11, neural_transition_steps=17)
    assert _neural_source_steps(legacy_nuisance_steps) == 11
    assert _neural_direct_one_step_steps(legacy_nuisance_steps) == 17
    with pytest.raises(ValueError, match="profile"):
        OccupancyRatioBenchmarkConfig(profile="not-a-profile")
    with pytest.raises(ValueError, match="estimators"):
        OccupancyRatioBenchmarkConfig(estimators=("not_an_estimator",))
    with pytest.raises(ValueError, match="cv_scoring"):
        OccupancyRatioBenchmarkConfig(cv_scoring="bad")
    with pytest.raises(ValueError, match="google_learning_rates"):
        OccupancyRatioBenchmarkConfig(google_learning_rates=(0.0,))
    with pytest.raises(ValueError, match="discrete_policy_shifts"):
        OccupancyRatioBenchmarkConfig(discrete_policy_shifts=(-0.1,))
    with pytest.raises(ValueError, match="boosted_density_ratio_loss"):
        OccupancyRatioBenchmarkConfig(boosted_density_ratio_loss="bad")
    with pytest.raises(ValueError, match="boosted_logistic_logit_clip"):
        OccupancyRatioBenchmarkConfig(boosted_logistic_logit_clip=0.0)
    with pytest.raises(ValueError, match="openml_task_ids"):
        OccupancyRatioBenchmarkConfig(openml_task_ids=(0,))
    with pytest.raises(ValueError, match="tabular_state_cap"):
        OccupancyRatioBenchmarkConfig(tabular_state_cap=1)
    with pytest.raises(ValueError, match="source_state_correction_mode"):
        OccupancyRatioBenchmarkConfig(source_state_correction_mode="bad")
    with pytest.raises(ValueError, match="density_ratio_loss"):
        SourceStateRatioConfig(density_ratio_loss="bad")
    with pytest.raises(ValueError, match="density_ratio_loss"):
        NeuralSourceStateRatioConfig(density_ratio_loss="bad")
    assert OccupancyRatioBenchmarkConfig(automl_tuning="fast").automl_tuning == "fast"
    with pytest.raises(ValueError, match="automl_tuning"):
        OccupancyRatioBenchmarkConfig(automl_tuning="wide")
    factored_benchmark = OccupancyRatioBenchmarkConfig(
        boosted_stabilization_presets=("stable_factored",),
        neural_stabilization_presets=("stable_factored",),
        include_google_dual_dice=False,
    )
    assert factored_benchmark.boosted_stabilization_presets == ("stable_factored",)
    assert factored_benchmark.neural_stabilization_presets == ("stable_factored",)


def test_benchmark_cli_accepts_discrete_policy_shifts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["occupancy-ratio-benchmark", "--discrete-policy-shifts", "0.0", "0.35", "1.5"],
    )
    args = parse_benchmark_args()
    assert args.discrete_policy_shifts == [0.0, 0.35, 1.5]


def test_benchmark_profile_and_estimator_expansion() -> None:
    config = OccupancyRatioBenchmarkConfig.for_profile(
        "medium",
        include_google_dual_dice=False,
    )
    assert config.profile == "medium"
    assert config.stage == "medium"
    assert config.gammas == (0.5, 0.9, 0.95, 0.99)
    assert config.discrete_policy_shifts == ()
    assert config.linear_gaussian_policy_shifts == (0.25, 0.5, 1.0, 2.0)
    assert "openml_contextual_bandit" in config.settings
    assert "openml_finite_mdp" in config.settings
    assert not any("dualdice" in estimator for estimator in config.resolved_estimators())

    expanded_default = _expanded_estimators(
        OccupancyRatioBenchmarkConfig(
            estimators=("boosted_tree", "neural_network"),
            include_google_dual_dice=False,
        )
    )
    assert expanded_default == ("boosted_tree_stable", "neural_network_stable")

    expanded = _expanded_estimators(
        OccupancyRatioBenchmarkConfig(
            estimators=("oracle", "boosted_tree", "neural_network", "google_dualdice"),
            boosted_estimator_presets=("squared", "stable", "transition_norm"),
            neural_estimator_presets=("stable", "calibrated"),
            neural_stabilization_presets=(),
        )
    )
    assert expanded == (
        "oracle",
        "boosted_tree_squared",
        "boosted_tree_stable",
        "boosted_tree_transition_norm",
        "neural_network_stable",
        "neural_network_calibrated",
        "google_dualdice_neural",
    )
    expanded_auto = _expanded_estimators(
        OccupancyRatioBenchmarkConfig(
            estimators=("neural_network",),
            neural_estimator_presets=("auto", "stable_logistic_nuisance"),
            neural_stabilization_presets=(),
        )
    )
    assert expanded_auto == ("neural_network_auto", "neural_network_stable_logistic_nuisance")
    expanded_factored = _expanded_estimators(
        OccupancyRatioBenchmarkConfig(
            estimators=("boosted_tree", "neural_network"),
            boosted_estimator_presets=("stable_factored",),
            neural_estimator_presets=("stable_factored",),
            include_google_dual_dice=False,
        )
    )
    assert expanded_factored == ("boosted_tree_stable_factored", "neural_network_stable_factored")

    overnight = OccupancyRatioBenchmarkConfig.for_profile("overnight", include_google_dual_dice=False)
    assert "gym_halfcheetah" in overnight.settings
    assert "gym_hopper" in overnight.settings
    assert "minari_pointmaze" in overnight.settings
    assert "minari_minigrid" in overnight.settings
    assert "boosted_tree_stable_logistic_nuisance" in overnight.estimators
    assert "neural_network_google_parity" not in overnight.estimators
    assert overnight.sample_sizes == (1_000, 2_000)
    assert overnight.estimator_timeout_sec is None
    assert not any("dualdice" in estimator for estimator in overnight.resolved_estimators())

    high_stakes = OccupancyRatioBenchmarkConfig.for_profile("high_stakes", include_google_dual_dice=False)
    assert high_stakes.profile == "high_stakes"
    assert high_stakes.estimator_timeout_sec == 900.0
    assert high_stakes.source_state_correction_mode == "auto"
    assert high_stakes.boosted_density_ratio_loss == "lsif"
    assert high_stakes.neural_density_ratio_loss == "lsif"
    assert high_stakes.neural_fixed_point_damping == 0.5
    assert {"boosted_tree_stable", "neural_network_stable"} <= set(high_stakes.resolved_estimators())


def test_discrete_policy_shift_sweeps_are_expanded_and_recorded() -> None:
    config = OccupancyRatioBenchmarkConfig(discrete_policy_shifts=(0.0, 0.35, 1.5))
    assert _policy_shifts_for_setting(config, "discrete_chain") == (0.0, 0.35, 1.5)
    assert _policy_shifts_for_setting(OccupancyRatioBenchmarkConfig(), "discrete_grid") == (None,)

    dataset = make_discrete_dataset(
        setting="discrete_grid",
        gamma=0.9,
        sample_size=40,
        seed=7,
        policy_shift=1.5,
    )
    assert dataset.metadata["policy_shift"] == 1.5
    assert dataset.true_ratio is not None
    assert float(np.std(dataset.true_ratio)) > 0.0


def test_conservatism_audit_flags_uniform_collapse_and_google_delta(tmp_path) -> None:
    rows = [
        {
            "setting": "discrete_chain",
            "estimator": "boosted_tree_stable",
            "status": "ok",
            "gamma": 0.9,
            "seed": 0,
            "sample_size": 200,
            "effective_sample_size_fraction": 0.99,
            "true_effective_sample_size_fraction": 0.62,
            "weight_cv_ratio_to_truth": 0.25,
            "ratio_corr": 0.4,
            "log_ratio_rmse": 0.8,
            "ope_value_abs_error": 0.2,
        },
        {
            "setting": "gym_pendulum",
            "estimator": "google_dualdice_neural",
            "status": "ok",
            "gamma": 0.9,
            "seed": 0,
            "sample_size": 200,
            "ope_value_abs_error": 1.0,
        },
        {
            "setting": "gym_pendulum",
            "estimator": "neural_network_stable",
            "status": "ok",
            "gamma": 0.9,
            "seed": 0,
            "sample_size": 200,
            "ope_value_abs_error": 1.2,
            "clipping_fraction": 0.03,
        },
    ]
    audit_rows = build_conservatism_audit_rows(rows)
    chain = next(row for row in audit_rows if row["setting"] == "discrete_chain")
    assert chain["audit_status"] == "fail"
    assert "near-uniform ESS" in chain["audit_reason"]
    assert "below half oracle spread" in chain["audit_reason"]
    gym = next(row for row in audit_rows if row["estimator"] == "neural_network_stable")
    assert gym["audit_status"] == "warn"
    assert gym["google_ope_value_abs_error"] == pytest.approx(1.0)
    assert gym["ope_abs_error_delta_vs_google"] == pytest.approx(0.2)

    paths = write_conservatism_audit(rows, tmp_path)
    assert paths["audit"].exists()
    assert paths["report"].exists()
    report = render_conservatism_report(audit_rows)
    assert "Default Decisions" in report
    assert "Conservatism Failures" in report


def test_conservatism_audit_json_configs_load() -> None:
    config_dir = Path(__file__).resolve().parents[1] / "configs"
    smoke = load_config_file(config_dir / "conservatism_audit_smoke.json")
    full = load_config_file(config_dir / "conservatism_audit_full.json")
    required = {
        "discrete_chain",
        "discrete_grid",
        "linear_gaussian",
        "gym_pendulum",
        "gym_mountain_car_continuous",
    }
    assert required.issubset(set(smoke.settings))
    assert required.issubset(set(full.settings))
    assert {"gym_hopper", "gym_halfcheetah"}.issubset(set(full.settings))
    assert "boosted_tree_stable_factored" in smoke.estimators
    assert "neural_network_stable_factored" in smoke.estimators
    assert "google_dualdice_neural" in smoke.estimators
    assert smoke.cv_fixed_point_dampings == (1.0, 0.75, 0.5)
    assert smoke.cv_occupancy_ratio_max_values == (None, 100.0, 50.0)


def test_gym_control_dataset_has_optional_ratio_truth_and_target_value() -> None:
    pytest.importorskip("gymnasium")
    dataset = make_gym_control_dataset(
        setting="gym_pendulum",
        gamma=0.9,
        sample_size=24,
        seed=42,
        target_value_rollouts=2,
    )
    assert dataset.true_ratio is None
    assert dataset.states.shape == dataset.next_states.shape
    assert dataset.actions.shape == dataset.target_actions.shape
    assert dataset.rewards.shape == (24,)
    assert np.all(np.isfinite(dataset.rewards))
    assert np.all((dataset.masks == 0.0) | (dataset.masks == 1.0))
    assert np.isfinite(float(dataset.metadata["target_policy_value"]))
    oracle = estimate_oracle(dataset)
    assert oracle.status == "skipped"
    assert oracle.diagnostics["ratio_truth_available"] == 0.0


def test_openml_contextual_bandit_fixture_has_exact_ratio_truth() -> None:
    features = np.array(
        [
            [0.0, 1.0],
            [1.0, 0.0],
            [0.5, 0.5],
            [-1.0, 2.0],
            [2.0, -1.0],
            [1.5, 1.0],
        ],
        dtype=np.float64,
    )
    labels = np.array([0, 1, 0, 1, 2, 2])
    dataset = make_openml_contextual_bandit_from_arrays(
        features=features,
        labels=labels,
        gamma=0.9,
        sample_size=80,
        seed=7,
        task_id="fixture",
        dataset_name="fixture",
    )
    assert dataset.setting == "openml_contextual_bandit"
    assert dataset.states.shape[0] == 80
    assert dataset.actions.shape[0] == 80
    assert dataset.true_ratio is not None
    assert dataset.true_action_ratio is not None
    assert np.all(np.isfinite(dataset.true_ratio))
    assert np.all(dataset.true_ratio > 0.0)
    assert np.allclose(dataset.true_ratio, dataset.true_action_ratio)
    assert np.allclose(dataset.true_transition_ratio, 1.0)
    assert estimate_oracle(dataset).status == "ok"


def test_openml_finite_mdp_fixture_solved_ratio_truth() -> None:
    rng = np.random.default_rng(11)
    features = rng.normal(size=(24, 3))
    labels = np.arange(24) % 3
    dataset = make_openml_finite_mdp_from_arrays(
        features=features,
        labels=labels,
        gamma=0.5,
        sample_size=100,
        seed=3,
        task_id="fixture",
        dataset_name="fixture",
        state_cap=12,
    )
    assert dataset.setting == "openml_finite_mdp"
    assert dataset.true_ratio is not None
    assert dataset.metadata["n_states"] == 12
    assert np.all(np.isfinite(dataset.true_ratio))
    assert np.all(dataset.true_ratio >= 0.0)

    transition = np.zeros((2, 2, 2), dtype=np.float64)
    transition[:, :, :] = 0.5
    target_policy = np.ones((2, 2), dtype=np.float64) / 2.0
    reference = np.ones(2, dtype=np.float64) / 2.0
    solved = solve_discounted_occupancy(transition, target_policy, reference, gamma=0.7)
    assert np.allclose(solved, reference)


def test_tabular_dataset_unavailable_becomes_skipped_rows(tmp_path, monkeypatch) -> None:
    def _missing(**kwargs):
        raise OptionalDatasetUnavailable("fixture missing optional data")

    monkeypatch.setattr(runner_module, "make_openml_contextual_bandit_dataset", _missing)
    config = OccupancyRatioBenchmarkConfig(
        stage="smoke",
        output_root=tmp_path,
        seeds=(0,),
        sample_sizes=(20,),
        gammas=(0.5,),
        settings=("openml_contextual_bandit",),
        estimators=("oracle", "boosted_tree_stable"),
        openml_task_ids=(31,),
        include_google_dual_dice=False,
        write_plots=False,
    )
    result = run_benchmark(config)
    assert {row["estimator"] for row in result.rows} == {"oracle", "boosted_tree_stable"}
    assert all(row["status"] == "skipped" for row in result.rows)
    assert all(row["dataset_variant"] == "31" for row in result.rows)
    assert all("fixture missing optional data" in row["skip_reason"] for row in result.rows)


def test_runner_manual_tabular_fixture_writes_boosted_rows(tmp_path, monkeypatch) -> None:
    features = np.column_stack([np.linspace(-1.0, 1.0, 80), np.cos(np.linspace(0.0, 3.0, 80))])
    labels = np.arange(80) % 2

    def _fixture(**kwargs):
        return make_openml_contextual_bandit_from_arrays(
            features=features,
            labels=labels,
            gamma=float(kwargs["gamma"]),
            sample_size=int(kwargs["sample_size"]),
            seed=int(kwargs["seed"]),
            task_id=int(kwargs["task_id"]),
            dataset_name="fixture",
        )

    monkeypatch.setattr(runner_module, "make_openml_contextual_bandit_dataset", _fixture)
    config = OccupancyRatioBenchmarkConfig(
        stage="smoke",
        output_root=tmp_path,
        seeds=(0,),
        sample_sizes=(60,),
        gammas=(0.5,),
        settings=("openml_contextual_bandit",),
        estimators=("oracle", "boosted_tree_stable"),
        openml_task_ids=(31,),
        include_google_dual_dice=False,
        boosted_num_iterations=1,
        boosted_mcmc_samples=1,
        boosted_batch_size=32,
        write_plots=False,
    )
    result = run_benchmark(config)
    boosted_rows = [row for row in result.rows if row["estimator"] == "boosted_tree_stable"]
    assert boosted_rows
    assert boosted_rows[0]["status"] == "ok"
    assert boosted_rows[0]["setting"] == "openml_contextual_bandit"
    assert boosted_rows[0]["dataset_variant"] == "31"
    assert boosted_rows[0]["ratio_truth_available"] == 1.0
    assert result.conservatism_audit_path.exists()
    assert result.conservatism_report_path.exists()
    assert "conservatism_audit.md" in result.benchmark_readout_path.read_text(encoding="utf-8")


def test_public_package_exports_and_console_modules() -> None:
    import importlib
    import inspect
    import occupancy_ratio

    assert occupancy_ratio.NeuralDiscountedOccupancyRatioModel.__name__ == "NeuralDiscountedOccupancyRatioModel"
    assert occupancy_ratio.DiscountedOccupancyRatioNeuralModel is occupancy_ratio.NeuralDiscountedOccupancyRatioModel
    assert importlib.import_module("occupancy_ratio.boosted").fit_discounted_occupancy_ratio
    assert importlib.import_module("occupancy_ratio.neural").fit_discounted_occupancy_ratio_neural
    assert importlib.import_module("occupancy_ratio.nuisance").fit_importance_ratio_lgbm
    assert importlib.import_module("occupancy_ratio.configs").ActionRatioConfig
    assert importlib.import_module("occupancy_ratio.models").DiscountedOccupancyRatioModel
    assert importlib.import_module("occupancy_ratio.validation")._resolve_initial_ratio_mode
    assert importlib.import_module("occupancy_ratio.nuisance_lgbm").fit_state_density_ratio_lgbm
    assert importlib.import_module("occupancy_ratio.targets").make_forward_occupancy_dataset
    assert importlib.import_module("occupancy_ratio.stabilization")._project_nonnegative_normalized
    assert importlib.import_module("occupancy_ratio.neural_configs").NeuralActionRatioConfig
    assert importlib.import_module("occupancy_ratio.neural_models").NeuralDiscountedOccupancyRatioModel
    assert importlib.import_module("occupancy_ratio.neural_nuisance").fit_action_ratio_neural
    assert importlib.import_module("occupancy_ratio.neural_targets")._NeuralTargetBuilder
    assert importlib.import_module("occupancy_ratio.neural_fit").fit_discounted_occupancy_ratio_neural
    assert importlib.import_module("occupancy_ratio._tuning_candidates")._make_candidates
    assert importlib.import_module("occupancy_ratio._tuning_cv")._make_folds
    assert importlib.import_module("occupancy_ratio._tuning_refit")._select_refit_candidate
    assert importlib.import_module("occupancy_ratio._tuning_scoring")._score_candidates
    assert importlib.import_module("occupancy_ratio.fit_occupancy_ratio")._resolve_initial_ratio_mode
    assert importlib.import_module("occupancy_ratio.fit_occupancy_ratio_neural")._NeuralTargetBuilder
    assert inspect.signature(occupancy_ratio.tune_occupancy_ratio_auto).parameters["families"].default == ("neural",)
    assert inspect.signature(occupancy_ratio.OccupancyTuningConfig).parameters["families"].default == ("neural",)
    assert occupancy_ratio.ActionRatioConfig.__module__ == "occupancy_ratio.fit_occupancy_ratio"
    assert occupancy_ratio.NeuralActionRatioConfig.__module__ == "occupancy_ratio.fit_occupancy_ratio_neural"
    assert occupancy_ratio.OccupancyTuningConfig.__module__ == "occupancy_ratio.tuning"
    assert importlib.import_module("occupancy_ratio_benchmark.run").main
    assert importlib.import_module("occupancy_ratio_benchmark.dualdice_grid").main


def test_dualdice_json_configs_load() -> None:
    config_dir = Path(__file__).resolve().parents[1] / "configs"
    smoke = load_config_file(config_dir / "dualdice_smoke.json")
    core = load_config_file(config_dir / "dualdice_core.json")
    high_stakes = load_config_file(config_dir / "dualdice_high_stakes.json")
    assert smoke.config_path is not None
    assert smoke.config_sha256
    assert smoke.profile == "smoke"
    assert "neural_network_google_parity" in smoke.estimators
    assert core.profile == "medium"
    assert core.sample_sizes == (1000, 5000)
    assert core.gammas == (0.9, 0.95, 0.99)
    assert "google_dualdice_neural" in core.estimators
    assert high_stakes.profile == "high_stakes"
    assert high_stakes.source_state_correction_mode == "auto"


def test_google_dualdice_preflight_skip_is_json_friendly(tmp_path) -> None:
    preflight = preflight_google_dualdice(tmp_path / "missing-google-research")
    assert not preflight.available
    config = OccupancyRatioBenchmarkConfig(
        stage="smoke",
        output_root=tmp_path,
        estimators=("oracle", "google_dualdice_neural"),
        include_google_dual_dice=True,
        external_repo_path=tmp_path / "missing-google-research",
        seeds=(0,),
        sample_sizes=(50,),
        gammas=(0.5,),
        settings=("discrete_chain",),
        write_plots=False,
    )
    result = run_benchmark(config)
    skipped = [row for row in result.rows if row["estimator"] == "google_dualdice_neural"]
    assert skipped
    assert skipped[0]["status"] == "skipped"
    assert skipped[0]["skip_reason"]
    for row in result.rows:
        for value in row.values():
            assert isinstance(value, (str, int, float, bool, type(None), np.integer, np.floating))


def test_winner_table_selects_primary_metric() -> None:
    rows = [
        {
            "profile": "smoke",
            "stage": "smoke",
            "setting": "linear_gaussian",
            "estimator": "boosted_tree_stable",
            "status": "ok",
            "gamma": 0.9,
            "sample_size": 100,
            "policy_shift": 1.0,
            "ope_value_abs_error": 0.4,
            "runtime_sec": 1.0,
        },
        {
            "profile": "smoke",
            "stage": "smoke",
            "setting": "linear_gaussian",
            "estimator": "neural_network_stable",
            "status": "ok",
            "gamma": 0.9,
            "sample_size": 100,
            "policy_shift": 1.0,
            "ope_value_abs_error": 0.2,
            "runtime_sec": 2.0,
        },
    ]
    winners = make_winner_table(rows)
    assert winners[0]["winning_estimator"] == "neural_network_stable"
    assert winners[0]["primary_metric"] == "ope_value_abs_error"


def test_high_stakes_recommendation_requires_passing_estimators() -> None:
    base = {
        "profile": "high_stakes",
        "stage": "high_stakes",
        "setting": "gym_pendulum",
        "gamma": 0.9,
        "sample_size": 100,
        "seed": 0,
        "status": "ok",
    }
    rows = [
        {
            **base,
            "estimator": "boosted_tree_stable",
            "diagnostic_status": "pass",
            "ope_value_estimate": 1.0,
        },
        {
            **base,
            "estimator": "neural_network_stable",
            "diagnostic_status": "pass",
            "ope_value_estimate": 3.0,
        },
        {
            **base,
            "estimator": "google_dualdice_neural",
            "diagnostic_status": "fail",
            "ope_value_estimate": 100.0,
        },
    ]
    recommendations = make_high_stakes_recommendations(rows)
    assert recommendations[0]["decision_status"] == "pass"
    assert recommendations[0]["recommended_ope_value"] == 2.0
    assert recommendations[0]["passing_estimator_count"] == 2


def test_weight_diagnostics_separate_postprocessing_from_clipping() -> None:
    raw = np.array([2.0, 4.0, 6.0])
    normalized = raw / np.mean(raw)
    diagnostics = summarize_weights(normalized, raw_weights=raw)
    assert diagnostics["postprocessing_changed_fraction"] == 1.0
    assert diagnostics["clipping_fraction"] == 0.0
    assert diagnostics["negative_raw_fraction"] == 0.0

    clipped = summarize_weights(np.array([0.0, 1.0, 2.0]), raw_weights=np.array([-1.0, 1.0, 2.0]))
    assert clipped["clipping_fraction"] == pytest.approx(1.0 / 3.0)
    assert clipped["negative_raw_fraction"] == pytest.approx(1.0 / 3.0)


def test_high_stakes_guardrail_uses_final_clipping_not_normalization() -> None:
    base = {
        "status": "ok",
        "estimator": "neural_network_stable",
        "ope_value_estimate": 1.0,
        "effective_sample_size_fraction": 0.5,
        "clipping_fraction": 0.0,
        "postprocessing_changed_fraction": 1.0,
        "fixed_point_rel_change_final": 0.01,
        "weight_max": 5.0,
        "weight_q99_to_median": 2.0,
    }
    assert _high_stakes_diagnostic_status(base)["diagnostic_status"] == "pass"

    capped = {**base, "projection_clipped_fraction_final": 0.20}
    capped_status = _high_stakes_diagnostic_status(capped)
    assert capped_status["diagnostic_status"] == "fail"
    assert "clipping fraction" in capped_status["diagnostic_reason"]


def test_projection_and_damping_helpers() -> None:
    projected, info = _project_nonnegative_normalized(
        np.array([-1.0, 0.0, np.nan, np.inf, 4.0]),
        max_value=5.0,
        normalize=True,
        return_info=True,
    )
    assert np.all(np.isfinite(projected))
    assert np.all(projected >= 0.0)
    assert np.max(projected) <= 5.0
    assert np.isclose(np.mean(projected), 1.0)
    assert info["projection_clipped_fraction"] > 0.0

    all_zero = _project_nonnegative_normalized(np.zeros(4), normalize=True)
    assert np.allclose(all_zero, 1.0)

    current = np.array([1.0, 2.0, 3.0])
    update = np.array([3.0, 2.0, 1.0])
    assert np.allclose(_damped_update(current, update, 1.0), update)
    assert np.allclose(_damped_update(current, update, 0.25), 0.75 * current + 0.25 * update)


def test_pseudo_clipping_and_sample_weights_helpers() -> None:
    clipped, diag = _clip_pseudo_outcomes(
        np.array([0.0, 1.0, 2.0, 100.0]),
        enabled=True,
        pseudo_outcome_max=None,
        pseudo_outcome_upper_quantile=0.75,
        pseudo_outcome_min=0.0,
        target_min=0.0,
        target_max=None,
    )
    assert np.all(np.isfinite(clipped))
    assert diag["pseudo_outcome_cap"] < 100.0
    assert diag["pseudo_outcome_clipped_fraction"] > 0.0
    for key in ("pseudo_outcome_p95", "pseudo_outcome_p99", "pseudo_outcome_max", "pseudo_outcome_mean"):
        assert key in diag

    weights, weight_diag = _make_occupancy_sample_weights(
        mode="sqrt_target",
        action_ratio=np.ones(4),
        target=np.array([0.0, 1.0, 4.0, 100.0]),
        max_value=3.0,
    )
    assert np.all(np.isfinite(weights))
    assert np.max(weights) <= 3.0
    assert np.isclose(np.mean(weights), 1.0, atol=0.2)
    assert weight_diag["sample_weight_clipped_fraction"] > 0.0


def test_fold_splits_and_moment_calibration() -> None:
    folds = _make_fold_indices(17, 4, 123)
    combined = np.sort(np.concatenate(folds))
    assert np.array_equal(combined, np.arange(17))
    assert sum(len(fold) for fold in folds) == 17

    rng = np.random.default_rng(44)
    states = rng.normal(size=(120, 2))
    actions = rng.normal(size=(120, 1))
    fit = fit_importance_ratio_lgbm(
        S=states,
        A=actions,
        A_pi=actions.copy(),
        num_boost_round=3,
        early_stopping_rounds=0,
        refit_on_all_data=False,
        show_tqdm=False,
        moment_calibration="scalar",
        prediction_max=None,
        lgb_params={"min_data_in_leaf": 10, "num_leaves": 7, "verbose": -1},
    )
    assert np.isclose(np.mean(fit["w_hat"]), 1.0)
    assert fit["calibration"]["applied"]


def test_crossfit_and_cv_smoke_on_discrete() -> None:
    dataset = make_discrete_dataset(setting="discrete_chain", gamma=0.5, sample_size=90, seed=1)
    occupancy = OccupancyRegressionConfig(
        num_iterations=2,
        mcmc_samples=3,
        batch_size=64,
        show_progress=False,
        lgb_params={"min_data_in_leaf": 10, "num_leaves": 7, "verbose": -1},
    )
    action = ActionRatioConfig(
        num_boost_round=3,
        early_stopping_rounds=0,
        refit_on_all_data=False,
        show_progress=False,
        crossfit_folds=2,
        moment_calibration="scalar",
        lgb_params={"min_data_in_leaf": 10, "num_leaves": 7, "verbose": -1},
    )
    transition = TransitionRatioConfig(
        num_boost_round=3,
        permutation_samples=2,
        early_stopping_rounds=0,
        refit_on_all_data=False,
        show_progress=False,
        crossfit_folds=2,
        lgb_params={"min_data_in_leaf": 10, "num_leaves": 7, "verbose": -1},
    )
    tuned = tune_discounted_occupancy_ratio_cv(
        states=dataset.states,
        actions=dataset.actions,
        next_states=dataset.next_states,
        target_actions=dataset.target_actions,
        gamma=dataset.gamma,
        occupancy=occupancy,
        action_ratio=action,
        transition_ratio=transition,
        occupancy_grid=({"fixed_point_damping": 0.5},),
        cv_folds=2,
        seed=7,
        fit_final=True,
    )
    model = tuned["model"]
    weights = model.predict_state_action_ratio(dataset.states, dataset.actions)
    assert np.all(np.isfinite(weights))
    assert model.to_legacy_dict()["nuisance_crossfit"]["enabled"]


def test_discounted_occupancy_accepts_logistic_nuisance_configs() -> None:
    dataset = make_discrete_dataset(setting="discrete_chain", gamma=0.5, sample_size=80, seed=3)
    model = fit_discounted_occupancy_ratio(
        states=dataset.states,
        actions=dataset.actions,
        next_states=dataset.next_states,
        target_actions=dataset.target_actions,
        gamma=dataset.gamma,
        occupancy=OccupancyRegressionConfig(
            num_iterations=2,
            mcmc_samples=3,
            batch_size=64,
            show_progress=False,
            lgb_params={"min_data_in_leaf": 10, "num_leaves": 7, "verbose": -1},
        ),
        action_ratio=ActionRatioConfig(
            density_ratio_loss="logistic",
            logistic_logit_clip=8.0,
            num_boost_round=3,
            early_stopping_rounds=0,
            refit_on_all_data=False,
            show_progress=False,
            lgb_params={"min_data_in_leaf": 10, "num_leaves": 7, "verbose": -1},
        ),
        transition_ratio=TransitionRatioConfig(
            density_ratio_loss="logistic",
            logistic_logit_clip=8.0,
            num_boost_round=3,
            permutation_samples=2,
            early_stopping_rounds=0,
            refit_on_all_data=False,
            show_progress=False,
            lgb_params={"min_data_in_leaf": 10, "num_leaves": 7, "verbose": -1},
        ),
    )
    weights = model.predict_state_action_ratio(dataset.states, dataset.actions)
    assert np.all(np.isfinite(weights))
    legacy = model.to_legacy_dict()
    assert legacy["iw_density_ratio_loss"] == "logistic"
    assert legacy["k_density_ratio_loss"] == "logistic"


def test_boosted_huber_and_squared_smoke_on_discrete() -> None:
    dataset = make_discrete_dataset(setting="discrete_chain", gamma=0.5, sample_size=120, seed=0)
    config = OccupancyRatioBenchmarkConfig(
        stage="smoke",
        seeds=(0,),
        sample_sizes=(120,),
        gammas=(0.5,),
        settings=("discrete_chain",),
        estimators=("boosted_tree",),
        include_google_dual_dice=False,
        boosted_num_iterations=4,
        boosted_mcmc_samples=4,
        boosted_batch_size=64,
        boosted_losses=("squared", "huber"),
        write_plots=False,
    )
    squared = estimate_boosted_tree(dataset, config, loss="squared")
    huber = estimate_boosted_tree(dataset, config, loss="huber")
    assert squared.status == "ok"
    assert huber.status == "ok"
    assert np.isfinite(squared.diagnostics["log_ratio_rmse"])
    assert np.isfinite(huber.diagnostics["log_ratio_rmse"])
    assert "huber_delta_final" in huber.diagnostics
    assert "fixed_point_rel_change_final" in huber.diagnostics
    assert huber.diagnostics["history_json_friendly"] == 1.0


def test_neural_action_ratio_no_shift_stays_near_one() -> None:
    rng = np.random.default_rng(123)
    states = rng.normal(size=(80, 2)).astype(np.float32)
    actions = rng.normal(size=(80, 1)).astype(np.float32)
    x = np.concatenate([states, actions], axis=1)
    fit = fit_action_ratio_neural(
        x,
        x.copy(),
        NeuralActionRatioConfig(max_steps=5, batch_size=32, hidden_dims=(16,), seed=123),
    )
    assert np.max(np.abs(fit["w_hat"] - 1.0)) < 1e-8
    assert fit["calibration"]["method"] == "none"
    assert "clipped_fraction" in fit["w_hat_summary"]


def test_neural_logistic_nuisance_smokes() -> None:
    rng = np.random.default_rng(456)
    states = rng.normal(size=(90, 2)).astype(np.float32)
    actions = rng.normal(size=(90, 1)).astype(np.float32)
    x_beh = np.concatenate([states, actions], axis=1)
    action_fit = fit_action_ratio_neural(
        x_beh,
        x_beh.copy(),
        NeuralActionRatioConfig(
            max_steps=2,
            batch_size=32,
            hidden_dims=(12,),
            seed=456,
            density_ratio_loss="logistic",
        ),
    )
    assert np.all(np.isfinite(action_fit["w_hat"]))
    assert np.isclose(np.mean(action_fit["w_hat"]), 1.0)
    assert action_fit["density_ratio_loss"] == "logistic"
    assert np.isfinite(action_fit["prior_correction"])

    x_sa = rng.normal(size=(90, 3)).astype(np.float32)
    s_next = rng.normal(size=(90, 2)).astype(np.float32)
    s_ref = rng.normal(size=(90, 2)).astype(np.float32)
    trans_fit = fit_transition_ratio_neural(
        x_sa,
        s_next,
        s_ref,
        NeuralTransitionRatioConfig(
            max_steps=2,
            batch_size=32,
            hidden_dims=(12,),
            permutation_samples=2,
            seed=457,
            density_ratio_loss="logistic",
        ),
    )
    assert np.all(np.isfinite(trans_fit["k_hat"]))
    assert np.all(trans_fit["k_hat"] >= 0.0)
    assert trans_fit["density_ratio_loss"] == "logistic"
    assert np.isfinite(trans_fit["prior_correction"])
    assert trans_fit["reference_uses_initial_states"]


def test_neural_moment_calibration_and_crossfit_smoke() -> None:
    rng = np.random.default_rng(321)
    x_beh = rng.normal(size=(72, 3)).astype(np.float32)
    x_pi = (x_beh + rng.normal(scale=0.15, size=x_beh.shape)).astype(np.float32)
    fit = fit_action_ratio_neural(
        x_beh,
        x_pi,
        NeuralActionRatioConfig(
            max_steps=2,
            batch_size=24,
            hidden_dims=(12,),
            seed=321,
            moment_calibration="scalar",
            crossfit_folds=2,
            patience=2,
        ),
    )
    assert np.all(np.isfinite(fit["w_hat"]))
    assert np.isclose(np.mean(fit["w_hat"]), 1.0)
    assert fit["calibration"]["applied"]
    assert fit["crossfit"]["enabled"]
    assert fit["crossfit"]["folds"] == 2
    params = list(fit["predictor"].model.parameters())
    assert {param.device.type for param in params} == {"cpu"}
    assert not any(param.requires_grad for param in params)


def test_neural_optional_torch_error_is_actionable(monkeypatch) -> None:
    import occupancy_ratio.fit_occupancy_ratio_neural as neural

    monkeypatch.setattr(neural, "torch", None)
    monkeypatch.setattr(neural, "nn", None)
    with pytest.raises(ModuleNotFoundError, match=r"occupancy-ratio\[neural\]"):
        neural.fit_action_ratio_neural(
            np.zeros((4, 2), dtype=np.float32),
            np.zeros((4, 2), dtype=np.float32),
            NeuralActionRatioConfig(max_steps=1),
        )


def test_neural_model_prediction_helpers_smoke() -> None:
    dataset = make_discrete_dataset(setting="discrete_chain", gamma=0.5, sample_size=80, seed=0)
    model = fit_discounted_occupancy_ratio_neural(
        states=dataset.states,
        actions=dataset.actions,
        next_states=dataset.next_states,
        target_actions=dataset.target_actions,
        gamma=float(dataset.gamma),
        occupancy=NeuralOccupancyRegressionConfig(
            num_iterations=2,
            gradient_steps_per_iteration=1,
            mcmc_samples=2,
            batch_size=32,
            hidden_dims=(16,),
            seed=0,
            loss="squared",
        ),
        action_ratio=NeuralActionRatioConfig(max_steps=2, batch_size=32, hidden_dims=(16,), seed=1),
        transition_ratio=NeuralTransitionRatioConfig(
            max_steps=2,
            batch_size=32,
            permutation_samples=2,
            hidden_dims=(16,),
            seed=2,
        ),
    )
    raw = model.predict_state_action_ratio(dataset.states, dataset.actions, clip=False)
    weights = model.predict_state_action_ratio(dataset.states, dataset.actions, clip=True)
    action = model.predict_action_ratio(dataset.states, dataset.actions)
    state = model.predict_state_ratio(dataset.states, dataset.actions)
    assert raw.shape == (dataset.n,)
    assert weights.shape == (dataset.n,)
    assert model.diagnostics["validation_warmup_iterations"] == 1
    assert model.diagnostics["accepted_count"] >= 1
    assert np.all(np.isfinite(raw))
    assert np.all(np.isfinite(weights))
    assert np.all(weights >= 0.0)
    assert np.all(np.isfinite(action))
    assert state.shape == (dataset.n,)
    assert isinstance(model.to_legacy_dict(), dict)


def test_neural_discounted_crossfit_uses_fold_target_builder() -> None:
    dataset = make_discrete_dataset(setting="discrete_chain", gamma=0.5, sample_size=54, seed=6)
    model = fit_discounted_occupancy_ratio_neural(
        states=dataset.states,
        actions=dataset.actions,
        next_states=dataset.next_states,
        target_actions=dataset.target_actions,
        gamma=float(dataset.gamma),
        occupancy=NeuralOccupancyRegressionConfig(
            num_iterations=1,
            gradient_steps_per_iteration=1,
            mcmc_samples=1,
            batch_size=32,
            hidden_dims=(8,),
            seed=60,
        ),
        action_ratio=NeuralActionRatioConfig(
            max_steps=1,
            batch_size=32,
            hidden_dims=(8,),
            seed=61,
            crossfit_folds=2,
            crossfit_seed=600,
        ),
        transition_ratio=NeuralTransitionRatioConfig(
            max_steps=1,
            batch_size=32,
            permutation_samples=1,
            hidden_dims=(8,),
            seed=62,
            crossfit_folds=2,
            crossfit_seed=600,
        ),
    )
    legacy = model.to_legacy_dict()
    assert model.diagnostics["nuisance_crossfit_enabled"]
    assert legacy["nuisance_crossfit"]["enabled"]
    assert legacy["nuisance_crossfit"]["target_builder"]
    assert any(row.get("crossfit_target_builder") is True for row in model.history)


def test_neural_discounted_logistic_nuisance_combinations() -> None:
    dataset = make_discrete_dataset(setting="discrete_chain", gamma=0.5, sample_size=60, seed=3)
    for action_loss, transition_loss in (("logistic", "lsif"), ("lsif", "logistic"), ("logistic", "logistic")):
        model = fit_discounted_occupancy_ratio_neural(
            states=dataset.states,
            actions=dataset.actions,
            next_states=dataset.next_states,
            target_actions=dataset.target_actions,
            gamma=float(dataset.gamma),
            occupancy=NeuralOccupancyRegressionConfig(
                num_iterations=1,
                gradient_steps_per_iteration=1,
                mcmc_samples=2,
                batch_size=32,
                hidden_dims=(12,),
                seed=30,
            ),
            action_ratio=NeuralActionRatioConfig(
                max_steps=1,
                batch_size=32,
                hidden_dims=(12,),
                seed=31,
                density_ratio_loss=action_loss,
            ),
            transition_ratio=NeuralTransitionRatioConfig(
                max_steps=1,
                batch_size=32,
                permutation_samples=1,
                hidden_dims=(12,),
                seed=32,
                density_ratio_loss=transition_loss,
            ),
        )
        weights = model.predict_state_action_ratio(dataset.states, dataset.actions)
        assert np.all(np.isfinite(weights))
        assert model.diagnostics["action_density_ratio_loss"] == action_loss
        assert model.diagnostics["transition_density_ratio_loss"] == transition_loss


def test_neural_huber_and_squared_smoke_on_discrete() -> None:
    dataset = make_discrete_dataset(setting="discrete_chain", gamma=0.5, sample_size=80, seed=0)
    config = OccupancyRatioBenchmarkConfig(
        stage="smoke",
        seeds=(0,),
        sample_sizes=(80,),
        gammas=(0.5,),
        settings=("discrete_chain",),
        estimators=("neural_network",),
        include_google_dual_dice=False,
        neural_num_iterations=2,
        neural_gradient_steps_per_iteration=1,
        neural_mcmc_samples=2,
        neural_batch_size=32,
        neural_hidden_dims=(16,),
        neural_action_steps=2,
        neural_transition_steps=2,
        neural_transition_permutation_samples=2,
        neural_losses=("squared", "huber"),
        write_plots=False,
    )
    squared = estimate_neural_network(dataset, config, loss="squared")
    huber = estimate_neural_network(dataset, config, loss="huber")
    assert squared.status == "ok"
    assert huber.status == "ok"
    assert np.isfinite(squared.diagnostics["log_ratio_rmse"])
    assert np.isfinite(huber.diagnostics["log_ratio_rmse"])
    assert "fixed_point_rel_change_final" in squared.diagnostics
    assert "huber_delta_final" in huber.diagnostics
    assert huber.diagnostics["history_json_friendly"] == 1.0


def test_neural_cv_smoke_on_discrete() -> None:
    dataset = make_discrete_dataset(setting="discrete_chain", gamma=0.5, sample_size=50, seed=2)
    tuned = tune_discounted_occupancy_ratio_neural_cv(
        states=dataset.states,
        actions=dataset.actions,
        next_states=dataset.next_states,
        target_actions=dataset.target_actions,
        gamma=float(dataset.gamma),
        occupancy=NeuralOccupancyRegressionConfig(
            num_iterations=1,
            gradient_steps_per_iteration=1,
            mcmc_samples=2,
            batch_size=32,
            hidden_dims=(12,),
            seed=10,
        ),
        action_ratio=NeuralActionRatioConfig(max_steps=1, batch_size=32, hidden_dims=(12,), seed=11),
        transition_ratio=NeuralTransitionRatioConfig(
            max_steps=1,
            batch_size=32,
            permutation_samples=2,
            hidden_dims=(12,),
            seed=12,
        ),
        occupancy_grid=({"fixed_point_damping": 0.5}, {"fixed_point_damping": 0.75}),
        cv_folds=2,
        seed=7,
        fit_final=True,
    )
    assert np.isfinite(tuned["best_score"])
    assert len(tuned["cv_rows"]) >= 2
    weights = tuned["model"].predict_state_action_ratio(dataset.states, dataset.actions)
    assert np.all(np.isfinite(weights))


def test_product_automl_boosted_smoke_and_capped_candidates() -> None:
    dataset = make_discrete_dataset(setting="discrete_chain", gamma=0.5, sample_size=48, seed=22)
    space = OccupancySearchSpace(
        boosted_occupancy=OccupancyRegressionConfig(
            num_iterations=1,
            mcmc_samples=2,
            batch_size=32,
            show_progress=False,
            lgb_params={"min_data_in_leaf": 1, "num_leaves": 7, "verbose": -1},
        ),
        boosted_action_ratio=ActionRatioConfig(
            num_boost_round=1,
            early_stopping_rounds=0,
            refit_on_all_data=False,
            show_progress=False,
            lgb_params={"min_data_in_leaf": 1, "num_leaves": 7, "verbose": -1},
        ),
        boosted_source_state_ratio=SourceStateRatioConfig(
            num_boost_round=1,
            early_stopping_rounds=0,
            refit_on_all_data=False,
            show_progress=False,
            lgb_params={"min_data_in_leaf": 1, "num_leaves": 7, "verbose": -1},
        ),
        boosted_transition_ratio=TransitionRatioConfig(
            num_boost_round=1,
            permutation_samples=1,
            early_stopping_rounds=0,
            refit_on_all_data=False,
            show_progress=False,
            lgb_params={"min_data_in_leaf": 1, "num_leaves": 7, "verbose": -1},
        ),
    )
    result = tune_occupancy_ratio_auto(
        states=dataset.states,
        actions=dataset.actions,
        next_states=dataset.next_states,
        target_actions=dataset.target_actions,
        gamma=float(dataset.gamma),
        rewards=dataset.rewards,
        search_space=space,
        config=OccupancyTuningConfig(families=("boosted",), cv_folds=2, max_candidates=2, promotion_candidates=1, seed=9),
    )
    assert result.model is not None
    assert result.selected_family == "boosted"
    assert len([row for row in result.candidates if row.budget_stage == "screen"]) == 2
    assert all("moment_balance" in row.metrics for row in result.candidates)
    assert any(np.isfinite(row.metrics.get("moment_balance", np.nan)) for row in result.candidates)
    weights = result.model.predict_state_action_ratio(dataset.states, dataset.actions)
    assert np.all(np.isfinite(weights))
    assert not any("truth" in key for row in result.candidates for key in row.metrics)


def test_product_automl_grouped_folds_and_source_candidates() -> None:
    dataset = make_discrete_dataset(setting="discrete_chain", gamma=0.5, sample_size=40, seed=23)
    groups = np.repeat(np.arange(10), 4)
    from occupancy_ratio.tuning import _make_folds

    for fold_idx in _make_folds(groups.shape[0], 2, 12, groups=groups):
        assert all(np.sum(groups[fold_idx] == group) == np.sum(groups == group) for group in np.unique(groups[fold_idx]))
    source = dataset.states[:10]
    result = tune_occupancy_ratio_auto(
        states=dataset.states,
        actions=dataset.actions,
        next_states=dataset.next_states,
        target_actions=dataset.target_actions,
        gamma=float(dataset.gamma),
        initial_states=source,
        initial_weights=np.ones(source.shape[0]),
        groups=groups,
        search_space=OccupancySearchSpace(
            boosted_occupancy=OccupancyRegressionConfig(
                num_iterations=1,
                mcmc_samples=1,
                batch_size=32,
                show_progress=False,
                lgb_params={"min_data_in_leaf": 1, "num_leaves": 7, "verbose": -1},
            ),
            boosted_action_ratio=ActionRatioConfig(
                num_boost_round=1,
                early_stopping_rounds=0,
                refit_on_all_data=False,
                show_progress=False,
                lgb_params={"min_data_in_leaf": 1, "num_leaves": 7, "verbose": -1},
            ),
            boosted_source_state_ratio=SourceStateRatioConfig(
                num_boost_round=1,
                early_stopping_rounds=0,
                refit_on_all_data=False,
                show_progress=False,
                lgb_params={"min_data_in_leaf": 1, "num_leaves": 7, "verbose": -1},
            ),
            boosted_transition_ratio=TransitionRatioConfig(
                num_boost_round=1,
                permutation_samples=1,
                early_stopping_rounds=0,
                refit_on_all_data=False,
                show_progress=False,
                lgb_params={"min_data_in_leaf": 1, "num_leaves": 7, "verbose": -1},
            ),
            boosted_candidates=(
                {
                    "occupancy": {"fixed_point_damping": 0.5},
                    "action_ratio": {"prediction_max": 50.0},
                    "transition_ratio": {"prediction_max": 50.0},
                    "source_state_ratio": {"prediction_max": 10.0},
                },
            ),
        ),
        config=OccupancyTuningConfig(families=("boosted",), cv_folds=2, max_candidates=1, promotion_candidates=1, seed=12),
    )
    assert result.candidates[0].overrides["source_state_ratio"]
    assert {fold.fold for fold in result.folds} == {0, 1}
    assert result.model.diagnostics["source_state_ratio_enabled"]


def test_product_automl_fast_budget_and_stable_fallback_policy() -> None:
    from occupancy_ratio.tuning import (
        _final_refit_penalty,
        _make_candidates,
        _near_uniform_penalty,
        _should_fallback_to_baseline,
        _weak_moment_instability_fallback,
        _weight_quality_from_values,
    )

    assert tuple(OccupancyTuningConfig().families) == ("neural",)
    fast_cfg = OccupancyTuningConfig(budget="fast", max_candidates=16, promotion_candidates=4)
    candidates = _make_candidates(OccupancySearchSpace(), fast_cfg, has_initial_states=False)
    assert len(candidates) == 8
    assert not any(row["overrides"].get("backend", {}).get("name") == "google_dualdice" for row in candidates)
    google_candidates = _make_candidates(
        OccupancySearchSpace(),
        OccupancyTuningConfig(budget="fast", max_candidates=16, promotion_candidates=4, include_google_dualdice=True),
        has_initial_states=True,
        has_initial_actions=True,
    )
    assert any(row["overrides"].get("backend", {}).get("name") == "google_dualdice" for row in google_candidates)
    no_initial_action_candidates = _make_candidates(
        OccupancySearchSpace(),
        OccupancyTuningConfig(budget="fast", max_candidates=16, promotion_candidates=4, include_google_dualdice=True),
        has_initial_states=True,
        has_initial_actions=False,
    )
    assert not any(row["overrides"].get("backend", {}).get("name") == "google_dualdice" for row in no_initial_action_candidates)
    assert candidates[0]["candidate_id"] == "neural_000"
    assert candidates[0]["candidate_label"] == "neural_stable"
    assert candidates[0]["overrides"]["occupancy"]["loss"] == "huber"
    assert candidates[0]["overrides"]["action_ratio"]["density_ratio_loss"] == "lsif"
    assert candidates[1]["candidate_label"] == "neural_google_parity"
    assert candidates[1]["overrides"]["occupancy"]["hidden_dims"][0] >= 256
    assert candidates[1]["overrides"]["occupancy"]["activation"] == "relu"
    source_candidates = _make_candidates(
        OccupancySearchSpace(),
        OccupancyTuningConfig(max_candidates=32, promotion_candidates=4),
        has_initial_states=True,
    )
    assert source_candidates[0]["overrides"].get("modes", {}) == {}
    assert any(
        row["overrides"].get("modes", {}).get("initial_ratio_mode") == "factored"
        and row["overrides"].get("modes", {}).get("one_step_ratio_mode") == "factored"
        for row in source_candidates
    )
    boosted_candidates = _make_candidates(
        OccupancySearchSpace(),
        OccupancyTuningConfig(families=("boosted",), budget="fast", max_candidates=16, promotion_candidates=4),
        has_initial_states=False,
    )
    assert boosted_candidates[0]["candidate_id"] == "boosted_000"
    assert boosted_candidates[0]["overrides"]["occupancy"]["loss"] == "huber"
    assert boosted_candidates[0]["overrides"]["action_ratio"]["density_ratio_loss"] == "lsif"
    balanced_candidates = _make_candidates(
        OccupancySearchSpace(),
        OccupancyTuningConfig(budget="balanced", max_candidates=16, promotion_candidates=4),
        has_initial_states=False,
    )
    assert 8 <= len(balanced_candidates) <= 16
    assert not any(row["candidate_label"] in {"neural_low_lr", "neural_high_lr"} for row in balanced_candidates)
    assert not any(
        row["overrides"].get("occupancy", {}).get("loss") == "squared"
        for row in balanced_candidates
    )
    with pytest.raises(ValueError, match="budget"):
        OccupancyTuningConfig(budget="slow")

    assert _near_uniform_penalty(1.0, 1.0, 0.0) > 0.0
    assert _near_uniform_penalty(1.0, 1.0, 0.20) == 0.0
    assert _near_uniform_penalty(1.0, 0.0, 0.0) == 0.0
    low_ess_truth_like = np.r_[np.full(5, 100.0), np.full(995, 500.0 / 995.0)]
    uniform_collapse = np.ones(1000)
    occupancy = OccupancyRegressionConfig(occupancy_ratio_max=200.0)
    assert _weight_quality_from_values(low_ess_truth_like, occupancy, action_shift=1.0) < (
        _weight_quality_from_values(uniform_collapse, occupancy, action_shift=1.0)
    )
    assert _final_refit_penalty(
        {
            "weight_quality": 0.0,
            "ess_fraction": 0.01,
            "clipped_fraction": 0.0,
            "n_weights": 1000.0,
        }
    ) == 0.0
    assert _final_refit_penalty(
        {
            "weight_quality": 0.0,
            "ess_fraction": 0.0005,
            "clipped_fraction": 0.0,
            "n_weights": 1000.0,
        }
    ) > 0.0

    selected = CandidateResult(
        candidate_id="boosted_003",
        family="boosted",
        budget_stage="full",
        overrides={},
        fold_results=[],
        metrics={"final_weight_quality": 0.030, "final_ess_fraction": 0.96, "final_clipped_fraction": 0.0},
        score=0.20,
        runtime_sec=10.0,
    )
    baseline = CandidateResult(
        candidate_id="boosted_000",
        family="boosted",
        budget_stage="full",
        overrides={},
        fold_results=[],
        metrics={"final_weight_quality": 0.032, "final_ess_fraction": 0.95, "final_clipped_fraction": 0.0},
        score=0.22,
        runtime_sec=2.0,
    )
    assert _should_fallback_to_baseline(
        selected=selected,
        baseline=baseline,
        selected_score=0.20,
        baseline_score=0.22,
        cfg=OccupancyTuningConfig(),
    )

    selected_low_ess = CandidateResult(
        candidate_id="boosted_004",
        family="boosted",
        budget_stage="full",
        overrides={},
        fold_results=[],
        metrics={
            "final_weight_quality": 0.010,
            "final_ess_fraction": 0.01,
            "final_clipped_fraction": 0.0,
            "final_n_weights": 1000.0,
        },
        score=0.20,
        runtime_sec=2.0,
    )
    smooth_baseline = CandidateResult(
        candidate_id="boosted_000",
        family="boosted",
        budget_stage="full",
        overrides={},
        fold_results=[],
        metrics={
            "final_weight_quality": 0.012,
            "final_ess_fraction": 0.99,
            "final_clipped_fraction": 0.0,
            "final_n_weights": 1000.0,
        },
        score=0.22,
        runtime_sec=2.0,
    )
    assert not _should_fallback_to_baseline(
        selected=selected_low_ess,
        baseline=smooth_baseline,
        selected_score=0.20,
        baseline_score=0.22,
        cfg=OccupancyTuningConfig(),
    )

    weak_moment_unstable = CandidateResult(
        candidate_id="boosted_004",
        family="boosted",
        budget_stage="full",
        overrides={},
        fold_results=[],
        metrics={
            "moment_balance": 10.0,
            "final_ess_fraction": 0.05,
            "final_weight_cv": 4.0,
            "final_weight_quality": 0.01,
        },
        score=0.10,
        runtime_sec=2.0,
    )
    stable_default = CandidateResult(
        candidate_id="boosted_000",
        family="boosted",
        budget_stage="full",
        overrides={},
        fold_results=[],
        metrics={
            "moment_balance": 10.5,
            "final_ess_fraction": 0.20,
            "final_weight_cv": 2.0,
            "final_weight_quality": 0.50,
        },
        score=0.80,
        runtime_sec=2.0,
    )
    assert _weak_moment_instability_fallback(
        selected=weak_moment_unstable,
        baseline=stable_default,
        cfg=OccupancyTuningConfig(),
    )
    assert _should_fallback_to_baseline(
        selected=weak_moment_unstable,
        baseline=stable_default,
        selected_score=0.10,
        baseline_score=1.00,
        cfg=OccupancyTuningConfig(),
    )
    strong_moment_unstable = CandidateResult(
        candidate_id="boosted_001",
        family="boosted",
        budget_stage="full",
        overrides={},
        fold_results=[],
        metrics={
            "moment_balance": 5.0,
            "final_ess_fraction": 0.05,
            "final_weight_cv": 4.0,
            "final_weight_quality": 0.01,
        },
        score=0.10,
        runtime_sec=2.0,
    )
    assert not _weak_moment_instability_fallback(
        selected=strong_moment_unstable,
        baseline=stable_default,
        cfg=OccupancyTuningConfig(),
    )
    collapsed_stable_default = CandidateResult(
        candidate_id="boosted_000",
        family="boosted",
        budget_stage="full",
        overrides={},
        fold_results=[],
        metrics={
            "moment_balance": 10.5,
            "final_action_shift": 1.0,
            "final_ess_fraction": 1.0,
            "final_weight_cv": 0.0,
            "final_weight_quality": 1.0,
        },
        score=0.80,
        runtime_sec=2.0,
    )
    assert not _weak_moment_instability_fallback(
        selected=weak_moment_unstable,
        baseline=collapsed_stable_default,
        cfg=OccupancyTuningConfig(),
    )


def test_product_moment_extra_blocks_are_candidate_independent() -> None:
    dataset = make_discrete_dataset(setting="discrete_chain", gamma=0.5, sample_size=80, seed=30)
    train = np.arange(50)
    valid = np.arange(50, 80)
    builder = _FoldFeatureBuilder(
        S_train=dataset.states[train],
        A_train=dataset.actions[train],
        rewards_train=dataset.rewards[train],
        gamma=float(dataset.gamma),
        seed=301,
        geometry_features=2,
        rff_features=4,
        value_iterations=2,
        value_patience=1,
        S_next_train=dataset.next_states[train],
        A_next_train=dataset.target_actions[train],
        A_target_train=dataset.target_actions[train],
        extra_blocks=("second_order", "multiscale_rff", "support", "policy_shift"),
        multiscale_rff_scales=(0.5, 2.0),
        strata_quantiles=(0.25, 0.5, 0.75),
    )
    blocks = builder.blocks(
        S_eval=dataset.states[valid],
        A_eval=dataset.actions[valid],
        S_next=dataset.next_states[valid],
        A_next=dataset.target_actions[valid],
        S_initial=dataset.initial_states[:20],
        A_initial=dataset.initial_actions[:20],
    )
    expected = {
        "second_order",
        "rff_multiscale",
        "support",
        "support_strata",
        "policy_shift",
        "policy_shift_strata",
    }
    assert expected.issubset(blocks)
    for name in expected:
        current, successor, initial = blocks[name]
        assert current.shape[0] == valid.shape[0]
        assert successor.shape[1] == current.shape[1]
        assert initial.shape[1] == current.shape[1]
        assert np.all(np.isfinite(current))


def test_moment_evaluator_ablation_pairs_against_current() -> None:
    base = {
        "stage": "smoke",
        "profile": "smoke",
        "matrix_id": "tabular",
        "setting": "linear_gaussian",
        "dataset_variant": "",
        "policy_shift": "1.0",
        "gamma": "0.9",
        "sample_size": "300",
        "seed": "0",
        "estimator": "neural_network_stable",
        "status": "ok",
        "ratio_truth_available": "1.0",
        "log_ratio_rmse": "0.2",
        "ope_value_abs_error": "0.1",
        "effective_sample_size_fraction": "0.7",
        "true_effective_sample_size_fraction": "0.6",
        "weight_cv": "0.4",
        "true_weight_cv": "0.5",
        "runtime_sec": "1.0",
    }
    rows = [
        {**base, "evaluator_id": "current", "ratio_normalized_l1": "0.5"},
        {**base, "evaluator_id": "support", "evaluator_extra_blocks": "support", "ratio_normalized_l1": "0.3"},
    ]
    deltas = paired_selection_delta_rows(rows)
    assert len(deltas) == 1
    assert deltas[0]["outcome"] == "win"
    summary = summarize_evaluator_rows(rows, deltas)
    support = next(row for row in summary if row["evaluator_id"] == "support")
    assert support["wins_vs_current"] == 1
    assert "support" in render_evaluator_report(summary, deltas)


def test_product_automl_can_tune_google_dualdice_as_neural_candidate(monkeypatch) -> None:
    import occupancy_ratio.tuning as tuning

    dataset = make_discrete_dataset(setting="discrete_chain", gamma=0.5, sample_size=24, seed=29)
    calls = []

    class FakeGoogleDualDICEModel:
        history = [{"loss": 0.25}]

        def predict_state_action_ratio(self, states, actions, *, clip=True):
            n = np.asarray(states).shape[0]
            return np.linspace(0.75, 1.25, n)

    def fake_fit_google_dualdice_occupancy_ratio(**kwargs):
        calls.append(kwargs)
        assert kwargs["initial_states"] is not None
        assert kwargs["initial_actions"] is not None
        assert kwargs["target_next_actions"] is not None
        return FakeGoogleDualDICEModel()

    monkeypatch.setattr(tuning, "fit_google_dualdice_occupancy_ratio", fake_fit_google_dualdice_occupancy_ratio)
    result = tune_occupancy_ratio_auto(
        states=dataset.states,
        actions=dataset.actions,
        next_states=dataset.next_states,
        target_actions=dataset.target_actions,
        gamma=float(dataset.gamma),
        initial_states=dataset.states[:6],
        initial_actions=dataset.target_actions[:6],
        search_space=OccupancySearchSpace(
            neural_candidates=(
                {
                    "backend": {"name": "google_dualdice"},
                    "google_dualdice": {"num_updates": 5, "batch_size": 8, "normalize_predictions": True},
                },
            ),
        ),
        config=OccupancyTuningConfig(
            families=("neural",),
            cv_folds=2,
            max_candidates=1,
            promotion_candidates=1,
            seed=17,
            include_google_dualdice=True,
        ),
    )

    assert calls
    assert result.selected_family == "neural"
    assert result.selected_overrides["backend"]["name"] == "google_dualdice"
    assert result.model is not None
    assert any(row.metrics.get("backend_google_dualdice") == 1.0 for row in result.candidates)
    assert all(np.isfinite(row.metrics.get("validation_loss", np.inf)) for row in result.candidates if row.fold_results)


def test_top_level_tuning_exports() -> None:
    import occupancy_ratio

    assert occupancy_ratio.tune_occupancy_ratio_auto
    assert occupancy_ratio.tune_occupancy_ratio
    assert occupancy_ratio.OccupancyTuningConfig
    assert occupancy_ratio.tune_discounted_occupancy_ratio_neural_cv


def test_runner_smoke_writes_outputs(tmp_path) -> None:
    config = OccupancyRatioBenchmarkConfig(
        stage="smoke",
        output_root=tmp_path,
        seeds=(0,),
        sample_sizes=(80,),
        gammas=(0.5,),
        settings=("discrete_chain",),
        estimators=("oracle", "boosted_tree", "neural_network"),
        include_google_dual_dice=False,
        boosted_num_iterations=3,
        boosted_mcmc_samples=3,
        boosted_batch_size=64,
        boosted_losses=("squared", "huber"),
        neural_num_iterations=2,
        neural_gradient_steps_per_iteration=1,
        neural_mcmc_samples=2,
        neural_batch_size=32,
        neural_hidden_dims=(16,),
        neural_action_steps=2,
        neural_transition_steps=2,
        neural_transition_permutation_samples=2,
        write_plots=False,
    )
    result = run_benchmark(config)
    assert result.results_path.exists()
    assert result.summary_path.exists()
    assert result.manifest_path.exists()
    assert result.defaults_report_path.exists()
    assert result.neural_vs_dice_path.exists()
    assert result.benchmark_readout_path.exists()
    assert {row["estimator"] for row in result.rows} >= {
        "oracle",
        "boosted_tree_stable",
        "neural_network_stable",
    }
    neural_rows = [row for row in result.rows if str(row["estimator"]).startswith("neural_network")]
    assert neural_rows
    assert neural_rows[0]["neural_stabilization_preset"] == "stable"
    assert neural_rows[0]["neural_normalize_transition_cache"] == 0.0
    assert neural_rows[0]["source_state_correction_applied"] == 1.0
    assert neural_rows[0]["source_state_ratio_enabled"] == 0.0
    assert neural_rows[0]["initial_joint_ratio_enabled"] == 1.0
    assert neural_rows[0]["one_step_direct_ratio_enabled"] == 1.0
    assert neural_rows[0]["initial_joint_ratio_updates"] > 0.0
    assert neural_rows[0]["one_step_direct_ratio_updates"] > 0.0


def test_runner_gym_smoke_writes_ope_rows_without_ratio_truth(tmp_path) -> None:
    pytest.importorskip("gymnasium")
    config = OccupancyRatioBenchmarkConfig(
        stage="smoke",
        output_root=tmp_path,
        seeds=(0,),
        sample_sizes=(30,),
        gammas=(0.9,),
        settings=("gym_pendulum",),
        estimators=("oracle", "boosted_tree_stable"),
        include_google_dual_dice=False,
        boosted_num_iterations=1,
        boosted_mcmc_samples=1,
        boosted_batch_size=32,
        gym_target_value_rollouts=2,
        write_plots=False,
    )
    result = run_benchmark(config)
    oracle_rows = [row for row in result.rows if row["estimator"] == "oracle"]
    boosted_rows = [row for row in result.rows if row["estimator"] == "boosted_tree_stable"]
    assert oracle_rows[0]["status"] == "skipped"
    assert boosted_rows[0]["status"] == "ok"
    assert boosted_rows[0]["ratio_truth_available"] == 0.0
    assert boosted_rows[0]["ope_value_abs_error"] != ""
    assert "log_ratio_rmse" not in boosted_rows[0]
    assert boosted_rows[0]["source_state_correction_applied"] == 1.0
    assert boosted_rows[0]["source_state_ratio_enabled"] == 0.0
    assert boosted_rows[0]["initial_joint_ratio_enabled"] == 1.0
    assert boosted_rows[0]["one_step_direct_ratio_enabled"] == 1.0


def test_neural_default_ablation_tags_rows_by_candidate_id() -> None:
    candidate = next(candidate for candidate in CANDIDATES if candidate.candidate_id == "stage_budget")
    rows = _tag_rows(
        [
            {
                "profile": "high_stakes",
                "setting": "discrete_chain",
                "gamma": 0.9,
                "sample_size": 1000,
                "seed": 0,
                "estimator": "neural_network_stable",
                "status": "ok",
                "ratio_normalized_l1": 0.2,
                "log_ratio_rmse": 0.1,
                "ope_value_abs_error": 0.05,
                "effective_sample_size_fraction": 0.7,
                "true_effective_sample_size_fraction": 0.6,
                "weight_cv": 0.4,
                "true_weight_cv": 0.5,
                "runtime_sec": 12.0,
            }
        ],
        candidate=candidate,
        matrix_id="tabular",
    )
    assert rows[0]["candidate_id"] == "stage_budget"
    assert rows[0]["matrix_id"] == "tabular"

    summary = summarize_ablation_rows(rows, audit_rows=[])
    assert summary[0]["candidate_id"] == "stage_budget"
    assert summary[0]["estimator"] == "neural_network_stable"
    assert summary[0]["controlled_rows"] == 1
    assert "stage_budget" in render_ablation_report(summary)


def test_defaults_report_selects_from_generated_csv(tmp_path) -> None:
    rows = [
        {
            "profile": "overnight",
            "setting": "openml_contextual_bandit",
            "dataset_variant": "31",
            "gamma": 0.9,
            "sample_size": 100,
            "seed": 0,
            "estimator": "boosted_tree_stable",
            "status": "ok",
            "ope_value_abs_error": 0.20,
            "effective_sample_size_fraction": 0.90,
        },
        {
            "profile": "overnight",
            "setting": "openml_contextual_bandit",
            "dataset_variant": "31",
            "gamma": 0.9,
            "sample_size": 100,
            "seed": 0,
            "estimator": "boosted_tree_stable_logistic_nuisance",
            "status": "ok",
            "ope_value_abs_error": 0.19,
            "effective_sample_size_fraction": 0.91,
            "ratio_normalized_l1": 0.05,
        },
        {
            "profile": "overnight",
            "setting": "openml_contextual_bandit",
            "dataset_variant": "31",
            "gamma": 0.9,
            "sample_size": 100,
            "seed": 0,
            "estimator": "boosted_tree_auto",
            "status": "ok",
            "ope_value_abs_error": 0.21,
            "effective_sample_size_fraction": 0.92,
        },
    ]
    results_path = tmp_path / "results.csv"
    write_csv(results_path, rows)
    paths = generate_defaults_report(results_path)
    assert paths["report"].exists()
    text = paths["report"].read_text(encoding="utf-8")
    assert "Recommended default: `boosted_tree_stable`" in text
    assert paths["neural_vs_dice"].exists()


def test_runner_boosted_logistic_auto_resume_and_timeout(tmp_path) -> None:
    base = dict(
        stage="smoke",
        output_root=tmp_path,
        seeds=(0,),
        sample_sizes=(50,),
        gammas=(0.5,),
        settings=("discrete_chain",),
        include_google_dual_dice=False,
        boosted_num_iterations=1,
        boosted_mcmc_samples=1,
        boosted_batch_size=32,
        neural_num_iterations=1,
        neural_gradient_steps_per_iteration=1,
        neural_mcmc_samples=1,
        neural_batch_size=32,
        neural_hidden_dims=(8,),
        neural_action_steps=1,
        neural_transition_steps=1,
        neural_transition_permutation_samples=1,
        write_plots=False,
    )
    config = OccupancyRatioBenchmarkConfig(
        **base,
        estimators=("boosted_tree_stable_logistic_nuisance", "boosted_tree_auto"),
    )
    result = run_benchmark(config)
    assert {row["estimator"] for row in result.rows} >= {
        "boosted_tree_stable_logistic_nuisance",
        "boosted_tree_auto",
    }
    logistic_rows = [row for row in result.rows if row["estimator"] == "boosted_tree_stable_logistic_nuisance"]
    assert logistic_rows[0]["nuisance_density_ratio_loss"] == "logistic"
    auto_rows = [row for row in result.rows if row["estimator"] == "boosted_tree_auto"]
    assert auto_rows[0]["selected_preset"]
    assert np.isfinite(float(auto_rows[0]["selection_score"]))

    resumed = run_benchmark(config)
    assert len(resumed.rows) == len(result.rows)

    timeout_base = {**base, "output_root": tmp_path / "timeout"}
    timeout_config = OccupancyRatioBenchmarkConfig(
        **timeout_base,
        estimators=("boosted_tree_stable_logistic_nuisance",),
        estimator_timeout_sec=0.001,
        resume=False,
    )
    timeout_result = run_benchmark(timeout_config)
    assert timeout_result.rows[0]["status"] == "timeout"
    assert timeout_result.rows[0]["error_type"] == "TimeoutError"
    assert float(timeout_result.rows[0]["timeout_sec"]) == pytest.approx(0.001)


def test_runner_neural_logistic_auto_resume_and_timeout(tmp_path) -> None:
    base = dict(
        stage="smoke",
        output_root=tmp_path,
        seeds=(0,),
        sample_sizes=(50,),
        gammas=(0.5,),
        settings=("discrete_chain",),
        include_google_dual_dice=False,
        boosted_num_iterations=1,
        boosted_mcmc_samples=1,
        boosted_batch_size=32,
        neural_num_iterations=1,
        neural_gradient_steps_per_iteration=1,
        neural_mcmc_samples=1,
        neural_batch_size=32,
        neural_hidden_dims=(8,),
        neural_action_steps=1,
        neural_transition_steps=1,
        neural_transition_permutation_samples=1,
        write_plots=False,
    )
    config = OccupancyRatioBenchmarkConfig(
        **base,
        estimators=("neural_network_stable_logistic_nuisance", "neural_network_auto"),
    )
    result = run_benchmark(config)
    assert {row["estimator"] for row in result.rows} >= {
        "neural_network_stable_logistic_nuisance",
        "neural_network_auto",
    }
    logistic_rows = [row for row in result.rows if row["estimator"] == "neural_network_stable_logistic_nuisance"]
    assert logistic_rows[0]["neural_density_ratio_loss"] == "logistic"
    assert logistic_rows[0]["neural_action_density_ratio_loss"] == "logistic"
    assert logistic_rows[0]["neural_transition_density_ratio_loss"] == "logistic"
    auto_rows = [row for row in result.rows if row["estimator"] == "neural_network_auto"]
    assert auto_rows[0]["selected_preset"]
    assert np.isfinite(float(auto_rows[0]["selection_score"]))

    resumed = run_benchmark(config)
    assert len(resumed.rows) == len(result.rows)

    timeout_base = {**base, "output_root": tmp_path / "timeout"}
    timeout_config = OccupancyRatioBenchmarkConfig(
        **timeout_base,
        estimators=("neural_network_stable_logistic_nuisance",),
        estimator_timeout_sec=0.001,
        resume=False,
    )
    timeout_result = run_benchmark(timeout_config)
    assert timeout_result.rows[0]["status"] == "timeout"
    assert timeout_result.rows[0]["error_type"] == "TimeoutError"
    assert float(timeout_result.rows[0]["timeout_sec"]) == pytest.approx(0.001)


def test_runner_cv_tuning_writes_tuning_rows(tmp_path) -> None:
    config = OccupancyRatioBenchmarkConfig(
        stage="smoke",
        output_root=tmp_path,
        seeds=(0,),
        sample_sizes=(120,),
        gammas=(0.5,),
        settings=("discrete_chain",),
        estimators=("boosted_tree",),
        boosted_estimator_presets=("stable",),
        include_google_dual_dice=False,
        tune_cv=True,
        cv_folds=2,
        cv_fixed_point_dampings=(0.5,),
        cv_occupancy_ratio_max_values=(50.0,),
        cv_nuisance_prediction_max_values=(50.0,),
        cv_moment_calibrations=("none",),
        boosted_num_iterations=2,
        boosted_mcmc_samples=3,
        boosted_batch_size=64,
        write_plots=False,
    )
    result = run_benchmark(config)
    assert result.tuning_path.exists()
    assert result.tuning_rows
    assert {
        "candidate_id",
        "family",
        "budget_stage",
        "metric_validation_loss",
        "metric_weight_quality",
        "metric_action_shift",
        "metric_weight_cv",
    } <= set(result.tuning_rows[0])
    selected_rows = [
        row
        for row in result.tuning_rows
        if row.get("tuning_stage") == "automl_candidate"
        and row.get("budget_stage") == "full"
        and float(row.get("selected", 0.0)) == 1.0
    ]
    assert selected_rows
    assert float(selected_rows[0]["promoted"]) == 1.0
    assert result.rows[0]["profile"] == "smoke"
