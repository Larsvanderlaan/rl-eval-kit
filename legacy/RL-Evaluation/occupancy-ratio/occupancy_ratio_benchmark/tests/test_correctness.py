from __future__ import annotations

import numpy as np
import pytest

from occupancy_ratio_benchmark.config import OccupancyRatioBenchmarkConfig
from occupancy_ratio_benchmark.defaults_report import generate_defaults_report

from occupancy_ratio.fit_importance_and_transition_ratios import (
    fit_importance_ratio_lgbm,
    fit_transition_ratio_lgbm,
)
from occupancy_ratio.fit_occupancy_ratio import (
    ActionRatioConfig,
    DiscountedOccupancyRatioModel,
    OccupancyRegressionConfig,
    TransitionRatioConfig,
    _clip_pseudo_outcomes,
    _damped_update,
    _make_fold_indices,
    _make_occupancy_sample_weights,
    _project_nonnegative_normalized,
    fit_discounted_occupancy_ratio,
    tune_discounted_occupancy_ratio_cv,
)
from occupancy_ratio.fit_occupancy_ratio_neural import (
    NeuralActionRatioConfig,
    NeuralOccupancyRegressionConfig,
    NeuralTransitionRatioConfig,
    fit_action_ratio_neural,
    fit_discounted_occupancy_ratio_neural,
    fit_transition_ratio_neural,
    tune_discounted_occupancy_ratio_neural_cv,
)
from occupancy_ratio_benchmark.external_baselines import preflight_google_dualdice
from occupancy_ratio_benchmark.discrete import exact_ratio_table, make_chain_mdp, make_discrete_dataset
from occupancy_ratio_benchmark.estimators import estimate_boosted_tree, estimate_neural_network, estimate_oracle
from occupancy_ratio_benchmark.gym_control import make_gym_control_dataset
from occupancy_ratio_benchmark.io import write_csv
from occupancy_ratio_benchmark import runner as runner_module
from occupancy_ratio_benchmark.runner import _expanded_estimators, make_winner_table, run_benchmark
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


def test_invalid_occupancy_config_raises() -> None:
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
    with pytest.raises(ValueError, match="profile"):
        OccupancyRatioBenchmarkConfig(profile="not-a-profile")
    with pytest.raises(ValueError, match="estimators"):
        OccupancyRatioBenchmarkConfig(estimators=("not_an_estimator",))
    with pytest.raises(ValueError, match="cv_scoring"):
        OccupancyRatioBenchmarkConfig(cv_scoring="bad")
    with pytest.raises(ValueError, match="google_learning_rates"):
        OccupancyRatioBenchmarkConfig(google_learning_rates=(0.0,))
    with pytest.raises(ValueError, match="boosted_density_ratio_loss"):
        OccupancyRatioBenchmarkConfig(boosted_density_ratio_loss="bad")
    with pytest.raises(ValueError, match="boosted_logistic_logit_clip"):
        OccupancyRatioBenchmarkConfig(boosted_logistic_logit_clip=0.0)
    with pytest.raises(ValueError, match="openml_task_ids"):
        OccupancyRatioBenchmarkConfig(openml_task_ids=(0,))
    with pytest.raises(ValueError, match="tabular_state_cap"):
        OccupancyRatioBenchmarkConfig(tabular_state_cap=1)


def test_benchmark_profile_and_estimator_expansion() -> None:
    config = OccupancyRatioBenchmarkConfig.for_profile(
        "medium",
        include_google_dual_dice=False,
    )
    assert config.profile == "medium"
    assert config.stage == "medium"
    assert config.gammas == (0.5, 0.9, 0.95, 0.99)
    assert config.linear_gaussian_policy_shifts == (0.25, 0.5, 1.0, 2.0)
    assert "openml_contextual_bandit" in config.settings
    assert "openml_finite_mdp" in config.settings
    assert not any("dualdice" in estimator for estimator in config.resolved_estimators())

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

    overnight = OccupancyRatioBenchmarkConfig.for_profile("overnight", include_google_dual_dice=False)
    assert "gym_halfcheetah" in overnight.settings
    assert "gym_hopper" in overnight.settings
    assert "minari_pointmaze" in overnight.settings
    assert "minari_minigrid" in overnight.settings
    assert "neural_network_google_parity" in overnight.estimators
    assert not any("dualdice" in estimator for estimator in overnight.resolved_estimators())


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


def test_public_package_exports_and_console_modules() -> None:
    import importlib
    import occupancy_ratio

    assert occupancy_ratio.NeuralDiscountedOccupancyRatioModel.__name__ == "NeuralDiscountedOccupancyRatioModel"
    assert occupancy_ratio.DiscountedOccupancyRatioNeuralModel is occupancy_ratio.NeuralDiscountedOccupancyRatioModel
    assert importlib.import_module("occupancy_ratio.boosted").fit_discounted_occupancy_ratio
    assert importlib.import_module("occupancy_ratio.neural").fit_discounted_occupancy_ratio_neural
    assert importlib.import_module("occupancy_ratio.nuisance").fit_importance_ratio_lgbm
    assert importlib.import_module("occupancy_ratio_benchmark.run").main
    assert importlib.import_module("occupancy_ratio_benchmark.dualdice_grid").main


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
    assert {row["estimator"] for row in result.rows} >= {
            "oracle",
            "boosted_tree_squared",
            "boosted_tree_huber",
            "neural_network_stable",
        }
    neural_rows = [row for row in result.rows if str(row["estimator"]).startswith("neural_network")]
    assert neural_rows
    assert neural_rows[0]["neural_stabilization_preset"] == "stable"
    assert neural_rows[0]["neural_normalize_transition_cache"] == 0.0


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


def test_defaults_report_selects_from_generated_csv(tmp_path) -> None:
    rows = [
        {
            "profile": "overnight",
            "setting": "gym_pendulum",
            "gamma": 0.9,
            "sample_size": 100,
            "seed": 0,
            "estimator": "google_dualdice_neural",
            "status": "ok",
            "ope_value_abs_error": 0.20,
            "effective_sample_size_fraction": 0.30,
        },
        {
            "profile": "overnight",
            "setting": "gym_pendulum",
            "gamma": 0.9,
            "sample_size": 100,
            "seed": 0,
            "estimator": "neural_network_stable",
            "status": "ok",
            "ope_value_abs_error": 0.18,
            "effective_sample_size_fraction": 0.25,
            "ratio_normalized_l1": "",
        },
    ]
    results_path = tmp_path / "results.csv"
    write_csv(results_path, rows)
    paths = generate_defaults_report(results_path)
    assert paths["report"].exists()
    text = paths["report"].read_text(encoding="utf-8")
    assert "Recommended default" in text
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
    assert result.rows[0]["profile"] == "smoke"
