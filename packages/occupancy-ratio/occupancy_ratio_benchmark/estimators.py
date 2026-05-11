from __future__ import annotations

from dataclasses import dataclass, field, replace
from itertools import product
import time
from typing import Any, Sequence

import numpy as np

from occupancy_ratio.calibration import calibrate_occupancy_bellman_binning
from occupancy_ratio.fit_occupancy_ratio import (
    ActionRatioConfig,
    OccupancyRegressionConfig,
    SourceStateRatioConfig,
    TransitionRatioConfig,
    fit_discounted_occupancy_ratio,
)
from occupancy_ratio.fit_occupancy_ratio_neural import (
    NeuralActionRatioConfig,
    NeuralOccupancyRegressionConfig,
    NeuralSourceStateRatioConfig,
    NeuralTransitionRatioConfig,
    fit_discounted_occupancy_ratio_neural,
)
from occupancy_ratio.tuning import (
    OccupancySearchSpace,
    OccupancyTuningConfig,
    OccupancyTuningResult,
    tune_occupancy_ratio_auto,
)
from occupancy_ratio_benchmark.config import OccupancyRatioBenchmarkConfig
from occupancy_ratio_benchmark.data import BenchmarkDataset
from occupancy_ratio_benchmark.diagnostics import estimator_diagnostics_optional
from occupancy_ratio_benchmark.external_baselines import (
    DICE_RL_BEST_REGULARIZED_FLAGS,
    DICE_RL_DUALDICE_RECOVERY_FLAGS,
    GoogleDICERLPreflight,
    GoogleDualDICEPreflight,
    estimate_google_dice_rl_neural,
    estimate_google_dualdice_neural,
    preflight_google_dice_rl,
)


Array = np.ndarray


@dataclass
class EstimatorResult:
    estimator: str
    status: str
    weights: Array | None
    raw_weights: Array | None
    runtime_sec: float
    diagnostics: dict[str, Any]
    skip_reason: str = ""
    tuning_rows: list[dict[str, Any]] = field(default_factory=list)


def estimate_oracle(dataset: BenchmarkDataset) -> EstimatorResult:
    start = time.perf_counter()
    if dataset.true_ratio is None:
        return EstimatorResult(
            estimator="oracle",
            status="skipped",
            weights=None,
            raw_weights=None,
            runtime_sec=time.perf_counter() - start,
            diagnostics={"ratio_truth_available": 0.0},
            skip_reason="Oracle ratio is unavailable for this benchmark setting.",
        )
    weights = np.asarray(dataset.true_ratio, dtype=np.float64).reshape(-1)
    return EstimatorResult(
        estimator="oracle",
        status="ok",
        weights=weights,
        raw_weights=weights,
        runtime_sec=time.perf_counter() - start,
        diagnostics=estimator_diagnostics_optional(
            true_ratio=dataset.true_ratio,
            estimated_ratio=weights,
            raw_ratio=weights,
            reference_weights=dataset.reference_weights,
            feature_matrix=_diagnostic_features(dataset),
        ),
    )


def _boosted_lgb_params(
    config: OccupancyRatioBenchmarkConfig,
    dataset: BenchmarkDataset,
    *,
    role: str,
) -> dict[str, Any]:
    """LightGBM defaults for off-the-shelf boosted OPE fits.

    Modern-control benchmark cells often use 1k-5k rows. A fixed
    ``min_data_in_leaf=100`` made the boosted ratios nearly uniform at 1k rows,
    so scale the leaf floor with available data while keeping larger runs
    regularized.
    """
    n_rows = max(1, int(dataset.states.shape[0]))
    min_leaf = max(20, min(100, n_rows // 20))
    params: dict[str, Any] = {
        "num_leaves": 31 if config.stage == "smoke" else 63,
        "min_data_in_leaf": int(min_leaf),
        "verbose": -1,
        "num_threads": 0,
    }
    if role == "occupancy":
        params["learning_rate"] = 0.08
    return params


def estimate_boosted_tree(
    dataset: BenchmarkDataset,
    config: OccupancyRatioBenchmarkConfig,
    *,
    loss: str,
    preset: str | None = None,
    initial_ratio_mode: str = "auto",
    one_step_ratio_mode: str = "auto",
) -> EstimatorResult:
    start = time.perf_counter()
    preset = str(loss if preset is None else preset)
    occupancy_options, nuisance_prediction_max = _boosted_stabilization_options(config, preset)
    nuisance_crossfit_folds, nuisance_moment_calibration = _boosted_nuisance_options(config, preset)
    density_ratio_loss = _boosted_density_ratio_loss(config, preset)
    effective_loss = str(occupancy_options.pop("loss"))
    occupancy_lgb_params = _boosted_lgb_params(config, dataset, role="occupancy")
    nuisance_lgb_params = _boosted_lgb_params(config, dataset, role="nuisance")
    occupancy_config = OccupancyRegressionConfig(
        num_iterations=int(config.boosted_num_iterations),
        trees_per_iteration=int(config.boosted_trees_per_iteration),
        mcmc_samples=int(config.boosted_mcmc_samples),
        batch_size=int(config.boosted_batch_size),
        loss=effective_loss,
        huber_delta=config.huber_delta,
        huber_delta_scale=float(config.huber_delta_scale),
        huber_hessian_floor=float(config.huber_hessian_floor),
        validation_fraction=0.20,
        patience=8 if config.stage == "smoke" else 12,
        seed=int(dataset.seed),
        show_progress=False,
        lgb_params=occupancy_lgb_params,
        **occupancy_options,
    )
    action_config = ActionRatioConfig(
        num_boost_round=30 if config.stage == "smoke" else 120,
        early_stopping_rounds=5,
        validation_fraction=0.20,
        show_progress=False,
        refit_on_all_data=True,
        lgb_params=nuisance_lgb_params,
        prediction_max=nuisance_prediction_max,
        crossfit_folds=int(nuisance_crossfit_folds),
        moment_calibration=str(nuisance_moment_calibration),
        density_ratio_loss=str(density_ratio_loss),
        logistic_logit_clip=config.boosted_logistic_logit_clip,
    )
    source_config = SourceStateRatioConfig(
        num_boost_round=30 if config.stage == "smoke" else 120,
        early_stopping_rounds=5,
        validation_fraction=0.20,
        show_progress=False,
        refit_on_all_data=True,
        lgb_params=nuisance_lgb_params,
        prediction_max=nuisance_prediction_max,
        crossfit_folds=int(nuisance_crossfit_folds),
        moment_calibration=str(nuisance_moment_calibration),
        density_ratio_loss=str(density_ratio_loss),
        logistic_logit_clip=config.boosted_logistic_logit_clip,
    )
    transition_config = TransitionRatioConfig(
        num_boost_round=40 if config.stage == "smoke" else 180,
        permutation_samples=5 if config.stage == "smoke" else 20,
        early_stopping_rounds=5,
        validation_fraction=0.20,
        show_progress=False,
        refit_on_all_data=True,
        lgb_params=nuisance_lgb_params,
        prediction_max=nuisance_prediction_max,
        crossfit_folds=int(nuisance_crossfit_folds),
        moment_calibration=str(nuisance_moment_calibration),
        density_ratio_loss=str(density_ratio_loss),
        logistic_logit_clip=config.boosted_logistic_logit_clip,
    )
    initial_states, initial_actions, initial_weights, source_applied = _initial_ratio_inputs(dataset, config)
    target_next_actions = _target_next_actions_input(dataset)
    tuning_rows: list[dict[str, Any]] = []
    if bool(config.tune_cv):
        tuned = tune_occupancy_ratio_auto(
            states=dataset.states,
            actions=dataset.actions,
            next_states=dataset.next_states,
            target_actions=dataset.target_actions,
            gamma=float(dataset.gamma),
            initial_states=initial_states,
            initial_actions=initial_actions,
            initial_weights=initial_weights,
            target_next_actions=target_next_actions,
            rewards=dataset.rewards,
            search_space=_boosted_tuning_search_space(
                config,
                preset,
                occupancy_config=occupancy_config,
                action_config=action_config,
                source_config=source_config,
                transition_config=transition_config,
                automl=bool(str(config.automl_tuning) in {"fast", "balanced"}),
                include_source=initial_states is not None,
            ),
            config=_benchmark_tuning_config(
                config,
                families=("boosted",),
                seed=int(dataset.seed + 60_001),
                candidate_count=_benchmark_candidate_count(config, family="boosted", preset=preset, include_source=initial_states is not None),
            ),
            initial_ratio_mode=initial_ratio_mode,
            one_step_ratio_mode=one_step_ratio_mode,
        )
        model = tuned.model
        tuning_rows = _flatten_boosted_tuning_rows(tuned, estimator=f"boosted_tree_{preset}", dataset=dataset)
    else:
        model = fit_discounted_occupancy_ratio(
            states=dataset.states,
            actions=dataset.actions,
            next_states=dataset.next_states,
            target_actions=dataset.target_actions,
            gamma=float(dataset.gamma),
            initial_states=initial_states,
            initial_actions=initial_actions,
            initial_weights=initial_weights,
            target_next_actions=target_next_actions,
            occupancy=occupancy_config,
            action_ratio=action_config,
            source_state_ratio=source_config,
            transition_ratio=transition_config,
            initial_ratio_mode=initial_ratio_mode,
            one_step_ratio_mode=one_step_ratio_mode,
        )
    raw = model.predict_state_action_ratio(dataset.states, dataset.actions, clip=False)
    weights = model.predict_state_action_ratio(dataset.states, dataset.actions, clip=True)
    bellman_diagnostics: dict[str, Any] = {}
    if preset == "bellman_moment_calibrated":
        weights, bellman_diagnostics = _calibrate_bellman_moment_weights(
            dataset,
            weights,
            w_max=config.boosted_occupancy_ratio_max,
        )
    diagnostics = estimator_diagnostics_optional(
        true_ratio=dataset.true_ratio,
        estimated_ratio=weights,
        raw_ratio=raw,
        reference_weights=dataset.reference_weights,
        feature_matrix=_diagnostic_features(dataset),
    )
    diagnostics.update(_value_diagnostics(dataset, weights))
    diagnostics.update(bellman_diagnostics)
    diagnostics.update(_first_stage_diagnostics(dataset, model.to_legacy_dict()))
    diagnostics.update(
        {
            "occupancy_loss": effective_loss,
            "occupancy_stabilization_preset": preset,
            "trees_used": float(model.diagnostics.get("trees_used") or 0),
            "stopped_early": float(bool(model.diagnostics.get("stopped_early"))),
            "refresh_count": float(model.diagnostics.get("refresh_count") or 0),
            "history_json_friendly": float(_history_json_friendly(model.history)),
            "nuisance_crossfit_folds": float(nuisance_crossfit_folds),
            "nuisance_moment_calibration": str(nuisance_moment_calibration),
            "nuisance_density_ratio_loss": str(density_ratio_loss),
            "action_density_ratio_loss": str(model.diagnostics.get("action_density_ratio_loss", "lsif")),
            "transition_density_ratio_loss": str(model.diagnostics.get("transition_density_ratio_loss", "lsif")),
            "logistic_logit_clip": _finite_or_blank(config.boosted_logistic_logit_clip),
            "action_prior_correction": float(model.diagnostics.get("action_prior_correction") or 1.0),
            "transition_prior_correction": float(model.diagnostics.get("transition_prior_correction") or 1.0),
            "source_state_correction_mode": str(config.source_state_correction_mode),
            "source_state_correction_applied": float(source_applied),
            "source_state_initial_rows": float(0 if initial_states is None else np.asarray(initial_states).shape[0]),
            "source_state_ratio_enabled": float(bool(model.diagnostics.get("source_state_ratio_enabled", False))),
            "source_state_ratio_mean": _finite_or_blank(model.diagnostics.get("source_state_ratio_mean")),
            "source_state_ratio_max": _finite_or_blank(model.diagnostics.get("source_state_ratio_max")),
            "source_state_ratio_ess_fraction": _finite_or_blank(
                model.diagnostics.get("source_state_ratio_ess_fraction")
            ),
            "source_state_ratio_loss": _finite_or_blank(model.diagnostics.get("source_state_ratio_loss")),
            "source_state_ratio_updates": _finite_or_blank(model.diagnostics.get("source_state_ratio_updates")),
            "source_state_ratio_clipped_fraction": _finite_or_blank(
                model.diagnostics.get("source_state_ratio_clipped_fraction")
            ),
            "source_state_ratio_query_clipped_fraction": _finite_or_blank(
                model.diagnostics.get("source_state_ratio_query_clipped_fraction")
            ),
            "source_state_ratio_prediction_max": _finite_or_blank(
                model.diagnostics.get("source_state_ratio_prediction_max")
            ),
            "source_state_ratio_prediction_scale": _finite_or_blank(
                model.diagnostics.get("source_state_ratio_prediction_scale")
            ),
            "source_state_ratio_density_ratio_loss": str(
                model.diagnostics.get("source_state_ratio_density_ratio_loss", "none")
            ),
            "initial_ratio_mode": str(model.diagnostics.get("initial_ratio_mode", "")),
            "one_step_ratio_mode": str(model.diagnostics.get("one_step_ratio_mode", "")),
            "initial_joint_ratio_enabled": float(bool(model.diagnostics.get("initial_joint_ratio_enabled", False))),
            "initial_joint_ratio_mean": _finite_or_blank(model.diagnostics.get("initial_joint_ratio_mean")),
            "initial_joint_ratio_max": _finite_or_blank(model.diagnostics.get("initial_joint_ratio_max")),
            "initial_joint_ratio_ess_fraction": _finite_or_blank(
                model.diagnostics.get("initial_joint_ratio_ess_fraction")
            ),
            "initial_joint_ratio_loss": _finite_or_blank(model.diagnostics.get("initial_joint_ratio_loss")),
            "initial_joint_ratio_updates": _finite_or_blank(model.diagnostics.get("initial_joint_ratio_updates")),
            "initial_joint_ratio_clipped_fraction": _finite_or_blank(
                model.diagnostics.get("initial_joint_ratio_clipped_fraction")
            ),
            "initial_joint_ratio_query_clipped_fraction": _finite_or_blank(
                model.diagnostics.get("initial_joint_ratio_query_clipped_fraction")
            ),
            "initial_joint_ratio_prediction_max": _finite_or_blank(
                model.diagnostics.get("initial_joint_ratio_prediction_max")
            ),
            "initial_joint_ratio_prediction_scale": _finite_or_blank(
                model.diagnostics.get("initial_joint_ratio_prediction_scale")
            ),
            "initial_joint_ratio_density_ratio_loss": str(
                model.diagnostics.get("initial_joint_ratio_density_ratio_loss", "none")
            ),
            "one_step_direct_ratio_enabled": float(
                bool(model.diagnostics.get("one_step_direct_ratio_enabled", False))
            ),
            "one_step_direct_ratio_mean": _finite_or_blank(model.diagnostics.get("one_step_direct_ratio_mean")),
            "one_step_direct_ratio_max": _finite_or_blank(model.diagnostics.get("one_step_direct_ratio_max")),
            "one_step_direct_ratio_ess_fraction": _finite_or_blank(
                model.diagnostics.get("one_step_direct_ratio_ess_fraction")
            ),
            "one_step_direct_ratio_loss": _finite_or_blank(model.diagnostics.get("one_step_direct_ratio_loss")),
            "one_step_direct_ratio_updates": _finite_or_blank(
                model.diagnostics.get("one_step_direct_ratio_updates")
            ),
            "one_step_direct_ratio_clipped_fraction": _finite_or_blank(
                model.diagnostics.get("one_step_direct_ratio_clipped_fraction")
            ),
            "one_step_direct_ratio_query_clipped_fraction": _finite_or_blank(
                model.diagnostics.get("one_step_direct_ratio_query_clipped_fraction")
            ),
            "one_step_direct_ratio_prediction_max": _finite_or_blank(
                model.diagnostics.get("one_step_direct_ratio_prediction_max")
            ),
            "one_step_direct_ratio_prediction_scale": _finite_or_blank(
                model.diagnostics.get("one_step_direct_ratio_prediction_scale")
            ),
            "one_step_direct_ratio_density_ratio_loss": str(
                model.diagnostics.get("one_step_direct_ratio_density_ratio_loss", "none")
            ),
            "tuned_cv": float(bool(config.tune_cv)),
        }
    )
    diagnostics.update(_occupancy_history_diagnostics(model.history))
    return EstimatorResult(
        estimator=f"boosted_tree_{preset}",
        status="ok",
        weights=weights,
        raw_weights=raw,
        runtime_sec=time.perf_counter() - start,
        diagnostics=diagnostics,
        tuning_rows=tuning_rows,
    )


def estimate_boosted_tree_auto(
    dataset: BenchmarkDataset,
    config: OccupancyRatioBenchmarkConfig,
) -> EstimatorResult:
    candidates = (
        "stable",
        "staged_cv",
        "stable_factored",
        "relaxed_tail",
        "transition_norm",
        "calibrated",
        "stable_logistic_nuisance",
    )
    results = [
        estimate_boosted_tree(
            dataset,
            config,
            loss="huber",
            preset=preset,
            initial_ratio_mode="factored" if preset == "stable_factored" else "auto",
            one_step_ratio_mode="factored" if preset == "stable_factored" else "auto",
        )
        for preset in candidates
    ]
    scored = [(_selection_score(result), result) for result in results]
    score, selected = min(scored, key=lambda item: item[0])
    diagnostics = dict(selected.diagnostics)
    diagnostics.update(
        {
            "selected_preset": str(selected.diagnostics.get("occupancy_stabilization_preset", selected.estimator)),
            "selection_score": float(score),
            "auto_candidate_count": float(len(candidates)),
        }
    )
    tuning_rows: list[dict[str, Any]] = []
    for candidate_score, result in scored:
        tuning_rows.extend(result.tuning_rows)
        tuning_rows.append(
            {
                "estimator": "boosted_tree_auto",
                "candidate_estimator": result.estimator,
                "score": float(candidate_score),
                "selected": float(result is selected),
            }
        )
    return EstimatorResult(
        estimator="boosted_tree_auto",
        status=selected.status,
        weights=selected.weights,
        raw_weights=selected.raw_weights,
        runtime_sec=sum(float(result.runtime_sec) for result in results),
        diagnostics=diagnostics,
        skip_reason=selected.skip_reason,
        tuning_rows=tuning_rows,
    )


def estimate_neural_network(
    dataset: BenchmarkDataset,
    config: OccupancyRatioBenchmarkConfig,
    *,
    loss: str,
    preset: str | None = None,
    initial_ratio_mode: str = "auto",
    one_step_ratio_mode: str = "auto",
) -> EstimatorResult:
    start = time.perf_counter()
    _stabilize_torch_runtime()
    preset = str(loss if preset is None else preset)
    occupancy_options, nuisance_prediction_max = _neural_stabilization_options(config, preset)
    nuisance_crossfit_folds, nuisance_moment_calibration, density_ratio_loss = _neural_nuisance_options(config, preset)
    effective_loss = str(occupancy_options.pop("loss"))
    hidden_dims = (256, 256) if preset == "google_parity" else tuple(int(width) for width in config.neural_hidden_dims)
    action_hidden_dims = _neural_stage_hidden_dims(config.neural_action_hidden_dims, fallback=hidden_dims)
    source_hidden_dims = _neural_stage_hidden_dims(config.neural_source_hidden_dims, fallback=hidden_dims)
    transition_hidden_dims = _neural_stage_hidden_dims(config.neural_transition_hidden_dims, fallback=hidden_dims)
    direct_one_step_hidden_dims = _neural_stage_hidden_dims(config.neural_direct_one_step_hidden_dims, fallback=source_hidden_dims)
    activation = "relu" if preset == "google_parity" else str(config.neural_activation)
    source_steps = _neural_source_steps(config)
    direct_one_step_steps = _neural_direct_one_step_steps(config)
    occupancy_config = NeuralOccupancyRegressionConfig(
        num_iterations=int(config.neural_num_iterations),
        gradient_steps_per_iteration=int(config.neural_gradient_steps_per_iteration),
        mcmc_samples=int(config.neural_mcmc_samples),
        batch_size=int(config.neural_batch_size),
        hidden_dims=hidden_dims,
        activation=activation,
        learning_rate=float(config.neural_learning_rate),
        weight_decay=float(config.neural_weight_decay),
        loss=effective_loss,
        huber_delta=config.huber_delta,
        huber_delta_scale=float(config.huber_delta_scale),
        validation_fraction=0.20,
        patience=8 if config.stage == "smoke" else 12,
        validation_warmup_iterations=int(config.neural_validation_warmup_iterations),
        seed=int(dataset.seed),
        grad_clip_norm=config.neural_grad_clip_norm,
        device=str(config.neural_device),
        show_progress=False,
        direct_one_step_density_ratio_loss=str(density_ratio_loss),
        direct_one_step_prediction_max=nuisance_prediction_max,
        direct_one_step_logistic_logit_clip=config.neural_logistic_logit_clip,
        direct_one_step_moment_calibration=str(nuisance_moment_calibration),
        direct_one_step_max_steps=direct_one_step_steps,
        direct_one_step_hidden_dims=direct_one_step_hidden_dims,
        direct_adjoint_steps=config.neural_direct_adjoint_steps,
        direct_adjoint_learning_rate=config.neural_direct_adjoint_learning_rate,
        direct_adjoint_weight_decay=config.neural_direct_adjoint_weight_decay,
        **occupancy_options,
    )
    action_config = NeuralActionRatioConfig(
        hidden_dims=action_hidden_dims,
        activation=activation,
        learning_rate=float(config.neural_nuisance_learning_rate),
        weight_decay=float(config.neural_weight_decay),
        batch_size=int(config.neural_batch_size),
        max_steps=int(config.neural_action_steps),
        validation_fraction=0.20,
        patience=8 if config.stage == "smoke" else 20,
        seed=int(dataset.seed + 7_001),
        prediction_max=nuisance_prediction_max,
        grad_clip_norm=config.neural_grad_clip_norm,
        moment_calibration=str(nuisance_moment_calibration),
        crossfit_folds=int(nuisance_crossfit_folds),
        crossfit_seed=int(dataset.seed + 17_001),
        density_ratio_loss=str(density_ratio_loss),
        logistic_logit_clip=config.neural_logistic_logit_clip,
        device=str(config.neural_device),
    )
    source_prediction_max = nuisance_prediction_max
    source_config = NeuralSourceStateRatioConfig(
        hidden_dims=source_hidden_dims,
        activation=activation,
        learning_rate=float(config.neural_nuisance_learning_rate),
        weight_decay=float(config.neural_weight_decay),
        batch_size=int(config.neural_batch_size),
        max_steps=source_steps,
        validation_fraction=0.20,
        patience=8 if config.stage == "smoke" else 20,
        seed=int(dataset.seed + 9_001),
        prediction_max=source_prediction_max,
        grad_clip_norm=config.neural_grad_clip_norm,
        moment_calibration=str(nuisance_moment_calibration),
        crossfit_folds=int(nuisance_crossfit_folds),
        crossfit_seed=int(dataset.seed + 19_001),
        density_ratio_loss=str(density_ratio_loss),
        logistic_logit_clip=config.neural_logistic_logit_clip,
        device=str(config.neural_device),
    )
    transition_config = NeuralTransitionRatioConfig(
        hidden_dims=transition_hidden_dims,
        activation=activation,
        learning_rate=float(config.neural_nuisance_learning_rate),
        weight_decay=float(config.neural_weight_decay),
        batch_size=int(config.neural_batch_size),
        max_steps=int(config.neural_transition_steps),
        permutation_samples=int(config.neural_transition_permutation_samples),
        validation_fraction=0.20,
        patience=8 if config.stage == "smoke" else 20,
        seed=int(dataset.seed + 8_001),
        prediction_max=nuisance_prediction_max,
        grad_clip_norm=config.neural_grad_clip_norm,
        moment_calibration=str(nuisance_moment_calibration),
        crossfit_folds=int(nuisance_crossfit_folds),
        crossfit_seed=int(dataset.seed + 18_001),
        density_ratio_loss=str(density_ratio_loss),
        logistic_logit_clip=config.neural_logistic_logit_clip,
        device=str(config.neural_device),
    )
    initial_states, initial_actions, initial_weights, source_applied = _initial_ratio_inputs(dataset, config)
    target_next_actions = _target_next_actions_input(dataset)
    tuning_rows: list[dict[str, Any]] = []
    if bool(config.tune_cv):
        tuned = tune_occupancy_ratio_auto(
            states=dataset.states,
            actions=dataset.actions,
            next_states=dataset.next_states,
            target_actions=dataset.target_actions,
            gamma=float(dataset.gamma),
            initial_states=initial_states,
            initial_actions=initial_actions,
            initial_weights=initial_weights,
            target_next_actions=target_next_actions,
            rewards=dataset.rewards,
            families=("neural",),
            search_space=_neural_tuning_search_space(
                config,
                preset,
                occupancy_config=occupancy_config,
                action_config=action_config,
                source_config=source_config,
                transition_config=transition_config,
                automl=bool(str(config.automl_tuning) in {"fast", "balanced"}),
                include_source=initial_states is not None,
            ),
            config=_benchmark_tuning_config(
                config,
                families=("neural",),
                seed=int(dataset.seed + 70_001),
                candidate_count=_benchmark_candidate_count(config, family="neural", preset=preset, include_source=initial_states is not None),
            ),
            initial_ratio_mode=initial_ratio_mode,
            one_step_ratio_mode=one_step_ratio_mode,
        )
        model = tuned.model
        tuning_rows = _flatten_neural_tuning_rows(tuned, estimator=f"neural_network_{preset}", dataset=dataset)
    else:
        model = fit_discounted_occupancy_ratio_neural(
            states=dataset.states,
            actions=dataset.actions,
            next_states=dataset.next_states,
            target_actions=dataset.target_actions,
            gamma=float(dataset.gamma),
            initial_states=initial_states,
            initial_actions=initial_actions,
            initial_weights=initial_weights,
            target_next_actions=target_next_actions,
            occupancy=occupancy_config,
            action_ratio=action_config,
            source_state_ratio=source_config,
            transition_ratio=transition_config,
            initial_ratio_mode=initial_ratio_mode,
            one_step_ratio_mode=one_step_ratio_mode,
        )
    raw = model.predict_state_action_ratio(dataset.states, dataset.actions, clip=False)
    weights = model.predict_state_action_ratio(dataset.states, dataset.actions, clip=True)
    bellman_diagnostics: dict[str, Any] = {}
    if preset == "bellman_moment_calibrated":
        weights, bellman_diagnostics = _calibrate_bellman_moment_weights(
            dataset,
            weights,
            w_max=config.neural_occupancy_ratio_max,
        )
    diagnostics = estimator_diagnostics_optional(
        true_ratio=dataset.true_ratio,
        estimated_ratio=weights,
        raw_ratio=raw,
        reference_weights=dataset.reference_weights,
        feature_matrix=_diagnostic_features(dataset),
    )
    diagnostics.update(_value_diagnostics(dataset, weights))
    diagnostics.update(bellman_diagnostics)
    diagnostics.update(_first_stage_diagnostics(dataset, model.to_legacy_dict()))
    diagnostics.update(
        {
            "occupancy_loss": effective_loss,
            "occupancy_stabilization_preset": preset,
            "neural_stabilization_preset": preset,
            "neural_loss": effective_loss,
            "neural_hidden_dims": "x".join(str(int(width)) for width in hidden_dims),
            "neural_action_hidden_dims": "x".join(str(int(width)) for width in action_hidden_dims),
            "neural_source_hidden_dims": "x".join(str(int(width)) for width in source_hidden_dims),
            "neural_transition_hidden_dims": "x".join(str(int(width)) for width in transition_hidden_dims),
            "neural_direct_one_step_hidden_dims": "x".join(str(int(width)) for width in direct_one_step_hidden_dims),
            "neural_activation": activation,
            "neural_gradient_steps_used": float(model.diagnostics.get("gradient_steps_used") or 0),
            "neural_accepted_count": float(model.diagnostics.get("accepted_count") or 0),
            "neural_validation_warmup_accepts": float(model.diagnostics.get("validation_warmup_accepts") or 0),
            "neural_validation_warmup_iterations": float(
                model.diagnostics.get("validation_warmup_iterations") or 0
            ),
            "neural_action_steps": float(config.neural_action_steps),
            "neural_action_updates": float(model.diagnostics.get("action_updates") or 0),
            "neural_source_steps": float(source_steps),
            "neural_transition_steps": float(config.neural_transition_steps),
            "neural_transition_updates": float(model.diagnostics.get("transition_updates") or 0),
            "neural_direct_one_step_steps": float(direct_one_step_steps),
            "neural_direct_adjoint_steps": _finite_or_blank(model.diagnostics.get("direct_adjoint_steps")),
            "neural_direct_adjoint_learning_rate": _finite_or_blank(
                model.diagnostics.get("direct_adjoint_learning_rate")
            ),
            "neural_action_best_valid_loss": _finite_or_blank(model.diagnostics.get("action_best_valid_loss")),
            "neural_transition_best_valid_loss": _finite_or_blank(
                model.diagnostics.get("transition_best_valid_loss")
            ),
            "neural_occupancy_best_valid_loss": _finite_or_blank(
                model.diagnostics.get("occupancy_best_valid_loss")
            ),
            "neural_occupancy_final_valid_loss": _finite_or_blank(
                model.diagnostics.get("occupancy_final_valid_loss")
            ),
            "neural_action_prediction_scale": float(model.diagnostics.get("action_prediction_scale") or 1.0),
            "neural_transition_prediction_scale": float(model.diagnostics.get("transition_prediction_scale") or 1.0),
            "neural_action_crossfit_folds": float(model.diagnostics.get("action_crossfit_folds") or 1.0),
            "neural_transition_crossfit_folds": float(model.diagnostics.get("transition_crossfit_folds") or 1.0),
            "neural_moment_calibration": str(nuisance_moment_calibration),
            "neural_density_ratio_loss": str(density_ratio_loss),
            "neural_action_density_ratio_loss": str(model.diagnostics.get("action_density_ratio_loss", "lsif")),
            "neural_transition_density_ratio_loss": str(model.diagnostics.get("transition_density_ratio_loss", "lsif")),
            "neural_logistic_logit_clip": _finite_or_blank(config.neural_logistic_logit_clip),
            "neural_action_prior_correction": float(model.diagnostics.get("action_prior_correction") or 1.0),
            "neural_transition_prior_correction": float(model.diagnostics.get("transition_prior_correction") or 1.0),
            "neural_normalize_transition_cache": float(
                bool(getattr(occupancy_config, "normalize_transition_cache", False))
            ),
            "source_state_correction_mode": str(config.source_state_correction_mode),
            "source_state_correction_applied": float(source_applied),
            "source_state_initial_rows": float(0 if initial_states is None else np.asarray(initial_states).shape[0]),
            "source_state_ratio_enabled": float(bool(model.diagnostics.get("source_state_ratio_enabled", False))),
            "source_state_ratio_mean": _finite_or_blank(model.diagnostics.get("source_state_ratio_mean")),
            "source_state_ratio_max": _finite_or_blank(model.diagnostics.get("source_state_ratio_max")),
            "source_state_ratio_ess_fraction": _finite_or_blank(
                model.diagnostics.get("source_state_ratio_ess_fraction")
            ),
            "source_state_ratio_loss": _finite_or_blank(model.diagnostics.get("source_state_ratio_loss")),
            "source_state_ratio_updates": _finite_or_blank(model.diagnostics.get("source_state_ratio_updates")),
            "source_state_ratio_clipped_fraction": _finite_or_blank(
                model.diagnostics.get("source_state_ratio_clipped_fraction")
            ),
            "source_state_ratio_query_clipped_fraction": _finite_or_blank(
                model.diagnostics.get("source_state_ratio_query_clipped_fraction")
            ),
            "source_state_ratio_prediction_max": _finite_or_blank(
                model.diagnostics.get("source_state_ratio_prediction_max")
            ),
            "source_state_ratio_prediction_scale": _finite_or_blank(
                model.diagnostics.get("source_state_ratio_prediction_scale")
            ),
            "source_state_ratio_density_ratio_loss": str(
                model.diagnostics.get("source_state_ratio_density_ratio_loss", "none")
            ),
            "initial_ratio_mode": str(model.diagnostics.get("initial_ratio_mode", "")),
            "one_step_ratio_mode": str(model.diagnostics.get("one_step_ratio_mode", "")),
            "initial_joint_ratio_enabled": float(bool(model.diagnostics.get("initial_joint_ratio_enabled", False))),
            "initial_joint_ratio_mean": _finite_or_blank(model.diagnostics.get("initial_joint_ratio_mean")),
            "initial_joint_ratio_max": _finite_or_blank(model.diagnostics.get("initial_joint_ratio_max")),
            "initial_joint_ratio_ess_fraction": _finite_or_blank(
                model.diagnostics.get("initial_joint_ratio_ess_fraction")
            ),
            "initial_joint_ratio_loss": _finite_or_blank(model.diagnostics.get("initial_joint_ratio_loss")),
            "initial_joint_ratio_updates": _finite_or_blank(model.diagnostics.get("initial_joint_ratio_updates")),
            "initial_joint_ratio_clipped_fraction": _finite_or_blank(
                model.diagnostics.get("initial_joint_ratio_clipped_fraction")
            ),
            "initial_joint_ratio_query_clipped_fraction": _finite_or_blank(
                model.diagnostics.get("initial_joint_ratio_query_clipped_fraction")
            ),
            "initial_joint_ratio_prediction_max": _finite_or_blank(
                model.diagnostics.get("initial_joint_ratio_prediction_max")
            ),
            "initial_joint_ratio_prediction_scale": _finite_or_blank(
                model.diagnostics.get("initial_joint_ratio_prediction_scale")
            ),
            "initial_joint_ratio_density_ratio_loss": str(
                model.diagnostics.get("initial_joint_ratio_density_ratio_loss", "none")
            ),
            "one_step_direct_ratio_enabled": float(
                bool(model.diagnostics.get("one_step_direct_ratio_enabled", False))
            ),
            "one_step_direct_ratio_mean": _finite_or_blank(model.diagnostics.get("one_step_direct_ratio_mean")),
            "one_step_direct_ratio_max": _finite_or_blank(model.diagnostics.get("one_step_direct_ratio_max")),
            "one_step_direct_ratio_ess_fraction": _finite_or_blank(
                model.diagnostics.get("one_step_direct_ratio_ess_fraction")
            ),
            "one_step_direct_ratio_loss": _finite_or_blank(model.diagnostics.get("one_step_direct_ratio_loss")),
            "one_step_direct_ratio_updates": _finite_or_blank(
                model.diagnostics.get("one_step_direct_ratio_updates")
            ),
            "one_step_direct_ratio_clipped_fraction": _finite_or_blank(
                model.diagnostics.get("one_step_direct_ratio_clipped_fraction")
            ),
            "one_step_direct_ratio_query_clipped_fraction": _finite_or_blank(
                model.diagnostics.get("one_step_direct_ratio_query_clipped_fraction")
            ),
            "one_step_direct_ratio_prediction_max": _finite_or_blank(
                model.diagnostics.get("one_step_direct_ratio_prediction_max")
            ),
            "one_step_direct_ratio_prediction_scale": _finite_or_blank(
                model.diagnostics.get("one_step_direct_ratio_prediction_scale")
            ),
            "one_step_direct_ratio_density_ratio_loss": str(
                model.diagnostics.get("one_step_direct_ratio_density_ratio_loss", "none")
            ),
            "stopped_early": float(bool(model.diagnostics.get("stopped_early"))),
            "refresh_count": float(model.diagnostics.get("refresh_count") or 0),
            "history_json_friendly": float(_history_json_friendly(model.history)),
            "tuned_cv": float(bool(config.tune_cv)),
        }
    )
    diagnostics.update(_occupancy_history_diagnostics(model.history))
    return EstimatorResult(
        estimator=f"neural_network_{preset}",
        status="ok",
        weights=weights,
        raw_weights=raw,
        runtime_sec=time.perf_counter() - start,
        diagnostics=diagnostics,
        tuning_rows=tuning_rows,
    )


def estimate_neural_network_auto(
    dataset: BenchmarkDataset,
    config: OccupancyRatioBenchmarkConfig,
) -> EstimatorResult:
    candidates = ("stable", "stable_factored", "relaxed_tail", "calibrated", "stable_logistic_nuisance")
    results = [
        estimate_neural_network(
            dataset,
            config,
            loss="huber",
            preset=preset,
            initial_ratio_mode="factored" if preset == "stable_factored" else "auto",
            one_step_ratio_mode="factored" if preset == "stable_factored" else "auto",
        )
        for preset in candidates
    ]
    scored = [(_selection_score(result), result) for result in results]
    score, selected = min(scored, key=lambda item: item[0])
    diagnostics = dict(selected.diagnostics)
    diagnostics.update(
        {
            "selected_preset": str(selected.diagnostics.get("neural_stabilization_preset", selected.estimator)),
            "selection_score": float(score),
            "auto_candidate_count": float(len(candidates)),
        }
    )
    tuning_rows: list[dict[str, Any]] = []
    for candidate_score, result in scored:
        tuning_rows.extend(result.tuning_rows)
        tuning_rows.append(
            {
                "estimator": "neural_network_auto",
                "candidate_estimator": result.estimator,
                "score": float(candidate_score),
                "selected": float(result is selected),
            }
        )
    return EstimatorResult(
        estimator="neural_network_auto",
        status=selected.status,
        weights=selected.weights,
        raw_weights=selected.raw_weights,
        runtime_sec=sum(float(result.runtime_sec) for result in results),
        diagnostics=diagnostics,
        skip_reason=selected.skip_reason,
        tuning_rows=tuning_rows,
    )


def _stabilize_torch_runtime() -> None:
    """Avoid macOS/OpenMP threadpool crashes during benchmark neural fits."""
    try:
        import torch

        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
    except Exception:
        pass


def estimate_google_dualdice(
    dataset: BenchmarkDataset,
    config: OccupancyRatioBenchmarkConfig,
    preflight: GoogleDualDICEPreflight,
) -> EstimatorResult:
    result = estimate_google_dualdice_neural(
        dataset,
        preflight=preflight,
        num_updates=int(config.google_num_updates),
        batch_size=int(config.google_batch_size),
        diagnostic_features=_diagnostic_features(dataset),
        value_diagnostics={},
    )
    diagnostics = dict(result["diagnostics"])
    if result["weights"] is not None:
        diagnostics.update(_value_diagnostics(dataset, np.asarray(result["weights"], dtype=np.float64)))
    return EstimatorResult(
        estimator=str(result["estimator"]),
        status=str(result["status"]),
        weights=result["weights"],
        raw_weights=result["raw_weights"],
        runtime_sec=float(result["runtime_sec"]),
        diagnostics=diagnostics,
        skip_reason=str(result["skip_reason"]),
    )


def estimate_google_dualdice_default(
    dataset: BenchmarkDataset,
    config: OccupancyRatioBenchmarkConfig,
    preflight: GoogleDualDICEPreflight,
) -> EstimatorResult:
    """Official Google DualDICE with fixed documented defaults."""
    tuned_config = replace(config, google_num_updates=1_000, google_batch_size=128)
    result = estimate_google_dualdice(dataset, tuned_config, preflight)
    return _rename_result(result, "google_dualdice_default")


def estimate_google_dualdice_published_default(
    dataset: BenchmarkDataset,
    config: OccupancyRatioBenchmarkConfig,
    preflight: GoogleDualDICEPreflight,
) -> EstimatorResult:
    """Explicit alias for the published Google Research policy_eval DualDICE."""
    return _rename_result(estimate_google_dualdice_default(dataset, config, preflight), "google_dualdice_published_default")


def estimate_dice_rl_dualdice_recovered(
    dataset: BenchmarkDataset,
    config: OccupancyRatioBenchmarkConfig,
    preflight: GoogleDICERLPreflight,
) -> EstimatorResult:
    """Google DICE-RL exact DualDICE recovery form from the README flags."""
    result = _estimate_dice_rl_candidate(
        dataset,
        config,
        preflight,
        {
            "num_steps": int(config.dice_rl_num_steps),
            "batch_size": int(config.dice_rl_batch_size),
            "learning_rate": float(config.dice_rl_learning_rate),
            "hidden_dims": tuple(int(width) for width in config.dice_rl_hidden_dims),
            "flags": dict(DICE_RL_DUALDICE_RECOVERY_FLAGS),
            "label_prefix": "dice_rl_dualdice_recovered_candidate",
        },
    )
    return _rename_result(result, "dice_rl_dualdice_recovered")


def estimate_dice_rl_best_regularized(
    dataset: BenchmarkDataset,
    config: OccupancyRatioBenchmarkConfig,
    preflight: GoogleDICERLPreflight,
) -> EstimatorResult:
    """Google DICE-RL README best regularized DICE-family comparator."""
    result = _estimate_dice_rl_candidate(
        dataset,
        config,
        preflight,
        {
            "num_steps": int(config.dice_rl_num_steps),
            "batch_size": int(config.dice_rl_batch_size),
            "learning_rate": float(config.dice_rl_learning_rate),
            "hidden_dims": tuple(int(width) for width in config.dice_rl_hidden_dims),
            "flags": dict(DICE_RL_BEST_REGULARIZED_FLAGS),
            "label_prefix": "dice_rl_best_regularized_candidate",
        },
    )
    return _rename_result(result, "dice_rl_best_regularized")


def estimate_dice_rl_dualdice_gmm_tuned(
    dataset: BenchmarkDataset,
    config: OccupancyRatioBenchmarkConfig,
    preflight: GoogleDICERLPreflight,
) -> EstimatorResult:
    """Tune the DICE-RL DualDICE recovery form using non-oracle Bellman-GMM moments."""
    results = [_estimate_dice_rl_candidate(dataset, config, preflight, candidate) for candidate in _dice_rl_dualdice_candidates(config)]
    if not results:
        return _skipped_diagnostic_oracle("dice_rl_dualdice_gmm_tuned", "No DICE-RL candidates were configured.")
    scored = [(_dualdice_gmm_score(dataset, result, config), result) for result in results]
    score, selected_original = min(scored, key=lambda item: item[0])
    selected = _rename_result(selected_original, "dice_rl_dualdice_gmm_tuned")
    diagnostics = dict(selected.diagnostics)
    diagnostics.update(
        {
            "gmm_tuned": 1.0,
            "selected_candidate": str(selected_original.estimator),
            "selection_score": float(score),
            "dice_rl_candidate_count": float(len(results)),
        }
    )
    return EstimatorResult(
        estimator="dice_rl_dualdice_gmm_tuned",
        status=selected.status,
        weights=selected.weights,
        raw_weights=selected.raw_weights,
        runtime_sec=sum(float(result.runtime_sec) for _, result in scored),
        diagnostics=diagnostics,
        skip_reason=selected.skip_reason,
        tuning_rows=_selector_tuning_rows("dice_rl_dualdice_gmm_tuned", dataset, scored, selected_result=selected_original, scoring="bellman_gmm"),
    )


def estimate_dice_rl_dualdice_oracle(
    dataset: BenchmarkDataset,
    config: OccupancyRatioBenchmarkConfig,
    preflight: GoogleDICERLPreflight,
) -> EstimatorResult:
    """Diagnostic oracle envelope over DICE-RL DualDICE recovery candidates."""
    if dataset.true_ratio is None:
        return _skipped_diagnostic_oracle("dice_rl_dualdice_oracle", "Oracle ratio is unavailable for this setting.")
    results = [_estimate_dice_rl_candidate(dataset, config, preflight, candidate) for candidate in _dice_rl_dualdice_candidates(config)]
    scored = [(_oracle_ratio_score(dataset, result), result) for result in results]
    score, selected_original = min(scored, key=lambda item: item[0])
    selected = _rename_result(selected_original, "dice_rl_dualdice_oracle")
    diagnostics = dict(selected.diagnostics)
    diagnostics.update(
        {
            "oracle_tuned": 1.0,
            "selected_candidate": str(selected_original.estimator),
            "selection_score": float(score),
            "dice_rl_candidate_count": float(len(results)),
        }
    )
    return EstimatorResult(
        estimator="dice_rl_dualdice_oracle",
        status=selected.status,
        weights=selected.weights,
        raw_weights=selected.raw_weights,
        runtime_sec=sum(float(result.runtime_sec) for _, result in scored),
        diagnostics=diagnostics,
        skip_reason=selected.skip_reason,
        tuning_rows=_selector_tuning_rows("dice_rl_dualdice_oracle", dataset, scored, selected_result=selected_original, scoring="oracle_ratio_l1"),
    )


def estimate_neural_fori_cv_size(dataset: BenchmarkDataset, config: OccupancyRatioBenchmarkConfig) -> EstimatorResult:
    """FORI neural size/model selection with non-oracle Bellman-GMM scoring."""
    tuned_config = replace(
        config,
        tune_cv=True,
        automl_tuning="balanced",
        cv_score_method="bellman_gmm",
        cv_gmm_objective="ratio",
        cv_moment_extra_blocks=("second_order", "multiscale_rff", "support", "policy_shift"),
    )
    result = estimate_neural_network(dataset, tuned_config, loss="huber", preset="stable")
    return _rename_result(result, "neural_fori_cv_size")


def estimate_neural_fori_oracle(dataset: BenchmarkDataset, config: OccupancyRatioBenchmarkConfig) -> EstimatorResult:
    """Diagnostic upper envelope over fixed FORI neural candidates using ratio truth."""
    if dataset.true_ratio is None:
        return _skipped_diagnostic_oracle("neural_fori_oracle", "Oracle ratio is unavailable for this setting.")
    candidate_presets = ("stable", "stable_logistic_nuisance", "relaxed_tail", "google_parity")
    results = [
        estimate_neural_network(dataset, config, loss="huber", preset=preset)
        for preset in candidate_presets
    ]
    scored = [(_oracle_ratio_score(dataset, result), result) for result in results]
    score, selected_original = min(scored, key=lambda item: item[0])
    selected = _rename_result(selected_original, "neural_fori_oracle")
    diagnostics = dict(selected.diagnostics)
    diagnostics.update(
        {
            "oracle_tuned": 1.0,
            "selected_candidate": str(selected_original.estimator),
            "selection_score": float(score),
            "oracle_candidate_count": float(len(candidate_presets)),
        }
    )
    tuning_rows = _selector_tuning_rows(
        "neural_fori_oracle",
        dataset,
        [(score_value, result) for score_value, result in scored],
        selected_result=selected_original,
        scoring="oracle_ratio_l1",
    )
    return EstimatorResult(
        estimator="neural_fori_oracle",
        status=selected.status,
        weights=selected.weights,
        raw_weights=selected.raw_weights,
        runtime_sec=sum(float(result.runtime_sec) for _, result in scored),
        diagnostics=diagnostics,
        skip_reason=selected.skip_reason,
        tuning_rows=tuning_rows,
    )


def estimate_dualdice_gmm_tuned(
    dataset: BenchmarkDataset,
    config: OccupancyRatioBenchmarkConfig,
    preflight: GoogleDualDICEPreflight,
) -> EstimatorResult:
    """Tune DualDICE over standard knobs using non-oracle Bellman-GMM moments."""
    results = [_estimate_dualdice_candidate(dataset, config, preflight, candidate) for candidate in _dualdice_candidates(config)]
    if not results:
        return _skipped_diagnostic_oracle("dualdice_gmm_tuned", "No DualDICE candidates were configured.")
    scored = [(_dualdice_gmm_score(dataset, result, config), result) for result in results]
    score, selected_original = min(scored, key=lambda item: item[0])
    selected = _rename_result(selected_original, "dualdice_gmm_tuned")
    diagnostics = dict(selected.diagnostics)
    diagnostics.update(
        {
            "gmm_tuned": 1.0,
            "selected_candidate": str(selected_original.estimator),
            "selection_score": float(score),
            "dualdice_candidate_count": float(len(results)),
        }
    )
    return EstimatorResult(
        estimator="dualdice_gmm_tuned",
        status=selected.status,
        weights=selected.weights,
        raw_weights=selected.raw_weights,
        runtime_sec=sum(float(result.runtime_sec) for _, result in scored),
        diagnostics=diagnostics,
        skip_reason=selected.skip_reason,
        tuning_rows=_selector_tuning_rows("dualdice_gmm_tuned", dataset, scored, selected_result=selected_original, scoring="bellman_gmm"),
    )


def estimate_dualdice_oracle(
    dataset: BenchmarkDataset,
    config: OccupancyRatioBenchmarkConfig,
    preflight: GoogleDualDICEPreflight,
) -> EstimatorResult:
    """Diagnostic upper envelope over DualDICE candidates using ratio truth."""
    if dataset.true_ratio is None:
        return _skipped_diagnostic_oracle("dualdice_oracle", "Oracle ratio is unavailable for this setting.")
    results = [_estimate_dualdice_candidate(dataset, config, preflight, candidate) for candidate in _dualdice_candidates(config)]
    scored = [(_oracle_ratio_score(dataset, result), result) for result in results]
    score, selected_original = min(scored, key=lambda item: item[0])
    selected = _rename_result(selected_original, "dualdice_oracle")
    diagnostics = dict(selected.diagnostics)
    diagnostics.update(
        {
            "oracle_tuned": 1.0,
            "selected_candidate": str(selected_original.estimator),
            "selection_score": float(score),
            "dualdice_candidate_count": float(len(results)),
        }
    )
    return EstimatorResult(
        estimator="dualdice_oracle",
        status=selected.status,
        weights=selected.weights,
        raw_weights=selected.raw_weights,
        runtime_sec=sum(float(result.runtime_sec) for _, result in scored),
        diagnostics=diagnostics,
        skip_reason=selected.skip_reason,
        tuning_rows=_selector_tuning_rows("dualdice_oracle", dataset, scored, selected_result=selected_original, scoring="oracle_ratio_l1"),
    )


def _rename_result(result: EstimatorResult, estimator: str) -> EstimatorResult:
    diagnostics = dict(result.diagnostics)
    diagnostics["base_estimator"] = result.estimator
    return EstimatorResult(
        estimator=estimator,
        status=result.status,
        weights=result.weights,
        raw_weights=result.raw_weights,
        runtime_sec=result.runtime_sec,
        diagnostics=diagnostics,
        skip_reason=result.skip_reason,
        tuning_rows=list(result.tuning_rows),
    )


def _skipped_diagnostic_oracle(estimator: str, reason: str) -> EstimatorResult:
    return EstimatorResult(
        estimator=estimator,
        status="skipped",
        weights=None,
        raw_weights=None,
        runtime_sec=0.0,
        diagnostics={"oracle_tuned": float("oracle" in estimator)},
        skip_reason=reason,
    )


def _dualdice_candidates(config: OccupancyRatioBenchmarkConfig) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for updates, batch_size, zeta_lr, weight_decay, hidden_width in product(
        tuple(int(value) for value in config.google_update_grid),
        tuple(int(value) for value in config.google_batch_sizes),
        tuple(float(value) for value in config.google_learning_rates),
        tuple(float(value) for value in config.google_weight_decays),
        tuple(int(value) for value in config.google_hidden_dims),
    ):
        rows.append(
            {
                "num_updates": int(updates),
                "batch_size": int(batch_size),
                "zeta_learning_rate": float(zeta_lr),
                "nu_learning_rate": float(max(zeta_lr / 10.0, 1e-6)),
                "weight_decay": float(weight_decay),
                "hidden_dims": (int(hidden_width), int(hidden_width)),
            }
        )
    return rows


def _estimate_dualdice_candidate(
    dataset: BenchmarkDataset,
    config: OccupancyRatioBenchmarkConfig,
    preflight: GoogleDualDICEPreflight,
    candidate: dict[str, Any],
) -> EstimatorResult:
    label = (
        "google_dualdice_candidate_"
        f"u{int(candidate['num_updates'])}_"
        f"b{int(candidate['batch_size'])}_"
        f"lr{float(candidate['zeta_learning_rate']):.0e}_"
        f"wd{float(candidate['weight_decay']):.0e}_"
        f"h{'x'.join(str(width) for width in candidate['hidden_dims'])}"
    )
    result = estimate_google_dualdice_neural(
        dataset,
        preflight=preflight,
        num_updates=int(candidate["num_updates"]),
        batch_size=int(candidate["batch_size"]),
        weight_decay=float(candidate["weight_decay"]),
        nu_learning_rate=float(candidate["nu_learning_rate"]),
        zeta_learning_rate=float(candidate["zeta_learning_rate"]),
        hidden_dims=tuple(int(width) for width in candidate["hidden_dims"]),
        estimator_name=label,
        diagnostic_features=_diagnostic_features(dataset),
        value_diagnostics={},
    )
    diagnostics = dict(result["diagnostics"])
    if result["weights"] is not None:
        diagnostics.update(_value_diagnostics(dataset, np.asarray(result["weights"], dtype=np.float64)))
    diagnostics.update({f"candidate_{key}": value for key, value in candidate.items() if key != "hidden_dims"})
    diagnostics["candidate_hidden_dims"] = "x".join(str(width) for width in candidate["hidden_dims"])
    return EstimatorResult(
        estimator=str(result["estimator"]),
        status=str(result["status"]),
        weights=result["weights"],
        raw_weights=result["raw_weights"],
        runtime_sec=float(result["runtime_sec"]),
        diagnostics=diagnostics,
        skip_reason=str(result["skip_reason"]),
    )


def _dice_rl_dualdice_candidates(config: OccupancyRatioBenchmarkConfig) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for num_steps, batch_size, learning_rate, hidden_width in product(
        tuple(int(value) for value in config.dice_rl_update_grid),
        tuple(int(value) for value in config.dice_rl_batch_sizes),
        tuple(float(value) for value in config.dice_rl_learning_rates),
        tuple(int(value) for value in config.dice_rl_hidden_dim_grid),
    ):
        rows.append(
            {
                "num_steps": int(num_steps),
                "batch_size": int(batch_size),
                "learning_rate": float(learning_rate),
                "hidden_dims": (int(hidden_width), int(hidden_width)),
                "flags": dict(DICE_RL_DUALDICE_RECOVERY_FLAGS),
                "label_prefix": "dice_rl_dualdice_candidate",
            }
        )
    return rows


def _estimate_dice_rl_candidate(
    dataset: BenchmarkDataset,
    config: OccupancyRatioBenchmarkConfig,
    preflight: GoogleDICERLPreflight,
    candidate: dict[str, Any],
) -> EstimatorResult:
    _ = config
    flags = dict(candidate["flags"])
    label = (
        f"{candidate['label_prefix']}_"
        f"u{int(candidate['num_steps'])}_"
        f"b{int(candidate['batch_size'])}_"
        f"lr{float(candidate['learning_rate']):.0e}_"
        f"h{'x'.join(str(width) for width in candidate['hidden_dims'])}"
    )
    result = estimate_google_dice_rl_neural(
        dataset,
        preflight=preflight,
        num_steps=int(candidate["num_steps"]),
        batch_size=int(candidate["batch_size"]),
        learning_rate=float(candidate["learning_rate"]),
        hidden_dims=tuple(int(width) for width in candidate["hidden_dims"]),
        flags=flags,
        estimator_name=label,
        diagnostic_features=_diagnostic_features(dataset),
        value_diagnostics={},
    )
    diagnostics = dict(result["diagnostics"])
    if result["weights"] is not None:
        diagnostics.update(_value_diagnostics(dataset, np.asarray(result["weights"], dtype=np.float64)))
    diagnostics.update(
        {
            "candidate_num_steps": int(candidate["num_steps"]),
            "candidate_batch_size": int(candidate["batch_size"]),
            "candidate_learning_rate": float(candidate["learning_rate"]),
            "candidate_hidden_dims": "x".join(str(width) for width in candidate["hidden_dims"]),
        }
    )
    for key, value in flags.items():
        diagnostics[f"candidate_{key}"] = float(value)
    return EstimatorResult(
        estimator=str(result["estimator"]),
        status=str(result["status"]),
        weights=result["weights"],
        raw_weights=result["raw_weights"],
        runtime_sec=float(result["runtime_sec"]),
        diagnostics=diagnostics,
        skip_reason=str(result["skip_reason"]),
    )


def _dualdice_gmm_score(
    dataset: BenchmarkDataset,
    result: EstimatorResult,
    config: OccupancyRatioBenchmarkConfig,
) -> float:
    if result.status != "ok" or result.weights is None:
        return float("inf")
    weights = np.asarray(result.weights, dtype=np.float64).reshape(-1)
    h, h_next, init_moments = _bellman_moment_test_arrays(dataset)
    if weights.shape[0] != h.shape[0]:
        return float("inf")
    moments = weights[:, None] * (h - float(dataset.gamma) * h_next)
    residual = np.mean(moments, axis=0) - (1.0 - float(dataset.gamma)) * init_moments
    centered = moments - np.mean(moments, axis=0, keepdims=True)
    cov = centered.T @ centered / max(centered.shape[0] - 1, 1)
    ridge = float(config.cv_gmm_cov_ridge) * np.eye(cov.shape[0], dtype=np.float64)
    try:
        score = float(residual @ np.linalg.solve(cov + ridge + 1e-8 * np.eye(cov.shape[0]), residual))
    except np.linalg.LinAlgError:
        score = float(np.linalg.norm(residual))
    return score + _stability_penalty(result.diagnostics)


def _oracle_ratio_score(dataset: BenchmarkDataset, result: EstimatorResult) -> float:
    if result.status != "ok" or result.weights is None or dataset.true_ratio is None:
        return float("inf")
    truth = np.asarray(dataset.true_ratio, dtype=np.float64).reshape(-1)
    weights = np.asarray(result.weights, dtype=np.float64).reshape(-1)
    if truth.shape != weights.shape:
        return float("inf")
    return float(np.mean(np.abs(weights - truth)) / max(float(np.mean(np.abs(truth))), 1e-12))


def _selector_tuning_rows(
    estimator: str,
    dataset: BenchmarkDataset,
    scored: list[tuple[float, EstimatorResult]],
    *,
    selected_result: EstimatorResult,
    scoring: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, (score, result) in enumerate(scored):
        row = {
            **_tuning_context(estimator, dataset),
            "tuning_stage": "selector_candidate",
            "candidate_index": int(index),
            "candidate_estimator": result.estimator,
            "score": float(score),
            "selected": float(result is selected_result),
            "scoring": scoring,
            "status": result.status,
            "skip_reason": result.skip_reason,
            "runtime_sec": float(result.runtime_sec),
        }
        for key in (
            "google_num_updates",
            "google_batch_size",
            "google_weight_decay",
            "google_nu_learning_rate",
            "google_zeta_learning_rate",
            "google_hidden_dims",
            "dice_rl_num_steps",
            "dice_rl_batch_size",
            "dice_rl_learning_rate",
            "dice_rl_hidden_dims",
            "dice_rl_exact_dualdice_recovery",
            "dice_rl_best_regularized_form",
        ):
            if key in result.diagnostics:
                row[key] = result.diagnostics[key]
        rows.append(row)
    return rows


def _boosted_stabilization_options(
    config: OccupancyRatioBenchmarkConfig,
    preset: str,
) -> tuple[dict[str, Any], float | None]:
    """Map benchmark presets to estimator knobs without changing the public API."""
    if preset == "squared":
        return (
            dict(
                loss="squared",
                fixed_point_damping=1.0,
                normalize_occupancy=False,
                occupancy_ratio_max=None,
                clip_pseudo_outcomes=False,
                occupancy_sample_weight_mode="uniform",
            ),
            None,
        )
    if preset in {"huber", "logistic_nuisance"}:
        return (
            dict(
                loss="huber",
                fixed_point_damping=1.0,
                normalize_occupancy=False,
                occupancy_ratio_max=None,
                clip_pseudo_outcomes=False,
                occupancy_sample_weight_mode="uniform",
            ),
            None,
        )
    if preset == "huber_projection":
        return (
            dict(
                loss="huber",
                fixed_point_damping=1.0,
                normalize_occupancy=True,
                occupancy_ratio_max=config.boosted_occupancy_ratio_max,
                clip_pseudo_outcomes=True,
                pseudo_outcome_upper_quantile=float(config.boosted_pseudo_outcome_upper_quantile),
                occupancy_sample_weight_mode="uniform",
                occupancy_sample_weight_max=config.boosted_sample_weight_max,
            ),
            config.boosted_nuisance_prediction_max,
        )
    if preset == "relaxed_tail":
        return (
            dict(
                loss="huber",
                fixed_point_damping=0.75,
                normalize_occupancy=True,
                occupancy_ratio_max=100.0,
                clip_pseudo_outcomes=True,
                pseudo_outcome_upper_quantile=0.999,
                occupancy_sample_weight_mode="uniform",
                occupancy_sample_weight_max=config.boosted_sample_weight_max,
                normalize_transition_cache=False,
            ),
            100.0,
        )
    if preset in {
        "stable",
        "staged_cv",
        "stable_factored",
        "transition_norm",
        "crossfit2",
        "calibrated",
        "crossfit2_calibrated",
        "bellman_moment_calibrated",
        "stable_logistic_nuisance",
        "huber_projection_damping",
        "huber_projection_damping_transition_norm",
        "huber_projection_damping_weighted",
    }:
        weight_mode = "uniform" if preset != "huber_projection_damping_weighted" else str(config.boosted_sample_weight_mode)
        transition_norm = preset in {"transition_norm", "huber_projection_damping_transition_norm"}
        return (
            dict(
                loss="huber",
                fixed_point_damping=float(config.boosted_fixed_point_damping),
                normalize_occupancy=True,
                occupancy_ratio_max=config.boosted_occupancy_ratio_max,
                clip_pseudo_outcomes=True,
                pseudo_outcome_upper_quantile=float(config.boosted_pseudo_outcome_upper_quantile),
                occupancy_sample_weight_mode=weight_mode,
                occupancy_sample_weight_max=config.boosted_sample_weight_max,
                normalize_transition_cache=bool(
                    config.boosted_normalize_transition_cache
                    or transition_norm
                ),
            ),
            config.boosted_nuisance_prediction_max,
        )
    raise ValueError(f"Unknown boosted-tree stabilization preset '{preset}'.")


def _neural_stabilization_options(
    config: OccupancyRatioBenchmarkConfig,
    preset: str,
) -> tuple[dict[str, Any], float | None]:
    if preset == "squared":
        return (
            dict(
                loss="squared",
                fixed_point_damping=1.0,
                normalize_occupancy=False,
                occupancy_ratio_max=None,
                clip_pseudo_outcomes=False,
                occupancy_sample_weight_mode="uniform",
            ),
            None,
        )
    if preset in {"huber", "logistic_nuisance"}:
        return (
            dict(
                loss="huber",
                fixed_point_damping=1.0,
                normalize_occupancy=False,
                occupancy_ratio_max=None,
                clip_pseudo_outcomes=False,
                occupancy_sample_weight_mode="uniform",
            ),
            None,
        )
    if preset == "huber_projection":
        return (
            dict(
                loss="huber",
                fixed_point_damping=1.0,
                normalize_occupancy=True,
                occupancy_ratio_max=config.neural_occupancy_ratio_max,
                clip_pseudo_outcomes=True,
                pseudo_outcome_upper_quantile=float(config.neural_pseudo_outcome_upper_quantile),
                occupancy_sample_weight_mode="uniform",
                occupancy_sample_weight_max=config.neural_sample_weight_max,
            ),
            config.neural_nuisance_prediction_max,
        )
    if preset == "relaxed_tail":
        return (
            dict(
                loss="huber",
                fixed_point_damping=0.75,
                normalize_occupancy=True,
                occupancy_ratio_max=100.0,
                clip_pseudo_outcomes=True,
                pseudo_outcome_upper_quantile=0.999,
                occupancy_sample_weight_mode="uniform",
                occupancy_sample_weight_max=config.neural_sample_weight_max,
                normalize_transition_cache=False,
            ),
            100.0,
        )
    if preset in {
        "stable",
        "stable_factored",
        "transition_norm",
        "crossfit2",
        "calibrated",
        "crossfit2_calibrated",
        "bellman_moment_calibrated",
        "stable_logistic_nuisance",
        "google_parity",
        "huber_projection_damping",
        "huber_projection_damping_transition_norm",
        "huber_projection_damping_weighted",
    }:
        weight_mode = "uniform" if preset != "huber_projection_damping_weighted" else str(config.neural_sample_weight_mode)
        transition_norm = preset in {"transition_norm", "huber_projection_damping_transition_norm"}
        return (
            dict(
                loss="huber",
                fixed_point_damping=float(config.neural_fixed_point_damping),
                normalize_occupancy=True,
                occupancy_ratio_max=config.neural_occupancy_ratio_max,
                clip_pseudo_outcomes=True,
                pseudo_outcome_upper_quantile=float(config.neural_pseudo_outcome_upper_quantile),
                occupancy_sample_weight_mode=weight_mode,
                occupancy_sample_weight_max=config.neural_sample_weight_max,
                normalize_transition_cache=bool(transition_norm),
            ),
            config.neural_nuisance_prediction_max,
        )
    raise ValueError(f"Unknown neural-network stabilization preset '{preset}'.")


def _boosted_nuisance_options(config: OccupancyRatioBenchmarkConfig, preset: str) -> tuple[int, str]:
    folds = int(config.boosted_crossfit_folds)
    calibration = str(config.boosted_moment_calibration)
    if preset in {"crossfit2", "crossfit2_calibrated"}:
        folds = max(2, folds)
    if preset in {"calibrated", "crossfit2_calibrated"}:
        calibration = "scalar"
    return folds, calibration


def _boosted_density_ratio_loss(config: OccupancyRatioBenchmarkConfig, preset: str) -> str:
    loss = str(config.boosted_density_ratio_loss)
    if preset in {"logistic_nuisance", "stable_logistic_nuisance"}:
        loss = "logistic"
    if loss not in {"lsif", "logistic"}:
        raise ValueError("boosted_density_ratio_loss must be 'lsif' or 'logistic'.")
    return loss


def _neural_nuisance_options(config: OccupancyRatioBenchmarkConfig, preset: str) -> tuple[int, str, str]:
    folds = int(config.neural_crossfit_folds)
    calibration = str(config.neural_moment_calibration)
    density_ratio_loss = str(config.neural_density_ratio_loss)
    if preset in {"crossfit2", "crossfit2_calibrated"}:
        folds = max(2, folds)
    if preset in {"calibrated", "crossfit2_calibrated"}:
        calibration = "scalar"
    if preset in {"logistic_nuisance", "stable_logistic_nuisance"}:
        density_ratio_loss = "logistic"
    return folds, calibration, density_ratio_loss


def _neural_source_steps(config: OccupancyRatioBenchmarkConfig) -> int:
    value = config.neural_source_steps
    return int(config.neural_action_steps if value is None else value)


def _neural_direct_one_step_steps(config: OccupancyRatioBenchmarkConfig) -> int:
    value = config.neural_direct_one_step_steps
    return int(config.neural_transition_steps if value is None else value)


def _neural_stage_hidden_dims(value: Sequence[int] | None, *, fallback: Sequence[int]) -> tuple[int, ...]:
    return tuple(int(width) for width in (fallback if value is None else value))


def _nuisance_cv_grid(config: OccupancyRatioBenchmarkConfig) -> tuple[dict[str, Any], ...]:
    grid = []
    for cap in config.cv_nuisance_prediction_max_values:
        for calibration in config.cv_moment_calibrations:
            grid.append(
                {
                    "prediction_max": cap,
                    "moment_calibration": str(calibration),
                }
            )
    return tuple(grid) or ({},)


def _boosted_occupancy_cv_grid(
    config: OccupancyRatioBenchmarkConfig,
    preset: str,
) -> tuple[dict[str, Any], ...]:
    transition_values = (preset == "transition_norm",)
    if preset not in {"transition_norm", "huber_projection_damping_transition_norm"}:
        transition_values = (False, True)
    return tuple(
        {
            "fixed_point_damping": float(damping),
            "occupancy_ratio_max": cap,
            "normalize_transition_cache": bool(norm),
        }
        for damping in config.cv_fixed_point_dampings
        for cap in config.cv_occupancy_ratio_max_values
        for norm in transition_values
    )


def _neural_occupancy_cv_grid(
    config: OccupancyRatioBenchmarkConfig,
    preset: str,
) -> tuple[dict[str, Any], ...]:
    return _boosted_occupancy_cv_grid(config, preset)


def _benchmark_tuning_config(
    config: OccupancyRatioBenchmarkConfig,
    *,
    families: tuple[str, ...],
    seed: int,
    candidate_count: int,
) -> OccupancyTuningConfig:
    budget = str(config.automl_tuning)
    automl = budget in {"fast", "balanced"}
    max_candidates = (8 if budget == "fast" else 16) if automl else max(1, int(candidate_count))
    promotion_candidates = (3 if budget == "fast" else 4) if automl else max(1, min(int(candidate_count), 4))
    default_tuning = OccupancyTuningConfig()
    moment_extra_blocks = tuple(str(block) for block in config.cv_moment_extra_blocks) or tuple(default_tuning.moment_extra_blocks)
    return OccupancyTuningConfig(
        families=families,
        cv_folds=int(config.cv_folds),
        seed=int(seed),
        budget=budget if automl else "balanced",
        max_candidates=max_candidates,
        promotion_candidates=promotion_candidates,
        refit=True,
        score_method=str(config.cv_score_method),
        gmm_objective=str(config.cv_gmm_objective),
        gmm_cov_ridge=float(config.cv_gmm_cov_ridge),
        gmm_complexity_weight=float(config.cv_gmm_complexity_weight),
        gmm_ope_broad_weight=float(config.cv_gmm_ope_broad_weight),
        gmm_refit_fraction=float(config.cv_gmm_refit_fraction),
        stable_fallback=str(config.cv_score_method) != "validation_loss",
        staged_bootstrap_cv=bool(config.staged_bootstrap_cv),
        staged_cv_iterations=int(config.staged_cv_iterations),
        staged_cv_n_bootstrap=int(config.staged_cv_n_bootstrap),
        moment_extra_blocks=moment_extra_blocks,
        moment_multiscale_rff_scales=tuple(float(scale) for scale in config.cv_moment_multiscale_rff_scales),
        moment_strata_quantiles=tuple(float(quantile) for quantile in config.cv_moment_strata_quantiles),
    )


def _boosted_tuning_search_space(
    config: OccupancyRatioBenchmarkConfig,
    preset: str,
    *,
    occupancy_config: OccupancyRegressionConfig,
    action_config: ActionRatioConfig,
    source_config: SourceStateRatioConfig,
    transition_config: TransitionRatioConfig,
    automl: bool,
    include_source: bool,
) -> OccupancySearchSpace:
    return OccupancySearchSpace(
        boosted_occupancy=occupancy_config,
        boosted_action_ratio=action_config,
        boosted_source_state_ratio=source_config,
        boosted_transition_ratio=transition_config,
        boosted_candidates=None if automl else _benchmark_candidate_grid(config, preset, include_source=include_source),
    )


def _neural_tuning_search_space(
    config: OccupancyRatioBenchmarkConfig,
    preset: str,
    *,
    occupancy_config: NeuralOccupancyRegressionConfig,
    action_config: NeuralActionRatioConfig,
    source_config: NeuralSourceStateRatioConfig,
    transition_config: NeuralTransitionRatioConfig,
    automl: bool,
    include_source: bool,
) -> OccupancySearchSpace:
    return OccupancySearchSpace(
        neural_occupancy=occupancy_config,
        neural_action_ratio=action_config,
        neural_source_state_ratio=source_config,
        neural_transition_ratio=transition_config,
        neural_candidates=None if automl else _benchmark_candidate_grid(config, preset, include_source=include_source),
    )


def _benchmark_candidate_count(
    config: OccupancyRatioBenchmarkConfig,
    *,
    family: str,
    preset: str,
    include_source: bool,
) -> int:
    if str(config.automl_tuning) == "fast":
        return 8
    if str(config.automl_tuning) == "balanced":
        return 16
    return len(_benchmark_candidate_grid(config, preset, include_source=include_source))


def _benchmark_candidate_grid(
    config: OccupancyRatioBenchmarkConfig,
    preset: str,
    *,
    include_source: bool,
) -> tuple[dict[str, dict[str, Any]], ...]:
    rows = []
    nuisance_grid = _nuisance_cv_grid(config)
    for occ_over, action_over, transition_over in product(
        _boosted_occupancy_cv_grid(config, preset),
        nuisance_grid,
        nuisance_grid,
    ):
        candidate = {
            "occupancy": dict(occ_over),
            "action_ratio": dict(action_over),
            "transition_ratio": dict(transition_over),
        }
        if include_source:
            candidate["source_state_ratio"] = dict(action_over)
        rows.append(candidate)
    return tuple(rows)


def _flatten_boosted_tuning_rows(
    tuned: dict[str, Any],
    *,
    estimator: str,
    dataset: BenchmarkDataset,
) -> list[dict[str, Any]]:
    if isinstance(tuned, OccupancyTuningResult):
        return _flatten_product_tuning_rows(tuned, estimator=estimator, dataset=dataset)
    rows: list[dict[str, Any]] = []
    selected = tuned.get("selected_indices", {})
    for stage_name, scores_key in (
        ("action_ratio", "action_scores"),
        ("transition_ratio", "transition_scores"),
        ("occupancy", "occupancy_scores"),
    ):
        for idx, score_row in enumerate(tuned.get(scores_key, [])):
            rows.append(
                {
                    **_tuning_context(estimator, dataset),
                    "tuning_stage": stage_name,
                    "candidate_index": int(idx),
                    "score": float(score_row.get("score", float("nan"))),
                    "selected": float(int(selected.get(stage_name, -1)) == idx),
                    "scoring": str(tuned.get("scoring", "")),
                }
            )
    return rows


def _flatten_neural_tuning_rows(
    tuned: dict[str, Any],
    *,
    estimator: str,
    dataset: BenchmarkDataset,
) -> list[dict[str, Any]]:
    if isinstance(tuned, OccupancyTuningResult):
        return _flatten_product_tuning_rows(tuned, estimator=estimator, dataset=dataset)
    rows: list[dict[str, Any]] = []
    for idx, row in enumerate(tuned.get("cv_rows", [])):
        rows.append(
            {
                **_tuning_context(estimator, dataset),
                "tuning_stage": "joint_neural",
                "candidate_index": int(idx),
                "fold": int(row.get("fold", -1)),
                "score": float(row.get("score", float("nan"))),
                "selected": "",
                "scoring": "validation_history_loss",
            }
        )
    return rows


def _flatten_product_tuning_rows(
    tuned: OccupancyTuningResult,
    *,
    estimator: str,
    dataset: BenchmarkDataset,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    selected_id = str(tuned.selected_candidate_id)
    scoring_label = str(getattr(tuned.config, "score_method", "legacy_rank"))
    if scoring_label == "bellman_gmm":
        scoring_label = f"bellman_gmm_{getattr(tuned.config, 'gmm_objective', 'ratio')}"
    for row in tuned.candidate_rows():
        rows.append(
            {
                **_tuning_context(estimator, dataset),
                "tuning_stage": "automl_candidate",
                "candidate_id": row.get("candidate_id", ""),
                "candidate_label": row.get("candidate_label", ""),
                "family": row.get("family", ""),
                "budget_stage": row.get("budget_stage", ""),
                "candidate_index": _candidate_index(row.get("candidate_id", "")),
                "score": row.get("score", float("nan")),
                "selected": float(str(row.get("candidate_id", "")) == selected_id and row.get("budget_stage") == "full"),
                "promoted": row.get("promoted", ""),
                "runtime_sec": row.get("runtime_sec", ""),
                "scoring": scoring_label,
                **{key: value for key, value in row.items() if str(key).startswith("metric_")},
            }
        )
    for row in tuned.fold_rows():
        rows.append(
            {
                **_tuning_context(estimator, dataset),
                "tuning_stage": "automl_fold",
                "candidate_id": row.get("candidate_id", ""),
                "candidate_label": row.get("candidate_label", ""),
                "family": row.get("family", ""),
                "budget_stage": row.get("budget_stage", ""),
                "candidate_index": _candidate_index(row.get("candidate_id", "")),
                "fold": row.get("fold", ""),
                "score": "",
                "selected": float(str(row.get("candidate_id", "")) == selected_id and row.get("budget_stage") == "full"),
                "promoted": "",
                "runtime_sec": row.get("runtime_sec", ""),
                "scoring": scoring_label,
                "metric_validation_loss": row.get("validation_loss", ""),
                "metric_norm_error": row.get("norm_error", ""),
                "metric_ess_fraction": row.get("ess_fraction", ""),
                "metric_p99": row.get("p99", ""),
                "metric_max_weight": row.get("max_weight", ""),
                "metric_clipped_fraction": row.get("clipped_fraction", ""),
                "metric_reward_value": row.get("reward_value", ""),
                "metric_moment_balance": row.get("moment_balance", ""),
                "metric_moment_balance_targeted": row.get("moment_balance_targeted", ""),
                "metric_moment_balance_broad": row.get("moment_balance_broad", ""),
                "metric_moment_balance_mass": row.get("moment_balance_mass", ""),
                "metric_moment_balance_reward": row.get("moment_balance_reward", ""),
                "metric_moment_balance_value": row.get("moment_balance_value", ""),
                "metric_moment_balance_value_strata": row.get("moment_balance_value_strata", ""),
                "metric_moment_balance_geometry": row.get("moment_balance_geometry", ""),
                "metric_moment_balance_rff": row.get("moment_balance_rff", ""),
                "metric_moment_balance_rff_multiscale": row.get("moment_balance_rff_multiscale", ""),
                "metric_selection_risk": row.get("selection_risk", ""),
                "metric_selection_risk_raw": row.get("selection_risk_raw", ""),
                "metric_selection_effective_dim": row.get("selection_effective_dim", ""),
                "metric_selection_complexity_penalty": row.get("selection_complexity_penalty", ""),
            }
        )
    for row in tuned.first_stage_candidate_rows():
        rows.append(
            {
                **_tuning_context(estimator, dataset),
                "tuning_stage": "automl_first_stage_candidate",
                "candidate_id": row.get("candidate_id", ""),
                "family": row.get("family", ""),
                "ratio_task": row.get("task", ""),
                "ratio_mode": row.get("mode", ""),
                "score": row.get("score", float("nan")),
                "selected": row.get("selected", ""),
                "runtime_sec": row.get("runtime_sec", ""),
                "update_count_mean": row.get("update_count_mean", ""),
                "scoring": "density_ratio_cv_loss",
                "error": row.get("error", ""),
            }
        )
    for row in tuned.first_stage_fold_rows():
        rows.append(
            {
                **_tuning_context(estimator, dataset),
                "tuning_stage": "automl_first_stage_fold",
                "candidate_id": row.get("candidate_id", ""),
                "family": row.get("family", ""),
                "ratio_task": row.get("task", ""),
                "ratio_mode": row.get("mode", ""),
                "fold": row.get("fold", ""),
                "score": row.get("score", ""),
                "selected": "",
                "runtime_sec": "",
                "update_count": row.get("update_count", ""),
                "scoring": "density_ratio_cv_loss",
            }
        )
    for row in tuned.first_stage_skipped_rows():
        rows.append(
            {
                **_tuning_context(estimator, dataset),
                "tuning_stage": "automl_first_stage_skipped",
                "family": row.get("family", ""),
                "ratio_task": row.get("task", ""),
                "ratio_mode": row.get("mode", ""),
                "skip_reason": row.get("reason", ""),
                "selected": "",
                "scoring": "density_ratio_cv_loss",
            }
        )
    for row in tuned.staged_cv_candidate_rows():
        rows.append(
            {
                **_tuning_context(estimator, dataset),
                "tuning_stage": "automl_staged_cv_candidate",
                "candidate_id": row.get("candidate_id", ""),
                "candidate_label": row.get("candidate_label", ""),
                "family": row.get("family", ""),
                "budget_stage": row.get("budget_stage", ""),
                "stage": row.get("stage", ""),
                "candidate_index": _candidate_index(row.get("candidate_id", "")),
                "score": row.get("loss_mean", float("nan")),
                "selected": row.get("selected", row.get("selected_min_loss", "")),
                "selected_min_loss": row.get("selected_min_loss", ""),
                "kept": row.get("kept", ""),
                "active": row.get("active", row.get("kept", "")),
                "pruned": row.get("pruned", ""),
                "baseline_forced_eval": row.get("baseline_forced_eval", ""),
                "stage_loss": row.get("stage_loss", row.get("loss_mean", "")),
                "stage_loss_se": row.get("stage_loss_se", row.get("loss_se", "")),
                "metric_staged_cv_loss": row.get("loss_mean", ""),
                "metric_staged_cv_loss_se": row.get("loss_se", ""),
                "metric_staged_cv_threshold": row.get("threshold", ""),
                "staged_cv_n_bootstrap": row.get("bootstrap_iterations", ""),
                "scoring": "staged_bootstrapped_loss",
                "reason": row.get("reason", ""),
            }
        )
    for row in tuned.staged_cv_fold_rows():
        rows.append(
            {
                **_tuning_context(estimator, dataset),
                "tuning_stage": "automl_staged_cv_fold",
                "candidate_id": row.get("candidate_id", ""),
                "family": row.get("family", ""),
                "budget_stage": row.get("budget_stage", ""),
                "stage": row.get("stage", ""),
                "candidate_index": _candidate_index(row.get("candidate_id", "")),
                "fold": row.get("fold", ""),
                "score": row.get("loss", ""),
                "stage_loss": row.get("stage_loss", row.get("loss", "")),
                "stage_loss_se": row.get("stage_loss_se", ""),
                "active": row.get("active", ""),
                "pruned": row.get("pruned", ""),
                "selected": row.get("selected", ""),
                "baseline_forced_eval": row.get("baseline_forced_eval", ""),
                "scoring": "staged_bootstrapped_loss",
                "staged_loss_metric": row.get("metric", ""),
            }
        )
    return rows


def _candidate_index(candidate_id: Any) -> int:
    try:
        return int(str(candidate_id).rsplit("_", 1)[-1])
    except (TypeError, ValueError):
        return -1


def _tuning_context(estimator: str, dataset: BenchmarkDataset) -> dict[str, Any]:
    out = {
        "setting": dataset.setting,
        "estimator": estimator,
        "gamma": float(dataset.gamma),
        "seed": int(dataset.seed),
        "sample_size": int(dataset.sample_size),
    }
    out.update(dataset.metadata)
    return out


def run_estimator(
    estimator: str,
    dataset: BenchmarkDataset,
    config: OccupancyRatioBenchmarkConfig,
    google_preflight: GoogleDualDICEPreflight,
    dice_rl_preflight: GoogleDICERLPreflight | None = None,
) -> EstimatorResult:
    if estimator == "oracle":
        return estimate_oracle(dataset)
    if estimator == "neural_fori_default_stable":
        return _rename_result(estimate_neural_network(dataset, config, loss="huber", preset="stable"), estimator)
    if estimator == "neural_fori_cv_size":
        return estimate_neural_fori_cv_size(dataset, config)
    if estimator == "neural_fori_oracle":
        return estimate_neural_fori_oracle(dataset, config)
    if estimator == "google_dualdice_default":
        return estimate_google_dualdice_default(dataset, config, google_preflight)
    if estimator == "google_dualdice_published_default":
        return estimate_google_dualdice_published_default(dataset, config, google_preflight)
    if estimator == "dualdice_gmm_tuned":
        return estimate_dualdice_gmm_tuned(dataset, config, google_preflight)
    if estimator == "dualdice_oracle":
        return estimate_dualdice_oracle(dataset, config, google_preflight)
    if estimator == "dice_rl_dualdice_recovered":
        return estimate_dice_rl_dualdice_recovered(dataset, config, _dice_rl_preflight_or_default(config, dice_rl_preflight))
    if estimator == "dice_rl_dualdice_gmm_tuned":
        return estimate_dice_rl_dualdice_gmm_tuned(dataset, config, _dice_rl_preflight_or_default(config, dice_rl_preflight))
    if estimator == "dice_rl_dualdice_oracle":
        return estimate_dice_rl_dualdice_oracle(dataset, config, _dice_rl_preflight_or_default(config, dice_rl_preflight))
    if estimator == "dice_rl_best_regularized":
        return estimate_dice_rl_best_regularized(dataset, config, _dice_rl_preflight_or_default(config, dice_rl_preflight))
    if estimator == "boosted_tree":
        return estimate_boosted_tree(dataset, config, loss="huber", preset="stable")
    if estimator == "boosted_tree_auto":
        return estimate_boosted_tree_auto(dataset, config)
    if estimator == "boosted_tree_squared":
        return estimate_boosted_tree(dataset, config, loss="squared")
    if estimator == "boosted_tree_huber":
        return estimate_boosted_tree(dataset, config, loss="huber")
    if estimator == "boosted_tree_stable":
        return estimate_boosted_tree(dataset, config, loss="huber", preset="stable")
    if estimator == "boosted_tree_staged_cv":
        staged_config = replace(config, tune_cv=True, automl_tuning="balanced", staged_bootstrap_cv=True)
        return _rename_result(estimate_boosted_tree(dataset, staged_config, loss="huber", preset="stable"), estimator)
    if estimator == "boosted_tree_stable_factored":
        return estimate_boosted_tree(
            dataset,
            config,
            loss="huber",
            preset="stable_factored",
            initial_ratio_mode="factored",
            one_step_ratio_mode="factored",
        )
    if estimator == "boosted_tree_relaxed_tail":
        return estimate_boosted_tree(dataset, config, loss="huber", preset="relaxed_tail")
    if estimator == "boosted_tree_logistic_nuisance":
        return estimate_boosted_tree(dataset, config, loss="huber", preset="logistic_nuisance")
    if estimator == "boosted_tree_stable_logistic_nuisance":
        return estimate_boosted_tree(dataset, config, loss="huber", preset="stable_logistic_nuisance")
    if estimator == "boosted_tree_transition_norm":
        return estimate_boosted_tree(dataset, config, loss="huber", preset="transition_norm")
    if estimator == "boosted_tree_crossfit2":
        return estimate_boosted_tree(dataset, config, loss="huber", preset="crossfit2")
    if estimator == "boosted_tree_calibrated":
        return estimate_boosted_tree(dataset, config, loss="huber", preset="calibrated")
    if estimator == "boosted_tree_crossfit2_calibrated":
        return estimate_boosted_tree(dataset, config, loss="huber", preset="crossfit2_calibrated")
    if estimator == "boosted_tree_bellman_moment_calibrated":
        return estimate_boosted_tree(dataset, config, loss="huber", preset="bellman_moment_calibrated")
    if estimator == "boosted_tree_huber_projection":
        return estimate_boosted_tree(dataset, config, loss="huber", preset="huber_projection")
    if estimator == "boosted_tree_huber_projection_damping":
        return estimate_boosted_tree(dataset, config, loss="huber", preset="huber_projection_damping")
    if estimator == "boosted_tree_huber_projection_damping_transition_norm":
        return estimate_boosted_tree(
            dataset,
            config,
            loss="huber",
            preset="huber_projection_damping_transition_norm",
        )
    if estimator == "boosted_tree_huber_projection_damping_weighted":
        return estimate_boosted_tree(dataset, config, loss="huber", preset="huber_projection_damping_weighted")
    if estimator == "neural_network":
        return estimate_neural_network(dataset, config, loss="huber", preset="stable")
    if estimator == "neural_network_auto":
        return estimate_neural_network_auto(dataset, config)
    if estimator == "neural_network_squared":
        return estimate_neural_network(dataset, config, loss="squared")
    if estimator == "neural_network_huber":
        return estimate_neural_network(dataset, config, loss="huber")
    if estimator == "neural_network_stable":
        return estimate_neural_network(dataset, config, loss="huber", preset="stable")
    if estimator == "neural_network_staged_cv":
        staged_config = replace(
            config,
            tune_cv=True,
            automl_tuning=str(config.automl_tuning) if str(config.automl_tuning) != "off" else "balanced",
            staged_bootstrap_cv=True,
        )
        return _rename_result(estimate_neural_network(dataset, staged_config, loss="huber", preset="stable"), estimator)
    if estimator == "neural_network_naive_final_bellman_cv":
        naive_config = replace(
            config,
            tune_cv=True,
            automl_tuning=str(config.automl_tuning) if str(config.automl_tuning) != "off" else "balanced",
            staged_bootstrap_cv=False,
            cv_score_method="validation_loss",
        )
        return _rename_result(estimate_neural_network(dataset, naive_config, loss="huber", preset="stable"), estimator)
    if estimator == "neural_network_stable_factored":
        return estimate_neural_network(
            dataset,
            config,
            loss="huber",
            preset="stable_factored",
            initial_ratio_mode="factored",
            one_step_ratio_mode="factored",
        )
    if estimator == "neural_network_relaxed_tail":
        return estimate_neural_network(dataset, config, loss="huber", preset="relaxed_tail")
    if estimator == "neural_network_transition_norm":
        return estimate_neural_network(dataset, config, loss="huber", preset="transition_norm")
    if estimator == "neural_network_crossfit2":
        return estimate_neural_network(dataset, config, loss="huber", preset="crossfit2")
    if estimator == "neural_network_calibrated":
        return estimate_neural_network(dataset, config, loss="huber", preset="calibrated")
    if estimator == "neural_network_crossfit2_calibrated":
        return estimate_neural_network(dataset, config, loss="huber", preset="crossfit2_calibrated")
    if estimator == "neural_network_bellman_moment_calibrated":
        return estimate_neural_network(dataset, config, loss="huber", preset="bellman_moment_calibrated")
    if estimator == "neural_network_logistic_nuisance":
        return estimate_neural_network(dataset, config, loss="huber", preset="logistic_nuisance")
    if estimator == "neural_network_stable_logistic_nuisance":
        return estimate_neural_network(dataset, config, loss="huber", preset="stable_logistic_nuisance")
    if estimator == "neural_network_google_parity":
        return estimate_neural_network(dataset, config, loss="huber", preset="google_parity")
    if estimator == "neural_network_huber_projection":
        return estimate_neural_network(dataset, config, loss="huber", preset="huber_projection")
    if estimator == "neural_network_huber_projection_damping":
        return estimate_neural_network(dataset, config, loss="huber", preset="huber_projection_damping")
    if estimator == "neural_network_huber_projection_damping_transition_norm":
        return estimate_neural_network(
            dataset,
            config,
            loss="huber",
            preset="huber_projection_damping_transition_norm",
        )
    if estimator == "neural_network_huber_projection_damping_weighted":
        return estimate_neural_network(dataset, config, loss="huber", preset="huber_projection_damping_weighted")
    if estimator in {"google_dualdice", "google_dualdice_neural"}:
        return estimate_google_dualdice(dataset, config, google_preflight)
    if estimator == "google_tabular_dualdice_gridwalk":
        return EstimatorResult(
            estimator="google_tabular_dualdice_gridwalk",
            status="skipped",
            weights=None,
            raw_weights=None,
            runtime_sec=0.0,
            diagnostics={},
            skip_reason="Run the dualdice-paper profile or occupancy_ratio_benchmark.dualdice_grid for GridWalk.",
        )
    raise ValueError(f"Unknown estimator '{estimator}'.")


def _dice_rl_preflight_or_default(
    config: OccupancyRatioBenchmarkConfig,
    preflight: GoogleDICERLPreflight | None,
) -> GoogleDICERLPreflight:
    if preflight is not None:
        return preflight
    if not bool(config.include_dice_rl):
        return GoogleDICERLPreflight(False, "Google DICE-RL disabled by config.", config.dice_rl_repo_path)
    return preflight_google_dice_rl(config.dice_rl_repo_path)


def _first_stage_diagnostics(dataset: BenchmarkDataset, legacy: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    if dataset.true_action_ratio is not None and "pred_iw" in legacy:
        pred = np.maximum(np.asarray(legacy["pred_iw"], dtype=np.float64).reshape(-1), 1e-12)
        truth = np.maximum(np.asarray(dataset.true_action_ratio, dtype=np.float64).reshape(-1), 1e-12)
        if pred.shape == truth.shape:
            out["action_ratio_log_rmse"] = float(np.sqrt(np.mean((np.log(pred) - np.log(truth)) ** 2)))
            out["action_ratio_mae"] = float(np.mean(np.abs(pred - truth)))
    if dataset.true_transition_ratio is not None:
        k_fit = legacy.get("k_fit")
        if isinstance(k_fit, dict) and "k_hat" in k_fit:
            pred = np.maximum(np.asarray(k_fit["k_hat"], dtype=np.float64).reshape(-1), 1e-12)
            truth = np.maximum(np.asarray(dataset.true_transition_ratio, dtype=np.float64).reshape(-1), 1e-12)
            if pred.shape == truth.shape:
                out["transition_ratio_log_rmse"] = float(np.sqrt(np.mean((np.log(pred) - np.log(truth)) ** 2)))
                out["transition_ratio_mae"] = float(np.mean(np.abs(pred - truth)))
    return out


def _value_diagnostics(dataset: BenchmarkDataset, weights: Array) -> dict[str, float]:
    rewards = getattr(dataset, "rewards", None)
    if rewards is None:
        return {}
    reward_arr = np.asarray(rewards, dtype=np.float64).reshape(-1)
    weights_arr = np.asarray(weights, dtype=np.float64).reshape(-1)
    if reward_arr.shape != weights_arr.shape:
        return {}
    estimated_value = float(np.mean(weights_arr * reward_arr))
    if dataset.true_ratio is not None:
        truth_arr = np.asarray(dataset.true_ratio, dtype=np.float64).reshape(-1)
        if reward_arr.shape != truth_arr.shape:
            return {}
        oracle_value = float(np.mean(truth_arr * reward_arr))
        return {
            "ope_value_estimate": estimated_value,
            "ope_value_oracle": oracle_value,
            "ope_value_error": float(estimated_value - oracle_value),
            "ope_value_abs_error": float(abs(estimated_value - oracle_value)),
        }
    if "target_policy_value" not in dataset.metadata:
        return {}
    oracle_value = float(dataset.metadata["target_policy_value"])
    se = float(dataset.metadata.get("target_policy_value_se", np.nan))
    error = float(estimated_value - oracle_value)
    out = {
        "ope_value_estimate": estimated_value,
        "ope_value_target": oracle_value,
        "ope_value_oracle": oracle_value,
        "ope_value_error": error,
        "ope_value_abs_error": float(abs(error)),
        "ope_value_target_se": se if np.isfinite(se) else "",
    }
    if np.isfinite(se) and se > 0.0:
        out["ope_value_abs_error_se_units"] = float(abs(error) / se)
    return {
        key: value
        for key, value in out.items()
        if not (isinstance(value, float) and not np.isfinite(value))
    }


def _initial_ratio_inputs(
    dataset: BenchmarkDataset,
    config: OccupancyRatioBenchmarkConfig,
) -> tuple[Array | None, Array | None, Array | None, bool]:
    mode = str(config.source_state_correction_mode)
    if mode == "never":
        return None, None, None, False
    if mode == "auto" and not _source_state_correction_auto_applies(dataset):
        return None, None, None, False
    initial_states = getattr(dataset, "initial_states", None)
    if initial_states is None:
        if mode == "always":
            raise ValueError("source_state_correction_mode='always' requires dataset.initial_states.")
        return None, None, None, False
    initial_states_arr = np.asarray(initial_states, dtype=np.float64)
    if initial_states_arr.ndim == 0 or initial_states_arr.shape[0] == 0:
        if mode == "always":
            raise ValueError("source_state_correction_mode='always' requires nonempty initial_states.")
        return None, None, None, False
    initial_actions = getattr(dataset, "initial_actions", None)
    initial_actions_arr = None if initial_actions is None else np.asarray(initial_actions, dtype=np.float64)
    initial_weights = getattr(dataset, "initial_weights", None)
    return (
        initial_states_arr,
        initial_actions_arr,
        None if initial_weights is None else np.asarray(initial_weights, dtype=np.float64),
        True,
    )


def _target_next_actions_input(dataset: BenchmarkDataset) -> Array | None:
    target_next_actions = getattr(dataset, "next_target_actions", None)
    if target_next_actions is None:
        return None
    target_next_actions_arr = np.asarray(target_next_actions, dtype=np.float64)
    if target_next_actions_arr.ndim == 0 or target_next_actions_arr.shape[0] == 0:
        return None
    return target_next_actions_arr


def _source_state_correction_auto_applies(dataset: BenchmarkDataset) -> bool:
    initial_states = getattr(dataset, "initial_states", None)
    if initial_states is not None:
        initial_states_arr = np.asarray(initial_states)
        if initial_states_arr.ndim > 0 and initial_states_arr.shape[0] > 0:
            return True
    setting = str(getattr(dataset, "setting", ""))
    if setting.startswith("gym_") or setting in {
        "openml_finite_mdp",
        "obp_logged_bandit",
        "rtbgym_discrete",
        "recgym_recommender",
        "minari_pointmaze",
        "minari_minigrid",
    }:
        return True
    reference = str(getattr(dataset, "metadata", {}).get("reference_distribution", "")).lower()
    return "behavior_discounted" in reference or "logged" in reference


def _finite_or_blank(value: Any) -> float | str:
    if value is None:
        return ""
    out = float(value)
    return out if np.isfinite(out) else ""


def _selection_score(result: EstimatorResult) -> float:
    if result.status != "ok":
        return float("inf")
    for name in (
        "best_validation_loss",
        "final_validation_loss",
        "neural_occupancy_best_valid_loss",
        "neural_occupancy_final_valid_loss",
        "fixed_point_rel_change_final",
        "weight_cv",
    ):
        value = result.diagnostics.get(name)
        if value in ("", None):
            continue
        try:
            score = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(score):
            return score + _stability_penalty(result.diagnostics)
    return float("inf")


def _stability_penalty(diagnostics: dict[str, Any]) -> float:
    penalty = 0.0
    for name, scale in (
        ("normalization_error", 0.10),
        ("clipping_fraction", 0.05),
        ("negative_raw_fraction", 1.00),
        ("projection_clipped_fraction_final", 0.05),
        ("pseudo_outcome_clipped_fraction_final", 0.05),
    ):
        try:
            value = abs(float(diagnostics.get(name, 0.0)))
        except (TypeError, ValueError):
            value = 0.0
        if np.isfinite(value):
            penalty += scale * value
    try:
        ess = float(diagnostics.get("effective_sample_size_fraction", 1.0))
    except (TypeError, ValueError):
        ess = 1.0
    if np.isfinite(ess):
        penalty += _near_uniform_collapse_penalty(diagnostics, ess)
    return float(penalty)


def _near_uniform_collapse_penalty(diagnostics: dict[str, Any], ess: float) -> float:
    """Discourage selecting a nearly constant ratio when nuisance fits see policy shift."""
    try:
        cv = abs(float(diagnostics.get("weight_cv", 0.0)))
    except (TypeError, ValueError):
        cv = 0.0
    shift_signal = 0.0
    for name in (
        "action_ratio_mae",
        "transition_ratio_mae",
        "action_ratio_log_rmse",
        "transition_ratio_log_rmse",
    ):
        try:
            value = abs(float(diagnostics.get(name, 0.0)))
        except (TypeError, ValueError):
            value = 0.0
        if np.isfinite(value):
            shift_signal = max(shift_signal, value)
    if not np.isfinite(ess) or ess < 0.995 or cv > 0.05 or shift_signal < 0.05:
        return 0.0
    collapse = min(1.0, max(0.0, (ess - 0.995) / 0.005))
    shift = min(1.0, shift_signal)
    return float(0.25 * collapse * shift)


def _occupancy_history_diagnostics(history: list[dict[str, Any]]) -> dict[str, float]:
    out: dict[str, float] = {}
    if not history:
        return out
    risk_new = [float(row["risk_new"]) for row in history if "risk_new" in row and np.isfinite(float(row["risk_new"]))]
    if risk_new:
        out["final_validation_loss"] = float(risk_new[-1])
        out["best_validation_loss"] = float(np.min(risk_new))
    deltas = [float(row["huber_delta"]) for row in history if "huber_delta" in row]
    if deltas:
        out["huber_delta_mean"] = float(np.mean(deltas))
        out["huber_delta_median"] = float(np.median(deltas))
        out["huber_delta_final"] = float(deltas[-1])
    accepted = [float(bool(row.get("accepted"))) for row in history if "accepted" in row]
    if accepted:
        out["accepted_iteration_fraction"] = float(np.mean(accepted))
    validation_improved = [
        float(bool(row.get("validation_improved"))) for row in history if "validation_improved" in row
    ]
    if validation_improved:
        out["validation_improved_iteration_fraction"] = float(np.mean(validation_improved))
    warmup_accepts = [
        float(bool(row.get("validation_warmup_accept")))
        for row in history
        if "validation_warmup_accept" in row
    ]
    if warmup_accepts:
        out["validation_warmup_accept_fraction"] = float(np.mean(warmup_accepts))
    for source, dest in (
        ("fixed_point_rel_change_eval", "fixed_point_rel_change_final"),
        ("ess_fraction", "ess_fraction_final"),
        ("weight_p99", "weight_p99_final"),
        ("weight_max", "weight_max_final"),
        ("projection_clipped_fraction", "projection_clipped_fraction_final"),
        ("projection_post_normalization_clipped_fraction", "projection_post_normalization_clipped_fraction_final"),
        ("projection_negative_fraction", "projection_negative_fraction_final"),
        ("projection_nonfinite_fraction", "projection_nonfinite_fraction_final"),
        ("pseudo_outcome_clipped_fraction", "pseudo_outcome_clipped_fraction_final"),
    ):
        vals = [float(row[source]) for row in history if source in row and np.isfinite(float(row[source]))]
        if vals:
            out[dest] = float(vals[-1])
    return out


def _history_json_friendly(history: list[dict[str, Any]]) -> bool:
    allowed = (str, int, float, bool, type(None), np.integer, np.floating)
    return all(isinstance(value, allowed) for row in history for value in row.values())


def _calibrate_bellman_moment_weights(
    dataset: BenchmarkDataset,
    weights: Array,
    *,
    w_max: float | None,
) -> tuple[Array, dict[str, Any]]:
    h, h_next, init_moments = _bellman_moment_test_arrays(dataset)
    result = calibrate_occupancy_bellman_binning(
        omega_hat=weights,
        h=h,
        h_next=h_next,
        init_moments=init_moments,
        gamma=float(dataset.gamma),
        n_bins=10,
        min_bin_size=30,
        w_max=w_max,
        lambda_bellman=1.0,
        lambda_shrink=1.0,
        ridge=1e-6,
        normalize=True,
        return_diagnostics=True,
    )
    omega_cal = np.asarray(result["omega_cal"], dtype=np.float64).reshape(-1)
    diagnostics = result["diagnostics"]
    multipliers = np.asarray(diagnostics["multipliers"], dtype=np.float64).reshape(-1)
    bin_counts = np.asarray(diagnostics["bin_counts"], dtype=np.float64).reshape(-1)
    out: dict[str, Any] = {
        "bellman_moment_bin_count": float(bin_counts.size),
        "bellman_moment_bin_min_count": float(np.min(bin_counts)) if bin_counts.size else 0.0,
        "bellman_moment_bin_max_count": float(np.max(bin_counts)) if bin_counts.size else 0.0,
        "bellman_moment_multiplier_min": float(np.min(multipliers)) if multipliers.size else 0.0,
        "bellman_moment_multiplier_max": float(np.max(multipliers)) if multipliers.size else 0.0,
        "bellman_moment_multiplier_mean": float(np.mean(multipliers)) if multipliers.size else 0.0,
        "bellman_moment_normalization_scale": float(diagnostics["normalization_scale"]),
        "bellman_moment_objective_value": float(diagnostics["objective_value"]),
        "bellman_moment_residual_norm_before": float(diagnostics["residual_norm_before"]),
        "bellman_moment_residual_norm_after": float(diagnostics["residual_norm_after"]),
        "bellman_moment_ess_before": float(diagnostics["ess_before"]),
        "bellman_moment_ess_after": float(diagnostics["ess_after"]),
        "bellman_moment_max_weight_before": float(diagnostics["max_weight_before"]),
        "bellman_moment_max_weight_after": float(diagnostics["max_weight_after"]),
        "bellman_moment_residual_reduction_fraction": float(diagnostics["residual_reduction_fraction"]),
        "bellman_moment_ess_loss_fraction": float(diagnostics["ess_loss_fraction"]),
        "bellman_moment_q99_increase_fraction": float(diagnostics["q99_increase_fraction"]),
        "bellman_moment_max_weight_increase_fraction": float(diagnostics["max_weight_increase_fraction"]),
        "bellman_moment_recommendation": str(diagnostics["calibration_recommendation"]),
        "bellman_moment_recommendation_reasons": " | ".join(
            str(reason) for reason in diagnostics["calibration_recommendation_reasons"]
        ),
    }
    if "clipped_fraction" in diagnostics:
        out["bellman_moment_clipped_fraction"] = float(diagnostics["clipped_fraction"])
    return omega_cal, out


def _bellman_moment_test_arrays(dataset: BenchmarkDataset) -> tuple[Array, Array, Array]:
    h = _state_action_features(
        np.asarray(dataset.states, dtype=np.float64),
        np.asarray(dataset.actions, dtype=np.float64),
    )
    h_next = _state_action_features(
        np.asarray(dataset.next_states, dtype=np.float64),
        np.asarray(dataset.next_target_actions, dtype=np.float64),
    )
    init_features = _state_action_features(
        np.asarray(dataset.initial_states, dtype=np.float64),
        np.asarray(dataset.initial_actions, dtype=np.float64),
    )
    width = min(h.shape[1], h_next.shape[1], init_features.shape[1], 32)
    h = h[:, :width]
    h_next = h_next[:, :width]
    init_features = init_features[:, :width]
    init_weights = np.asarray(dataset.initial_weights, dtype=np.float64).reshape(-1)
    if init_weights.shape[0] != init_features.shape[0] or not np.all(np.isfinite(init_weights)):
        init_weights = np.ones(init_features.shape[0], dtype=np.float64)
    weight_sum = float(np.sum(init_weights))
    if not np.isfinite(weight_sum) or abs(weight_sum) <= 1e-12:
        init_weights = np.ones(init_features.shape[0], dtype=np.float64)
        weight_sum = float(init_features.shape[0])
    init_moments = np.sum(init_features * init_weights[:, None], axis=0) / weight_sum
    return h, h_next, init_moments


def _diagnostic_features(dataset: BenchmarkDataset) -> Array:
    features = _state_action_features(dataset.states, dataset.actions)
    if features.shape[1] > 32:
        return features[:, :32]
    return features


def _state_action_features(states: Array, actions: Array) -> Array:
    states_arr = np.asarray(states, dtype=np.float64).reshape(np.asarray(states).shape[0], -1)
    actions_arr = np.asarray(actions, dtype=np.float64).reshape(np.asarray(actions).shape[0], -1)
    if states_arr.shape[0] != actions_arr.shape[0]:
        raise ValueError("states and actions must have the same number of rows.")
    return np.concatenate([np.ones((states_arr.shape[0], 1), dtype=np.float64), states_arr, actions_arr], axis=1)
