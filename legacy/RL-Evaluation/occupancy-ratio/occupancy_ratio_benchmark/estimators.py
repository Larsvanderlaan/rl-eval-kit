from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any

import numpy as np

from occupancy_ratio.fit_occupancy_ratio import (
    ActionRatioConfig,
    OccupancyRegressionConfig,
    TransitionRatioConfig,
    fit_discounted_occupancy_ratio,
    tune_discounted_occupancy_ratio_cv,
)
from occupancy_ratio.fit_occupancy_ratio_neural import (
    NeuralActionRatioConfig,
    NeuralOccupancyRegressionConfig,
    NeuralTransitionRatioConfig,
    fit_discounted_occupancy_ratio_neural,
    tune_discounted_occupancy_ratio_neural_cv,
)
from occupancy_ratio_benchmark.config import OccupancyRatioBenchmarkConfig
from occupancy_ratio_benchmark.data import BenchmarkDataset
from occupancy_ratio_benchmark.diagnostics import estimator_diagnostics_optional
from occupancy_ratio_benchmark.external_baselines import (
    GoogleDualDICEPreflight,
    estimate_google_dualdice_neural,
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


def estimate_boosted_tree(
    dataset: BenchmarkDataset,
    config: OccupancyRatioBenchmarkConfig,
    *,
    loss: str,
    preset: str | None = None,
) -> EstimatorResult:
    start = time.perf_counter()
    preset = str(loss if preset is None else preset)
    occupancy_options, nuisance_prediction_max = _boosted_stabilization_options(config, preset)
    nuisance_crossfit_folds, nuisance_moment_calibration = _boosted_nuisance_options(config, preset)
    density_ratio_loss = _boosted_density_ratio_loss(config, preset)
    effective_loss = str(occupancy_options.pop("loss"))
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
        lgb_params={
            "learning_rate": 0.08,
            "num_leaves": 31 if config.stage == "smoke" else 63,
            "min_data_in_leaf": 20 if config.stage == "smoke" else 100,
            "verbose": -1,
            "num_threads": 0,
        },
        **occupancy_options,
    )
    action_config = ActionRatioConfig(
        num_boost_round=30 if config.stage == "smoke" else 120,
        early_stopping_rounds=5,
        validation_fraction=0.20,
        show_progress=False,
        refit_on_all_data=True,
        lgb_params={
            "num_leaves": 31,
            "min_data_in_leaf": 20 if config.stage == "smoke" else 100,
            "verbose": -1,
            "num_threads": 0,
        },
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
        lgb_params={
            "num_leaves": 31,
            "min_data_in_leaf": 20 if config.stage == "smoke" else 100,
            "verbose": -1,
            "num_threads": 0,
        },
        prediction_max=nuisance_prediction_max,
        crossfit_folds=int(nuisance_crossfit_folds),
        moment_calibration=str(nuisance_moment_calibration),
        density_ratio_loss=str(density_ratio_loss),
        logistic_logit_clip=config.boosted_logistic_logit_clip,
    )
    tuning_rows: list[dict[str, Any]] = []
    if bool(config.tune_cv):
        tuned = tune_discounted_occupancy_ratio_cv(
            states=dataset.states,
            actions=dataset.actions,
            next_states=dataset.next_states,
            target_actions=dataset.target_actions,
            gamma=float(dataset.gamma),
            occupancy=occupancy_config,
            action_ratio=action_config,
            transition_ratio=transition_config,
            occupancy_grid=_boosted_occupancy_cv_grid(config, preset),
            action_ratio_grid=_nuisance_cv_grid(config),
            transition_ratio_grid=_nuisance_cv_grid(config),
            cv_folds=int(config.cv_folds),
            scoring=str(config.cv_scoring),
            lambda_norm=float(config.cv_lambda_norm),
            lambda_tail=float(config.cv_lambda_tail),
            seed=int(dataset.seed + 60_001),
            fit_final=True,
        )
        model = tuned["model"]
        tuning_rows = _flatten_boosted_tuning_rows(tuned, estimator=f"boosted_tree_{preset}", dataset=dataset)
    else:
        model = fit_discounted_occupancy_ratio(
            states=dataset.states,
            actions=dataset.actions,
            next_states=dataset.next_states,
            target_actions=dataset.target_actions,
            gamma=float(dataset.gamma),
            occupancy=occupancy_config,
            action_ratio=action_config,
            transition_ratio=transition_config,
        )
    raw = model.predict_state_action_ratio(dataset.states, dataset.actions, clip=False)
    weights = model.predict_state_action_ratio(dataset.states, dataset.actions, clip=True)
    diagnostics = estimator_diagnostics_optional(
        true_ratio=dataset.true_ratio,
        estimated_ratio=weights,
        raw_ratio=raw,
        reference_weights=dataset.reference_weights,
        feature_matrix=_diagnostic_features(dataset),
    )
    diagnostics.update(_value_diagnostics(dataset, weights))
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
    candidates = ("huber", "stable")
    results = [estimate_boosted_tree(dataset, config, loss="huber", preset=preset) for preset in candidates]
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
) -> EstimatorResult:
    start = time.perf_counter()
    _stabilize_torch_runtime()
    preset = str(loss if preset is None else preset)
    occupancy_options, nuisance_prediction_max = _neural_stabilization_options(config, preset)
    nuisance_crossfit_folds, nuisance_moment_calibration, density_ratio_loss = _neural_nuisance_options(config, preset)
    effective_loss = str(occupancy_options.pop("loss"))
    hidden_dims = (256, 256) if preset == "google_parity" else tuple(int(width) for width in config.neural_hidden_dims)
    activation = "relu" if preset == "google_parity" else str(config.neural_activation)
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
        **occupancy_options,
    )
    action_config = NeuralActionRatioConfig(
        hidden_dims=hidden_dims,
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
    transition_config = NeuralTransitionRatioConfig(
        hidden_dims=hidden_dims,
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
    tuning_rows: list[dict[str, Any]] = []
    if bool(config.tune_cv):
        tuned = tune_discounted_occupancy_ratio_neural_cv(
            states=dataset.states,
            actions=dataset.actions,
            next_states=dataset.next_states,
            target_actions=dataset.target_actions,
            gamma=float(dataset.gamma),
            occupancy=occupancy_config,
            action_ratio=action_config,
            transition_ratio=transition_config,
            occupancy_grid=_neural_occupancy_cv_grid(config, preset),
            action_grid=_nuisance_cv_grid(config),
            transition_grid=_nuisance_cv_grid(config),
            cv_folds=int(config.cv_folds),
            seed=int(dataset.seed + 70_001),
            fit_final=True,
        )
        model = tuned["model"]
        tuning_rows = _flatten_neural_tuning_rows(tuned, estimator=f"neural_network_{preset}", dataset=dataset)
    else:
        model = fit_discounted_occupancy_ratio_neural(
            states=dataset.states,
            actions=dataset.actions,
            next_states=dataset.next_states,
            target_actions=dataset.target_actions,
            gamma=float(dataset.gamma),
            occupancy=occupancy_config,
            action_ratio=action_config,
            transition_ratio=transition_config,
        )
    raw = model.predict_state_action_ratio(dataset.states, dataset.actions, clip=False)
    weights = model.predict_state_action_ratio(dataset.states, dataset.actions, clip=True)
    diagnostics = estimator_diagnostics_optional(
        true_ratio=dataset.true_ratio,
        estimated_ratio=weights,
        raw_ratio=raw,
        reference_weights=dataset.reference_weights,
        feature_matrix=_diagnostic_features(dataset),
    )
    diagnostics.update(_value_diagnostics(dataset, weights))
    diagnostics.update(_first_stage_diagnostics(dataset, model.to_legacy_dict()))
    diagnostics.update(
        {
            "occupancy_loss": effective_loss,
            "occupancy_stabilization_preset": preset,
            "neural_stabilization_preset": preset,
            "neural_loss": effective_loss,
            "neural_hidden_dims": "x".join(str(int(width)) for width in hidden_dims),
            "neural_activation": activation,
            "neural_gradient_steps_used": float(model.diagnostics.get("gradient_steps_used") or 0),
            "neural_accepted_count": float(model.diagnostics.get("accepted_count") or 0),
            "neural_validation_warmup_accepts": float(model.diagnostics.get("validation_warmup_accepts") or 0),
            "neural_validation_warmup_iterations": float(
                model.diagnostics.get("validation_warmup_iterations") or 0
            ),
            "neural_action_updates": float(model.diagnostics.get("action_updates") or 0),
            "neural_transition_updates": float(model.diagnostics.get("transition_updates") or 0),
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
    candidates = ("stable", "calibrated", "stable_logistic_nuisance")
    results = [estimate_neural_network(dataset, config, loss="huber", preset=preset) for preset in candidates]
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
    if preset in {
        "stable",
        "transition_norm",
        "crossfit2",
        "calibrated",
        "crossfit2_calibrated",
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
    if preset in {
        "stable",
        "transition_norm",
        "crossfit2",
        "calibrated",
        "crossfit2_calibrated",
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


def _flatten_boosted_tuning_rows(
    tuned: dict[str, Any],
    *,
    estimator: str,
    dataset: BenchmarkDataset,
) -> list[dict[str, Any]]:
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
) -> EstimatorResult:
    if estimator == "oracle":
        return estimate_oracle(dataset)
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
    if estimator == "neural_network_transition_norm":
        return estimate_neural_network(dataset, config, loss="huber", preset="transition_norm")
    if estimator == "neural_network_crossfit2":
        return estimate_neural_network(dataset, config, loss="huber", preset="crossfit2")
    if estimator == "neural_network_calibrated":
        return estimate_neural_network(dataset, config, loss="huber", preset="calibrated")
    if estimator == "neural_network_crossfit2_calibrated":
        return estimate_neural_network(dataset, config, loss="huber", preset="crossfit2_calibrated")
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
        penalty += max(0.0, 0.05 - ess) * 10.0
    return float(penalty)


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
        ("pseudo_outcome_clipped_fraction", "pseudo_outcome_clipped_fraction_final"),
    ):
        vals = [float(row[source]) for row in history if source in row and np.isfinite(float(row[source]))]
        if vals:
            out[dest] = float(vals[-1])
    return out


def _history_json_friendly(history: list[dict[str, Any]]) -> bool:
    allowed = (str, int, float, bool, type(None), np.integer, np.floating)
    return all(isinstance(value, allowed) for row in history for value in row.values())


def _diagnostic_features(dataset: BenchmarkDataset) -> Array:
    states = np.asarray(dataset.states, dtype=np.float64).reshape(dataset.n, -1)
    actions = np.asarray(dataset.actions, dtype=np.float64).reshape(dataset.n, -1)
    features = np.concatenate([np.ones((dataset.n, 1), dtype=np.float64), states, actions], axis=1)
    if features.shape[1] > 32:
        return features[:, :32]
    return features
