from __future__ import annotations

import argparse
from dataclasses import fields
import hashlib
import json
from pathlib import Path
from typing import Any

from occupancy_ratio_benchmark.config import (
    BOOSTED_ESTIMATOR_PRESETS,
    NEURAL_ESTIMATOR_PRESETS,
    OccupancyRatioBenchmarkConfig,
)
from occupancy_ratio_benchmark.runner import run_benchmark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run occupancy-ratio estimator benchmarks.")
    profiles = ("smoke", "medium", "full", "overnight", "high_stakes", "dualdice-paper")
    parser.add_argument("--config", type=str, default=None, help="JSON benchmark config file.")
    parser.add_argument("--stage", choices=profiles, default=None)
    parser.add_argument("--profile", choices=profiles, default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--external-repo-path", default=None)
    parser.add_argument("--dice-rl-repo-path", default=None)
    parser.add_argument("--no-google-dualdice", action="store_true")
    parser.add_argument("--no-dice-rl", action="store_true")
    parser.add_argument("--include-dualdice-gridwalk", action="store_true")
    parser.add_argument("--gridwalk-alphas", nargs="*", type=float, default=None)
    parser.add_argument("--tune-cv", action="store_true")
    parser.add_argument("--automl-tuning", choices=("off", "fast", "balanced"), default=None)
    parser.add_argument("--cv-folds", type=int, default=None)
    parser.add_argument("--cv-scoring", choices=("composite", "loss"), default=None)
    parser.add_argument(
        "--cv-moment-extra-blocks",
        nargs="*",
        default=None,
        choices=("second_order", "multiscale_rff", "support", "policy_shift"),
    )
    parser.add_argument("--cv-moment-multiscale-rff-scales", nargs="*", type=float, default=None)
    parser.add_argument("--cv-moment-strata-quantiles", nargs="*", type=float, default=None)
    parser.add_argument("--cv-score-method", choices=("legacy_rank", "bellman_gmm"), default=None)
    parser.add_argument("--cv-gmm-objective", choices=("ratio", "ope"), default=None)
    parser.add_argument("--cv-gmm-cov-ridge", type=float, default=None)
    parser.add_argument("--cv-gmm-complexity-weight", type=float, default=None)
    parser.add_argument("--cv-gmm-ope-broad-weight", type=float, default=None)
    parser.add_argument("--cv-gmm-refit-fraction", type=float, default=None)
    parser.add_argument("--staged-cv", action="store_true")
    parser.add_argument("--staged-cv-iterations", type=int, default=None)
    parser.add_argument("--staged-cv-n-bootstrap", type=int, default=None)
    parser.add_argument("--settings", nargs="*", default=None)
    parser.add_argument("--estimators", nargs="*", default=None)
    parser.add_argument("--boosted-estimator-presets", nargs="*", default=None, choices=BOOSTED_ESTIMATOR_PRESETS)
    parser.add_argument("--neural-estimator-presets", nargs="*", default=None, choices=NEURAL_ESTIMATOR_PRESETS)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--sample-sizes", nargs="*", type=int, default=None)
    parser.add_argument("--gammas", nargs="*", type=float, default=None)
    parser.add_argument("--discrete-policy-shifts", nargs="*", type=float, default=None)
    parser.add_argument("--linear-gaussian-policy-shifts", nargs="*", type=float, default=None)
    parser.add_argument("--random-tabular-state-counts", nargs="*", type=int, default=None)
    parser.add_argument("--random-tabular-action-counts", nargs="*", type=int, default=None)
    parser.add_argument("--boosted-losses", nargs="*", default=None, choices=("squared", "huber"))
    parser.add_argument("--boosted-num-iterations", type=int, default=None)
    parser.add_argument("--boosted-mcmc-samples", type=int, default=None)
    parser.add_argument("--boosted-batch-size", type=int, default=None)
    parser.add_argument("--boosted-crossfit-folds", type=int, default=None)
    parser.add_argument("--boosted-moment-calibration", choices=("none", "scalar"), default=None)
    parser.add_argument("--boosted-density-ratio-loss", choices=("lsif", "logistic"), default=None)
    parser.add_argument("--boosted-logistic-logit-clip", type=float, default=None)
    parser.add_argument(
        "--boosted-stabilization-presets",
        nargs="*",
        default=None,
        choices=(
            "squared",
            "huber",
            "stable",
            "stable_factored",
            "relaxed_tail",
            "transition_norm",
            "crossfit2",
            "calibrated",
            "crossfit2_calibrated",
            "bellman_moment_calibrated",
            "logistic_nuisance",
            "stable_logistic_nuisance",
            "google_parity",
            "auto",
            "staged_cv",
            "huber_projection",
            "huber_projection_damping",
            "huber_projection_damping_transition_norm",
            "huber_projection_damping_weighted",
        ),
    )
    parser.add_argument("--neural-losses", nargs="*", default=None, choices=("squared", "huber"))
    parser.add_argument(
        "--neural-stabilization-presets",
        nargs="*",
        default=None,
        choices=(
            "squared",
            "huber",
            "stable",
            "stable_factored",
            "relaxed_tail",
            "transition_norm",
            "crossfit2",
            "calibrated",
            "crossfit2_calibrated",
            "bellman_moment_calibrated",
            "logistic_nuisance",
            "stable_logistic_nuisance",
            "auto",
            "staged_cv",
            "huber_projection",
            "huber_projection_damping",
            "huber_projection_damping_transition_norm",
            "huber_projection_damping_weighted",
        ),
    )
    parser.add_argument("--neural-num-iterations", type=int, default=None)
    parser.add_argument("--neural-gradient-steps-per-iteration", type=int, default=None)
    parser.add_argument("--neural-mcmc-samples", type=int, default=None)
    parser.add_argument("--neural-batch-size", type=int, default=None)
    parser.add_argument("--neural-hidden-dims", nargs="*", type=int, default=None)
    parser.add_argument("--neural-action-hidden-dims", nargs="*", type=int, default=None)
    parser.add_argument("--neural-source-hidden-dims", nargs="*", type=int, default=None)
    parser.add_argument("--neural-transition-hidden-dims", nargs="*", type=int, default=None)
    parser.add_argument("--neural-direct-one-step-hidden-dims", nargs="*", type=int, default=None)
    parser.add_argument("--neural-activation", choices=("relu", "tanh", "silu", "gelu"), default=None)
    parser.add_argument("--neural-learning-rate", type=float, default=None)
    parser.add_argument("--neural-nuisance-learning-rate", type=float, default=None)
    parser.add_argument("--neural-weight-decay", type=float, default=None)
    parser.add_argument("--neural-action-steps", type=int, default=None)
    parser.add_argument("--neural-source-steps", type=int, default=None)
    parser.add_argument("--neural-transition-steps", type=int, default=None)
    parser.add_argument("--neural-direct-one-step-steps", type=int, default=None)
    parser.add_argument("--neural-transition-permutation-samples", type=int, default=None)
    parser.add_argument("--neural-density-ratio-loss", choices=("lsif", "logistic"), default=None)
    parser.add_argument("--neural-logistic-logit-clip", type=float, default=None)
    parser.add_argument("--neural-validation-warmup-iterations", type=int, default=None)
    parser.add_argument("--neural-crossfit-folds", type=int, default=None)
    parser.add_argument("--neural-moment-calibration", choices=("none", "scalar"), default=None)
    parser.add_argument("--neural-direct-adjoint-steps", type=int, default=None)
    parser.add_argument("--neural-direct-adjoint-learning-rate", type=float, default=None)
    parser.add_argument("--neural-direct-adjoint-weight-decay", type=float, default=None)
    parser.add_argument("--neural-device", default=None)
    parser.add_argument("--source-state-correction-mode", choices=("auto", "always", "never"), default=None)
    parser.add_argument("--estimator-timeout-sec", type=float, default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--google-num-updates", type=int, default=None)
    parser.add_argument("--google-batch-size", type=int, default=None)
    parser.add_argument("--google-batch-sizes", nargs="*", type=int, default=None)
    parser.add_argument("--google-learning-rates", nargs="*", type=float, default=None)
    parser.add_argument("--google-weight-decays", nargs="*", type=float, default=None)
    parser.add_argument("--google-hidden-dims", nargs="*", type=int, default=None)
    parser.add_argument("--google-update-grid", nargs="*", type=int, default=None)
    parser.add_argument("--dice-rl-num-steps", type=int, default=None)
    parser.add_argument("--dice-rl-batch-size", type=int, default=None)
    parser.add_argument("--dice-rl-learning-rate", type=float, default=None)
    parser.add_argument("--dice-rl-hidden-dims", nargs="*", type=int, default=None)
    parser.add_argument("--dice-rl-update-grid", nargs="*", type=int, default=None)
    parser.add_argument("--dice-rl-batch-sizes", nargs="*", type=int, default=None)
    parser.add_argument("--dice-rl-learning-rates", nargs="*", type=float, default=None)
    parser.add_argument("--dice-rl-hidden-dim-grid", nargs="*", type=int, default=None)
    parser.add_argument("--huber-delta", type=float, default=None)
    parser.add_argument("--huber-delta-scale", type=float, default=None)
    parser.add_argument("--mc-truth-samples", type=int, default=None)
    parser.add_argument("--gym-target-value-rollouts", type=int, default=None)
    parser.add_argument("--application-target-value-rollouts", type=int, default=None)
    parser.add_argument("--openml-task-ids", nargs="*", type=int, default=None)
    parser.add_argument("--openml-max-tasks", type=int, default=None)
    parser.add_argument("--tabular-state-cap", type=int, default=None)
    parser.add_argument("--obp-campaigns", nargs="*", default=None)
    parser.add_argument("--minari-dataset-ids", nargs="*", default=None)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def load_config_file(path: str | Path) -> OccupancyRatioBenchmarkConfig:
    """Load a benchmark config JSON, starting from the named profile defaults."""
    config_path = Path(path)
    raw_bytes = config_path.read_bytes()
    payload = json.loads(raw_bytes.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Benchmark config JSON must contain an object.")

    overrides_payload = payload.get("overrides")
    if overrides_payload is not None and not isinstance(overrides_payload, dict):
        raise ValueError("Benchmark config 'overrides' must be an object when supplied.")
    overrides: dict[str, Any] = dict(overrides_payload or payload)
    profile = str(
        payload.get("profile")
        or payload.get("stage")
        or overrides.get("profile")
        or overrides.get("stage")
        or "smoke"
    )
    base = OccupancyRatioBenchmarkConfig.for_profile(profile)
    allowed = {field.name for field in fields(OccupancyRatioBenchmarkConfig)}
    unknown = sorted(key for key in overrides if key not in allowed)
    if unknown:
        raise ValueError(f"Unknown benchmark config field(s): {', '.join(unknown)}")

    updates = _coerce_config_updates(overrides)
    updates["config_path"] = config_path
    updates["config_sha256"] = hashlib.sha256(raw_bytes).hexdigest()
    return OccupancyRatioBenchmarkConfig(**{**base.__dict__, **updates})


def _coerce_config_updates(updates: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in updates.items():
        if key in {"output_root", "external_repo_path", "dice_rl_repo_path", "config_path"} and value is not None:
            out[key] = Path(value)
        elif isinstance(value, list):
            out[key] = tuple(value)
        else:
            out[key] = value
    return out


def main() -> None:
    args = parse_args()
    if args.config is not None:
        config = load_config_file(args.config)
    else:
        profile = args.stage or args.profile or "smoke"
        config = OccupancyRatioBenchmarkConfig.for_profile(
            profile,
            output_root=Path(args.output_root or "outputs/occupancy_ratio_benchmark"),
            external_repo_path=Path(args.external_repo_path or "/tmp/google-research"),
            include_google_dual_dice=not args.no_google_dualdice,
        )
    updates = {}
    for name, value in (
        ("stage", args.stage),
        ("profile", args.profile),
        ("output_root", Path(args.output_root) if args.output_root is not None else None),
        ("external_repo_path", Path(args.external_repo_path) if args.external_repo_path is not None else None),
        ("dice_rl_repo_path", Path(args.dice_rl_repo_path) if args.dice_rl_repo_path is not None else None),
        ("settings", args.settings),
        ("estimators", args.estimators),
        ("boosted_estimator_presets", args.boosted_estimator_presets),
        ("neural_estimator_presets", args.neural_estimator_presets),
        ("seeds", args.seeds),
        ("sample_sizes", args.sample_sizes),
        ("gammas", args.gammas),
        ("discrete_policy_shifts", args.discrete_policy_shifts),
        ("linear_gaussian_policy_shifts", args.linear_gaussian_policy_shifts),
        ("random_tabular_state_counts", args.random_tabular_state_counts),
        ("random_tabular_action_counts", args.random_tabular_action_counts),
        ("gridwalk_alphas", args.gridwalk_alphas),
        ("boosted_losses", args.boosted_losses),
        ("boosted_num_iterations", args.boosted_num_iterations),
        ("boosted_mcmc_samples", args.boosted_mcmc_samples),
        ("boosted_batch_size", args.boosted_batch_size),
        ("boosted_crossfit_folds", args.boosted_crossfit_folds),
        ("boosted_moment_calibration", args.boosted_moment_calibration),
        ("boosted_density_ratio_loss", args.boosted_density_ratio_loss),
        ("boosted_logistic_logit_clip", args.boosted_logistic_logit_clip),
        ("boosted_stabilization_presets", args.boosted_stabilization_presets),
        ("neural_losses", args.neural_losses),
        ("neural_stabilization_presets", args.neural_stabilization_presets),
        ("neural_num_iterations", args.neural_num_iterations),
        ("neural_gradient_steps_per_iteration", args.neural_gradient_steps_per_iteration),
        ("neural_mcmc_samples", args.neural_mcmc_samples),
        ("neural_batch_size", args.neural_batch_size),
        ("neural_hidden_dims", args.neural_hidden_dims),
        ("neural_action_hidden_dims", args.neural_action_hidden_dims),
        ("neural_source_hidden_dims", args.neural_source_hidden_dims),
        ("neural_transition_hidden_dims", args.neural_transition_hidden_dims),
        ("neural_direct_one_step_hidden_dims", args.neural_direct_one_step_hidden_dims),
        ("neural_activation", args.neural_activation),
        ("neural_learning_rate", args.neural_learning_rate),
        ("neural_nuisance_learning_rate", args.neural_nuisance_learning_rate),
        ("neural_weight_decay", args.neural_weight_decay),
        ("neural_action_steps", args.neural_action_steps),
        ("neural_source_steps", args.neural_source_steps),
        ("neural_transition_steps", args.neural_transition_steps),
        ("neural_direct_one_step_steps", args.neural_direct_one_step_steps),
        ("neural_transition_permutation_samples", args.neural_transition_permutation_samples),
        ("neural_density_ratio_loss", args.neural_density_ratio_loss),
        ("neural_logistic_logit_clip", args.neural_logistic_logit_clip),
        ("neural_validation_warmup_iterations", args.neural_validation_warmup_iterations),
        ("neural_crossfit_folds", args.neural_crossfit_folds),
        ("neural_moment_calibration", args.neural_moment_calibration),
        ("neural_direct_adjoint_steps", args.neural_direct_adjoint_steps),
        ("neural_direct_adjoint_learning_rate", args.neural_direct_adjoint_learning_rate),
        ("neural_direct_adjoint_weight_decay", args.neural_direct_adjoint_weight_decay),
        ("neural_device", args.neural_device),
        ("source_state_correction_mode", args.source_state_correction_mode),
        ("estimator_timeout_sec", args.estimator_timeout_sec),
        ("google_num_updates", args.google_num_updates),
        ("google_batch_size", args.google_batch_size),
        ("google_batch_sizes", args.google_batch_sizes),
        ("google_learning_rates", args.google_learning_rates),
        ("google_weight_decays", args.google_weight_decays),
        ("google_hidden_dims", args.google_hidden_dims),
        ("google_update_grid", args.google_update_grid),
        ("dice_rl_num_steps", args.dice_rl_num_steps),
        ("dice_rl_batch_size", args.dice_rl_batch_size),
        ("dice_rl_learning_rate", args.dice_rl_learning_rate),
        ("dice_rl_hidden_dims", args.dice_rl_hidden_dims),
        ("dice_rl_update_grid", args.dice_rl_update_grid),
        ("dice_rl_batch_sizes", args.dice_rl_batch_sizes),
        ("dice_rl_learning_rates", args.dice_rl_learning_rates),
        ("dice_rl_hidden_dim_grid", args.dice_rl_hidden_dim_grid),
        ("huber_delta", args.huber_delta),
        ("automl_tuning", args.automl_tuning),
        ("cv_folds", args.cv_folds),
        ("cv_scoring", args.cv_scoring),
        ("cv_moment_extra_blocks", args.cv_moment_extra_blocks),
        ("cv_moment_multiscale_rff_scales", args.cv_moment_multiscale_rff_scales),
        ("cv_moment_strata_quantiles", args.cv_moment_strata_quantiles),
        ("cv_score_method", args.cv_score_method),
        ("cv_gmm_objective", args.cv_gmm_objective),
        ("cv_gmm_cov_ridge", args.cv_gmm_cov_ridge),
        ("cv_gmm_complexity_weight", args.cv_gmm_complexity_weight),
        ("cv_gmm_ope_broad_weight", args.cv_gmm_ope_broad_weight),
        ("cv_gmm_refit_fraction", args.cv_gmm_refit_fraction),
        ("staged_cv_iterations", args.staged_cv_iterations),
        ("staged_cv_n_bootstrap", args.staged_cv_n_bootstrap),
        ("mc_truth_samples", args.mc_truth_samples),
        ("gym_target_value_rollouts", args.gym_target_value_rollouts),
        ("application_target_value_rollouts", args.application_target_value_rollouts),
        ("openml_task_ids", args.openml_task_ids),
        ("openml_max_tasks", args.openml_max_tasks),
        ("tabular_state_cap", args.tabular_state_cap),
        ("obp_campaigns", args.obp_campaigns),
        ("minari_dataset_ids", args.minari_dataset_ids),
    ):
        if value is not None:
            updates[name] = tuple(value) if isinstance(value, list) else value
    if args.neural_losses is not None and args.neural_stabilization_presets is None:
        updates["neural_stabilization_presets"] = ()
    if args.huber_delta_scale is not None:
        updates["huber_delta_scale"] = float(args.huber_delta_scale)
    if args.include_dualdice_gridwalk:
        updates["include_dualdice_gridwalk"] = True
    if args.no_google_dualdice:
        updates["include_google_dual_dice"] = False
    if args.no_dice_rl:
        updates["include_dice_rl"] = False
    if args.tune_cv:
        updates["tune_cv"] = True
        updates.setdefault("automl_tuning", "balanced")
    if args.staged_cv:
        updates["staged_bootstrap_cv"] = True
        updates["tune_cv"] = True
        updates.setdefault("automl_tuning", "balanced")
    if args.no_plots:
        updates["write_plots"] = False
    if args.no_resume:
        updates["resume"] = False
    if updates:
        config = OccupancyRatioBenchmarkConfig(**{**config.__dict__, **updates})
    result = run_benchmark(config)
    print(f"Wrote results: {result.results_path}")
    print(f"Wrote summary: {result.summary_path}")
    print(f"Wrote winners: {result.winner_path}")
    print(f"Wrote tuning: {result.tuning_path}")
    print(f"Wrote diagnostics: {result.diagnostics_path}")
    print(f"Wrote conservatism audit: {result.conservatism_audit_path}")
    print(f"Wrote conservatism report: {result.conservatism_report_path}")
    print(f"Wrote readout: {result.benchmark_readout_path}")
    print(result.plot_status)


if __name__ == "__main__":
    main()
