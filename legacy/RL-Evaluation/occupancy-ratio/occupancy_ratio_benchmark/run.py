from __future__ import annotations

import argparse
from pathlib import Path

from occupancy_ratio_benchmark.config import (
    BOOSTED_ESTIMATOR_PRESETS,
    NEURAL_ESTIMATOR_PRESETS,
    OccupancyRatioBenchmarkConfig,
)
from occupancy_ratio_benchmark.runner import run_benchmark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run occupancy-ratio estimator benchmarks.")
    profiles = ("smoke", "medium", "full", "overnight", "dualdice-paper")
    parser.add_argument("--stage", choices=profiles, default=None)
    parser.add_argument("--profile", choices=profiles, default="smoke")
    parser.add_argument("--output-root", default="outputs/occupancy_ratio_benchmark")
    parser.add_argument("--external-repo-path", default="/tmp/google-research")
    parser.add_argument("--no-google-dualdice", action="store_true")
    parser.add_argument("--include-dualdice-gridwalk", action="store_true")
    parser.add_argument("--tune-cv", action="store_true")
    parser.add_argument("--cv-folds", type=int, default=None)
    parser.add_argument("--cv-scoring", choices=("composite", "loss"), default=None)
    parser.add_argument("--settings", nargs="*", default=None)
    parser.add_argument("--estimators", nargs="*", default=None)
    parser.add_argument("--boosted-estimator-presets", nargs="*", default=None, choices=BOOSTED_ESTIMATOR_PRESETS)
    parser.add_argument("--neural-estimator-presets", nargs="*", default=None, choices=NEURAL_ESTIMATOR_PRESETS)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--sample-sizes", nargs="*", type=int, default=None)
    parser.add_argument("--gammas", nargs="*", type=float, default=None)
    parser.add_argument("--linear-gaussian-policy-shifts", nargs="*", type=float, default=None)
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
            "transition_norm",
            "crossfit2",
            "calibrated",
            "crossfit2_calibrated",
            "logistic_nuisance",
            "stable_logistic_nuisance",
            "auto",
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
            "transition_norm",
            "crossfit2",
            "calibrated",
            "crossfit2_calibrated",
            "logistic_nuisance",
            "stable_logistic_nuisance",
            "auto",
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
    parser.add_argument("--neural-activation", choices=("relu", "tanh", "silu", "gelu"), default=None)
    parser.add_argument("--neural-learning-rate", type=float, default=None)
    parser.add_argument("--neural-nuisance-learning-rate", type=float, default=None)
    parser.add_argument("--neural-weight-decay", type=float, default=None)
    parser.add_argument("--neural-action-steps", type=int, default=None)
    parser.add_argument("--neural-transition-steps", type=int, default=None)
    parser.add_argument("--neural-transition-permutation-samples", type=int, default=None)
    parser.add_argument("--neural-density-ratio-loss", choices=("lsif", "logistic"), default=None)
    parser.add_argument("--neural-logistic-logit-clip", type=float, default=None)
    parser.add_argument("--neural-validation-warmup-iterations", type=int, default=None)
    parser.add_argument("--neural-crossfit-folds", type=int, default=None)
    parser.add_argument("--neural-moment-calibration", choices=("none", "scalar"), default=None)
    parser.add_argument("--neural-device", default=None)
    parser.add_argument("--estimator-timeout-sec", type=float, default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--google-num-updates", type=int, default=None)
    parser.add_argument("--google-batch-size", type=int, default=None)
    parser.add_argument("--huber-delta", type=float, default=None)
    parser.add_argument("--huber-delta-scale", type=float, default=None)
    parser.add_argument("--mc-truth-samples", type=int, default=None)
    parser.add_argument("--gym-target-value-rollouts", type=int, default=None)
    parser.add_argument("--openml-task-ids", nargs="*", type=int, default=None)
    parser.add_argument("--openml-max-tasks", type=int, default=None)
    parser.add_argument("--tabular-state-cap", type=int, default=None)
    parser.add_argument("--obp-campaigns", nargs="*", default=None)
    parser.add_argument("--minari-dataset-ids", nargs="*", default=None)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profile = args.stage or args.profile
    config = OccupancyRatioBenchmarkConfig.for_profile(
        profile,
        output_root=Path(args.output_root),
        external_repo_path=Path(args.external_repo_path),
        include_google_dual_dice=not args.no_google_dualdice,
    )
    updates = {}
    for name, value in (
        ("settings", args.settings),
        ("estimators", args.estimators),
        ("boosted_estimator_presets", args.boosted_estimator_presets),
        ("neural_estimator_presets", args.neural_estimator_presets),
        ("seeds", args.seeds),
        ("sample_sizes", args.sample_sizes),
        ("gammas", args.gammas),
        ("linear_gaussian_policy_shifts", args.linear_gaussian_policy_shifts),
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
        ("neural_activation", args.neural_activation),
        ("neural_learning_rate", args.neural_learning_rate),
        ("neural_nuisance_learning_rate", args.neural_nuisance_learning_rate),
        ("neural_weight_decay", args.neural_weight_decay),
        ("neural_action_steps", args.neural_action_steps),
        ("neural_transition_steps", args.neural_transition_steps),
        ("neural_transition_permutation_samples", args.neural_transition_permutation_samples),
        ("neural_density_ratio_loss", args.neural_density_ratio_loss),
        ("neural_logistic_logit_clip", args.neural_logistic_logit_clip),
        ("neural_validation_warmup_iterations", args.neural_validation_warmup_iterations),
        ("neural_crossfit_folds", args.neural_crossfit_folds),
        ("neural_moment_calibration", args.neural_moment_calibration),
        ("neural_device", args.neural_device),
        ("estimator_timeout_sec", args.estimator_timeout_sec),
        ("google_num_updates", args.google_num_updates),
        ("google_batch_size", args.google_batch_size),
        ("huber_delta", args.huber_delta),
        ("cv_folds", args.cv_folds),
        ("cv_scoring", args.cv_scoring),
        ("mc_truth_samples", args.mc_truth_samples),
        ("gym_target_value_rollouts", args.gym_target_value_rollouts),
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
    if args.tune_cv:
        updates["tune_cv"] = True
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
    print(result.plot_status)


if __name__ == "__main__":
    main()
