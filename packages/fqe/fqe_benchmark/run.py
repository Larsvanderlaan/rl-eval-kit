from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from fqe_benchmark.runner import run_benchmark
from fqe_benchmark.types import BenchmarkConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FQE estimator benchmarks.")
    parser.add_argument("--stage", choices=("smoke", "core", "full"), default="smoke")
    parser.add_argument("--output-root", default="outputs/fqe_benchmark")
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--estimators", nargs="*", default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--sample-sizes", nargs="*", type=int, default=None)
    parser.add_argument("--gammas", nargs="*", type=float, default=None)
    parser.add_argument("--policy-shifts", nargs="*", type=float, default=None)
    parser.add_argument("--stationary-gamma-ratio", type=float, default=None)
    parser.add_argument("--gym-target-value-rollouts", type=int, default=None)
    parser.add_argument("--google-research-path", default=None)
    parser.add_argument("--google-dualdice-num-updates", type=int, default=None)
    parser.add_argument("--google-dualdice-batch-size", type=int, default=None)
    parser.add_argument("--dice-rl-repo-path", default=None)
    parser.add_argument("--dice-rl-num-steps", type=int, default=None)
    parser.add_argument("--dice-rl-batch-size", type=int, default=None)
    parser.add_argument("--dice-rl-learning-rate", type=float, default=None)
    parser.add_argument("--scope-rl-repo-path", default=None)
    parser.add_argument("--scope-rl-n-steps", type=int, default=None)
    parser.add_argument("--scope-rl-n-steps-per-epoch", type=int, default=None)
    parser.add_argument("--scope-rl-batch-size", type=int, default=None)
    parser.add_argument("--scope-rl-learning-rate", type=float, default=None)
    parser.add_argument("--tune-cv", action="store_true")
    parser.add_argument("--automl-tuning", choices=("off", "fast", "balanced"), default=None)
    parser.add_argument("--staged-cv", action="store_true")
    parser.add_argument("--staged-cv-iterations", nargs="*", type=int, default=None)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = BenchmarkConfig.for_stage(args.stage, output_root=Path(args.output_root))
    updates = {}
    for name, value in (
        ("datasets", args.datasets),
        ("estimators", args.estimators),
        ("seeds", args.seeds),
        ("sample_sizes", args.sample_sizes),
        ("gammas", args.gammas),
        ("policy_shifts", args.policy_shifts),
    ):
        if value is not None:
            updates[name] = tuple(value)
    if args.stationary_gamma_ratio is not None:
        updates["stationary_gamma_ratio"] = float(args.stationary_gamma_ratio)
    if args.gym_target_value_rollouts is not None:
        updates["gym_target_value_rollouts"] = int(args.gym_target_value_rollouts)
    if args.google_research_path is not None:
        updates["google_research_path"] = Path(args.google_research_path)
    if args.google_dualdice_num_updates is not None:
        updates["google_dualdice_num_updates"] = int(args.google_dualdice_num_updates)
    if args.google_dualdice_batch_size is not None:
        updates["google_dualdice_batch_size"] = int(args.google_dualdice_batch_size)
    if args.dice_rl_repo_path is not None:
        updates["dice_rl_repo_path"] = Path(args.dice_rl_repo_path)
    if args.dice_rl_num_steps is not None:
        updates["dice_rl_num_steps"] = int(args.dice_rl_num_steps)
    if args.dice_rl_batch_size is not None:
        updates["dice_rl_batch_size"] = int(args.dice_rl_batch_size)
    if args.dice_rl_learning_rate is not None:
        updates["dice_rl_learning_rate"] = float(args.dice_rl_learning_rate)
    if args.scope_rl_repo_path is not None:
        updates["scope_rl_repo_path"] = Path(args.scope_rl_repo_path)
    if args.scope_rl_n_steps is not None:
        updates["scope_rl_n_steps"] = int(args.scope_rl_n_steps)
    if args.scope_rl_n_steps_per_epoch is not None:
        updates["scope_rl_n_steps_per_epoch"] = int(args.scope_rl_n_steps_per_epoch)
    if args.scope_rl_batch_size is not None:
        updates["scope_rl_batch_size"] = int(args.scope_rl_batch_size)
    if args.scope_rl_learning_rate is not None:
        updates["scope_rl_learning_rate"] = float(args.scope_rl_learning_rate)
    if args.no_plots:
        updates["output_plots"] = False
    if args.fail_fast:
        updates["fail_fast"] = True
    if args.automl_tuning is not None:
        updates["automl_tuning"] = args.automl_tuning
    if args.staged_cv:
        updates["staged_cv"] = True
        updates.setdefault("automl_tuning", "balanced")
    if args.staged_cv_iterations is not None:
        updates["staged_cv_iterations"] = tuple(args.staged_cv_iterations)
    if args.tune_cv:
        updates["tune_cv"] = True
        updates.setdefault("automl_tuning", "balanced")
    if updates:
        config = replace(config, **updates)
    result = run_benchmark(config)
    print(f"Wrote results: {result.results_path}")
    print(f"Wrote summary: {result.summary_path}")
    print(f"Wrote diagnostics: {result.diagnostics_path}")
    print(f"Wrote manifest: {result.manifest_path}")
    print(f"Wrote tuning results: {result.tuning_results_path}")


if __name__ == "__main__":
    main()
