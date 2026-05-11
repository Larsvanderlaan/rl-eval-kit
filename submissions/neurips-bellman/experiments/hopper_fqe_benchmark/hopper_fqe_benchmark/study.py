from __future__ import annotations

import argparse
import json

from .runner import BenchmarkConfig, run_benchmark


def parse_args() -> BenchmarkConfig:
    parser = argparse.ArgumentParser(description="Run the Hopper medium benchmark for FQE, DualDICE, and stationary-weighted FQE.")
    parser.add_argument("--data-dir", type=str, default="hopper_fqe_benchmark/artifacts")
    parser.add_argument("--artifact-dir", type=str, default="hopper_fqe_benchmark/artifacts")
    parser.add_argument("--benchmark-dir", type=str, default="hopper_fqe_benchmark/artifacts/benchmark/dope")
    parser.add_argument("--output-dir", type=str, default="hopper_fqe_benchmark/outputs")
    parser.add_argument("--gamma-eval", type=float, default=0.99)
    parser.add_argument("--gamma-ratio", type=float, default=0.99)
    parser.add_argument("--noise-scale", type=float, default=0.25)
    parser.add_argument("--max-trajectories", type=int, default=None)
    parser.add_argument("--max-transitions", type=int, default=None)
    parser.add_argument(
        "--target-policies",
        type=str,
        nargs="+",
        default=[f"hopper-medium_{idx:02d}" for idx in range(11)],
    )
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=[
            "standard_fqe",
            "weighted_dual_dice",
            "weighted_linear",
        ],
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--min-weight", type=float, default=1e-4)
    parser.add_argument("--max-weight", type=float, default=20.0)
    parser.add_argument("--clip-quantile", type=float, default=0.995)
    parser.add_argument("--uniform-mix", type=float, default=0.05)
    parser.add_argument("--target-ess-fraction", type=float, default=0.4)
    parser.add_argument("--ratio-feature-quadratic", action="store_true")
    parser.add_argument("--ratio-feature-cross-terms", action="store_true")
    parser.add_argument("--fqe-num-updates", type=int, default=20_000)
    parser.add_argument("--dice-num-updates", type=int, default=20_000)
    parser.add_argument("--saddle-max-steps", type=int, default=5_000)
    parser.add_argument("--rkhs-max-steps", type=int, default=3_000)
    parser.add_argument("--saddle-normalization-penalty", type=float, default=2.0)
    parser.add_argument("--saddle-uniform-mix", type=float, default=0.0)
    parser.add_argument("--saddle-target-ess-fraction", type=float, default=0.2)
    parser.add_argument("--saddle-max-weight", type=float, default=100.0)
    parser.add_argument("--rkhs-normalization-penalty", type=float, default=2.0)
    parser.add_argument("--rkhs-uniform-mix", type=float, default=0.0)
    parser.add_argument("--rkhs-target-ess-fraction", type=float, default=0.2)
    parser.add_argument("--rkhs-max-weight", type=float, default=100.0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--budget",
        type=str,
        choices=["smoke", "pilot", "paper"],
        default=None,
        help="Convenience preset for update counts. smoke=2k, pilot=100k, paper=1M updates for FQE and DualDICE.",
    )
    args = parser.parse_args()
    fqe_num_updates = args.fqe_num_updates
    dice_num_updates = args.dice_num_updates
    if args.budget == "smoke":
        fqe_num_updates = 2_000
        dice_num_updates = 2_000
    elif args.budget == "pilot":
        fqe_num_updates = 100_000
        dice_num_updates = 100_000
    elif args.budget == "paper":
        fqe_num_updates = 1_000_000
        dice_num_updates = 1_000_000
    return BenchmarkConfig(
        data_dir=args.data_dir,
        artifact_dir=args.artifact_dir,
        benchmark_dir=args.benchmark_dir,
        output_dir=args.output_dir,
        gamma_eval=args.gamma_eval,
        gamma_ratio=args.gamma_ratio,
        max_trajectories=args.max_trajectories,
        max_transitions=args.max_transitions,
        target_policies=tuple(args.target_policies),
        methods=tuple(args.methods),
        seeds=tuple(args.seeds),
        noise_scale=args.noise_scale,
        min_weight=args.min_weight,
        max_weight=args.max_weight,
        clip_quantile=args.clip_quantile,
        uniform_mix=args.uniform_mix,
        target_ess_fraction=args.target_ess_fraction,
        ratio_feature_quadratic=bool(args.ratio_feature_quadratic),
        ratio_feature_cross_terms=bool(args.ratio_feature_cross_terms),
        fqe_num_updates=fqe_num_updates,
        dice_num_updates=dice_num_updates,
        saddle_max_steps=args.saddle_max_steps,
        rkhs_max_steps=args.rkhs_max_steps,
        saddle_normalization_penalty=args.saddle_normalization_penalty,
        saddle_uniform_mix=args.saddle_uniform_mix,
        saddle_target_ess_fraction=args.saddle_target_ess_fraction,
        saddle_max_weight=args.saddle_max_weight,
        rkhs_normalization_penalty=args.rkhs_normalization_penalty,
        rkhs_uniform_mix=args.rkhs_uniform_mix,
        rkhs_target_ess_fraction=args.rkhs_target_ess_fraction,
        rkhs_max_weight=args.rkhs_max_weight,
        device=args.device,
    )


def main() -> None:
    cfg = parse_args()
    result = run_benchmark(cfg)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
