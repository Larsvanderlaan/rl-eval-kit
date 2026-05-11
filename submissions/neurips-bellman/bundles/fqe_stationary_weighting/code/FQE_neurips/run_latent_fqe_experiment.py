from __future__ import annotations

import argparse
import json

from .latent_garnet_benchmark import (
    LatentGarnetConfig,
    build_latent_garnet_benchmark,
    evaluate_fqe_methods_on_benchmark,
    evaluate_weight_estimators_on_benchmark,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run latent Garnet weight and FQE comparison experiment.")
    parser.add_argument("--n-states", type=int, default=100)
    parser.add_argument("--n-actions", type=int, default=4)
    parser.add_argument("--latent-dim", type=int, default=3)
    parser.add_argument("--branching-factor", type=int, default=5)
    parser.add_argument("--dataset-size", type=int, default=10000)
    parser.add_argument("--data-mode", type=str, default="mixed")
    parser.add_argument("--n-trajectories", type=int, default=200)
    parser.add_argument("--iid-fraction", type=float, default=0.5)
    parser.add_argument("--behavior-coverage", type=float, default=0.3)
    parser.add_argument("--observation-mode", type=str, default="compact_nonlinear")
    parser.add_argument("--gamma-ratio", type=float, default=1.0)
    parser.add_argument("--gamma-eval", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    benchmark = build_latent_garnet_benchmark(
        LatentGarnetConfig(
            n_states=args.n_states,
            n_actions=args.n_actions,
            latent_dim=args.latent_dim,
            branching_factor=args.branching_factor,
            dataset_size=args.dataset_size,
            data_mode=args.data_mode,
            n_trajectories=args.n_trajectories,
            iid_fraction=args.iid_fraction,
            behavior_coverage=args.behavior_coverage,
            observation_mode=args.observation_mode,
            seed=args.seed,
        )
    )

    weight_results = evaluate_weight_estimators_on_benchmark(
        benchmark=benchmark,
        gamma_ratio=args.gamma_ratio,
        seed=args.seed,
    )
    fqe_results = evaluate_fqe_methods_on_benchmark(
        benchmark=benchmark,
        gamma_eval=args.gamma_eval,
        gamma_ratio=args.gamma_ratio,
        seed=args.seed,
    )
    output = {
        "config": {
            "n_states": args.n_states,
            "n_actions": args.n_actions,
            "latent_dim": args.latent_dim,
            "branching_factor": args.branching_factor,
            "dataset_size": args.dataset_size,
            "data_mode": args.data_mode,
            "n_trajectories": args.n_trajectories,
            "iid_fraction": args.iid_fraction,
            "behavior_coverage": args.behavior_coverage,
            "observation_mode": args.observation_mode,
            "gamma_ratio": args.gamma_ratio,
            "gamma_eval": args.gamma_eval,
            "seed": args.seed,
        },
        "weight_results": weight_results,
        "fqe_results": fqe_results,
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
